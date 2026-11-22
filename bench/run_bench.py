"""Speed benchmark harness around llama.cpp's `llama-bench`, and the project's speed gate.

What
    * `run_bench(model_gguf, backend, n_gpu_layers)` runs `llama-bench` of the LittleMore engine (engine/build/bin) on one
      GGUF file: generation of 128 tokens (`tg128`) first, then prompt processing of 512 tokens (`pp512`), in tokens per
      second,
      with every repetition kept (the reported value is the median of the repetitions), plus the GPU memory the run used (`vram_mb`, sampled with nvidia-smi) and the GPU's SM
      clock during the run (to see whether the GPU was at its boost clock, see "Clocks" below).
    * `run_plan(...)` runs a list of files in interleaved rounds (A B C A B C ...) so slow drifts of the machine (clocks,
      temperature, DRAM bandwidth) hit every file alike, and writes all raw results to a JSON file.
    * `check_gate(results)` is the hard speed gate of the project:
      LittleBit at 0.55 bits per weight generates tokens faster than the bf16 dense model, for BOTH Qwen3-0.6B and
      Qwen3-1.7B.  Pure logic, tested in tests/bench/test_check_gate.py.
    * `render_markdown(results)` turns the JSON into the tables of bench/results/phase1.md.
Why
    The gate needs numbers that can be reproduced and whose spread is known: this laptop GPU changes its clock with load
    (idles at 1605 MHz, boosts to 2040-2340 MHz after about a second of load) and its DRAM bandwidth varies between runs
    (128-188 GB/s measured with engine/tools/littlebit-bench/read_bandwidth.cu).  So every number here comes with its
    repetitions, the rounds are interleaved, and the clock is recorded.
Clocks
    `llama-bench` runs one untimed warm-up pass per test and then the timed repetitions.  Before each measured
    invocation this harness starts a GPU load (bench/gpu_load.py: by default memory-bound copies of a 256 MiB buffer;
    float16 matrix products with GPU_LOAD_ARGS = ["--mode", "gemm"]) for `warmup_s` seconds
    (default 3 s), keeps it running while llama-bench loads the model, and stops it when llama-bench starts its own
    warm-up pass, so the GPU never idles between the warm-up and the measurement; the SM clock at the start of the first
    timed repetition is recorded and a run below BOOST_MIN_MHZ is flagged and repeated once; generation is measured
    before prompt processing (see `run_bench`).  The SM clock sampled during the timed repetitions (from llama-bench's
    first timed repetition to its end; the untimed warm-up pass and the model load are not included) is stored
    (`sm_clock_mhz`: min / median / max) so a run at a low clock is visible in the results instead of silently lowering
    a number.  (Results written before 2026-11-22 15:00, e.g. bench/results/phase1.json, sampled from 1 s after the load
    stopped, i.e. they include the warm-up pass.)

Usage
    python -m bench.run_bench --plan bench/plans/phase1.json --rounds 3 --out bench/results/phase1.json
    python -m bench.run_bench --markdown bench/results/phase1.json          (print the tables)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "engine" / "build" / "bin"
EXE = ".exe" if os.name == "nt" else ""
# CUDA 12.9 runtime DLLs must be on PATH for ggml-cuda (the project pins CUDA 12.9, see docs/setup-windows.md)
CUDA_BIN = Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin")

# The two base models the hard gate is defined on (the rule is defined on exactly these two sizes).
GATE_MODELS = ("qwen3-0.6b", "qwen3-1.7b")
# Results measured on PROXY weights (a base model rebuilt from a quantized GGUF with tools/dev/dequant_gguf_to_hf.py:
# right shapes and types, other weight values; valid for speed only) are stored under "<model>-proxy" and marked
# "weights": "proxy".  They never satisfy the gate silently: check_gate uses them only with allow_proxy=True, and
# gate_status then reports the result as provisional.
PROXY_SUFFIX = "-proxy"

# backend name -> llama-bench device argument ("none" = no GPU device at all, i.e. pure CPU)
_DEVICES = {"cuda": "CUDA0", "vulkan": "Vulkan0", "cpu": "none"}


def _env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the engine programs: CUDA 12.9 first on PATH, plus `extra` (a plan entry's "env", e.g.
    {"GGML_CUDA_LITTLEBIT_F32_PREFILL": "1"} to measure the engine's float32 prefill path next to its default)."""
    env = dict(os.environ)
    env.update(extra or {})
    if CUDA_BIN.exists():
        env["PATH"] = str(CUDA_BIN) + os.pathsep + env.get("PATH", "")
    return env


class _GpuSampler:
    """Samples nvidia-smi (memory used in MiB, SM clock in MHz) every `period_ms` in a background process.

    nvidia-smi reports the memory used on the whole GPU (all processes, including the CUDA context of the benchmark),
    so `peak_mb - base_mb` is what the benchmark process added.  Without nvidia-smi every value is None.
    """

    def __init__(self, period_ms: int = 100) -> None:
        """Prepare an idle sampler; nothing runs until the `with` block is entered.  `period_ms`: sampling period."""
        self.period_ms = period_ms
        self.samples: list[tuple[float, float]] = []     # (memory used MiB, SM clock MHz)
        self.times: list[float] = []                     # time.perf_counter() of each sample
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def query() -> tuple[float, float] | None:
        """One sample (memory used MiB, SM clock MHz) or None when nvidia-smi is not available."""
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,clocks.sm", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            mem, clk = out.strip().splitlines()[0].split(",")
            return float(mem), float(clk)
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    def __enter__(self) -> "_GpuSampler":
        """Start `nvidia-smi -lms <period>` and a reader thread that appends to `samples` / `times`; without nvidia-smi
        the sampler stays empty (samples == []) and the benchmark still runs."""
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=memory.used,clocks.sm", "--format=csv,noheader,nounits",
                 f"-lms", str(self.period_ms)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self._proc = None
            return self

        def read() -> None:
            """Reader thread: append each nvidia-smi line (memory MiB, SM clock MHz) and its time to the samples."""
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                try:
                    mem, clk = line.strip().split(",")
                    self.samples.append((float(mem), float(clk)))
                    self.times.append(time.perf_counter())
                except ValueError:
                    pass

        self._thread = threading.Thread(target=read, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop nvidia-smi (terminate, then kill after 10 s) and join the reader thread; never suppresses exceptions."""
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=10)


def _bench_cmd(model_gguf: Path, backend: str, n_gpu_layers: int, args: Iterable[str]) -> list[str]:
    """The llama-bench command line for one model on one backend."""
    if backend not in _DEVICES:
        raise ValueError(f"unknown backend {backend!r}; expected one of {sorted(_DEVICES)}")
    exe = BIN / ("llama-bench" + EXE)
    if not exe.exists():
        raise FileNotFoundError(f"{exe} not found: build the engine first (docs/development.md)")
    ngl = 0 if backend == "cpu" else n_gpu_layers
    return [str(exe), "-m", str(model_gguf), "-ngl", str(ngl), "-dev", _DEVICES[backend], *args]


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    """Run one command in the engine's bin directory with the benchmark environment (_env), capturing UTF-8 output.
    Raises subprocess.TimeoutExpired after `timeout` seconds; stdin is closed so llama-bench can never wait for input."""
    return subprocess.run(cmd, cwd=BIN, env=_env(), capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, stdin=subprocess.DEVNULL)


# a measured run whose SM clock at the first timed repetition is below this is flagged ("clock_low"): the GPU of this
# project idles at 1605 MHz and boosts to 2040-2340 MHz; memory-bound runs settle around 1.8-2.0 GHz
BOOST_MIN_MHZ = 1800.0
# seconds after the warm-up load stopped before memory / clock samples count (see run_bench)
VRAM_SETTLE_S = 1.0
# arguments of bench/gpu_load.py (the warm-up load); see its --mode
GPU_LOAD_ARGS: list[str] = []


def _bench_with_load(cmd: list[str], warmup_s: float, use_load: bool,
                     env: dict[str, str] | None = None) -> tuple[subprocess.CompletedProcess, dict]:
    """Run one llama-bench command with the GPU kept busy until llama-bench's own warm-up pass starts.

    With `use_load`, bench/gpu_load.py runs first for `warmup_s` seconds, then llama-bench is started while the load
    keeps running (so the GPU stays at its boost clock while llama-bench loads the model), and the load is stopped the
    moment llama-bench reports its untimed warm-up pass (`--progress`).  The SM clock is read when llama-bench reports
    its first timed repetition.  Returns the completed llama-bench process and
    {"clock_at_start_mhz", "load_s", "load_stop_t", "run1_t"} (load_stop_t: time.perf_counter() when the load was
    stopped; run1_t: when llama-bench reported its first timed repetition, None if it did not).
    """
    info: dict = {"clock_at_start_mhz": None, "load_s": 0.0, "load_stop_t": time.perf_counter(), "run1_t": None}
    load = None
    if use_load:
        t0 = time.perf_counter()
        load = subprocess.Popen([sys.executable, "-m", "bench.gpu_load", *GPU_LOAD_ARGS], cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True)
        assert load.stdout is not None
        if load.stdout.readline().strip() != "ready":
            load.kill()
            raise RuntimeError("bench/gpu_load.py did not start (is torch with CUDA installed in this environment?)")
        while time.perf_counter() - t0 < warmup_s:
            time.sleep(0.05)
    stop_load = lambda: (load.terminate(), load.wait(timeout=30)) if load is not None and load.poll() is None else None  # noqa: E731
    proc = subprocess.Popen([*cmd, "--progress"], cwd=BIN, env=_env(env), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace")
    out_chunks: list[str] = []
    reader = threading.Thread(target=lambda: out_chunks.append(proc.stdout.read()), daemon=True)
    reader.start()
    err_lines: list[str] = []
    try:
        assert proc.stderr is not None
        for line in proc.stderr:
            err_lines.append(line)
            if "warmup" in line and load is not None and load.poll() is None:
                stop_load()
                info["load_stop_t"] = time.perf_counter()
                info["load_s"] = info["load_stop_t"] - t0
            if " run 1/" in line and info["clock_at_start_mhz"] is None:
                info["run1_t"] = time.perf_counter()
                q = _GpuSampler.query()
                info["clock_at_start_mhz"] = q[1] if q else None
        proc.wait(timeout=3600)
    finally:
        stop_load()
        if proc.poll() is None:
            proc.kill()
    reader.join(timeout=60)
    return subprocess.CompletedProcess(cmd, proc.returncode, "".join(out_chunks), "".join(err_lines)), info


def run_bench(model_gguf: Path, backend: str, n_gpu_layers: int, *, reps: int = 10, n_prompt: int = 512,
              n_gen: int = 128, warmup_s: float = 3.0, extra_args: Iterable[str] = (),
              env: dict[str, str] | None = None) -> dict[str, Any]:
    """Benchmark one GGUF with llama-bench; returns a dict with at least the keys `pp512`, `tg128` and `vram_mb`.

    `backend`: "cuda", "vulkan" or "cpu" (cpu = no GPU device at all, `-dev none -ngl 0`).  `n_gpu_layers`: llama.cpp's
    -ngl (99 = every layer on the GPU).  `reps`: timed repetitions per test (llama-bench additionally runs one untimed
    warm-up pass per test).  Generation is measured first, then prompt processing, in two llama-bench runs (measured on
    this GPU: when tg128 follows pp512 in the same run, its first repetitions are 10-20 % slower than the rest, 191 vs
    237 tokens/s for Qwen3-0.6B LittleBit, while the GPU recovers from the power-capped prompt phase).  Before each of
    the two runs a GPU load (bench/gpu_load.py) runs for `warmup_s` seconds and keeps running until llama-bench starts
    its own warm-up pass (see `_bench_with_load`); on the CPU backend there is no load.  Returned (speeds in tokens per
    second; keys follow n_prompt / n_gen):
        pp512 / tg128             MEDIAN over the repetitions
        pp512_mean / tg128_mean   mean over the repetitions;  pp512_sd / tg128_sd  their standard deviation
        pp512_samples / ...       every repetition
        tg128_clock_at_start_mhz  SM clock when the first timed repetition started (same for pp512)
        clock_low                 True if either of them was below BOOST_MIN_MHZ (the run is flagged, not dropped)
        vram_mb                   peak GPU memory used during the two runs minus the memory used before them (MiB);
                                  only measured without the warm-up load (warmup_s=0), None otherwise, without
                                  nvidia-smi and on the CPU backend
        vram_peak_mb, vram_base_mb  the two raw nvidia-smi values (whole GPU)
        sm_clock_mhz              [min, median, max] SM clock sampled while llama-bench measured (after the load stopped)
        cmd, file_mib, warmup_s   the command lines, the file size and the load time actually spent before each run
    Raises RuntimeError when llama-bench fails (e.g. the model does not fit) with its error output.
    """
    model_gguf = Path(model_gguf).resolve()   # llama-bench runs in engine/build/bin
    out: dict[str, Any] = {"file_mib": model_gguf.stat().st_size / 2**20, "backend": backend,
                           "n_gpu_layers": n_gpu_layers, "cmd": [], "warmup_s": []}
    use_load = backend != "cpu" and warmup_s > 0
    base = _GpuSampler.query()
    windows: list[tuple[float, float]] = []
    with _GpuSampler() as smp:
        for key, p, n in ((f"tg{n_gen}", 0, n_gen), (f"pp{n_prompt}", n_prompt, 0)):
            cmd = _bench_cmd(model_gguf, backend, n_gpu_layers,
                             ["-p", str(p), "-n", str(n), "-r", str(reps), "-o", "json", *extra_args])
            out["cmd"].append(cmd)
            r, info = _bench_with_load(cmd, warmup_s, use_load, env)
            # memory and clock samples count from the first timed repetition (so the clock range is that of the measured
            # repetitions, not of the warm-up pass), and at least VRAM_SETTLE_S after the load was stopped: nvidia-smi
            # still shows the load's memory for a moment after it exited (measured: +549 MiB on one bf16 run)
            start = info["load_stop_t"] + (VRAM_SETTLE_S if use_load else 0.0)
            if info["run1_t"] is not None:
                start = max(start, info["run1_t"])
            windows.append((start, time.perf_counter()))
            out["warmup_s"].append(info["load_s"])
            out[key + "_clock_at_start_mhz"] = info["clock_at_start_mhz"]
            if r.returncode != 0:
                raise RuntimeError(f"llama-bench failed ({r.returncode}) on {model_gguf.name}:\n{r.stderr[-3000:]}")
            rows = json.loads(r.stdout[r.stdout.index("["):])
            if len(rows) != 1:
                raise RuntimeError(f"llama-bench gave {len(rows)} results for {key}: {r.stdout[-2000:]}")
            samples = [float(v) for v in rows[0]["samples_ts"]]
            out[key] = statistics.median(samples)
            out[key + "_mean"] = statistics.fmean(samples)
            out[key + "_sd"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
            out[key + "_samples"] = samples
            out.setdefault("build_commit", rows[0].get("build_commit"))
            out.setdefault("gpu_info", rows[0].get("gpu_info"))
    clocks_start = [out.get(f"{k}_clock_at_start_mhz") for k in (f"tg{n_gen}", f"pp{n_prompt}")]
    out["clock_low"] = use_load and any(c is not None and c < BOOST_MIN_MHZ for c in clocks_start)
    inside = [s for s, t in zip(smp.samples, smp.times) if any(a <= t <= b for a, b in windows)]
    if inside and base is not None and backend != "cpu":
        peak = max(m for m, _ in inside)
        clocks = sorted(c for _, c in inside)
        out.update(vram_mb=peak - base[0], vram_peak_mb=peak, vram_base_mb=base[0],
                   sm_clock_mhz=[clocks[0], statistics.median(clocks), clocks[-1]])
        if use_load:
            # not valid with the warm-up load: its memory lingers in nvidia-smi for a moment after it exits, and a
            # settle delay long enough for that misses the whole pp512 run of a small model (measured: 535 instead of
            # 821 MiB); measure VRAM in a separate pass with warmup_s=0
            out.update(vram_mb=None, vram_peak_mb=None)
    else:
        out.update(vram_mb=None, vram_peak_mb=None, vram_base_mb=None, sm_clock_mhz=None)
    return out


def check_gate(results: dict, lb_key: str = "littlebit-0.55", dense_key: str = "bf16",
               models: tuple[str, ...] = GATE_MODELS, metric: str = "tg128", allow_proxy: bool = False) -> bool:
    """True only if, for every gate model, `results[model][lb_key][metric]` > `results[model][dense_key][metric]`.

    `results` maps model name -> variant name -> measurement dict (as `run_bench` returns, or aggregated by
    `aggregate`).  A missing model or variant, or a value that is not a finite positive number, fails the gate: an
    unmeasured gate cannot pass.  Equal speed is not faster.  Results on proxy weights (key "<model>-proxy") count only
    with `allow_proxy=True`, and only for a model that has no result on its original weights.
    """
    for model in models:
        variants = results.get(model)
        if variants is None and allow_proxy:
            variants = results.get(model + PROXY_SUFFIX)
        if not isinstance(variants, dict):
            return False
        try:
            lb = float(variants[lb_key][metric])
            dense = float(variants[dense_key][metric])
        except (KeyError, TypeError, ValueError):
            return False
        if not all(math.isfinite(v) and v > 0 for v in (lb, dense)):
            return False
        if not lb > dense:
            return False
    return True


def gate_status(results: dict, lb_key: str = "littlebit-0.55", dense_key: str = "bf16",
                models: tuple[str, ...] = GATE_MODELS) -> str:
    """The gate as a line of text: "PASS", "PASS (provisional: <models> on proxy weights)" or "FAIL".

    Provisional means the gate passes only when proxy-weight results stand in for the listed models, which have no
    result on their original weights; it is not the formal gate.
    """
    if check_gate(results, lb_key, dense_key, models):
        return "PASS"
    if check_gate(results, lb_key, dense_key, models, allow_proxy=True):
        proxies = [m for m in models if m not in results and m + PROXY_SUFFIX in results]
        return f"PASS (provisional: {', '.join(proxies)} on proxy weights)"
    return "FAIL"


def aggregate(runs: list[dict]) -> dict:
    """Raw runs (one dict per model, variant and round) -> model -> variant -> summary.

    Per variant: `tg128` / `pp512` = mean of the per-round medians, `*_rounds` the per-round medians, `*_all` every
    repetition of every round, `*_min` / `*_max` over the per-round medians, `vram_mb` the largest of the rounds, and the SM
    clock range over all rounds.
    """
    out: dict = {}
    for run in runs:
        v = out.setdefault(run["model"], {}).setdefault(run["variant"], {"file": run["file"], "rounds": 0,
                                                                        "weights": run.get("weights", "original")})
        v["rounds"] += 1
        for key in ("pp512", "tg128"):
            v.setdefault(key + "_rounds", []).append(run[key])
            v.setdefault(key + "_all", []).extend(run[key + "_samples"])
        if run.get("vram_mb") is not None:
            v["vram_mb"] = max(v.get("vram_mb") or 0.0, run["vram_mb"])
        if run.get("sm_clock_mhz"):
            lo, _, hi = run["sm_clock_mhz"]
            cur = v.get("sm_clock_range_mhz") or [lo, hi]
            v["sm_clock_range_mhz"] = [min(cur[0], lo), max(cur[1], hi)]
        v["file_mib"] = run["file_mib"]
    for variants in out.values():
        for v in variants.values():
            for key in ("pp512", "tg128"):
                rm = v[key + "_rounds"]
                v[key] = statistics.fmean(rm)
                v[key + "_min"], v[key + "_max"] = min(rm), max(rm)
                v[key + "_sd_all"] = statistics.stdev(v[key + "_all"]) if len(v[key + "_all"]) > 1 else 0.0
    return out


def provenance() -> dict:
    """What was measured: the engine's git commit and whether its working tree had changes, the commit llama-bench reports
    (its build info, written when CMake last configured the build, so it can lag behind a rebuilt tree), and this
    repository's commit.  Stored in the results JSON by run_plan."""
    def git(*args: str, cwd: Path) -> str:
        """Stdout of `git <args>` in `cwd`, stripped; an empty string when git is missing or the command fails."""
        try:
            return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    eng = ROOT / "engine"
    try:
        ver = subprocess.run([str(BIN / ("llama-bench" + EXE)), "--version"], cwd=BIN, env=_env(), capture_output=True,
                             text=True, timeout=60)
        version = (ver.stdout + ver.stderr).strip().splitlines()
        version = next((v for v in version if "commit" in v), "")
    except (OSError, subprocess.SubprocessError):
        version = ""
    return {"engine_commit": git("rev-parse", "HEAD", cwd=eng),
            "engine_dirty": bool(git("status", "--porcelain", "--untracked-files=no", cwd=eng)),
            "llama_bench_version": version,
            "repo_commit": git("rev-parse", "HEAD", cwd=ROOT)}


def run_plan(plan: list[dict], rounds: int, out_json: Path, backend: str = "cuda", n_gpu_layers: int = 99,
             reps: int = 10, warmup_s: float = 3.0) -> list[dict]:
    """Run every plan entry ({"model", "variant", "file"}) once per round, rounds interleaved; save raw runs as JSON.

    An entry may carry "weights": "proxy" (its model key must then end in "-proxy"); it is copied into the record.
    An entry may carry "env": {name: value} for the engine programs (copied into the record), e.g. to run the same file
    once with the engine's default and once with GGML_CUDA_LITTLEBIT_F32_PREFILL=1 in the same interleaved rounds.
    The JSON is rewritten after every run, so an interrupted plan keeps what was measured.  An entry that fails (e.g.
    out of GPU memory) is recorded with its error instead of aborting the plan.  A run flagged `clock_low` (the GPU was
    below BOOST_MIN_MHZ when a measurement started) is repeated once; the repeat replaces it and carries
    `"retried_after_clock_low": true` (if the repeat is flagged too, it stays flagged).  The engine provenance
    (`provenance()`) is stored at the top of the JSON.
    """
    runs: list[dict] = []
    prov = provenance()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    for rnd in range(rounds):
        for entry in plan:
            weights = entry.get("weights", "original")
            if (weights == "proxy") != entry["model"].endswith(PROXY_SUFFIX):
                raise ValueError(f"plan entry {entry}: proxy weights need a model key ending in {PROXY_SUFFIX!r} and vice versa")
            rec = {"model": entry["model"], "variant": entry["variant"], "file": entry["file"], "round": rnd,
                   "weights": weights,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            if entry.get("env"):
                rec["env"] = dict(entry["env"])
            try:
                rec.update(run_bench(ROOT / entry["file"], backend, n_gpu_layers, reps=reps, warmup_s=warmup_s,
                                         env=entry.get("env")))
                if rec.get("clock_low"):
                    rec.update(run_bench(ROOT / entry["file"], backend, n_gpu_layers, reps=reps, warmup_s=warmup_s,
                                         env=entry.get("env")))
                    rec["retried_after_clock_low"] = True
                print(f"round {rnd} {entry['model']:<12} {entry['variant']:<26} tg128 {rec['tg128']:8.1f} "
                      f"pp512 {rec['pp512']:8.1f}  vram {rec['vram_mb']}  clock {rec['sm_clock_mhz']}  at start "
                      f"{rec.get('tg128_clock_at_start_mhz')}/{rec.get('pp512_clock_at_start_mhz')}"
                      f"{'  CLOCK LOW' if rec.get('clock_low') else ''}", flush=True)
            except (RuntimeError, FileNotFoundError, subprocess.SubprocessError) as e:
                rec["error"] = str(e)[-3000:]
                print(f"round {rnd} {entry['model']} {entry['variant']}: FAILED {str(e)[:300]}", flush=True)
            runs.append(rec)
            out_json.write_text(json.dumps({"backend": backend, "n_gpu_layers": n_gpu_layers, "reps": reps,
                                            "warmup_s": warmup_s, "provenance": prov,
                                            "clock_window": "timed_repetitions", "runs": runs}, indent=1),
                                encoding="utf-8")
    return runs


def render_markdown(raw: dict, vram_raw: dict | None = None) -> str:
    """Markdown tables (per model: every variant's tg128 / pp512 / VRAM) and the gate lines, from a run_plan JSON.

    `vram_raw`: optionally another run_plan JSON whose VRAM values replace those of `raw` (used when the speed run's
    VRAM samples were not valid, see bench/results/phase1.md); the column heading says so.
    """
    runs = [r for r in raw["runs"] if "error" not in r]
    agg = aggregate(runs)
    vram_note = ""
    if vram_raw is not None:
        vagg = aggregate([r for r in vram_raw["runs"] if "error" not in r])
        for model, variants in agg.items():
            for name, v in variants.items():
                v["vram_mb"] = vagg.get(model, {}).get(name, {}).get("vram_mb")
        vram_note = " (separate pass)"
    # where the SM clock samples come from: the timed repetitions (current harness), or, in results written before that
    # change (no "clock_window" key), from 1 s after the warm-up load stopped, which includes llama-bench's warm-up pass
    clock_note = (" during the timed repetitions" if raw.get("clock_window") == "timed_repetitions"
                  else " from 1 s after the load stopped, incl. the untimed warm-up pass")
    lines: list[str] = []
    for model, variants in agg.items():
        proxy = any(v.get("weights") == "proxy" for v in variants.values())
        lines += [f"### {model}" + (" (PROXY weights: speed only, not the original model)" if proxy else ""), "",
                  f"| variant | file MiB | tg128 t/s (mean of the round medians; min..max of them) | pp512 t/s (same) | VRAM MiB{vram_note} | SM clock MHz (min..max sampled{clock_note}) |",
                  "|---|---|---|---|---|---|"]
        for name, v in variants.items():
            clk = v.get("sm_clock_range_mhz")
            lines.append(f"| {name} | {v['file_mib']:.0f} | {v['tg128']:.1f} ({v['tg128_min']:.1f}..{v['tg128_max']:.1f}) | "
                         f"{v['pp512']:.0f} ({v['pp512_min']:.0f}..{v['pp512_max']:.0f}) | "
                         f"{v['vram_mb']:.0f} | {f'{clk[0]:.0f}..{clk[1]:.0f}' if clk else '-'} |"
                         if v.get("vram_mb") is not None else
                         f"| {name} | {v['file_mib']:.0f} | {v['tg128']:.1f} | {v['pp512']:.0f} | - | - |")
        lines.append("")
    failed = [r for r in raw["runs"] if "error" in r]
    for r in failed:
        lines.append(f"- FAILED: {r['model']} {r['variant']} (round {r['round']}): {r['error'].strip().splitlines()[-1][:200]}")
    for lb_key in sorted({k for v in agg.values() for k in v if k.startswith("littlebit-0.55")}):
        for dense_key in ("bf16", "f16"):
            status = gate_status(agg, lb_key=lb_key, dense_key=dense_key)
            lines.append(f"- gate {lb_key} > {dense_key} (tg128, {' and '.join(GATE_MODELS)}): **{status}**")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="LittleMore speed benchmarks (llama-bench) and the speed gate.")
    ap.add_argument("--plan", type=Path, help="JSON list of {model, variant, file} (file relative to the repo root)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=float, default=3.0, help="seconds of GPU warm-up before each measured run")
    ap.add_argument("--backend", default="cuda", choices=sorted(_DEVICES))
    ap.add_argument("-ngl", "--n-gpu-layers", type=int, default=99)
    ap.add_argument("--out", type=Path, help="raw results JSON (with --plan)")
    ap.add_argument("--markdown", type=Path, help="print the tables of an existing results JSON")
    ap.add_argument("--vram", type=Path, help="with --markdown: take the VRAM column from this results JSON")
    a = ap.parse_args(argv)
    if a.markdown:
        vram = json.loads(a.vram.read_text(encoding="utf-8")) if a.vram else None
        print(render_markdown(json.loads(a.markdown.read_text(encoding="utf-8")), vram))
        return 0
    if not a.plan or not a.out:
        ap.error("--plan and --out are required unless --markdown is given")
    plan = json.loads(a.plan.read_text(encoding="utf-8"))
    run_plan(plan, a.rounds, a.out, a.backend, a.n_gpu_layers, a.reps, a.warmup)
    print(render_markdown(json.loads(a.out.read_text(encoding="utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
