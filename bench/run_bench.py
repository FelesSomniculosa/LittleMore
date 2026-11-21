"""Speed benchmark harness around llama.cpp's `llama-bench`, and the project's speed gate.

What
    * `run_bench(model_gguf, backend, n_gpu_layers)` runs `llama-bench` of the LittleMore engine (engine/build/bin) on one
      GGUF file: prompt processing of 512 tokens (`pp512`) and generation of 128 tokens (`tg128`), in tokens per second,
      with every repetition kept (the reported value is the median of the repetitions), plus the GPU memory the run used (`vram_mb`, sampled with nvidia-smi) and the GPU's SM
      clock during the run (to see whether the GPU was at its boost clock, see "Clocks" below).
    * `run_plan(...)` runs a list of files in interleaved rounds (A B C A B C ...) so slow drifts of the machine (clocks,
      temperature, DRAM bandwidth) hit every file alike, and writes all raw results to a JSON file.
    * `check_gate(results)` is the hard speed gate of the project (docs/specs/2026-11-21-littlemore-design.md, "Fast"):
      LittleBit at 0.55 bits per weight generates tokens faster than the bf16 dense model, for BOTH Qwen3-0.6B and
      Qwen3-1.7B.  Pure logic, tested in tests/bench/test_check_gate.py.
    * `render_markdown(results)` turns the JSON into the tables of bench/results/phase1.md.
Why
    The gate needs numbers that can be reproduced and whose spread is known: this laptop GPU changes its clock with load
    (idles at 1605 MHz, boosts to 2040-2340 MHz after about a second of load) and its DRAM bandwidth varies between runs
    (128-188 GB/s measured with engine/tools/littlebit-bench/read_bandwidth.cu).  So every number here comes with its
    repetitions, the rounds are interleaved, and the clock is recorded.
Clocks
    `llama-bench` runs one untimed warm-up pass per test and then the timed repetitions; before the measured invocations
    this harness keeps the GPU busy for `warmup_s` seconds (default 3 s) with short `llama-bench` generation runs of the
    same model (`-p 0 -n 64`, repeated until the time is up), so the GPU does not start the measured run from its idle
    clock; generation is measured before prompt processing (see `run_bench`).  The SM clock sampled during the measured run is stored (`sm_clock_mhz`: min / median / max) so a run at a low
    clock is visible in the results instead of silently lowering a number.

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

# The two base models the hard gate is defined on (spec: "Speed gate is on Qwen3-0.6B and 1.7B").
GATE_MODELS = ("qwen3-0.6b", "qwen3-1.7b")

# backend name -> llama-bench device argument ("none" = no GPU device at all, i.e. pure CPU)
_DEVICES = {"cuda": "CUDA0", "vulkan": "Vulkan0", "cpu": "none"}


def _env() -> dict[str, str]:
    """Environment for the engine programs: CUDA 12.9 first on PATH."""
    env = dict(os.environ)
    if CUDA_BIN.exists():
        env["PATH"] = str(CUDA_BIN) + os.pathsep + env.get("PATH", "")
    return env


class _GpuSampler:
    """Samples nvidia-smi (memory used in MiB, SM clock in MHz) every `period_ms` in a background process.

    nvidia-smi reports the memory used on the whole GPU (all processes, including the CUDA context of the benchmark),
    so `peak_mb - base_mb` is what the benchmark process added.  Without nvidia-smi every value is None.
    """

    def __init__(self, period_ms: int = 100) -> None:
        self.period_ms = period_ms
        self.samples: list[tuple[float, float]] = []
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
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=memory.used,clocks.sm", "--format=csv,noheader,nounits",
                 f"-lms", str(self.period_ms)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self._proc = None
            return self

        def read() -> None:
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                try:
                    mem, clk = line.strip().split(",")
                    self.samples.append((float(mem), float(clk)))
                except ValueError:
                    pass

        self._thread = threading.Thread(target=read, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
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
    return subprocess.run(cmd, cwd=BIN, env=_env(), capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, stdin=subprocess.DEVNULL)


def _warm_up(model_gguf: Path, backend: str, n_gpu_layers: int, seconds: float) -> float:
    """Keep the GPU busy with short generation runs of the same model for at least `seconds`; returns the time spent."""
    if seconds <= 0 or backend == "cpu":
        return 0.0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        r = _run(_bench_cmd(model_gguf, backend, n_gpu_layers, ["-p", "0", "-n", "64", "-r", "3", "-o", "json"]), 600)
        if r.returncode != 0:
            raise RuntimeError(f"warm-up run failed ({r.returncode}):\n{r.stderr[-2000:]}")
    return time.perf_counter() - t0


def run_bench(model_gguf: Path, backend: str, n_gpu_layers: int, *, reps: int = 10, n_prompt: int = 512,
              n_gen: int = 128, warmup_s: float = 3.0, extra_args: Iterable[str] = ()) -> dict[str, Any]:
    """Benchmark one GGUF with llama-bench; returns a dict with at least the keys `pp512`, `tg128` and `vram_mb`.

    `backend`: "cuda", "vulkan" or "cpu" (cpu = no GPU device at all, `-dev none -ngl 0`).  `n_gpu_layers`: llama.cpp's
    -ngl (99 = every layer on the GPU).  `reps`: timed repetitions per test (llama-bench additionally runs one untimed
    warm-up pass per test).  Generation is measured first, right after the warm-up, then prompt processing, in two
    llama-bench runs (measured on this GPU: when tg128 follows pp512 in the same run, its first repetitions are 10-20 %
    slower than the rest, 191 vs 237 tokens/s for Qwen3-0.6B LittleBit, while the GPU recovers from the power-capped
    prompt phase).  Returned (speeds in tokens per second; keys follow n_prompt / n_gen):
        pp512 / tg128             MEDIAN over the repetitions (robust against the one slow first repetition that remains
                                  after the warm-up: 226 vs 238-240 tokens/s in a tg-only run of the same file)
        pp512_mean / tg128_mean   mean over the repetitions;  pp512_sd / tg128_sd  their standard deviation
        pp512_samples / ...       every repetition
        vram_mb                   peak GPU memory used during the two runs minus the memory used before them (MiB;
                                  None without nvidia-smi or on the CPU backend)
        vram_peak_mb, vram_base_mb  the two raw nvidia-smi values (whole GPU)
        sm_clock_mhz              [min, median, max] SM clock sampled during the measured runs (None without nvidia-smi)
        cmd, file_mib, warmup_s   the command lines, the file size and the warm-up time actually spent
    Raises RuntimeError when llama-bench fails (e.g. the model does not fit) with its error output.
    """
    model_gguf = Path(model_gguf).resolve()   # llama-bench runs in engine/build/bin
    out: dict[str, Any] = {"file_mib": model_gguf.stat().st_size / 2**20, "backend": backend,
                           "n_gpu_layers": n_gpu_layers, "cmd": []}
    out["warmup_s"] = _warm_up(model_gguf, backend, n_gpu_layers, warmup_s)
    base = _GpuSampler.query()
    with _GpuSampler() as smp:
        for key, p, n in ((f"tg{n_gen}", 0, n_gen), (f"pp{n_prompt}", n_prompt, 0)):
            cmd = _bench_cmd(model_gguf, backend, n_gpu_layers,
                             ["-p", str(p), "-n", str(n), "-r", str(reps), "-o", "json", *extra_args])
            out["cmd"].append(cmd)
            r = _run(cmd, 3600)
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
    if smp.samples and base is not None and backend != "cpu":
        peak = max(m for m, _ in smp.samples)
        clocks = sorted(c for _, c in smp.samples)
        out.update(vram_mb=peak - base[0], vram_peak_mb=peak, vram_base_mb=base[0],
                   sm_clock_mhz=[clocks[0], statistics.median(clocks), clocks[-1]])
    else:
        out.update(vram_mb=None, vram_peak_mb=None, vram_base_mb=None, sm_clock_mhz=None)
    return out


def check_gate(results: dict, lb_key: str = "littlebit-0.55", dense_key: str = "bf16",
               models: tuple[str, ...] = GATE_MODELS, metric: str = "tg128") -> bool:
    """True only if, for every gate model, `results[model][lb_key][metric]` > `results[model][dense_key][metric]`.

    `results` maps model name -> variant name -> measurement dict (as `run_bench` returns, or aggregated by
    `aggregate`).  A missing model or variant, or a value that is not a finite positive number, fails the gate: an
    unmeasured gate cannot pass.  Equal speed is not faster.
    """
    for model in models:
        variants = results.get(model)
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


def aggregate(runs: list[dict]) -> dict:
    """Raw runs (one dict per model, variant and round) -> model -> variant -> summary.

    Per variant: `tg128` / `pp512` = mean of the per-round medians, `*_rounds` the per-round medians, `*_all` every
    repetition of every round, `*_min` / `*_max` over the per-round medians, `vram_mb` the largest of the rounds, and the SM
    clock range over all rounds.
    """
    out: dict = {}
    for run in runs:
        v = out.setdefault(run["model"], {}).setdefault(run["variant"], {"file": run["file"], "rounds": 0})
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


def run_plan(plan: list[dict], rounds: int, out_json: Path, backend: str = "cuda", n_gpu_layers: int = 99,
             reps: int = 10, warmup_s: float = 3.0) -> list[dict]:
    """Run every plan entry ({"model", "variant", "file"}) once per round, rounds interleaved; save raw runs as JSON.

    The JSON is rewritten after every run, so an interrupted plan keeps what was measured.  An entry that fails (e.g.
    out of GPU memory) is recorded with its error instead of aborting the plan.
    """
    runs: list[dict] = []
    out_json.parent.mkdir(parents=True, exist_ok=True)
    for rnd in range(rounds):
        for entry in plan:
            rec = {"model": entry["model"], "variant": entry["variant"], "file": entry["file"], "round": rnd,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            try:
                rec.update(run_bench(ROOT / entry["file"], backend, n_gpu_layers, reps=reps, warmup_s=warmup_s))
                print(f"round {rnd} {entry['model']:<12} {entry['variant']:<26} tg128 {rec['tg128']:8.1f} "
                      f"pp512 {rec['pp512']:8.1f}  vram {rec['vram_mb']}  clock {rec['sm_clock_mhz']}", flush=True)
            except (RuntimeError, FileNotFoundError, subprocess.SubprocessError) as e:
                rec["error"] = str(e)[-3000:]
                print(f"round {rnd} {entry['model']} {entry['variant']}: FAILED {str(e)[:300]}", flush=True)
            runs.append(rec)
            out_json.write_text(json.dumps({"backend": backend, "n_gpu_layers": n_gpu_layers, "reps": reps,
                                            "warmup_s": warmup_s, "runs": runs}, indent=1), encoding="utf-8")
    return runs


def render_markdown(raw: dict) -> str:
    """Markdown tables (per model: every variant's tg128 / pp512 / VRAM) and the gate lines, from a run_plan JSON."""
    runs = [r for r in raw["runs"] if "error" not in r]
    agg = aggregate(runs)
    lines: list[str] = []
    for model, variants in agg.items():
        lines += [f"### {model}", "",
                  "| variant | file MiB | tg128 t/s (mean of the round medians; min..max of them) | pp512 t/s (same) | VRAM MiB | SM clock MHz (min..max sampled, incl. model loading) |",
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
            ok = check_gate(agg, lb_key=lb_key, dense_key=dense_key)
            lines.append(f"- gate {lb_key} > {dense_key} (tg128, {' and '.join(GATE_MODELS)}): **{'PASS' if ok else 'FAIL'}**")
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
    a = ap.parse_args(argv)
    if a.markdown:
        print(render_markdown(json.loads(a.markdown.read_text(encoding="utf-8"))))
        return 0
    if not a.plan or not a.out:
        ap.error("--plan and --out are required unless --markdown is given")
    plan = json.loads(a.plan.read_text(encoding="utf-8"))
    run_plan(plan, a.rounds, a.out, a.backend, a.n_gpu_layers, a.reps, a.warmup)
    print(render_markdown(json.loads(a.out.read_text(encoding="utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
