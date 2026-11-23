"""F. Sanitizers: NVIDIA compute-sanitizer over the CUDA LittleBit kernels, and MSVC AddressSanitizer over the C API.

compute-sanitizer (CUDA 12.9 toolkit) runs four tools: memcheck (out-of-bounds and misaligned device accesses),
racecheck (shared-memory data races), synccheck (invalid __syncthreads / warp synchronisation) and initcheck (reads of
uninitialised device memory). They run over:
  * `test-backend-ops test -o LITTLEBIT -b CUDA0`: the op-level cases of GGML_OP_LITTLEBIT (every kernel path at its
    batch sizes), once with the tensor-core prefill path and once with GGML_CUDA_LITTLEBIT_F32_PREFILL=1 (the
    chunked-decode bands and the float32 GEMM path);
  * `llama-debug` on the tiny model (whole graph on CUDA) at prompt lengths at the C dispatch boundaries.
A run passes when the tool reports "ERROR SUMMARY: 0 errors" and the program itself succeeded.

AddressSanitizer: a separate CPU-only build with /fsanitize=address (.cache/robust/build-asan, configure line in
docs/development.md, section 9) of liblittlemore, the engine, tests/c/test_api.c and the generated NULL harness.
ASan's quarantine holds freed memory back on purpose, which test_api's own leak check (private bytes) would read as a
leak, so that run sets quarantine_size_mb=0; a second run of the NULL harness keeps the default quarantine.

Slow (racecheck in particular): opt-in like the rest of tests/robustness.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

import _rb
from conftest import ENGINE_BIN, ROOT

EXE = ".exe" if os.name == "nt" else ""
SANITIZER = Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/compute-sanitizer/compute-sanitizer.exe")
ASAN_BIN = ROOT / ".cache" / "robust" / "build-asan" / "bin"
TOOLS = ["memcheck", "racecheck", "synccheck", "initcheck"]
# Prompt lengths for the whole-graph runs: decode (1, 8), first prefill (9), chunk-band edges (16/17, 24/25), larger.
LENGTHS = [1, 8, 9, 17, 25, 33, 64, 129]


def _sanitize(tool: str, cmd: list[str], env: dict | None = None, timeout: float = 7200) -> tuple[int, str]:
    if not SANITIZER.exists():
        pytest.skip(f"{SANITIZER} not found")
    # --report-api-errors no: CUDA API calls that return an error on purpose are not kernel faults. ggml-cuda tries to
    # update an instantiated CUDA graph in place and, when the update is refused (cudaErrorGraphExecUpdateFailure),
    # instantiates it again; memcheck counted each such refusal (144 in the op test, every printed one of this kind).
    # initcheck: --check-api-memory-access no. test-backend-ops copies whole graphs between backends
    # (ggml_backend_graph_copy -> ggml_backend_tensor_get) BEFORE computing them, i.e. it reads destination tensors that
    # are not written yet; initcheck reported each such cudaMemcpy (2.37 million, every printed one with that
    # backtrace). Without the API checks it reports what kernels read uninitialised.
    extra = ["--check-api-memory-access", "no"] if tool == "initcheck" else []
    p = subprocess.run([str(SANITIZER), "--tool", tool, "--print-limit", "50", "--report-api-errors", "no", *extra, *cmd],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout, env=_rb.child_env(env), stdin=subprocess.DEVNULL)
    return p.returncode, p.stdout + p.stderr


def _summary(out: str) -> str:
    """Error count of the sanitizer's summary line (racecheck prints "RACECHECK SUMMARY: N hazards displayed (E errors,
    W warnings)"; the others "ERROR SUMMARY: N errors"); "?" when there is none."""
    m = re.findall(r"RACECHECK SUMMARY: \d+ hazards displayed \((\d+) errors, (\d+) warnings\)", out)
    if m:
        return str(int(m[-1][0]) + int(m[-1][1]))
    m = re.findall(r"ERROR SUMMARY: (\d+) error", out)
    return m[-1] if m else "?"


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("path", ["bp", "f32"])
def test_compute_sanitizer_littlebit_op(has_cuda: bool, tool: str, path: str) -> None:
    if not has_cuda:
        pytest.skip("needs CUDA")
    env = {"GGML_CUDA_LITTLEBIT_F32_PREFILL": "1"} if path == "f32" else None
    rc, out = _sanitize(tool, [str(ENGINE_BIN / ("test-backend-ops" + EXE)), "test", "-o", "LITTLEBIT", "-b", "CUDA0"], env)
    log = _rb.SCRATCH / f"sanitizer_op_{tool}_{path}.log"
    log.write_text(out, encoding="utf-8")
    print(f"{tool} {path}: exit {rc}, errors {_summary(out)}, {'OK' if 'backends passed' in out else 'no pass line'} ({log})")
    assert _summary(out) == "0" and rc == 0 and "4/4 backends passed" in out, out[-3000:]


@pytest.mark.parametrize("tool", TOOLS)
def test_compute_sanitizer_whole_graph(tiny: Path, has_cuda: bool, tool: str) -> None:
    """llama-debug on the tiny model with every layer on CUDA, at each prompt length, both prefill paths."""
    if not has_cuda:
        pytest.skip("needs CUDA")
    words = ("one two three four five six seven eight nine ten " * 20).split()
    bad = []
    # racecheck and synccheck are 10x to 100x slower: they run the decode, first-prefill and a larger length only
    lengths = LENGTHS if tool in ("memcheck", "initcheck") else [1, 9, 33]
    for n in lengths:
        for env in (None, {"GGML_CUDA_LITTLEBIT_F32_PREFILL": "1"}):
            prompt = " ".join(words[:n])
            rc, out = _sanitize(tool, [str(ENGINE_BIN / ("llama-debug" + EXE)), "-m", str(tiny), "-p", prompt, "-ngl", "99",
                                       "-c", "1024", "-b", "1024", "--save-logits", "--logits-output-dir", str(_rb.SCRATCH / "san_logits")], env)
            s = _summary(out)
            print(f"{tool} n~{n} {'f32' if env else 'bp'}: exit {rc}, errors {s}")
            if s != "0" or rc != 0:
                bad.append((n, env, rc, s, out[-2000:]))
    assert not bad, bad[:2]


def _asan_runtime_dir() -> Path | None:
    """The MSVC folder that holds the AddressSanitizer runtime DLL (clang_rt.asan_dynamic-x86_64.dll)."""
    root = Path(r"C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC")
    hits = sorted(root.glob("*/bin/Hostx64/x64/clang_rt.asan_dynamic-x86_64.dll"))
    return hits[-1].parent if hits else None


def _asan(cmd: list[str], options: str) -> subprocess.CompletedProcess:
    if not (ASAN_BIN / ("test_api" + EXE)).exists():
        pytest.skip(f"AddressSanitizer build not found in {ASAN_BIN} (configure line in docs/development.md, section 9)")
    rt = _asan_runtime_dir()
    if rt is None:
        pytest.skip("MSVC AddressSanitizer runtime not found")
    env = _rb.child_env({"LITTLEMORE_LOG": "error", "ASAN_OPTIONS": options})
    env["PATH"] = str(rt) + os.pathsep + env["PATH"]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, env=env,
                          stdin=subprocess.DEVNULL)


@pytest.mark.parametrize("prog,options", [
    ("test_api", "halt_on_error=1:windows_hook_rtl_allocators=true:quarantine_size_mb=0"),
    ("null_harness", "halt_on_error=1:windows_hook_rtl_allocators=true"),
    ("null_harness", "halt_on_error=1:windows_hook_rtl_allocators=true:quarantine_size_mb=0")])
def test_address_sanitizer(tiny: Path, prog: str, options: str) -> None:
    exe = ASAN_BIN / (prog + EXE)
    if not exe.exists():
        pytest.skip(f"{exe} not built")
    args = [str(exe), str(tiny)] + (["cpu"] if prog == "test_api" else [])
    p = _asan(args, options)
    out = p.stdout + p.stderr
    print(ascii(f"{prog} [{options}]: exit {p.returncode}; {out.strip().splitlines()[-1] if out.strip() else ''}"))
    assert "AddressSanitizer" not in out and p.returncode == 0, out[-4000:]
