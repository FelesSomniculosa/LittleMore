"""E. Resource behaviour: GPU memory taken by another process, RAM limits, no CUDA, leaks, a 10-minute soak.

* VRAM starvation: a helper process (PyTorch) allocates and writes 3, 5 or 5.8 GB of GPU memory and keeps it; then a
  child loads a model with the default auto-fit and generates. Windows (WDDM) does not report other processes' GPU
  memory to cudaMemGetInfo (docs/c-api.md, "Known limits"), so the fit may over-commit; required is that nothing
  crashes or hangs and the result is either a working model or a clean error. The same with `vram_budget` set to what
  is really free shows the documented remedy.
* RAM pressure: the child puts itself into a Windows job object with a hard memory limit, so allocations fail for
  real; the load or session must fail with a status, not crash.
* CUDA unavailable: CUDA_VISIBLE_DEVICES=-1, and a copy of the library folder without the CUDA plugin.
* CUDA out of memory: a context whose KV cache cannot fit on the GPU; and GPU memory taken away after the session
  exists (all device memory of a session is reserved when it is created, so generation must go on).
* Leaks: 500 short generations and 100 load/free cycles with private bytes and nvidia-smi samples; a 10-minute soak
  of mixed requests. Peaks per phase are printed for the report.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import _rb
from conftest import ROOT

MEM = """
import ctypes, subprocess
from ctypes import wintypes
class PMC(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t), ('PrivateUsage', ctypes.c_size_t)]
_k32 = ctypes.WinDLL('kernel32'); _k32.GetCurrentProcess.restype = wintypes.HANDLE
_psapi = ctypes.WinDLL('psapi'); _psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
def mem():
    c = PMC(); c.cb = ctypes.sizeof(c)
    assert _psapi.GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return dict(private_mib=c.PrivateUsage / 2**20, ws_mib=c.WorkingSetSize / 2**20, peak_ws_mib=c.PeakWorkingSetSize / 2**20,
                peak_private_mib=c.PeakPagefileUsage / 2**20)
def cuda_used():
    # GPU memory this process holds, MiB (cudaMemGetInfo seen from this process: total minus free); -1 without CUDA
    try:
        rt = ctypes.CDLL('cudart64_12.dll')
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        if rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
            return -1
        return (total.value - free.value) >> 20
    except OSError:
        return -1
def vram():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.split()[0])
    except Exception:
        return -1
"""

HOG = r"""
import sys, time, torch
gb = float(sys.argv[1])
n = int(gb * 2**30)
x = torch.empty(n, dtype=torch.uint8, device='cuda')
x.fill_(1)
torch.cuda.synchronize()
print('HOLDING', gb, flush=True)
time.sleep(float(sys.argv[2]))
"""


def _hog(gb: float, seconds: float = 900) -> subprocess.Popen:
    """Start the helper that holds `gb` GiB of GPU memory; returns once it holds it (or fails)."""
    script = _rb.SCRATCH / "vram_hog.py"
    script.write_text(HOG, encoding="utf-8")
    p = subprocess.Popen([sys.executable, str(script), str(gb), str(seconds)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=_rb.child_env())
    line = p.stdout.readline()
    if "HOLDING" not in line:
        p.kill()
        err = p.stderr.read()
        pytest.skip(f"the helper could not hold {gb} GiB of GPU memory: {err[-400:]}")
    return p


def _gpu_busy_elsewhere() -> int | None:
    """GPU memory in use before the test (another program, e.g. a parallel benchmark, would distort the result)."""
    return _rb.gpu_used_mib()


@pytest.fixture()
def quiet_gpu(has_cuda: bool):
    if not has_cuda:
        pytest.skip("needs CUDA")
    used = _gpu_busy_elsewhere()
    if used is not None and used > 600:
        pytest.skip(f"{used} MiB of GPU memory already in use by other programs: the measurement would be distorted")
    yield used


# --------------------------------------------------------------------------------------------------------------
# VRAM starvation
# --------------------------------------------------------------------------------------------------------------
def _starved_run(model: Path, opts: dict, n_ctx: int, timeout: float = 900) -> _rb.Outcome:
    return _rb.run_child(MEM + f"""
import time
t0 = time.time()
st, m = load({str(model)!r}, {opts!r})
r = dict(load=st, load_s=round(time.time() - t0, 1), error=err()[:300])
if m:
    i = info(m); r.update(n_gpu_layers=i['n_gpu_layers'], n_layer=i['n_layer'], budget_mib=i['fit_budget_bytes'] >> 20)
    st2, s = session(m, {{'n_ctx': {n_ctx}}}); r['session'] = st2; r['session_error'] = err()[:300] if not s else ''
    if s:
        t1 = time.time(); g = generate(s, b'The capital of France is', {{'temperature': 0.0, 'max_tokens': 16}})
        r.update(gen=g[0], gen_s=round(time.time() - t1, 1), pieces=len(g[2].pieces), gen_error=err()[:300] if g[0] != 'LM_OK' else '')
        r['vram_mib'] = vram(); r.update(mem())
        free_session(s)
    free_model(m)
emit(**r)
""", timeout=timeout)


@pytest.mark.parametrize("held_gb", [3.0, 5.0, 5.8])
@pytest.mark.parametrize("model_name", ["qwen06", "gguf_8b"])
def test_vram_starvation(request, quiet_gpu, held_gb: float, model_name: str) -> None:
    model = request.getfixturevalue(model_name)
    hog = _hog(held_gb)
    try:
        free = _rb.gpu_free_mib()
        auto = _starved_run(model, {}, 4096)
        budget = max(0, (free or 0) - 64) << 20
        fitted = _starved_run(model, {"vram_budget": budget or 1}, 4096)
    finally:
        hog.kill()
        hog.wait()
    for label, o in (("auto", auto), ("budget", fitted)):
        print(ascii(f"{model.name} held {held_gb} GiB (nvidia-smi free {free} MiB), {label}: {o.kind} {o.result}"))
        assert o.clean, f"{label}: {o.describe()}"
        r = o.result
        assert r["load"] in ("LM_OK", "LM_ERR_OUT_OF_MEMORY", "LM_ERR_MODEL_LOAD"), r
        if r["load"] == "LM_OK" and r.get("session") == "LM_OK":
            assert r["gen"] in ("LM_OK", "LM_ERR_DECODE", "LM_ERR_OUT_OF_MEMORY"), r


# --------------------------------------------------------------------------------------------------------------
# RAM pressure (Windows job object with a memory limit)
# --------------------------------------------------------------------------------------------------------------
JOB = r"""
import ctypes
from ctypes import wintypes
k32 = ctypes.WinDLL('kernel32', use_last_error=True)
class BASIC(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64), ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]
class IOC(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in ('r', 'w', 'o', 'rb', 'wb', 'ob')]
class EXT(ctypes.Structure):
    _fields_ = [('Basic', BASIC), ('Io', IOC), ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
k32.CreateJobObjectW.restype = wintypes.HANDLE
k32.GetCurrentProcess.restype = wintypes.HANDLE
k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
job = k32.CreateJobObjectW(None, None)
e = EXT(); e.Basic.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
e.ProcessMemoryLimit = LIMIT_MIB << 20
assert k32.SetInformationJobObject(job, 9, ctypes.byref(e), ctypes.sizeof(e)), ctypes.get_last_error()
assert k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()), ctypes.get_last_error()
"""


def _system_commit() -> str:
    """The machine's commit charge right now (GlobalMemoryStatusEx: "page file" = commit limit), for failure messages:
    a child that dies when the WHOLE system is short of commit memory is a different case from one that dies under
    its own job limit."""
    import ctypes
    from ctypes import wintypes

    class MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD)] + [
            (n, ctypes.c_uint64) for n in ("total_phys", "avail_phys", "total_commit", "avail_commit", "total_virtual",
                                           "avail_virtual", "avail_ext")]
    m = MEMSTAT()
    m.dwLength = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return "system commit: unknown"
    return (f"system commit available {m.avail_commit >> 20} of {m.total_commit >> 20} MiB, physical available "
            f"{m.avail_phys >> 20} of {m.total_phys >> 20} MiB")


def _ram_limit_diagnosis(o: _rb.Outcome) -> str:
    """Everything known about one test_ram_limit child, for the assertion messages: how it ended (classification and
    exit code), the last step it reported (see the markers in the snippet), every RESULT line, the full stdout and the
    stderr tail, and the system's commit state just after it ended."""
    last = o.results[-1].get("step", "?") if o.results else "none (died before the first marker)"
    return (f"child ended with {o.kind} (exit {o.returncode}); last step reached: {last}; results: {o.results}; "
            f"stdout: {o.stdout[-3000:]!r}; stderr tail: {o.stderr[-3000:]!r}; {_system_commit()}")


@pytest.mark.parametrize("limit_mib,use_mmap,n_ctx", [(lim, mm, 512) for lim in (200, 300, 420, 600, 900, 1500, 2500)
                                                     for mm in (0, 1)] + [(1500, 1, 40960)])
def test_ram_limit(qwen06: Path, limit_mib: int, use_mmap: int, n_ctx: int) -> None:
    """Private memory of the child is capped (the 0.6B file is 642 MiB; its KV cache at 40960 tokens is 4.4 GiB): the
    load or the session must fail cleanly with a status, or work; never a crash. CPU only (devices="none").
    Before the engine's out-of-memory fixes (docs/engine-changes.md, "Robustness fixes"), limits of 260 to 420 MiB (mmap) aborted the process (ggml_init's GGML_ASSERT on
    its buffer, then GGML_ABORT in ggml_backend_sched_split_graph after unchecked mallocs in ggml_backend_sched_new).

    The child reports a marker after every step ("started": the library is loaded, no limit yet; "job-assigned": the
    limit is in force; then "loaded", "session", "generated", "freed"; the final result has step "end"), and every
    assertion message carries the whole diagnosis (`_ram_limit_diagnosis`).

    KNOWN ISSUE (diagnosed, not fixed yet; see docs/development.md, section 9): at limits of 420 MiB and less the CUDA plugin
    normally cannot be loaded inside the limit, is skipped, and the child passes. When ANOTHER process already has the
    CUDA plugin's DLLs loaded (in a pytest session the `has_cuda` fixture's `backends_info()` call does that in the
    pytest process itself), the limited child can load and initialise it, and then often ends with a fail-fast
    (0xC0000409) during `lm_model_load`, after "loaded CUDA backend" and before the GGUF is read, with no message.
    Measured: 7 of 16 runs of the 300 MiB cases failed after a `has_cuda` test in the same session, 0 of 50 without;
    with a helper process holding ggml-cuda.dll 11 of 32 failed (9 fail-fast; 2 access violations at exit in a shorter
    snippet that did not free the model), and 16 of 16 with the Vulkan plugin removed, while
    0 of 16 failed with the CUDA plugin removed. So run alone (`-k ram_limit`) the test passes; in a full robustness
    run it can fail."""
    o = _rb.run_child("emit(step='started')\n" + JOB.replace("LIMIT_MIB", str(limit_mib)) + f"""
emit(step='job-assigned')
st, m = load({str(qwen06)!r}, {{'devices': 'none', 'use_mmap': {use_mmap}}})
r = dict(load=st, error=err()[:300]); emit(step='loaded', **r)
if m:
    st2, s = session(m, {{'n_ctx': {n_ctx}}}); r['session'] = st2; r['session_error'] = err()[:300]; emit(step='session', **r)
    if s:
        r['gen'] = generate(s, b'Hello', {{'temperature': 0.0, 'max_tokens': 8}})[0]; r['gen_error'] = err()[:300]
        emit(step='generated', **r)
        free_session(s)
    free_model(m); emit(step='freed')
emit(step='end', **r)
""", timeout=600)
    why = _ram_limit_diagnosis(o)
    print(ascii(f"limit {limit_mib} MiB, mmap {use_mmap}, n_ctx {n_ctx}: {o.kind} {o.result} {o.stderr[-300:]}"))
    assert o.clean and o.result.get("step") == "end", why
    r = o.result
    assert r["load"] in ("LM_OK", "LM_ERR_MODEL_LOAD", "LM_ERR_OUT_OF_MEMORY"), why
    if r["load"] != "LM_OK":
        assert r["error"].startswith("lm_model_load"), why
    elif r.get("session") not in (None, "LM_OK"):
        assert r["session"] in ("LM_ERR_OUT_OF_MEMORY", "LM_ERR_SESSION") and r["session_error"].startswith("lm_session_new"), why


# --------------------------------------------------------------------------------------------------------------
# CUDA unavailable
# --------------------------------------------------------------------------------------------------------------
INFO = """
names = [ln.split(':')[0] for ln in _lib.backends_text().splitlines() if ln.strip()]
st, m = load(MODEL, None)
r = dict(backends=names, load=st, error=err()[:200])
if m:
    i = info(m); r['gpu_devices'] = i['gpu_devices']; r['n_gpu_layers'] = i['n_gpu_layers']
    _, s = session(m, {'n_ctx': 128}); r['gen'] = generate(s, b'Hello', {'temperature': 0.0, 'max_tokens': 8})[0]
    free_session(s); free_model(m)
emit(**r)
"""


def test_cuda_hidden(tiny: Path) -> None:
    o = _rb.run_child("from littlemore import _lib\n" + INFO.replace("MODEL", repr(str(tiny))), env={"CUDA_VISIBLE_DEVICES": "-1"})
    print(ascii(o.result))
    assert o.clean, o.describe()
    r = o.result
    assert not any(b.startswith("CUDA") for b in r["backends"]) and "CPU" in r["backends"], r
    assert r["load"] == "LM_OK" and r["gen"] == "LM_OK", r


@pytest.mark.parametrize("remove", [["ggml-cuda.dll"], ["ggml-cuda.dll", "ggml-vulkan.dll"]])
def test_plugin_missing(tiny: Path, lib_path: Path, remove: list[str]) -> None:
    """A copy of the library folder without the CUDA (and Vulkan) plugin: the library loads, lists what is left and
    runs the model there."""
    if os.name != "nt":
        pytest.skip("Windows file names")
    d = _rb.SCRATCH / ("noplugin_" + "_".join(x.split(".")[0] for x in remove))
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in lib_path.parent.iterdir():
        if f.suffix == ".dll" and f.name not in remove:
            shutil.copy2(f, d / f.name)
    o = _rb.run_child("from littlemore import _lib\n" + INFO.replace("MODEL", repr(str(tiny))),
                      env={"LITTLEMORE_LIBRARY": str(d / "littlemore.dll")})
    print(ascii(o.result))
    assert o.clean, o.describe()
    r = o.result
    assert not any(b.startswith("CUDA") for b in r["backends"]) and r["load"] == "LM_OK" and r["gen"] == "LM_OK", r
    if "ggml-vulkan.dll" in remove:
        assert r["backends"] == ["CPU"] and r["gpu_devices"] == "" and r["n_gpu_layers"] == 0, r


def test_only_cpu_plugin_missing_is_a_clean_backend_error(tiny: Path, lib_path: Path) -> None:
    d = _rb.SCRATCH / "nocpu"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in lib_path.parent.iterdir():
        if f.suffix == ".dll" and not f.name.startswith("ggml-cpu"):
            shutil.copy2(f, d / f.name)
    o = _rb.run_child(f"st, m = load({str(tiny)!r}, None)\nemit(load=st, error=err())", env={"LITTLEMORE_LIBRARY": str(d / "littlemore.dll")})
    print(ascii(o.result))
    assert o.clean and o.result["load"] == "LM_ERR_BACKEND" and "CPU" in o.result["error"], o.describe()


# --------------------------------------------------------------------------------------------------------------
# CUDA out of memory
# --------------------------------------------------------------------------------------------------------------
def test_cuda_oom_at_session_and_memory_taken_after_creation(qwen06: Path, quiet_gpu) -> None:
    """A session whose KV cache cannot be allocated on the GPU (n_ctx 400000 on Qwen3-0.6B: about 44 GiB of f16 KV) fails
    with a status. Then a normal session is created, a helper takes all remaining GPU memory, and generation goes on
    (everything a session needs on the device is allocated at creation)."""
    hog_script = _rb.SCRATCH / "vram_hog.py"
    hog_script.write_text(HOG, encoding="utf-8")
    o = _rb.run_child(MEM + f"""
import subprocess, sys
st, m = load({str(qwen06)!r}, {{'devices': 'CUDA0', 'n_gpu_layers': 99}})
r = dict(load=st)
st2, s = session(m, {{'n_ctx': 400000}}); r['huge'] = st2; r['huge_error'] = err()[:300]
st3, s = session(m, {{'n_ctx': 1024}}); r['normal'] = st3
before = generate(s, b'The capital of France is', {{'temperature': 0.0, 'max_tokens': 32}})[1]
free_mib = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()[0])
hog = subprocess.Popen([sys.executable, {str(hog_script)!r}, str(max(0.1, free_mib / 1024 - 0.15)), '120'], stdout=subprocess.PIPE, text=True)
r['hog'] = hog.stdout.readline().strip()
r['free_before_hog_mib'] = free_mib
after = generate(s, b'The capital of France is', {{'temperature': 0.0, 'max_tokens': 32}})
r['after'] = after[0]; r['after_same'] = after[1] == before
long = generate(s, ('word ' * 600).encode(), {{'temperature': 0.0, 'max_tokens': 32}}); r['long_prompt'] = long[0]; r['long_error'] = err()[:200]
hog.kill(); free_session(s); free_model(m)
emit(**r)
""", timeout=600)
    print(ascii(o.result))
    assert o.clean, o.describe()
    r = o.result
    assert r["huge"] in ("LM_ERR_OUT_OF_MEMORY", "LM_ERR_SESSION") and r["huge_error"].startswith("lm_session_new"), r
    assert r["normal"] == "LM_OK" and r["after"] == "LM_OK" and r["after_same"], r
    assert r["long_prompt"] in ("LM_OK", "LM_ERR_DECODE", "LM_ERR_OUT_OF_MEMORY"), r


# --------------------------------------------------------------------------------------------------------------
# Leaks and soak
# --------------------------------------------------------------------------------------------------------------
def _growth(samples: list[dict], key: str) -> float:
    """Growth of `key` between the start (after a warm-up sample) and the end, robust to the allocator's bimodal
    states (measured: the CPU heap of a load/free cycle alternates between two levels 32 MiB apart, with no trend over
    200 cycles): the change of the floor (minimum) and of the ceiling (maximum) of the first and last samples (up to five each, never overlapping),
    whichever is larger."""
    vals = [x[key] for x in samples[1:]]
    k = max(1, min(5, len(vals) // 2))  # non-overlapping head and tail, up to five samples each
    head, tail = vals[:k], vals[-k:]
    return max(min(tail) - min(head), max(tail) - max(head))


LEAK_GEN = MEM + """
import time
st, m = load(MODEL, OPTS)
_, s = session(m, {'n_ctx': 256})
samples = []
t0 = time.time()
for i in range(__N__):
    g = generate(s, b'The capital of France is', {'temperature': 0.0, 'max_tokens': 8})
    assert g[0] == 'LM_OK', (i, g[0], err())
    if i % 50 == 0 or i == __N__ - 1:
        samples.append(dict(i=i, vram=vram(), own=cuda_used() if OWN else -1, **mem()))
emit(samples=samples, seconds=round(time.time() - t0, 1))
free_session(s); free_model(m)
"""


@pytest.mark.parametrize("devices", ["none", "CUDA0"])
def test_no_leak_over_500_generations(qwen06: Path, has_cuda: bool, devices: str) -> None:
    """500 sequential short generations on one session: private bytes from generation 50 to 499 within 16 MiB, GPU
    memory (nvidia-smi, all processes) within 64 MiB."""
    if devices != "none" and not has_cuda:
        pytest.skip("needs CUDA")
    o = _rb.run_child(LEAK_GEN.replace("MODEL", repr(str(qwen06))).replace("OPTS", repr({"devices": devices})).replace("__N__", "500").replace("OWN", str(devices != "none")),
                      timeout=1800)
    assert o.clean, o.describe()
    s = o.result["samples"]
    print(ascii(json.dumps(s[:2] + s[-2:])), o.result["seconds"], "s")
    grow = _growth(s, "private_mib")
    vgrow = _growth(s, "own") if devices != "none" else 0
    print(f"{devices}: private growth {grow:.1f} MiB, own GPU memory growth {vgrow} MiB (nvidia-smi {s[1]['vram']} -> "
          f"{s[-1]['vram']} MiB), peak working set {s[-1]['peak_ws_mib']:.0f} MiB")
    assert grow <= 16 and vgrow <= 64, (grow, vgrow)


LEAK_LOAD = MEM + """
import time
samples = []
for i in range(__N__):
    st, m = load(MODEL, OPTS)
    assert st == 'LM_OK', (i, st, err())
    _, s = session(m, {'n_ctx': 256})
    assert generate(s, b'Hi', {'temperature': 0.0, 'max_tokens': 2})[0] == 'LM_OK'
    free_session(s); free_model(m)
    if i % 10 == 0 or i == __N__ - 1:
        samples.append(dict(i=i, vram=vram(), own=cuda_used() if OWN else -1, **mem()))
emit(samples=samples)
"""


@pytest.mark.parametrize("devices", ["none", "CUDA0"])
def test_no_leak_over_100_load_free_cycles(qwen06: Path, has_cuda: bool, devices: str) -> None:
    """100 load / session / generate / free cycles of Qwen3-0.6B: private bytes after cycle 10 and after cycle 99
    within 32 MiB, GPU memory within 64 MiB."""
    if devices != "none" and not has_cuda:
        pytest.skip("needs CUDA")
    o = _rb.run_child(LEAK_LOAD.replace("MODEL", repr(str(qwen06))).replace("OPTS", repr({"devices": devices})).replace("__N__", "100").replace("OWN", str(devices != "none")),
                      timeout=3600)
    assert o.clean, o.describe()
    s = o.result["samples"]
    grow = _growth(s, "private_mib")
    vgrow = _growth(s, "vram") if devices != "none" else 0   # the model is freed each cycle: nvidia-smi between cycles
    print(ascii(json.dumps([(x["i"], round(x["private_mib"], 1), x["vram"]) for x in s])))
    print(f"{devices}: 100 cycles, private growth {grow:.1f} MiB, VRAM growth {vgrow} MiB, peak working set {s[-1]['peak_ws_mib']:.0f} MiB")
    assert grow <= 32 and vgrow <= 64, (grow, vgrow)


SOAK = MEM + r"""
import time, random, threading
rng = random.Random(0)
st, m = load(MODEL, None)
assert st == 'LM_OK', err()
sessions = [session(m, {'n_ctx': 1024})[1] for _ in range(3)]
samples, errors, counts = [], [], {}
t0 = time.time()
next_sample = 0.0
while time.time() - t0 < SECONDS:
    kind = rng.choice(['generate', 'chat', 'stop-early', 'tokenize', 'long', 'bad-sampling', 'threads', 'context-full', 'sampled'])
    counts[kind] = counts.get(kind, 0) + 1
    s = rng.choice(sessions)
    if kind == 'generate':
        r = generate(s, b'The capital of France is', {'temperature': 0.0, 'max_tokens': rng.randint(1, 48)})[0]; ok = r == 'LM_OK'
    elif kind == 'sampled':
        r = generate(s, b'Once upon a time', {'temperature': 0.9, 'seed': rng.randint(0, 2**32 - 2), 'max_tokens': 32})[0]; ok = r == 'LM_OK'
    elif kind == 'chat':
        r = chat(s, [('system', 'Be brief.'), ('user', 'Hello ' * rng.randint(1, 50))], {'temperature': 0.0, 'max_tokens': 24})[0]; ok = r == 'LM_OK'
    elif kind == 'stop-early':
        r = generate(s, b'Count:', {'temperature': 0.0, 'max_tokens': 64}, Collector(stop_after=rng.randint(1, 10)))[0]; ok = r == 'LM_OK'
    elif kind == 'tokenize':
        st_, ids = tokenize(m, ('text ' * rng.randint(1, 2000)).encode()); ok = st_ == 'LM_OK' and detokenize(m, ids)[0] == 'LM_OK'
    elif kind == 'long':
        r = generate(s, ('word ' * rng.randint(100, 900)).encode(), {'temperature': 0.0, 'max_tokens': 8})[0]; ok = r == 'LM_OK'
    elif kind == 'bad-sampling':
        ok = generate(s, b'x', {'max_tokens': 0})[0] == 'LM_ERR_INVALID_ARG'
    elif kind == 'context-full':
        ok = generate(s, ('word ' * 1100).encode(), {'temperature': 0.0, 'max_tokens': 8})[0] == 'LM_ERR_CONTEXT_FULL'
    else:
        res = []
        ts = [threading.Thread(target=lambda ss=ss: res.append(generate(ss, b'Hello', {'temperature': 0.0, 'max_tokens': 16})[0])) for ss in sessions]
        [t.start() for t in ts]; [t.join() for t in ts]; ok = res == ['LM_OK'] * 3
    if not ok:
        errors.append((kind, err()[:200]))
    if time.time() - t0 >= next_sample:
        samples.append(dict(t=round(time.time() - t0), vram=vram(), own=cuda_used(), **mem())); next_sample += 30
samples.append(dict(t=round(time.time() - t0), vram=vram(), own=cuda_used(), **mem()))
emit(samples=samples, errors=errors[:20], n_errors=len(errors), counts=counts)
for s in sessions: free_session(s)
free_model(m)
"""


def test_soak_10_minutes(qwen06: Path, quiet_gpu) -> None:
    """Ten minutes of mixed requests (greedy, sampled, chat, early stop, tokenizer, long prompts, refused requests,
    three sessions in three threads) on Qwen3-0.6B with auto-fit: no unexpected status, private bytes between the
    60-second sample and the end within 32 MiB, GPU memory within 64 MiB."""
    seconds = int(os.environ.get("LITTLEMORE_SOAK_SECONDS", "600"))
    o = _rb.run_child(SOAK.replace("MODEL", repr(str(qwen06))).replace("SECONDS", str(seconds)), timeout=seconds + 600)
    assert o.clean, o.describe()
    r = o.result
    s = r["samples"]
    base = next(x for x in s if x["t"] >= 60) if seconds >= 120 else s[0]
    grow = s[-1]["private_mib"] - base["private_mib"]
    vgrow = s[-1]["own"] - base["own"]   # this process's own GPU memory (nvidia-smi also counts other programs)
    print(ascii(json.dumps([(x["t"], round(x["private_mib"], 1), x["vram"], x["own"]) for x in s])))
    print(ascii(json.dumps({"counts": r["counts"], "n_errors": r["n_errors"], "first": s[0], "base": base, "last": s[-1],
                            "private_growth_mib": round(grow, 1), "vram_growth_mib": vgrow,
                            "peak_vram_mib": max(x["vram"] for x in s)})))
    assert r["n_errors"] == 0, r["errors"]
    assert grow <= 32 and vgrow <= 64, (grow, vgrow)
