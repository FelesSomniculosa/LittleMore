"""B. Option extremes: every field of every lm_* options struct at its edges must be accepted and sane, or refused
with a clear status. Never a crash, a hang, NaN output or a silent clamp to a meaningless value.

Each case runs in its own child process (`_rb.run_child`) so that a crash is a classified failure. "Sane" is checked,
not assumed: an accepted case must also generate (where generation is possible), report what it did (the auto-fit's
n_gpu_layers through lm_model_get_info), and a refused case must name the failing function in lm_last_error().

Fields that the C API does not have (min_p, repeat penalty) are not applicable: the API exposes temperature, top_k,
top_p, seed and max_tokens only (include/littlemore.h, lm_sampling).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

import _rb

INT32_MAX, UINT32_MAX = 2**31 - 1, 2**32 - 1


def _session_case(model: Path, sopts: dict, load: dict | None = None, gen: dict | None = None, timeout: float = 300) -> dict:
    """Load `model`, open a session with `sopts`, and generate 8 greedy tokens if the session opened."""
    load = load if load is not None else {"devices": "none"}
    gen = gen or {"temperature": 0.0, "max_tokens": 8}
    o = _rb.run_child(f"""
st, m = load({str(model)!r}, {load!r})
r = dict(load=st)
if m:
    st2, s = session(m, {sopts!r})
    r['session'] = st2
    r['session_error'] = err() if s is None else ''
    if s:
        g = generate(s, b'Hello world', {gen!r})
        r['gen'] = g[0]; r['text'] = g[1].hex(); r['gen_error'] = err() if g[0] != 'LM_OK' else ''
        r['free_s'] = free_session(s)
    r['free_m'] = free_model(m)
emit(**r)
""", timeout=timeout)
    assert o.clean, f"{sopts} / {load}: {o.describe()}"
    return o.result


# --------------------------------------------------------------------------------------------------------------
# lm_session_options
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_ctx,expect", [
    (0, "ok"), (1, "full"), (2, "full"), (16, "ok"), (512, "ok"), (128, "ok"),      # 128 = the tiny model's maximum
    (129, "ok"), (4096, "ok"), (INT32_MAX, "oom"), (INT32_MAX + 1, "invalid"), (UINT32_MAX - 200, "invalid"),
    (UINT32_MAX, "invalid")])
def test_n_ctx(tiny: Path, n_ctx: int, expect: str) -> None:
    r = _session_case(tiny, {"n_ctx": n_ctx})
    print(n_ctx, r)
    if expect == "ok":
        assert r["session"] == "LM_OK" and r["gen"] == "LM_OK", r
    elif expect == "full":  # a context of 1 or 2 tokens cannot hold the 12-token prompt
        assert r["session"] == "LM_OK" and r["gen"] == "LM_ERR_CONTEXT_FULL", r
    elif expect == "oom":
        assert r["session"] == "LM_ERR_OUT_OF_MEMORY" and r["session_error"].startswith("lm_session_new"), r
    else:
        assert r["session"] == "LM_ERR_INVALID_ARG" and "n_ctx" in r["session_error"], r


@pytest.mark.parametrize("n_batch", [0, 1, 2, 7, 333, 512, 513, 4096, INT32_MAX, UINT32_MAX])
def test_n_batch(tiny: Path, n_batch: int) -> None:
    """Every batch size works (the engine caps it at n_ctx); the text is the same as with the default batch."""
    ref = _session_case(tiny, {"n_ctx": 128})
    r = _session_case(tiny, {"n_ctx": 128, "n_batch": n_batch})
    assert r["session"] == "LM_OK" and r["gen"] == "LM_OK" and r["text"] == ref["text"], (r, ref)


@pytest.mark.parametrize("n_threads,ok", [(0, True), (1, True), (3, True), (16, True), (512, True), (513, False),
                                          (5000, False), (INT32_MAX, False), (-1, False), (-(2**31), False)])
@pytest.mark.parametrize("devices", ["none", None], ids=["cpu", "default"])
def test_n_threads(tiny: Path, n_threads: int, ok: bool, devices) -> None:
    """Thread counts outside [0, LM_MAX_THREADS] are refused. Before the fix INT32_MAX crashed the process (fail-fast)
    and 5000 started 5000 OS threads for every token."""
    r = _session_case(tiny, {"n_ctx": 64, "n_threads": n_threads}, load={"devices": devices} if devices else {})
    if ok:
        assert r["session"] == "LM_OK" and r["gen"] == "LM_OK", r
    else:
        assert r["session"] == "LM_ERR_INVALID_ARG" and "n_threads" in r["session_error"], r


def test_session_reserved_and_struct_size(tiny: Path) -> None:
    o = _rb.run_child(f"""
st, m = load({str(tiny)!r}, {{'devices': 'none'}})
out = {{}}
so = session_options(reserved0=1); out['reserved'] = session(m, so)[0]
for n in (0, 1, 7, 8, 10, 12, 16, 20, 23, 24, 32):
    so = session_options(); so.struct_size = n
    st2, s = session(m, so); out[f'size{{n}}'] = st2
    if s: free_session(s)
emit(**out)
""")
    assert o.clean, o.describe()
    r = o.result
    assert r["reserved"] == "LM_ERR_UNSUPPORTED", r
    for n in (0, 1, 7, 10, 23):  # not the end of a field
        assert r[f"size{n}"] == "LM_ERR_INVALID_ARG", (n, r)
    for n in (8, 12, 16, 20, 24):  # field boundaries (8 = only struct_size, all defaults)
        assert r[f"size{n}"] == "LM_OK", (n, r)
    assert r["size32"] in ("LM_OK", "LM_ERR_UNSUPPORTED"), r  # bytes after a Python struct: zero or not


# --------------------------------------------------------------------------------------------------------------
# lm_model_options
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ngl", [-1, 0, 1, 14, 28, 29, 30, 1000, INT32_MAX, -2, -5, -(2**31)])
def test_n_gpu_layers(qwen06: Path, has_cuda: bool, ngl: int) -> None:
    """Qwen3-0.6B has 28 blocks; 29 = everything including the head. Larger values are clamped by the engine to
    everything (documented: N >= L + 1 offloads all), and lm_model_get_info reports what was used."""
    if not has_cuda:
        pytest.skip("needs a CUDA device")
    o = _rb.run_child(f"""
st, m = load({str(qwen06)!r}, {{'n_gpu_layers': {ngl}, 'devices': 'CUDA0'}})
r = dict(load=st, error=err())
if m:
    i = info(m); r['n_gpu_layers'] = i['n_gpu_layers']; r['n_layer'] = i['n_layer']
    _, s = session(m, {{'n_ctx': 128}})
    g = generate(s, b'The capital of France is', {{'temperature': 0.0, 'max_tokens': 8}})
    r['gen'] = g[0]; r['text'] = g[1].hex()
    free_session(s); free_model(m)
emit(**r)
""", timeout=300)
    assert o.clean, o.describe()
    r = o.result
    print(ngl, {k: v for k, v in r.items() if k != "text"})
    if ngl < -1:
        assert r["load"] == "LM_ERR_INVALID_ARG" and "n_gpu_layers" in r["error"], r
        return
    assert r["load"] == "LM_OK" and r["gen"] == "LM_OK", r
    if ngl == -1:
        assert 0 <= r["n_gpu_layers"] <= 29
    else:
        assert r["n_gpu_layers"] == min(ngl, 29), r


@pytest.mark.parametrize("devices,expect", [
    (None, "LM_OK"), ("", "LM_OK"), ("none", "LM_OK"), ("CUDA0", "LM_OK"), ("cuda0", "?"), ("CUDA9", "LM_ERR_BACKEND"),
    ("nosuchdevice", "LM_ERR_BACKEND"), ("CPU", "LM_ERR_INVALID_ARG"), ("CUDA0,CUDA0", "LM_OK"), (" CUDA0 ", "LM_OK"),
    ("CUDA0,", "LM_ERR_INVALID_ARG"), (",", "LM_ERR_INVALID_ARG"), ("none,CUDA0", "LM_ERR_BACKEND"),
    ("NONE", "?"), ("X" * 100_000, "LM_ERR_BACKEND"), ("CUDA0," * 5000 + "CUDA0", "LM_OK"), ("Vulkan0", "?"),
    ("CUDA0\tCUDA1", "LM_ERR_BACKEND")])
def test_devices(tiny: Path, has_cuda: bool, devices, expect: str) -> None:
    """Device lists, including invalid, empty, very long and oddly spelt ones. "?" = either a clean load or a clean
    error is acceptable (the result is printed): e.g. whether device names are case-insensitive is the engine's."""
    if not has_cuda and devices and "CUDA0" in devices and expect == "LM_OK":
        pytest.skip("needs a CUDA device")
    o = _rb.run_child(f"""
st, m = load({str(tiny)!r}, {{'devices': {devices!r}}} if {devices!r} is not None else None)
r = dict(load=st, error=err()[:300])
if m:
    i = info(m); r['gpu_devices'] = i['gpu_devices']
    _, s = session(m, {{'n_ctx': 64}}); r['gen'] = generate(s, b'Hi', {{'temperature': 0.0, 'max_tokens': 4}})[0]
    free_session(s); free_model(m)
emit(**r)
""")
    assert o.clean, o.describe()
    r = o.result
    print(repr(devices)[:40], r)
    if expect != "?":
        assert r["load"] == expect, r
    if r["load"] == "LM_OK":
        assert r["gen"] == "LM_OK", r
    else:
        assert r["error"].startswith("lm_model_load"), r


def test_model_options_struct_reserved_budget_mmap(tiny: Path, has_cuda: bool) -> None:
    o = _rb.run_child(f"""
out = {{}}
o = model_options(devices='none', reserved0=7); out['reserved'] = load({str(tiny)!r}, o)[0]
for n in (0, 4, 8, 12, 13, 16, 17, 24, 32, 36, 38, 40):
    o = model_options(devices='none'); o.struct_size = n
    st, m = load({str(tiny)!r}, o); out[f'size{{n}}'] = st
    if m: free_model(m)
for mm in (0, 1, 2, -1):
    st, m = load({str(tiny)!r}, {{'devices': 'none', 'use_mmap': mm}}); out[f'mmap{{mm}}'] = st
    if m: free_model(m)
for b in (1, 2**20, 2**64 - 1):
    st, m = load({str(tiny)!r}, {{'vram_budget': b}}); out[f'budget{{b}}'] = (st, info(m)['n_gpu_layers'] if m else None)
    if m: free_model(m)
for c in (0, 1, 128, 2**32 - 1):
    st, m = load({str(tiny)!r}, {{'fit_n_ctx': c}}); out[f'fitctx{{c}}'] = (st, info(m)['n_gpu_layers'] if m else None)
    if m: free_model(m)
emit(**out)
""")
    assert o.clean, o.describe()
    r = o.result
    print(r)
    assert r["reserved"] == "LM_ERR_UNSUPPORTED"
    for n in (0, 4, 13, 17, 38):
        assert r[f"size{n}"] == "LM_ERR_INVALID_ARG", (n, r)
    # 8 = struct_size only: devices stays at its default (all GPUs), still a valid load
    for n in (8, 12, 16, 24, 32, 36, 40):
        assert r[f"size{n}"] == "LM_OK", (n, r)
    assert all(r[f"mmap{mm}"] == "LM_OK" for mm in (0, 1, 2, -1)), r
    if has_cuda:
        assert r["budget1"] == ["LM_OK", 0], r                 # nothing fits into 1 byte
        assert r["budget18446744073709551615"] == ["LM_OK", 3], r  # everything (2 blocks + head) fits a huge budget
        assert r["fitctx4294967295"][0] == "LM_OK", r          # a huge fit context saturates: fewer layers, no crash
    for c in (0, 1, 128):
        assert r[f"fitctx{c}"][0] == "LM_OK", r


# --------------------------------------------------------------------------------------------------------------
# lm_sampling
# --------------------------------------------------------------------------------------------------------------
SAMPLING_CASES = {
    # name: (fields, expectation) ; "greedy" = must equal the greedy text, "ok" = any text, "invalid" = refused
    "temp0": ({"temperature": 0.0}, "greedy"),
    "temp-neg": ({"temperature": -1.0}, "greedy"),
    "temp-neg-inf": ({"temperature": float("-inf")}, "greedy"),
    "temp-nan": ({"temperature": float("nan")}, "invalid"),
    "temp-inf": ({"temperature": float("inf")}, "ok"),
    "temp-1e9": ({"temperature": 1e9}, "ok"),
    "temp-1": ({"temperature": 1.0}, "ok"),
    "temp-1e-5": ({"temperature": 1e-5}, "greedy"),
    "temp-1e-30": ({"temperature": 1e-30}, "greedy"),
    "temp-1e-38": ({"temperature": 1e-38}, "greedy"),
    "temp-denormal": ({"temperature": 1e-45}, "greedy"),   # logit / 1e-45 overflowed to inf -> NaN probabilities before the fix
    "topk0": ({"top_k": 0}, "ok"),
    "topk1": ({"top_k": 1}, "greedy"),
    "topk-neg": ({"top_k": -1}, "invalid"),
    "topk-max": ({"top_k": INT32_MAX}, "ok"),
    "topp0": ({"top_p": 0.0}, "invalid"),
    "topp-tiny": ({"top_p": 1e-30}, "greedy"),
    "topp1": ({"top_p": 1.0}, "ok"),
    "topp1.5": ({"top_p": 1.5}, "invalid"),
    "topp-nan": ({"top_p": float("nan")}, "invalid"),
    "topp-neg": ({"top_p": -0.5}, "invalid"),
    "seed0": ({"seed": 0}, "ok"),
    "seed-random": ({"seed": 0xFFFFFFFF}, "ok"),
    "seed-max-1": ({"seed": 0xFFFFFFFE}, "ok"),
    "max1": ({"max_tokens": 1}, "ok"),
    "max0": ({"max_tokens": 0}, "invalid"),
    "max-neg": ({"max_tokens": -5}, "invalid"),
    "max-huge": ({"max_tokens": INT32_MAX}, "ok"),
    "reserved": ({"reserved0": 1}, "unsupported"),
}


@pytest.mark.parametrize("model_name", ["tiny", "qwen06"])
def test_sampling_extremes(request, model_name: str) -> None:
    """Every case of SAMPLING_CASES on one session (prompt "Hello world", 16 tokens, default top_k 40 / top_p 0.95,
    temperature 1 where the case does not set it). Greedy cases must reproduce the greedy text exactly."""
    model = request.getfixturevalue(model_name)
    o = _rb.run_child(f"""
nan, inf = float('nan'), float('inf')  # repr() of the cases writes these names
cases = {SAMPLING_CASES!r}
st, m = load({str(model)!r}, {{'devices': 'none'}})
_, s = session(m, {{'n_ctx': 128}})
out = {{}}
g = generate(s, b'Hello world', dict(temperature=0.0, max_tokens=16)); out['_greedy'] = (g[0], g[1].hex())
for name_, (fields, _exp) in cases.items():
    f = dict(temperature=1.0, max_tokens=16); f.update(fields)
    st, text, c = generate(s, b'Hello world', f)
    out[name_] = (st, text.hex(), len(c.pieces), err()[:200] if st != 'LM_OK' else '')
emit(**out)
""", timeout=600)
    assert o.clean, o.describe()
    r = o.result
    greedy = r["_greedy"][1]
    bad = []
    for name, (fields, exp) in SAMPLING_CASES.items():
        st, text, n, e = r[name]
        print(f"{model_name} {name}: {st} {n} pieces {e}")
        if exp == "invalid":
            ok = st == "LM_ERR_INVALID_ARG" and e.startswith("lm_generate")
        elif exp == "unsupported":
            ok = st == "LM_ERR_UNSUPPORTED"
        elif exp == "greedy":
            ok = st == "LM_OK" and text == greedy
        else:
            ok = st == "LM_OK" and (n >= 1 or name == "max1")
        if not ok:
            bad.append((name, r[name]))
    assert not bad, bad


def test_seed_semantics(tiny: Path) -> None:
    """A fixed seed reproduces its text; LM_SEED_RANDOM gives different texts across calls (a 300-token vocabulary at
    temperature 2 over 32 tokens: identical random texts would be a broken seed); two fixed seeds differ."""
    o = _rb.run_child(f"""
st, m = load({str(tiny)!r}, {{'devices': 'none'}})
_, s = session(m, {{'n_ctx': 128}})
def t(seed): return generate(s, b'Hello world', dict(temperature=2.0, top_k=0, top_p=1.0, max_tokens=32, seed=seed))[1]
emit(fixed=[t(7) == t(7), t(0) == t(0)], fixed_differ=t(7) != t(8), random=len({{t(0xFFFFFFFF) for _ in range(4)}}))
""")
    assert o.clean, o.describe()
    r = o.result
    assert r["fixed"] == [True, True] and r["fixed_differ"] and r["random"] >= 3, r


# --------------------------------------------------------------------------------------------------------------
# Python SDK: no silent wraparound of the numbers it passes to the C structs
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [{"top_k": 2**32 + 5}, {"top_k": 2**31}, {"max_tokens": 2**32 + 1}, {"max_tokens": 2**31},
                                {"top_k": 1.5}, {"max_tokens": 2.5}, {"top_k": True}])
def test_sdk_sampling_ints_are_not_wrapped(lib_path: Path, kw: dict) -> None:
    """ctypes stores Python ints into int32 fields modulo 2**32: top_k=2**32+5 used to become 5 and
    max_tokens=2**32+1 became 1, silently. The SDK must refuse such values (ValueError / TypeError)."""
    from littlemore import model as M
    with pytest.raises((ValueError, TypeError)):
        M._make_sampling(**kw)


def test_sdk_sampling_accepts_integer_likes(lib_path: Path) -> None:
    """Integer-like values (numpy integers) stay accepted; the struct holds exactly the value given."""
    import numpy as np
    from littlemore import model as M
    s = M._make_sampling(top_k=np.int64(7), max_tokens=np.int32(9), seed=np.uint32(5))
    assert (s.top_k, s.max_tokens, s.seed) == (7, 9, 5)


def test_sdk_session_limits(lib_path: Path) -> None:
    import littlemore
    for kw in ({"n_threads": 513}, {"n_ctx": 2**31}):
        with pytest.raises(ValueError):
            littlemore.Model("does-not-matter.gguf", **kw)


def test_sdk_api_version_mismatch_is_refused(lib_path: Path) -> None:
    """A library of another API version is refused before any struct is used (checked in a child: it must not
    affect this process's loaded library)."""
    o = _rb.run_child("""
from littlemore import _lib as L
class Fake:
    def lm_api_version(self): return L.EXPECTED_API_VERSION + 1
try:
    L._check_api_version(Fake()); emit(refused=False)
except L.LibraryNotFoundError as e:
    emit(refused=True, msg=str(e))
emit(c_version=lib.lm_api_version())
""")
    assert o.clean and o.results[0]["refused"] and o.results[1]["c_version"] == 1, o.describe()
