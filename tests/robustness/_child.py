"""Child-process driver of the robustness tests: runs one snippet of Python against the raw C API and reports.

Why a child process: the robustness tests feed the library broken files, absurd options and misuse. A defect may abort
or crash the process (GGML_ASSERT -> abort(), an access violation, a hang). Running every probe in its own process
with a timeout turns such a defect into a classified test failure (see `_rb.run_child`) instead of killing pytest.

Usage (the tests do this through `_rb.run_child`):

    python _child.py <python source of the snippet>

The snippet runs with the helpers below in scope and reports with `emit(...)`, which prints one line
`RESULT {json}`; the parent collects those lines. Nothing here uses the high-level `littlemore.Model` class: the
helpers call the C functions directly through the SDK's ctypes declarations (`littlemore._lib`), so a test can pass
anything the C API accepts, including invalid UTF-8, NUL bytes, garbage struct sizes or freed handles.

Windows error dialogs are switched off first (SetErrorMode), so a crash ends the process at once with its
exception code instead of waiting for someone to close a "program has stopped working" window.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

if os.name == "nt":
    # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX: no modal error boxes on a crash.
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    try:
        # abort() must not show the "abort() has been called" dialog or call Windows Error Reporting:
        # _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT). The process still ends with exit code 3.
        ctypes.cdll.ucrtbase._set_abort_behavior(0, 0x1 | 0x2)
    except (OSError, AttributeError):
        pass

from littlemore import _lib  # noqa: E402  (path set up above)

lib = _lib.load()
c_vp = ctypes.c_void_p


def emit(**kw) -> None:
    """Report a result to the parent (one JSON line; flushed at once so it survives a crash right after)."""
    print("RESULT " + json.dumps(kw, default=str), flush=True)


def name(status: int) -> str:
    """Constant name of an lm_status value, e.g. "LM_ERR_MODEL_LOAD"."""
    return lib.lm_status_name(status).decode()


def err() -> str:
    """lm_last_error() of this thread."""
    return lib.lm_last_error().decode("utf-8", "replace")


def _fill(struct, values: dict | None) -> None:
    """Set the fields of a ctypes struct from a dict (strings are encoded as UTF-8 for char* fields)."""
    for k, v in (values or {}).items():
        if isinstance(v, str):
            v = v.encode("utf-8")
        setattr(struct, k, v)


def model_options(**values) -> _lib.ModelOptions:
    """lm_model_options from the library defaults, with `values` applied."""
    o = _lib.ModelOptions()
    lib.lm_model_options_default(ctypes.byref(o))
    _fill(o, values)
    return o


def session_options(**values) -> _lib.SessionOptions:
    """lm_session_options from the library defaults, with `values` applied."""
    o = _lib.SessionOptions()
    lib.lm_session_options_default(ctypes.byref(o))
    _fill(o, values)
    return o


def sampling(**values) -> _lib.Sampling:
    """lm_sampling from the library defaults, with `values` applied."""
    s = _lib.Sampling()
    lib.lm_sampling_default(ctypes.byref(s))
    _fill(s, values)
    return s


def load(path, opts=None):
    """lm_model_load. `path` str or bytes; `opts` an lm_model_options, a dict of fields, or None (NULL).
    Returns (status name, model handle or None)."""
    if isinstance(path, str):
        path = path.encode("utf-8")
    if isinstance(opts, dict):
        opts = model_options(**opts)
    h = c_vp()
    st = lib.lm_model_load(path, ctypes.byref(opts) if opts is not None else None, ctypes.byref(h))
    return name(st), (h if h.value else None)


def session(model, opts=None):
    """lm_session_new with an lm_session_options, a dict of fields, or None. Returns (status name, handle or None)."""
    if isinstance(opts, dict):
        opts = session_options(**opts)
    h = c_vp()
    st = lib.lm_session_new(model, ctypes.byref(opts) if opts is not None else None, ctypes.byref(h))
    return name(st), (h if h.value else None)


class Collector:
    """Token callback that collects the pieces; `stop_after` > 0 returns non-zero after that many pieces."""

    def __init__(self, stop_after: int = 0) -> None:
        self.pieces: list[bytes] = []
        self.stop_after = stop_after
        self.cb = _lib.TokenCallback(self._on)

    def _on(self, ptr, length, _user) -> int:
        self.pieces.append(ctypes.string_at(ptr, length))
        return 1 if self.stop_after and len(self.pieces) >= self.stop_after else 0

    @property
    def text(self) -> bytes:
        return b"".join(self.pieces)


def generate(sess, prompt: bytes | str, smp=None, collector: Collector | None = None):
    """lm_generate; `smp` an lm_sampling, a dict, or None. Returns (status name, generated bytes, collector)."""
    if isinstance(prompt, str):
        prompt = prompt.encode("utf-8")
    if isinstance(smp, dict):
        smp = sampling(**smp)
    c = collector or Collector()
    st = lib.lm_generate(sess, prompt, ctypes.byref(smp) if smp is not None else None, c.cb, None)
    return name(st), c.text, c


def chat(sess, messages, smp=None, collector: Collector | None = None):
    """lm_chat over [(role, content), ...] (str or bytes). Returns (status name, generated bytes, collector)."""
    arr = (_lib.Message * max(len(messages), 1))()
    keep = []
    for i, (role, content) in enumerate(messages):
        role = role.encode("utf-8") if isinstance(role, str) else role
        content = content.encode("utf-8") if isinstance(content, str) else content
        keep += [role, content]
        arr[i].role, arr[i].content = role, content
    if isinstance(smp, dict):
        smp = sampling(**smp)
    c = collector or Collector()
    st = lib.lm_chat(sess, arr, len(messages), ctypes.byref(smp) if smp is not None else None, c.cb, None)
    return name(st), c.text, c


def tokenize(model, text: bytes | str, add_special: bool = False):
    """lm_tokenize (two calls: size, then ids). Returns (status name, list of ids)."""
    if isinstance(text, str):
        text = text.encode("utf-8")
    n = ctypes.c_size_t(0)
    st = lib.lm_tokenize(model, text, len(text), None, 0, int(add_special), ctypes.byref(n))
    if st not in (0, _lib.LM_ERR_BUFFER_TOO_SMALL):
        return name(st), []
    buf = (ctypes.c_int32 * max(n.value, 1))()
    st = lib.lm_tokenize(model, text, len(text), buf, n.value, int(add_special), ctypes.byref(n))
    return name(st), list(buf[: n.value])


def detokenize(model, ids, special: bool = False):
    """lm_detokenize (two calls). Returns (status name, bytes)."""
    arr = (ctypes.c_int32 * max(len(ids), 1))(*ids)
    n = ctypes.c_size_t(0)
    st = lib.lm_detokenize(model, arr, len(ids), None, 0, int(special), ctypes.byref(n))
    if st not in (0, _lib.LM_ERR_BUFFER_TOO_SMALL):
        return name(st), b""
    buf = ctypes.create_string_buffer(n.value + 1)
    st = lib.lm_detokenize(model, arr, len(ids), buf, len(buf), int(special), ctypes.byref(n))
    return name(st), buf.raw[: n.value]


def info(model) -> dict:
    """lm_model_get_info as a dict (strings decoded)."""
    mi = _lib.ModelInfo()
    mi.struct_size = ctypes.sizeof(_lib.ModelInfo)
    st = lib.lm_model_get_info(model, ctypes.byref(mi))
    out = {"status": name(st)}
    for f, _ in _lib.ModelInfo._fields_:
        v = getattr(mi, f)
        out[f] = v.decode("utf-8", "replace") if isinstance(v, bytes) else v
    return out


def free_session(sess) -> str:
    return name(lib.lm_session_free(sess))


def free_model(model) -> str:
    return name(lib.lm_model_free(model))


def exercise(path, opts=None, n_ctx: int = 64, max_tokens: int = 4) -> dict:
    """Load `path`; if that works, also open a session, generate a few greedy tokens, tokenize and free everything.
    A broken file that loads must still not crash at run time. Returns a dict of the statuses."""
    r: dict = {}
    r["load"], m = load(path, opts)
    if m is None:
        r["load_error"] = err()
        return r
    r["session"], s = session(m, {"n_ctx": n_ctx})
    if s is not None:
        r["generate"], text, _ = generate(s, b"Hello world", {"temperature": 0.0, "max_tokens": max_tokens})
        r["text_len"] = len(text)
        if r["generate"] != "LM_OK":
            r["generate_error"] = err()
        r["free_session"] = free_session(s)
    else:
        r["session_error"] = err()
    r["tokenize"], ids = tokenize(m, b"Hello world")
    r["n_ids"] = len(ids)
    r["free_model"] = free_model(m)
    return r


if __name__ == "__main__":
    src = sys.argv[1]
    if src.startswith("@"):
        src = Path(src[1:]).read_text(encoding="utf-8")
    try:
        exec(compile(src, "<snippet>", "exec"), dict(globals(), __name__="__snippet__"))
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - a Python-level error of the snippet is reported, exit code 1
        traceback.print_exc()
        sys.exit(1)
