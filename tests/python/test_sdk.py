"""Tests of the LittleMore Python SDK (`python/littlemore`) and its command line.

What is covered
    * the five basic behaviours: streaming yields pieces, `chat` applies the model's chat template, `info()` lists
      the compute backends, a bad path raises `LittleMoreError`, and `littlemore convert` then `littlemore run`
      works end to end on a tiny synthetic model;
    * the ctypes struct layouts against the C header (parsed here, independently of the SDK) and against the
      shared library itself (it accepts a `struct_size` only on a field boundary);
    * lifetime rules: `close()` is idempotent, sessions are freed before models, use after close is an error;
    * streaming rules: an abandoned or garbage-collected stream stops generation and releases the session lock,
      UTF-8 sequences split across pieces are decoded without errors, two threads can chat on one model;
    * the Windows DLL-directory rule: CUDA0 must show up in `info()` from a plain `python.exe` run from another
      working directory with no CUDA folder on PATH (skipped when there is no CUDA device or runtime).

Test tiers
    * Tests that need only the built shared library (layout, backends, bad path, CUDA visibility) are in the
      default suite; they are skipped, with a reason, when the library is not built.
    * Tests that also need the tiny model (a 0.7 MB random LittleBit GGUF that tests/c/make_tiny_gguf.py builds
      and caches under .cache/) or build a tiny checkpoint are marked `smoke` like the other end-to-end tests.
    The model is random, so its text is gibberish; the tests check plumbing, never quality.
"""
from __future__ import annotations

import ctypes
import gc
import importlib.util
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PYDIR = ROOT / "python"
HEADER = ROOT / "include" / "littlemore.h"
TINY_GGUF = ROOT / ".cache" / "gguf" / "tiny-llama-lb2-chatml.gguf"

import littlemore  # noqa: E402  (python/ is on pytest's pythonpath, see pytest.ini)
from littlemore import _lib  # noqa: E402
from littlemore import model as _model_mod  # noqa: E402


# --------------------------------------------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def lib_path() -> Path:
    """Path of the built shared library, or skip the test with the reason."""
    try:
        return _lib.find_library()
    except _lib.LibraryNotFoundError as e:  # pragma: no cover - depends on the machine
        pytest.skip(f"littlemore shared library not built: {e}")


@pytest.fixture(scope="session")
def tiny_gguf(lib_path: Path) -> Path:
    """The tiny test GGUF (built once by tests/c/make_tiny_gguf.py, cached), or skip."""
    spec = importlib.util.spec_from_file_location("make_tiny_gguf", ROOT / "tests" / "c" / "make_tiny_gguf.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod.build(TINY_GGUF)
    except ImportError as e:  # pragma: no cover - missing torch/transformers
        pytest.skip(f"cannot build the tiny test model: {e!r}")


@pytest.fixture()
def model(tiny_gguf: Path):
    """A CPU-only tiny model (devices="none": no GPU memory touched, fast and deterministic)."""
    m = littlemore.Model(tiny_gguf, devices="none", n_ctx=512)
    yield m
    m.close()


def _cli(*args: str, timeout: int = 600, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run `python -m littlemore <args>` against the source tree in a child process (UTF-8 output)."""
    env = dict(os.environ, PYTHONPATH=str(PYDIR), PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-m", "littlemore", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, env=env, cwd=cwd,
                          stdin=subprocess.DEVNULL)


# --------------------------------------------------------------------------------------------------------------
# The five basic behaviours
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.smoke
def test_generate_stream_yields_tokens(model) -> None:
    pieces = list(model.generate("Hello", stream=True, max_tokens=8))
    assert len(pieces) >= 2, "a stream should yield several pieces, not one blob"
    assert all(isinstance(p, str) and p for p in pieces)
    # Streaming and non-streaming give the same text (the default sampling is deterministic).
    assert "".join(pieces) == model.generate("Hello", stream=False, max_tokens=8)


@pytest.mark.smoke
def test_chat_applies_template(model) -> None:
    """`chat` formats the messages with the model's ChatML template: its answer to a user message equals the
    answer `generate` gives to the same conversation written out as ChatML text (greedy, and also with a seeded
    sampler, where equal text is much stronger evidence), and differs when the message or the system prompt changes."""
    def framed(content: str) -> str:
        """The ChatML text of one user turn, as the model's chat template renders it."""
        return "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n"

    user = [{"role": "user", "content": "Hello"}]
    greedy = dict(temperature=0, max_tokens=12)
    assert model.chat(user, **greedy) == model.generate(framed("Hello"), **greedy)
    sampled = dict(temperature=1.5, seed=7, max_tokens=16)  # varied text: an accidental match is very unlikely
    a = model.chat(user, **sampled)
    assert a == model.generate(framed("Hello"), **sampled)
    assert a == model.chat(user, **sampled), "the same seed must give the same text"
    assert a != model.chat([{"role": "user", "content": "Goodbye"}], **sampled), "the user text must reach the model"
    with_system = [{"role": "system", "content": "Be brief."}] + user
    assert a != model.chat(with_system, **sampled), "the system message must reach the model"
    assert a != model.generate("Hello", **sampled), "generate must not apply the template"


def test_info_lists_backends(lib_path: Path) -> None:
    info = littlemore.backends_info()
    names = [b["name"] for b in info]
    assert "CPU" in names, info
    assert all(b["description"] for b in info)


def test_version_comes_from_one_place(lib_path: Path) -> None:
    """The package version, the header's LM_VERSION_* and the loaded library all agree."""
    text = HEADER.read_text(encoding="utf-8")
    parts = [re.search(rf"#define LM_VERSION_{n} (\d+)", text).group(1) for n in ("MAJOR", "MINOR", "PATCH")]
    assert littlemore.version() == ".".join(parts) == littlemore.__version__


@pytest.mark.smoke
def test_model_info_fields(model) -> None:
    info = model.info()
    assert info["library_version"] == littlemore.version()
    assert any(b["name"] == "CPU" for b in info["backends"])
    m = info["model"]
    assert m["n_layer"] == 2 and m["n_vocab"] == 300 and m["n_embd"] == 256
    assert m["n_gpu_layers"] == 0 and m["gpu_devices"] == ""  # devices="none": nothing on a GPU
    assert m["size_bytes"] > 0 and m["n_params"] > 0 and m["description"]


def test_bad_path_raises_littlemore_error(lib_path: Path, tmp_path: Path) -> None:
    with pytest.raises(littlemore.LittleMoreError) as ei:
        littlemore.Model(tmp_path / "does-not-exist.gguf")
    err = ei.value
    assert err.status_name == "LM_ERR_MODEL_LOAD" and err.status == 4
    assert "lm_model_load" in str(err), "the message must carry lm_last_error() text"
    # A directory and a non-GGUF file are also reported as errors, not crashes.
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"not a gguf file at all")
    for bad in (tmp_path, junk):
        with pytest.raises(littlemore.LittleMoreError):
            littlemore.Model(bad)


@pytest.mark.smoke
def test_cli_convert_then_run(tmp_path: Path, lib_path: Path, monkeypatch) -> None:
    """`littlemore convert` on a tiny synthetic checkpoint gives a GGUF that `littlemore run --prompt` answers."""
    for p in (ROOT / "tools" / "dev", ROOT / "tests" / "convert", ROOT):
        monkeypatch.syspath_prepend(str(p))  # undone when the test ends
    try:
        import _tiny
        ck = _tiny.make_tiny_checkpoint("llama", tmp_path, bpw=2.0)
    except ImportError as e:  # pragma: no cover - missing torch/transformers
        pytest.skip(f"cannot build a tiny checkpoint: {e!r}")
    out = tmp_path / "tiny.gguf"
    r = _cli("convert", "--repo", str(ROOT), str(ck), "-o", str(out))
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.stat().st_size > 10_000
    r = _cli("run", str(out), "--prompt", "Hello", "--max-tokens", "6", "--cpu-only")
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip(), "the answer must be printed to stdout"
    # A missing model is a clean error message and a non-zero exit code, not a traceback.
    r = _cli("run", str(tmp_path / "nope.gguf"), "--prompt", "x")
    assert r.returncode != 0 and "Traceback" not in r.stderr and "lm_model_load" in r.stderr


@pytest.mark.smoke
def test_cli_bench_smoke(tiny_gguf: Path) -> None:
    r = _cli("bench", str(tiny_gguf), "--cpu-only", "--n-prompt", "16", "--n-gen", "8", "--reps", "1", "--json")
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(r.stdout)
    for key in ("load_s", "first_piece_ms", "generation_pieces_per_s", "n_prompt_tokens", "pieces"):
        assert key in res, res
    assert res["generation_pieces_per_s"] > 0 and res["n_prompt_tokens"] >= 1


# --------------------------------------------------------------------------------------------------------------
# Struct layout: ctypes vs the C header and vs the library
# --------------------------------------------------------------------------------------------------------------
_C_SIZES = {"size_t": 8, "int32_t": 4, "uint32_t": 4, "uint64_t": 8, "float": 4, "char": 1}


def _parse_header_structs() -> dict[str, tuple[int, list[tuple[str, int, int]]]]:
    """Parse the struct typedefs of the header: {name: (sizeof, [(field, offset, size)])} using the natural
    alignment rules of LP64/LLP64 (every type aligned to its own size), independently of the SDK."""
    text = re.sub(r"/\*.*?\*/", "", HEADER.read_text(encoding="utf-8"), flags=re.S)
    out: dict[str, tuple[int, list[tuple[str, int, int]]]] = {}
    for body, name in re.findall(r"typedef struct \w+ \{(.*?)\} (\w+);", text, flags=re.S):
        off, maxal, fields = 0, 1, []
        for decl in [d.strip() for d in body.split(";") if d.strip()]:
            m = re.fullmatch(r"(const char \*|\w+)\s*(\w+)(?:\[(\d+)\])?", decl)
            assert m, f"unparsed field {decl!r} in {name}"
            ctype, fname, count = m.group(1).strip(), m.group(2), int(m.group(3) or 1)
            size = 8 if "*" in ctype else _C_SIZES[ctype]  # a pointer is 8 bytes on the 64-bit targets
            off = (off + size - 1) // size * size
            fields.append((fname, off, size * count))
            off += size * count
            maxal = max(maxal, size)
        out[name] = ((off + maxal - 1) // maxal * maxal, fields)
    return out


def test_struct_layout_matches_header() -> None:
    parsed = _parse_header_structs()
    classes = {"lm_model_options": _lib.ModelOptions, "lm_session_options": _lib.SessionOptions,
               "lm_sampling": _lib.Sampling, "lm_model_info": _lib.ModelInfo, "lm_message": _lib.Message}
    assert set(classes) <= set(parsed), parsed.keys()
    for cname, cls in classes.items():
        size, fields = parsed[cname]
        assert ctypes.sizeof(cls) == size, f"{cname}: sizeof {ctypes.sizeof(cls)} != header {size}"
        assert [f[0] for f in cls._fields_] == [f[0] for f in fields], f"{cname}: field names/order differ"
        for fname, off, fsize in fields:
            member = getattr(cls, fname)
            assert (member.offset, member.size) == (off, fsize), f"{cname}.{fname}: {(member.offset, member.size)}"
    # The sizes the header's static asserts (tests/c/test_api.c) pin down at compile time.
    assert [ctypes.sizeof(c) for c in (_lib.ModelOptions, _lib.SessionOptions, _lib.Sampling, _lib.ModelInfo,
                                       _lib.Message)] == [40, 24, 32, 336, 16]


def test_struct_size_accepted_by_library_on_field_boundaries(lib_path: Path, tmp_path: Path) -> None:
    """The library validates struct_size against ITS field boundaries (INVALID_ARG when a size cuts a field in
    two). Every field end of the ctypes layout must be accepted and every size inside a field refused: this ties
    the Python offsets to the compiled C layout, not just to the header text."""
    lib = _lib.load()
    cls = _lib.ModelOptions
    ends = [getattr(cls, n).offset + getattr(cls, n).size for n, _ in cls._fields_]
    for size in range(1, ctypes.sizeof(cls) + 1):
        opts = cls()
        lib.lm_model_options_default(ctypes.byref(opts))
        opts.struct_size = size
        out = ctypes.c_void_p()
        status = lib.lm_model_load(str(tmp_path / "none.gguf").encode(), ctypes.byref(opts), ctypes.byref(out))
        if size in ends:
            assert status == 4, f"struct_size {size} (a field end) gave status {status}, expected MODEL_LOAD (4)"
        else:
            assert status == 1, f"struct_size {size} (inside a field) gave status {status}, expected INVALID_ARG (1)"


@pytest.mark.smoke
def test_model_info_prefix_reads_match_full_struct(model) -> None:
    """lm_model_get_info with a truncated struct_size fills exactly the fields before the cut: equal values at
    equal offsets prove the ctypes ModelInfo offsets are the library's."""
    lib, cls = _lib.load(), _lib.ModelInfo
    full = cls()
    full.struct_size = ctypes.sizeof(cls)
    assert lib.lm_model_get_info(model._model, ctypes.byref(full)) == 0
    full_bytes = bytes(full)
    for name, _ in cls._fields_[1:]:
        part = cls()
        part.struct_size = getattr(cls, name).offset + getattr(cls, name).size
        assert lib.lm_model_get_info(model._model, ctypes.byref(part)) == 0
        assert bytes(part)[8:part.struct_size] == full_bytes[8:part.struct_size], name
        assert not any(bytes(part)[part.struct_size:]), f"bytes after the cut at {name} were written"


# --------------------------------------------------------------------------------------------------------------
# Lifetime
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.smoke
def test_close_is_idempotent_and_ordered(tiny_gguf: Path) -> None:
    m = littlemore.Model(tiny_gguf, devices="none")
    assert not m.closed
    m.close()
    m.close()  # second call: no error, no double free
    assert m.closed
    for call in (lambda: m.generate("x"), lambda: m.chat([{"role": "user", "content": "x"}]), m.info):
        with pytest.raises(littlemore.LittleMoreError) as ei:
            call()
        assert ei.value.status_name == "CLOSED"
    # Context manager and __del__ also free cleanly. (lm_model_free refuses with LM_ERR_BUSY while a session is
    # alive and close() raises on any failed free, so a wrong order would fail here.)
    with littlemore.Model(tiny_gguf, devices="none") as m2:
        assert m2.generate("Hello", max_tokens=2) is not None
    assert m2.closed
    m3 = littlemore.Model(tiny_gguf, devices="none")
    del m3
    gc.collect()
    for _ in range(3):  # repeated load/close cycles leave the library usable
        with littlemore.Model(tiny_gguf, devices="none") as m4:
            m4.generate("Hello", max_tokens=1)


@pytest.mark.smoke
def test_close_during_stream_stops_it(tiny_gguf: Path) -> None:
    """Closing the model in the middle of a stream stops the generation: far fewer pieces arrive than an
    uninterrupted run produces (at most the 64 already queued plus the one in hand)."""
    with littlemore.Model(tiny_gguf, devices="none", n_ctx=512) as ref:
        full = len(list(ref.generate("Hello", stream=True, max_tokens=200, temperature=0)))
    assert full >= 150, f"the reference run is too short to show a stop ({full} pieces)"
    m = littlemore.Model(tiny_gguf, devices="none", n_ctx=512)
    gen = m.generate("Hello", stream=True, max_tokens=200, temperature=0)
    assert next(gen)
    m.close()  # must stop the running generation, wait for it, and free everything
    assert m.closed
    rest = list(gen)  # the stream just ends (queued pieces may still be delivered); no exception
    assert all(isinstance(p, str) for p in rest)
    assert 1 + len(rest) <= 66 < full, f"generation was not stopped: {1 + len(rest)} of {full} pieces"


# --------------------------------------------------------------------------------------------------------------
# Streaming rules (the byte-level fake stands in for lm_generate: it feeds scripted raw pieces to the callback)
# --------------------------------------------------------------------------------------------------------------
def _fake_invoke(pieces: list[bytes], delivered: list[bytes]):
    """An `invoke` for the streaming helper that plays the part of lm_generate: it hands scripted raw byte pieces
    to the callback, stops as soon as the callback returns non-zero, and returns LM_OK (0) like the C library."""
    def invoke(cb) -> int:
        """Fake C generation call: deliver the pieces through the callback until it asks to stop."""
        for p in pieces:
            buf = ctypes.create_string_buffer(p)
            delivered.append(p)
            if cb(ctypes.cast(buf, ctypes.c_void_p), len(p), None):
                break
        return 0
    return invoke


def _stream(pieces: list[bytes], delivered: list[bytes] | None = None, lock: threading.Lock | None = None):
    """The SDK's stream generator over a fake C call."""
    return _model_mod._stream_text(_fake_invoke(pieces, [] if delivered is None else delivered),
                                   lock or threading.Lock(), _model_mod._ActiveStreams())


def _stream_threads() -> list[threading.Thread]:
    """Live stream worker threads of the SDK (named `littlemore-stream`); tests use it to prove none is left behind."""
    return [t for t in threading.enumerate() if t.name == "littlemore-stream"]


def test_utf8_split_and_invalid_bytes_never_raise() -> None:
    """A multi-byte character split across pieces is reassembled; an invalid byte becomes U+FFFD; no exception."""
    euro, eacute, smile = "€".encode(), "é".encode(), "😀".encode()
    pieces = [b"a", euro[:1], euro[1:], b"\xff", eacute[:1], eacute[1:], smile[:2], smile[2:], b"z"]
    assert "".join(_stream(pieces)) == "a€\ufffdé😀z"
    # A truncated sequence at the very end is flushed as one replacement character, not dropped or raised.
    assert "".join(_stream([b"ok", euro[:2]])) == "ok\ufffd"


def test_early_abandon_stops_generation_and_frees_lock() -> None:
    delivered: list[bytes] = []
    lock = threading.Lock()
    gen = _stream([b"x"] * 10_000, delivered, lock)
    assert next(gen) == "x"
    gen.close()
    assert not lock.locked(), "the session lock must be released when the consumer abandons the stream"
    assert len(delivered) < 200, f"generation kept going after abandon ({len(delivered)} of 10000 pieces)"
    assert not _stream_threads(), "worker thread leaked"


def test_abandoned_stream_is_cleaned_up_by_garbage_collection() -> None:
    lock = threading.Lock()
    gen = _stream([b"y"] * 10_000, None, lock)
    next(gen)
    del gen
    gc.collect()
    deadline = time.time() + 5
    while (lock.locked() or _stream_threads()) and time.time() < deadline:
        time.sleep(0.01)
    assert not lock.locked() and not _stream_threads()


def test_stream_consumed_from_another_thread() -> None:
    """The C call runs in a worker thread, so the generator may be consumed from any thread."""
    gen = _stream([b"a", b"b", b"c"])
    result: list[str] = []
    t = threading.Thread(target=lambda: result.append("".join(gen)))
    t.start()
    t.join(10)
    assert result == ["abc"]


def test_error_in_the_c_call_reaches_the_consumer() -> None:
    def failing(cb) -> int:
        """Fake C generation call that fails, to test error propagation out of a stream."""
        raise _lib.LittleMoreError("lm_chat: boom", status=10, status_name="LM_ERR_DECODE")
    gen = _model_mod._stream_text(failing, threading.Lock(), _model_mod._ActiveStreams())
    with pytest.raises(littlemore.LittleMoreError, match="boom"):
        list(gen)
    assert not _stream_threads()


@pytest.mark.smoke
def test_real_stream_abandon_releases_session(model) -> None:
    gen = model.generate("Hello", stream=True, max_tokens=200)
    assert next(gen)
    gen.close()
    t0 = time.time()
    again = model.generate("Hello", max_tokens=4)  # would hang forever if the lock leaked
    assert again and time.time() - t0 < 60
    gen2 = model.chat([{"role": "user", "content": "Hi"}], stream=True, max_tokens=200)
    next(gen2)
    del gen2
    gc.collect()
    assert model.generate("Hello", max_tokens=4) == again
    assert not _stream_threads()


@pytest.mark.smoke
def test_concurrent_chat_from_two_threads(model) -> None:
    msgs = [{"role": "user", "content": "Hello"}]
    expected = model.chat(msgs, max_tokens=10)
    results: dict[int, str] = {}
    errors: list[BaseException] = []

    def work(i: int) -> None:
        """Worker thread: chat three times and check each answer equals the single-thread result."""
        try:
            for _ in range(3):
                results[i] = model.chat(msgs, max_tokens=10)
                assert results[i] == expected
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(300)
    assert not errors, errors
    assert results == {0: expected, 1: expected}


# --------------------------------------------------------------------------------------------------------------
# Argument handling
# --------------------------------------------------------------------------------------------------------------
def test_n_gpu_layers_and_sampling_arguments_are_validated(lib_path: Path) -> None:
    assert _model_mod._parse_n_gpu_layers("auto") == -1
    assert _model_mod._parse_n_gpu_layers(0) == 0 and _model_mod._parse_n_gpu_layers(12) == 12
    for bad in ("many", -2, 1.5, True, None):
        with pytest.raises((ValueError, TypeError)):
            _model_mod._parse_n_gpu_layers(bad)
    s = _model_mod._make_sampling(temperature=0, seed=littlemore.SEED_RANDOM, max_tokens=5)
    assert s.temperature == 0 and s.seed == 0xFFFFFFFF and s.max_tokens == 5 and s.top_k == 40  # others: C defaults
    assert _model_mod._make_sampling(seed="random").seed == littlemore.SEED_RANDOM
    for kw in ({"seed": -1}, {"seed": 2**32}, {"max_tokens": 0}, {"top_p": 0.0}, {"top_k": -1}):
        with pytest.raises(ValueError):
            _model_mod._make_sampling(**kw)
    with pytest.raises(TypeError):
        _model_mod._make_sampling(temprature=1.0)  # a typo must not be silently ignored


def test_chat_message_validation(lib_path: Path) -> None:
    for bad in ([], [{"role": "user"}], [{"role": "user", "content": 3}], ["hi"], [{"role": "u\0", "content": "x"}]):
        with pytest.raises((ValueError, TypeError)):
            _model_mod._encode_messages(bad)


# --------------------------------------------------------------------------------------------------------------
# Windows DLL directories and CUDA visibility
# --------------------------------------------------------------------------------------------------------------
class _Handle:
    """Stand-in for the object os.add_dll_directory returns."""

    def close(self) -> None:
        """Stub with the one method the code under test calls."""


@pytest.mark.skipif(os.name != "nt", reason="os.add_dll_directory exists only on Windows")
def test_dll_directories_are_registered_before_the_library_loads(monkeypatch, tmp_path: Path, lib_path: Path) -> None:
    """Python 3.8+ ignores PATH for DLL dependencies, so the SDK must call os.add_dll_directory() for the CUDA
    runtime folder BEFORE it loads littlemore.dll (whose plugins load the CUDA runtime)."""
    cuda = tmp_path / "cudabin"
    cuda.mkdir()
    (cuda / "cudart64_12.dll").write_bytes(b"")
    events: list[tuple[str, str]] = []
    monkeypatch.setenv("LITTLEMORE_DLL_DIR", str(cuda))
    monkeypatch.setattr(os, "add_dll_directory", lambda p: events.append(("add", str(p))) or _Handle())
    monkeypatch.setattr(_lib.ctypes, "CDLL", lambda *a, **k: events.append(("load", str(a[0]))) or object())
    monkeypatch.setattr(_lib, "_LIB", None)
    monkeypatch.setattr(_lib, "_DLL_HANDLES", [])
    monkeypatch.setattr(_lib, "_declare_prototypes", lambda dll: None)
    monkeypatch.setattr(_lib, "_check_api_version", lambda dll: None)
    _lib.load()
    kinds = [k for k, _ in events]
    assert "add" in kinds and "load" in kinds
    assert max(i for i, k in enumerate(kinds) if k == "add") < kinds.index("load"), events
    assert ("add", str(cuda)) in events


@pytest.mark.skipif(os.name != "nt", reason="DLL directories are a Windows concept")
def test_dll_search_dirs_order_and_content(monkeypatch, tmp_path: Path) -> None:
    a, b, pathdir = tmp_path / "a", tmp_path / "b", tmp_path / "onpath"
    for d in (a, b, pathdir):
        d.mkdir()
    (pathdir / "cudart64_12.dll").write_bytes(b"")
    cuda_root = tmp_path / "cuda"
    (cuda_root / "bin").mkdir(parents=True)
    monkeypatch.setenv("LITTLEMORE_DLL_DIR", os.pathsep.join([str(a), str(b)]))
    monkeypatch.setenv("CUDA_PATH", str(cuda_root))
    monkeypatch.setenv("PATH", str(pathdir) + os.pathsep + os.environ.get("PATH", ""))
    dirs = [Path(d) for d in _lib.dll_search_dirs()]
    assert dirs[:2] == [a, b], "LITTLEMORE_DLL_DIR comes first, in the order given"
    assert cuda_root / "bin" in dirs, "CUDA_PATH/bin is searched"
    assert pathdir in dirs, "a PATH folder holding a CUDA runtime DLL is searched (Python ignores PATH itself)"
    assert len(dirs) == len(set(dirs)), "no duplicates"


@pytest.mark.smoke
def test_cuda_backend_visible_from_plain_python_in_another_directory(lib_path: Path, tmp_path: Path) -> None:
    """The user-visible rule: python.exe started in an unrelated directory, with no CUDA folder on PATH and no
    helper variable, still sees CUDA0 in info() (when the machine has a CUDA device and runtime)."""
    if not (lib_path.parent / ("ggml-cuda.dll" if os.name == "nt" else "libggml-cuda.so")).exists():
        pytest.skip("the build has no CUDA plugin")
    if os.name == "nt" and not [d for d in _lib.dll_search_dirs() if list(Path(d).glob("cudart64_*.dll"))]:
        pytest.skip("no CUDA runtime DLL folder found (CUDA toolkit not installed)")
    smi = shutil.which("nvidia-smi")
    probe = subprocess.run([smi, "-L"], capture_output=True, text=True) if smi else None
    if probe is None or probe.returncode != 0 or "GPU" not in probe.stdout:
        pytest.skip("no NVIDIA GPU / driver on this machine")
    # PATH reduced to the Windows system folders and Python itself: nothing CUDA related on it.
    sysroot = os.environ.get("SystemRoot", r"C:\Windows")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("CUDA", "LITTLEMORE"))}
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), sysroot, rf"{sysroot}\System32"])
    env["PYTHONPATH"] = str(PYDIR)
    code = "import littlemore; print([b['name'] for b in littlemore.backends_info()])"
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "CUDA0" in r.stdout, f"CUDA0 missing from the backends: {r.stdout.strip()} / {r.stderr[-500:]}"


# --------------------------------------------------------------------------------------------------------------
# Open streams, locks and deadlocks (a regression must FAIL, never hang the suite: see _finishes_within)
# --------------------------------------------------------------------------------------------------------------
def _finishes_within(fn, seconds: float = 40):
    """Run `fn` in a daemon thread; fail (instead of hanging the test run) when it has not finished in `seconds`.
    Returns fn's result and re-raises its exception."""
    box: list = []
    err: list[BaseException] = []

    def run() -> None:
        """Run `fn` and keep its result or exception for the main thread."""
        try:
            box.append(fn())
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller
            err.append(e)

    t = threading.Thread(target=run, daemon=True, name="deadlock-guard")
    t.start()
    t.join(seconds)
    assert not t.is_alive(), f"deadlock: the call was still running after {seconds} s"
    if err:
        raise err[0]
    return box[0] if box else None


_HELLO = [{"role": "user", "content": "Hello"}]


@pytest.mark.smoke
def test_second_generation_while_a_stream_is_open_raises_instead_of_deadlocking(model) -> None:
    """Case A: a stream that was abandoned but is still referenced keeps the session. A new generation call on the
    same thread can never succeed (nobody would consume the stream), so it raises a clear error."""
    def body() -> None:
        """Abandon a stream early (still alive) and check that every new generation from this thread is refused."""
        s = model.chat(_HELLO, stream=True, max_tokens=200)
        for _piece in s:
            break  # abandoned early, but `s` is still alive
        for call in (lambda: model.chat(_HELLO, max_tokens=2), lambda: model.generate("x", max_tokens=2),
                     lambda: model.chat(_HELLO, stream=True), lambda: model.generate("x", stream=True)):
            with pytest.raises(littlemore.LittleMoreError) as ei:
                call()
            assert ei.value.status_name == "STREAM_OPEN" and "stream" in str(ei.value)
        s.close()  # closing the stream frees the session
        assert model.chat(_HELLO, max_tokens=2) is not None
        # A stream that was consumed to its end never blocks later calls either.
        assert list(model.generate("Hello", stream=True, max_tokens=3))
        assert model.generate("Hello", max_tokens=3)

    _finishes_within(body)


@pytest.mark.smoke
def test_tokenizer_and_info_work_inside_a_stream_loop(model) -> None:
    """Case B: tokenize, detokenize and info need no session, so they run while a stream is open (the C API allows
    it during generation)."""
    def body() -> None:
        """While a stream is open, tokenize/detokenize must keep working and give the same answers."""
        ids = model.tokenize("Hello world")
        seen = 0
        for _piece in model.generate("Hello", stream=True, max_tokens=100):
            assert model.tokenize("Hello world") == ids
            assert model.detokenize(ids) == "Hello world"
            assert model.info()["model"]["n_layer"] == 2
            seen += 1
            if seen >= 3:
                break
        assert seen == 3

    _finishes_within(body)


@pytest.mark.smoke
def test_other_thread_waits_for_an_open_stream_then_runs(model) -> None:
    """Another thread's call queues behind an open stream (serial use) and runs once the stream is closed."""
    s = model.generate("Hello", stream=True, max_tokens=200)
    assert next(s)
    result: list[str] = []
    t = threading.Thread(target=lambda: result.append(model.chat(_HELLO, max_tokens=3)), daemon=True)
    t.start()
    t.join(1.0)
    assert t.is_alive() and not result, "the second call must wait while the stream holds the session"
    s.close()
    t.join(40)
    assert not t.is_alive() and result, "the waiting call must run after the stream is closed"


@pytest.mark.smoke
def test_half_consumed_stream_at_interpreter_exit_does_not_hang(tiny_gguf: Path) -> None:
    """A stream left open in a module global when the interpreter exits must not hang the process."""
    code = ("import sys, littlemore\n"
            "m = littlemore.Model(sys.argv[1], devices='none')\n"
            "s = m.chat([{'role': 'user', 'content': 'Hello'}], stream=True, max_tokens=500)\n"
            "next(s)\n"
            "print('streaming', flush=True)\n")  # m and s stay alive in module globals until the interpreter exits
    env = dict(os.environ, PYTHONPATH=str(PYDIR))
    r = subprocess.run([sys.executable, "-c", code, str(tiny_gguf)], capture_output=True, text=True, timeout=30,
                       env=env)
    assert r.returncode == 0 and "streaming" in r.stdout, r.stderr[-1500:]


# --------------------------------------------------------------------------------------------------------------
# Callback failures, validation, constants, pickling
# --------------------------------------------------------------------------------------------------------------
def test_failing_callback_surfaces_the_error_in_stream_and_collect(monkeypatch) -> None:
    def boom(ptr, length):
        """Fake `ctypes.string_at` that fails, to test that errors in the callback surface."""
        raise RuntimeError("cannot copy the piece")

    monkeypatch.setattr(_model_mod.ctypes, "string_at", boom)
    with pytest.raises(RuntimeError, match="cannot copy"):
        list(_stream([b"abc"] * 5))
    delivered: list[bytes] = []
    with pytest.raises(RuntimeError, match="cannot copy"):
        _model_mod._collect_text(_fake_invoke([b"abc"] * 5, delivered), threading.Lock(), _model_mod._ActiveStreams(),
                                 lambda: None)
    assert len(delivered) == 1, "generation must stop at the first failing callback"


def test_keyboard_interrupt_in_the_callback_is_reraised_after_the_call(monkeypatch) -> None:
    def interrupt(ptr, length):
        """Fake `ctypes.string_at` that raises KeyboardInterrupt, to test Ctrl+C handling."""
        raise KeyboardInterrupt

    monkeypatch.setattr(_model_mod.ctypes, "string_at", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _model_mod._collect_text(_fake_invoke([b"abc"] * 5, []), threading.Lock(), _model_mod._ActiveStreams(),
                                 lambda: None)


def test_numeric_options_are_validated_not_wrapped(lib_path: Path) -> None:
    for kw in ({"n_ctx": -1}, {"n_ctx": 2**32}, {"n_batch": -5}, {"n_threads": 2**31}, {"vram_budget": -1}):
        with pytest.raises(ValueError):
            littlemore.Model("does-not-matter.gguf", **kw)  # refused before any file is touched


@pytest.mark.smoke
def test_sdk_normalizes_text_to_nfc(model) -> None:
    """The SDK normalizes text to NFC before tokenize / generate / chat (the Hugging Face tokenizers of Qwen and others
    do; llama.cpp does not, and the C API passes bytes unchanged). Negative control: the raw C call gives different
    tokens for the decomposed (NFD) and the composed (NFC) form of the same text, so the check can fail."""
    import unicodedata
    nfc = "café naïve Ångström 한국어"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfd != nfc
    lib = _lib.load()

    def raw(text: str) -> list[int]:
        b = text.encode("utf-8")
        buf = (ctypes.c_int32 * 256)()
        n = ctypes.c_size_t(0)
        _lib.check(lib.lm_tokenize(model._model, b, len(b), buf, 256, 0, ctypes.byref(n)), lib)
        return list(buf[: n.value])

    assert raw(nfd) != raw(nfc)                       # the C API does not normalize
    assert model.tokenize(nfd) == model.tokenize(nfc) == raw(nfc)
    g = dict(temperature=0, max_tokens=8)
    assert model.generate(nfd, **g) == model.generate(nfc, **g)
    assert model.chat([{"role": "user", "content": nfd}], **g) == model.chat([{"role": "user", "content": nfc}], **g)


@pytest.mark.smoke
def test_token_ids_are_range_checked(model) -> None:
    for bad in ([2**31], [-2**31 - 1], [1.5], ["a"]):
        with pytest.raises((ValueError, TypeError)):
            model.detokenize(bad)
    assert model.detokenize([]) == ""


def test_status_constants_match_the_header(lib_path: Path) -> None:
    text = HEADER.read_text(encoding="utf-8")
    enum = dict(re.findall(r"(LM_ERR_\w+)\s*=\s*(\d+)", text))
    assert int(enum["LM_ERR_BUFFER_TOO_SMALL"]) == _lib.LM_ERR_BUFFER_TOO_SMALL
    assert int(enum["LM_ERR_REENTRANT"]) == 13 and int(enum["LM_ERR_BUSY"]) == 14


def test_errors_are_picklable() -> None:
    for err in (littlemore.LittleMoreError("lm_chat: boom", 10, "LM_ERR_DECODE"),
                littlemore.LibraryNotFoundError("no library here")):
        back = pickle.loads(pickle.dumps(err))
        assert type(back) is type(err) and str(back) == str(err)
        assert (back.status, back.status_name, back.message) == (err.status, err.status_name, err.message)


# --------------------------------------------------------------------------------------------------------------
# DLL directory hygiene and library precedence
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(os.name != "nt", reason="DLL directories are a Windows concept")
def test_dll_dirs_are_absolute_and_never_the_working_directory(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "cudart64_12.dll").write_bytes(b"")  # a runtime DLL in the working directory must NOT be trusted
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LITTLEMORE_DLL_DIR", os.pathsep.join([".", "sub"]))
    monkeypatch.setenv("CUDA_PATH", "relative-cuda")
    monkeypatch.setenv("PATH", os.pathsep.join([".", os.environ.get("PATH", "")]))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        dirs = _lib.dll_search_dirs()
    assert all(os.path.isabs(d) for d in dirs), dirs
    assert str(tmp_path) not in [os.path.normpath(d) for d in dirs], "the working directory must never be registered"


@pytest.mark.skipif(os.name != "nt", reason="DLL directories are a Windows concept")
def test_nvidia_wheel_dirs_come_from_site_packages_not_the_working_directory(monkeypatch, tmp_path: Path) -> None:
    import sysconfig
    fake = tmp_path / "nvidia" / "cublas" / "bin"
    fake.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))  # an `nvidia` folder first on sys.path (like python -m from this cwd)
    assert str(fake) not in _lib._nvidia_wheel_bin_dirs()
    site = tmp_path / "site" / "nvidia" / "cuda_runtime" / "bin"
    site.mkdir(parents=True)
    paths = {"purelib": str(tmp_path / "site"), "platlib": str(tmp_path / "site")}
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: paths)
    assert str(site) in _lib._nvidia_wheel_bin_dirs()


@pytest.mark.skipif(os.name != "nt", reason="os.add_dll_directory exists only on Windows")
def test_add_dll_directory_failure_is_tolerated(monkeypatch, lib_path: Path) -> None:
    def refuse(path):
        """Fake `os.add_dll_directory` that refuses, to test that the SDK carries on."""
        raise OSError("not allowed")

    monkeypatch.setattr(os, "add_dll_directory", refuse)
    monkeypatch.setattr(_lib.ctypes, "CDLL", lambda *a, **k: object())
    monkeypatch.setattr(_lib, "_LIB", None)
    monkeypatch.setattr(_lib, "_declare_prototypes", lambda dll: None)
    monkeypatch.setattr(_lib, "_check_api_version", lambda dll: None)
    assert _lib.load() is not None  # the library still loads; the folder is simply not registered


def test_newer_library_wins_over_a_stale_one_and_a_warning_is_given_once(monkeypatch, tmp_path: Path) -> None:
    name = _lib._library_file_names()[0]
    stale, fresh = tmp_path / "stale", tmp_path / "fresh"
    for d, size, age in ((stale, 10, 1_000_000_000), (fresh, 20, 1_900_000_000)):
        d.mkdir()
        (d / name).write_bytes(b"x" * size)
        os.utime(d / name, (age, age))
    monkeypatch.delenv("LITTLEMORE_LIBRARY", raising=False)
    monkeypatch.setattr(_lib, "_candidate_library_dirs", lambda: [stale, fresh])  # the stale one has priority by order
    monkeypatch.setattr(_lib, "_WARNED_STALE", False)
    with pytest.warns(UserWarning, match="stale") as rec:
        assert _lib.find_library() == (fresh / name).resolve()
    assert str(stale) in str(rec[0].message) and str(fresh) in str(rec[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _lib.find_library() == (fresh / name).resolve()  # second call: no second warning
    # Identical files (a copy of the same build) are not worth a warning.
    (stale / name).write_bytes(b"x" * 20)
    os.utime(stale / name, (1_900_000_000, 1_900_000_000))
    monkeypatch.setattr(_lib, "_WARNED_STALE", False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _lib.find_library()


# --------------------------------------------------------------------------------------------------------------
# Command line: usage errors, convert lookup, licence files
# --------------------------------------------------------------------------------------------------------------
def test_bench_zero_reps_is_a_usage_error(lib_path: Path, tmp_path: Path) -> None:
    for bad in (["--reps", "0"], ["--n-prompt", "0"], ["--n-gen", "-1"]):
        r = _cli("bench", str(tmp_path / "m.gguf"), *bad)
        assert r.returncode == 2 and "Traceback" not in r.stderr and "positive" in r.stderr, r.stderr[-500:]


@pytest.mark.smoke
def test_run_with_invalid_sampling_value_is_a_clean_usage_error(tiny_gguf: Path) -> None:
    r = _cli("run", str(tiny_gguf), "--prompt", "hi", "--cpu-only", "--max-tokens", "0")
    assert r.returncode == 2 and "Traceback" not in r.stderr and "max_tokens" in r.stderr, r.stderr[-500:]
    r = _cli("run", str(tiny_gguf), "--prompt", "hi", "--cpu-only", "--seed", "-3")
    assert r.returncode == 2 and "Traceback" not in r.stderr and "seed" in r.stderr, r.stderr[-500:]


@pytest.mark.smoke
def test_convert_finds_the_converter_despite_an_unrelated_tools_package(tmp_path: Path, lib_path: Path) -> None:
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "__init__.py").write_text("raise SystemExit('the wrong tools package was imported')\n",
                                                    encoding="utf-8")
    r = _cli("convert", "--repo", str(ROOT), "--help", cwd=tmp_path)  # `python -m` puts the working directory first on sys.path
    assert r.returncode == 0 and "checkpoint" in r.stdout.lower(), (r.stdout + r.stderr)[-800:]


def test_licence_files_are_in_place() -> None:
    top = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert top.startswith("MIT License") and "Copyright (c) 2026 LittleMore contributors" in top
    assert (PYDIR / "LICENSE").read_text(encoding="utf-8") == top, "python/LICENSE must be a copy of the top-level LICENSE"
    notices = (PYDIR / "THIRD_PARTY_NOTICES").read_text(encoding="utf-8")
    engine_licence = ROOT / "engine" / "LICENSE"
    if engine_licence.exists():  # the engine checkout is not part of this repository
        assert engine_licence.read_text(encoding="utf-8").strip() in notices
    assert "Copyright (c) 2023-2026 The ggml authors" in notices
    meta = (PYDIR / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "MIT"' in meta and "license-files" in meta and "THIRD_PARTY_NOTICES" in meta


# --------------------------------------------------------------------------------------------------------------
# `littlemore convert` never runs code from the current directory
# --------------------------------------------------------------------------------------------------------------
def _fake_repo(root: Path, marker: Path) -> Path:
    """A folder that looks like the LittleMore repository whose converter only writes `marker` (and returns 0)."""
    (root / "tools" / "convert").mkdir(parents=True)
    (root / "include").mkdir()
    (root / "include" / "littlemore.h").write_text("/* fake */\n")
    (root / "CMakeLists.txt").write_text("project(littlemore)\n")
    (root / "tools" / "convert" / "__init__.py").write_text("")
    (root / "tools" / "convert" / "lb_to_gguf.py").write_text(
        f"from pathlib import Path\ndef main(argv):\n    Path({str(marker)!r}).write_text(' '.join(argv))\n    return 0\n")
    return root


def test_convert_does_not_search_the_current_directory(tmp_path: Path, monkeypatch, capsys) -> None:
    """A converter in the working directory is NOT executed; without --repo / LITTLEMORE_REPO the command refuses."""
    from littlemore import cli
    marker = tmp_path / "ran.txt"
    _fake_repo(tmp_path / "cwd_repo", marker)
    monkeypatch.chdir(tmp_path / "cwd_repo")
    monkeypatch.delenv("LITTLEMORE_REPO", raising=False)
    assert cli.main(["convert", "ckpt", "-o", "out.gguf"]) == 1
    assert not marker.exists(), "code from the current directory was executed"
    assert "--repo" in capsys.readouterr().err


def test_convert_with_explicit_repo_runs_its_converter_and_names_it(tmp_path: Path, monkeypatch, capsys) -> None:
    """`--repo PATH` (or LITTLEMORE_REPO) is used, the path is printed and the other arguments are forwarded."""
    from littlemore import cli
    marker = tmp_path / "ran.txt"
    repo = _fake_repo(tmp_path / "repo", marker)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["convert", "--repo", str(repo), "ckpt", "-o", "out.gguf"]) == 0
    assert marker.read_text() == "ckpt -o out.gguf"
    assert str(repo.resolve()) in capsys.readouterr().err
    marker.unlink()
    monkeypatch.setenv("LITTLEMORE_REPO", str(repo))
    assert cli.main(["convert", "ckpt"]) == 0 and marker.read_text() == "ckpt"


def test_convert_refuses_a_folder_that_is_not_the_repository(tmp_path: Path, capsys) -> None:
    """A folder without the repository's marker files is refused, and nothing in it is executed."""
    from littlemore import cli
    marker = tmp_path / "ran.txt"
    (tmp_path / "fake" / "tools" / "convert").mkdir(parents=True)
    (tmp_path / "fake" / "tools" / "convert" / "lb_to_gguf.py").write_text(f"open({str(marker)!r}, 'w')\n")
    assert cli.main(["convert", "--repo", str(tmp_path / "fake"), "ckpt"]) == 1
    assert not marker.exists()
    assert "does not look like" in capsys.readouterr().err
