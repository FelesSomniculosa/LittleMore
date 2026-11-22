"""The `Model` class: load a GGUF model, chat with it, generate text, ask what it runs on.

Design in a nutshell
    * One `Model` owns one library model plus one session (the conversation state and KV cache). The C API runs
      one call per session at a time, and it forbids freeing a session while a call is running, so the SDK keeps
      a `threading.Lock` per model: calls from several threads queue up and never overlap, and `close()` waits for
      the running call.
    * Every `chat` / `generate` call is complete in itself (the library discards the session's previous state):
      for a multi-turn conversation pass the whole message history, including earlier assistant answers, on each
      call. `littlemore run` does exactly that.
    * Streaming runs the C call in a small worker thread and hands the pieces to the caller through a queue. The
      caller can therefore consume the generator from any thread, can stop whenever it likes (closing or just
      dropping the generator makes the next callback return non-zero, which ends the C call cleanly and releases
      the lock), and the C callback never runs user code or frees anything: it only enqueues bytes.
    * The library delivers raw bytes. Pieces are valid UTF-8 whenever the model produced valid UTF-8, but a model
      can produce invalid sequences, which the library passes through unchanged, so the text is decoded
      incrementally with errors="replace": streaming never raises a decoding error.
    * Locks: generation (`chat` / `generate`) needs the session, so it queues behind a running or still-open
      stream. `tokenize`, `detokenize` and `info` need only the model, which the C API allows during generation, so
      they work while a stream is open. A stream that has been started and not finished or closed owns the session:
      starting another generation on the SAME thread raises `LittleMoreError` ("STREAM_OPEN") instead of waiting for
      something that can never happen; other threads simply wait. Close a stream you abandon early
      (`stream.close()` or `contextlib.closing`), or let it be garbage collected.
    * Freeing: the session is always freed before the model (the model refuses while a session is alive), `close()`
      is idempotent and safe from any thread, and `__del__` / the context manager call it.
"""
from __future__ import annotations

import atexit
import ctypes
import queue
import re
import sys
import threading
import weakref
from codecs import getincrementaldecoder
from os import PathLike, fspath
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence, overload

from . import _lib
from ._lib import LittleMoreError

__all__ = ["Model", "backends_info", "version", "SEED_RANDOM"]

#: `seed` value that asks for a different random seed on every call (LM_SEED_RANDOM).
SEED_RANDOM: int = _lib.LM_SEED_RANDOM

_SAMPLING_KEYS = ("temperature", "top_k", "top_p", "seed", "max_tokens")


def version() -> str:
    """The version of the loaded C library (e.g. "0.1.0")."""
    return _lib.load().lm_version().decode("ascii")


_BACKEND_LINE = re.compile(r"^(?P<name>[^:\s]+):\s*(?P<description>.*?)\s*(?:\[(?P<details>[^\]]*)\])?\s*$")


def backends_info() -> list[dict[str, str]]:
    """The compute devices the library found, one dict per device.

    Each dict has `name` (what `devices=` accepts, e.g. "CPU", "CUDA0", "Vulkan0"), `description` (the device
    name, e.g. the GPU model) and `details` (the library's bracket text such as "GPU, free 5.1 GiB of 6.0 GiB").
    Loads the backend plugins if that has not happened yet.
    """
    out: list[dict[str, str]] = []
    for line in _lib.iter_lines(_lib.backends_text()):
        m = _BACKEND_LINE.match(line.strip())
        if m:
            out.append({"name": m["name"], "description": m["description"], "details": m["details"] or ""})
    return out


# --------------------------------------------------------------------------------------------------------------
# Argument handling
# --------------------------------------------------------------------------------------------------------------
def _parse_n_gpu_layers(value: int | str) -> int:
    """"auto" -> -1 (the library fits as many layers as the free GPU memory allows); an int is used as is.

    Raises:
        ValueError: a string other than "auto", or an int below -1.
        TypeError: any other type (bool is refused: `True` is almost certainly a mistake).
    """
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            return _lib.LM_GPU_LAYERS_AUTO
        raise ValueError(f'n_gpu_layers must be "auto" or an integer, not {value!r}')
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'n_gpu_layers must be "auto" or an int, not {type(value).__name__}')
    if value < _lib.LM_GPU_LAYERS_AUTO:
        raise ValueError("n_gpu_layers must be -1 (auto), 0 or positive")
    return value


def _int_in_range(name: str, value: Any, low: int, high: int) -> int:
    """`value` as an int in [low, high], else ValueError / TypeError (ctypes would silently wrap it around)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, not {type(value).__name__}")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, not {value}")
    return value


def _make_sampling(**kw: Any) -> _lib.Sampling:
    """Build an `lm_sampling` from keyword arguments, starting from the library's defaults.

    Accepted: temperature (<= 0 is greedy), top_k (0 = off), top_p (0 < p <= 1), seed (int 0 .. 2**32-1, or
    "random" / `SEED_RANDOM` for a new seed per call) and max_tokens (> 0). Anything else is a TypeError, so a
    misspelt option is never silently ignored.
    """
    unknown = set(kw) - set(_SAMPLING_KEYS)
    if unknown:
        raise TypeError(f"unknown sampling option(s) {sorted(unknown)}; expected {list(_SAMPLING_KEYS)}")
    lib = _lib.load()
    s = _lib.Sampling()
    _lib.check(lib.lm_sampling_default(ctypes.byref(s)), lib)
    if kw.get("temperature") is not None:
        s.temperature = float(kw["temperature"])
    if kw.get("top_k") is not None:
        if int(kw["top_k"]) < 0:
            raise ValueError("top_k must be >= 0 (0 turns it off)")
        s.top_k = int(kw["top_k"])
    if kw.get("top_p") is not None:
        if not 0.0 < float(kw["top_p"]) <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        s.top_p = float(kw["top_p"])
    if kw.get("seed") is not None:
        seed = SEED_RANDOM if kw["seed"] == "random" else kw["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError('seed must be an integer in 0 .. 2**32-1, or "random"')
        s.seed = seed
    if kw.get("max_tokens") is not None:
        if int(kw["max_tokens"]) <= 0:
            raise ValueError("max_tokens must be > 0")
        s.max_tokens = int(kw["max_tokens"])
    return s


def _encode_messages(messages: Sequence[Mapping[str, str]]) -> tuple[Any, list[bytes]]:
    """Convert chat messages to a C array of `lm_message`.

    Returns the array and the list of encoded byte strings; the caller must keep that list alive while the C
    call runs (the C structs only point into those bytes).

    Raises:
        ValueError / TypeError: empty list, an item that is not a mapping with str "role" and "content", or text
            containing a NUL character.
    """
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError("messages must be a list of {'role': ..., 'content': ...} dicts")
    if not messages:
        raise ValueError("messages must not be empty")
    keep: list[bytes] = []
    arr = (_lib.Message * len(messages))()
    for i, m in enumerate(messages):
        if not isinstance(m, Mapping) or "role" not in m or "content" not in m:
            raise TypeError(f"message {i} must be a dict with 'role' and 'content'")
        role, content = _lib.utf8(m["role"], f"message {i} role"), _lib.utf8(m["content"], f"message {i} content")
        keep += [role, content]
        arr[i].role, arr[i].content = role, content
    return arr, keep


# --------------------------------------------------------------------------------------------------------------
# Running one generation call: streaming (worker thread + queue) and collecting (calling thread)
# --------------------------------------------------------------------------------------------------------------
class _ActiveStreams:
    """The stop flags of the generations that are running or waiting on one model, so `Model.close()` can ask all
    of them to stop before it waits for the session lock. A streaming generation also records the thread that
    consumes it (its `owner`), which lets the model refuse a second generation call from that same thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[threading.Event, int | None] = {}

    def add(self, ev: threading.Event, owner: int | None = None) -> None:
        with self._lock:
            self._events[ev] = owner

    def discard(self, ev: threading.Event) -> None:
        with self._lock:
            self._events.pop(ev, None)

    def stop_all(self) -> None:
        with self._lock:
            for ev in self._events:
                ev.set()

    def owned_by(self, ident: int) -> bool:
        """True when a stream consumed by thread `ident` is open (started, not finished or closed)."""
        with self._lock:
            return any(o == ident for o in self._events.values())


_DONE = object()  # queue sentinel: the C call has returned


def _stream_text(invoke: Callable[[Any], int], lock: threading.Lock, active: _ActiveStreams,
                 check_open: Callable[[], None] | None = None,
                 before_start: Callable[[], None] | None = None) -> Iterator[str]:
    """Generator over the text pieces of one generation call.

    `invoke(cb)` makes the C call (`lm_chat` / `lm_generate`) with `cb` as the token callback and returns its
    status. It runs in a worker thread, with `lock` held, started when the first piece is requested.

    Stopping: when the consumer closes or drops the generator, the stop flag is set, the callback returns 1 at its
    next call (the C library then ends the call with LM_OK), the worker leaves the lock and is joined. The callback
    does nothing but enqueue bytes and test the flag; it never frees or calls back into the library.
    Errors: a failing C call raises `LittleMoreError` in the consumer, with the message fetched on the worker thread
    (`lm_last_error` is thread-local, so it must be read where the call failed). So does an exception raised by the
    callback itself (e.g. out of memory while copying a piece): it stops the generation and is re-raised to the
    consumer, never swallowed.
    `before_start` runs in the consumer thread when the first piece is requested, before anything is registered
    (the model uses it to refuse a second stream on a thread that still has one open).
    """
    if check_open:
        check_open()
    if before_start:
        before_start()
    pieces: queue.Queue[Any] = queue.Queue(maxsize=64)  # bounded: a slow consumer slows generation down
    stop = threading.Event()       # set by Model.close(): end the generation, but still tell the consumer it is over
    abandoned = threading.Event()  # set when the consumer closed/dropped the generator: nobody reads any more
    active.add(stop, threading.get_ident())
    failure: list[BaseException] = []  # an exception raised inside the callback (it cannot cross the C boundary)

    def put(item: Any, data: bool) -> bool:
        """Enqueue `item`. Data pieces are refused once the generation is being stopped; the end marker and errors
        are always delivered unless the consumer is gone (otherwise a consumer blocked in get() would wait forever).
        Returns False when the item was not enqueued."""
        while not abandoned.is_set() and not (data and stop.is_set()):
            try:
                pieces.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def on_piece(ptr: int, length: int, _user: int) -> int:
        # Runs on the worker thread inside the C call. Copy the bytes (valid only during the call) and return.
        try:
            return 0 if put(ctypes.string_at(ptr, length), True) else 1
        except BaseException as e:  # noqa: BLE001 - an exception must not cross the C boundary; stop generating
            failure.append(e)
            stop.set()
            return 1

    def work() -> None:
        cb = _lib.TokenCallback(on_piece)  # must stay referenced while the C call runs
        try:
            while not lock.acquire(timeout=0.05):  # wait for the session, but give up when abandoned or closed
                if stop.is_set() or abandoned.is_set():
                    return
            try:
                if stop.is_set() or abandoned.is_set():
                    return
                if check_open:
                    check_open()
                _lib.check(invoke(cb))
                if failure:
                    raise failure[0]
            finally:
                lock.release()
        except BaseException as e:  # noqa: BLE001 - handed to the consumer
            put(e, False)
        finally:
            put(_DONE, False)

    worker = threading.Thread(target=work, name="littlemore-stream", daemon=True)
    decoder = getincrementaldecoder("utf-8")(errors="replace")
    try:
        worker.start()
        while True:
            item = pieces.get()
            if item is _DONE:
                break
            if isinstance(item, BaseException):
                raise item
            text = decoder.decode(item)
            if text:
                yield text
        tail = decoder.decode(b"", final=True)  # the library flushes held-back bytes itself; this is a safety net
        if tail:
            yield tail
    finally:
        abandoned.set()  # stops the callback at its next call; the worker then leaves the lock
        stop.set()
        # Wait for the worker so the session lock is free when we return. Not while the interpreter is shutting
        # down: daemon threads are frozen then, so joining one would block forever (the exit handler has already
        # stopped and released every stream by that point).
        if not sys.is_finalizing():
            worker.join()
        active.discard(stop)


def _collect_text(invoke: Callable[[Any], int], lock: threading.Lock, active: _ActiveStreams,
                  check_open: Callable[[], None]) -> str:
    """Run one generation call in the calling thread and return the whole text (no worker thread needed)."""
    chunks: list[bytes] = []
    stop = threading.Event()
    failure: list[BaseException] = []

    def on_piece(ptr: int, length: int, _user: int) -> int:
        try:
            chunks.append(ctypes.string_at(ptr, length))
            return 1 if stop.is_set() else 0
        except BaseException as e:  # noqa: BLE001 - cannot cross the C boundary: stop, re-raise after the call
            failure.append(e)
            return 1

    cb = _lib.TokenCallback(on_piece)
    active.add(stop)
    try:
        with lock:
            check_open()
            _lib.check(invoke(cb))
    finally:
        active.discard(stop)
    if failure:
        raise failure[0]  # also KeyboardInterrupt: it was caught in the callback and is re-raised here
    if stop.is_set():
        raise LittleMoreError("the model was closed while a generation was running", None, "CLOSED")
    return b"".join(chunks).decode("utf-8", "replace")


# --------------------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------------------
# Every open model, so that interpreter exit can close them first. Why: a stream that is still open at exit (for
# example held in a module global) has a daemon worker thread inside the C call, holding the session lock. When
# the interpreter shuts down such daemon threads are frozen where they stand, so a later `Model.__del__` would wait
# for that lock forever. atexit handlers run BEFORE that freeze, while the worker can still react: closing the
# model stops the generation, releases the lock and frees everything in the right order.
_LIVE_MODELS: "weakref.WeakSet[Model]" = weakref.WeakSet()


def _close_all_models_at_exit() -> None:
    for m in list(_LIVE_MODELS):
        try:
            m.close()
        except Exception:  # noqa: BLE001 - exit must go on; the OS reclaims what could not be freed
            pass


atexit.register(_close_all_models_at_exit)


class Model:
    """A loaded GGUF model with one conversation session.

    Args:
        path: the `.gguf` file.
        n_gpu_layers: `"auto"` (default) fits as many layers on the GPU as the free GPU memory allows; an int
            offloads that many layers (0 = none). **Note:** with `n_gpu_layers=0` (or when auto-fit fits nothing)
            the GPU still speeds up large prompt batches (it costs about 0.7 to 1 GiB of GPU memory; generation
            stays on the CPU). This is intended; use `devices="none"` for no GPU use at all.
        devices: which devices to use, by the names `backends_info()` prints. `None` (default) = every GPU,
            `"none"` = CPU only (no GPU memory is allocated), `"CUDA0"` or `"CUDA0,CUDA1"` = exactly these.
        n_ctx: context size in tokens (prompt plus answer); 0 = the library default (4096, capped at the model's).
        n_threads: CPU threads; 0 = the library default.
        n_batch: prompt batch size; 0 = the library default (512).
        vram_budget: auto-fit only: GPU memory budget in bytes (0 = the free memory the devices report).
        use_mmap: memory-map the model file where possible.

    Raises:
        LittleMoreError: the file is missing or not a model, or no backend/device is available, etc. The error
            carries the status name and the library's own message.

    Use it as a context manager (`with Model(...) as m:`) or call `close()`; both free everything, in the right
    order, and may be repeated.
    """

    def __init__(self, path: str | PathLike[str], n_gpu_layers: int | str = "auto", *, devices: str | None = None,
                 n_ctx: int = 0, n_threads: int = 0, n_batch: int = 0, vram_budget: int = 0,
                 use_mmap: bool = True) -> None:
        # Fields first: __del__ must work even when validation below raises.
        self._model: ctypes.c_void_p | None = None
        self._session: ctypes.c_void_p | None = None
        self._closed = False
        self._lock = threading.Lock()  # the session: one generation at a time (also taken by close())
        self._model_lock = threading.Lock()  # the model pointer: held briefly by tokenize / info, and by close()
        self._active = _ActiveStreams()
        layers = _parse_n_gpu_layers(n_gpu_layers)
        n_ctx = _int_in_range("n_ctx", n_ctx, 0, 2**32 - 1)  # 0 = the library default
        n_threads = _int_in_range("n_threads", n_threads, 0, 2**31 - 1)
        n_batch = _int_in_range("n_batch", n_batch, 0, 2**32 - 1)
        vram_budget = _int_in_range("vram_budget", vram_budget, 0, 2**64 - 1)
        lib = _lib.load()
        path_bytes = _lib.utf8(fspath(path), "path")
        dev_bytes = None if devices is None else _lib.utf8(devices, "devices")

        opts = _lib.ModelOptions()
        _lib.check(lib.lm_model_options_default(ctypes.byref(opts)), lib)  # sets struct_size and the defaults
        opts.n_gpu_layers = layers
        opts.devices = dev_bytes  # a NULL pointer for None; the bytes object outlives the call (local variable)
        opts.vram_budget = int(vram_budget)
        opts.use_mmap = 1 if use_mmap else 0
        handle = ctypes.c_void_p()
        _lib.check(lib.lm_model_load(path_bytes, ctypes.byref(opts), ctypes.byref(handle)), lib)
        self._model = handle

        sopts = _lib.SessionOptions()
        try:
            _lib.check(lib.lm_session_options_default(ctypes.byref(sopts)), lib)
            sopts.n_ctx, sopts.n_threads, sopts.n_batch = int(n_ctx), int(n_threads), int(n_batch)
            sess = ctypes.c_void_p()
            _lib.check(lib.lm_session_new(self._model, ctypes.byref(sopts), ctypes.byref(sess)), lib)
            self._session = sess
        except BaseException:
            self.close()  # do not leak the loaded model when the session cannot be created
            raise
        self.path = fspath(path)
        _LIVE_MODELS.add(self)

    # -- lifetime -----------------------------------------------------------------------------------------------
    @property
    def closed(self) -> bool:
        """True once `close()` has been called."""
        return self._closed

    def close(self) -> None:
        """Free the session and the model. Safe to call repeatedly and from any thread.

        Running generations are asked to stop first and `close()` then waits for them (a stream ends quietly, a
        non-streaming call raises `LittleMoreError` with status name "CLOSED"). The session is freed before the
        model, as the C API requires. Raises `LittleMoreError` if the library refuses to free something.
        """
        self._closed = True
        self._active.stop_all()
        with self._lock:  # waits for a generation that is still finishing
            with self._model_lock:  # and for a tokenize / info call
                lib = _lib.load()
                if self._session is not None:
                    _lib.check(lib.lm_session_free(self._session), lib)
                    self._session = None
                if self._model is not None:
                    _lib.check(lib.lm_model_free(self._model), lib)
                    self._model = None

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - never raise from a finalizer (e.g. at interpreter shutdown)
            pass

    def __repr__(self) -> str:
        return f"Model({getattr(self, 'path', '?')!r}{', closed' if self._closed else ''})"

    def _check_open(self) -> None:
        if self._closed or self._session is None:
            raise LittleMoreError("the model is closed", None, "CLOSED")

    # -- generation ---------------------------------------------------------------------------------------------
    @overload
    def chat(self, messages: Sequence[Mapping[str, str]], stream: Literal[False] = False, **sampling: Any) -> str: ...
    @overload
    def chat(self, messages: Sequence[Mapping[str, str]], stream: Literal[True], **sampling: Any) -> Iterator[str]: ...

    def chat(self, messages: Sequence[Mapping[str, str]], stream: bool = False, **sampling: Any) -> str | Iterator[str]:
        """Answer a conversation, formatted with the model's own chat template.

        Every call starts from scratch: pass the **full history** (earlier user and assistant messages too) each
        time; the model keeps no memory between calls.

        Args:
            messages: list of `{"role": "system" | "user" | "assistant", "content": str}`.
            stream: False (default) returns the whole answer as a `str`; True returns a generator of `str`
                pieces. A stream may be consumed from any thread; stop early by closing or dropping it.
            **sampling: `temperature`, `top_k`, `top_p`, `seed`, `max_tokens` (see `_make_sampling`). The default
                is deterministic: the same input gives the same text. Pass `seed="random"` for varied answers.

        Special-token text inside messages (e.g. "<|im_end|>") is turned into the real control token by the
        library; remove it from untrusted input.
        """
        arr, keep = _encode_messages(messages)
        smp = _make_sampling(**sampling)
        lib = _lib.load()

        def invoke(cb: Any) -> int:
            _ = keep  # keeps the encoded bytes alive for the whole C call
            return lib.lm_chat(self._session, arr, len(arr), ctypes.byref(smp), cb, None)

        return self._run(invoke, stream)

    @overload
    def generate(self, prompt: str, stream: Literal[False] = False, **sampling: Any) -> str: ...
    @overload
    def generate(self, prompt: str, stream: Literal[True], **sampling: Any) -> Iterator[str]: ...

    def generate(self, prompt: str, stream: bool = False, **sampling: Any) -> str | Iterator[str]:
        """Continue a raw prompt (no chat template). Same `stream` and sampling rules as `chat`."""
        text = _lib.utf8(prompt, "prompt")
        smp = _make_sampling(**sampling)
        lib = _lib.load()

        def invoke(cb: Any) -> int:
            return lib.lm_generate(self._session, text, ctypes.byref(smp), cb, None)

        return self._run(invoke, stream)

    def _refuse_if_stream_open_here(self) -> None:
        """A stream that this thread started and neither finished nor closed owns the session, and this thread is
        the only one that could ever release it: waiting would deadlock, so say so instead."""
        if self._active.owned_by(threading.get_ident()):
            raise LittleMoreError(
                "a stream on this model is still open in this thread: consume it to the end or close it "
                "(stream.close() or contextlib.closing(stream)) before starting another generation",
                None, "STREAM_OPEN")

    def _run(self, invoke: Callable[[Any], int], stream: bool) -> str | Iterator[str]:
        self._check_open()
        self._refuse_if_stream_open_here()
        if stream:
            return _stream_text(invoke, self._lock, self._active, self._check_open, self._refuse_if_stream_open_here)
        return _collect_text(invoke, self._lock, self._active, self._check_open)

    # -- tokens -------------------------------------------------------------------------------------------------
    def tokenize(self, text: str, add_special: bool = False) -> list[int]:
        """Token ids of `text`. `add_special=True` adds the model's beginning-of-text token (if it uses one) and
        parses special-token text such as "<|im_start|>" as tokens."""
        data = _lib.utf8(text, "text")
        lib = _lib.load()
        with self._model_lock:  # the model only: works while a stream is open (the C API allows it)
            self._check_open()
            n = ctypes.c_size_t(0)
            st = lib.lm_tokenize(self._model, data, len(data), None, 0, int(add_special), ctypes.byref(n))
            if st not in (_lib.LM_OK, _lib.LM_ERR_BUFFER_TOO_SMALL):  # then n is the size needed
                _lib.check(st, lib)
            buf = (ctypes.c_int32 * max(n.value, 1))()
            _lib.check(lib.lm_tokenize(self._model, data, len(data), buf, n.value, int(add_special),
                                       ctypes.byref(n)), lib)
            return list(buf[: n.value])

    def detokenize(self, tokens: Sequence[int], special: bool = False) -> str:
        """Text of the token ids (`special=True` renders special tokens as text instead of leaving them out)."""
        ids = [_int_in_range("token id", t, -(2**31), 2**31 - 1) for t in tokens]  # no silent int32 wraparound
        arr = (ctypes.c_int32 * max(len(ids), 1))(*ids)
        lib = _lib.load()
        with self._model_lock:
            self._check_open()
            n = ctypes.c_size_t(0)
            st = lib.lm_detokenize(self._model, arr, len(ids), None, 0, int(special), ctypes.byref(n))
            if st not in (_lib.LM_OK, _lib.LM_ERR_BUFFER_TOO_SMALL):
                _lib.check(st, lib)
            buf = ctypes.create_string_buffer(n.value + 1)
            _lib.check(lib.lm_detokenize(self._model, arr, len(ids), buf, len(buf), int(special),
                                         ctypes.byref(n)), lib)
            return buf.raw[: n.value].decode("utf-8", "replace")

    # -- information --------------------------------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        """Facts about the library, its backends and this model.

        Returns a dict with `library_version`, `api_version`, `backends` (see `backends_info()`) and `model`:
        `n_layer`, `n_gpu_layers` (layers actually put on the GPU after auto-fit; `n_layer + 1` = everything),
        `n_ctx_train`, `n_vocab`, `n_embd`, `size_bytes`, `n_params`, the auto-fit figures (`fit_*`, 0 when
        auto-fit did not run), `gpu_devices` (comma-separated, "" = none) and `description`.
        """
        lib = _lib.load()
        with self._model_lock:
            self._check_open()
            mi = _lib.ModelInfo()
            mi.struct_size = ctypes.sizeof(_lib.ModelInfo)
            _lib.check(lib.lm_model_get_info(self._model, ctypes.byref(mi)), lib)
        model = {name: getattr(mi, name) for name, _ in _lib.ModelInfo._fields_
                 if name not in ("struct_size", "reserved0", "gpu_devices", "description")}
        model["gpu_devices"] = mi.gpu_devices.decode("utf-8", "replace")
        model["description"] = mi.description.decode("utf-8", "replace")
        return {"library_version": version(), "api_version": int(lib.lm_api_version()),
                "backends": backends_info(), "model": model}
