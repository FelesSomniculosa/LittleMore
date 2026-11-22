"""Low-level ctypes bindings to the LittleMore C API (`include/littlemore.h`).

Nothing here needs a compiler: the shared library is loaded with `ctypes` and every function is declared with
explicit argument and result types, so a wrong call fails loudly instead of corrupting memory. Application code
should use `littlemore.Model` (model.py); this module is the foundation under it.

Three things in here are easy to get wrong, so they are explained where they happen:

1. Finding the library and its dependencies (`find_library`, `dll_search_dirs`, `load`). On Windows, Python 3.8+
   no longer searches PATH for the DLLs that a DLL depends on. The CUDA backend plugin (ggml-cuda.dll) needs the
   CUDA runtime DLLs, so unless the SDK registers their folder with `os.add_dll_directory()` BEFORE the library
   is loaded, the plugin silently fails to load and only the CPU (and Vulkan) devices appear.
2. Struct layouts. The C option structs start with a `struct_size` field and have explicit `reserved` fields
   instead of hidden padding, so a ctypes `Structure` with the same fields in the same order has the same layout
   as the C struct on the supported 64-bit platforms. The tests compare the offsets with the header and with
   the compiled library.
3. Errors. Every fallible C function returns an `lm_status`; the human-readable message is in a thread-local
   buffer (`lm_last_error`), so `check()` reads it immediately, on the thread that made the failing call.
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable

# The C API version this binding was written for (LM_API_VERSION in the header). load() refuses a library
# with another value, because an ABI break would make the struct layouts below wrong.
EXPECTED_API_VERSION = 1

# lm_status values the SDK treats specially (the full list is in the header).
LM_OK = 0
LM_ERR_BUFFER_TOO_SMALL = 11  # an output buffer is too small; the needed size is reported (a test checks the value against the header)
LM_GPU_LAYERS_AUTO = -1
LM_SEED_RANDOM = 0xFFFFFFFF

_PKG_DIR = Path(__file__).resolve().parent
# A source checkout is python/littlemore/_lib.py inside the repository; its CMake build puts the library and the
# backend plugins in <repo>/build/bin. (In an installed wheel this path simply does not exist.)
_REPO_ROOT = _PKG_DIR.parents[1]


# --------------------------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------------------------
class LittleMoreError(Exception):
    """An error reported by the LittleMore library (or by the SDK itself).

    Attributes:
        status: the numeric `lm_status` code (1 to 14), or None for errors that do not come from a C call.
        status_name: the constant's name, e.g. "LM_ERR_MODEL_LOAD"; "CLOSED" for use of a closed model.
        message: the text of `lm_last_error()` fetched right after the failing call (it starts with the name of
            the failing C function), or the SDK's own explanation.
    """

    def __init__(self, message: str, status: int | None = None, status_name: str = "ERROR") -> None:
        super().__init__(f"{message} [{status_name}]")
        self.message = message
        self.status = status
        self.status_name = status_name

    def __reduce__(self):  # exceptions with extra constructor arguments need this to survive pickle / multiprocessing
        return (type(self), (self.message, self.status, self.status_name))


class LibraryNotFoundError(LittleMoreError):
    """The LittleMore shared library could not be found or does not match this SDK."""

    def __init__(self, message: str) -> None:
        super().__init__(message, None, "LIBRARY")

    def __reduce__(self):
        return (type(self), (self.message,))


# --------------------------------------------------------------------------------------------------------------
# Struct layouts (see the header for the meaning of every field)
# --------------------------------------------------------------------------------------------------------------
class ModelOptions(ctypes.Structure):
    """`lm_model_options`. Fill it with `lm_model_options_default` (which also sets `struct_size` and the defaults),
    then change the fields you need."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("n_gpu_layers", ctypes.c_int32),
        ("reserved0", ctypes.c_int32),
        ("devices", ctypes.c_char_p),
        ("vram_budget", ctypes.c_uint64),
        ("fit_n_ctx", ctypes.c_uint32),
        ("use_mmap", ctypes.c_int32),
    ]


class SessionOptions(ctypes.Structure):
    """`lm_session_options`."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("n_ctx", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_batch", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
    ]


class Sampling(ctypes.Structure):
    """`lm_sampling`."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("seed", ctypes.c_uint32),
        ("max_tokens", ctypes.c_int32),
        ("reserved0", ctypes.c_int32),
    ]


class Message(ctypes.Structure):
    """`lm_message`: a role and a text, both UTF-8, NUL-terminated. Fixed layout (no struct_size)."""

    _fields_ = [("role", ctypes.c_char_p), ("content", ctypes.c_char_p)]


class ModelInfo(ctypes.Structure):
    """`lm_model_info`."""

    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("n_layer", ctypes.c_int32),
        ("n_gpu_layers", ctypes.c_int32),
        ("n_ctx_train", ctypes.c_int32),
        ("n_vocab", ctypes.c_int32),
        ("n_embd", ctypes.c_int32),
        ("reserved0", ctypes.c_int32),
        ("size_bytes", ctypes.c_uint64),
        ("n_params", ctypes.c_uint64),
        ("fit_budget_bytes", ctypes.c_uint64),
        ("fit_reserve_bytes", ctypes.c_uint64),
        ("fit_layer_bytes", ctypes.c_uint64),
        ("fit_head_bytes", ctypes.c_uint64),
        ("gpu_devices", ctypes.c_char * 128),
        ("description", ctypes.c_char * 128),
    ]


# `int (LM_CALL *lm_token_cb)(const char *piece, size_t len, void *user)`. LM_CALL is __cdecl on Windows, which is
# what ctypes.CFUNCTYPE means (WINFUNCTYPE would be stdcall). The piece pointer is taken as a raw address, not
# as c_char_p, because ctypes would stop a c_char_p at the first NUL byte and the model may produce any bytes.
TokenCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)


# --------------------------------------------------------------------------------------------------------------
# Finding the library and its DLL dependencies
# --------------------------------------------------------------------------------------------------------------
def _library_file_names() -> list[str]:
    """Glob patterns for the shared library's file name on this platform (a versioned name on Linux)."""
    if sys.platform == "win32":
        return ["littlemore.dll"]
    if sys.platform == "darwin":
        return ["liblittlemore.dylib", "liblittlemore.*.dylib"]
    return ["liblittlemore.so", "liblittlemore.so.*"]


def _candidate_library_dirs() -> list[Path]:
    """Folders searched for the library, in priority order: the wheel's bundled folder, then a source
    checkout's CMake build output."""
    return [_PKG_DIR / "_bin", _REPO_ROOT / "build" / "bin"]


_WARNED_STALE = False  # the "two different builds found" warning is given once per process


def find_library() -> Path:
    """Return the path of the LittleMore shared library.

    Order: the file named by the environment variable `LITTLEMORE_LIBRARY` (always wins); otherwise the library
    in the `_bin` folder inside this package (where the wheel bundles it and the backend plugins) and in
    `<repo>/build/bin` (a source checkout built with CMake). When BOTH exist, the one with the NEWER modification
    time is used, so a fresh CMake build is not shadowed by an old copy left in `_bin`; if the two files differ
    (size or time), a `UserWarning` names both and the one chosen (once per process). The backend plugins are
    loaded by the library itself from ITS folder, so the chosen file's folder must hold them too.

    Raises:
        LibraryNotFoundError: nothing found; the message lists what was searched.
    """
    global _WARNED_STALE
    explicit = os.environ.get("LITTLEMORE_LIBRARY")
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise LibraryNotFoundError(f"LITTLEMORE_LIBRARY={explicit!r} is not a file")
        return p.resolve()
    searched = _candidate_library_dirs()
    found: list[Path] = []
    for d in searched:
        for pattern in _library_file_names():
            hits = sorted(glob.glob(str(d / pattern)))
            if hits:
                found.append(Path(hits[0]).resolve())
                break
    if not found:
        raise LibraryNotFoundError(
            "the littlemore shared library was not found; searched " + ", ".join(str(d) for d in searched)
            + ". Install a wheel that bundles it, build it with CMake (docs/development.md), or set LITTLEMORE_LIBRARY.")
    best = max(found, key=lambda f: f.stat().st_mtime)  # max() keeps the first (higher-priority) of equal times
    stale = [f for f in found if f != best
             and (f.stat().st_size != best.stat().st_size or f.stat().st_mtime != best.stat().st_mtime)]
    if stale and not _WARNED_STALE:
        _WARNED_STALE = True
        warnings.warn(f"found two different littlemore libraries: using the newer {best}; the stale copy "
                      f"{stale[0]} is ignored (delete it, or set LITTLEMORE_LIBRARY to choose explicitly)",
                      UserWarning, stacklevel=2)
    return best


def _split_paths(value: str | None) -> list[str]:
    """Split a PATH-style string into its non-empty folders."""
    return [p for p in (value or "").split(os.pathsep) if p]


def _has_cuda_runtime(folder: str | Path) -> bool:
    """True when `folder` holds a CUDA runtime DLL (cudart64_<major>.dll)."""
    try:
        return bool(glob.glob(os.path.join(str(folder), "cudart64_*.dll")))
    except OSError:
        return False


def _nvidia_wheel_bin_dirs() -> list[str]:
    """`bin` folders of the NVIDIA runtime wheels from PyPI (nvidia-cuda-runtime-cu12, nvidia-cublas-cu12, ...),
    which install their DLLs under site-packages/nvidia/<component>/bin on Windows.

    Only real site-packages folders are searched (the interpreter's purelib/platlib and its site directories):
    an `nvidia` folder that merely happens to be first on sys.path, e.g. in the working directory, is NOT used,
    because a folder registered for DLL loading is trusted to hold the CUDA libraries."""
    import site
    import sysconfig

    roots: list[str] = []
    paths = sysconfig.get_paths()
    roots += [paths.get("purelib", ""), paths.get("platlib", "")]
    try:
        roots += site.getsitepackages()
        roots.append(site.getusersitepackages())
    except (AttributeError, OSError):  # some embedded interpreters lack these
        pass
    found: list[str] = []
    for root in roots:
        if root and os.path.isabs(root):
            for d in sorted(glob.glob(os.path.join(root, "nvidia", "*", "bin"))):
                if d not in found:
                    found.append(d)
    return found


def _default_cuda_toolkit_bins() -> list[str]:
    """`bin` folders of CUDA toolkits in the installer's default location, newest version first."""
    root = os.environ.get("ProgramFiles", r"C:\Program Files")
    pattern = os.path.join(root, "NVIDIA GPU Computing Toolkit", "CUDA", "v*", "bin")

    def version_key(p: str) -> tuple[int, ...]:
        """Sort key: the numeric version of a toolkit folder such as CUDA/v12.9/bin."""
        name = Path(p).parent.name.lstrip("v")
        return tuple(int(x) for x in name.split(".") if x.isdigit())

    return sorted(glob.glob(pattern), key=version_key, reverse=True)


def dll_search_dirs(library_dir: str | Path | None = None) -> list[str]:
    """Folders to register with `os.add_dll_directory` before loading the library (Windows only; [] elsewhere).

    Why: Python 3.8+ does not search PATH for DLL dependencies, but the CUDA backend plugin needs the CUDA
    runtime DLLs (cudart, cublas). Sources, most specific first, without duplicates and only folders that exist:

    1. `LITTLEMORE_DLL_DIR` (several folders separated by `os.pathsep`, absolute paths only): the explicit override.
    Every folder below is registered with `os.add_dll_directory`; Windows defines no search order between
    folders added that way, so the order here is only the order of registration, not a priority.
    2. `CUDA_PATH` and `CUDA_PATH_V*` environment variables: `<toolkit>/bin`.
    3. Folders on PATH that contain a CUDA runtime DLL (what a user who "has CUDA on PATH" expects to work).
    4. `bin` folders of the NVIDIA runtime wheels installed in this Python (`pip install nvidia-cublas-cu12 ...`).
    5. CUDA toolkits in the installer's default location (newest first).
    6. `library_dir` (the library's own folder, which holds the plugins and anything a wheel bundles).
    """
    if sys.platform != "win32":
        return []
    candidates: list[str] = _split_paths(os.environ.get("LITTLEMORE_DLL_DIR"))
    for name in sorted(os.environ):
        if name == "CUDA_PATH" or name.startswith("CUDA_PATH_V"):
            candidates.append(os.path.join(os.environ[name], "bin"))
    candidates += [p for p in _split_paths(os.environ.get("PATH")) if os.path.isabs(p) and _has_cuda_runtime(p)]
    candidates += _nvidia_wheel_bin_dirs()
    candidates += [p for p in _default_cuda_toolkit_bins() if _has_cuda_runtime(p)]
    if library_dir is not None:
        candidates.append(str(library_dir))
    result: list[str] = []
    seen: set[str] = set()
    explicit = _split_paths(os.environ.get("LITTLEMORE_DLL_DIR"))
    for c in candidates:
        # Relative entries ("." or "bin") would depend on the working directory, and a DLL folder is trusted to
        # hold code that gets loaded: skip them (with a warning when the user asked for one explicitly).
        if not os.path.isabs(c):
            if c in explicit:
                warnings.warn(f"LITTLEMORE_DLL_DIR entry {c!r} is not an absolute path and is ignored", UserWarning,
                              stacklevel=2)
            continue
        key = os.path.normcase(os.path.abspath(c))
        if os.path.isdir(c) and key not in seen:
            seen.add(key)
            result.append(os.path.abspath(c))
    return result


# The handles os.add_dll_directory returns: the folder stays registered only while the handle lives, so they are
# kept for the whole process.
_DLL_HANDLES: list[object] = []
_LIB: ctypes.CDLL | None = None


def _declare_prototypes(dll: ctypes.CDLL) -> None:
    """Give every C function its exact argument and result types (see the header)."""
    c_i32, c_sz, c_vp, c_cp = ctypes.c_int32, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_char_p
    sig = {
        "lm_version": (c_cp, []),
        "lm_api_version": (c_i32, []),
        "lm_last_error": (c_cp, []),
        "lm_status_name": (c_cp, [c_i32]),
        "lm_backends_info": (c_sz, [c_vp, c_sz]),
        "lm_model_options_default": (c_i32, [ctypes.POINTER(ModelOptions)]),
        "lm_model_load": (c_i32, [c_cp, ctypes.POINTER(ModelOptions), ctypes.POINTER(c_vp)]),
        "lm_model_free": (c_i32, [c_vp]),
        "lm_model_get_info": (c_i32, [c_vp, ctypes.POINTER(ModelInfo)]),
        "lm_session_options_default": (c_i32, [ctypes.POINTER(SessionOptions)]),
        "lm_session_new": (c_i32, [c_vp, ctypes.POINTER(SessionOptions), ctypes.POINTER(c_vp)]),
        "lm_session_free": (c_i32, [c_vp]),
        "lm_sampling_default": (c_i32, [ctypes.POINTER(Sampling)]),
        "lm_chat": (c_i32, [c_vp, ctypes.POINTER(Message), c_sz, ctypes.POINTER(Sampling), TokenCallback, c_vp]),
        "lm_generate": (c_i32, [c_vp, c_cp, ctypes.POINTER(Sampling), TokenCallback, c_vp]),
        "lm_tokenize": (c_i32, [c_vp, c_cp, c_sz, ctypes.POINTER(c_i32), c_sz, ctypes.c_int, ctypes.POINTER(c_sz)]),
        "lm_detokenize": (c_i32, [c_vp, ctypes.POINTER(c_i32), c_sz, ctypes.c_char_p, c_sz, ctypes.c_int,
                                  ctypes.POINTER(c_sz)]),
    }
    for name, (restype, argtypes) in sig.items():
        fn = getattr(dll, name)
        fn.restype = restype
        fn.argtypes = argtypes


def _check_api_version(dll: ctypes.CDLL) -> None:
    """Refuse a library built for another ABI version (the header says to compare lm_api_version())."""
    found = dll.lm_api_version()
    if found != EXPECTED_API_VERSION:
        raise LibraryNotFoundError(f"the littlemore library has API version {found}, this SDK needs "
                                   f"{EXPECTED_API_VERSION}; install matching versions of both")


def load() -> ctypes.CDLL:
    """Load the shared library once and return it with all prototypes declared.

    On Windows the DLL folders of `dll_search_dirs()` are registered with `os.add_dll_directory()` first, so the
    CUDA runtime DLLs can be found when the backend plugins load (the plugins load lazily, at the first call
    that needs a backend, from the library's own folder).

    Raises:
        LibraryNotFoundError: library missing, unloadable, or of another API version.
    """
    global _LIB
    if _LIB is not None:
        return _LIB
    path = find_library()
    if sys.platform == "win32":
        for d in dll_search_dirs(path.parent):
            try:
                _DLL_HANDLES.append(os.add_dll_directory(d))
            except OSError:  # e.g. the folder vanished or is not accessible: the others may still be enough
                continue
    try:
        dll = ctypes.CDLL(str(path))
    except OSError as e:
        raise LibraryNotFoundError(f"cannot load {path}: {e}") from e
    _declare_prototypes(dll)
    _check_api_version(dll)
    _LIB = dll
    return dll


# --------------------------------------------------------------------------------------------------------------
# Small helpers shared by the SDK
# --------------------------------------------------------------------------------------------------------------
def last_error(dll: ctypes.CDLL | None = None) -> str:
    """The calling thread's `lm_last_error()` text (thread-local in the library: call it on the thread that made
    the failing call, immediately afterwards)."""
    return (dll or load()).lm_last_error().decode("utf-8", "replace")


def check(status: int, dll: ctypes.CDLL | None = None) -> None:
    """Raise `LittleMoreError` for a non-zero `lm_status`, with the status name and the library's message."""
    if status == LM_OK:
        return
    lib = dll or load()
    name = lib.lm_status_name(status).decode("ascii", "replace")
    raise LittleMoreError(last_error(lib), status, name)


def backends_text(dll: ctypes.CDLL | None = None) -> str:
    """The text of `lm_backends_info()`: one device per line. Retries with a bigger buffer because the text
    contains the devices' current free memory, so its length can change between two calls."""
    lib = dll or load()
    need = lib.lm_backends_info(None, 0)
    if need == 0:
        raise LittleMoreError(last_error(lib) or "lm_backends_info failed", None, "LM_ERR_BACKEND")
    for _ in range(4):
        buf = ctypes.create_string_buffer(need + 256)  # slack: the free-memory figures may print wider
        n = lib.lm_backends_info(buf, len(buf))
        if n < len(buf):
            return buf.value.decode("utf-8", "replace")
        need = n
    raise LittleMoreError("lm_backends_info: the text kept growing", None, "LM_ERR_INTERNAL")


def utf8(text: str, what: str) -> bytes:
    """Encode `text` for the C API; an embedded NUL would silently truncate it there, so it is refused."""
    if not isinstance(text, str):
        raise TypeError(f"{what} must be a str, not {type(text).__name__}")
    if "\0" in text:
        raise ValueError(f"{what} must not contain a NUL character")
    return text.encode("utf-8")


def iter_lines(text: str) -> Iterable[str]:
    """Non-empty lines of a text."""
    return (ln for ln in text.splitlines() if ln.strip())
