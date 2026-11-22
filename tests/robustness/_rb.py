"""Shared helpers of the robustness tests (tests/robustness): child processes, exit classification, GGUF layout.

* `run_child(snippet)` runs a snippet in `_child.py` (a fresh Python process with the raw C API helpers) with a
  timeout and classifies how it ended (`Outcome`): a clean exit, a Python exception, an abort, an access violation,
  any other crash code, or a hang. Robustness tests assert `outcome.clean`.
* `GGUFLayout` parses the metadata part of a GGUF file (header, key/value pairs, tensor infos) and records the byte
  offset and width of every field, so the fuzz tests can mutate exactly a length, a type tag, a dimension or an
  offset instead of only random bytes.
* `rss_mib` / `gpu_used_mib`: memory measurements for the leak and resource tests.
"""
from __future__ import annotations

import ctypes
import json
import os
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHILD = Path(__file__).resolve().parent / "_child.py"
SCRATCH = ROOT / ".cache" / "robust"
CUDA_BIN = Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin")

# Windows exit codes of a process that died instead of returning.
_CRASH_NAMES = {
    3: "abort",                       # abort() (GGML_ABORT / GGML_ASSERT end here)
    0xC0000005: "access-violation",
    0xC00000FD: "stack-overflow",
    0xC0000409: "fail-fast",          # __fastfail / stack buffer overrun / abort with _CALL_REPORTFAULT
    0xC0000374: "heap-corruption",
    0xC000001D: "illegal-instruction",
    0xC0000094: "integer-divide-by-zero",
    0xC0000420: "assertion-failure",
    0x40000015: "abort-fastfail",
    0xC0000135: "dll-not-found",
}


@dataclass
class Outcome:
    """How a child process ended."""

    returncode: int | None          # None = killed after the timeout
    stdout: str
    stderr: str
    results: list[dict] = field(default_factory=list)

    @property
    def kind(self) -> str:
        """"ok" (exit 0), "python-error" (exit 1: an exception in the snippet), "timeout", or a crash name."""
        if self.returncode is None:
            return "timeout"
        if self.returncode == 0:
            return "ok"
        if self.returncode == 1:
            return "python-error"
        code = self.returncode & 0xFFFFFFFF
        return _CRASH_NAMES.get(code, f"exit-0x{code:08X}")

    @property
    def clean(self) -> bool:
        """The process returned normally (exit 0) and reported at least one result."""
        return self.kind == "ok" and bool(self.results)

    @property
    def result(self) -> dict:
        """The last RESULT line (what the snippet emitted last)."""
        return self.results[-1] if self.results else {}

    def describe(self) -> str:
        """Short text for assertion messages."""
        return (f"child ended with {self.kind} (exit {self.returncode}); results {self.results[-3:]}; "
                f"stderr tail: {self.stderr[-1500:]!r}")


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment of a child: the CUDA 12.9 bin first on PATH (the CUDA plugin's runtime DLLs), quiet engine log."""
    env = dict(os.environ)
    if CUDA_BIN.is_dir():
        env["PATH"] = str(CUDA_BIN) + os.pathsep + env.get("PATH", "")
    env.setdefault("LITTLEMORE_LOG", "none")
    env["PYTHONIOENCODING"] = "utf-8"
    if extra:
        env.update(extra)
    return env


def run_child(snippet: str, timeout: float = 120, env: dict[str, str] | None = None, cwd: Path | None = None,
              python: str | None = None) -> Outcome:
    """Run `snippet` in a fresh `_child.py` process; returns its classified `Outcome` (never raises on a crash)."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    src = SCRATCH / f"snippet_{os.getpid()}_{abs(hash(snippet)) % 10**12}.py"
    src.write_text(snippet, encoding="utf-8")
    try:
        p = subprocess.run([python or sys.executable, str(CHILD), "@" + str(src)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, env=child_env(env), cwd=cwd,
                           stdin=subprocess.DEVNULL)
        rc, out, errs = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc = None
        out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        errs = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    finally:
        try:
            src.unlink()
        except OSError:
            pass
    results = []
    for line in out.splitlines():
        if line.startswith("RESULT "):
            try:
                results.append(json.loads(line[7:]))
            except json.JSONDecodeError:
                pass
    return Outcome(rc, out, errs, results)


# --------------------------------------------------------------------------------------------------------------
# GGUF metadata layout (GGUF v3: little-endian; see engine/ggml/include/gguf.h and ggml/src/gguf.cpp)
# --------------------------------------------------------------------------------------------------------------
# GGUF value types and the byte width of the scalar ones.
GGUF_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
GGUF_STRING, GGUF_ARRAY = 8, 9


@dataclass
class Field:
    """One fixed-width integer field of the metadata: where it is, how wide, and what it means."""

    offset: int
    width: int
    what: str            # e.g. "header.n_tensors", "kv[3].key_len", "tensor[5].dim[1]"
    value: int


@dataclass
class GGUFLayout:
    """Byte layout of a GGUF file's metadata (everything before the tensor data)."""

    fields: list[Field]
    kv: list[dict]             # per key/value pair: key, type, offset of the entry, offset after it
    tensors: list[dict]        # per tensor info: name, type, dims, data offset, entry offsets
    kv_start: int
    tensor_info_start: int
    tensor_info_end: int
    data_start: int
    file_size: int

    @staticmethod
    def parse(path: Path) -> "GGUFLayout":
        """Parse `path` (a well-formed GGUF v3 file) and record the offset of every field."""
        data = path.read_bytes() if path.stat().st_size < 64 << 20 else _head(path, 64 << 20)
        fields: list[Field] = []
        pos = 0

        def take(fmt: str, what: str) -> int:
            nonlocal pos
            w = struct.calcsize("<" + fmt)
            (v,) = struct.unpack_from("<" + fmt, data, pos)
            fields.append(Field(pos, w, what, v))
            pos += w
            return v

        if data[:4] != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        pos = 4
        take("I", "header.version")
        n_tensors = take("Q", "header.n_tensors")
        n_kv = take("Q", "header.n_kv")
        kv_start = pos
        kvs = []
        alignment = 32
        for i in range(n_kv):
            start = pos
            klen = take("Q", f"kv[{i}].key_len")
            key = data[pos:pos + klen].decode("utf-8", "replace")
            pos += klen
            vtype = take("I", f"kv[{i}].type")
            value = None
            if vtype == GGUF_STRING:
                slen = take("Q", f"kv[{i}].str_len")
                value = data[pos:pos + slen]
                pos += slen
            elif vtype == GGUF_ARRAY:
                atype = take("I", f"kv[{i}].arr_type")
                n = take("Q", f"kv[{i}].arr_n")
                if atype == GGUF_STRING:
                    for j in range(n):
                        (slen,) = struct.unpack_from("<Q", data, pos)
                        if j < 4:  # record the first few element lengths as fuzz targets (the vocab has 100k+)
                            fields.append(Field(pos, 8, f"kv[{i}].arr[{j}].str_len", slen))
                        pos += 8 + slen
                else:
                    pos += GGUF_SCALAR[atype] * n
            else:
                w = GGUF_SCALAR[vtype]
                value = int.from_bytes(data[pos:pos + w], "little")
                fields.append(Field(pos, w, f"kv[{i}].value", value))
                pos += w
            if key == "general.alignment" and isinstance(value, int):
                alignment = value
            kvs.append({"key": key, "type": vtype, "start": start, "end": pos, "value": value})
        ti_start = pos
        tensors = []
        for i in range(n_tensors):
            start = pos
            nlen = take("Q", f"tensor[{i}].name_len")
            tname = data[pos:pos + nlen].decode("utf-8", "replace")
            pos += nlen
            nd = take("I", f"tensor[{i}].n_dims")
            dims = [take("Q", f"tensor[{i}].dim[{d}]") for d in range(nd)]
            ttype = take("I", f"tensor[{i}].type")
            off = take("Q", f"tensor[{i}].offset")
            tensors.append({"name": tname, "type": ttype, "dims": dims, "offset": off, "start": start, "end": pos,
                            "name_start": start + 8, "name_len": nlen})
        ti_end = pos
        data_start = (ti_end + alignment - 1) // alignment * alignment
        return GGUFLayout(fields, kvs, tensors, kv_start, ti_start, ti_end, data_start, path.stat().st_size)

    def field(self, what: str) -> Field:
        """The field called `what` (e.g. "header.n_kv")."""
        for f in self.fields:
            if f.what == what:
                return f
        raise KeyError(what)

    def kv_index(self, key: str) -> int:
        """Index of a key/value pair by its key."""
        for i, kv in enumerate(self.kv):
            if kv["key"] == key:
                return i
        raise KeyError(key)

    def tensor_index(self, tname: str) -> int:
        """Index of a tensor info by tensor name."""
        for i, t in enumerate(self.tensors):
            if t["name"] == tname:
                return i
        raise KeyError(tname)


def _head(path: Path, n: int) -> bytes:
    """The first `n` bytes of a file (the metadata of a large model is at its start)."""
    with open(path, "rb") as f:
        return f.read(n)


def patch_int(buf: bytearray, f: Field, value: int) -> None:
    """Write `value` into the field `f` of `buf` (two's complement for negative values, truncated to the width)."""
    buf[f.offset:f.offset + f.width] = (value & ((1 << (8 * f.width)) - 1)).to_bytes(f.width, "little")


# --------------------------------------------------------------------------------------------------------------
# Memory measurements
# --------------------------------------------------------------------------------------------------------------
def gpu_used_mib() -> int | None:
    """GPU memory in use on GPU 0 according to nvidia-smi (all processes), or None without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def gpu_free_mib() -> int | None:
    """Free GPU memory on GPU 0 according to nvidia-smi, or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def free_ram_mib() -> int:
    """Free physical memory of the machine, MiB (Windows: GlobalMemoryStatusEx)."""
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return int(m.ullAvailPhys >> 20)
