"""The LittleMore GGUF format marker: a file written for another B1 type number or a newer format is refused cleanly.

Background
    A GGUF file stores each tensor's ggml type as a bare number. B1 (the 1-bit sign type of LittleBit models) exists
    only in the LittleMore engine and is numbered right after upstream llama.cpp's last type (44 today). If upstream
    ever claims that number, B1 must be renumbered, and a file written with the old number would be read as the other
    type without any error. The converter (tools/convert/lb_to_gguf.py) therefore writes two UINT32 keys:
    `littlemore.format_version` (1) and `littlemore.b1_type_id` (the B1 number used for the file's tensors). The
    engine's loader (llama_model_loader, engine/src/llama-model-loader.cpp) checks them before it reads any tensor.

What is checked here, through the C API (the Python SDK is a thin ctypes layer over `lm_model_load`)
    * the tiny test model carries both keys with the converter's values (so the "correct values" case below really
      has the marker);
    * `littlemore.b1_type_id` = 45 or 43 (a build whose B1 number differs from the file's): LM_ERR_MODEL_LOAD, and
      the message from `lm_last_error` names the key, both numbers and that the model must be reconverted;
    * `littlemore.format_version` = 2 (newer than this library): LM_ERR_MODEL_LOAD with a "newer" message;
    * a marker key stored with the wrong value type: LM_ERR_MODEL_LOAD;
    * the same file WITHOUT the two keys (as every file made before the marker): loads and generates exactly the same
      text as the file with the correct keys (the legacy path, which assumes the historical B1 number 44).

The broken files are loaded in a child process, as in test_malformed_littlebit.py: a regression that aborts then fails
the test instead of killing pytest. Needs the built library and the tiny model; skipped without them.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from test_sdk import PYDIR, ROOT, lib_path, tiny_gguf  # noqa: F401  (fixtures)

sys.path.insert(0, str(ROOT / "tests" / "smoke"))
import _assets as A  # noqa: E402

A._add_path(A.ENGINE / "gguf-py")  # the engine's gguf-py knows the B1 type; the pip `gguf` package does not
import gguf  # noqa: E402

import littlemore  # noqa: E402  (python/ is on pytest's pythonpath, see pytest.ini)

KEY_VERSION = "littlemore.format_version"
KEY_B1 = "littlemore.b1_type_id"
B1 = int(gguf.GGMLQuantizationType.B1)  # the number this engine build uses (gguf-py mirrors ggml.h; tests/convert checks it)
U32 = gguf.GGUFValueType.UINT32

# Load a file in a child process and print either LOADED or the error status and message. argv[3] selects the
# devices: "none" (CPU only, the plain engine loader) or "auto" (the library's default, which first runs its own
# GGUF parser for the GPU auto-fit when a GPU backend exists).
_LOAD = (
    "import sys; sys.path.insert(0, sys.argv[1])\n"
    "import littlemore\n"
    "kw = {'devices': 'none'} if sys.argv[3] == 'none' else {}\n"
    "try:\n"
    "    littlemore.Model(sys.argv[2], **kw)\n"
    "    print('LOADED')\n"
    "except littlemore.LittleMoreError as e:\n"
    "    print('ERR', e.status_name, e.message)\n"
)


def _try_load(path: Path, devices: str) -> subprocess.CompletedProcess:
    """Load `path` through the SDK in a child process and return what it printed."""
    return subprocess.run([sys.executable, "-c", _LOAD, str(PYDIR), str(path), devices], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=300)


def _expect_model_load_error(path: Path, devices: str) -> str:
    """The child must exit normally and report LM_ERR_MODEL_LOAD; returns the error message (lm_last_error)."""
    r = _try_load(path, devices)
    assert r.returncode == 0, f"the process died (exit code {r.returncode}): {r.stderr[-400:]}"
    assert r.stdout.startswith("ERR LM_ERR_MODEL_LOAD "), r.stdout + r.stderr[-400:]
    return r.stdout[len("ERR LM_ERR_MODEL_LOAD "):].strip()


def _fields(path: Path) -> dict:
    """Metadata of a GGUF file by key (gguf-py reader fields)."""
    return gguf.GGUFReader(str(path)).fields


def test_tiny_model_carries_the_marker(tiny_gguf: Path) -> None:
    """Precondition of the cases below: the converter wrote both keys, as UINT32, with this build's values."""
    f = _fields(tiny_gguf)
    assert f[KEY_VERSION].types == [U32] and f[KEY_VERSION].contents() == 1
    assert f[KEY_B1].types == [U32] and f[KEY_B1].contents() == B1


@pytest.mark.parametrize("devices", ["none", "auto"])
@pytest.mark.parametrize("bad", [B1 + 1, B1 - 1])
def test_other_b1_type_id_is_refused(tiny_gguf: Path, tmp_path: Path, bad: int, devices: str) -> None:
    """A file written for another B1 number (as after a renumbering) is never loaded; the message says why."""
    out = tmp_path / f"b1_{bad}.gguf"
    A.copy_gguf(tiny_gguf, out, extra_kv=[(KEY_B1, bad, U32)])
    msg = _expect_model_load_error(out, devices)
    assert msg.startswith("lm_model_load:"), msg
    assert KEY_B1 in msg and f"B1 = {bad}" in msg and f"GGML_TYPE_B1 = {B1}" in msg, msg
    assert "reconvert" in msg, msg


@pytest.mark.parametrize("devices", ["none", "auto"])
def test_newer_format_version_is_refused(tiny_gguf: Path, tmp_path: Path, devices: str) -> None:
    """format_version 2 is newer than this library (it understands 1): refused with a "newer" message."""
    out = tmp_path / "version2.gguf"
    A.copy_gguf(tiny_gguf, out, extra_kv=[(KEY_VERSION, 2, U32)])
    msg = _expect_model_load_error(out, devices)
    assert KEY_VERSION in msg and "version 2" in msg and "newer" in msg, msg


@pytest.mark.parametrize("key", [KEY_VERSION, KEY_B1])
def test_marker_with_wrong_value_type_is_refused(tiny_gguf: Path, tmp_path: Path, key: str) -> None:
    """A marker key stored as a string cannot be compared: refused (the loader's typed key reader), not ignored."""
    out = tmp_path / "string_marker.gguf"
    A.copy_gguf(tiny_gguf, out, extra_kv=[(key, "1", gguf.GGUFValueType.STRING)])
    msg = _expect_model_load_error(out, "none")
    assert key in msg and "wrong type" in msg, msg


def test_file_without_marker_loads_and_generates_identically(tiny_gguf: Path, tmp_path: Path) -> None:
    """Legacy files (no marker keys) load as before: same text as the file with the correct keys, greedy and sampled."""
    legacy = tmp_path / "legacy.gguf"
    A.copy_gguf(tiny_gguf, legacy, drop_kv={KEY_VERSION, KEY_B1})
    f = _fields(legacy)
    assert KEY_VERSION not in f and KEY_B1 not in f, "the legacy copy still has a marker key"
    texts = []
    for path in (tiny_gguf, legacy):
        with littlemore.Model(path, devices="none", n_ctx=512) as m:
            texts.append((m.generate("Hello", temperature=0, max_tokens=24),
                          m.generate("Hello", temperature=1.5, seed=7, max_tokens=24)))
    assert all(t for pair in texts for t in pair), texts
    assert texts[0] == texts[1], texts
