"""A malformed LittleBit GGUF must make `Model(...)` fail with LM_ERR_MODEL_LOAD, never abort the process.

The engine's loader used to hand the shape and type of a LittleBit linear to `ggml_littlebit_packed`, whose GGML_ASSERTs
abort the whole process (or, for a missing tensor, dereference a null pointer). The loader now checks everything first
and throws, which the C API reports as LM_ERR_MODEL_LOAD. Each case derives a broken file from the tiny test model
(tests/c/make_tiny_gguf.py) with `copy_gguf`, and loads it in a SUBPROCESS: a regression (an abort) then fails the test
instead of killing pytest.

Needs the built library and the tiny model; skipped without them (same fixtures as test_sdk.py).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from test_sdk import PYDIR, ROOT, lib_path, tiny_gguf  # noqa: F401  (fixtures)

sys.path.insert(0, str(ROOT / "tests" / "smoke"))
import _assets as A  # noqa: E402

A._add_path(A.ENGINE / "gguf-py")  # the engine's gguf-py knows the B1 type; the pip `gguf` package does not

# The tensors of one LittleBit linear (layer 0, attention query) that the loader reads.
LIN = "blk.0.attn_q."

_LOAD = (
    "import sys; sys.path.insert(0, sys.argv[1])\n"
    "import littlemore\n"
    "try:\n"
    "    littlemore.Model(sys.argv[2], devices='none')\n"
    "    print('LOADED')\n"
    "except littlemore.LittleMoreError as e:\n"
    "    print('ERR', e.status_name, e)\n"
)


def _try_load(path: Path) -> subprocess.CompletedProcess:
    """Load `path` through the SDK in a child process and return what it printed."""
    return subprocess.run([sys.executable, "-c", _LOAD, str(PYDIR), str(path)], capture_output=True, text=True, timeout=300)


def _expect_model_load_error(path: Path) -> None:
    """The child must exit normally and report LM_ERR_MODEL_LOAD (a crash or abort has a non-zero exit code)."""
    r = _try_load(path)
    assert r.returncode == 0, f"the process died (exit code {r.returncode}): {r.stderr[-400:]}"
    assert "ERR LM_ERR_MODEL_LOAD" in r.stdout, r.stdout + r.stderr[-400:]


def _f16(t):
    """Same tensor with its values stored as F16."""
    import gguf
    return np.asarray(t.data).astype(np.float16), gguf.GGMLQuantizationType.F16


def _case(tiny: Path, tmp: Path, name: str, replace=None, drop=None) -> Path:
    """Write a broken copy of the tiny model."""
    out = tmp / f"{name}.gguf"
    A.copy_gguf(tiny, out, replace=replace, drop=drop)
    return out


def test_unmodified_copy_loads(tiny_gguf: Path, tmp_path: Path) -> None:
    """Control: the copy machinery itself does not break the model (so the failures below come from the damage)."""
    r = _try_load(_case(tiny_gguf, tmp_path, "control"))
    assert r.returncode == 0 and "LOADED" in r.stdout, r.stdout + r.stderr[-400:]


def test_missing_u_factor(tiny_gguf: Path, tmp_path: Path) -> None:
    """lb_v present, lb_u absent."""
    _expect_model_load_error(_case(tiny_gguf, tmp_path, "no_u", drop=lambda n: n == LIN + "lb_u"))


@pytest.mark.parametrize("scale", ["lb_s_v2", "lb_s_mid", "lb_s_u1"])
def test_scale_not_f32(tiny_gguf: Path, tmp_path: Path, scale: str) -> None:
    """A scale vector stored as F16 instead of F32."""
    bad = _case(tiny_gguf, tmp_path, "f16_" + scale, replace=lambda t: _f16(t) if t.name == LIN + scale else None)
    _expect_model_load_error(bad)


def test_residual_pack_not_f32(tiny_gguf: Path, tmp_path: Path) -> None:
    """The packed residual scales stored as F16 (the tiny model has a residual path)."""
    _expect_model_load_error(_case(tiny_gguf, tmp_path, "f16_pkr", replace=lambda t: _f16(t) if t.name == LIN + "lb_s_pkr" else None))


def test_rank_not_multiple_of_128(tiny_gguf: Path, tmp_path: Path) -> None:
    """lb_v with 130 rows: the rank is not a whole number of 128-sign blocks."""
    import gguf

    def widen(t):
        if t.name != LIN + "lb_v":
            return None
        data = np.asarray(t.data)
        pad = np.zeros((2, data.shape[1]), dtype=data.dtype)
        return np.concatenate([data, pad], axis=0), gguf.GGMLQuantizationType.B1
    _expect_model_load_error(_case(tiny_gguf, tmp_path, "rank130", replace=widen))


def test_v_row_length_differs_from_input_width(tiny_gguf: Path, tmp_path: Path) -> None:
    """lb_v whose rows are longer than the layer's input width."""
    import gguf

    def widen(t):
        if t.name != LIN + "lb_v":
            return None
        data = np.asarray(t.data)
        return np.concatenate([data, np.zeros((data.shape[0], 16), dtype=data.dtype)], axis=1), gguf.GGMLQuantizationType.B1
    _expect_model_load_error(_case(tiny_gguf, tmp_path, "wide_v", replace=widen))


# --------------------------------------------------------------------------------------------------------------
# Auto-fit (plan_fit) reads sizes and layer numbers from the file: they must not be trusted
# --------------------------------------------------------------------------------------------------------------
_FIT = (
    "import sys; sys.path.insert(0, sys.argv[1])\n"
    "import littlemore\n"
    "names = [b['name'] for b in littlemore.backends_info()]\n"
    "if not any(n != 'CPU' for n in names):\n"
    "    print('NO-GPU'); sys.exit(0)\n"
    "try:\n"
    "    littlemore.Model(sys.argv[2])\n"      # default n_gpu_layers='auto' -> plan_fit runs
    "    print('LOADED')\n"
    "except littlemore.LittleMoreError as e:\n"
    "    print('ERR', e.status_name)\n"
)


def _fit_load(path: Path) -> subprocess.CompletedProcess:
    """Load with the default (auto-fit) settings in a child process; the auto-fit only runs when a GPU backend exists."""
    return subprocess.run([sys.executable, "-c", _FIT, str(PYDIR), str(path)], capture_output=True, text=True, timeout=300)


def _expect_clean_fit_failure(path: Path) -> None:
    """No crash, no huge allocation: the load either fails with an error status or succeeds; skip without a GPU."""
    r = _fit_load(path)
    if "NO-GPU" in r.stdout:
        pytest.skip("auto-fit only runs when a GPU backend is present")
    assert r.returncode == 0, f"the process died (exit code {r.returncode}): {r.stderr[-400:]}"
    assert "ERR LM_ERR_MODEL_LOAD" in r.stdout or "LOADED" in r.stdout, r.stdout + r.stderr[-400:]


def test_autofit_ignores_an_absurd_layer_index(tiny_gguf: Path, tmp_path: Path) -> None:
    """A tensor named blk.2000000000.* must not make the auto-fit allocate a per-layer table of 16 GB."""
    import gguf
    out = tmp_path / "far_layer.gguf"
    A.copy_gguf(tiny_gguf, out, extra=[("blk.2000000000.attn_q.weight", np.zeros(8, dtype=np.float32), gguf.GGMLQuantizationType.F32)])
    _expect_clean_fit_failure(out)


def test_autofit_survives_sizes_that_overflow_64_bits(tiny_gguf: Path, tmp_path: Path) -> None:
    """Head counts and key/value lengths near 2^32 make the KV-size products exceed 64 bits: no crash."""
    import gguf
    big = 2**32 - 1
    T = gguf.GGUFValueType.UINT32
    out = tmp_path / "huge_kv.gguf"
    A.copy_gguf(tiny_gguf, out, extra_kv=[("llama.attention.head_count_kv", big, T), ("llama.attention.key_length", big, T),
                                           ("llama.attention.value_length", big, T)])
    _expect_clean_fit_failure(out)
