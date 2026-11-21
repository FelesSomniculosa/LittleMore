"""GGML_TYPE_B1 support in the engine's gguf-py (pure 1-bit signs, bit 1 = +1, LSB first).

Skipped when engine/gguf-py is missing (engine is a separate, git-ignored clone).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

GGUF_PY = Path(__file__).resolve().parents[2] / "engine" / "gguf-py"
if not (GGUF_PY / "gguf").is_dir():
    pytest.skip("engine/gguf-py not present", allow_module_level=True)
sys.path.insert(0, str(GGUF_PY))

import gguf  # noqa: E402
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402

B1 = GGMLQuantizationType.B1


def _signs(pattern: dict) -> np.ndarray:
    x = -np.ones(128, dtype=np.float32)
    for i in pattern:
        x[i] = 1.0
    return x


def test_enum_and_sizes():
    assert int(B1) == 44
    assert gguf.GGML_QUANT_SIZES[B1] == (128, 16)


@pytest.mark.parametrize(
    "plus,expected",
    [
        ([0], [0x01] + [0] * 15),
        ([7], [0x80] + [0] * 15),
        ([8], [0x00, 0x01] + [0] * 14),
        (range(128), [0xFF] * 16),
        ([], [0x00] * 16),
    ],
)
def test_known_patterns(plus, expected):
    x = _signs(list(plus))
    q = quantize(x.reshape(1, 128), B1)
    assert q.shape == (1, 16) and q.dtype == np.uint8
    assert q.tolist() == [expected]
    assert np.array_equal(dequantize(q, B1), x.reshape(1, 128))


def test_zero_is_plus_one():
    q = quantize(np.zeros((1, 128), dtype=np.float32), B1)
    assert q.tolist() == [[0xFF] * 16]


def test_roundtrip_random():
    rng = np.random.default_rng(0)
    x = np.where(rng.integers(0, 2, size=(5, 384)) == 1, 1.0, -1.0).astype(np.float32)
    q = quantize(x, B1)
    assert q.shape == (5, 48)
    assert np.array_equal(dequantize(q, B1), x)


def test_gguf_write_read_roundtrip():
    rng = np.random.default_rng(1)
    x = np.where(rng.integers(0, 2, size=(4, 256)) == 1, 1.0, -1.0).astype(np.float32)
    q = quantize(x, B1)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "b1.gguf"
        w = GGUFWriter(str(path), "llama")
        w.add_tensor("w.b1", q, raw_dtype=B1)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

        r = GGUFReader(str(path))
        t = r.tensors[0]
        assert t.tensor_type == B1
        assert tuple(t.shape) == (256, 4)
        data = np.array(t.data).reshape(4, 32)  # copy: the reader memory-maps the file
        del t, r  # release the mapping so Windows can delete the temp file
        assert np.array_equal(data, q)
        assert np.array_equal(dequantize(data, B1), x)
