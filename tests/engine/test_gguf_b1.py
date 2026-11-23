"""GGML_TYPE_B1 support in the engine's gguf-py (pure 1-bit signs, bit 1 = +1, LSB first).

Skipped when engine/gguf-py is missing (a checkout without the engine submodule: `git submodule update --init`).
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
    """One 128-element block of -1.0 with +1.0 at the indices in `pattern`."""
    x = -np.ones(128, dtype=np.float32)
    for i in pattern:
        x[i] = 1.0
    return x


def test_enum_and_sizes():
    """B1 is type number 44 and a block is 128 signs in 16 bytes; the engine C code and the file format both
    rely on these numbers, so changing either breaks every B1 file."""
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
    """Bit layout of the quantizer: a +1 at element i sets bit (i % 8) of byte (i // 8) (least significant bit first),
    and -1 is a 0 bit.  If the polarity or bit order were swapped, the engine would read every sign wrong."""
    x = _signs(list(plus))
    q = quantize(x.reshape(1, 128), B1)
    assert q.shape == (1, 16) and q.dtype == np.uint8
    assert q.tolist() == [expected]
    assert np.array_equal(dequantize(q, B1), x.reshape(1, 128))


def test_zero_is_plus_one():
    """A value of exactly 0.0 maps to +1 (the `>= 0` rule), so all-zero input gives all-ones bytes."""
    q = quantize(np.zeros((1, 128), dtype=np.float32), B1)
    assert q.tolist() == [[0xFF] * 16]


def test_roundtrip_random():
    """Random +-1 data survives quantize then dequantize unchanged, for several rows of several blocks."""
    rng = np.random.default_rng(0)
    x = np.where(rng.integers(0, 2, size=(5, 384)) == 1, 1.0, -1.0).astype(np.float32)
    q = quantize(x, B1)
    assert q.shape == (5, 48)
    assert np.array_equal(dequantize(q, B1), x)


def test_gguf_write_read_roundtrip():
    """A B1 tensor written to a GGUF file and read back keeps its type, shape ([ne0, ne1] order) and bytes."""
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
