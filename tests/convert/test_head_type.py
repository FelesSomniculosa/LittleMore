"""Tests for the converter's `--head-type f16|q8_0|q6_k` (precision of the token embedding / output head).

What is tested
    * The two quantizers the option uses (gguf-py's Q8_0 and the LittleMore port of ggml's Q6_K reference quantizer in
      engine/gguf-py/gguf/quants.py) produce exactly the bytes of ggml's own C quantizer (`ggml_quantize_chunk` in
      ggml-base, called through ctypes), including all-zero blocks and blocks whose 16-value sub-blocks differ so much
      in magnitude that a sub-block scale rounds to 0 (a separate branch of the Q6_K code).
    * A tiny Qwen3 / Llama checkpoint converted with each head type: the embedding (and an untied output head) has the
      requested type, every other tensor is byte-identical to the default (f16) conversion, and the dequantized head is
      within the expected quantization error of the F16 head.
    * A head whose row length the type cannot take (Q6_K needs multiples of 256) is refused, not silently written as F16.

Why
    The head is the largest single tensor of a small model (311 MB of 636 MB for Qwen3-0.6B at F16) and is read once per
    generated token, so quantizing it is the main speed lever for small models (docs/engine-changes.md, decode profile).
    Correctness of everything else is measured with the F16 head; this option must not change anything else.

The ggml comparison needs the engine build (engine/build/bin/ggml-base.dll or libggml-base.so); it is skipped without
it.  The Qwen3 case needs the cached Qwen3-0.6B tokenizer (see _tiny.py); Llama needs nothing.
"""
from __future__ import annotations

import ctypes
import os
from math import prod
from pathlib import Path

import numpy as np
import pytest

import _tiny
from tools.convert import lb_to_gguf as lb  # also puts engine/gguf-py on sys.path

import gguf  # noqa: E402  (the engine's gguf-py)

Q8_0 = gguf.GGMLQuantizationType.Q8_0
Q6_K = gguf.GGMLQuantizationType.Q6_K
F16 = gguf.GGMLQuantizationType.F16

_BIN = _tiny.ROOT / "engine" / "build" / "bin"
_LIBGGML = _BIN / ("ggml-base.dll" if os.name == "nt" else "libggml-base.so")


def _ggml_quantize(data: np.ndarray, qtype: gguf.GGMLQuantizationType) -> np.ndarray:
    """ggml's C quantizer (ggml_quantize_chunk without an importance matrix) on a float32 [rows][n] array."""
    lib = ctypes.CDLL(str(_LIBGGML))
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_float))
    data = np.ascontiguousarray(data, dtype=np.float32)
    out = np.zeros(gguf.quant_shape_to_byte_shape(data.shape, qtype), dtype=np.uint8)
    n = lib.ggml_quantize_chunk(qtype.value, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                out.ctypes.data_as(ctypes.c_void_p), 0, prod(data.shape[:-1]), data.shape[-1],
                                ctypes.cast(0, ctypes.POINTER(ctypes.c_float)))
    assert n == out.size
    return out


def _test_data() -> np.ndarray:
    """Rows of 1024 values: random normal and embedding-like rows, an all-zero row, and mixed-magnitude rows.

    Mixed magnitude: within each 256-value Q6_K block, the first 16-value sub-block is 1e4 times larger than the rest,
    so the int8 sub-block scales of the small sub-blocks round to 0 and their quants keep the values of the per-sub-block
    search (the `d == 0` branch of quantize_row_q6_K_ref).
    """
    rng = np.random.default_rng(7)
    rows = [rng.standard_normal((48, 1024)), 0.02 * rng.standard_normal((16, 1024)),
            rng.standard_t(3, (8, 1024)), np.zeros((1, 1024))]
    mixed = rng.standard_normal((8, 1024))
    for b in range(0, 1024, 256):
        mixed[:, b:b + 16] *= 1e4   # the first sub-block of each block is large
    rows.append(mixed)
    return np.concatenate(rows).astype(np.float32)


@pytest.mark.skipif(not _LIBGGML.exists(), reason=f"engine build not found ({_LIBGGML})")
@pytest.mark.parametrize("qtype", [Q8_0, Q6_K], ids=["q8_0", "q6_k"])
def test_quantizer_bit_identical_to_ggml(qtype):
    """gguf-py's quantizer output equals ggml's C reference quantizer byte for byte."""
    data = _test_data()
    py = gguf.quants.quantize(data, qtype)
    c = _ggml_quantize(data, qtype)
    assert py.shape == c.shape
    bad = np.flatnonzero((py.reshape(-1, gguf.GGML_QUANT_SIZES[qtype][1]) != c.reshape(-1, gguf.GGML_QUANT_SIZES[qtype][1])).any(axis=1))
    assert bad.size == 0, f"{bad.size} of {py.size // gguf.GGML_QUANT_SIZES[qtype][1]} blocks differ (first {bad[:5]})"
    if qtype == Q6_K:
        # the mixed-magnitude rows must really reach the "sub-block scale rounds to 0" branch
        blocks = py.reshape(-1, 210)
        scales = blocks[:, 192:208].view(np.int8)
        d = blocks[:, 208:210].view(np.float16).reshape(-1)
        assert np.any((scales == 0).any(axis=1) & (d != 0)), "no block with a zero sub-block scale was produced"


def _convert(ck: Path, out: Path, head_type: str) -> dict:
    lb.convert_checkpoint(ck, out, head_type=head_type)
    return {t.name: t for t in gguf.GGUFReader(str(out)).tensors}


def _deq(t) -> np.ndarray:
    return gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).reshape(-1)


# Relative RMS error of the quantized head against the F16 head.  Expected: rounding to a uniform grid of step
# max|block| / levels has an rms error of step / sqrt(12); the tiny models' weights are normal (HF init, sigma 0.02), so
# max|x| is about 2.4 sigma over a Q8_0 block of 32 and 2.1 sigma over a Q6_K sub-block of 16, giving
# 2.4 / 127 / 3.46 = 5.5e-3 (Q8_0) and 2.1 / 32 / 3.46 = 1.9e-2 (Q6_K).  Measured on these models: 5.35e-3 .. 5.38e-3
# (Q8_0) and 1.78e-2 .. 1.79e-2 (Q6_K).  The bounds are 2x the expectation.
_REL_RMS_BOUND = lb.HEAD_REL_RMS_MAX   # {"q8_0": 1.1e-2, "q6_k": 3.8e-2}, the converter's own content check


@pytest.mark.parametrize("family,untied", [("llama", False), ("llama", True), ("qwen3", False)])
@pytest.mark.parametrize("head_type", ["q8_0", "q6_k"])
def test_head_type_converts_only_the_head(tmp_path, family, untied, head_type):
    """Only token_embd (and output, when untied) change type; all other tensors are byte-identical to the f16 file."""
    if family == "qwen3" and not _tiny.qwen_tokenizer_available():
        pytest.skip("Qwen3-0.6B tokenizer not cached")
    ck = _tiny.make_tiny_checkpoint(family, tmp_path, tie_word_embeddings=not untied)
    ref = _convert(ck, tmp_path / "f16.gguf", "f16")
    got = _convert(ck, tmp_path / f"{head_type}.gguf", head_type)
    want_type = {"q8_0": Q8_0, "q6_k": Q6_K}[head_type]
    heads = {"token_embd.weight"} | ({"output.weight"} if untied else set())
    assert heads <= set(ref) and set(got) == set(ref)
    assert ("output.weight" in ref) == untied
    for name, t in ref.items():
        g = got[name]
        if name in heads:
            assert t.tensor_type == F16 and g.tensor_type == want_type, name
            a, b = _deq(t).astype(np.float64), _deq(g).astype(np.float64)
            rel = np.sqrt(np.mean((a - b) ** 2) / np.mean(a ** 2))
            assert rel <= _REL_RMS_BOUND[head_type], f"{name}: relative rms error {rel:.3e}"
        else:
            assert g.tensor_type == t.tensor_type, name
            assert np.array_equal(np.asarray(g.data).view(np.uint8), np.asarray(t.data).view(np.uint8)), name


@pytest.mark.parametrize("head_type", ["q8_0", "q6_k"])
def test_converter_checks_head_content(tmp_path, monkeypatch, head_type):
    """The converter dequantizes the written head and compares it with the checkpoint (lb.HEAD_REL_RMS_MAX); a head
    whose quantized blocks are corrupted (here: one block in 64 of the embedding zeroed by the quantizer) is refused with
    a ConversionError and no output file, while the intact conversion reports its error."""
    ck = _tiny.make_tiny_checkpoint("llama", tmp_path)
    ok = tmp_path / "ok.gguf"
    report = lb.convert_checkpoint(ck, ok, head_type=head_type)
    assert 0 < report.head_rel_rms["token_embd.weight"] <= _REL_RMS_BOUND[head_type]

    qtype = {"q8_0": Q8_0, "q6_k": Q6_K}[head_type]
    cls = gguf.quants._type_traits[qtype]
    real = cls.quantize_blocks.__func__

    def corrupt(c, blocks):
        out = real(c, blocks)
        out[::64] = 0          # every 64th block: scale 0 and quants 0, i.e. those 32 / 256 values become 0
        return out

    monkeypatch.setattr(cls, "quantize_blocks", classmethod(corrupt))
    bad = tmp_path / "bad.gguf"
    with pytest.raises(lb.ConversionError, match="not the checkpoint's head"):
        lb.convert_checkpoint(ck, bad, head_type=head_type)
    assert not bad.exists()


def test_head_type_row_length_refused(tmp_path):
    """Q6_K blocks are 256 values: a hidden size of 128 cannot be stored and must fail (no silent F16 fallback)."""
    ck = _tiny.make_tiny_checkpoint("llama", tmp_path, hidden_size=128, intermediate_size=256)
    out = tmp_path / "q6k.gguf"
    with pytest.raises(lb.ConversionError, match="token_embd.weight"):
        lb.convert_checkpoint(ck, out, head_type="q6_k")
    assert not out.exists()


def test_head_type_unknown_refused():
    """An unknown head type is a ConversionError naming the option."""
    with pytest.raises(lb.ConversionError, match="head_type"):
        lb.convert(Path("nonexistent"), Path("x.gguf"), head_type="bogus")
