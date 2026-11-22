"""Gemma 3 sliding-window layer pattern: the converter writes it from `layer_types` or refuses.

llama.cpp's Gemma3Model converter writes no per-layer pattern, and the engine then assumes the real 5:1 layout.
`_write_gemma3_swa_pattern` fills the gap; the engine-side proof (logits match the reference for non-default
patterns) is in tests/engine/test_fused_graph_families.py.

The hazard these tests guard against: if the converter wrote nothing (or a wrong pattern), the engine would
silently fall back to 5 sliding-window layers then 1 full-attention layer, and a model with a different layout
would load and run but produce wrong logits with no error.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.convert import lb_to_gguf as lb

import gguf


class _Writer:
    """Stand-in for the GGUF writer: records what the converter asks it to write."""

    def __init__(self):
        self.calls = []

    def add_sliding_window_pattern(self, v):
        """Record the sliding-window pattern the converter asks for."""
        self.calls.append(v)


def _conv(hparams, n_layer=4, arch=gguf.MODEL_ARCH.GEMMA3):
    """Minimal fake of the llama.cpp model-converter object, with the given HF hyper-parameters."""
    return SimpleNamespace(model_arch=arch, hparams=hparams, block_count=n_layer, gguf_writer=_Writer())


def test_layer_types_become_a_bool_array():
    """HF `layer_types` (sliding/full per layer) must be written as a per-layer bool array (True = sliding)."""
    c = _conv({"layer_types": ["sliding_attention", "full_attention"] * 2})
    lb._write_gemma3_swa_pattern(c)
    assert c.gguf_writer.calls == [[True, False, True, False]]


def test_default_five_to_one_pattern_is_written_too():
    """Even the standard 5:1 layout is written explicitly, so the file never depends on the engine's fallback."""
    c = _conv({"layer_types": ["sliding_attention"] * 5 + ["full_attention"]}, n_layer=6)
    lb._write_gemma3_swa_pattern(c)
    assert c.gguf_writer.calls == [[True] * 5 + [False]]


@pytest.mark.parametrize("bad", [["sliding_attention", "chunked_attention", "full_attention", "full_attention"],
                                 ["sliding_attention", "full_attention"]])
def test_unknown_type_or_wrong_length_fails_loudly(bad):
    """An unknown layer type or a list of the wrong length must raise, not be guessed or truncated."""
    with pytest.raises(lb.ConversionError, match="layer_types"):
        lb._write_gemma3_swa_pattern(_conv({"layer_types": bad}))


def test_no_layer_types_leaves_engine_default_and_integer_pattern_is_forwarded():
    """Without `layer_types` nothing is written (the engine default applies); an integer `sliding_window_pattern`
    is forwarded as given unless it is 6 (the engine default; the converter also skips 1)."""
    c = _conv({})
    lb._write_gemma3_swa_pattern(c)
    assert c.gguf_writer.calls == []
    c = _conv({"sliding_window_pattern": 4})
    lb._write_gemma3_swa_pattern(c)
    assert c.gguf_writer.calls == [4]
    c = _conv({"sliding_window_pattern": 6})
    lb._write_gemma3_swa_pattern(c)
    assert c.gguf_writer.calls == []


def test_other_architectures_are_untouched():
    """Only Gemma 3 gets a pattern; any other architecture must not have the key written."""
    c = _conv({"layer_types": ["sliding_attention"] * 4}, arch=gguf.MODEL_ARCH.LLAMA)
    lb._write_gemma3_swa_pattern(c)
    assert c.gguf_writer.calls == []
