"""Gemma 3 sliding-window layer pattern: the converter writes it from `layer_types` or refuses.

llama.cpp's Gemma3Model converter writes no per-layer pattern, and the engine then assumes the real 5:1 layout.
`_write_gemma3_swa_pattern` fills the gap; the engine-side proof (logits match the reference for non-default
patterns) is in tests/engine/test_fused_graph_families.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.convert import lb_to_gguf as lb

import gguf


class _Writer:
    def __init__(self):
        self.calls = []

    def add_sliding_window_pattern(self, v):
        self.calls.append(v)


def _conv(hparams, n_layer=4, arch=gguf.MODEL_ARCH.GEMMA3):
    return SimpleNamespace(model_arch=arch, hparams=hparams, block_count=n_layer, gguf_writer=_Writer())


def test_layer_types_become_a_bool_array():
    c = _conv({"layer_types": ["sliding_attention", "full_attention"] * 2})
    lb._write_gemma3_swa_pattern(c)
    assert c.gguf_writer.calls == [[True, False, True, False]]


def test_default_five_to_one_pattern_is_written_too():
    c = _conv({"layer_types": ["sliding_attention"] * 5 + ["full_attention"]}, n_layer=6)
    lb._write_gemma3_swa_pattern(c)
    assert c.gguf_writer.calls == [[True] * 5 + [False]]


@pytest.mark.parametrize("bad", [["sliding_attention", "chunked_attention", "full_attention", "full_attention"],
                                 ["sliding_attention", "full_attention"]])
def test_unknown_type_or_wrong_length_fails_loudly(bad):
    with pytest.raises(lb.ConversionError, match="layer_types"):
        lb._write_gemma3_swa_pattern(_conv({"layer_types": bad}))


def test_no_layer_types_leaves_engine_default_and_integer_pattern_is_forwarded():
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
    c = _conv({"layer_types": ["sliding_attention"] * 4}, arch=gguf.MODEL_ARCH.LLAMA)
    lb._write_gemma3_swa_pattern(c)
    assert c.gguf_writer.calls == []
