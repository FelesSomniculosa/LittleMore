"""Converter tests for the per-linear bookkeeping scalars that trained LittleBit checkpoints carry.

Checkpoints written by the LittleBit training code (for example the community checkpoint
`LittleBitLLM/littlebit-qwen3-8b-0.55bpw`) store three scalars next to each LittleBit linear's
signs and scales:

    <module>._eff_bit_target    bits per weight the layer aimed for (bf16, e.g. 0.5508 = 0.55 in bf16)
    <module>._eff_bit_actual    bits per weight it reached          (bf16, e.g. 0.5469)
    <module>._split_dim_final   the rank the layer ended with        (int64, e.g. 544)

They are not weights and the engine has no use for them, but llama.cpp's converter does not know
them and stops with "Can not map tensor".  The converter therefore drops exactly these three names
on LittleBit linears, after checking `_split_dim_final` against the rank of the sign matrices
(a mismatch means the checkpoint is not what it claims to be, so it is refused).  The same names
on anything that is NOT a LittleBit linear are not dropped (they still fail), so nothing unknown
is skipped silently.

The tiny checkpoints are made offline by `_tiny.make_tiny_checkpoint` (needs the cached Qwen3-0.6B
tokenizer; skipped without it).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import _tiny
from tools.convert import lb_to_gguf as lb

import gguf  # noqa: E402

needs_qwen_tok = pytest.mark.skipif(not _tiny.qwen_tokenizer_available(), reason="Qwen3-0.6B tokenizer not cached")


@pytest.fixture(scope="module")
def q3(tmp_path_factory):
    """A tiny Qwen3 LittleBit-2 checkpoint (residual path on), made once for this module."""
    if not _tiny.qwen_tokenizer_available():
        pytest.skip("Qwen3-0.6B tokenizer not cached")
    return _tiny.make_tiny_checkpoint("qwen3", tmp_path_factory.mktemp("q3bk"))


def _add_bookkeeping(src: Path, dst: Path, rank_offset: dict[str, int] | None = None,
                     extra: dict[str, torch.Tensor] | None = None) -> list[str]:
    """Copy checkpoint `src` to `dst` and add the three bookkeeping scalars to every LittleBit linear.

    `rank_offset` maps a module name to a number added to its `_split_dim_final` (to fake a wrong rank);
    `extra` adds arbitrary further tensors.  Returns the module names that got bookkeeping scalars.
    """
    shutil.copytree(src, dst)
    f = dst / "model.safetensors"
    t = load_file(str(f))
    mods = sorted(k[: -len(".U_packed")] for k in t if k.endswith(".U_packed"))
    for m in mods:
        rank = int(t[f"{m}.U_shape"][1])
        t[f"{m}._eff_bit_target"] = torch.tensor(0.55, dtype=torch.bfloat16)
        t[f"{m}._eff_bit_actual"] = torch.tensor(0.546875, dtype=torch.bfloat16)
        t[f"{m}._split_dim_final"] = torch.tensor(rank + (rank_offset or {}).get(m, 0), dtype=torch.int64)
    t.update(extra or {})
    save_file(t, str(f), metadata={"format": "pt"})
    return mods


@needs_qwen_tok
def test_bookkeeping_scalars_are_dropped_after_the_rank_check(q3, tmp_path):
    """With the scalars present the conversion succeeds, validates every linear, and writes none of them."""
    mods = _add_bookkeeping(q3, tmp_path / "ck")
    report = lb.convert_checkpoint(tmp_path / "ck", tmp_path / "o.gguf")
    assert report.n_validated == report.n_linears == len(mods)
    assert len(report.bookkeeping) == 3 * len(mods)
    assert all(k.rsplit(".", 1)[0] in mods for k in report.bookkeeping)
    names = [t.name for t in gguf.GGUFReader(str(tmp_path / "o.gguf")).tensors]
    assert not [n for n in names if "eff_bit" in n or "split_dim" in n]


@needs_qwen_tok
def test_wrong_split_dim_final_is_refused(q3, tmp_path):
    """A `_split_dim_final` that differs from the rank of the sign matrices is a ConversionError naming the layer."""
    victim = "model.layers.1.mlp.up_proj"
    _add_bookkeeping(q3, tmp_path / "ck", rank_offset={victim: 8})
    rank = int(load_file(str(q3 / "model.safetensors"))[f"{victim}.U_shape"][1])
    with pytest.raises(lb.ConversionError, match=rf"{victim}\._split_dim_final is {rank + 8} but .* rank {rank}"):
        lb.convert(tmp_path / "ck", tmp_path / "o.gguf")
    assert not (tmp_path / "o.gguf").exists()


@needs_qwen_tok
def test_bookkeeping_name_on_a_plain_tensor_still_fails(q3, tmp_path):
    """The names are only dropped on LittleBit linears: on the final norm (a plain tensor) the conversion fails."""
    _add_bookkeeping(q3, tmp_path / "ck",
                     extra={"model.norm._eff_bit_actual": torch.tensor(0.5, dtype=torch.bfloat16)})
    with pytest.raises(lb.ConversionError, match=r"model\.norm\._eff_bit_actual"):
        lb.convert(tmp_path / "ck", tmp_path / "o.gguf")
    assert not (tmp_path / "o.gguf").exists()
