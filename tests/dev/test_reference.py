"""Tests for the lean PyTorch LittleBit reference (tools/dev/reference.py).

Run with `pytest tests/dev`.  Terms are defined in docs/development.md.
"""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import reference as ref


def _signs(rows, cols, gen):
    """Random +/-1 float matrix (rows, cols) drawn from the seeded generator `gen`."""
    return torch.where(torch.rand(rows, cols, generator=gen) < 0.5, -1.0, 1.0)


def _params(out_f, in_f, r, gen, residual=True):
    """Random tensors for one LittleBit layer: returns (primary, residual-or-None) dicts.

    Each dict holds sign matrices U (out_f, r), V (r, in_f) and positive scale
    vectors u1, u2, v1, v2 (kept in 0.5..1.5 so no scale is near zero).
    """
    def path():
        return {
            "U": _signs(out_f, r, gen), "V": _signs(r, in_f, gen),
            "u1": torch.rand(1, out_f, generator=gen) + 0.5,
            "u2": torch.rand(1, r, generator=gen) + 0.5,
            "v1": torch.rand(1, r, generator=gen) + 0.5,
            "v2": torch.rand(1, in_f, generator=gen) + 0.5,
        }
    return path(), (path() if residual else None)


def _dense(p):
    """Independent dense weight of one path: W = diag(u1) U diag(v1*u2) V diag(v2).

    Written separately from LBLinear.forward so the two can disagree if either is wrong.
    """
    return (p["u1"].T * p["U"]) @ ((p["v1"] * p["u2"]).T * p["V"]) * p["v2"]


def _make(p, res):
    """Build an LBLinear from the dicts returned by `_params` (`res` may be None)."""
    return ref.LBLinear(p["U"], p["V"], p["u1"], p["u2"], p["v1"], p["v2"],
                        residual=None if res is None else
                        (res["U"], res["V"], res["u1"], res["u2"], res["v1"], res["v2"]))


def test_unpack_matches_spec():
    """Pins the packed-sign bit order: LSB-first, bit 0 -> +1, bit 1 -> -1.

    Fails if the order or polarity changes, which would silently break every
    consumer (maker, converter, engine) that must agree on the format.
    """
    # bit0=0 -> +1, bit1=1 -> -1, LSB-first; bits 2.. are 0 -> +1
    word = torch.tensor([[0b10]], dtype=torch.int32)
    out = ref.unpack_signs(word, (1, 4))
    assert out.tolist() == [[1.0, -1.0, 1.0, 1.0]]
    # bit 31 (sign bit of int32) = column 31 -> -1
    word = torch.tensor([[-2**31]], dtype=torch.int32)
    out = ref.unpack_signs(word, (1, 32))
    assert out[0, 31] == -1.0 and (out[0, :31] == 1.0).all()


def test_pack_unpack_roundtrip():
    """pack_signs then unpack_signs returns the input, for widths around the 32-bit word size.

    Widths 31/32/33 and 70 cover just under, exactly at and just over a word
    boundary.  Fails on an off-by-one in padding or word count.
    """
    g = torch.Generator().manual_seed(0)
    for cols in (1, 31, 32, 33, 70):
        s = _signs(5, cols, g)
        packed = ref.pack_signs(s)
        assert packed.dtype == torch.int32 and packed.shape == (5, (cols + 31) // 32)
        assert torch.equal(ref.unpack_signs(packed, (5, cols)), s)


def test_forward_matches_dense():
    """The factor chain equals x @ W.T with W built densely, for primary + residual.

    Fails if the order of the scales or sign matmuls in forward() is wrong.
    """
    g = torch.Generator().manual_seed(1)
    p, r = _params(24, 40, 8, g)
    layer = _make(p, r)
    x = torch.randn(3, 40, generator=g)
    want = x @ (_dense(p) + _dense(r)).T
    torch.testing.assert_close(layer(x), want, rtol=1e-5, atol=1e-5)


def test_residual_optional():
    """A layer without a residual path works and equals its single dense path."""
    g = torch.Generator().manual_seed(2)
    p, _ = _params(16, 20, 4, g, residual=False)
    layer = _make(p, None)
    x = torch.randn(2, 20, generator=g)
    torch.testing.assert_close(layer(x), x @ _dense(p).T, rtol=1e-5, atol=1e-5)


def test_padded_rank_exact():
    """Padding bits in the last packed word are ignored, even when set to 1.

    Rank 33 needs 2 words, leaving 31 padding bits.  The test sets them to 1 on
    purpose (a converter or engine may write anything there); fails if
    unpack_signs lets them leak into the result or the layer output.
    """
    g = torch.Generator().manual_seed(3)
    p, r = _params(8, 70, 33, g)
    for q in (p, r):
        for m, cols in (("U", 33), ("V", 70)):
            original = q[m]
            packed = ref.pack_signs(original)
            pad = ~((1 << (cols % 32)) - 1) & 0xFFFFFFFF  # bits beyond `cols`
            last = (packed[:, -1].to(torch.int64) & 0xFFFFFFFF) | pad
            packed[:, -1] = torch.where(last >= 2**31, last - 2**32, last).to(torch.int32)
            # Prove the padding really is dirty (all ones) before unpacking.
            assert ((packed[:, -1].to(torch.int64) & 0xFFFFFFFF & pad) == pad).all()
            unpacked = ref.unpack_signs(packed, original.shape)
            assert torch.equal(unpacked, original)
    layer = _make(p, r)
    x = torch.randn(2, 70, generator=g)
    torch.testing.assert_close(layer(x), x @ (_dense(p) + _dense(r)).T, rtol=1e-5, atol=1e-5)


def test_unpack_chunked_matches(monkeypatch):
    """Chunked unpacking gives the same result as one pass, and honours `dtype`.

    _CHUNK_ELEMS = 256 is far below this matrix's 50 rows x 4 words x 32 bits,
    so the row loop runs many times.  Fails on a chunk-boundary bug.
    """
    g = torch.Generator().manual_seed(9)
    s = _signs(50, 100, g)
    monkeypatch.setattr(ref, "_CHUNK_ELEMS", 256)  # force many row chunks
    assert torch.equal(ref.unpack_signs(ref.pack_signs(s), (50, 100)), s)
    assert ref.unpack_signs(ref.pack_signs(s), (50, 100), dtype=torch.int8).dtype == torch.int8


def test_lblinear_sign_storage_is_one_byte():
    """Sign matrices are stored as int8 (1 byte/element); fails if they become fp32 (4x memory)."""
    g = torch.Generator().manual_seed(10)
    p, r = _params(16, 40, 8, g)
    layer = _make(p, r)
    for name in ("U_0", "V_0", "U_1", "V_1"):
        assert getattr(layer, name).element_size() == 1


def _tensors(prefix, out_f, in_f, r, g, suffix=""):
    """Checkpoint tensors (safetensors key -> tensor) for one path of module `prefix`.

    `suffix` is "" for the primary path or "_R" for the residual path.
    """
    U, V = _signs(out_f, r, g), _signs(r, in_f, g)
    return {
        f"{prefix}.U{suffix}_packed": ref.pack_signs(U),
        f"{prefix}.U{suffix}_shape": torch.tensor([out_f, r], dtype=torch.int32),
        f"{prefix}.V{suffix}_packed": ref.pack_signs(V),
        f"{prefix}.V{suffix}_shape": torch.tensor([r, in_f], dtype=torch.int32),
        f"{prefix}.u1{suffix}": torch.rand(1, out_f, generator=g),
        f"{prefix}.u2{suffix}": torch.rand(1, r, generator=g),
        f"{prefix}.v1{suffix}": torch.rand(1, r, generator=g),
        f"{prefix}.v2{suffix}": torch.rand(1, in_f, generator=g),
    }


def _write(d: Path, shards):
    """Write each dict in `shards` as one safetensors file in `d`, plus a config file."""
    d.mkdir(exist_ok=True)
    for i, t in enumerate(shards):
        save_file(t, str(d / f"model-{i}.safetensors"))
    (d / "littlebit_config.json").write_text(json.dumps({"rank": 4}))


def test_load_checkpoint_roundtrip(tmp_path):
    """Tensors spread over two shards are grouped by module; the residual is found only where present."""
    g = torch.Generator().manual_seed(4)
    t = {**_tensors("a.proj", 6, 10, 4, g), **_tensors("a.proj", 6, 10, 4, g, "_R")}
    _write(tmp_path, [t, _tensors("b.proj", 6, 10, 4, g)])
    params = ref.load_checkpoint(tmp_path)
    assert set(params) == {"a.proj", "b.proj"}
    assert params["a.proj"].residual is not None and params["b.proj"].residual is None
    assert params["a.proj"].primary.U.shape == (6, 4)


def test_load_checkpoint_duplicate_key_raises(tmp_path):
    """The same key in two shards is an error, not a silent overwrite."""
    g = torch.Generator().manual_seed(5)
    t = _tensors("a.proj", 6, 10, 4, g)
    _write(tmp_path, [t, {"a.proj.u1": t["a.proj.u1"]}])
    with pytest.raises(ValueError, match="duplicate"):
        ref.load_checkpoint(tmp_path)


def test_load_checkpoint_shape_mismatch_raises(tmp_path):
    """A `*_shape` that disagrees with the packed word count is rejected with a clear error."""
    g = torch.Generator().manual_seed(6)
    t = _tensors("a.proj", 6, 10, 4, g)
    t["a.proj.V_shape"] = torch.tensor([4, 100], dtype=torch.int32)  # needs 4 words, has 1
    _write(tmp_path, [t])
    with pytest.raises(ValueError, match="shape"):
        ref.load_checkpoint(tmp_path)


def _tiny_base(tmp_path, **kw):
    """Save a tiny random Llama model (2 layers, hidden 32) and return its directory.

    Extra keyword arguments go to LlamaConfig.  Embeddings and lm_head are
    untied (the Llama default), which the parameter count below relies on.
    """
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64, **kw)
    base = LlamaForCausalLM(cfg)
    base_dir = tmp_path / "base"
    base.save_pretrained(base_dir)
    return base_dir


def _tiny_ckpt(tmp_path):
    """Checkpoint with only two LittleBit layers: layer 0 up_proj and layer 1 q_proj."""
    g = torch.Generator().manual_seed(7)
    t = _tensors("model.layers.0.mlp.up_proj", 64, 32, 8, g)
    t.update(_tensors("model.layers.1.self_attn.q_proj", 32, 32, 8, g))
    _write(tmp_path / "ck", [t])
    return tmp_path / "ck"


def test_build_model_swaps_linears(tmp_path):
    """build_model replaces exactly the checkpoint's Linears and keeps the rest as real bf16.

    Fails if another Linear is swapped, a dense weight is kept, a tensor stays on
    the meta device, or the model cannot run a forward pass.
    """
    base_dir = _tiny_base(tmp_path)
    model = ref.build_model(str(base_dir), _tiny_ckpt(tmp_path))
    assert isinstance(model.model.layers[0].mlp.up_proj, ref.LBLinear)
    assert isinstance(model.model.layers[1].self_attn.q_proj, ref.LBLinear)
    assert isinstance(model.model.layers[0].mlp.down_proj, torch.nn.Linear)
    assert isinstance(model.lm_head, torch.nn.Linear)
    # Remaining params are bf16 and real; replaced dense weights are not retained.
    assert all(not t.is_meta and t.dtype == torch.bfloat16 for t in model.parameters())
    assert not hasattr(model.model.layers[0].mlp.up_proj, "weight")
    # Expected 21664 parameters (untied embeddings), all non-replaced ones:
    #   embed + lm_head 2*64*32 = 4096; norms (2 layers x 2 + final) x 32 = 160;
    #   attention 4*1024*2 - 1024 (q_proj of layer 1 replaced) = 7168;
    #   mlp 3*2048*2 - 2048 (up_proj of layer 0 replaced) = 10240.
    assert sum(t.numel() for t in model.parameters()) == 21664
    out = model(torch.tensor([[1, 2, 3]])).logits
    assert out.shape == (1, 3, 64) and torch.isfinite(out).all()


def test_build_model_warns_on_dropped_bias(tmp_path):
    """Replacing a Linear that has a bias warns, because LBLinear cannot represent the bias."""
    base_dir = _tiny_base(tmp_path, attention_bias=True)
    with pytest.warns(UserWarning, match="bias"):
        ref.build_model(str(base_dir), _tiny_ckpt(tmp_path))


def test_load_checkpoint_alias_packed_R(tmp_path):
    """The alternative residual key spelling (`U_packed_R`) is accepted like `U_R_packed`."""
    g = torch.Generator().manual_seed(11)
    t = _tensors("a.proj", 6, 10, 4, g)
    for k, v in _tensors("a.proj", 6, 10, 4, g, "_R").items():
        for m in "UV":  # U_R_packed -> U_packed_R alias spelling
            k = k.replace(f".{m}_R_packed", f".{m}_packed_R").replace(f".{m}_R_shape", f".{m}_shape_R")
        t[k] = v
    _write(tmp_path, [t])
    assert ref.load_checkpoint(tmp_path)["a.proj"].residual is not None


def test_load_checkpoint_rank_mismatch_raises(tmp_path):
    """U with 4 columns and V with 5 rows (different rank) is rejected."""
    g = torch.Generator().manual_seed(12)
    t = _tensors("a.proj", 6, 10, 4, g)
    t["a.proj.V_packed"] = ref.pack_signs(_signs(5, 10, g))
    t["a.proj.V_shape"] = torch.tensor([5, 10], dtype=torch.int32)
    _write(tmp_path, [t])
    with pytest.raises(ValueError, match="columns but V has"):
        ref.load_checkpoint(tmp_path)


def test_load_checkpoint_scale_length_raises(tmp_path):
    """A scale vector of the wrong length (u1 must have `out` entries) is rejected."""
    g = torch.Generator().manual_seed(13)
    t = _tensors("a.proj", 6, 10, 4, g)
    t["a.proj.u1"] = torch.rand(1, 7, generator=g)
    _write(tmp_path, [t])
    with pytest.raises(ValueError, match="u1"):
        ref.load_checkpoint(tmp_path)
