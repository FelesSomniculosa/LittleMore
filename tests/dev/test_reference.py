"""Tests for the lean PyTorch LittleBit reference (tools/dev/reference.py)."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import reference as ref


def _signs(rows, cols, gen):
    return torch.where(torch.rand(rows, cols, generator=gen) < 0.5, -1.0, 1.0)


def _params(out_f, in_f, r, gen, residual=True):
    def path(suffix):
        return {
            "U": _signs(out_f, r, gen), "V": _signs(r, in_f, gen),
            "u1": torch.rand(1, out_f, generator=gen) + 0.5,
            "u2": torch.rand(1, r, generator=gen) + 0.5,
            "v1": torch.rand(1, r, generator=gen) + 0.5,
            "v2": torch.rand(1, in_f, generator=gen) + 0.5,
        }
    return path("") , (path("_R") if residual else None)


def _dense(p):
    # W = diag(u1) Uq diag(v1*u2) Vq diag(v2)
    return (p["u1"].T * p["U"]) @ ((p["v1"] * p["u2"]).T * p["V"]) * p["v2"]


def _make(p, res):
    return ref.LBLinear(p["U"], p["V"], p["u1"], p["u2"], p["v1"], p["v2"],
                        residual=None if res is None else
                        (res["U"], res["V"], res["u1"], res["u2"], res["v1"], res["v2"]))


def test_unpack_matches_spec():
    # bit0=0 -> +1, bit1=1 -> -1, LSB-first; bits 2.. are 0 -> +1
    word = torch.tensor([[0b10]], dtype=torch.int32)
    out = ref.unpack_signs(word, (1, 4))
    assert out.tolist() == [[1.0, -1.0, 1.0, 1.0]]
    # bit 31 (sign bit of int32) = column 31 -> -1
    word = torch.tensor([[-2**31]], dtype=torch.int32)
    out = ref.unpack_signs(word, (1, 32))
    assert out[0, 31] == -1.0 and (out[0, :31] == 1.0).all()


def test_pack_unpack_roundtrip():
    g = torch.Generator().manual_seed(0)
    for cols in (1, 31, 32, 33, 70):
        s = _signs(5, cols, g)
        packed = ref.pack_signs(s)
        assert packed.dtype == torch.int32 and packed.shape == (5, (cols + 31) // 32)
        assert torch.equal(ref.unpack_signs(packed, (5, cols)), s)


def test_forward_matches_dense():
    g = torch.Generator().manual_seed(1)
    p, r = _params(24, 40, 8, g)
    layer = _make(p, r)
    x = torch.randn(3, 40, generator=g)
    want = x @ (_dense(p) + _dense(r)).T
    torch.testing.assert_close(layer(x), want, rtol=1e-5, atol=1e-5)


def test_residual_optional():
    g = torch.Generator().manual_seed(2)
    p, _ = _params(16, 20, 4, g, residual=False)
    layer = _make(p, None)
    x = torch.randn(2, 20, generator=g)
    torch.testing.assert_close(layer(x), x @ _dense(p).T, rtol=1e-5, atol=1e-5)


def test_padded_rank_exact():
    # rank 33 -> 2 packed words with 31 padding bits; padding must not leak in.
    g = torch.Generator().manual_seed(3)
    p, r = _params(8, 70, 33, g)
    for q in (p, r):
        q["U_packed"] = ref.pack_signs(q["U"])
        q["V_packed"] = ref.pack_signs(q["V"])
        # dirty the padding bits of the last word (set them to 1 = -1)
        for k, cols in (("U_packed", 33), ("V_packed", 70)):
            pad = ~((1 << (cols % 32)) - 1) & 0xFFFFFFFF
            last = q[k][:, -1].to(torch.int64) & 0xFFFFFFFF | pad
            q[k][:, -1] = torch.where(last >= 2**31, last - 2**32, last).to(torch.int32)
    for q in (p, r):
        q["U"] = ref.unpack_signs(q["U_packed"], q["U"].shape)
        q["V"] = ref.unpack_signs(q["V_packed"], q["V"].shape)
    layer = _make(p, r)
    x = torch.randn(2, 70, generator=g)
    torch.testing.assert_close(layer(x), x @ (_dense(p) + _dense(r)).T, rtol=1e-5, atol=1e-5)


def _tensors(prefix, out_f, in_f, r, g, suffix=""):
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
    d.mkdir(exist_ok=True)
    for i, t in enumerate(shards):
        save_file(t, str(d / f"model-{i}.safetensors"))
    (d / "littlebit_config.json").write_text(json.dumps({"rank": 4}))


def test_load_checkpoint_roundtrip(tmp_path):
    g = torch.Generator().manual_seed(4)
    t = {**_tensors("a.proj", 6, 10, 4, g), **_tensors("a.proj", 6, 10, 4, g, "_R")}
    _write(tmp_path, [t, _tensors("b.proj", 6, 10, 4, g)])
    params = ref.load_checkpoint(tmp_path)
    assert set(params) == {"a.proj", "b.proj"}
    assert params["a.proj"].residual is not None and params["b.proj"].residual is None
    assert params["a.proj"].primary.U.shape == (6, 4)


def test_load_checkpoint_duplicate_key_raises(tmp_path):
    g = torch.Generator().manual_seed(5)
    t = _tensors("a.proj", 6, 10, 4, g)
    _write(tmp_path, [t, {"a.proj.u1": t["a.proj.u1"]}])
    with pytest.raises(ValueError, match="duplicate"):
        ref.load_checkpoint(tmp_path)


def test_load_checkpoint_shape_mismatch_raises(tmp_path):
    g = torch.Generator().manual_seed(6)
    t = _tensors("a.proj", 6, 10, 4, g)
    t["a.proj.V_shape"] = torch.tensor([4, 100], dtype=torch.int32)  # needs 4 words, has 1
    _write(tmp_path, [t])
    with pytest.raises(ValueError, match="shape"):
        ref.load_checkpoint(tmp_path)


def test_build_model_swaps_linears(tmp_path, monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64)
    base = LlamaForCausalLM(cfg)
    base_dir = tmp_path / "base"
    base.save_pretrained(base_dir)
    g = torch.Generator().manual_seed(7)
    t = _tensors("model.layers.0.mlp.up_proj", 64, 32, 8, g)
    t.update(_tensors("model.layers.1.self_attn.q_proj", 32, 32, 8, g))
    _write(tmp_path / "ck", [t])
    model = ref.build_model(str(base_dir), tmp_path / "ck")
    assert isinstance(model.model.layers[0].mlp.up_proj, ref.LBLinear)
    assert isinstance(model.model.layers[1].self_attn.q_proj, ref.LBLinear)
    assert isinstance(model.model.layers[0].mlp.down_proj, torch.nn.Linear)
    assert isinstance(model.lm_head, torch.nn.Linear)
    out = model(torch.tensor([[1, 2, 3]])).logits
    assert out.shape == (1, 3, 64) and torch.isfinite(out).all()
