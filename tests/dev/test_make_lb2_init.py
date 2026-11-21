"""Tests for the init-only LittleBit-2 maker (tools/dev/make_lb2_init.py)."""
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import make_lb2_init as mk
import reference as ref


def _spiky_w(n=256, m=192, seed=0):
    """Random matrix with a decaying spectrum -> spiky SVD factors."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, m, generator=g)
    Uq, _ = torch.linalg.qr(torch.randn(n, m, generator=g))
    Vq, _ = torch.linalg.qr(torch.randn(m, m, generator=g))
    s = torch.arange(1, m + 1, dtype=torch.float32) ** -1.0
    return (Uq * s) @ Vq.T + 0.0 * A


def _factors(W, r):
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    sq = S[:r].sqrt()
    return U[:, :r] * sq, Vt[:r].T * sq


def _lam(Z):
    l1 = Z.abs().sum(1)
    l2 = Z.norm(dim=1)
    return (1 - (l1 / l2) ** 2 / Z.shape[1]).mean().item()


def _recon(Ub, Vb, h, g, l):
    return (h[:, None] * Ub) @ (l[:, None] * Vb) * g[None, :]


def _rel(W, What):
    return ((W - What).norm() / W.norm()).item()


def test_joint_itq_orthogonal():
    Z = torch.cat(_factors(_spiky_w(), 32), 0)
    R = mk.joint_itq(Z, n_iter=20, seed=1)
    assert torch.allclose(R.T @ R, torch.eye(32), atol=1e-4)


def test_joint_itq_reduces_distortion():
    U, V = _factors(_spiky_w(), 32)
    Z = torch.cat([U, V], 0)
    R = mk.joint_itq(Z, seed=0)
    assert _lam(Z @ R) < _lam(Z)


def test_rotation_preserves_product():
    U, V = _factors(_spiky_w(), 32)
    R = mk.joint_itq(torch.cat([U, V], 0), seed=0)
    assert torch.allclose((U @ R) @ (V @ R).T, U @ V.T, atol=1e-4)


@pytest.mark.parametrize("d_out,d_in,bpw", [(4096, 4096, 0.55), (4096, 11008, 0.1)])
def test_rank_for_bits_never_exceeds_budget(d_out, d_in, bpw):
    r = mk.rank_for_bits(d_out, d_in, bpw)
    assert r >= 8 and r % 8 == 0
    bits = 32 * (d_in + d_out) + 2 * r * (d_in + d_out + 16)
    assert bits / (d_in * d_out) <= bpw
    r1 = mk.rank_for_bits(d_out, d_in, bpw, residual=False)
    assert r1 % 8 == 0
    assert (16 * (d_in + d_out) + r1 * (d_in + d_out + 16)) / (d_in * d_out) <= bpw


def test_rank_for_bits_tiny_matrix_floor():
    assert mk.rank_for_bits(64, 64, 0.1) == 8


def test_roundtrip_pack_unpack():
    g = torch.Generator().manual_seed(0)
    s = torch.where(torch.rand(7, 70, generator=g) < 0.5, -1.0, 1.0)
    assert torch.equal(ref.unpack_signs(ref.pack_signs(s), (7, 70)), s)


def test_dual_svid_reconstructs():
    W = _spiky_w()
    U, V = _factors(W, 48)
    Ub, Vb, h, g, l = mk.dual_svid(U, V)
    assert Ub.shape == (256, 48) and Vb.shape == (48, 192)
    assert set(Ub.unique().tolist()) <= {-1.0, 1.0}
    e1 = _rel(W, _recon(Ub, Vb, h, g, l))
    assert e1 < 1.0
    res = W - _recon(Ub, Vb, h, g, l)
    Ur, Vr = _factors(res, 48)
    Ub2, Vb2, h2, g2, l2 = mk.dual_svid(Ur, Vr)
    e2 = _rel(W, _recon(Ub, Vb, h, g, l) + _recon(Ub2, Vb2, h2, g2, l2))
    assert e2 < e1


def test_itq_beats_plain_svid_on_reconstruction_after_binarization():
    W = _spiky_w()
    U, V = _factors(W, 48)
    plain = _rel(W, _recon(*mk.dual_svid(U, V)))
    R = mk.joint_itq(torch.cat([U, V], 0), seed=0)
    itq = _rel(W, _recon(*mk.dual_svid(U @ R, V @ R)))
    assert itq < plain


def _tiny_model(tmp_path: Path) -> Path:
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64)
    d = tmp_path / "base"
    LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(d)
    (d / "tokenizer.json").write_text("{}")
    return d


def test_make_checkpoint_loads_in_reference(tmp_path):
    base, out = _tiny_model(tmp_path), tmp_path / "ck"
    mk.make_checkpoint(str(base), out, bpw=1.0, seed=0)
    cfg = json.loads((out / "littlebit_config.json").read_text())
    assert cfg["eff_bit"] == 1.0 and cfg["residual"] and cfg["use_itq"]
    assert (out / "config.json").exists() and (out / "tokenizer.json").exists()
    params = ref.load_checkpoint(out)
    assert len(params) == 2 * 7 and all(p.residual is not None for p in params.values())
    model = ref.build_model(str(base), out)
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3, 4]])).logits
    assert torch.isfinite(logits.float()).all()


def test_make_checkpoint_without_residual(tmp_path):
    base, out = _tiny_model(tmp_path), tmp_path / "ck"
    mk.make_checkpoint(str(base), out, bpw=1.0, use_itq=False, residual=False)
    assert all(p.residual is None for p in ref.load_checkpoint(out).values())
    with safe_open(str(out / "model.safetensors"), framework="pt") as f:
        assert not any("_R" in k for k in f.keys())
    assert json.loads((out / "littlebit_config.json").read_text())["residual"] is False


def test_kv_factor_per_tensor_rank(tmp_path):
    base, out = _tiny_model(tmp_path), tmp_path / "ck"
    mk.make_checkpoint(str(base), out, bpw=8.0, kv_factor=0.5)
    ranks = {}
    with safe_open(str(out / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.endswith(".U_shape"):
                ranks[k[:-8]] = int(f.get_tensor(k)[1])
    assert all(r % 8 == 0 and r >= 8 for r in ranks.values())
    q = ranks["model.layers.0.self_attn.q_proj"]
    assert ranks["model.layers.0.self_attn.k_proj"] < q
    assert ranks["model.layers.0.self_attn.v_proj"] < q
    assert ranks["model.layers.0.self_attn.o_proj"] == q
