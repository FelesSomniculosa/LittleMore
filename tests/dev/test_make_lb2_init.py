"""Tests for the init-only LittleBit-2 maker (tools/dev/make_lb2_init.py).

Run with `pytest tests/dev`.  Terms (SVD, Joint-ITQ, Dual-SVID, GQA) are defined
in docs/development.md.
"""
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
    Uq, _ = torch.linalg.qr(torch.randn(n, m, generator=g))
    Vq, _ = torch.linalg.qr(torch.randn(m, m, generator=g))
    s = torch.arange(1, m + 1, dtype=torch.float32) ** -1.0
    return (Uq * s) @ Vq.T


def _factors(W, r):
    """Rank-r factors (U^, V^) of W with the singular values split evenly (sqrt on each side)."""
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    sq = S[:r].sqrt()
    return U[:, :r] * sq, Vt[:r].T * sq


def _lam(Z):
    """Mean non-binarizability of the rows of Z: 0 when every entry of a row has equal
    magnitude (signs alone then describe it exactly), larger when a few entries dominate."""
    l1 = Z.abs().sum(1)
    l2 = Z.norm(dim=1)
    return (1 - (l1 / l2) ** 2 / Z.shape[1]).mean().item()


def _recon(Ub, Vb, h, g, l):
    """Dense matrix diag(h) Ub diag(l) Vb diag(g) from Dual-SVID outputs."""
    return (h[:, None] * Ub) @ (l[:, None] * Vb) * g[None, :]


def _rel(W, What):
    """Relative error ||W - What|| / ||W||; 1.0 means no better than reconstructing zeros."""
    return ((W - What).norm() / W.norm()).item()


def test_joint_itq_orthogonal():
    """The Joint-ITQ rotation is orthogonal (R^T R = I); fails if the Procrustes update is wrong."""
    Z = torch.cat(_factors(_spiky_w(), 32), 0)
    R = mk.joint_itq(Z, n_iter=20, seed=1)
    assert torch.allclose(R.T @ R, torch.eye(32), atol=1e-4)


def test_joint_itq_reduces_distortion():
    """Rotating by R makes the factors more binarizable than before (the point of Joint-ITQ)."""
    U, V = _factors(_spiky_w(), 32)
    Z = torch.cat([U, V], 0)
    R = mk.joint_itq(Z, seed=0)
    assert _lam(Z @ R) < _lam(Z)


def test_rotation_preserves_product():
    """Rotating both factors by the same R leaves U V^T unchanged, so the weight is not distorted."""
    U, V = _factors(_spiky_w(), 32)
    R = mk.joint_itq(torch.cat([U, V], 0), seed=0)
    assert torch.allclose((U @ R) @ (V @ R).T, U @ V.T, atol=1e-4)


@pytest.mark.parametrize("d_out,d_in,bpw", [(4096, 4096, 0.55), (4096, 11008, 0.1)])
def test_rank_for_bits_never_exceeds_budget(d_out, d_in, bpw):
    """The chosen rank is a multiple of 8 and its bit cost (formula in rank_for_bits) stays
    within `bpw`, for two and for one path.  Shapes are 4096 x 4096 and 4096 x 11008,
    typical 7B-model sizes."""
    r = mk.rank_for_bits(d_out, d_in, bpw)
    assert r >= 8 and r % 8 == 0
    bits = 32 * (d_in + d_out) + 2 * r * (d_in + d_out + 16)
    assert bits / (d_in * d_out) <= bpw
    r1 = mk.rank_for_bits(d_out, d_in, bpw, residual=False)
    assert r1 % 8 == 0
    assert (16 * (d_in + d_out) + r1 * (d_in + d_out + 16)) / (d_in * d_out) <= bpw


def test_rank_for_bits_tiny_matrix_floor():
    """When the budget gives rank < 8 the result is clamped up to 8 (documented behaviour)."""
    assert mk.rank_for_bits(64, 64, 0.1) == 8


def test_roundtrip_pack_unpack():
    """Pack/unpack round trip, checked again here because the maker writes checkpoints with
    pack_signs; a change to the packing would break maker and reference together."""
    g = torch.Generator().manual_seed(0)
    s = torch.where(torch.rand(7, 70, generator=g) < 0.5, -1.0, 1.0)
    assert torch.equal(ref.unpack_signs(ref.pack_signs(s), (7, 70)), s)


def test_dual_svid_reconstructs():
    """Dual-SVID gives +/-1 signs and a useful approximation, and a second (residual) fit helps.

    Fails if signs are not +/-1, the error is not clearly below the zero baseline (0.9 vs
    1.0), or adding the residual path does not reduce the error.
    """
    W = _spiky_w()
    U, V = _factors(W, 48)
    Ub, Vb, h, g, l = mk.dual_svid(U, V)
    assert Ub.shape == (256, 48) and Vb.shape == (48, 192)
    assert set(Ub.unique().tolist()) <= {-1.0, 1.0}
    e1 = _rel(W, _recon(Ub, Vb, h, g, l))
    assert e1 < _rel(W, torch.zeros_like(W))  # beats the zero-reconstruction baseline (1.0)
    assert e1 < 0.9
    res = W - _recon(Ub, Vb, h, g, l)
    Ur, Vr = _factors(res, 48)
    Ub2, Vb2, h2, g2, l2 = mk.dual_svid(Ur, Vr)
    e2 = _rel(W, _recon(Ub, Vb, h, g, l) + _recon(Ub2, Vb2, h2, g2, l2))
    assert e2 < e1


def test_itq_beats_plain_svid_on_reconstruction_after_binarization():
    """Joint-ITQ rotation before Dual-SVID gives a smaller reconstruction error than no rotation."""
    W = _spiky_w()
    U, V = _factors(W, 48)
    plain = _rel(W, _recon(*mk.dual_svid(U, V)))
    R = mk.joint_itq(torch.cat([U, V], 0), seed=0)
    itq = _rel(W, _recon(*mk.dual_svid(U @ R, V @ R)))
    assert itq < plain


def _tiny_model(tmp_path: Path, hidden: int = 64, heads: int = 4) -> Path:
    """Save a tiny random Llama model (2 layers, 2 KV heads, GQA) as bf16 and return its directory.

    Writes a dummy tokenizer.json so the maker's side-file copying can be checked.
    """
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=2, num_attention_heads=heads, num_key_value_heads=2,
                      max_position_embeddings=64)
    d = tmp_path / "base"
    LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(d)
    (d / "tokenizer.json").write_text("{}")
    return d


def _path_dense(q):
    """Independent dense W for one loaded path: diag(u1) U diag(v1*u2) V diag(v2)."""
    U, V = q.U.float(), q.V.float()
    return (q.u1.reshape(-1, 1) * U) @ ((q.v1 * q.u2).reshape(-1, 1) * V) * q.v2.reshape(1, -1)


def _base_weight(base: Path, name: str) -> torch.Tensor:
    """Weight `<name>.weight` of the saved base model, as fp32."""
    with safe_open(str(base / "model.safetensors"), framework="pt") as f:
        return f.get_tensor(name + ".weight").float()


def _ok(W, What):
    """Check (a): clearly better than the zero reconstruction (rel. error 1.0)."""
    return _rel(W, What) < 0.95


@pytest.mark.parametrize("name", ["model.layers.0.self_attn.q_proj",
                                  "model.layers.1.self_attn.k_proj",
                                  "model.layers.1.mlp.down_proj"])
def test_checkpoint_reconstructs_base_weight(tmp_path, name):
    """The written checkpoint really approximates the base weight, through the reference.

    Checks: (a) primary alone and primary+residual are each clearly better than the zero
    reconstruction (error < 0.95; the margin below 1.0 absorbs random-weight variation);
    (b) adding the residual reduces the error; (c) the dense matrix rebuilt from the file
    equals the maker's own in-memory reconstruction, and LBLinear on an identity batch
    gives the same matrix.  Fails if the maker mislabels a scale, packs signs wrongly or
    the reference reads them differently.  bpw 8.0 gives a high rank so the error is small.
    """
    base, out = _tiny_model(tmp_path), tmp_path / "ck"
    mk.make_checkpoint(str(base), out, bpw=8.0, seed=0)
    W = _base_weight(base, name)
    p = ref.load_checkpoint(out)[name]
    prim, full = _path_dense(p.primary), _path_dense(p.primary) + _path_dense(p.residual)
    assert _ok(W, prim) and _ok(W, full)                       # (a)
    assert _rel(W, full) < _rel(W, prim)                       # (b)
    # (c) equals the maker's own dense reconstruction
    r = mk._layer_rank(name, W.shape[0], W.shape[1], 8.0, True, 1.0)
    d1, _ = mk._path(W, r, True, 0)
    d2, _ = mk._path(W - d1, r, True, 0)
    assert torch.allclose(prim, d1, atol=1e-4, rtol=1e-4)
    assert torch.allclose(full, d1 + d2, atol=1e-4, rtol=1e-4)
    # the reference module's forward on the identity must agree too
    lin = ref.LBLinear.from_params(p)
    assert torch.allclose(lin(torch.eye(W.shape[1])).T, full, atol=1e-3, rtol=1e-3)


def test_checkpoint_reconstruction_check_has_teeth(tmp_path):
    """Proves check (a) can fail: swapping the u1/v2 roles (square q_proj) must fail it.

    ("Has teeth" = the check fails when the code is wrong; see docs/development.md.)
    """
    from safetensors.torch import load_file, save_file
    base, out = _tiny_model(tmp_path), tmp_path / "ck"
    name = "model.layers.0.self_attn.q_proj"
    # Random weights have near-uniform row/column scales, so swapping u1/v2 would be
    # harmless; give q_proj strongly non-uniform scales (as real weights have).
    st = load_file(str(base / "model.safetensors"))
    g = torch.Generator().manual_seed(1)
    rs, cs = torch.exp(torch.randn(64, generator=g)), torch.exp(torch.randn(64, generator=g))
    st[name + ".weight"] = (st[name + ".weight"].float() * rs[:, None] * cs[None, :]).bfloat16()
    save_file(st, str(base / "model.safetensors"), metadata={"format": "pt"})
    mk.make_checkpoint(str(base), out, bpw=8.0, seed=0)
    W, p = _base_weight(base, name), ref.load_checkpoint(out)[name].primary
    assert _ok(W, _path_dense(p))
    p.u1, p.v2 = p.v2, p.u1
    assert not _ok(W, _path_dense(p))


def test_make_checkpoint_loads_in_reference(tmp_path):
    """End to end: the maker's output loads in the reference and the model runs.

    Checks the config file, copied side files, 14 modules (7 linears x 2 layers, each with a
    residual) and finite logits.
    """
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
    """With residual=False no `_R` tensor is written and the config records it."""
    base, out = _tiny_model(tmp_path), tmp_path / "ck"
    mk.make_checkpoint(str(base), out, bpw=1.0, use_itq=False, residual=False)
    assert all(p.residual is None for p in ref.load_checkpoint(out).values())
    with safe_open(str(out / "model.safetensors"), framework="pt") as f:
        assert not any("_R" in k for k in f.keys())
    assert json.loads((out / "littlebit_config.json").read_text())["residual"] is False


def _ranks(out: Path) -> dict:
    """Map module name -> stored rank, read from each `.U_shape` tensor of a checkpoint."""
    ranks = {}
    with safe_open(str(out / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.endswith(".U_shape"):
                ranks[k[:-8]] = int(f.get_tensor(k)[1])
    return ranks


def test_kv_factor_per_tensor_rank(tmp_path):
    """kv_factor scales only the k_proj / v_proj ranks (GQA); all other ranks are unchanged."""
    # hidden 256 / bpw 1.0: uncapped ranks (q 40, k/v 24 < min(dim)), so kv_factor binds.
    base = _tiny_model(tmp_path, hidden=256, heads=4)
    mk.make_checkpoint(str(base), tmp_path / "a", bpw=1.0, kv_factor=1.0)
    mk.make_checkpoint(str(base), tmp_path / "b", bpw=1.0, kv_factor=2.0)
    a, b = _ranks(tmp_path / "a"), _ranks(tmp_path / "b")
    assert a.keys() == b.keys()
    assert all(r % 8 == 0 and r >= 8 for r in [*a.values(), *b.values()])
    kv = [n for n in a if n.endswith(("k_proj", "v_proj"))]
    assert len(kv) == 4
    for n in a:
        if n in kv:
            assert b[n] == 2 * a[n]
        else:
            assert b[n] == a[n]
