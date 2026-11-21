"""Tests for the Samsung-format LittleBit checkpoint -> GGUF (B1) converter.

Run with `pytest tests/convert`.  All models are tiny random HF models built offline
(see _tiny.py); no network.  Terms (LittleBit, GGUF, GQA, RoPE) are defined
in the glossary of docs/development.md.  B1 is the engine's pure 1-bit sign type.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

import _tiny
import reference as ref
from tools.convert import lb_to_gguf as lb  # also puts engine/gguf-py on sys.path
from tools.convert import validate as val

import gguf  # noqa: E402  (the engine's gguf-py, put on sys.path by lb_to_gguf)

needs_qwen_tok = pytest.mark.skipif(
    not _tiny.qwen_tokenizer_available(),
    reason=f"Qwen3-0.6B tokenizer not cached at {_tiny.QWEN_TOK_DIR} "
           "(.venv/Scripts/hf.exe download Qwen/Qwen3-0.6B --local-dir .cache/models/Qwen3-0.6B)")

B1 = gguf.GGMLQuantizationType.B1


def _shape_of(t) -> tuple[int, ...]:
    """ggml shape [ne0, ne1] of a GGUF reader tensor (ne0 = contiguous dim)."""
    return tuple(int(x) for x in t.shape)


def _read(path: Path) -> dict:
    """Tensors of a GGUF file by name."""
    return {t.name: t for t in gguf.GGUFReader(str(path)).tensors}


@pytest.fixture(scope="module")
def qwen3_ckpt(tmp_path_factory):
    """A tiny Qwen3 init-only checkpoint (two paths: primary + residual)."""
    if not _tiny.qwen_tokenizer_available():
        pytest.skip("Qwen3-0.6B tokenizer not cached")
    return _tiny.make_tiny_checkpoint("qwen3", tmp_path_factory.mktemp("q3"))


@pytest.fixture(scope="module")
def qwen3_converted(qwen3_ckpt, tmp_path_factory):
    """Convert the tiny Qwen3 checkpoint once; returns (checkpoint dir, output GGUF path, conversion report)."""
    out = tmp_path_factory.mktemp("q3out") / "q3.gguf"
    report = lb.convert_checkpoint(qwen3_ckpt, out)
    return qwen3_ckpt, out, report


# --------------------------------------------------------------------------- core

@needs_qwen_tok
def test_each_layer_matches_reference(qwen3_converted):
    """Every LittleBit linear, re-decoded from the written GGUF, equals the PyTorch reference.

    Checked here with an independent decode in the test (not just the converter's own
    validation) for every linear, through the permutations the converter reported.
    """
    ckpt, out, report = qwen3_converted
    params = ref.load_checkpoint(ckpt)
    tensors = _read(out)
    assert report.n_linears == len(params) == report.n_validated
    assert report.max_rel_err < report.tolerance
    rng = np.random.default_rng(1)
    for mod, p in params.items():
        for pr in report.probes[mod]:
            base = pr.gguf_name.removesuffix(".weight")
            x = rng.standard_normal((3, p.primary.V.shape[1])).astype(np.float32)
            want = ref.LBLinear.from_params(p)(torch.from_numpy(x)).numpy()[:, pr.row_idx]
            y = 0
            for sfx in ("", "r"):
                if f"{base}.lb_v{sfx}" not in tensors:
                    continue
                g = {k: tensors[f"{base}.{k}{sfx}"] for k in ("lb_v", "lb_u", "lb_s_v2", "lb_s_mid", "lb_s_u1")}
                # own decode: B1 bit 1 = +1, bit 0 = -1, LSB first
                V = np.where(np.unpackbits(g["lb_v"].data, axis=-1, bitorder="little") == 1, 1.0, -1.0)
                U = np.where(np.unpackbits(g["lb_u"].data, axis=-1, bitorder="little") == 1, 1.0, -1.0)
                xg = x[:, pr.col_idx].astype(np.float64)
                h = (xg * g["lb_s_v2"].data) @ V.T * g["lb_s_mid"].data
                y = y + (h @ U.T) * g["lb_s_u1"].data
            np.testing.assert_allclose(y, want, rtol=1e-4, atol=1e-5 * np.abs(want).max())
    # the GGUF is the engine's expected layout: B1 signs, F32 scales
    t = tensors["blk.0.attn_q.lb_v"]
    assert t.tensor_type == B1 and tensors["blk.0.attn_q.lb_s_mid"].tensor_type == gguf.GGMLQuantizationType.F32
    assert report.written_names, "converter reported nothing written"


@needs_qwen_tok
def test_no_residual_checkpoint(tmp_path):
    """A checkpoint without residual paths converts, and no `r`-suffixed tensor is written."""
    ck = _tiny.make_tiny_checkpoint("qwen3", tmp_path, residual=False)
    out = tmp_path / "o.gguf"
    report = lb.convert_checkpoint(ck, out)
    names = _read(out)
    assert any(n.endswith(".lb_v") for n in names)
    assert not any(n.endswith(("lb_vr", "lb_ur", "lb_s_v2r", "lb_s_midr", "lb_s_u1r")) for n in names)
    assert report.n_validated == report.n_linears > 0


@needs_qwen_tok
def test_rank_not_multiple_of_128_pads_exactly(tmp_path):
    """Logical rank 136 is padded to 256; the padded mid entries are exactly 0; validation stays tiny."""
    # bpw 1.23 without residual gives rank 136 for the 256x256 attention matrices.
    ck = _tiny.make_tiny_checkpoint("qwen3", tmp_path, bpw=1.23, residual=False)
    params = ref.load_checkpoint(ck)
    assert params["model.layers.0.self_attn.q_proj"].primary.V.shape[0] == 136
    out = tmp_path / "o.gguf"
    report = lb.convert_checkpoint(ck, out)
    t = _read(out)
    mid = t["blk.0.attn_q.lb_s_mid"].data
    assert mid.shape == (256,) and np.all(mid[136:] == 0) and np.any(mid[:136] != 0)
    assert _shape_of(t["blk.0.attn_q.lb_v"]) == (256, 256)  # ggml ne = [n_in, r_padded]
    p = params["model.layers.0.self_attn.q_proj"].primary
    np.testing.assert_array_equal(mid[:136], (p.v1.float() * p.u2.float()).numpy().reshape(-1))
    assert report.max_rel_err < 1e-5
    # the padded sign bits are +1 (bit 1), whatever: harmless because mid = 0 there
    padded_v_rows = np.unpackbits(t["blk.0.attn_q.lb_v"].data[136:], axis=-1, bitorder="little")
    assert padded_v_rows.min() == 1


@needs_qwen_tok
def test_kv_factor_per_tensor_rank(tmp_path):
    """GQA: k/v have their own (smaller) rank; the converter pads each to its own multiple of 128."""
    ck = _tiny.make_tiny_checkpoint("qwen3", tmp_path, bpw=4.0, kv_factor=0.5)
    params = ref.load_checkpoint(ck)
    rq = params["model.layers.0.self_attn.q_proj"].primary.V.shape[0]
    rk = params["model.layers.0.self_attn.k_proj"].primary.V.shape[0]
    rv = params["model.layers.0.self_attn.v_proj"].primary.V.shape[0]
    assert rk == rv < 128 < rq  # sanity of the fixture: the ranks really differ (and pad differently)
    out = tmp_path / "o.gguf"
    lb.convert_checkpoint(ck, out)
    t = _read(out)
    for name, r in (("attn_q", rq), ("attn_k", rk), ("attn_v", rv)):
        want = lb.pad_to(r)
        assert _shape_of(t[f"blk.0.{name}.lb_v"])[1] == want, name
        assert _shape_of(t[f"blk.0.{name}.lb_u"])[0] == want, name
        assert t[f"blk.0.{name}.lb_s_mid"].data.shape == (want,), name
    assert _shape_of(t["blk.0.attn_k.lb_v"])[1] != _shape_of(t["blk.0.attn_q.lb_v"])[1]


def _shrink_residual(ckpt: Path, module: str, new_r: int) -> None:
    """Rewrite `module`'s residual path in `ckpt` at a smaller rank (so it differs from the primary)."""
    f = ckpt / "model.safetensors"
    t = load_file(str(f))
    U = ref.unpack_signs(t[f"{module}.U_R_packed"], t[f"{module}.U_R_shape"].tolist())[:, :new_r]
    V = ref.unpack_signs(t[f"{module}.V_R_packed"], t[f"{module}.V_R_shape"].tolist())[:new_r]
    t[f"{module}.U_R_packed"], t[f"{module}.U_R_shape"] = ref.pack_signs(U), torch.tensor(U.shape)
    t[f"{module}.V_R_packed"], t[f"{module}.V_R_shape"] = ref.pack_signs(V), torch.tensor(V.shape)
    for s in ("u2_R", "v1_R"):
        t[f"{module}.{s}"] = t[f"{module}.{s}"][:, :new_r].contiguous()
    save_file(t, str(f), metadata={"format": "pt"})


@needs_qwen_tok
def test_padded_rank_with_residual_different_rank(tmp_path):
    """Primary rank 136 (-> 256) with residual rank 72 (-> 128): each path padded on its own, exact."""
    # bpw 2.46 with a residual gives rank 136 on both paths of the 256x256 matrices.
    ck = _tiny.make_tiny_checkpoint("qwen3", tmp_path, bpw=2.46, residual=True)
    mod = "model.layers.0.self_attn.q_proj"
    assert ref.load_checkpoint(ck)[mod].primary.V.shape[0] == 136
    _shrink_residual(ck, mod, 72)
    out = tmp_path / "o.gguf"
    report = lb.convert_checkpoint(ck, out)
    t = _read(out)
    assert _shape_of(t["blk.0.attn_q.lb_v"])[1] == 256 and _shape_of(t["blk.0.attn_q.lb_vr"])[1] == 128
    assert np.all(t["blk.0.attn_q.lb_s_mid"].data[136:] == 0)
    assert np.all(t["blk.0.attn_q.lb_s_midr"].data[72:] == 0)
    assert report.max_rel_err < report.tolerance


# --------------------------------------------------------------------------- errors

@needs_qwen_tok
@pytest.mark.parametrize("missing", ["config.json", "littlebit_config.json"])
def test_missing_config_raises(qwen3_ckpt, tmp_path, missing):
    """A checkpoint without config.json (HF architecture) or littlebit_config.json is refused up front."""
    ck = tmp_path / "ck"
    shutil.copytree(qwen3_ckpt, ck)
    (ck / missing).unlink()
    with pytest.raises(lb.ConversionError, match=missing) as e:
        lb.convert(ck, tmp_path / "o.gguf")
    if missing == "littlebit_config.json":
        assert "--assume-no-littlebit-config" in str(e.value)  # the message says how to proceed
    assert not (tmp_path / "o.gguf").exists()


@needs_qwen_tok
def test_shape_mismatch_raises(qwen3_ckpt, tmp_path):
    """U_shape disagreeing with the packed tensor aborts, naming the layer."""
    ck = tmp_path / "ck"
    shutil.copytree(qwen3_ckpt, ck)
    t = load_file(str(ck / "model.safetensors"))
    key = "model.layers.1.mlp.up_proj.U_shape"
    t[key] = t[key] + torch.tensor([8, 0])
    save_file(t, str(ck / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(lb.ConversionError, match=r"layers\.1\.mlp\.up_proj"):
        lb.convert(ck, tmp_path / "o.gguf")
    assert not (tmp_path / "o.gguf").exists()


@needs_qwen_tok
def test_duplicate_shard_key_raises(qwen3_ckpt, tmp_path):
    """The same tensor key in two shards is ambiguous: refuse instead of picking one."""
    ck = tmp_path / "ck"
    shutil.copytree(qwen3_ckpt, ck)
    shutil.copy2(ck / "model.safetensors", ck / "model-extra.safetensors")
    with pytest.raises(lb.ConversionError, match="duplicate"):
        lb.convert(ck, tmp_path / "o.gguf")
    assert not (tmp_path / "o.gguf").exists()


@needs_qwen_tok
def test_no_partial_file_on_failure(qwen3_ckpt, tmp_path, monkeypatch):
    """A failed validation leaves neither the output file nor a temp file behind."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(lb.validate, "validate_layer", lambda *a, **k: 1.0)  # force "validation failed"
    with pytest.raises(lb.ConversionError, match="tolerance"):
        lb.convert(qwen3_ckpt, out_dir / "m.gguf")
    assert list(out_dir.iterdir()) == []


@needs_qwen_tok
def test_corrupting_writer_is_caught_and_named(qwen3_ckpt, tmp_path, monkeypatch):
    """If the sign packer misbehaves, convert(validate=True) aborts naming the layer; nothing is left."""
    real = lb.signs_to_b1

    def bad(signs):
        """The real packer, but with the first sign of the first row flipped."""
        out = real(signs)
        out[0, 0] ^= 0x01  # flip the first sign of the first row
        return out

    monkeypatch.setattr(lb, "signs_to_b1", bad)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(lb.ConversionError, match=r"layers\.0\.\S+"):
        lb.convert(qwen3_ckpt, out_dir / "m.gguf")
    assert list(out_dir.iterdir()) == []


@needs_qwen_tok
def test_skipped_linear_raises(qwen3_ckpt, tmp_path, monkeypatch):
    """A linear for which the HF converter yields no GGUF tensor must abort, not vanish silently."""
    real = lb.probe_linear
    monkeypatch.setattr(lb, "probe_linear",
                        lambda conv, name, *a, **k: [] if name.endswith("layers.0.mlp.up_proj.weight") else real(conv, name, *a, **k))
    with pytest.raises(lb.ConversionError, match="up_proj"):
        lb.convert(qwen3_ckpt, tmp_path / "o.gguf")
    assert not (tmp_path / "o.gguf").exists()


# --------------------------------------------------------------------------- bits

def test_polarity_flip_known_pattern():
    """Samsung words (+1 = bit 0, -1 = bit 1) -> B1 bytes (+1 = bit 1, -1 = bit 0), checked byte by byte.

    A converter that forgot the flip fails here (every byte would be the complement).
    NB: forgetting the flip on BOTH U and V is invisible in the layer output (the two sign
    errors cancel), which is exactly why the bytes are pinned directly.
    """
    # one row of 128 columns = 4 int32 words.  Samsung bits set (= -1) at columns 0, 9, 31, 32, 127.
    words = np.zeros(4, dtype=np.uint32)
    for col in (0, 9, 31, 32, 127):
        words[col // 32] |= np.uint32(1) << np.uint32(col % 32)
    packed = torch.from_numpy(words.view(np.int32).copy()).reshape(1, 4)
    signs = ref.unpack_signs(packed, (1, 128), dtype=torch.int8).numpy()
    assert signs[0, 0] == -1 and signs[0, 1] == 1  # sanity: bit 1 -> -1, bit 0 -> +1
    got = lb.signs_to_b1(signs)
    assert got.shape == (1, 16) and got.dtype == np.uint8
    expect = np.full(16, 0xFF, dtype=np.uint8)  # everything +1 -> every B1 bit 1 ...
    expect[0] &= ~np.uint8(1 << 0)               # ... except the -1 columns, which are bit 0
    expect[1] &= ~np.uint8(1 << 1)               # col 9  = byte 1 bit 1
    expect[3] &= ~np.uint8(1 << 7)               # col 31 = byte 3 bit 7
    expect[4] &= ~np.uint8(1 << 0)               # col 32 = byte 4 bit 0
    expect[15] &= ~np.uint8(1 << 7)              # col 127
    np.testing.assert_array_equal(got[0], expect)
    assert got[0, 0] == 0xFE and got[0, 1] == 0xFD and got[0, 2] == 0xFF
    # the engine's own gguf-py quantizer agrees on the byte layout
    dense = np.where(signs > 0, 1.0, -1.0).astype(np.float32)
    np.testing.assert_array_equal(gguf.quants.quantize(dense, B1), got)


def test_pad_to():
    """Ranks and widths round up to the next multiple of 128 (one B1 block); exact multiples stay put."""
    assert [lb.pad_to(n) for n in (1, 128, 129, 136, 256)] == [128, 128, 256, 256, 256]


# --------------------------------------------------------------------------- the validator must catch corruption

@needs_qwen_tok
def test_validator_catches_corruption(qwen3_converted):
    """validate_layer passes on the honest GGUF tensors and fails on a flipped sign bit or a changed scale."""
    ckpt, out, report = qwen3_converted
    p = ref.load_checkpoint(ckpt)["model.layers.0.self_attn.q_proj"]
    t = _read(out)
    names = ("lb_v", "lb_u", "lb_s_v2", "lb_s_mid", "lb_s_u1", "lb_vr", "lb_ur", "lb_s_v2r", "lb_s_midr", "lb_s_u1r")
    good = {k: t[f"blk.0.attn_q.{k}"] for k in names}
    tol = val.tolerance(p)
    assert val.validate_layer(p, good) < tol

    def mutated(key, fn):
        """Copy of the good tensors where tensor `key` has had `fn` applied to a copy of its data."""
        d = dict(good)
        data = np.array(d[key].data, copy=True)
        fn(data)
        d[key] = d[key]._replace(data=data)
        return d

    def flip_sign(a):
        """Corruption: flip one sign bit (B1 bytes)."""
        a[3, 5] ^= 0x04  # one sign bit inside the logical rank

    def bump_scale(a):
        """Corruption: double one scale entry (F32 vector)."""
        a[7] *= 2.0  # one scale entry doubled

    for key, fn in (("lb_v", flip_sign), ("lb_u", flip_sign), ("lb_vr", flip_sign),
                    ("lb_s_mid", bump_scale), ("lb_s_v2", bump_scale), ("lb_s_u1", bump_scale)):
        err = val.validate_layer(p, mutated(key, fn))
        assert err > 10 * tol, (key, err, tol)
    # a missing residual tensor is an error too (the residual is part of the layer)
    d = {k: v for k, v in good.items() if not k.endswith("r")}
    assert val.validate_layer(p, d) > 10 * tol


# --------------------------------------------------------------------------- bias

@needs_qwen_tok
def test_bias_preserved_or_rejected(tmp_path):
    """Qwen3 with attention_bias=True: the biases must survive into the GGUF (or the converter must raise).

    Verified behaviour: the biases are plain non-LittleBit tensors, so the HF converter writes them
    as `blk.N.attn_{q,k,v,output}.bias` next to the factored weight.  Whether a given engine
    architecture then CONSUMES them is the loader's business (Qwen3's does not load them; see the
    report) -- the converter's contract is that it never drops one silently.
    """
    ck = _tiny.make_tiny_checkpoint("qwen3", tmp_path, attention_bias=True)
    # make the biases non-trivial so equality is meaningful (HF initialises them to zero)
    f = ck / "model.safetensors"
    t = load_file(str(f))
    g = torch.Generator().manual_seed(5)
    bias_keys = [k for k in t if k.endswith("_proj.bias")]
    assert bias_keys, "tiny model has no attention biases; fixture is wrong"
    for k in bias_keys:
        t[k] = torch.randn(t[k].shape, generator=g).to(torch.bfloat16)
    save_file(t, str(f), metadata={"format": "pt"})
    out = tmp_path / "o.gguf"
    report = lb.convert_checkpoint(ck, out)
    tensors = _read(out)
    gmap = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output"}
    for k in bias_keys:
        layer = int(k.split(".")[2])
        gname = f"blk.{layer}.{gmap[k.split('.')[-2]]}.bias"
        assert gname in tensors, f"bias {k} was dropped"
        np.testing.assert_array_equal(tensors[gname].data.astype(np.float32), t[k].float().numpy())
    assert sorted(report.biases) == sorted(bias_keys)


# --------------------------------------------------------------------------- families

FAMILY_CASES = [
    pytest.param("qwen3", "qwen3", id="qwen3", marks=needs_qwen_tok),
    pytest.param("llama", "llama", id="llama"),
    pytest.param("gemma3", "gemma3", id="gemma3"),
    pytest.param("qwen35", "qwen35", id="qwen35-deltanet", marks=needs_qwen_tok),
]


@pytest.mark.parametrize("family,arch", FAMILY_CASES)
def test_family_converts_and_validates(family, arch, tmp_path):
    """Tiny random model of the family -> checkpoint -> GGUF; every linear validated; probes are permutations.

    Gemma 3 only exercises the generic path (identity permutations expected); Llama (q/k RoPE
    rows) and Qwen3.5 (V-head reorder) exercise real permutations.
    """
    ck = _tiny.make_tiny_checkpoint(family, tmp_path)
    out = tmp_path / f"{family}.gguf"
    report = lb.convert_checkpoint(ck, out)
    reader = gguf.GGUFReader(str(out))
    assert reader.fields["general.architecture"].contents() == arch
    params = ref.load_checkpoint(ck)
    assert report.n_linears == len(params) == report.n_validated
    assert report.max_rel_err < report.tolerance
    names = {t.name for t in reader.tensors}
    assert sum(n.endswith(".lb_v") for n in names) == sum(len(v) for v in report.probes.values())
    for mod, probes in report.probes.items():
        assert probes, mod
        for pr in probes:  # pure permutation of (a subset of) rows / columns: no repeats
            assert len(set(pr.row_idx.tolist())) == len(pr.row_idx)
            assert len(set(pr.col_idx.tolist())) == len(pr.col_idx)
    if family == "llama":  # HF -> GGUF permutes q/k rows for RoPE; the probe must have seen it
        q = report.probes["model.layers.0.self_attn.q_proj"][0]
        assert not np.array_equal(q.row_idx, np.arange(len(q.row_idx)))
    if family == "qwen35":  # DeltaNet V-head reorder (num_k_heads != num_v_heads)
        z = report.probes["model.layers.0.linear_attn.in_proj_z"][0]
        assert not np.array_equal(z.row_idx, np.arange(len(z.row_idx)))
    # the head/embedding tensors came from the checkpoint's bf16 copies, as F16 (head_type f16)
    emb = next(t for t in reader.tensors if t.name == "token_embd.weight")
    assert emb.tensor_type == gguf.GGMLQuantizationType.F16


def test_head_type_other_than_f16_not_yet():
    """q8_0 / q6_k heads are not implemented yet; asking for one must fail with a clear NotImplementedError (not convert silently as f16)."""
    for h in ("q8_0", "q6_k"):
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            lb.convert(Path("nonexistent"), Path("x.gguf"), head_type=h)
    with pytest.raises(lb.ConversionError, match="head_type"):
        lb.convert(Path("nonexistent"), Path("x.gguf"), head_type="bogus")


def test_cli_main_is_importable():
    """`python -m tools.convert.lb_to_gguf <ckpt> -o <out>` is argparse-based; --help exits 0."""
    with pytest.raises(SystemExit) as e:
        lb.main(["--help"])
    assert e.value.code == 0
