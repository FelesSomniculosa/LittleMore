"""Converter tests on: any rank (padding), a missing/overridden littlebit_config.json, byte-level sign polarity, the exact check, the memory preflight."""
from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

import _tiny
import reference as ref
from tools.convert import lb_to_gguf as lb
from tools.convert import validate as val

import gguf  # noqa: E402

needs_qwen_tok = pytest.mark.skipif(not _tiny.qwen_tokenizer_available(), reason="Qwen3-0.6B tokenizer not cached")


def _read(path: Path) -> dict:
    return {t.name: t for t in gguf.GGUFReader(str(path)).tensors}


def _set_rank(ckpt: Path, new_r: int) -> None:
    """Truncate EVERY LittleBit linear of `ckpt` (both paths) to logical rank `new_r`.

    Real checkpoints may have any rank >= 1; the maker only makes multiples of 8, so the
    factors are cut here and re-packed with reference.pack_signs.
    """
    f = ckpt / "model.safetensors"
    t = load_file(str(f))
    mods = [k[: -len(".U_packed")] for k in t if k.endswith(".U_packed")]
    for m in mods:
        for res in ("", "_R"):
            if f"{m}.U{res}_packed" not in t:
                continue
            U = ref.unpack_signs(t[f"{m}.U{res}_packed"], t[f"{m}.U{res}_shape"].tolist())[:, :new_r]
            V = ref.unpack_signs(t[f"{m}.V{res}_packed"], t[f"{m}.V{res}_shape"].tolist())[:new_r]
            t[f"{m}.U{res}_packed"], t[f"{m}.U{res}_shape"] = ref.pack_signs(U), torch.tensor(U.shape)
            t[f"{m}.V{res}_packed"], t[f"{m}.V{res}_shape"] = ref.pack_signs(V), torch.tensor(V.shape)
            for s in ("u2", "v1"):
                t[f"{m}.{s}{res}"] = t[f"{m}.{s}{res}"][:, :new_r].contiguous()
    save_file(t, str(f), metadata={"format": "pt"})


@pytest.fixture(scope="module")
def q3(tmp_path_factory):
    if not _tiny.qwen_tokenizer_available():
        pytest.skip("Qwen3-0.6B tokenizer not cached")
    return _tiny.make_tiny_checkpoint("qwen3", tmp_path_factory.mktemp("q3r"))


# ------------------------------------------------------------------ 1: ranks 8 and 37

@needs_qwen_tok
@pytest.mark.parametrize("r", [8, 37])
def test_tiny_and_odd_rank_pad_exactly(q3, tmp_path, r):
    """Logical rank 8 and the odd rank 37: padded to 128, mid == 0 beyond r, output equals an UNPADDED float64 evaluation."""
    ck = tmp_path / "ck"
    shutil.copytree(q3, ck)
    _set_rank(ck, r)
    params = ref.load_checkpoint(ck)
    assert all(p.primary.V.shape[0] == r and p.residual.V.shape[0] == r for p in params.values())
    out = tmp_path / "o.gguf"
    report = lb.convert_checkpoint(ck, out)
    assert report.max_rel_err < 1e-6  # ~1e-7-level, far below the 1e-5 tolerance
    tensors = _read(out)
    rng = np.random.default_rng(3)
    for mod, p in params.items():
        for pr in report.probes[mod]:
            base = pr.gguf_name.removesuffix(".weight")
            x = rng.standard_normal((3, p.primary.V.shape[1])).astype(np.float32)
            y_pad, y_unpad = 0, 0
            for sfx, path in (("", p.primary), ("r", p.residual)):
                g = {k: tensors[f"{base}.{k}{sfx}"] for k in ("lb_v", "lb_u", "lb_s_v2", "lb_s_mid", "lb_s_u1")}
                mid = np.asarray(g["lb_s_mid"].data)
                assert mid.shape == (128,) and np.all(mid[r:] == 0) and np.all(mid[:r] != 0)
                # padded evaluation straight from the written bytes (full 128 ranks)
                V = np.where(np.unpackbits(g["lb_v"].data, axis=-1, bitorder="little") == 1, 1.0, -1.0)
                U = np.where(np.unpackbits(g["lb_u"].data, axis=-1, bitorder="little") == 1, 1.0, -1.0)
                xg = x[:, pr.col_idx].astype(np.float64)
                y_pad = y_pad + (((xg * g["lb_s_v2"].data) @ V.T * mid) @ U.T) * g["lb_s_u1"].data
                # unpadded float64 evaluation from the ORIGINAL checkpoint, logical rank only
                Vl, Ul = path.V.numpy().astype(np.float64), path.U.numpy().astype(np.float64)
                midl = (path.v1.float() * path.u2.float()).numpy().astype(np.float64).reshape(-1)  # f32 fold, as stored
                xs = x.astype(np.float64) * path.v2.float().numpy().astype(np.float64).reshape(1, -1)
                yy = ((xs @ Vl.T * midl) @ Ul.T) * path.u1.float().numpy().astype(np.float64).reshape(1, -1)
                y_unpad = y_unpad + yy[:, pr.row_idx]
            assert np.abs(y_pad - y_unpad).max() <= 1e-9 * np.abs(y_unpad).max(), mod


# ------------------------------------------------------------------ 2: config override

@needs_qwen_tok
def test_missing_littlebit_config_message_and_override(q3, tmp_path, caplog):
    """Default: refuse with a helpful message; with the override: convert (residual detected from tensors)."""
    ck = tmp_path / "ck"
    shutil.copytree(q3, ck)
    (ck / "littlebit_config.json").unlink()
    with pytest.raises(lb.ConversionError) as e:
        lb.convert(ck, tmp_path / "o.gguf")
    msg = str(e.value)
    assert "--assume-no-littlebit-config" in msg and "assume_no_littlebit_config=True" in msg
    assert "written by the LittleBit" in msg and not (tmp_path / "o.gguf").exists()
    with caplog.at_level("WARNING", logger="littlemore.convert"):
        report = lb.convert_checkpoint(ck, tmp_path / "o.gguf", assume_no_littlebit_config=True)
    assert report.config_check_skipped and report.n_validated == report.n_linears
    assert "cross-check skipped" in caplog.text
    assert any(n.endswith(".lb_vr") for n in _read(tmp_path / "o.gguf"))  # residual found from the tensors


@needs_qwen_tok
def test_override_without_residual_and_cli_flag(tmp_path):
    """Override also works for a no-residual checkpoint (no residual tensors written); CLI flag is wired."""
    ck = _tiny.make_tiny_checkpoint("qwen3", tmp_path, residual=False)
    (ck / "littlebit_config.json").unlink()
    assert lb.main([str(ck), "-o", str(tmp_path / "x.gguf")]) == 1  # default refuses (exit code 1)
    assert lb.main([str(ck), "-o", str(tmp_path / "x.gguf"), "--assume-no-littlebit-config"]) == 0
    names = _read(tmp_path / "x.gguf")
    assert any(n.endswith(".lb_v") for n in names) and not any(n.endswith(".lb_vr") for n in names)


# ------------------------------------------------------------------ 3: byte-level polarity end to end

@needs_qwen_tok
def test_written_bytes_are_complement_of_samsung_bytes(q3, tmp_path, monkeypatch):
    """Identity layer, d_in=256, r=128: GGUF lb_v/lb_u bytes == bitwise complement of the Samsung packed bytes."""
    ck = tmp_path / "ck"
    shutil.copytree(q3, ck)
    _set_rank(ck, 128)
    out = tmp_path / "o.gguf"
    report = lb.convert_checkpoint(ck, out)
    mod = "model.layers.0.mlp.up_proj"
    (pr,) = report.probes[mod]
    assert np.array_equal(pr.row_idx, np.arange(len(pr.row_idx))) and np.array_equal(pr.col_idx, np.arange(len(pr.col_idx)))
    st = load_file(str(ck / "model.safetensors"))
    tensors = _read(out)
    base = pr.gguf_name.removesuffix(".weight")
    expected = {}
    for sfx, res in (("", ""), ("r", "_R")):
        for k, key in (("lb_v", f"{mod}.V{res}_packed"), ("lb_u", f"{mod}.U{res}_packed")):
            sam = st[key].numpy().view(np.uint8)  # int32 words, little endian: byte i/8 bit i%8 == element i
            got = np.asarray(tensors[f"{base}.{k}{sfx}"].data)
            assert got.shape == sam.shape == (st[key].shape[0], st[key].shape[1] * 4)
            np.testing.assert_array_equal(got, ~sam)
            expected[(k, sfx)] = ~sam
    # A test that fails when the code is wrong: a packer that forgets the flip writes the Samsung bytes themselves, which this test rejects.
    # (Validation alone cannot see a flip applied to U AND V together: the two sign errors cancel.)
    monkeypatch.setattr(lb, "signs_to_b1", lambda s: np.packbits(s < 0, axis=-1, bitorder="little"))
    out2 = tmp_path / "o2.gguf"
    lb.convert_checkpoint(ck, out2, validate=False)
    t2 = _read(out2)
    assert not np.array_equal(np.asarray(t2[f"{base}.lb_v"].data), expected[("lb_v", "")])
    np.testing.assert_array_equal(np.asarray(t2[f"{base}.lb_v"].data), st[f"{mod}.V_packed"].numpy().view(np.uint8))


# ------------------------------------------------------------------ 5: float64 exact check

def _synthetic(d_in: int, d_out: int, r: int, seed: int = 0):
    """Random LBLinearParams and the GGUF-style tensors the converter would write for it (identity permutation)."""
    g = torch.Generator().manual_seed(seed)
    sgn = lambda *s: (torch.randint(0, 2, s, generator=g) * 2 - 1).to(torch.int8)
    sc = lambda n: (torch.rand(1, n, generator=g) + 0.5)
    path = ref.LBPathParams(sgn(d_out, r), sgn(r, d_in), sc(d_out), sc(r), sc(r), sc(d_in))
    params = ref.LBLinearParams(path)
    arrays = lb._build_path(path, np.arange(d_out), np.arange(d_in), "synthetic")
    tensors = {k: SimpleNamespace(data=v, shape=(v.shape[1] * 8, v.shape[0]) if k in ("lb_v", "lb_u") else (v.shape[0],))
               for k, v in arrays.items()}
    return params, tensors


def test_float64_exact_check_catches_single_bit_where_float32_margin_is_small():
    """At 4096x4096 rank 4096 one flipped sign is only a few times above the float32 tolerance, but ~1e5x above the exact one."""
    params, tensors = _synthetic(4096, 64, 4096)
    tol = val.tolerance(params)
    assert val.validate_layer(params, tensors) < tol
    assert val.validate_layer_exact(params, tensors) < val.EXACT_TOL
    data = np.array(tensors["lb_v"].data, copy=True)
    data[100, 7] ^= 0x10  # one sign bit inside the logical rank
    bad = dict(tensors, lb_v=SimpleNamespace(data=data, shape=tensors["lb_v"].shape))
    e32, e64 = val.validate_layer(params, bad), val.validate_layer_exact(params, bad)
    print(f"flipped bit: float32 err {e32:.2e} (tol {tol:.2e}, margin {e32 / tol:.1f}x); float64 err {e64:.2e} (tol {val.EXACT_TOL:.0e}, margin {e64 / val.EXACT_TOL:.1e}x)")
    assert e32 > tol and e64 > 1e3 * val.EXACT_TOL
    assert e64 / val.EXACT_TOL > 100 * (e32 / tol)


# ------------------------------------------------------------------ 7, 4: cleanup and memory

@needs_qwen_tok
def test_unlink_failure_does_not_mask_original_error(q3, tmp_path, monkeypatch):
    """If removing the temp file fails (Windows mmap), the real ConversionError still propagates."""
    monkeypatch.setattr(lb.validate, "validate_layer", lambda *a, **k: 1.0)
    real_unlink = Path.unlink

    def bad_unlink(self, *a, **k):
        if self.name.endswith(".partial"):
            raise PermissionError("simulated mmap lock")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", bad_unlink)
    with pytest.raises(lb.ConversionError, match="tolerance"):
        lb.convert(q3, tmp_path / "o.gguf")
    monkeypatch.undo()
    for f in tmp_path.glob("*.partial"):
        f.unlink()


@needs_qwen_tok
def test_memory_preflight(q3, tmp_path, monkeypatch):
    """Not enough free RAM -> ConversionError with the estimate (no raw MemoryError); enough -> converts."""
    est = lb._estimate_bytes(q3)
    assert est > 1e8  # dominated by the tiny model's 151936 x 256 embedding
    monkeypatch.setattr(lb, "_available_memory", lambda: est // 2)
    with pytest.raises(lb.ConversionError, match=r"estimated \d+\.\d+ GiB.*available"):
        lb.convert(q3, tmp_path / "o.gguf")
    assert not (tmp_path / "o.gguf").exists()
    monkeypatch.setattr(lb, "_available_memory", lambda: None)  # unknown -> skipped
    lb.convert(q3, tmp_path / "o.gguf")


def test_estimate_scales_with_signs(tmp_path):
    """The estimate: 1/8 byte per sign (B1 copy) + 1 byte per sign of the LARGEST module only (the int8 copy of the
    module being converted; modules are read one at a time) + 8 bytes per element of the largest other tensor."""
    save_file({"m.U_packed": torch.zeros(10, 4, dtype=torch.int32)}, str(tmp_path / "model.safetensors"))
    assert lb._estimate_bytes(tmp_path) == int(10 * 4 * 32 * 1.125)
    two = tmp_path / "two"
    two.mkdir()
    save_file({"a.U_packed": torch.zeros(10, 4, dtype=torch.int32), "a.V_packed": torch.zeros(2, 4, dtype=torch.int32),
               "b.U_packed": torch.zeros(10, 4, dtype=torch.int32),
               "embed.weight": torch.zeros(100), "norm.weight": torch.zeros(50)}, str(two / "model.safetensors"))
    signs_a, signs_b = (10 + 2) * 4 * 32, 10 * 4 * 32
    assert lb._estimate_bytes(two) == int((signs_a + signs_b) / 8 + signs_a + 100 * 8)
