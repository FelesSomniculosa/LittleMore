"""Tests for the lean PyTorch LittleBit reference (tools/dev/reference.py).

Run with `pytest tests/dev`.  Terms are defined in docs/development.md.
"""
import json
import os
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
        """Random tensors for one LittleBit path (signs, scales) of the given shape."""
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


# ------------------------------------------------------------------ memory-lean variants (for 8B models)

def test_lazy_checkpoint_matches_load_checkpoint(tmp_path):
    """LazyCheckpoint gives the same modules, residual flags and tensors as load_checkpoint, one module per lookup."""
    g = torch.Generator().manual_seed(11)
    t = {**_tensors("a.proj", 6, 10, 4, g), **_tensors("a.proj", 6, 10, 4, g, "_R")}
    _write(tmp_path, [t, _tensors("b.proj", 6, 10, 4, g)])
    eager, lazy = ref.load_checkpoint(tmp_path), ref.LazyCheckpoint(tmp_path)
    assert list(lazy) == list(eager) and len(lazy) == 2
    assert lazy.has_residual("a.proj") and not lazy.has_residual("b.proj")
    for mod, p in eager.items():
        q = lazy[mod]
        for path_e, path_l in ((p.primary, q.primary), (p.residual, q.residual)):
            if path_e is None:
                assert path_l is None
                continue
            for f in ("U", "V", "u1", "u2", "v1", "v2"):
                assert torch.equal(getattr(path_e, f), getattr(path_l, f))


def test_lazy_checkpoint_membership_reads_nothing(tmp_path, monkeypatch):
    """`mod in lazy` answers from the index: no module is read (the converter asks it for every checkpoint key;
    reading the module each time made an 8B conversion 90 s slower)."""
    g = torch.Generator().manual_seed(14)
    _write(tmp_path, [_tensors("a.proj", 6, 10, 4, g)])
    lazy = ref.LazyCheckpoint(tmp_path)
    calls = []
    monkeypatch.setattr(ref, "_build_path", lambda *a: calls.append(a))
    assert "a.proj" in lazy and "b.proj" not in lazy and None not in lazy
    assert calls == []


def _private_commit_bytes() -> int:
    """This process's private commit charge (Win32 GetProcessMemoryInfo, PagefileUsage)."""
    import ctypes
    import ctypes.wintypes as W

    class _Counters(ctypes.Structure):
        """ctypes mirror of the Win32 PROCESS_MEMORY_COUNTERS struct."""
        _fields_ = [("cb", W.DWORD), ("PageFaultCount", W.DWORD)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                           "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                           "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    c = _Counters()
    c.cb = ctypes.sizeof(c)
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = W.HANDLE
    ctypes.WinDLL("psapi").GetProcessMemoryInfo(W.HANDLE(k32.GetCurrentProcess()), ctypes.byref(c), c.cb)
    return int(c.PagefileUsage)


@pytest.mark.skipif(os.name != "nt", reason="measures the Windows commit charge")
def test_lazy_checkpoint_values_do_not_pin_the_file(tmp_path):
    """Kept module values must not pin a private mapping of the whole checkpoint file.

    safetensors returns views into a private mapping of the whole file, one per `safe_open`; on Windows each one is
    charged in full against the commit limit while any view lives (measured: +2.77 GiB per kept tensor of the 2.97 GB
    8B checkpoint, which crashed an 8B reference build).  Here: a 64 MiB file, 8 modules kept; without the copies the
    charge grows by about 8 x 64 MiB, with them by almost nothing.
    """
    g = torch.Generator().manual_seed(15)
    t = {}
    for i in range(8):
        t.update(_tensors(f"m{i}.proj", 6, 10, 4, g))
    t["filler.weight"] = torch.zeros(16 * 2**20)            # 64 MiB of plain tensor: makes the mapping big
    _write(tmp_path, [t])
    lazy = ref.LazyCheckpoint(tmp_path)
    before = _private_commit_bytes()
    kept = [lazy[f"m{i}.proj"] for i in range(8)]
    grown = _private_commit_bytes() - before
    assert len(kept) == 8 and grown < 64 * 2**20, f"commit grew by {grown / 2**20:.0f} MiB for 8 small modules"


def test_lazy_checkpoint_rank_reads_only_the_shape(tmp_path, monkeypatch):
    """`rank(mod)` is the column count of U, from the `U_shape` tensor alone (no module is unpacked)."""
    g = torch.Generator().manual_seed(16)
    _write(tmp_path, [_tensors("a.proj", 6, 10, 4, g)])
    lazy = ref.LazyCheckpoint(tmp_path)
    expected = lazy["a.proj"].primary.U.shape[1]
    monkeypatch.setattr(ref, "_build_path", lambda *a: pytest.fail("rank() unpacked the module"))
    assert lazy.rank("a.proj") == expected == 4


def test_lazy_checkpoint_errors(tmp_path):
    """Missing tensors fail when indexing (names only); a shape mismatch fails when the module is looked up."""
    g = torch.Generator().manual_seed(12)
    t = _tensors("a.proj", 6, 10, 4, g)
    del t["a.proj.v2"]
    _write(tmp_path / "missing", [t])
    with pytest.raises(ValueError, match="missing tensors"):
        ref.LazyCheckpoint(tmp_path / "missing")
    t = _tensors("a.proj", 6, 10, 4, g)
    t["a.proj.V_shape"] = torch.tensor([4, 100], dtype=torch.int32)
    _write(tmp_path / "shape", [t])
    lazy = ref.LazyCheckpoint(tmp_path / "shape")   # names are fine
    with pytest.raises(ValueError, match="shape"):
        lazy["a.proj"]


@pytest.mark.parametrize("residual", [False, True])
def test_packed_lblinear_is_bit_identical(residual):
    """packed=True stores 1 bit per sign (int32 words) and computes exactly the same output as the int8 storage."""
    g = torch.Generator().manual_seed(13)
    p, res = _params(24, 40, 9, g, residual=residual)
    names = ("U", "V", "u1", "u2", "v1", "v2")
    a = _make(p, res)
    b = ref.LBLinear(*(p[n] for n in names), residual=None if res is None else tuple(res[n] for n in names),
                     packed=True)
    assert b.packed and b.U_0.dtype == torch.int32 and b.U_0.shape == (24, 1)   # 9 signs per row -> one word
    assert b.V_0.shape == (9, 2) and a.U_0.dtype == torch.int8                 # 40 signs per row -> two words
    x = torch.randn(5, 40, generator=g)
    assert torch.equal(a(x), b(x))


def test_build_model_packed_and_lean_float32_match_float(tmp_path):
    """build_model(packed_signs=True) + lean_float32 gives the same logits as build_model(...).float() (fp32).

    The head is applied in blocks (Bf16StoredLinear), so the sum order inside the head matmul may differ: the
    comparison allows float32 rounding (1e-5 relative), far below any real error (a wrong scale moves logits by %).
    """
    base_dir = _tiny_base(tmp_path)
    ck = _tiny_ckpt(tmp_path)
    full = ref.build_model(str(base_dir), ck).float().eval()
    lean = ref.lean_float32(ref.build_model(str(base_dir), ck, packed_signs=True)).eval()
    assert lean.model.layers[0].mlp.up_proj.packed and lean.lm_head.weight.dtype == torch.bfloat16
    assert lean.get_input_embeddings().weight.dtype == torch.bfloat16
    assert lean.model.norm.weight.dtype == torch.float32
    ids = torch.tensor([[1, 2, 3, 5, 8, 13]])
    with torch.no_grad():
        a, b = full(ids).logits, lean(ids).logits
    assert b.dtype == torch.float32
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-6), (a - b).abs().max()
    lean.lm_head.rows = 7   # blocks that do not divide the vocabulary (64) still cover every row
    with torch.no_grad():
        assert torch.allclose(lean(ids).logits, b, rtol=1e-5, atol=1e-6)
