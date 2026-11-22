"""Real-checkpoint compatibility: the cached Qwen3-0.6B init-only checkpoint -> B1 GGUF.

Marked `smoke` (slow-ish, needs cached assets; skipped when absent).  Checks that the
converter handles a real-size Samsung-format checkpoint (28 layers, 196 LittleBit linears,
residual paths, GQA), that per-layer validation passes, and that B1 really stores the
sign matrices at exactly 1.0 bit per sign, i.e. 16/18 of the Q1_0 slow-path file's bytes
for the same tensors (Q1_0 = 16 bytes of signs + a 2-byte unused scale per 128 signs = 1.125 bpw).
"""
from __future__ import annotations

import pytest

import _assets as A
import _tiny
from tools.convert import lb_to_gguf as lb

import gguf

pytestmark = pytest.mark.smoke

Q1_0_GGUF = A.LB_GGUF
LB_SIGN_SUFFIXES = (".lb_v", ".lb_u", ".lb_vr", ".lb_ur")


def _sign_tensor_stats(path):
    """(bytes, number of signs) summed over all LittleBit sign tensors of a GGUF."""
    nbytes = nsigns = 0
    for t in gguf.GGUFReader(str(path)).tensors:
        if t.name.endswith(LB_SIGN_SUFFIXES):
            nbytes += int(t.n_bytes)
            nsigns += int(t.shape[0]) * int(t.shape[1])
    return nbytes, nsigns


def test_real_checkpoint_converts_validates_and_is_one_bit(tmp_path):
    """The real Qwen3-0.6B checkpoint converts, every one of its 196 linears validates, and the sign
    tensors take exactly 1 bit per sign (a layout or padding mistake would change the byte count)."""
    if not (A.CKPT_DIR / "model.safetensors").exists():
        pytest.skip(f"Qwen3-0.6B LittleBit checkpoint not cached at {A.CKPT_DIR} (built by tests/smoke/_assets.ensure_gguf)")
    if not _tiny.qwen_tokenizer_available():
        pytest.skip("Qwen3-0.6B tokenizer files missing")
    out = tmp_path / "real-b1.gguf"
    report = lb.convert_checkpoint(A.CKPT_DIR, out)
    assert report.n_linears == 28 * 7 and report.n_validated == report.n_linears
    assert report.max_rel_err < report.tolerance
    b1_bytes, b1_signs = _sign_tensor_stats(out)
    assert b1_bytes * 8 == b1_signs, "B1 must be exactly 1.0 bit per sign"
    print(f"\nB1 sign tensors: {b1_bytes} bytes for {b1_signs} signs ({b1_bytes * 8 / b1_signs:.4f} bpw); "
          f"max rel err {report.max_rel_err:.2e}, tolerance {report.tolerance:.2e}, file {out.stat().st_size} bytes")
    if Q1_0_GGUF.exists():
        q_bytes, q_signs = _sign_tensor_stats(Q1_0_GGUF)
        assert q_signs == b1_signs, "same padded sign matrices expected in both files"
        print(f"Q1_0 sign tensors: {q_bytes} bytes ({q_bytes * 8 / q_signs:.4f} bpw); B1/Q1_0 = {b1_bytes / q_bytes:.4f}")
        assert abs(b1_bytes / q_bytes - 16 / 18) < 1e-9
