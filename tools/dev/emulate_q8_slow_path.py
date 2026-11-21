"""Emulate llama.cpp's Q1_0 x Q8_0 activation noise on top of reference.py.

Why this exists
---------------
llama.cpp evaluates a LittleBit linear as
    mul(v2) -> mul_mat(Q1_0 V) -> mul(v1*u2) -> mul_mat(Q1_0 U) -> mul(u1)
(one such chain per path; two paths for LittleBit-2).  A Q1_0 weight times an
F32 activation is computed by first quantizing the activation to Q8_0 (ggml's
`vec_dot_type` for Q1_0; the CUDA backend likewise quantizes to int8 blocks),
then doing an integer dot product.  `reference.py` keeps activations in fp32, so
the two differ by int8 activation noise injected at every binary matmul.

This module reproduces exactly that one source of error in PyTorch so the
engine can be compared against a reference that has the *same* noise model.  If
the engine matches the emulation to near fp precision while sitting far from the
plain fp32 reference, the gap is fully explained by Q8_0 activation
quantization and the conversion/graph are structurally right.

Noise model (mirrors ggml `quantize_row_q8_0_ref`, ggml-quants.c)
-----------------------------------------------------------------
For each block of 32 consecutive elements of the activation row:
    d  = max|x| / 127,  id = 1/d  (0 if d == 0)
    q  = round_half_away_from_zero(x * id)        # int8 in [-127, 127]
    x' = q * fp16(d)                              # the scale is stored in fp16
Applied to the vector entering each binary matmul: (x * v2) before V, and
((.) * v1*u2) before U.  The rank dimension is zero-padded (as the converter
pads it to a multiple of 128).  This emulation only pads to a multiple of 32
(its block size), which is equivalent: the padded entries of the middle scale
v1*u2 are zero, so the padded values entering the second matmul are zero and a
block's max |x| never grows; padding amount therefore does not matter.
Everything else (norms, softmax, scale vectors) stays fp32, except the KV
cache: the engine stores K (after RoPE) and V in f16 and feeds Q to flash
attention as f16, so q/k/v are rounded through f16 (`kv_f16=True`).  This is NOT
negligible: nudging V by ~5e-4 relative flips Q8_0 rounding decisions in the next
(o_proj) matmul, which then shows up as noise of the same order as Q8_0 itself.
Verified on layer 0 of the smoke model: with f16-rounded V the emulated o_proj
first binary matmul output equals the engine's to 6 digits (-42.98644 vs
-42.98641); without it the match is only ~3 digits.

Mutations (for tests that prove a check fails when the code is wrong)
-----------------------------------------------------------
`drop_residual` removes the LittleBit-2 residual path; `permute_qk` applies a
wrong RoPE-style row permutation (interleaved <-> half-split within each head)
to q_proj / k_proj outputs.  Both should make the engine agree clearly worse.
"""
from __future__ import annotations

import types

import numpy as np
import torch
from torch import Tensor, nn

import reference as R

_BLOCK = 32


def q8_0_roundtrip(x: Tensor) -> Tensor:
    """Quantize the last dim to ggml Q8_0 (blocks of 32) and dequantize, in fp32."""
    x = x.float()
    n = x.shape[-1]
    pad = (-n) % _BLOCK
    xp = nn.functional.pad(x, (0, pad)) if pad else x
    blocks = xp.reshape(*xp.shape[:-1], -1, _BLOCK)
    d = blocks.abs().amax(-1, keepdim=True) / 127.0
    id_ = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    v = blocks * id_
    q = torch.sign(v) * torch.floor(v.abs() + 0.5)      # C roundf: ties away from zero
    out = (q * d.half().float()).reshape(xp.shape)       # scale stored as fp16
    return out[..., :n]


def _interleave_perm(n: int, head_dim: int) -> Tensor:
    """Row permutation interleaved<->half-split inside each head (a wrong-RoPE mutation)."""
    idx = torch.arange(n).reshape(-1, head_dim)
    half = torch.cat([idx[:, 0::2], idx[:, 1::2]], dim=1)
    return half.reshape(-1)


def _forward(self: R.LBLinear, x: Tensor) -> Tensor:
    """LBLinear.forward with optional Q8_0 activation noise and mutations."""
    xf = x.to(torch.float32)
    q = q8_0_roundtrip if self._emu_quant else (lambda t: t)
    y = 0
    n_paths = 1 if self._emu_drop_residual else self.n_paths
    for i in range(n_paths):
        U, V, u1, u2, v1, v2 = (getattr(self, f"{n}_{i}").float() for n in R._NAMES)
        h = (q(xf * v2) @ V.T) * (v1 * u2)
        y = y + (q(h) @ U.T) * u1
    if self._emu_perm is not None:
        y = y[..., self._emu_perm]
    if self._emu_f16_out:
        y = _f16(y)
    return y.to(x.dtype)


def _f16(t: Tensor) -> Tensor:
    """Round to half precision (fp16) and back to fp32: emulates the engine's f16 storage."""
    return t.half().float()


_ORIG_ROPE = None


def _patch_rope(enable: bool) -> None:
    """Round q and k (post-RoPE) through f16 to mimic the engine's f16 K cache / FA Q.

    Qwen3 only: it replaces `apply_rotary_pos_emb` in transformers' Qwen3 module,
    a process-wide global, so it affects every Qwen3 model in this process until
    called again with enable=False (emulate() calls it on every use).  RoPE is
    rotary position embedding, see docs/development.md.
    """
    global _ORIG_ROPE
    from transformers.models.qwen3 import modeling_qwen3 as mq
    if _ORIG_ROPE is None:
        _ORIG_ROPE = mq.apply_rotary_pos_emb
    if enable:
        def rope(q, k, *a, **kw):
            q, k = _ORIG_ROPE(q, k, *a, **kw)
            return _f16(q).to(q.dtype), _f16(k).to(k.dtype)
        mq.apply_rotary_pos_emb = rope
    else:
        mq.apply_rotary_pos_emb = _ORIG_ROPE


def emulate(model: nn.Module, quantize: bool = True, drop_residual: bool = False,
            permute_qk: bool = False, head_dim: int = 128, kv_f16: bool = True) -> nn.Module:
    """Patch every LBLinear in `model` in place (returns it); see module docstring."""
    _patch_rope(kv_f16)
    for name, mod in model.named_modules():
        if isinstance(mod, R.LBLinear):
            mod._emu_quant = quantize
            mod._emu_drop_residual = drop_residual
            mod._emu_perm = (_interleave_perm(mod.out_features, head_dim)
                             if permute_qk and name.endswith(("q_proj", "k_proj")) else None)
            mod._emu_f16_out = kv_f16 and name.endswith("v_proj")  # V cache is f16
            mod.forward = types.MethodType(_forward, mod)
    return model


@torch.no_grad()
def last_logits(model: nn.Module, token_ids: list[int]) -> np.ndarray:
    """Last-position logits (fp32 numpy) for `token_ids` on CPU."""
    out = model(torch.tensor([token_ids], dtype=torch.long))
    return out.logits[0, -1].float().numpy()
