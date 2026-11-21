"""Init-only LittleBit-2 checkpoint maker (dev tool).

What: builds a LittleBit / LittleBit-2 style checkpoint from any HF causal LM by
      running only the *initialisation* of the method -- exact SVD, optional
      Joint-ITQ rotation, Dual-SVID scale fitting, sign factors -- with no
      quantisation-aware training.
Why:  no trained LittleBit-2 checkpoints are published.  Init-only ones produce
      gibberish text, but they are exact for correctness and speed testing of the
      runtime (reference model, GGUF converter, engine kernels): the file format
      and the arithmetic are identical to a trained checkpoint.

Algorithm (LittleBit-2, arXiv:2603.00042, Algorithm 1; written from the
published description only):
  1. W = U S V^T (exact).  U^ = U[:, :r] sqrt(S), V^ = V[:, :r] sqrt(S).  The
     singular values are split evenly (sqrt on each side) so both factors have
     comparable scale; rows are NOT normalised.
  2. Joint-ITQ: stack Z = [U^; V^] and find an orthogonal r x r rotation R that
     makes sign(Z R) a good 1-bit code of Z R (alternate B = sign(Z R) and the
     orthogonal Procrustes solution).  Rotating both factors by the same R keeps
     the product: (U^ R)(V^ R)^T = U^ V^^T.
  3. Dual-SVID: signs Ub = sign(U~), Vb = sign(V~)^T; the magnitudes |U~|, |V~|
     are approximated by rank-1 outer products, giving scale vectors
     h (out), g (in) and the rank-space scales urk, vrk.
     W ~ diag(h) Ub diag(urk * vrk) Vb diag(g).
  4. Residual path: repeat on W - W_primary, summed at inference.

Checkpoint layout (read by `reference.load_checkpoint`): per linear module
`<name>.U_packed/.U_shape/.V_packed/.V_shape` (+ `.u1 .u2 .v1 .v2`) and the
residual variants `<name>.U_R_packed/.U_R_shape/.V_R_packed/.V_R_shape/.u1_R
.u2_R .v1_R .v2_R`.  Signs are bit-packed int32 (see `reference.pack_signs`),
scales float32.  Non-LittleBit tensors (embeddings, norms, lm_head) are copied
in bf16 into the same file so the checkpoint directory is self-contained.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import torch
from torch import Tensor

from reference import pack_signs

_LAYER_RE = re.compile(r"(^|\.)layers\.\d+\.")
_SIDE_FILES = ("config.json", "generation_config.json", "tokenizer.json",
               "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
               "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja")


def _svd(W: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Exact thin SVD in float32; GPU when available and it fits, else CPU."""
    W = W.float()
    if torch.cuda.is_available():
        try:
            U, S, Vt = torch.linalg.svd(W.cuda(), full_matrices=False)
            return U.cpu(), S.cpu(), Vt.cpu()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
    return torch.linalg.svd(W.cpu(), full_matrices=False)


def _sign(x: Tensor) -> Tensor:
    """+/-1 sign with sign(0) -> +1 (0 would otherwise be unrepresentable in 1 bit)."""
    return torch.where(x >= 0, 1.0, -1.0).to(x.dtype)


def joint_itq(Z: Tensor, n_iter: int = 50, seed: int = 0) -> Tensor:
    """Joint-ITQ: orthogonal R (r x r) such that sign(Z R) approximates Z R well.

    Z is the row-stacked factor matrix [U^; V^] of shape (out+in, r).  Starts from
    a seeded random orthogonal R, then alternates B = sign(Z R) and the Procrustes
    update: SVD(B^T Z) = Phi Omega Psi^T, R = Psi Phi^T.
    """
    r = Z.shape[1]
    gen = torch.Generator().manual_seed(seed)
    R = torch.empty(r, r)
    torch.nn.init.orthogonal_(R, generator=gen)
    Zf, R = Z.float(), R.to(Z.device)
    for _ in range(n_iter):
        B = _sign(Zf @ R)
        Phi, _, Psit = torch.linalg.svd(B.T @ Zf)
        R = Psit.T @ Phi.T
    return R


def _rank1_abs(M: Tensor) -> tuple[Tensor, Tensor]:
    """Rank-1 fit |M| ~ a b^T with a, b >= 0 (sqrt of the top singular value on each)."""
    U, S, Vt = _svd(M.abs())
    a, b = U[:, 0], Vt[0]
    if a.sum() < 0:  # |M| >= 0 so the top singular pair is single-signed; fix the flip
        a, b = -a, -b
    s = S[0].sqrt()
    return a * s, b * s


def _dual_svid_parts(U: Tensor, V: Tensor):
    Ub = _sign(U.float())
    Vb = _sign(V.float()).T.contiguous()
    h, urk = _rank1_abs(U)
    g, vrk = _rank1_abs(V)
    return Ub, Vb, h, g, urk, vrk


def dual_svid(U: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Dual-SVID of factors U (out, r), V (in, r).

    Returns (Ub (out,r), Vb (r,in), h (out), g (in), l (r)) with +/-1 signs and
    W ~ diag(h) Ub diag(l) Vb diag(g), where l is the product of the two
    rank-space scales.
    """
    Ub, Vb, h, g, urk, vrk = _dual_svid_parts(U, V)
    return Ub, Vb, h, g, urk * vrk


def rank_for_bits(d_out: int, d_in: int, bpw: float, residual: bool = True) -> int:
    """Largest rank (multiple of 8, min 8) whose storage fits `bpw` bits/weight.

    Two paths: bits = 32(d_in+d_out) + 2r(d_in+d_out+16) (16-bit scales, 1-bit
    signs, the two rank-space scales counted as one folded vector).  One path:
    16(d_in+d_out) + r(d_in+d_out+16).  Rounded down to a multiple of 8 (kernel
    alignment).  If the formula gives r < 8 the rank is padded up to 8, so for
    tiny matrices the real bpw can exceed the target.
    """
    n, s = d_in * d_out, d_in + d_out
    if residual:
        r = int((bpw * n - 32 * s) // (2 * (s + 16)))
    else:
        r = int((bpw * n - 16 * s) // (s + 16))
    return max(8, r // 8 * 8)


def _layer_rank(name: str, d_out: int, d_in: int, bpw: float, residual: bool,
                kv_factor: float) -> int:
    r = rank_for_bits(d_out, d_in, bpw, residual)
    if name.endswith(("k_proj", "v_proj")):  # GQA K/V are small: per-tensor rank
        r = max(8, int(r * kv_factor) // 8 * 8)
    cap = max(8, min(d_out, d_in) // 8 * 8)  # cannot exceed the matrix rank
    return min(r, cap)


def _path(W: Tensor, r: int, use_itq: bool, seed: int):
    """One path's init: returns (dense reconstruction, tensors to store)."""
    U, S, Vt = _svd(W)
    sq = S[:r].sqrt()  # even sqrt(S) split between the two factors
    Uh, Vh = U[:, :r] * sq, Vt[:r].T * sq
    del U, S, Vt
    if use_itq:
        R = joint_itq(torch.cat([Uh, Vh], 0), seed=seed).cpu()
        Uh, Vh = Uh @ R, Vh @ R
    Ub, Vb, h, g, urk, vrk = _dual_svid_parts(Uh, Vh)
    dense = (h[:, None] * Ub) @ ((urk * vrk)[:, None] * Vb) * g[None, :]
    return dense, dict(Ub=Ub, Vb=Vb, u1=h, v2=g, u2=urk, v1=vrk)


def _emit(out: dict, name: str, t: dict, res: str) -> None:
    for m, key in (("U", "Ub"), ("V", "Vb")):
        out[f"{name}.{m}{res}_packed"] = pack_signs(t[key].cpu())
        out[f"{name}.{m}{res}_shape"] = torch.tensor(t[key].shape, dtype=torch.int64)
    for s in ("u1", "u2", "v1", "v2"):
        out[f"{name}.{s}{res}"] = t[s].float().cpu().reshape(1, -1).contiguous()


def make_checkpoint(hf_id: str, out_dir: Path, bpw: float, use_itq: bool = True,
                    residual: bool = True, seed: int = 0, kv_factor: float = 1.0) -> None:
    """Write an init-only LittleBit(-2) checkpoint of `hf_id` into `out_dir`.

    `hf_id` is a hub id or a local directory.  Every 2-D `*.weight` inside a
    transformer block (`layers.<i>.`) is factorised, one tensor at a time to bound
    memory; everything else (embeddings, norms, lm_head) is stored as bf16 so the
    directory is self-contained for the converter.  `kv_factor` scales the rank
    of k_proj / v_proj (GQA).  Also writes `littlebit_config.json` and copies the
    base config and tokenizer files.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    base = Path(hf_id)
    if not base.is_dir():
        from huggingface_hub import snapshot_download
        base = Path(snapshot_download(hf_id, allow_patterns=["*.json", "*.safetensors",
                                                             "*.model", "*.txt", "*.jinja"]))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, Tensor] = {}
    for shard in sorted(base.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                w = f.get_tensor(key)
                if key.endswith(".weight") and w.dim() == 2 and _LAYER_RE.search(key):
                    name = key[: -len(".weight")]
                    d_out, d_in = w.shape
                    r = _layer_rank(name, d_out, d_in, bpw, residual, kv_factor)
                    w = w.float()
                    dense, t = _path(w, r, use_itq, seed)
                    _emit(tensors, name, t, "")
                    if residual:
                        w -= dense  # residual of the dequantised primary path
                        del dense
                        _, t2 = _path(w, r, use_itq, seed)
                        _emit(tensors, name, t2, "_R")
                    del w, t
                else:
                    tensors[key] = w.to(torch.bfloat16).contiguous()
    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    (out_dir / "littlebit_config.json").write_text(json.dumps({
        "eff_bit": bpw, "residual": residual, "use_itq": use_itq, "kv_factor": kv_factor,
        "seed": seed, "base_model": hf_id, "split_dim_rule": "rank_for_bits"}, indent=2))
    for fn in _SIDE_FILES:
        if (base / fn).exists():
            shutil.copy2(base / fn, out_dir / fn)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Make an init-only LittleBit-2 checkpoint.")
    p.add_argument("hf_id")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--bpw", type=float, required=True)
    p.add_argument("--no-itq", action="store_true")
    p.add_argument("--no-residual", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kv-factor", type=float, default=1.0)
    a = p.parse_args()
    make_checkpoint(a.hf_id, a.out_dir, a.bpw, not a.no_itq, not a.no_residual, a.seed, a.kv_factor)
