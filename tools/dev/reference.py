"""Lean PyTorch reference for LittleBit (sub-1-bit) linear layers.

What: loads a LittleBit safetensors checkpoint and runs it in plain PyTorch.
Why:  this is the ground truth every later correctness test compares against
      (checkpoint maker, GGUF converter, engine ops, perplexity parity).  It is
      written from the published formulas only, and deliberately small.

Per linear layer (primary path; the residual path is identical, `_R` tensors,
and its output is summed in; the residual is optional):

    y = ((((x * v2) @ Vq.T) * (v1 * u2)) @ Uq.T) * u1

with Vq the (r, in) sign matrix, Uq the (out, r) sign matrix, and scales
u1 (1,out), u2 (1,r), v1 (1,r), v2 (1,in).

Memory (the 16 GB / 8B-model goal): signs are held as int8 (1 byte/element),
never as fp32, and the dense W is never built.  A LittleBit model has roughly
0.55 sign elements per original parameter, so about 0.55 GB per 1B parameters
for signs, plus 2 bytes/param (bf16) for the non-LittleBit tensors
(embeddings, norms, lm_head), plus small scale vectors.  build_model creates
the base model with parameters on the meta device, so the dense weights of the
replaced Linears are never allocated.  forward() casts one layer's int8 signs
to fp32 transiently (4 bytes/element of that single layer) -- on purpose, to
keep the matmul exact and simple.
"""
from __future__ import annotations

import re
import contextlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

# Bit order of packed sign matrices.  +1 -> bit 0, -1 -> bit 1, LSB-first,
# 32 columns per int32 word.  This matches the checkpoint layout produced by the
# LittleBit tooling; every consumer (maker, converter, engine) must agree on it.
_BITS = 32


# Upper bound on elements of the transient (rows, words, 32) bit tensor.
_CHUNK_ELEMS = 1 << 24


def unpack_signs(packed: Tensor, shape: tuple[int, int],
                 dtype: torch.dtype = torch.float32) -> Tensor:
    """Unpack an int32 bit-matrix into a +/-1 matrix (`dtype`) of logical `shape`.

    `packed` is (rows, ceil(cols/32)); bit k of word w is column 32*w+k
    (LSB-first), 0 -> +1, 1 -> -1.  Padding bits beyond `cols` in the last
    word are ignored (truncated), whatever their value.  Rows are processed in
    chunks so the int32 bit transient stays bounded for large matrices.
    """
    rows, cols = int(shape[0]), int(shape[1])
    shifts = torch.arange(_BITS, device=packed.device, dtype=torch.int32)
    out = torch.empty(rows, cols, dtype=dtype, device=packed.device)
    step = max(1, _CHUNK_ELEMS // max(1, packed.shape[1] * _BITS))
    for r0 in range(0, rows, step):
        chunk = packed[r0:r0 + step]
        # `& 1` after the shift removes the sign extension of arithmetic >> on bit 31.
        bits = ((chunk.unsqueeze(-1) >> shifts) & 1).reshape(chunk.shape[0], -1)[:, :cols]
        out[r0:r0 + step] = (1 - 2 * bits).to(dtype)
    return out


def pack_signs(signs: Tensor) -> Tensor:
    """Inverse of `unpack_signs`: (rows, cols) +/-1 values -> (rows, ceil(cols/32)) int32.

    Negative values become bit 1, others bit 0; padding bits are written as 0.
    """
    rows, cols = signs.shape
    words = (cols + _BITS - 1) // _BITS
    bits = torch.zeros(rows, words * _BITS, dtype=torch.int64)
    bits[:, :cols] = (signs < 0).to(torch.int64)
    weights = torch.tensor(1, dtype=torch.int64) << torch.arange(_BITS, dtype=torch.int64)
    w = (bits.reshape(rows, words, _BITS) * weights).sum(-1)
    # Wrap unsigned 32-bit values into int32's two's-complement range.
    return torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)


@dataclass
class LBPathParams:
    """One factorized path: sign matrices U (out,r), V (r,in) and four scales."""
    U: Tensor
    V: Tensor
    u1: Tensor
    u2: Tensor
    v1: Tensor
    v2: Tensor


@dataclass
class LBLinearParams:
    """All tensors for one linear layer; `residual` is None for no-residual checkpoints."""
    primary: LBPathParams
    residual: LBPathParams | None = None


_NAMES = ("U", "V", "u1", "u2", "v1", "v2")


class LBLinear(nn.Module):
    """Linear layer evaluated as the LittleBit factor chain.

    The dense weight W is NEVER materialized: forward applies the scales and the
    two sign matmuls to the activations directly.  This is the whole point of the
    format -- an 8B model must fit in 16 GB of RAM, and a dense W would not.
    """

    def __init__(self, U: Tensor, V: Tensor, u1: Tensor, u2: Tensor, v1: Tensor,
                 v2: Tensor, residual: tuple[Tensor, ...] | None = None) -> None:
        """Takes the primary path's (U, V, u1, u2, v1, v2) and optionally the
        residual path's six tensors as a tuple.  Signs are stored as int8."""
        super().__init__()
        paths = [(U, V, u1, u2, v1, v2)] + ([tuple(residual)] if residual else [])
        self.n_paths = len(paths)
        for i, path in enumerate(paths):
            for name, t in zip(_NAMES, path):
                # Signs are exact in int8; scales keep their stored dtype.
                self.register_buffer(f"{name}_{i}", t.to(torch.int8) if name in ("U", "V") else t)
        self.out_features, self.in_features = U.shape[0], V.shape[1]

    @classmethod
    def from_params(cls, p: LBLinearParams) -> "LBLinear":
        """Build from loaded checkpoint params."""
        def tup(q: LBPathParams) -> tuple[Tensor, ...]:
            return (q.U, q.V, q.u1, q.u2, q.v1, q.v2)
        return cls(*tup(p.primary), residual=tup(p.residual) if p.residual else None)

    def forward(self, x: Tensor) -> Tensor:
        """Sum over paths of ((((x*v2) @ Vq.T) * (v1*u2)) @ Uq.T) * u1, computed in fp32."""
        xf = x.to(torch.float32)
        y = 0
        for i in range(self.n_paths):
            # .float() makes a per-layer fp32 transient on purpose (see module header).
            U, V, u1, u2, v1, v2 = (getattr(self, f"{n}_{i}").float() for n in _NAMES)
            h = ((xf * v2) @ V.T) * (v1 * u2)
            y = y + (h @ U.T) * u1
        return y.to(x.dtype)


_KEY = re.compile(r"^(?P<mod>.+)\.(?P<name>U_packed|U_shape|V_packed|V_shape|u1|u2|v1|v2)(?P<res>_R)?$")
# Residual variants are stored as `U_R_packed`, `u1_R`, ...; normalise both spellings.
_KEY_R = re.compile(r"^(?P<mod>.+)\.(?P<name>U|V)_R_(?P<kind>packed|shape)$")


def _build_path(mod: str, t: dict[str, Tensor], res: str) -> LBPathParams:
    signs = {}
    for m in ("U", "V"):
        packed, shape = t[f"{m}{res}_packed"], t[f"{m}{res}_shape"].tolist()
        rows, cols = int(shape[0]), int(shape[1])
        want = (rows, (cols + _BITS - 1) // _BITS)
        if packed.dim() != 2 or tuple(packed.shape) != want:
            raise ValueError(
                f"{mod}.{m}{res}: shape {shape} inconsistent with packed tensor "
                f"of shape {tuple(packed.shape)} (expected {want})")
        signs[m] = unpack_signs(packed, (rows, cols), dtype=torch.int8)  # keep int8 only
    (out_f, r), (r2, in_f) = signs["U"].shape, signs["V"].shape
    if r != r2:
        raise ValueError(f"{mod}{res}: U has {r} columns but V has {r2} rows")
    scales = {s: t[f"{s}{res}"] for s in ("u1", "u2", "v1", "v2")}
    for s_, want in (("u1", out_f), ("u2", r), ("v1", r), ("v2", in_f)):
        if scales[s_].numel() != want:
            raise ValueError(f"{mod}.{s_}{res}: has {scales[s_].numel()} elements, expected {want}")
    return LBPathParams(signs["U"], signs["V"], *scales.values())


def load_checkpoint(ckpt_dir: Path) -> dict[str, LBLinearParams]:
    """Read every `*.safetensors` shard in `ckpt_dir` and group tensors by module prefix.

    Raises ValueError on a key present in more than one shard, on a module with
    missing tensors, or when `*_shape` disagrees with the packed word count.
    """
    from safetensors import safe_open

    seen: dict[str, Path] = {}
    mods: dict[str, dict[str, Tensor]] = {}
    for shard in sorted(Path(ckpt_dir).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                if key in seen:
                    raise ValueError(f"duplicate key {key!r} in {shard.name} and {seen[key].name}")
                seen[key] = shard
                # Normalise to "<name>[_R]" where the residual suffix is `_R` on
                # the sign key stem (U_R_packed) or on the scale name (u1_R).
                mr = _KEY_R.match(key)
                if mr:
                    mod, name = mr["mod"], f"{mr['name']}_R_{mr['kind']}"
                else:
                    m = _KEY.match(key)
                    if m is None:
                        continue  # non-LittleBit tensor (e.g. norms); not ours
                    mod, name = m["mod"], m["name"]
                    if m["res"]:  # tolerate `U_packed_R` as an alias of `U_R_packed`
                        name = name[0] + "_R_" + name[2:] if name[0] in "UV" else name + "_R"
                mods.setdefault(mod, {})[name] = f.get_tensor(key)
    out: dict[str, LBLinearParams] = {}
    for mod, t in mods.items():
        need = ["U_packed", "U_shape", "V_packed", "V_shape", "u1", "u2", "v1", "v2"]
        missing = [k for k in need if k not in t]
        if missing:
            raise ValueError(f"{mod}: missing tensors {missing}")
        res_need = ["U_R_packed", "U_R_shape", "V_R_packed", "V_R_shape", "u1_R", "u2_R", "v1_R", "v2_R"]
        has_res = any(k in t for k in res_need)
        if has_res:
            missing = [k for k in res_need if k not in t]
            if missing:
                raise ValueError(f"{mod}: missing residual tensors {missing}")
        out[mod] = LBLinearParams(_build_path(mod, t, ""), _build_path(mod, t, "_R") if has_res else None)
    return out


@contextlib.contextmanager
def _meta_parameters():
    """Create nn.Module parameters on the meta device (buffers stay real).

    Same idea as accelerate's init_empty_weights: the dense weights of Linears we
    are about to replace are never allocated.  Buffers (e.g. rotary inv_freq)
    stay real because they are computed at construction time.
    """
    orig = nn.Module.register_parameter

    def register(self, name, param):
        orig(self, name, param)
        if param is not None:
            self._parameters[name] = nn.Parameter(
                param.data.to("meta"), requires_grad=param.requires_grad)

    nn.Module.register_parameter = register
    try:
        yield
    finally:
        nn.Module.register_parameter = orig


def build_model(base_hf_id: str, ckpt_dir: Path) -> nn.Module:
    """Build the base HF model and swap checkpoint-listed Linears for LBLinear.

    Modules are matched by name against the checkpoint; embeddings, norms and
    lm_head (not in the checkpoint) are loaded from the base safetensors in bf16.
    The base model is built on the meta device and only non-replaced tensors are
    read, so the dense weights of replaced Linears are never held in memory.
    A checkpoint module with no matching nn.Linear raises ValueError.  A replaced
    Linear that had a bias loses it (LBLinear has none): a warning is emitted.
    """
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    params = load_checkpoint(Path(ckpt_dir))
    base = Path(base_hf_id)
    if not base.is_dir():
        from huggingface_hub import snapshot_download
        base = Path(snapshot_download(base_hf_id, allow_patterns=["*.json", "*.safetensors"]))
    config = AutoConfig.from_pretrained(str(base))
    with _meta_parameters():
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    replaced = []
    for name, p in params.items():
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, child, None)
        if not isinstance(old, nn.Linear):
            raise ValueError(f"checkpoint module {name!r} is not an nn.Linear in the base model")
        if old.bias is not None:
            warnings.warn(f"{name}: base Linear has a bias; LBLinear drops it", stacklevel=2)
        setattr(parent, child, LBLinear.from_params(p))
        replaced.append(name + ".")
    state = {}
    for shard in sorted(base.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                if not key.startswith(tuple(replaced)):
                    state[key] = f.get_tensor(key).to(torch.bfloat16)
    model.load_state_dict(state, strict=False, assign=True)
    model.tie_weights()
    meta = [n for n, t in model.named_parameters() if t.is_meta]
    if meta:
        raise ValueError(f"base model tensors missing from {base}: {meta[:5]}")
    return model.eval()
