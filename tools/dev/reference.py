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
never as fp32, and the dense W is never built.  At a budget of B bits per
weight a LittleBit model has a little under B sign elements per original
parameter (the rest of the budget pays for the scale vectors), so the signs
take about B bytes per 1B parameters (B GB per 1B), e.g. about 0.5 GB per 1B
at the 0.55 setting used by the smoke test.  On top of that come 2 bytes/param
(bf16) for the non-LittleBit tensors (embeddings, norms, lm_head) and the
small scale vectors.  Terms such as "LittleBit" and "GGUF" are defined in
docs/development.md (glossary).  build_model creates
the base model with parameters on the meta device, so the dense weights of the
replaced Linears are never allocated.  forward() casts one layer's int8 signs
to fp32 transiently (4 bytes/element of that single layer) -- on purpose, to
keep the matmul exact and simple.
"""
from __future__ import annotations

import re
import contextlib
import warnings
from collections.abc import Mapping
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


# Order of the six tensors of one path; used for both constructor arguments and
# the buffer names `<name>_<path index>` (shared with emulate_q8_slow_path.py).
_NAMES = ("U", "V", "u1", "u2", "v1", "v2")


class LBLinear(nn.Module):
    """Linear layer evaluated as the LittleBit factor chain.

    The dense weight W is NEVER materialized: forward applies the scales and the
    two sign matmuls to the activations directly.  This is the whole point of the
    format -- an 8B model must fit in 16 GB of RAM, and a dense W would not.
    """

    def __init__(self, U: Tensor, V: Tensor, u1: Tensor, u2: Tensor, v1: Tensor,
                 v2: Tensor, residual: tuple[Tensor, ...] | None = None, packed: bool = False) -> None:
        """Takes the primary path's (U, V, u1, u2, v1, v2) and optionally the
        residual path's six tensors as a tuple.  Signs are stored as int8 (1 byte per sign), or with
        `packed=True` bit-packed as int32 words (1 bit per sign, the checkpoint's own layout, see
        `pack_signs`) and unpacked again in every forward call: 8x less memory for the signs, the same
        +/-1 values, a slower forward.  `packed` is for running large models (8B) on a 16 GB machine."""
        super().__init__()
        paths = [(U, V, u1, u2, v1, v2)] + ([tuple(residual)] if residual else [])
        self.n_paths = len(paths)
        self.packed = bool(packed)
        self._sign_shapes: dict[str, tuple[int, int]] = {}
        for i, path in enumerate(paths):
            for name, t in zip(_NAMES, path):
                if name not in ("U", "V"):
                    self.register_buffer(f"{name}_{i}", t)  # scales keep their stored dtype
                elif self.packed:
                    self.register_buffer(f"{name}_{i}", pack_signs(t))
                    self._sign_shapes[f"{name}_{i}"] = (int(t.shape[0]), int(t.shape[1]))
                else:
                    self.register_buffer(f"{name}_{i}", t.to(torch.int8))  # signs are exact in int8
        self.out_features, self.in_features = U.shape[0], V.shape[1]

    @classmethod
    def from_params(cls, p: LBLinearParams, packed: bool = False) -> "LBLinear":
        """Build from loaded checkpoint params (`packed`: see `__init__`)."""
        def tup(q: LBPathParams) -> tuple[Tensor, ...]:
            return (q.U, q.V, q.u1, q.u2, q.v1, q.v2)
        return cls(*tup(p.primary), residual=tup(p.residual) if p.residual else None, packed=packed)

    def _f32(self, name: str) -> Tensor:
        """Buffer `name` (e.g. "U_0", "u1_1") as float32; packed sign matrices are unpacked to +/-1 first."""
        t = getattr(self, name)
        if name in self._sign_shapes:
            return unpack_signs(t, self._sign_shapes[name], dtype=torch.float32)
        return t.float()

    def forward(self, x: Tensor) -> Tensor:
        """Sum over paths of ((((x*v2) @ Vq.T) * (v1*u2)) @ Uq.T) * u1, computed in fp32.

        The product v1*u2 is the single "mid" scale between the two sign matmuls.
        The GGUF converter pre-multiplies it and the engine stores it as one
        tensor, `lb_s_mid`.  Name mapping to the engine tensors:

            reference   u1       v1 * u2    v2
            engine      lb_s_u1  lb_s_mid   lb_s_v2
        """
        xf = x.to(torch.float32)
        y = 0
        for i in range(self.n_paths):
            # A per-layer fp32 transient on purpose (see module header); packed signs are unpacked here.
            U, V, u1, u2, v1, v2 = (self._f32(f"{n}_{i}") for n in _NAMES)
            h = ((xf * v2) @ V.T) * (v1 * u2)
            y = y + (h @ U.T) * u1
        return y.to(x.dtype)


# Checkpoint keys come in two spellings for the residual path, so two patterns:
#   _KEY   : `<module>.U_packed`, `<module>.u1`, and the scale form `<module>.u1_R`
#            (also the alias `U_packed_R`, accepted for tolerance);
#   _KEY_R : the sign form `<module>.U_R_packed` that the maker writes.
# load_checkpoint normalises both to the same internal name.
_KEY = re.compile(r"^(?P<mod>.+)\.(?P<name>U_packed|U_shape|V_packed|V_shape|u1|u2|v1|v2)(?P<res>_R)?$")
# Residual variants are stored as `U_R_packed`, `u1_R`, ...; normalise both spellings.
_KEY_R = re.compile(r"^(?P<mod>.+)\.(?P<name>U|V)_R_(?P<kind>packed|shape)$")


def _build_path(mod: str, t: dict[str, Tensor], res: str) -> LBPathParams:
    """Build one path (`res` is "" for primary, "_R" for residual) from raw tensors.

    Unpacks the sign matrices to int8 and validates every shape: packed word
    count vs `*_shape`, equal rank of U and V, and each scale's length.
    Raises ValueError with the module name on any mismatch.
    """
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


class LazyCheckpoint(Mapping):
    """Module name -> `LBLinearParams`, read from the checkpoint files on each access, one module at a time.

    What: the same keys, checks and values as `load_checkpoint` (which is simply `dict(LazyCheckpoint(d))`), but only
    the tensor NAMES are read when it is created; a module's sign matrices are read and unpacked when it is looked up.
    Why: a tool that walks the model module by module (the GGUF converter) then holds one layer's int8 signs at a
    time instead of all of them; for an 8B model at 0.55 bits per weight that is 3.5 GiB less RAM, which decides
    whether the conversion fits next to other programs on a 16 GB machine.
    Cost: values are NOT cached, so every lookup reads and unpacks that module again (a full pass over an 8B model
    takes about 3.5 s); keep the value while you need it, drop it when done.

    Raises ValueError (as `load_checkpoint`) on a key present in more than one shard or a module with missing
    tensors when created, and on shape mismatches when the module is looked up.
    """

    def __init__(self, ckpt_dir: Path) -> None:
        """Index the tensor names of every `*.safetensors` shard in `ckpt_dir` (no tensor data is read)."""
        from safetensors import safe_open

        seen: dict[str, Path] = {}
        # module -> internal name -> (shard, checkpoint key); internal names as in `_build_path` ("U_packed", "u1_R", ...)
        self._index: dict[str, dict[str, tuple[Path, str]]] = {}
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
                    self._index.setdefault(mod, {})[name] = (shard, key)
        self._residual: dict[str, bool] = {}
        for mod, t in self._index.items():
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
            self._residual[mod] = has_res

    def has_residual(self, mod: str) -> bool:
        """True when module `mod` has a residual path (known from the tensor names alone, no data read)."""
        return self._residual[mod]

    def __getitem__(self, mod: str) -> LBLinearParams:
        """Read, unpack and check module `mod`'s tensors (KeyError for a module that is not in the checkpoint)."""
        from safetensors import safe_open

        names = self._index[mod]
        t: dict[str, Tensor] = {}
        by_shard: dict[Path, list[tuple[str, str]]] = {}
        for name, (shard, key) in names.items():
            by_shard.setdefault(shard, []).append((name, key))
        for shard, items in by_shard.items():
            with safe_open(str(shard), framework="pt") as f:
                for name, key in items:
                    # .clone(): a tensor from get_tensor is a view into a private (copy-on-write) mapping of the WHOLE
                    # file, and keeping it keeps that mapping alive.  On Windows each such mapping is charged in full
                    # against the commit limit (measured: +2.77 GiB per kept tensor of the 2.97 GB 8B checkpoint, each
                    # lookup opening its own), so an 8B model built module by module crashed with an access violation
                    # once the commit limit was reached.  The copies are small (one module's packed signs and scales).
                    t[name] = f.get_tensor(key).clone()
        return LBLinearParams(_build_path(mod, t, ""), _build_path(mod, t, "_R") if self._residual[mod] else None)

    def __contains__(self, mod: object) -> bool:
        """Membership from the index alone (the inherited Mapping version would read and unpack the module)."""
        return mod in self._index

    def __iter__(self):
        """Module names, in the order they first appear in the checkpoint files."""
        return iter(self._index)

    def __len__(self) -> int:
        return len(self._index)


def load_checkpoint(ckpt_dir: Path) -> dict[str, LBLinearParams]:
    """Read every `*.safetensors` shard in `ckpt_dir` and group tensors by module prefix (all modules in memory).

    Raises ValueError on a key present in more than one shard, on a module with
    missing tensors, or when `*_shape` disagrees with the packed word count.
    For a module-at-a-time view with the same checks and values, use `LazyCheckpoint`.
    """
    return dict(LazyCheckpoint(ckpt_dir))


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


def build_model(base_hf_id: str, ckpt_dir: Path, packed_signs: bool = False) -> nn.Module:
    """Build the base HF model and swap checkpoint-listed Linears for LBLinear.

    Modules are matched by name against the checkpoint; embeddings, norms and
    lm_head (not in the checkpoint) are loaded from the base safetensors in bf16.
    The base model is built on the meta device and only non-replaced tensors are
    read, so the dense weights of replaced Linears are never held in memory.
    A checkpoint module with no matching nn.Linear raises ValueError.  A replaced
    Linear that had a bias loses it (LBLinear has none): a warning is emitted.

    `packed_signs=True` keeps the signs bit-packed (`LBLinear(packed=True)`) and reads the checkpoint one module at
    a time (`LazyCheckpoint`), so the unpacked signs of the whole model are never in memory at once (8B at 0.55
    bits per weight: 0.43 GiB of packed signs instead of 3.5 GiB of int8).  The values computed are the same.
    """
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    params = LazyCheckpoint(Path(ckpt_dir)) if packed_signs else load_checkpoint(Path(ckpt_dir))
    base = Path(base_hf_id)
    if not base.is_dir():
        from huggingface_hub import snapshot_download
        base = Path(snapshot_download(base_hf_id, allow_patterns=["*.json", "*.safetensors"]))
    config = AutoConfig.from_pretrained(str(base))
    with _meta_parameters():
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    replaced = []
    for name in list(params):
        p = params[name]   # with LazyCheckpoint: read now, freed once packed into its LBLinear
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, child, None)
        if not isinstance(old, nn.Linear):
            raise ValueError(f"checkpoint module {name!r} is not an nn.Linear in the base model")
        if old.bias is not None:
            warnings.warn(f"{name}: base Linear has a bias; LBLinear drops it", stacklevel=2)
        setattr(parent, child, LBLinear.from_params(p, packed=packed_signs))
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


class Bf16StoredLinear(nn.Module):
    """A bias-free linear whose weight is STORED in bf16 but applied in float32, a block of output rows at a time.

    What: `y = x.float() @ W.float().T`, with W upcast `rows` output rows at a time.
    Why: the output head of a large-vocabulary model (Qwen3-8B: 151936 x 4096) is 1.16 GiB in bf16 and 2.3 GiB in
    float32.  The checkpoint stores it in bf16, so float32 STORAGE adds no information; float32 COMPUTE is what the
    reference needs.  This gives float32 compute (the same products as `nn.Linear` with the weight cast to float32,
    up to the summation order inside the matrix product) at bf16 storage.
    """

    def __init__(self, weight: Tensor, rows: int = 16384) -> None:
        """`weight`: (out, in), kept in bf16; `rows`: output rows upcast per block (transient rows x in x 4 bytes)."""
        super().__init__()
        self.register_buffer("weight", weight.detach().to(torch.bfloat16))
        self.rows = int(rows)
        self.out_features, self.in_features = int(weight.shape[0]), int(weight.shape[1])

    def forward(self, x: Tensor) -> Tensor:
        """float32 result of x @ W.T (x is cast to float32 first)."""
        xf = x.to(torch.float32)
        parts = [xf @ self.weight[r0:r0 + self.rows].float().T for r0 in range(0, self.out_features, self.rows)]
        return torch.cat(parts, dim=-1)


def lean_float32(model: nn.Module) -> nn.Module:
    """Make a `build_model` result compute in float32 while its embedding and output head stay stored in bf16.

    Same numbers as `model.float()` for every op that matters here (the embedding lookup returns bf16 values
    widened to float32, exactly as a float32 copy of the table would; the head is a `Bf16StoredLinear`), with about
    half the memory of the two largest tensors.  Every other parameter and floating buffer becomes float32, as
    `.float()` does.  Used to run the 8B reference on a 16 GB machine.  Returns the model (changed in place).
    """
    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    tied = head is not None and head.weight is emb.weight
    keep = {id(emb.weight)} | ({id(head.weight)} if head is not None else set())
    for module in model.modules():
        for name, prm in list(module._parameters.items()):
            if prm is not None and id(prm) not in keep and prm.is_floating_point():
                module._parameters[name] = nn.Parameter(prm.data.float(), requires_grad=prm.requires_grad)
        for name, buf in list(module._buffers.items()):
            if buf is not None and buf.is_floating_point():
                module._buffers[name] = buf.float()
    if head is not None:
        model.set_output_embeddings(Bf16StoredLinear(head.weight.data))
        if tied:
            model.config.tie_word_embeddings = False  # the head now owns a (shared-storage) bf16 copy
    # the decoder layers see float32 hidden states: widen the embedding's output (bf16 values, exact in float32)
    emb.register_forward_hook(lambda _m, _inp, out: out.float())
    return model

