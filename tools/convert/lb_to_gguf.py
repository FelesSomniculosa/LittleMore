"""LittleBit checkpoint (Samsung format) -> GGUF with GGML_TYPE_B1 weights, with validation.

What
    `convert()` turns a directory holding a Samsung-format LittleBit checkpoint
    (bit-packed sign matrices + scale vectors, plus the original HF config/tokenizer and the
    non-LittleBit tensors) into ONE self-contained GGUF file that the LittleMore engine can
    load: every LittleBit linear becomes five tensors `blk.N.<sub>.lb_v / lb_u / lb_s_v2 /
    lb_s_mid / lb_s_u1` (plus the same five with an `r` suffix for the residual path), the
    signs stored as the engine's pure 1-bit type B1 (exactly 1.0 bit per sign, no scale).
    With a residual path, `blk.N.<sub>.lb_s_pkr` holds the three residual scale vectors once
    more, packed as [lb_s_v2r | lb_s_midr | lb_s_u1r] (F32, length n_in + r_pad + n_out): the
    operand layout of the engine's fused op, which then needs no per-call packing.  The
    separate vectors stay for readers that use them (the Q1_0 chain, older engines).
Why
    The sign matrices are the bulk of the model; B1 stores them at exactly 1 bit per sign
    (the older Q1_0 slow path stored an unused per-block scale as well: 1.125 bits).  The
    row/column permutations the HF-to-GGUF conversion applies (RoPE layout of Q/K, Qwen3.5
    value-head reorder, ...) are NOT reimplemented here: see "Probes" below.

Starting point and licence
    The structure of this file (probe mechanism, scale folding, rank padding, naming) is
    derived from `tools/littlebit/lb_to_gguf.py` by Cybertiron
    (https://github.com/Cybertiron/llama.cpp-littlebit, "Architecture-generic LittleBit ->
    GGUF converter"), which wrote Q1_0 tensors from a base GGUF.  That file carries no
    copyright line of its own; it is part of the llama.cpp fork whose LICENSE reads
    (verbatim, from engine/LICENSE):

        MIT License

        Copyright (c) 2023-2026 The ggml authors

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

    Changes here: B1 output with the polarity flip, one self-contained input directory (the
    base GGUF is produced inside, by llama.cpp's own converter classes), per-layer numerical
    validation by reading the written file back, and an atomic write.

Families exercised by tests/convert (tiny random models): Qwen3, Llama, Gemma 3, Qwen3.5
(DeltaNet).  Llama (RoPE row permutation of q/k) and Qwen3.5 (value-head reorder) exercise
real permutations.  Gemma 3 exercises only the GENERIC path (its converter applies identity
permutations to the linears); nothing Gemma-specific is claimed beyond that.

Input widths: every LittleBit linear's input width (hidden size / intermediate size) must be
a multiple of 128, because a B1 row is made of whole 128-sign blocks and the fused kernel
reads whole blocks; the converter rejects anything else with a ConversionError rather than
padding (padding the input would change the activation the engine feeds in).  The RANK has
no such restriction: any logical rank >= 1 is padded up (see `_build_path`).

Running the result and checking it
    The engine runs these B1 GGUFs through its fused LittleBit op.  Validation here does
    not depend on the engine: every linear is read back from the written file and compared
    with the PyTorch reference (`validate.py`), so a converter bug cannot be hidden by a
    matching engine bug.  This module does not modify the engine.

Naming map (checkpoint -> GGUF); the product v1*u2 is folded because nothing sits between
the two rank-space scales (same table as reference.py):

        reference/checkpoint   u1        v1 * u2     v2
        GGUF tensor            lb_s_u1   lb_s_mid    lb_s_v2
        sign matrices          U (out,r) -> lb_u [r, out]    V (r,in) -> lb_v [in, r]   (ggml order)

    y = s_u1 * (Ub @ (s_mid * (Vb^T @ (s_v2 * x))))  [+ the `r` path]

Choices for the non-LittleBit tensors (embeddings, norms, lm_head, small non-factored
linears, biases): they come from the same checkpoint directory (the maker stores them as
bf16) and go through llama.cpp's real converter class for the architecture, with
`--outtype f16` semantics: 2-D+ weights (token embeddings, output head, any small dense
linear) are written as F16, 1-D tensors (norms, biases, scale-like vectors) as F32.

Head type (`--head-type f16|q8_0|q6_k`, default f16): the precision of the token embedding and the output head
(`token_embd.weight`, plus `output.weight` when the head is not tied to the embedding).  These are the largest single
tensors of a small model (Qwen3-0.6B: 311 MB of a 636 MB file at F16) and the head is read once per generated token,
so a Q8_0 (8.5 bits per value) or Q6_K (6.56 bits) head cuts those bytes 1.9x or 2.4x.  The quantizers are gguf-py's
Q8_0 and Q6_K, both bit-identical to ggml's C reference quantizers (tests/convert/test_head_type.py); the source values
are the checkpoint's, widened to float32 first (never F16-rounded before quantization).  Nothing else in the file
changes.  Correctness of the LittleBit linears is measured with the default F16 head; a quantized head adds its own
(small, measured: bench/results/phase1.md) error.  A head whose row length the type cannot take (Q8_0: multiples of
32, Q6_K: multiples of 256) is refused with a ConversionError instead of being written as F16.

Terms (LittleBit, GGUF, rank, RoPE, GQA) are defined in docs/development.md (glossary).

Note on history: Cybertiron's file was briefly vendored under engine/ and removed there once
this converter replaced it.

Usage: python -m tools.convert.lb_to_gguf <ckpt_dir> -o <out.gguf> [--head-type f16|q8_0|q6_k] [--no-validate] [--assume-no-littlebit-config]
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

# The engine's gguf-py (has GGMLQuantizationType.B1; the pip-installed gguf may not) and its
# `conversion` package (llama.cpp's HF -> GGUF converter classes), and tools/dev/reference.py.
_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO / "tools" / "dev", _REPO / "engine", _REPO / "engine" / "gguf-py"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gguf  # noqa: E402
import reference as ref  # noqa: E402
from conversion import get_model_class  # noqa: E402
from conversion.base import ModelBase, ModelType, get_model_architecture  # noqa: E402

from . import validate as _validate  # noqa: E402
# Public alias so callers/tests can reach the validator as `lb_to_gguf.validate`.  Inside
# `convert_checkpoint` the parameter `validate` shadows it, hence the private `_validate`.
validate = _validate

logger = logging.getLogger("littlemore.convert")

QK = 128  # signs per B1 block (16 bytes); the rank and the input width must be multiples of it
_HEAD_TYPES = ("f16", "q8_0", "q6_k")
# ggml type of the embedding / output head per head_type
_HEAD_QTYPE = {"f16": gguf.GGMLQuantizationType.F16, "q8_0": gguf.GGMLQuantizationType.Q8_0,
               "q6_k": gguf.GGMLQuantizationType.Q6_K}
# GGUF names of the head tensors (the output head exists only when it is not tied to the embedding)
_HEAD_TENSORS = ("token_embd.weight", "output.weight")
# Names the maker / Samsung tooling use for the scale tensors of a path; used to recognise
# which checkpoint keys belong to LittleBit linears (everything else is a plain tensor).
_LB_KEY = ref._KEY
_LB_KEY_R = ref._KEY_R


class ConversionError(Exception):
    """The checkpoint cannot be converted correctly (bad input, unsupported layout, failed validation)."""


@dataclass
class Probe:
    """Where the HF converter put one GGUF tensor made from a checkpoint linear.

    `gguf_name`: GGUF tensor name (with `.weight`); GGUF output row i = checkpoint row
    `row_idx[i]`, GGUF input column j = checkpoint column `col_idx[j]`.
    """
    gguf_name: str
    row_idx: np.ndarray
    col_idx: np.ndarray


@dataclass
class ConversionReport:
    """What `convert_checkpoint` did (returned for tests and the CLI summary)."""
    n_linears: int = 0                      # LittleBit linears listed in the checkpoint
    n_validated: int = 0                    # ... of which read back and validated
    max_rel_err: float = 0.0                # worst validation error over all linears
    tolerance: float = 0.0                  # the (smallest) tolerance used
    probes: dict[str, list[Probe]] = field(default_factory=dict)   # module -> GGUF tensors made from it
    written_names: list[str] = field(default_factory=list)        # every tensor name in the GGUF
    biases: list[str] = field(default_factory=list)               # checkpoint bias tensors carried over
    dropped: list[str] = field(default_factory=list)              # source tensors the HF converter dropped
    arch: str = ""
    config_check_skipped: bool = False      # littlebit_config.json cross-check skipped (override flag)
    lb_bytes: int = 0                       # bytes of the B1 sign matrices written
    head_type: str = "f16"                  # precision of token_embd / output (see "Head type" above)


def pad_to(n: int, m: int = QK) -> int:
    """Smallest multiple of `m` that is >= n (the rank is padded to a multiple of 128)."""
    return (n + m - 1) // m * m


def pack_residual_scales(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """[s_v2 | s_mid | s_u1] of one path as one float32 vector (the `lb_s_pkr` tensor of the residual path).

    `arrays`: the output of `_build_path` for the residual path.  The order and the lengths
    (n_in, r_pad, n_out) are those of the engine's `ggml_littlebit_packed` source 8.
    """
    return np.ascontiguousarray(np.concatenate(
        [arrays["lb_s_v2"], arrays["lb_s_mid"], arrays["lb_s_u1"]]).astype(np.float32, copy=False))


def signs_to_b1(signs: np.ndarray) -> np.ndarray:
    """Pack a (rows, cols) matrix of +1/-1 (int8) into B1 bytes, shape (rows, cols // 8).

    THE POLARITY FLIP.  Samsung's checkpoints pack +1 as bit 0 and -1 as bit 1
    (`reference.unpack_signs` documents this); the engine's B1 type packs +1 as bit 1 and
    -1 as bit 0.  Both are least-significant-bit first (element i in byte i // 8, bit
    i % 8).  We never touch the Samsung bits directly: the checkpoint is first decoded to
    real +1/-1 values by `reference.unpack_signs` (inside `reference.load_checkpoint`),
    and here the VALUES are re-encoded with the B1 convention, `bit = (value > 0)`.  That
    is the flip, expressed as "encode the meaning, not the bits", so it cannot be
    forgotten or applied twice by accident.  This equals `gguf.quants.quantize(x, B1)`
    for x = +/-1 (checked in tests/convert).
    """
    if signs.ndim != 2 or signs.shape[1] % 8:
        raise ValueError(f"signs must be (rows, cols) with cols % 8 == 0, got {signs.shape}")
    return np.packbits(signs > 0, axis=-1, bitorder="little")


def _f32(t: torch.Tensor) -> np.ndarray:
    """A checkpoint scale tensor as a flat contiguous float32 array (stored scales may be bf16/f16)."""
    return np.ascontiguousarray(t.float().numpy().reshape(-1), dtype=np.float32)


def _build_path(p: ref.LBPathParams, ri: np.ndarray, ci: np.ndarray, where: str) -> dict[str, np.ndarray]:
    """GGUF arrays of one path (primary or residual) of one linear, for the permutation (ri, ci).

    Returns {"lb_v", "lb_u", "lb_s_v2", "lb_s_mid", "lb_s_u1"} (no residual suffix).

    Row/column permutation: the HF converter may reorder the linear's output rows (`ri`:
    GGUF row i = checkpoint row ri[i]) and input columns (`ci`).  For W = diag(u1) U diag(mid)
    V diag(v2) a row permutation of W is the same permutation of the rows of U and of u1;
    a column permutation of W is the same permutation of the columns of V and of v2.  The
    rank dimension is untouched.

    Rank padding: B1 rows are 128-sign blocks, and the fused kernel walks the rank in blocks
    too, so the rank r is padded to rp = pad_to(r).  Padded V rows / U columns are filled with
    +1 (any sign would do) and the padded `lb_s_mid` entries are exactly 0.  Every padded rank
    index k then contributes  s_u1 * U[:, k] * (0 * (V[k] . (s_v2 * x)))  =  0  to the output,
    so the padded layer computes bit-for-bit the same function as the unpadded one.  (The
    padded products are exact zeros, not small numbers, so even float rounding is unchanged.)
    """
    r = p.V.shape[0]
    rp = pad_to(r)
    n_in, n_out = len(ci), len(ri)
    if n_in % QK:
        raise ConversionError(f"{where}: input width {n_in} is not a multiple of {QK}; B1 rows are 128-sign blocks")
    V = p.V.numpy()  # int8 +/-1, (r, in), already decoded from the Samsung bits
    U = p.U.numpy()  # int8 +/-1, (out, r)
    Vp = np.ones((rp, n_in), dtype=np.int8)
    Vp[:r] = V[:, ci]
    Up = np.ones((n_out, rp), dtype=np.int8)
    Up[:, :r] = U[ri]
    # mid = v1 * u2, formed in float32 exactly like reference.LBLinear (v1 * u2), then zero-padded.
    mid = np.zeros(rp, dtype=np.float32)
    mid[:r] = _f32(p.v1) * _f32(p.u2)
    return {
        "lb_v": signs_to_b1(Vp),                          # ggml [n_in, rp]: rp rows of n_in/8 bytes
        "lb_u": signs_to_b1(Up),                          # ggml [rp, n_out]: n_out rows of rp/8 bytes
        "lb_s_v2": np.ascontiguousarray(_f32(p.v2)[ci]),
        "lb_s_mid": mid,
        "lb_s_u1": np.ascontiguousarray(_f32(p.u1)[ri]),
    }


def _convert_one(conv: ModelBase, name: str, data: torch.Tensor, bid: int | None) -> list[tuple[str, torch.Tensor]]:
    """Push one tensor through the HF converter exactly as it would: filter_tensors, then modify_tensors."""
    item = conv.filter_tensors((name, lambda: data))
    if item is None:
        return []
    name, gen = item
    return list(conv.modify_tensors(gen(), name, bid))


def probe_linear(conv: ModelBase, name: str, bid: int | None, out_f: int, in_f: int) -> list[Probe]:
    """Find where llama.cpp's converter puts the rows and columns of the HF linear `name`.

    The probe mechanism.  The converter classes rename tensors and may permute, split or
    merge rows/columns (Q/K rows for RoPE, Qwen3.5 V-head reorder, ...).  Instead of
    copying that logic per architecture we feed the real converter two index tensors of
    the linear's shape:  R[i, j] = i  (row probe)  and  C[i, j] = j  (column probe).  What
    comes out says, for each output tensor, which source row / column sits at each position.
    That is only meaningful when the transform is a *pure permutation*: it moves whole rows
    and whole columns around (or selects a subset of them) without mixing values -- then
    R comes out constant along each row and C constant along each column, and the factors
    can be permuted the same way (see `_build_path`).  Anything else (a scaling, a
    sum of rows, ...) would be mangled by us, so it is rejected.  Float32 holds integers
    exactly up to 2**24, far beyond any layer dimension.
    """
    R = torch.arange(out_f, dtype=torch.float32)[:, None].expand(out_f, in_f).contiguous()
    rows = _convert_one(conv, name, R, bid)
    del R
    C = torch.arange(in_f, dtype=torch.float32)[None, :].expand(out_f, in_f).contiguous()
    cols = _convert_one(conv, name, C, bid)
    del C
    res: list[Probe] = []
    if len(rows) != len(cols):
        raise ConversionError(f"{name}: converter produced {len(rows)} tensors for the row probe but {len(cols)} for the column probe")
    for (gn, r), (gn2, c) in zip(rows, cols):
        if gn != gn2 or r.shape != c.shape or r.ndim != 2:
            raise ConversionError(f"{name}: inconsistent probe outputs ({gn}, {gn2}, {tuple(r.shape)}, {tuple(c.shape)})")
        ri = r[:, 0].round().long()
        ci = c[0, :].round().long()
        if not (torch.equal(r, ri[:, None].float().expand_as(r)) and torch.equal(c, ci[None, :].float().expand_as(c))):
            raise ConversionError(f"{name}: converter transform is not a pure row/column permutation")
        if len(torch.unique(ri)) != len(ri) or len(torch.unique(ci)) != len(ci):
            raise ConversionError(f"{name}: converter repeats rows or columns (not a permutation)")
        res.append(Probe(gn, ri.numpy(), ci.numpy()))
    return res


_LAYER_NAME = re.compile(r"^(?P<pre>.*?layers\.)(?P<bid>\d+)\.(?P<sub>.+)$")


def _split_layer_name(mod: str) -> tuple[int | None, str]:
    """('model.layers.3.mlp.up_proj') -> (3, 'mlp.up_proj'); (None, mod) when not inside a layer."""
    m = _LAYER_NAME.match(mod)
    return (int(m["bid"]), m["sub"]) if m else (None, mod)


def _shard_files(ckpt: Path) -> list[Path]:
    """The `*.safetensors` shards of the checkpoint directory (same rule as reference.load_checkpoint)."""
    return sorted(ckpt.glob("*.safetensors"))


def _checkpoint_keys(ckpt: Path) -> list[str]:
    """All tensor keys of the checkpoint shards."""
    keys: list[str] = []
    for shard in _shard_files(ckpt):
        with safe_open(str(shard), framework="pt") as f:
            keys.extend(f.keys())
    return keys


def _lb_module_of(key: str) -> str | None:
    """Module name if `key` is one of the LittleBit tensors (signs / shapes / scales), else None."""
    m = _LB_KEY_R.match(key) or _LB_KEY.match(key)
    return m["mod"] if m else None


def _check_inputs(ckpt: Path, head_type: str, assume_no_littlebit_config: bool = False) -> dict[str, Any]:
    """Early, cheap checks: head_type, required config files, shards present.  Returns littlebit_config.

    `littlebit_config.json` is only used for one cross-check (does `residual` match the tensors);
    shapes come from the tensors.  Without the override flag a missing file is an error whose
    message says what it is and how to proceed; with the flag it is skipped (with a warning).
    """
    if head_type not in _HEAD_TYPES:
        raise ConversionError(f"unknown head_type {head_type!r}; expected one of {_HEAD_TYPES}")
    if not ckpt.is_dir():
        raise ConversionError(f"checkpoint directory not found: {ckpt}")
    if not (ckpt / "config.json").is_file():
        raise ConversionError(f"{ckpt}: required file config.json is missing (the Hugging Face config: architecture and hyper-parameters)")
    if not _shard_files(ckpt):
        raise ConversionError(f"{ckpt}: no *.safetensors files")
    cfg = ckpt / "littlebit_config.json"
    if not cfg.is_file():
        if not assume_no_littlebit_config:
            raise ConversionError(
                f"{ckpt}: littlebit_config.json is missing.  This small JSON file is written by the LittleBit "
                "checkpoint tooling next to the weights and records how the checkpoint was made (bit budget, "
                "whether a residual path exists, ...); legacy checkpoints may lack it.  The converter only uses it "
                "to cross-check that residual paths are present exactly when it says so.  If you trust the "
                "checkpoint, re-run with --assume-no-littlebit-config (Python: assume_no_littlebit_config=True): "
                "that skips only this cross-check and detects residual paths from the tensors themselves.")
        logger.warning("littlebit_config.json is missing: residual cross-check skipped (--assume-no-littlebit-config)")
        return {}
    if assume_no_littlebit_config:
        logger.warning("--assume-no-littlebit-config given: residual cross-check against littlebit_config.json skipped")
        return {}
    try:
        return json.loads(cfg.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ConversionError(f"{cfg}: not valid JSON ({e})") from e


def _available_memory() -> int | None:
    """Bytes of RAM currently available, or None when it cannot be determined.

    psutil if installed (it is not a project dependency); else the standard library only:
    GlobalMemoryStatusEx via ctypes on Windows, sysconf on POSIX.
    """
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:
        if os.name == "nt":
            import ctypes

            class _MemStatus(ctypes.Structure):
                # Layout of the Win32 MEMORYSTATUSEX struct filled in by GlobalMemoryStatusEx.
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
                return int(st.ullAvailPhys)
            return None
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, AttributeError):
        return None


def _estimate_bytes(ckpt: Path) -> int:
    """Peak RAM the conversion needs, computed from the tensor shapes in the checkpoint headers.

    Per packed sign matrix of n bits: n bytes (reference.load_checkpoint holds int8 signs) plus
    n/8 bytes (the B1 copy waiting in the GGUF writer); per scale vector 8 bytes/element (the
    checkpoint copy + the float32 copy); per other tensor 4 bytes/element (source + F16/F32 output).
    That is roughly 0.55 GB per billion model parameters at a 0.55 bits-per-weight budget.
    """
    total = 0.0
    for shard in _shard_files(ckpt):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                n = 1
                for d in f.get_slice(key).get_shape():
                    n *= int(d)
                if key.endswith("_packed"):
                    total += n * 32 * (1 + 1 / 8)
                elif _lb_module_of(key) is not None:
                    total += n * 8
                else:
                    total += n * 4
    return int(total)


def _memory_preflight(ckpt: Path) -> None:
    """Raise ConversionError with the estimate if the conversion clearly cannot fit in free RAM."""
    avail = _available_memory()
    if avail is None:
        logger.debug("available memory unknown; skipping the memory preflight")
        return
    need = _estimate_bytes(ckpt)
    logger.info("memory preflight: need ~%.2f GiB, %.2f GiB available", need / 2**30, avail / 2**30)
    if need > avail:
        raise ConversionError(f"converting {ckpt} needs an estimated {need / 2**30:.2f} GiB of RAM "
                              f"but only {avail / 2**30:.2f} GiB are available; free memory or use a machine with more RAM")


def _make_converter(ckpt: Path, tmp_out: Path, head_type: str = "f16") -> ModelBase:
    """llama.cpp's converter class for the checkpoint's HF architecture, in F16 mode.

    `_SHARED_INPUT_MERGES = ()` switches off the merging of tensors that share an input
    (Qwen3.5 fuses in_proj_qkv+in_proj_z and in_proj_b+in_proj_a): a merged linear cannot be
    factored, so each linear stays its own GGUF tensor (the loader accepts both layouts).

    `head_type` other than f16: the converter's per-tensor type hook (`tensor_force_quant`, consulted for every
    tensor before the file-type default) returns Q8_0 / Q6_K for the token embedding and the output head, so the
    converter quantizes them from the float32-widened source values with gguf-py's quantizers; every other tensor
    keeps the F16 / F32 choice of the file type.
    """
    try:
        hparams = ModelBase.load_hparams(ckpt, False)
        arch = get_model_architecture(hparams, ModelType.TEXT)
        cls = get_model_class(arch)
        # MTP = multi-token-prediction head (speculative decoding), carried by some Qwen3.5 / Qwen3-Next
        # models.  The engine has no use for it here, and without this flag the Qwen converters insist
        # that the config declares one.  Set on the class only for the duration of construction.
        had = "no_mtp" in cls.__dict__
        old = cls.__dict__.get("no_mtp")
        cls.no_mtp = True
        try:
            conv = cls(ckpt, gguf.LlamaFileType.MOSTLY_F16, tmp_out, hparams=hparams)
        finally:
            if had:
                cls.no_mtp = old
            else:
                del cls.no_mtp
    except ConversionError:
        raise
    except Exception as e:
        raise ConversionError(f"cannot set up llama.cpp's converter for {ckpt} ({type(e).__name__}: {e})") from e
    conv._SHARED_INPUT_MERGES = ()
    if head_type != "f16":
        qtype = _HEAD_QTYPE[head_type]
        base_force = conv.tensor_force_quant
        head_keys = (gguf.MODEL_TENSOR.TOKEN_EMBD, gguf.MODEL_TENSOR.OUTPUT)

        def force_head(name: str, new_name: str, bid: int | None, n_dims: int):
            """Q8_0 / Q6_K for the embedding and the output head, the converter's own choice for everything else."""
            if n_dims >= 2 and any(conv.match_model_tensor_name(new_name, k, bid) for k in head_keys):
                return qtype
            return base_force(name, new_name, bid, n_dims)

        conv.tensor_force_quant = force_head
    logger.info("converter class %s (arch %s), head type %s", cls.__name__, conv.model_arch.name, head_type)
    return conv


def convert_checkpoint(ckpt_dir: Path, out: Path, head_type: str = "f16", validate: bool = True,
                       assume_no_littlebit_config: bool = False) -> ConversionReport:
    """Convert and (by default) validate; same as `convert` but returns a `ConversionReport`.

    Steps: (1) check the inputs; (2) load and cross-check the LittleBit params with
    `reference.load_checkpoint` (duplicate keys, shape mismatches -> ConversionError naming
    the layer); (3) run llama.cpp's converter over the NON-LittleBit tensors (this also writes
    the metadata and vocabulary); (4) probe every LittleBit linear and add its B1/F32 tensors;
    (5) write to a temp file next to `out`; (6) read it back and validate every linear against
    the reference; (7) rename onto `out` only if everything passed.  On any failure no
    output and no temp file remain.
    """
    ckpt_dir, out = Path(ckpt_dir), Path(out)
    lbcfg = _check_inputs(ckpt_dir, head_type, assume_no_littlebit_config)
    _memory_preflight(ckpt_dir)
    try:
        params = ref.load_checkpoint(ckpt_dir)
    except ValueError as e:
        raise ConversionError(str(e)) from e
    if not params:
        raise ConversionError(f"{ckpt_dir}: no LittleBit linear found (no <module>.U_packed tensors)")
    has_res = [m for m, p in params.items() if p.residual is not None]
    if not assume_no_littlebit_config and "residual" in lbcfg and bool(lbcfg["residual"]) != bool(has_res):
        raise ConversionError(f"littlebit_config.json says residual={lbcfg['residual']} but "
                              f"{len(has_res)} of {len(params)} linears have a residual path")
    for mod, p in params.items():
        if p.residual is not None and (p.residual.U.shape[0], p.residual.V.shape[1]) != (p.primary.U.shape[0], p.primary.V.shape[1]):
            raise ConversionError(f"{mod}: residual path shape differs from the primary path")

    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=out.name + ".", suffix=".partial", dir=out.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        report = _build(ckpt_dir, params, tmp, validate, head_type)
        report.config_check_skipped = bool(assume_no_littlebit_config)
        gc.collect()  # drop any memory map of `tmp` before renaming it (Windows refuses otherwise)
        os.replace(tmp, out)
        return report
    finally:
        gc.collect()
        try:
            tmp.unlink()  # only still there when something failed
        except FileNotFoundError:
            pass
        except OSError as e:  # e.g. Windows still holds a memory map; must not mask the real error
            logger.warning("could not remove temporary file %s (%s)", tmp, e)


def _build(ckpt: Path, params: dict[str, ref.LBLinearParams], tmp: Path, validate: bool,
           head_type: str = "f16") -> ConversionReport:
    """Everything between 'inputs checked' and 'rename': build the GGUF in `tmp` and validate it."""
    report = ConversionReport(n_linears=len(params), head_type=head_type)
    conv = _make_converter(ckpt, tmp, head_type)
    report.arch = conv.model_arch.name.lower()

    # (3) Drop the LittleBit tensors from the converter's index so it only handles plain ones.
    # `_source_tensor_names` maps each source key to the name the converter indexed it under.
    lb_filtered: set[str] = set()
    for src, filt in conv._source_tensor_names.items():
        if _lb_module_of(src) in params and filt is not None:
            lb_filtered.add(filt)
    for filt in lb_filtered:
        conv.model_tensors.pop(filt, None)
    report.dropped = sorted(src for src, filt in conv._source_tensor_names.items()
                            if filt is None and _lb_module_of(src) not in params)
    keys = set(_checkpoint_keys(ckpt))
    report.biases = sorted(f"{m}.bias" for m in params if f"{m}.bias" in keys)
    if report.biases:
        # Biases are plain tensors: the HF converter carries them over, but whether the engine's loader
        # for this architecture READS a bias next to a factored weight is the loader's business (Qwen3's
        # does not).  Say so loudly instead of letting it pass unnoticed.
        logger.warning("checkpoint has %d bias tensor(s) on LittleBit linears (%s ...); they are written to the GGUF "
                       "as ordinary bias tensors; make sure the engine loader for this architecture uses them",
                       len(report.biases), report.biases[0])
    layer_drops = [s for s in report.dropped if re.search(r"layers\.\d+\.", s) and not s.startswith("mtp.")]
    if layer_drops:
        raise ConversionError(f"llama.cpp's converter drops decoder-layer tensors that are not LittleBit linears "
                              f"(would be silently missing): {layer_drops[:5]}")
    try:
        conv.prepare_tensors()
    except Exception as e:
        raise ConversionError(f"llama.cpp's converter failed on the non-LittleBit tensors ({type(e).__name__}: {e})") from e
    writer = conv.gguf_writer

    # (4) The LittleBit linears.
    probe_cache: dict[tuple, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    layer_types = conv.hparams.get("layer_types") or (conv.hparams.get("text_config") or {}).get("layer_types")
    for mod in sorted(params):
        p = params[mod]
        bid, sub = _split_layer_name(mod)
        out_f, in_f = p.primary.U.shape[0], p.primary.V.shape[1]
        ltype = layer_types[bid] if (layer_types and bid is not None and bid < len(layer_types)) else None
        key = (ltype, sub, out_f, in_f)
        if key not in probe_cache:
            probes = probe_linear(conv, mod + ".weight", bid, out_f, in_f)
            if not probes:
                raise ConversionError(f"{mod}: llama.cpp's converter produced no GGUF tensor for this linear "
                                      "(unsupported or dropped); refusing to continue without it")
            # Store names with the layer index abstracted so other layers of the same kind reuse the probe.
            probe_cache[key] = [(re.sub(r"^blk\.\d+\.", "blk.{}.", pr.gguf_name), pr.row_idx, pr.col_idx) for pr in probes]
        report.probes[mod] = []
        for tmpl, ri, ci in probe_cache[key]:
            gname = tmpl.format(bid) if bid is not None else tmpl
            base = gname.removesuffix(".weight")
            report.probes[mod].append(Probe(gname, ri, ci))
            for sfx, path in (("", p.primary), ("r", p.residual)):
                if path is None:
                    continue
                arrays = _build_path(path, ri, ci, f"{mod}{'' if not sfx else ' (residual)'}")
                for k, arr in arrays.items():
                    name = f"{base}.{k}{sfx}"
                    try:
                        if k in ("lb_v", "lb_u"):
                            writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.B1)
                            report.lb_bytes += arr.size
                        else:
                            writer.add_tensor(name, arr)
                    except ValueError as e:
                        raise ConversionError(f"{mod}: cannot add GGUF tensor {name} ({e})") from e
                if sfx == "r":
                    # the residual scales once more, packed in the order GGML_OP_LITTLEBIT takes them
                    name = f"{base}.lb_s_pkr"
                    try:
                        writer.add_tensor(name, pack_residual_scales(arrays))
                    except ValueError as e:
                        raise ConversionError(f"{mod}: cannot add GGUF tensor {name} ({e})") from e

    # (5) Metadata (incl. vocabulary) and file.
    try:
        conv.prepare_metadata(vocab_only=False)
    except Exception as e:
        raise ConversionError(f"llama.cpp's converter failed on metadata/vocabulary ({type(e).__name__}: {e}); "
                              "does the checkpoint directory contain the tokenizer files?") from e
    _write_gemma3_swa_pattern(conv)
    report.written_names = sorted(n for shard in writer.tensors for n in shard)
    writer.write_header_to_file(path=tmp)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    del conv, writer
    gc.collect()

    # (6) Read back and check: counts first (nothing silently skipped), then numerics.
    _check_written(tmp, params, report, validate)
    return report


def _write_gemma3_swa_pattern(conv: ModelBase) -> None:
    """Gemma 3: tell the engine which layers use sliding-window attention, from the config's `layer_types`.

    llama.cpp's Gemma3Model converter writes no per-layer pattern, and the engine then assumes the real
    Gemma 3 layout (every 6th layer global, the others sliding-window with their own rope base).  A config
    with any other `layer_types` (a custom or tiny model) would run silently wrong.  The engine does read
    the array key `gemma3.attention.sliding_window_pattern` (bool per layer, true = sliding window) when present, so it is
    written here from `layer_types`.  Without `layer_types` (older configs) the key is left out and the engine default (the
    real 5:1 layout) applies; an integer `sliding_window_pattern` other than 1 or 6 is written as the integer (6 is the engine default; 1 is left out too).
    An unknown layer type or a wrong list length is refused: guessing would give wrong logits.
    """
    if conv.model_arch != gguf.MODEL_ARCH.GEMMA3:
        return
    hp = conv.hparams
    layer_types = hp.get("layer_types") or (hp.get("text_config") or {}).get("layer_types")
    n_layer = conv.block_count
    if layer_types:
        known = {"sliding_attention", "full_attention"}
        if len(layer_types) != n_layer or not set(layer_types) <= known:
            raise ConversionError(
                f"Gemma 3 layer_types {list(layer_types)} is not supported: expected {n_layer} entries, each "
                f"'sliding_attention' or 'full_attention'")
        conv.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in layer_types])
        return
    pattern = hp.get("sliding_window_pattern")
    if isinstance(pattern, int) and pattern not in (1, 6):
        conv.gguf_writer.add_sliding_window_pattern(pattern)


def _check_written(path: Path, params: dict[str, ref.LBLinearParams], report: ConversionReport, validate: bool) -> None:
    """Read the written GGUF back and verify counts and (if `validate`) every linear numerically.

    Why read back: the arrays the converter held in memory are not what the engine sees; the
    file is.  Reading it with gguf-py's reader checks the shapes, types and byte layout
    that actually reached the disk, and the validator decodes the B1 bytes independently.

    The checking itself returns an error MESSAGE instead of raising: gguf-py memory-maps the
    file, and an exception in flight keeps those mappings alive through its traceback, which
    on Windows would stop `convert_checkpoint` from deleting the temp file.  So the mapping
    is released first, then the error is raised.
    """
    msg = _verify(path, params, report, validate)
    gc.collect()
    if msg is not None:
        raise ConversionError(msg)


def _verify(path: Path, params: dict[str, ref.LBLinearParams], report: ConversionReport, validate: bool) -> str | None:
    """The checks of `_check_written`; returns None when fine, else the error message."""
    reader = gguf.GGUFReader(str(path))
    by_name = {t.name: t for t in reader.tensors}
    # the head tensors must have the requested type: llama.cpp's converter falls back to F16 with only a warning when
    # a type cannot take the row length (e.g. Q6_K and a hidden size that is not a multiple of 256)
    want = _HEAD_QTYPE[report.head_type]
    for hname in _HEAD_TENSORS:
        t = by_name.get(hname)
        if t is not None and t.tensor_type != want:
            return (f"{hname} was written as {t.tensor_type.name}, not {want.name} as --head-type {report.head_type} "
                    f"asks (row length {int(t.shape[0])}: Q8_0 needs a multiple of 32, Q6_K of 256)")
    n_lb_v = sum(1 for n in by_name if n.endswith(".lb_v"))
    expected = sum(len(v) for v in report.probes.values())
    if n_lb_v != expected or set(report.probes) != set(params):
        return (f"count check failed: {len(params)} checkpoint linears / {expected} expected GGUF linears, "
                f"{n_lb_v} lb_v tensors found in the file")
    for mod in sorted(params):
        p = params[mod]
        for pr in report.probes[mod]:
            base = pr.gguf_name.removesuffix(".weight")
            group: dict[str, Any] = {}
            for sfx, path_ in (("", p.primary), ("r", p.residual)):
                if path_ is None:
                    continue
                for k in ("lb_v", "lb_u", "lb_s_v2", "lb_s_mid", "lb_s_u1"):
                    t = by_name.get(f"{base}.{k}{sfx}")
                    if t is None:
                        return f"{mod}: tensor {base}.{k}{sfx} is missing from the written file"
                    group[k + sfx] = t
            if f"{base}.lb_vr" in by_name and p.residual is None:
                return f"{mod}: residual tensors written for a checkpoint linear without a residual path"
            packed = by_name.get(f"{base}.lb_s_pkr")
            if p.residual is None and packed is not None:
                return f"{mod}: {base}.lb_s_pkr written for a checkpoint linear without a residual path"
            if p.residual is not None:
                if packed is None:
                    return f"{mod}: tensor {base}.lb_s_pkr is missing from the written file"
                try:
                    group_pk = _validate.with_packed_residual_scales(group, packed)
                except ValueError as e:
                    return f"{mod}: {e}"
                for k in ("lb_s_v2r", "lb_s_midr", "lb_s_u1r"):
                    a = np.asarray(group[k].data, dtype=np.float32).reshape(-1)
                    if not np.array_equal(a.view(np.uint32), np.asarray(group_pk[k].data).view(np.uint32)):
                        return f"{mod}: {base}.lb_s_pkr does not hold the bytes of {base}.{k}"
            if not validate:
                continue
            tol = _validate.tolerance(p)
            try:
                err = _validate.validate_layer(p, group, row_idx=pr.row_idx, col_idx=pr.col_idx)
            except ValueError as e:
                return f"{mod}: {e}"
            try:
                err64 = _validate.validate_layer_exact(p, group, row_idx=pr.row_idx, col_idx=pr.col_idx)
            except ValueError as e:
                return f"{mod}: {e}"
            if not err64 <= _validate.EXACT_TOL:
                return (f"{mod} -> {pr.gguf_name}: float64 exact check failed: error {err64:.3e} exceeds "
                        f"{_validate.EXACT_TOL:.0e} (the conversion must be bit-exact; tolerance of the float32 check "
                        f"was {tol:.3e})")
            if p.residual is not None:
                # the packed residual scales, checked on their own against the float64 reference decode
                try:
                    err64_pk = _validate.validate_layer_exact(p, group_pk, row_idx=pr.row_idx, col_idx=pr.col_idx)
                except ValueError as e:
                    return f"{mod}: {e}"
                if not err64_pk <= _validate.EXACT_TOL:
                    return (f"{mod} -> {pr.gguf_name}: float64 exact check with the packed residual scales "
                            f"({base}.lb_s_pkr) failed: error {err64_pk:.3e} exceeds {_validate.EXACT_TOL:.0e}")
            report.tolerance = tol if not report.tolerance else min(report.tolerance, tol)
            report.max_rel_err = max(report.max_rel_err, err)
            if not err <= tol:
                return f"{mod} -> {pr.gguf_name}: max relative error {err:.3e} exceeds tolerance {tol:.3e}"
            report.n_validated += 1
    return None


def convert(ckpt_dir: Path, out: Path, head_type: str = "f16", validate: bool = True,
            assume_no_littlebit_config: bool = False) -> None:
    """Convert the Samsung-format LittleBit checkpoint in `ckpt_dir` to a B1 GGUF at `out`.

    `head_type`: precision of the embedding / output head, "f16" (default), "q8_0" or "q6_k" (see "Head type" in the
    module docstring; a row length the type cannot take is a ConversionError).  With `validate=True` (default)
    every LittleBit linear is read back from the written file and compared with the PyTorch
    reference; any error above tolerance, a missing/skipped linear or a bad input raises
    `ConversionError` and leaves no file at `out` (the file is written to a temp name and
    renamed only after validation).  `assume_no_littlebit_config=True` lets a checkpoint without
    `littlebit_config.json` through: only the residual cross-check against that file is skipped
    (with a warning, recorded in `ConversionReport.config_check_skipped`).
    """
    convert_checkpoint(ckpt_dir, out, head_type, validate, assume_no_littlebit_config)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry: `python -m tools.convert.lb_to_gguf <ckpt_dir> -o <out.gguf>`."""
    ap = argparse.ArgumentParser(description="Convert a Samsung-format LittleBit checkpoint to a B1 GGUF.")
    ap.add_argument("ckpt", type=Path, help="checkpoint directory (safetensors + config.json + tokenizer + littlebit_config.json)")
    ap.add_argument("-o", "--out", type=Path, required=True, help="output .gguf path")
    ap.add_argument("--head-type", default="f16", choices=_HEAD_TYPES,
                    help="precision of the token embedding / output head (default f16; q8_0 and q6_k shrink it)")
    ap.add_argument("--assume-no-littlebit-config", action="store_true",
                    help="accept a checkpoint without littlebit_config.json (skips only the residual cross-check)")
    ap.add_argument("--no-validate", action="store_true", help="skip the per-layer read-back validation (not recommended)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        report = convert_checkpoint(args.ckpt, args.out, args.head_type, not args.no_validate,
                                    args.assume_no_littlebit_config)
    except ConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {args.out}: {args.out.stat().st_size / 2**20:.1f} MiB, {report.n_linears} LittleBit linears "
          f"({report.lb_bytes / 2**20:.1f} MiB of B1 signs), validated {report.n_validated}, "
          f"max rel err {report.max_rel_err:.2e} (tolerance {report.tolerance:.2e}), head {report.head_type}"
          + (f"; WARNING {len(report.biases)} bias tensors on LittleBit linears carried over" if report.biases else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
