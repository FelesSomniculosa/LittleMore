"""Numerical validation of a converted LittleBit GGUF against the PyTorch reference.

What: `validate_layer` re-decodes the tensors of ONE LittleBit linear exactly as they
      were written in the GGUF file (B1 sign bytes + F32 scale vectors), runs the
      padded factor chain on random inputs in float64, and compares the result with
      `reference.LBLinear` built from the original Samsung-format checkpoint.
Why:  the converter reorders rows/columns, flips the bit polarity, folds two scales
      into one and pads the rank.  Each of those is a way to silently produce a
      wrong model.  Reading the file back (instead of reusing the in-memory arrays
      the converter had) also catches mistakes in the byte layout and in the GGUF
      writing itself, which is the part the engine will actually see.

Independence: the B1 bit decoding below (`decode_b1_rows`) is written here from the
format definition (bit 1 = +1, bit 0 = -1, element i of a block in byte i // 8, bit
i % 8, least significant bit first) and does not call any helper of the writer
(`lb_to_gguf.py`).  The reference side uses `reference.unpack_signs` through
`reference.load_checkpoint`, which decodes the *Samsung* convention (bit 0 = +1).
If the two conventions were confused, the checks fail.

Math checked (names: GGUF tensor suffix -> checkpoint scale, see reference.py):

    y = s_u1 * ( Ub @ ( s_mid * ( Vb^T @ ( s_v2 * x ) ) ) )  [+ the same for the residual `r` path]

    lb_s_u1 = u1,   lb_s_mid = v1 * u2 (zero-padded to a multiple of 128),   lb_s_v2 = v2

Tolerance: see `tolerance`.  Terms (LittleBit, GGUF, rank) are defined in the
glossary of docs/development.md.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

import reference as ref

# float32 unit roundoff, 2**-24: the relative error of one rounded float32 operation.
_U32 = 2.0 ** -24

# Rows of a sign matrix decoded to float64 at once; bounds the transient to ~chunk*cols*8 bytes.
_CHUNK_ROWS = 512

# Suffixes of the five tensors of one path; the residual path appends "r" to each.
_PATH_KEYS = ("lb_v", "lb_u", "lb_s_v2", "lb_s_mid", "lb_s_u1")


def tolerance(params: ref.LBLinearParams) -> float:
    """Largest relative error (2-norm of the difference / 2-norm of the reference) accepted.

    Derivation.  The converter copies the scale vectors bit-exactly (float32 -> float32,
    products v1*u2 formed in float32 exactly as the reference forms them) and the signs are
    exact, so the two sides compute the same real number.  The validator accumulates in
    float64 (error ~1e-16, negligible), so the difference is the rounding error of the
    REFERENCE's own float32 evaluation.  A float32 dot product of n terms accumulates
    rounding error that, with independent roundings, grows like sqrt(n) * u relative to the
    norm of the result (u = 2**-24; the worst case n*u is far too pessimistic for random
    rounding).  The chain does two long dot products (length d_in, then length r per path),
    so n = d_in + sum of the padded ranks, and we allow a safety factor of 16 on top of the
    sqrt(n) * u estimate for the extra elementwise roundings and summation-order effects:

        tol = 16 * sqrt(d_in + r_primary_padded + r_residual_padded) * 2**-24

    (about 3e-5 for a 256-wide layer, 1.5e-4 for an 8B-sized one).  Measured errors on the
    tiny test models are ~1e-7, i.e. 100x below this; a single flipped sign or a 2x wrong
    scale is 100-1000x above it (tests/convert/test_convert.py, test_validator_catches_corruption).
    """
    n = params.primary.V.shape[1]
    for path in (params.primary, params.residual):
        if path is not None:
            n += -(-path.V.shape[0] // 128) * 128
    return 16.0 * float(np.sqrt(n)) * _U32


def decode_b1_rows(data: np.ndarray, r0: int, r1: int) -> np.ndarray:
    """Decode rows [r0, r1) of a B1 byte matrix into float64 +/-1 values.

    `data` is (rows, bytes_per_row) uint8 as stored in the GGUF.  Written from the format
    definition: bit 1 = +1, bit 0 = -1; element i of a row is bit (i % 8) of byte (i // 8),
    least significant bit first (so `bitorder="little"`).  There is no scale in a B1 block.
    """
    bits = np.unpackbits(data[r0:r1], axis=-1, bitorder="little")  # (rows, bytes*8), 0/1
    return bits.astype(np.float64) * 2.0 - 1.0  # 1 -> +1, 0 -> -1


def _b1_matrix_shape(t: Any, what: str) -> tuple[int, int]:
    """(rows, cols) of a B1 tensor read from a GGUF, validating that bytes match the shape.

    GGUF stores shapes as [ne0, ne1] with ne0 the contiguous dimension, so a tensor of
    logical (rows, cols) has shape (cols, rows) and `rows` x `cols / 8` bytes.
    """
    cols, rows = int(t.shape[0]), int(t.shape[1])
    if cols % 128:
        raise ValueError(f"{what}: row length {cols} is not a multiple of the 128-sign B1 block")
    data = np.asarray(t.data)
    if data.dtype != np.uint8 or data.shape != (rows, cols // 8):
        raise ValueError(f"{what}: data is {data.dtype} {data.shape}, expected uint8 {(rows, cols // 8)} for shape {(cols, rows)}")
    return rows, cols


def _path_forward(x: np.ndarray, g: Mapping[str, Any], sfx: str, what: str) -> np.ndarray:
    """One path (`sfx` "" or "r") of the padded chain, from GGUF tensors only, in float64.

    y = s_u1 * (U @ (s_mid * (V^T @ (s_v2 * x)))): V is stored (r_pad, n_in) so `V^T x` is
    `x @ V.T`; U is stored (n_out, r_pad).  Sign matrices are decoded a chunk of rows at a time.
    """
    tv, tu = g["lb_v" + sfx], g["lb_u" + sfx]
    r_pad, n_in = _b1_matrix_shape(tv, f"{what}.lb_v{sfx}")
    n_out, r_u = _b1_matrix_shape(tu, f"{what}.lb_u{sfx}")
    if r_u != r_pad:
        raise ValueError(f"{what}: lb_u{sfx} has rank {r_u} but lb_v{sfx} has {r_pad}")
    if r_pad % 128:
        raise ValueError(f"{what}: padded rank {r_pad} is not a multiple of 128")
    s_v2 = np.asarray(g["lb_s_v2" + sfx].data, dtype=np.float64).reshape(-1)
    s_mid = np.asarray(g["lb_s_mid" + sfx].data, dtype=np.float64).reshape(-1)
    s_u1 = np.asarray(g["lb_s_u1" + sfx].data, dtype=np.float64).reshape(-1)
    if (s_v2.size, s_mid.size, s_u1.size) != (n_in, r_pad, n_out):
        raise ValueError(f"{what}: scale lengths {(s_v2.size, s_mid.size, s_u1.size)} do not match "
                         f"(n_in, r_pad, n_out) = {(n_in, r_pad, n_out)}")
    if x.shape[1] != n_in:
        raise ValueError(f"{what}: input width {x.shape[1]} != n_in {n_in}")
    xs = x * s_v2
    h = np.empty((x.shape[0], r_pad), dtype=np.float64)
    vdata = np.asarray(tv.data)
    for r0 in range(0, r_pad, _CHUNK_ROWS):
        r1 = min(r0 + _CHUNK_ROWS, r_pad)
        h[:, r0:r1] = xs @ decode_b1_rows(vdata, r0, r1).T
    h *= s_mid  # zero in the padding, so padded sign bits contribute exactly nothing
    y = np.empty((x.shape[0], n_out), dtype=np.float64)
    udata = np.asarray(tu.data)
    for r0 in range(0, n_out, _CHUNK_ROWS):
        r1 = min(r0 + _CHUNK_ROWS, n_out)
        y[:, r0:r1] = h @ decode_b1_rows(udata, r0, r1).T
    return y * s_u1


def validate_layer(ckpt_params: ref.LBLinearParams, gguf_tensors: Mapping[str, Any],
                   n_samples: int = 4, row_idx: np.ndarray | None = None,
                   col_idx: np.ndarray | None = None, seed: int = 0) -> float:
    """Max relative error between the reference layer and the layer decoded from GGUF tensors.

    `ckpt_params`: `reference.LBLinearParams` of the linear (from `reference.load_checkpoint`).
    `gguf_tensors`: the tensors of ONE GGUF linear keyed by suffix (`lb_v`, `lb_u`, `lb_s_v2`,
    `lb_s_mid`, `lb_s_u1`, and the residual `lb_vr`, ... if present); each needs `.data` (the
    stored array) and `.shape` (GGUF [ne0, ne1]) -- a `gguf.GGUFReader` tensor fits.
    `row_idx` / `col_idx`: the converter's row / column permutation: GGUF output row i is
    checkpoint output row `row_idx[i]`, GGUF input column j reads checkpoint input `col_idx[j]`
    (None = identity).  Draws `n_samples` standard-normal inputs, evaluates the reference in
    float32 (as the engine would see it) and the GGUF tensors in float64, and returns
    the largest ||y_gguf - y_ref||_2 / ||y_ref||_2 over the samples.  Raises ValueError when the
    tensors are inconsistent with each other (wrong shapes / lengths), never for a numeric mismatch.
    Compare the result with `tolerance(ckpt_params)`.
    """
    what = "linear"
    n_in = ckpt_params.primary.V.shape[1]
    gen = np.random.default_rng(seed)
    x = gen.standard_normal((n_samples, n_in)).astype(np.float32)
    with torch.no_grad():
        y_ref = ref.LBLinear.from_params(ckpt_params)(torch.from_numpy(x)).numpy().astype(np.float64)
    if row_idx is not None:
        y_ref = y_ref[:, np.asarray(row_idx)]
    xg = x if col_idx is None else x[:, np.asarray(col_idx)]
    y = 0.0
    n_paths = 0
    for sfx in ("", "r"):
        if "lb_v" + sfx in gguf_tensors:
            missing = [k + sfx for k in _PATH_KEYS if k + sfx not in gguf_tensors]
            if missing:
                raise ValueError(f"{what}: GGUF path '{sfx or 'primary'}' is missing tensors {missing}")
            y = y + _path_forward(xg.astype(np.float64), gguf_tensors, sfx, what)
            n_paths += 1
    if n_paths == 0:
        raise ValueError(f"{what}: no lb_v tensor given")
    err = 0.0
    for i in range(n_samples):
        denom = np.linalg.norm(y_ref[i])
        diff = np.linalg.norm(y[i] - y_ref[i])
        err = max(err, float(diff / denom) if denom > 0 else float(diff))
    return err
