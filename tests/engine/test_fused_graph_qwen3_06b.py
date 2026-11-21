"""Task 8, real model: init-only LittleBit-2 Qwen3-0.6B through the fused op, on CPU and with GPU offload.

The checkpoint and the B1 GGUF come from tests/smoke/_assets.py (cached under .cache/, built on first use with
tools/convert/lb_to_gguf.py); the Q1_0 slow-path baseline is derived from the B1 file (`b1_to_q1_0_gguf`).
Compared with `reference.py` (fp32, CPU) on the engine's own token ids: one token ("Hello") and a 16-token prompt.

Measured numbers behind the thresholds (logit rms about 2.5-2.9; max-abs over 151936 logits):

  config                              1 token      16 tokens
  CPU, default (flash attn, f16 KV)   1.35e-2      1.34e-2     <- the f16 attention path of flash attention
  CPU, -fa off, f32 KV                1.5e-4       9.8e-5      <- the fused op itself
  GPU (-ngl 99), default              1.45e-2      3.7e-2
  GPU (-ngl 99), -fa off, f32 KV      4.9e-3       6.0e-2      <- the CUDA half of the graph (f16 matmuls)
  slow path (Q1_0 chain), CPU         7.3e-1       1.36        <- 50x to 100x worse

The default CPU error is NOT the fused op: switching off flash attention and keeping K/V in f32 brings it to 1e-4.
With layers on the GPU, only the LittleBit linears stay on the CPU (see docs/engine-changes.md); the rest of the
graph runs in CUDA's f16 kernels, which adds a few 1e-2 at the logits.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

import _assets as A
from test_fused_graph_families import fmt, metrics

pytestmark = pytest.mark.engine_run

# Exact configuration: CPU, no flash attention, f32 KV cache: measured 1.5e-4 worst; limit 1e-3 (6x margin).
EXACT_ARGS = ("-fa", "off", "-ctk", "f32", "-ctv", "f32")
CPU_EXACT_MAX_ABS = 1e-3
# Default configuration on CPU: measured 1.35e-2; limit 2.5e-2 (1.9x margin, 1% of the logit scale).
CPU_DEFAULT_MAX_ABS = 2.5e-2
# With GPU offload: measured up to 6.0e-2; limit 1.0e-1 (1.7x margin, 3.5% of the logit scale).
GPU_MAX_ABS = 1.0e-1
# The Q1_0 slow path is at least 50x worse in every measurement; require at least 10x.
SLOW_MIN_RATIO = 10.0


@pytest.fixture(scope="module")
def ggufs():
    reason = A.missing_assets()
    if reason:
        pytest.skip(reason)
    return A.ensure_b1_gguf(), A.ensure_gguf()


@pytest.mark.parametrize("ngl", [0, 99], ids=["cpu", "gpu"])
@pytest.mark.parametrize("prompt", [A.PROMPT_ONE, A.PROMPT_LONG], ids=["1tok", "16tok"])
def test_fused_matches_reference_and_beats_slow_path(ggufs, prompt, ngl):
    b1, q1 = ggufs
    fused, ids = A.engine_logits(b1, prompt, ngl)
    slow, ids_slow = A.engine_logits(q1, prompt, ngl)
    assert ids == ids_slow
    want = A.reference_logits(ids)
    mf, ms = metrics(fused, want), metrics(slow, want)
    print(f"ngl={ngl} {len(ids)}tok fused: {fmt(mf)}\nngl={ngl} {len(ids)}tok slow : {fmt(ms)}  ({ms['max_abs'] / mf['max_abs']:.0f}x)")
    assert mf["max_abs"] <= (GPU_MAX_ABS if ngl else CPU_DEFAULT_MAX_ABS)
    assert mf["top1"]
    assert ms["max_abs"] >= SLOW_MIN_RATIO * mf["max_abs"]
    if ngl == 0:
        exact, _ = A.engine_logits(b1, prompt, 0, EXACT_ARGS)
        me = metrics(exact, want)
        print(f"ngl=0 {len(ids)}tok fused, -fa off f32 KV: {fmt(me)}")
        assert me["max_abs"] <= CPU_EXACT_MAX_ABS and me["top1"]
        # teeth: the same engine logits against a reference without the residual path
        nores = metrics(exact, A.reference_logits(ids, drop_residual=True, quantize=False, kv_f16=False))
        assert nores["max_abs"] >= 100 * me["max_abs"]


@pytest.mark.parametrize("ngl", [0, 99], ids=["cpu", "gpu"])
def test_generation_exits_zero(ggufs, ngl):
    """llama-cli and llama-completion generate 8 tokens from the B1 GGUF at both offload settings."""
    b1, _ = ggufs
    res = A.run_llama_cli(b1, "Hello", n_predict=8, ngl=ngl)
    assert res.returncode == 0, res.stdout[-2000:]
    assert A.count_generated_tokens(b1, "Hello", n_predict=8, ngl=ngl) == 8


def _tokens_per_second(gguf, ngl: int = 0, n: int = 32) -> float:
    r = A._run([A.BIN / ("llama-completion" + A.EXE), "-m", gguf, "-p", "Hello", "-n", n, "-ngl", ngl, "-no-cnv"])
    assert r.returncode == 0, r.stdout[-1500:]
    m = re.search(r"(?<!prompt )eval time =.*?([\d.]+) tokens per second", r.stdout.replace("prompt eval time", "PE"))
    assert m, r.stdout[-1500:]
    return float(m.group(1))


def test_cpu_speed_fused_vs_chain_is_reported(ggufs):
    """Informational: generation speed, fused op vs the Q1_0 chain, CPU. Only checks both run."""
    b1, q1 = ggufs
    f, s = _tokens_per_second(b1), _tokens_per_second(q1)
    print(f"CPU generation: fused (B1) {f:.2f} tokens/s, Q1_0 chain {s:.2f} tokens/s")
    assert f > 0 and s > 0
