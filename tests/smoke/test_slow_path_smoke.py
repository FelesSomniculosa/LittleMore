"""Smoke test: first light for the whole slow-path chain.

    Qwen3-0.6B (HF) -> make_lb2_init (init-only LittleBit-2, 0.55 bpw)
                    -> engine lb_to_gguf.py (Q1_0 path) -> llama.cpp fork

It checks that (1) llama-cli loads the converted GGUF and generates on GPU and
CPU, (2) exactly 8 tokens are produced, and (3) the engine's numerics agree with
the pure-PyTorch reference (tools/dev/reference.py) -- see below for how.

The model is init-only (no QAT), so generated text is gibberish by design; only
plumbing and numerics are under test.  The test skips when the cached base model
or engine binaries are missing (CI stays green) and caches generated artifacts
under .cache/ keyed on a hash of the tool sources (see _assets.py).

What the numeric checks establish (and what they do not)
--------------------------------------------------------
llama.cpp multiplies a Q1_0 weight by an F32 activation by first quantizing the
activation to Q8_0 (ggml-cpu: vec_dot_type = Q8_0; CUDA uses int8 blocks too),
and stores K/V in f16.  The plain fp32 reference has neither, so engine-vs-fp32
logits differ by ~2 (max-abs, CPU) -- that is NOT a 1e-2 comparison and cannot
be.  So the comparison target is tools/dev/emulate_q8_slow_path.py, the same
reference with ggml's Q8_0 activation quantization and f16 KV rounding
reproduced.  Three levels of evidence:

1. Single token ("Hello", position 0: RoPE and attention are trivial).  The
   whole computation is deterministic and identical on both sides, so engine ==
   emulation to ~4e-6 on CPU and GPU, and all 28 layer outputs agree to <2e-3 (sum per layer)
   (vs errors up to 5.5 against fp32).  This isolates conversion, scales, rank
   padding and the residual path.
2. Multi-token prompts.  Attention and softmax now differ at the last bit
   between ggml and PyTorch; Q8_0 rounding is a chaotic amplifier (a 1e-4
   change flips some rounding decisions, and flips compound over 28 layers), so
   engine and emulation can only agree to the "noise floor": the distance
   between two emulations that differ by nothing but f16-vs-f32 KV rounding.
   We assert engine-vs-emulation rms <= 2x that floor.
3. Mutation check (deliberately broken references): the same comparison against a reference with the residual
   path removed, or with q/k rows wrongly permuted (a RoPE-layout error), must
   be at least 2x worse than against the correct reference (measured 2.5x-20x).

Not detected: errors smaller than the noise floor (a few tenths of a logit rms
after 28 layers).  Single-token exactness bounds such errors for everything
except attention/RoPE over several positions, where only the mutation test
(gross layout errors) protects.  Top-1 agreement is asserted only for the
single token; for an init-only model top-1 of a multi-token prompt is
noise-dominated (it flips between two emulations).
"""
from __future__ import annotations

import re

import numpy as np
import pytest

import _assets as A

pytestmark = pytest.mark.smoke

# --- thresholds, tied to measured values -------------------
# Single token, engine vs emulation, max-abs over 151936 logits.  Measured
# 3.9e-6 on CPU and 3.8e-6 on GPU; 1e-4 leaves headroom for kernel/thread-order jitter.
SINGLE_TOKEN_MAX_ABS = 1e-4
# Per-layer output sum |engine - emulation|; measured <= 4e-4, sums are O(1..700).
LAYER_SUM_ABS = 2e-3
# Multi-token: engine-vs-emulation rms must be <= this x the noise-floor rms.
# Measured ratios 0.44-1.54 (GPU prefill is the noisiest); 2.0 is the limit.
FLOOR_RATIO_LIMIT = 2.0
# Mutations must be at least this x farther from the engine than the correct
# emulation is.  Measured minimum 2.5 (GPU, q/k permutation, 5-token prompt).
MUTATION_MIN_RATIO = 2.0


def rms(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square difference of two arrays."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


@pytest.fixture(scope="module")
def gguf_path():
    """The slow-path GGUF (skips the test when the model assets are missing)."""
    reason = A.missing_assets()
    if reason:
        pytest.skip(reason)
    return A.ensure_gguf()


@pytest.mark.parametrize("ngl", [99, 0], ids=["gpu", "cpu"])
def test_llama_cli_generates(gguf_path, ngl):
    """llama-cli exits 0 on GPU and CPU; llama-completion reports 8 generated tokens."""
    res = A.run_llama_cli(gguf_path, "Hello", n_predict=8, ngl=ngl)
    assert res.returncode == 0, res.stdout[-2000:]
    # llama-cli prints a timing line with "<number> t/s"; match the unit, not the wording.
    assert re.search(r"\d+(\.\d+)?\s*t/s", res.stdout), res.stdout[-500:]
    assert A.count_generated_tokens(gguf_path, "Hello", n_predict=8, ngl=ngl) == 8


@pytest.mark.parametrize("ngl", [0, 99], ids=["cpu", "gpu"])
def test_single_token_matches_emulation_exactly(gguf_path, ngl):
    """Position 0: engine logits == Q8_0-emulated reference (isolates conversion)."""
    engine, tokens = A.engine_logits(gguf_path, A.PROMPT_ONE, ngl=ngl)
    assert tokens == [9707]
    emu = A.reference_logits(tokens, quantize=True, kv_f16=True)
    fp = A.reference_logits(tokens)
    d_emu, d_fp = float(np.abs(engine - emu).max()), float(np.abs(engine - fp).max())
    print(f"single-token ngl={ngl}: |engine-emu|max={d_emu:.2e} |engine-fp32|max={d_fp:.3f}")
    assert d_emu <= SINGLE_TOKEN_MAX_ABS
    assert int(engine.argmax()) == int(emu.argmax()) == int(fp.argmax())
    assert d_fp > 100 * max(d_emu, 1e-6), "fp32 reference should be far worse than the emulation"
    # Dropping the residual path must be clearly detected (a test that fails when the code is wrong).
    no_res = A.reference_logits(tokens, drop_residual=True)
    assert float(np.abs(engine - no_res).max()) > 1.0


def test_single_token_layerwise_matches_emulation(gguf_path):
    """All 28 layer outputs (sum per layer) match the emulation; error does not grow with depth."""
    eng = A.engine_layer_sums(gguf_path, A.PROMPT_ONE, ngl=0)
    ref = A.layer_sums_reference([9707])
    assert sorted(eng) == list(range(len(ref))) == list(range(28))
    errs = [abs(eng[i] - ref[i]) for i in range(28)]
    print("layer sum errors, max =", max(errs))
    assert max(errs) <= LAYER_SUM_ABS


@pytest.mark.parametrize("ngl", [0, 99], ids=["cpu", "gpu"])
@pytest.mark.parametrize("prompt", [A.PROMPT, A.PROMPT_LONG], ids=["5tok", "16tok"])
def test_multi_token_logits_within_noise_floor(gguf_path, prompt, ngl):
    """Engine vs emulation is at the chaos floor; gross mutations are clearly worse."""
    engine, ids = A.engine_logits(gguf_path, prompt, ngl=ngl)
    if prompt == A.PROMPT:
        assert ids == A.PROMPT_IDS, f"engine tokenization changed: {ids}"
    emu = A.reference_logits(ids, quantize=True, kv_f16=True)
    # Noise floor: identical model, differing only by f16-vs-f32 KV rounding.
    floor = rms(emu, A.reference_logits(ids, quantize=True, kv_f16=False))
    d = rms(engine, emu)
    d_nores = rms(engine, A.reference_logits(ids, drop_residual=True))
    d_perm = rms(engine, A.reference_logits(ids, permute_qk=True))
    print(f"{len(ids)}tok ngl={ngl}: rms engine-emu={d:.3f} floor={floor:.3f} "
          f"no-residual={d_nores:.3f} permuted-qk={d_perm:.3f}")
    assert d <= FLOOR_RATIO_LIMIT * floor
    assert d_nores >= MUTATION_MIN_RATIO * d
    assert d_perm >= MUTATION_MIN_RATIO * d
