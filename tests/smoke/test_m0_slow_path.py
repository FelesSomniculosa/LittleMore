"""M0 smoke test: first light for the whole slow-path chain.

    Qwen3-0.6B (HF) -> make_lb2_init (init-only LittleBit-2, 0.55 bpw)
                    -> engine lb_to_gguf.py (Q1_0 path) -> llama.cpp fork

It checks that (1) llama-cli loads the converted GGUF and generates on both GPU
and CPU, (2) exactly 8 tokens are produced, and (3) the last-position logits of
the engine agree with the pure-PyTorch reference (tools/dev/reference.py).

The model is init-only (no QAT), so the generated text is gibberish by design;
only the plumbing and the numerics are under test.

Skipped automatically when the cached base model or the engine binaries are
missing, so CI without those assets stays green.  Generated artifacts are
cached under .cache/ and reused, so re-runs are fast.

Numeric tolerance -- why it is NOT 1e-2
---------------------------------------
llama.cpp multiplies a Q1_0 weight by an F32 activation by first quantizing the
activation to Q8_0 (ggml-cpu: vec_dot_type = Q8_0; CUDA does the same with
Q8_1).  The reference keeps activations in fp32.  Every LittleBit linear has two
binary matmuls per path, so that int8 activation noise is injected ~400 times
per forward.  Measured on this model: engine(CPU) vs fp32 reference gives
max-abs-diff 2.02 / rms 0.41 / corr 0.992 on logits spanning [-18, 16]; an
independent PyTorch emulation of Q8_0 activation quantization differs from the
fp32 reference by a similar amount (max 3.0, rms 0.59), i.e. the gap is
quantization noise, not a structural bug (a wrong permutation or sign
convention would decorrelate the logits completely).  The thresholds below are
that measured baseline plus a margin.  They are a regression guard, not a claim
of 1e-2 accuracy.
"""
from __future__ import annotations

import pytest

import _assets as A

pytestmark = pytest.mark.smoke

# Baselines measured on the CPU backend (-ngl 0, deterministic), see header.
MAX_ABS_DIFF_BASELINE = 2.02
MAX_ABS_DIFF_LIMIT = 2.5      # baseline + ~25% margin (thread-order jitter)
MIN_CORRELATION = 0.98        # measured 0.992
MIN_TOP5_OVERLAP = 3          # measured 4 of 5


@pytest.fixture(scope="module")
def gguf_path():
    reason = A.missing_assets()
    if reason:
        pytest.skip(reason)
    return A.ensure_gguf()


@pytest.mark.parametrize("ngl", [99, 0], ids=["gpu", "cpu"])
def test_llama_cli_generates(gguf_path, ngl):
    """llama-cli exits 0 on GPU and CPU; llama-completion reports 8 generated tokens."""
    res = A.run_llama_cli(gguf_path, "Hello", n_predict=8, ngl=ngl)
    assert res.returncode == 0, res.stdout[-2000:]
    assert "Generation:" in res.stdout  # llama-cli printed its timing line
    assert A.count_generated_tokens(gguf_path, "Hello", n_predict=8, ngl=ngl) == 8


def test_logits_match_reference(gguf_path):
    """Last-position logits agree with reference.py on the same fixed token ids."""
    import numpy as np

    engine, tokens = A.engine_logits(gguf_path, A.PROMPT, ngl=0)
    assert tokens == A.PROMPT_IDS, f"engine tokenization changed: {tokens}"
    ref = A.reference_logits(A.PROMPT_IDS)
    assert engine.shape == ref.shape

    diff = float(np.abs(engine - ref).max())
    corr = float(np.corrcoef(engine, ref)[0, 1])
    top5_e, top5_r = np.argsort(-engine)[:5], np.argsort(-ref)[:5]
    print(f"max_abs_diff={diff:.4f} corr={corr:.4f} top1={top5_e[0]}/{top5_r[0]}")

    assert top5_e[0] == top5_r[0], "top-1 token disagrees"
    assert len(set(top5_e) & set(top5_r)) >= MIN_TOP5_OVERLAP
    assert corr >= MIN_CORRELATION
    assert diff <= MAX_ABS_DIFF_LIMIT, f"{diff} > {MAX_ABS_DIFF_LIMIT} (baseline {MAX_ABS_DIFF_BASELINE})"
