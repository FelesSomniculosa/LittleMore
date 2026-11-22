"""Real model: init-only LittleBit-2 Qwen3-0.6B through the fused op, on CPU and with GPU offload.

The checkpoint and the B1 GGUF come from tests/smoke/_assets.py (cached under .cache/, built on first use with
tools/convert/lb_to_gguf.py); the Q1_0 slow-path baseline is derived from the B1 file (`b1_to_q1_0_gguf`).
Compared with `reference.py` (fp32, CPU) on the engine's own token ids: one token ("Hello") and a 16-token prompt.

Measured max-abs over 151936 logits against the fp32 reference (logit rms 2.5 / 2.9).  These numbers are the values
printed (`pytest -s`) by test_fused_matches_reference_and_beats_slow_path and the two GPU tests below, recorded when
the tests were written; the thresholds further down are set from them.  The "50x to 100x" is the slow-path
error divided by the fused default-config error on the same prompt (0.73 / 1.35e-2 = 54, 1.36 / 1.34e-2 = 101):

  config                              1 token      16 tokens
  CPU, default (flash attn, f16 KV)   1.35e-2      1.34e-2     <- the f16 attention path of flash attention
  CPU, -fa off, f32 KV                1.5e-4       9.8e-5      <- the fused op itself
  GPU (-ngl 99), default              1.38e-2      3.14e-2     (LittleBit linears on the CUDA kernels)
  GPU (-ngl 99), -fa off, f32 KV      4.8e-3       5.53e-2
  GPU, -fa off, f32 KV, F32 head      2.6e-5       -           <- control: B1 GGUF with the F16 head widened to F32
  slow path (Q1_0 chain), CPU         7.3e-1       1.36        <- 50x to 100x worse

The GPU rows are with the LittleBit weights on the GPU and the fused op on the CUDA kernels (decode kernels for 1 token,
chunked decode kernels for 16).  When this file was first written the LittleBit tensors were pinned to the CPU and the
op ran there inside an otherwise offloaded graph; the GPU rows were then 1.45e-2 / 3.7e-2 (default), 4.9e-3 / 6.0e-2
(-fa off, f32 KV) and 1.1e-4 (F32 head, 1 token).

Where the GPU error comes from (numbers from `python tools/dev/dense_floor.py`, table in docs/engine-changes.md):
  * 1 token: the 4.8e-3 is the F16 output head on CUDA. With the head widened to F32 the same run is 2.6e-5.
  * 16 tokens: NOT the LittleBit op. The DENSE Qwen3-0.6B (no LittleBit at all, F32 weights) run through the same CUDA
    path against its own fp32 HF reference is 2.5e-2 (default) and 3.9e-2 (-fa off, f32 KV); LittleBit is 3.14e-2 and
    5.53e-2, i.e. 1.25x and 1.43x the dense floor (1.46x / 1.56x while the LittleBit op still ran on the CPU). The same
    dense model is 1.7e-5 at one token, so the floor appears only with several tokens, i.e. in the CUDA batched matmuls
    (ggml-cuda's cuBLAS handle allows TF32 for F32 matrix products; see docs/engine-changes.md). The LittleBit op itself
    is float32 on CUDA (1e-6 relative in test-littlebit-ref); the 1.25x-1.43x excess over the dense floor is NOT
    attributed (the LittleBit model has different activation magnitudes; not isolated).
"""
from __future__ import annotations

import os
import re
import subprocess

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
# With GPU offload: measured up to 5.5e-2 (6.0e-2 while the LittleBit op ran on the CPU); limit 1.0e-1 (1.8x margin,
# 3.5% of the logit scale).
GPU_MAX_ABS = 1.0e-1
# The Q1_0 slow path is at least 50x worse in every measurement; require at least 10x.
SLOW_MIN_RATIO = 10.0


@pytest.fixture(scope="module")
def ggufs():
    """(B1 GGUF, Q1_0 slow-path GGUF) of the cached Qwen3-0.6B LittleBit model; skips if the assets are missing."""
    reason = A.missing_assets()
    if reason:
        pytest.skip(reason)
    return A.ensure_b1_gguf(), A.ensure_gguf()


@pytest.mark.parametrize("ngl", [0, 99], ids=["cpu", "gpu"])
@pytest.mark.parametrize("prompt", [A.PROMPT_ONE, A.PROMPT_LONG], ids=["1tok", "16tok"])
def test_fused_matches_reference_and_beats_slow_path(ggufs, prompt, ngl):
    """The fused op's logits stay within the per-configuration limit of the fp32 reference and keep the same top-1
    token, and the old Q1_0 slow path is at least SLOW_MIN_RATIO x farther away.  On CPU with `-fa off` and f32 KV
    it must be near exact (the op itself), and a reference without the residual path must be far off (so the
    comparison can detect a missing residual).  Fails if the fused kernel, the loader or the graph is wrong."""
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
        # The same engine logits against a reference without the residual path
        nores = metrics(exact, A.reference_logits(ids, drop_residual=True, quantize=False, kv_f16=False))
        assert nores["max_abs"] >= 100 * me["max_abs"]


# GPU regression guards, from the dense-floor measurement (see the module docstring).
# LittleBit-vs-reference / dense-vs-dense-reference at the same settings, 16 tokens, -ngl 99: measured 1.25 (default)
# and 1.43 (-fa off, f32 KV) with the LittleBit op on CUDA (1.46 / 1.56 when it ran on the CPU); limit 2.0.
GPU_OVER_DENSE_FLOOR_MAX = 2.0
# Prefill (44 tokens) at -ngl 99, exact config: measured 1.58 x the dense floor (limit above, 2.0).  Default config:
# LittleBit on the GPU / the same model on the pure CPU in the same config, measured 1.40 (3.09e-2 / 2.20e-2); limit
# 2.0 (see test_gpu_error_is_within_dense_floor_factor for why the dense factor does not apply there).
GPU_OVER_CPU_DEFAULT_MAX = 2.0
# One token, -ngl 99, -fa off, f32 KV: with the F16 head 4.8e-3 (limit 1e-2); with an F32 head 2.6e-5 with the op on
# CUDA (1.1e-4 when it ran on the CPU), limit 2.5e-4 (about 10x; was 1e-3).
GPU_1TOK_F16_HEAD_MAX_ABS = 1e-2
GPU_1TOK_F32_HEAD_MAX_ABS = 2.5e-4


@pytest.mark.parametrize("cfg", ["default", "exact"])
@pytest.mark.parametrize("prompt", [A.PROMPT_LONG, A.PROMPT_PREFILL], ids=["16tok", "prefill"])
def test_gpu_error_is_within_dense_floor_factor(ggufs, cfg, prompt):
    """-ngl 99: LittleBit's error stays within GPU_OVER_DENSE_FLOOR_MAX x the DENSE model's CUDA floor.

    16 tokens: the chunked decode kernels of the CUDA op.  The PROMPT_PREFILL prompt (44 tokens) is above the chunk limit
    of every Qwen3-0.6B linear (16 or 24 tokens), so every LittleBit linear of the prompt batch runs the prefill path
    (dequantized sign tiles + float32 cuBLAS GEMMs).

    Prefill, default configuration (flash attention + f16 KV): the dense factor does NOT apply, and the bound is
    instead GPU_OVER_CPU_DEFAULT_MAX x the same LittleBit model's error on the pure CPU (`-dev none`) in the same
    configuration.  Diagnosis (measured, 44 tokens, max-abs against the respective fp32 reference): on the pure CPU,
    with no GPU involved, LittleBit 2.20e-2 vs dense 8.31e-3, i.e. 2.65x already.  What carries it, isolated on the
    pure CPU (5 prompts of 22..62 tokens): f16 rounding of keys / values in attention.  Flash attention gives the same
    error with an f16 or an f32 KV cache (identical values), and with flash attention off the f16 KV cache alone gives the
    same level (LittleBit 2.4e-2..1.1e-1, dense 5.5e-3..8.6e-3); with neither, LittleBit is at 1.3e-4..5.2e-4 (dense
    1.7e-5..2.5e-5).  So the init-only LittleBit model amplifies that rounding far more than the dense model.  On the
    GPU: 3.09e-2 vs dense 1.49e-2 (2.08x).  The LittleBit prefill path is not the carrier: with the CUDA op forced onto
    the chunked decode kernels (temporary build) the same run gives 2.52e-2 (default) and 2.91e-2 (exact) against
    3.09e-2 and 1.94e-2 with the prefill path, i.e. the two float32 paths land on the same error level in different
    order.  In the exact configuration the dense factor holds for THIS prompt (1.58).

    THESE GUARDS ARE COARSE, and the dense-factor one is not a sound detector in general (Task 13 fix round 1, measured):
      * spread over correct paths: 5 prompts (this one and 4 more, 22..62 tokens) x {prefill path, forced chunked
        path}: LittleBit / dense-floor ratios of 1.69..21.0 (default) and 1.58..25.1 (exact): the 2.0 factor FAILS on
        correct code for most prompts; it holds only where this test applies it (PROMPT_LONG in both configurations,
        PROMPT_PREFILL exact).  GPU / pure-CPU (default config) is 1.14..2.42 over
        the same 10 correct runs (one above 2.0: the forced chunked path on a 48-token prompt).
      * mutations of the prefill epilogue (residual term times 1.001, 1.01, 1.1, residual dropped) are all caught by both
        guards on PROMPT_PREFILL (smallest effect, x1.001: default 0.172 = 7.8x the pure CPU, exact 0.163 = 13x the
        dense floor).  The op-level tests (test-littlebit-ref, test-backend-ops, NMSE <= 1e-10) are the precise
        detectors (precise to rounding); this model-level test only catches gross integration errors.
      * decision (the user's, Task 13 fix round 2, from these measurements): keep both guards as they are (option "B1"):
        default config LittleBit-GPU <= 2.0x LittleBit-pure-CPU, exact config LittleBit <= 2.0x the dense floor.
        Measured false-failure rate of the default-config guard over the 10 correct runs of the study: 1/10 (the forced
        chunked path on the 48-token prompt, 2.42; the committed prefill path 0/5); the exact-config dense factor holds
        only on this test's prompt (9/10 correct runs of the study exceed it).  Smallest mutation caught on this prompt
        by both: the residual term x1.001.  Reproduce: tools/dev/prefill_guard_study.sh / .py; raw data in
        docs/data/prefill_guard_study_2026-11-22.jsonl and docs/data/prefill_guard_cpu_2026-11-22.jsonl.
    """
    b1, _ = ggufs
    dense = A.ensure_dense_floor_gguf()
    extra = () if cfg == "default" else EXACT_ARGS
    lb, ids = A.engine_logits(b1, prompt, 99, extra)
    dn, ids_d = A.engine_logits(dense, prompt, 99, extra)
    assert ids == ids_d
    if prompt is A.PROMPT_PREFILL:
        # above the chunk limit of every Qwen3-0.6B linear (16 tokens for k / v, 24 for the others,
        # ggml-cuda/littlebit-dispatch.h)
        assert len(ids) > 24, len(ids)
    e_lb = metrics(lb, A.reference_logits(ids))["max_abs"]
    e_dense = metrics(dn, A.dense_reference_logits(ids))["max_abs"]
    print(f"gpu {cfg} {len(ids)} tokens: LittleBit {e_lb:.2e}, dense floor {e_dense:.2e}, ratio {e_lb / e_dense:.2f}")
    if prompt is A.PROMPT_PREFILL and cfg == "default":
        cpu, _ = A.engine_logits(b1, prompt, 0, ("-dev", "none"))
        e_cpu = metrics(cpu, A.reference_logits(ids))["max_abs"]
        print(f"gpu default {len(ids)} tokens: LittleBit on the pure CPU {e_cpu:.2e}, GPU / CPU {e_lb / e_cpu:.2f}")
        assert e_lb <= GPU_OVER_CPU_DEFAULT_MAX * e_cpu
    else:
        assert e_lb <= GPU_OVER_DENSE_FLOOR_MAX * e_dense


def test_gpu_one_token_error_is_the_f16_head(ggufs):
    """1 token at -ngl 99 without flash attention: F16 head 4.8e-3; the same file with an F32 head is exact to 3e-5."""
    b1, _ = ggufs
    f32head = A.ensure_b1_f32_head_gguf()
    lg16, ids = A.engine_logits(b1, A.PROMPT_ONE, 99, EXACT_ARGS)
    lg32, _ = A.engine_logits(f32head, A.PROMPT_ONE, 99, EXACT_ARGS)
    want = A.reference_logits(ids)
    e16, e32 = metrics(lg16, want)["max_abs"], metrics(lg32, want)["max_abs"]
    print(f"1 token gpu exact: F16 head {e16:.2e}, F32 head {e32:.2e}")
    assert e16 <= GPU_1TOK_F16_HEAD_MAX_ABS and e32 <= GPU_1TOK_F32_HEAD_MAX_ABS


def littlebit_placement(gguf, ngl: int, extra: tuple = ()) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """Every LITTLEBIT node the scheduler assigned in a short llama-completion run, parsed from GGML_SCHED_DEBUG=2.

    Returns (node name, backend of the node, [(source name, backend of the source), ...]) per LITTLEBIT node of every
    graph the scheduler printed.  A source name with a `#` (e.g. `CUDA0#blk.0.attn_q.lb_v`) is a per-call copy of a
    tensor that lives elsewhere; a weight source on the node's own backend means the weight is stored there.
    """
    env = A._env()
    env["GGML_SCHED_DEBUG"] = "2"
    r = subprocess.run([str(A.BIN / ("llama-completion" + A.EXE)), "-m", str(gguf), "-p", "Hello", "-n", "1",
                        "-ngl", str(ngl), "-no-cnv", "-v", *extra], cwd=A.ROOT, env=env, text=True, encoding="utf-8",
                       errors="replace", stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=600)
    assert r.returncode == 0, r.stdout[-2000:]
    text = re.sub(r"\d+\.\d+\.\d+\.\d+ [DIWE] ", "", r.stdout)   # drop the log timestamps
    nodes = []
    for chunk in text.split("node #")[1:]:
        m = re.match(r"\s*\d+ \(\s*LITTLEBIT\):\s*(\S+) \([^)]*\) \[\s*(\S+)\s*\]", chunk)
        if not m:
            continue
        body = chunk[m.end():].split("## SPLIT")[0]
        srcs = re.findall(r"(\S+) \(\s*\d+[KMG]\) \[\s*(\S+)\s*\]", body)   # sizes are printed in K, M or G
        nodes.append((m.group(1), m.group(2), srcs))
    return nodes


def test_littlebit_nodes_and_weights_on_cuda_at_ngl99(ggufs):
    """-ngl 99: every LITTLEBIT node runs on CUDA0 and every one of its sources (activation and weights) lives there,
    with no per-call weight copy (Task 11's placement by capability).  Teeth: test_placement_check_detects_littlebit_off_cuda."""
    b1, _ = ggufs

    def off_cuda(nodes):
        return [(n, d, s) for n, d, s in nodes if d != "CUDA0" or any(sd != "CUDA0" or "#" in sn for sn, sd in s)]

    gpu = littlebit_placement(b1, 99)
    assert gpu and len(gpu) % 196 == 0, len(gpu)     # 28 layers x 7 linears per printed graph
    assert not off_cuda(gpu), off_cuda(gpu)[:3]
    assert all(len(s) >= 9 for _, _, s in gpu)       # x + 8 weight tensors (packed residual scales) per node


def _engine_devices() -> str:
    """The device list of the engine build (`llama-completion --list-devices`)."""
    r = subprocess.run([str(A.BIN / ("llama-completion" + A.EXE)), "--list-devices"], cwd=A.ROOT, env=A._env(),
                       text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=120)
    return r.stdout


def test_placement_check_detects_littlebit_off_cuda(ggufs):
    """Teeth of the placement test: with a Vulkan device (no LittleBit kernel there; any Vulkan device will do, on the
    test machine Vulkan0 is the integrated Intel GPU) the LittleBit nodes and weights stay on the CPU, and the same
    check as above flags every node.  Skipped on machines without a Vulkan0 device."""
    if not re.search(r"^\s*Vulkan0:", _engine_devices(), re.M):
        pytest.skip("no Vulkan0 device in this engine build / machine: the negative control needs one")
    b1, _ = ggufs
    vk = littlebit_placement(b1, 99, ("-dev", "Vulkan0"))
    off_cuda = [(n, d, s) for n, d, s in vk if d != "CUDA0" or any(sd != "CUDA0" or "#" in sn for sn, sd in s)]
    assert vk and len(off_cuda) == len(vk) and any(d == "CPU" for _, d, _ in vk), vk[:3]


@pytest.mark.parametrize("ngl", [0, 99], ids=["cpu", "gpu"])
def test_generation_exits_zero(ggufs, ngl):
    """llama-cli and llama-completion generate 8 tokens from the B1 GGUF at both offload settings."""
    b1, _ = ggufs
    res = A.run_llama_cli(b1, "Hello", n_predict=8, ngl=ngl)
    assert res.returncode == 0, res.stdout[-2000:]
    assert A.count_generated_tokens(b1, "Hello", n_predict=8, ngl=ngl) == 8


def _tokens_per_second(gguf, ngl: int = 0, n: int = 32) -> float:
    """Generation speed (tokens/s) of `n` tokens of "Hello", read from llama-completion's timing line."""
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
