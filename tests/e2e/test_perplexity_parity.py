"""End-to-end perplexity parity: engine (llama-perplexity) vs the PyTorch reference.

What this proves
----------------
Both sides score WikiText-2 (raw test split) with the SAME tokens, the SAME chunking
and the SAME scoring rule, and the resulting perplexities must agree within 1 %
(relative).  The model is the init-only Qwen3-0.6B LittleBit checkpoint (0.55 bits
per weight, LittleBit-2, see docs/development.md).

  reference side : tools/dev/reference.py, plain PyTorch on the CPU, fp32 compute, the
                   dense weights are never built (the factorized chain is evaluated);
  engine side    : engine/build/bin/llama-perplexity on the converted B1 GGUF, once on
                   the GPU (CUDA fused LittleBit op, `-ngl 99`) and once on the CPU
                   (AVX2 path, `-ngl 0 -dev none`), with the engine's default settings
                   (flash attention auto, f16 KV cache, F16 head).

Why `-dev none` for the CPU run: with only `-ngl 0` the CUDA build still offloads the
weight matmuls of every batch of >= 32 tokens to the GPU (ggml's "op offload"), so a
512-token chunk would NOT run on the CPU kernels.  Measured on this model (rms logit
difference to the reference at 64 tokens, CPU exact settings): 2.9e-2 with op offload,
8.6e-5 with `--no-op-offload`/`-dev none`.  `-dev none` runs everything on the CPU.

The model is untrained (no QAT), so its perplexity is enormous (millions).  That is
fine: the point is that two independent implementations of the same function give the
same number, and perplexity is a sensitive summary of every logit of every scored
token.  It does NOT measure quality, and it covers only a subset of the test split.

Scoring rule (mirrors engine/tools/perplexity/perplexity.cpp, perplexity())
---------------------------------------------------------------------------
* The text is tokenised once (Qwen3 adds no BOS) and cut into consecutive,
  non-overlapping chunks of `n_ctx` = 512 tokens; only the first N chunks are used.
* Each chunk is evaluated on its own (fresh context, positions 0..511).
* Within a chunk only the second half is scored: the logits at positions
  256..510 predict the tokens at positions 257..511 (255 tokens per chunk), because
  early positions have little context and would add noise.
* perplexity = exp(mean negative log-likelihood over all scored tokens of all chunks).

Tokenizer check
---------------
llama-perplexity tokenises the raw text itself.  `test_token_streams_identical`
compares the whole id stream of the engine's tokenizer (llama-tokenize) with the one
from the Hugging Face tokenizer that the reference side uses, so a mismatch is caught
(and located) before it can masquerade as a numeric difference.

Diagnosis aids
--------------
`llama-perplexity` prints the running perplexity after every chunk; from it the
per-chunk mean NLL is recovered and compared chunk by chunk, so a failure shows which
chunk (and by how much) the two sides disagree on.

Mutation test proves the comparison can fail: a perturbed scale vector in the
reference, and a one-token shift of the reference's token stream, must both exceed the
1 % bound.

Skips (never fails) when the dataset, base model, GGUF or engine binaries are absent.
Run:  .venv\\Scripts\\python.exe -m pytest tests/e2e -m smoke -v -s
Env:  LITTLEMORE_PPL_CHUNKS = number of 512-token chunks (default DEFAULT_CHUNKS).
"""
from __future__ import annotations

import ast
import hashlib
import math
import os
import re
import time

import numpy as np
import pytest

import _assets as A

pytestmark = pytest.mark.smoke

# --- configuration ---------------------------------------------------------------------------------------------
N_CTX = 512            # tokens per chunk (the context length used for the perplexity rows in bench/results/phase1.md)
FIRST = N_CTX // 2     # llama-perplexity scores only positions >= n_ctx/2 of each chunk
SCORED_PER_CHUNK = N_CTX - FIRST - 1   # 255 tokens per chunk
DEFAULT_CHUNKS = 32    # first 32 chunks = 16384 tokens (8160 scored); the full split is 583 chunks
REL_TOL = 0.01         # the bound under test; never loosen it, find the cause of a gap instead

# Pinned dataset: WikiText-2 raw test split from the Hugging Face dataset repo `Salesforce/wikitext`
# (config wikitext-2-raw-v1, revision b08601e04326c79dfdd32d625aee71d232d685c3), file
# wikitext-2-raw-v1/test-00000-of-00001.parquet.  The raw text is the "text" column joined with "" (4358 rows).
# Download (about 0.7 MB; into .cache, which is git-ignored):
#   .venv\Scripts\hf.exe download Salesforce/wikitext wikitext-2-raw-v1/test-00000-of-00001.parquet
#       --repo-type dataset --revision b08601e04326c79dfdd32d625aee71d232d685c3 --local-dir .cache/data/wikitext
DATA_DIR = A.CACHE / "data"
PARQUET = DATA_DIR / "wikitext" / "wikitext-2-raw-v1" / "test-00000-of-00001.parquet"
RAW = DATA_DIR / "wiki.test.raw"
PARQUET_SHA256 = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"
RAW_SHA256 = "bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a"
RAW_BYTES = 1287656
TOTAL_TOKENS = 298938  # Qwen3 tokenisation of RAW (no BOS)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def n_chunks() -> int:
    return int(os.environ.get("LITTLEMORE_PPL_CHUNKS", DEFAULT_CHUNKS))


# --- fixtures --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def wiki_raw():
    """Path of the pinned WikiText-2 raw test text; rebuilt from the pinned parquet if only that is cached."""
    if not RAW.exists() and PARQUET.exists():
        import pyarrow.parquet as pq
        assert _sha256(PARQUET) == PARQUET_SHA256, "cached parquet differs from the pinned WikiText-2 test split"
        rows = pq.read_table(PARQUET).column("text").to_pylist()
        RAW.write_bytes("".join(rows).encode("utf-8"))
    if not RAW.exists():
        pytest.skip(f"WikiText-2 test split not cached at {RAW} (download command: see the constants in this file)")
    assert RAW.stat().st_size == RAW_BYTES and _sha256(RAW) == RAW_SHA256, \
        f"{RAW} is not the pinned WikiText-2 raw test split (sha256 {RAW_SHA256})"
    return RAW


@pytest.fixture(scope="module")
def assets(wiki_raw):
    """(B1 GGUF, llama-perplexity path, llama-tokenize path); skips when anything is missing."""
    reason = A.missing_assets()
    if reason:
        pytest.skip(reason)
    for tool in ("llama-perplexity", "llama-tokenize"):
        if not (A.BIN / (tool + A.EXE)).exists():
            pytest.skip(f"engine binary {tool} not built in {A.BIN} (see docs/engine-changes.md)")
    return A.ensure_b1_gguf(), A.BIN / ("llama-perplexity" + A.EXE), A.BIN / ("llama-tokenize" + A.EXE)


@pytest.fixture(scope="module")
def hf_ids(wiki_raw):
    """Token ids of the whole text from the Hugging Face tokenizer (no special tokens: Qwen3 has no BOS)."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(str(A.MODEL_DIR))
    text = wiki_raw.read_bytes().decode("utf-8")
    return tk(text, add_special_tokens=False)["input_ids"]


@pytest.fixture(scope="module")
def chunks(hf_ids):
    """The first N consecutive 512-token chunks (what llama-perplexity scores with `--chunks N`)."""
    n = n_chunks()
    assert 1 <= n <= len(hf_ids) // N_CTX, "requested chunk count outside the text"
    return np.asarray(hf_ids[: n * N_CTX], dtype=np.int64).reshape(n, N_CTX)


# --- reference side --------------------------------------------------------------------------------------------
def reference_chunk_nll(model, ids: np.ndarray) -> float:
    """Mean negative log-likelihood (nats, float64) of one 512-token chunk, scored like llama-perplexity."""
    import torch
    with torch.no_grad():
        logits = model(torch.from_numpy(ids)[None]).logits[0].double()
    # logits at position j predict token j+1; score j = FIRST .. N_CTX-2 (255 targets), as perplexity.cpp does.
    logp = torch.log_softmax(logits[FIRST:N_CTX - 1], dim=-1)
    target = torch.from_numpy(ids[FIRST + 1:N_CTX])
    return float(-logp.gather(1, target[:, None]).mean())


def reference_nlls(model, chunk_ids: np.ndarray) -> list[float]:
    return [reference_chunk_nll(model, c) for c in chunk_ids]


@pytest.fixture(scope="module")
def ref_run(chunks, assets):
    """Per-chunk mean NLL of the reference (fp32, CPU) and its wall time; the slowest step, so done once."""
    model = A.reference_model()
    t0 = time.time()
    nlls = reference_nlls(model, chunks)
    return nlls, time.time() - t0


# --- engine side -----------------------------------------------------------------------------------------------
def engine_chunk_nlls(assets, wiki_raw, n: int, device_args: tuple) -> tuple[list[float], float, float]:
    """Run llama-perplexity; returns (per-chunk mean NLL, final perplexity, seconds).

    The tool prints the running perplexity after each chunk, `[k]ppl,`; the cumulative NLL after k chunks is
    k * ln(ppl_k) (in units of one chunk's mean), so the per-chunk means follow by differencing.  The printed values
    carry 4 decimals, ample here: the mean NLL is ln(ppl) ~ 16, so the resolution is ~1e-8 relative.
    """
    gguf, ppl_exe, _ = assets
    t0 = time.time()
    proc = A._run([ppl_exe, "-m", gguf, "-f", wiki_raw, "-c", N_CTX, "--chunks", n, *device_args], timeout=3600)
    dt = time.time() - t0
    assert proc.returncode == 0, proc.stdout[-2000:]
    running = [float(m) for m in re.findall(r"\[\d+\]([0-9.eE+-]+),", proc.stdout)]
    assert len(running) == n, f"expected {n} running perplexities, got {len(running)}: {proc.stdout[-800:]}"
    cum = [math.log(p) * (k + 1) for k, p in enumerate(running)]       # cumulative mean-NLL * chunk index
    per_chunk = [cum[0]] + [cum[k] - cum[k - 1] for k in range(1, n)]
    final = float(re.search(r"Final estimate: PPL = ([0-9.eE+-]+)", proc.stdout).group(1))
    return per_chunk, final, dt


def ppl(nlls) -> float:
    return math.exp(sum(nlls) / len(nlls))


# --- tests -----------------------------------------------------------------------------------------------------
def test_token_streams_identical(assets, wiki_raw, hf_ids):
    """The engine's tokenizer and the Hugging Face tokenizer give the identical id stream for the whole text.

    `--no-parse-special` matches llama-perplexity (it tokenises with special-token parsing off); Qwen3 adds no BOS.
    """
    gguf, _, tok_exe = assets
    proc = A._run([tok_exe, "-m", gguf, "-f", wiki_raw, "--ids", "--no-parse-special", "--log-disable"])
    assert proc.returncode == 0, proc.stdout[-500:]
    engine_ids = ast.literal_eval(proc.stdout.strip().splitlines()[-1])
    assert len(hf_ids) == TOTAL_TOKENS
    first_diff = next((i for i, (a, b) in enumerate(zip(engine_ids, hf_ids)) if a != b), None)
    assert first_diff is None and len(engine_ids) == len(hf_ids), \
        f"token streams differ (lengths {len(engine_ids)} vs {len(hf_ids)}, first difference at {first_diff})"


CUDA = ("-ngl", 99)
CPU = ("-ngl", 0, "-dev", "none")   # see the module docstring: `-ngl 0` alone still offloads prefill matmuls to CUDA


# CPU with every f16 rounding of the attention path removed (no flash attention, f32 KV cache).  The engine's default
# attention path rounds Q/K/V to f16, which the reference (fp32) does not; that rounding is the ~0.3 % default-settings
# gap on the CPU (measured over 8 chunks: default 3.4e-3, exact 2.7e-5).  In this configuration the two sides compute
# the same function, so a much tighter bound applies (1e-3, >10x the measured gap).
CPU_EXACT = CPU + ("-fa", "off", "-ctk", "f32", "-ctv", "f32")
EXACT_TOL = 1e-3


@pytest.mark.parametrize("device,tol", [(CUDA, REL_TOL), (CPU, REL_TOL), (CPU_EXACT, EXACT_TOL)],
                         ids=["cuda", "cpu_avx2", "cpu_avx2_exact"])
def test_perplexity_parity(assets, wiki_raw, chunks, ref_run, device, tol):
    """Engine perplexity (CUDA / CPU / CPU with exact attention) == reference perplexity on the same chunks.

    The bound is 1 % for the engine's default settings; the exact-attention CPU run must agree to 0.1 %."""
    ref_nll, ref_secs = ref_run
    eng_nll, eng_ppl, eng_secs = engine_chunk_nlls(assets, wiki_raw, len(chunks), device)
    ref_ppl, eng_ppl_recomputed = ppl(ref_nll), ppl(eng_nll)
    rel = abs(eng_ppl - ref_ppl) / ref_ppl
    per_chunk = [abs(math.exp(e - r) - 1) for e, r in zip(eng_nll, ref_nll)]   # per-chunk perplexity ratio - 1
    print(f"\n[ppl parity] device={" ".join(map(str, device))} chunks={len(chunks)} scored={len(chunks) * SCORED_PER_CHUNK} "
          f"reference={ref_ppl:.6g} ({ref_secs:.0f}s) engine={eng_ppl:.6g} ({eng_secs:.0f}s) rel={rel:.3e} "
          f"worst-chunk rel={max(per_chunk):.3e} (chunk {int(np.argmax(per_chunk))})")
    # The differenced running values must reproduce the tool's own final number (guards the parsing above).
    assert abs(eng_ppl_recomputed - eng_ppl) / eng_ppl < 1e-4
    assert rel <= tol, f"perplexity differs by {rel:.3%} (> {tol:.1%}); per-chunk rel: " \
                           f"{[f'{x:.1e}' for x in per_chunk]}"


def _mutated_rel(model, chunk_ids, eng_nll, n_mut: int) -> float:
    """Relative perplexity difference of `model` (already mutated) against the engine on the first n_mut chunks."""
    mut = reference_nlls(model, chunk_ids[:n_mut])
    rel = abs(ppl(eng_nll[:n_mut]) - ppl(mut)) / ppl(mut)
    print(f"\n[ppl mutation] relative perplexity difference to the engine: {rel:.3e}")
    return rel


def test_mutations_fail_the_bound(assets, wiki_raw, chunks, ref_run):
    """The check is not vacuous: a perturbed scale vector, or a one-token shift of the reference's input, breaks it.

    Both mutations are evaluated on the first 2 chunks only (cost) against the unmodified engine result; the
    un-mutated reference must pass on the same 2 chunks, so the failure is caused by the mutation alone.
    """
    n_mut = min(2, len(chunks))
    eng_nll, _, _ = engine_chunk_nlls(assets, wiki_raw, n_mut, CUDA)
    ref_nll, _ = ref_run
    assert abs(ppl(eng_nll) - ppl(ref_nll[:n_mut])) / ppl(ref_nll[:n_mut]) <= REL_TOL      # control: clean passes

    # (1) one-token shift of the token stream fed to the reference (a tokenizer / chunk-offset bug).
    shifted = np.concatenate([chunks.reshape(-1)[1:], [0]]).reshape(chunks.shape)
    assert _mutated_rel(A.reference_model(), shifted, eng_nll, n_mut) > REL_TOL

    # (2) a 2 % error on one scale vector (the output scale u1 of layer 0's q_proj): a conversion/scale bug.
    model = A.reference_model()
    lin = model.model.layers[0].self_attn.q_proj
    saved = lin.u1_0.clone()
    try:
        lin.u1_0.mul_(1.02)
        assert _mutated_rel(model, chunks, eng_nll, n_mut) > REL_TOL
    finally:
        lin.u1_0.copy_(saved)
