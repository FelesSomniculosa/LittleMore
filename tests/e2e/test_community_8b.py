"""Real weights, end to end: the community LittleBit checkpoint of Qwen3-8B on a 6 GB GPU.

What this checks
----------------
The checkpoint `LittleBitLLM/littlebit-qwen3-8b-0.55bpw` (Qwen3-8B at 0.55 bits per weight, LittleBit-2 with a
residual path, half an epoch of quantization-aware distillation; licence CC BY-NC 4.0) converted to a B1 GGUF with
`tools/convert/lb_to_gguf.py` (default F16 embedding and head):

  * `test_loads_within_6gb_and_generates_32_tokens`: the Python SDK loads it with auto-fit (`n_gpu_layers="auto"`),
    the fit puts every layer and the output head on the GPU, the GPU memory in use stays below 6 GiB (measured with
    nvidia-smi while the model is loaded), and greedy generation yields 32 tokens of text.
  * `test_engine_matches_reference_on_real_weights`: engine (llama-perplexity, CUDA, `-ngl 99`) and the PyTorch
    reference (tools/dev/reference.py, float32 on the CPU) give the same perplexity, within the same 1 % bound as
    tests/e2e/test_perplexity_parity.py, on the first chunks of the WikiText-2 test split (512 tokens per chunk,
    second half scored, exactly as that test).  The reference runs memory-lean: signs kept bit-packed
    (`build_model(..., packed_signs=True)`) and the bf16 embedding / head applied in float32 (`lean_float32`), about
    4 GiB of RAM instead of about 9; the numbers are those of the plain float32 reference (tests/dev/test_reference.py).
  * `test_engine_matches_reference_tightly_with_f32_head_compute`: the same within 0.1 % with the CUDA backend's
    cuBLAS matmuls forced to 32-bit compute (the only source of the CUDA gap on this model, see `CUBLAS_F32`), and
    `test_tight_bound_detects_a_missing_residual_path` shows that this tight bound catches a fault the 1 % bound
    misses on a trained model (its docstring has the measured sensitivities).
  * `test_token_streams_identical`: the engine's tokenizer and the checkpoint's Hugging Face tokenizer give the same
    ids for the scored text, so a tokenizer difference cannot pass for a numeric one.

Heavy and opt-in
----------------
It needs about 3 GB of downloads, a converted 2.84 GiB GGUF, about 4 GiB of free RAM, a CUDA GPU with 6 GB, and a few
minutes, so it runs only with `LITTLEMORE_RUN_8B=1` (and it carries the `smoke` marker like the other end-to-end
tests).  It never downloads or converts anything itself: missing files are a skip whose message gives the command.

    .venv\\Scripts\\hf.exe download LittleBitLLM/littlebit-qwen3-8b-0.55bpw --revision 6b2c47aa9707e624e2e737d16d64a66be8090ae9
        --local-dir .cache/models/littlebit-qwen3-8b-0.55bpw --exclude "logs/*" "runs/*"
    .venv\\Scripts\\python.exe -m tools.convert.lb_to_gguf .cache/models/littlebit-qwen3-8b-0.55bpw
        -o .cache/gguf/littlebit-qwen3-8b-0.55bpw-b1.gguf
    $env:LITTLEMORE_RUN_8B = "1"; .venv\\Scripts\\python.exe -m pytest tests/e2e/test_community_8b.py -v -s

Env: LITTLEMORE_8B_PPL_CHUNKS = number of 512-token chunks for the parity test (default DEFAULT_CHUNKS).
Quality is not tested: the checkpoint is an early one (its card reports WikiText-2 perplexity 24.5 against 9.7 for
the original model), and the measured numbers are in docs/quickstart.md.
"""
from __future__ import annotations

import ast
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import _assets as A

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_perplexity_parity as P  # noqa: E402  (same directory; reuses its dataset pinning and engine helpers)

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(os.environ.get("LITTLEMORE_RUN_8B") != "1",
                       reason="heavy 8B real-weights test: set LITTLEMORE_RUN_8B=1 to run it (see the module docstring)"),
]

CKPT = A.CACHE / "models" / "littlebit-qwen3-8b-0.55bpw"
GGUF = A.CACHE / "gguf" / "littlebit-qwen3-8b-0.55bpw-b1.gguf"
DEFAULT_CHUNKS = 4          # 4 x 512 tokens, 1020 scored; the CPU reference takes about 15 s per chunk
GPU_LIMIT_MIB = 6 * 1024    # "fits a 6 GB GPU"
N_GEN = 32
PROMPT = "The capital of France is"


def n_chunks() -> int:
    """Chunk count for the parity test (LITTLEMORE_8B_PPL_CHUNKS, default DEFAULT_CHUNKS)."""
    return int(os.environ.get("LITTLEMORE_8B_PPL_CHUNKS", DEFAULT_CHUNKS))


def _gpu_used_mib() -> int | None:
    """GPU memory in use on the first GPU (nvidia-smi), or None without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


# --- fixtures --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ckpt() -> Path:
    """The downloaded checkpoint directory, or skip with the download command."""
    if not (CKPT / "model.safetensors").exists() or not (CKPT / "tokenizer.json").exists():
        pytest.skip(f"checkpoint not cached at {CKPT}; download command in this file's docstring (about 3 GB)")
    return CKPT


@pytest.fixture(scope="module")
def gguf(ckpt) -> Path:
    """The converted GGUF, or skip with the conversion command (this test never converts by itself)."""
    if not GGUF.exists():
        pytest.skip(f"GGUF not found at {GGUF}; convert with: python -m tools.convert.lb_to_gguf {CKPT} -o {GGUF}")
    return GGUF


@pytest.fixture(scope="module")
def engine_assets(gguf):
    """(GGUF, llama-perplexity, llama-tokenize), in the tuple layout test_perplexity_parity's helpers take."""
    for tool in ("llama-perplexity", "llama-tokenize"):
        if not (A.BIN / (tool + A.EXE)).exists():
            pytest.skip(f"engine binary {tool} not built in {A.BIN} (see docs/development.md)")
    return gguf, A.BIN / ("llama-perplexity" + A.EXE), A.BIN / ("llama-tokenize" + A.EXE)


@pytest.fixture(scope="module")
def hf_ids(ckpt):
    """Ids of the pinned WikiText-2 test text from the checkpoint's own Hugging Face tokenizer (no special tokens)."""
    from transformers import AutoTokenizer
    if not P.RAW.exists():
        pytest.skip(f"WikiText-2 test split not cached at {P.RAW} (see tests/e2e/test_perplexity_parity.py)")
    assert P.RAW.stat().st_size == P.RAW_BYTES and P._sha256(P.RAW) == P.RAW_SHA256, "not the pinned WikiText-2 text"
    tk = AutoTokenizer.from_pretrained(str(ckpt))
    return tk(P.RAW.read_bytes().decode("utf-8"), add_special_tokens=False)["input_ids"]


# --- tests -----------------------------------------------------------------------------------------------------
def test_loads_within_6gb_and_generates_32_tokens(gguf):
    """SDK + auto-fit: all 36 layers and the head on the GPU, < 6 GiB GPU memory in use, 32 greedy tokens of text."""
    littlemore = pytest.importorskip("littlemore")
    if not any(b.get("name", "").startswith("CUDA") for b in littlemore.backends_info()):
        pytest.skip("no CUDA device visible to the LittleMore library")
    peak, stop = [0], threading.Event()

    def sample() -> None:
        """Poll nvidia-smi every 0.2 s and keep the largest value seen while the model is loaded."""
        while not stop.is_set():
            used = _gpu_used_mib()
            if used is not None:
                peak[0] = max(peak[0], used)
            stop.wait(0.2)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        with littlemore.Model(gguf) as model:              # n_gpu_layers="auto", n_ctx = library default (4096)
            info = model.info()["model"]
            pieces = list(model.generate(PROMPT, stream=True, temperature=0, max_tokens=N_GEN))
            text = "".join(pieces)
            n_tokens = len(model.tokenize(PROMPT + text)) - len(model.tokenize(PROMPT))
    finally:
        stop.set()
        sampler.join()
    print(f"\n[8b] n_gpu_layers={info['n_gpu_layers']}/{info['n_layer'] + 1} devices={info['gpu_devices']} "
          f"gpu peak={peak[0]} MiB pieces={len(pieces)} tokens={n_tokens}\n[8b] {PROMPT!r} -> {text!r}")
    assert info["n_layer"] == 36 and info["gpu_devices"].startswith("CUDA")
    assert info["n_gpu_layers"] == info["n_layer"] + 1, "auto-fit did not put the whole model on the GPU"
    assert 0 < peak[0] < GPU_LIMIT_MIB, f"GPU memory in use {peak[0]} MiB (limit {GPU_LIMIT_MIB})"
    # One piece per token unless the library had to merge tokens to complete a UTF-8 character; retokenizing the
    # text gives the token count back (equal here: the output is plain text).
    assert len(pieces) == N_GEN and n_tokens == N_GEN, (len(pieces), n_tokens)
    words = text.split()
    assert len(words) >= 10 and sum(w.strip(".,;:!?'\"()").isalpha() for w in words) >= 0.8 * len(words), text


def test_token_streams_identical(engine_assets, hf_ids):
    """The engine's tokenizer gives the same ids as the checkpoint's Hugging Face tokenizer for the whole text."""
    gguf, _, tok_exe = engine_assets
    proc = A._run([tok_exe, "-m", gguf, "-f", P.RAW, "--ids", "--no-parse-special", "--log-disable"], timeout=900)
    assert proc.returncode == 0, proc.stdout[-500:]
    engine_ids = ast.literal_eval(proc.stdout.strip().splitlines()[-1])
    first_diff = next((i for i, (a, b) in enumerate(zip(engine_ids, hf_ids)) if a != b), None)
    assert first_diff is None and len(engine_ids) == len(hf_ids), \
        f"token streams differ (lengths {len(engine_ids)} vs {len(hf_ids)}, first difference at {first_diff})"


def lean_reference_model(ckpt: Path):
    """The memory-lean float32 reference of the checkpoint (packed signs, bf16-stored embedding and head)."""
    A._add_path(A.ROOT / "tools" / "dev")
    import reference as ref
    return ref.lean_float32(ref.build_model(str(ckpt), ckpt, packed_signs=True)).eval()


def lean_reference_nlls(model, chunk_ids: np.ndarray) -> list[float]:
    """Per-chunk mean NLL of `model` (see `lean_reference_model`), scored like llama-perplexity (second half).

    The head is applied only to the scored positions (FIRST .. N_CTX-2); the result is the same as scoring the full
    logits (P.reference_chunk_nll) with a third of the head's cost.
    """
    import torch
    out = []
    with torch.no_grad():
        for ids in chunk_ids:
            hidden = model.model(torch.from_numpy(ids)[None]).last_hidden_state[0]
            logp = torch.log_softmax(model.lm_head(hidden[P.FIRST:P.N_CTX - 1]).double(), dim=-1)
            target = torch.from_numpy(ids[P.FIRST + 1:P.N_CTX])
            out.append(float(-logp.gather(1, target[:, None]).mean()))
    return out


@pytest.fixture(scope="module")
def chunk_ids(hf_ids) -> np.ndarray:
    """The first n_chunks() consecutive 512-token chunks (what llama-perplexity scores with `--chunks N`)."""
    n = n_chunks()
    return np.asarray(hf_ids[: n * P.N_CTX], dtype=np.int64).reshape(n, P.N_CTX)


@pytest.fixture(scope="module")
def reference(ckpt, chunk_ids):
    """(lean reference model, its per-chunk NLL, seconds); built once (about 12 s, 4 GiB of RAM) for this module."""
    t0 = time.time()
    model = lean_reference_model(ckpt)
    nll = lean_reference_nlls(model, chunk_ids)
    return model, nll, time.time() - t0


# The CUDA backend multiplies an F16 matrix such as the output head through cuBLAS with 16-bit accumulation for prompt
# batches (CUBLAS_COMPUTE_16F; ggml-cuda.cu).  On this trained model that is the whole engine-vs-reference gap on CUDA
# (measured, 32 chunks: +9.5e-4 relative with it, -1.2e-5 with 32-bit compute; on the CPU with exact attention the gap
# is 1.4e-6).  llama.cpp's own switch GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 forces 32-bit compute, so with it a much tighter
# bound applies: TIGHT_TOL, about 80x the gap measured over 32 chunks (1.2e-5; 6e-6 over the first 4).
CUBLAS_F32 = {"GGML_CUDA_CUBLAS_COMPUTE_TYPE": "f32"}
TIGHT_TOL = 1e-3


def _engine_nll(engine_assets, n: int, env: dict[str, str], monkeypatch) -> tuple[list[float], float, float]:
    """Per-chunk NLL, perplexity and seconds of llama-perplexity on CUDA (`-ngl 99`) with extra environment `env`."""
    with monkeypatch.context() as m:
        for k, v in env.items():
            m.setenv(k, v)                     # A._run copies os.environ for the engine process
        nll, eng_ppl, secs, _ = P.engine_chunk_nlls(engine_assets, P.RAW, n, P.CUDA)
    assert abs(P.ppl(nll) - eng_ppl) / eng_ppl < 1e-4      # the differenced running values reproduce the total
    return nll, eng_ppl, secs


@pytest.fixture(scope="module")
def engine_f32_head(engine_assets, chunk_ids):
    """Engine result on CUDA with 32-bit cuBLAS compute (shared by the tight check and its mutation check)."""
    with pytest.MonkeyPatch.context() as mp:
        return _engine_nll(engine_assets, len(chunk_ids), CUBLAS_F32, mp)


def _report(tag: str, ref_nll, eng_nll, eng_ppl, ref_secs, eng_secs) -> float:
    """Print the comparison and return the relative perplexity difference."""
    ref_ppl = P.ppl(ref_nll)
    rel = abs(eng_ppl - ref_ppl) / ref_ppl
    gaps = [e - r for e, r in zip(eng_nll, ref_nll)]
    print(f"\n[8b ppl parity, {tag}] chunks={len(ref_nll)} reference={ref_ppl:.5f} ({ref_secs:.0f}s) "
          f"engine={eng_ppl:.5f} ({eng_secs:.0f}s) rel={rel:.3e} per-chunk NLL gaps={[f'{g:+.2e}' for g in gaps]}")
    assert all(math.isfinite(x) for x in list(ref_nll) + list(eng_nll))
    return rel


def test_engine_matches_reference_on_real_weights(engine_assets, chunk_ids, reference, monkeypatch):
    """Engine (CUDA, default settings) vs reference (CPU, float32) perplexity: within 1 % (as for Qwen3-0.6B)."""
    _, ref_nll, ref_secs = reference
    eng_nll, eng_ppl, eng_secs = _engine_nll(engine_assets, len(chunk_ids), {}, monkeypatch)
    rel = _report("default", ref_nll, eng_nll, eng_ppl, ref_secs, eng_secs)
    assert rel <= P.REL_TOL, f"perplexity differs by {rel:.3%} (> {P.REL_TOL:.0%})"


def test_engine_matches_reference_tightly_with_f32_head_compute(reference, engine_f32_head):
    """With 32-bit cuBLAS compute the engine matches the reference within 0.1 % (see CUBLAS_F32)."""
    _, ref_nll, ref_secs = reference
    eng_nll, eng_ppl, eng_secs = engine_f32_head
    rel = _report("GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32", ref_nll, eng_nll, eng_ppl, ref_secs, eng_secs)
    assert rel <= TIGHT_TOL, f"perplexity differs by {rel:.3%} (> {TIGHT_TOL:.1%})"


def test_tight_bound_detects_a_missing_residual_path(reference, chunk_ids, engine_f32_head):
    """The tight check has teeth on real weights: the reference without the residual path of ONE linear fails it.

    Measured sensitivity on this trained model (4 chunks, relative perplexity change of the reference): residual path
    of layer 18's down_proj left out +8.8e-3 (caught by TIGHT_TOL, NOT by the 1 % bound), a one-token shift of the
    input +1.6e-2, but a 2 % error on one output scale only +2.5e-4 (layer 0 down_proj) or -2.8e-5 (layer 0 q_proj):
    single scale errors of that size are left to the converter's bit-exact per-layer check (tools/convert/validate.py).
    """
    model, ref_nll, _ = reference
    eng_nll, eng_ppl, _ = engine_f32_head
    lin = model.model.layers[18].mlp.down_proj
    assert lin.n_paths == 2
    lin.n_paths = 1                                     # drop the residual path of this one linear
    try:
        mutated = lean_reference_nlls(model, chunk_ids)
    finally:
        lin.n_paths = 2
    rel = abs(eng_ppl - P.ppl(mutated)) / P.ppl(mutated)
    print(f"\n[8b mutation] residual of layer 18 down_proj dropped: relative difference to the engine {rel:.3e}")
    assert rel > TIGHT_TOL
