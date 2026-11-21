"""Asset and subprocess helpers for the M0 smoke test.

Everything here is cached under `<repo>/.cache/` (git-ignored; never commit
weights or GGUFs).  Each `ensure_*` function is a no-op when its output exists,
so a second run only pays for the llama.cpp invocations.

Layout:
    .cache/models/Qwen3-0.6B/                      base HF model (downloaded by hand/Task 5 setup)
    .cache/ckpt/qwen3-0.6b-lb2-0.55/               init-only LittleBit-2 checkpoint (tools/dev/make_lb2_init.py)
    .cache/gguf/qwen3-0.6b-base-f32.gguf           plain F32 GGUF; lb_to_gguf copies embeddings/metadata from it
    .cache/gguf/qwen3-0.6b-lb2-0.55-q1_0.gguf      the LittleBit GGUF under test
    .cache/logits/                                 llama-debug --save-logits output
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache"
ENGINE = ROOT / "engine"
BIN = ENGINE / "build" / "bin"
EXE = ".exe" if os.name == "nt" else ""
# ggml-cuda.dll links CUDA 12.9 runtime DLLs; CUDA 13 is also installed here, so
# the 12.9 bin dir must be found first on PATH or the CUDA backend fails to load.
CUDA_BIN = Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin")

MODEL_DIR = CACHE / "models" / "Qwen3-0.6B"
CKPT_DIR = CACHE / "ckpt" / "qwen3-0.6b-lb2-0.55"
BASE_GGUF = CACHE / "gguf" / "qwen3-0.6b-base-f32.gguf"
LB_GGUF = CACHE / "gguf" / "qwen3-0.6b-lb2-0.55-q1_0.gguf"
LOGITS_DIR = CACHE / "logits"
BPW = 0.55

PROMPT = "The capital of France is"
# Token ids of PROMPT for Qwen3 (no BOS); asserted against the engine's tokenizer.
PROMPT_IDS = [785, 6722, 315, 9625, 374]


def missing_assets() -> str | None:
    """Return a human-readable skip reason, or None when everything needed exists."""
    if not (MODEL_DIR / "model.safetensors").exists():
        return f"base model not cached at {MODEL_DIR} (download Qwen/Qwen3-0.6B there)"
    for tool in ("llama-cli", "llama-completion", "llama-debug"):
        if not (BIN / (tool + EXE)).exists():
            return f"engine binary {tool} not built in {BIN} (see docs/engine-changes.md)"
    if not (ENGINE / "tools" / "littlebit" / "lb_to_gguf.py").exists():
        return "engine/tools/littlebit/lb_to_gguf.py missing"
    return None


def _env() -> dict[str, str]:
    """Environment with the CUDA 12.9 bin dir first on PATH and engine gguf-py importable."""
    env = dict(os.environ)
    if CUDA_BIN.exists():
        env["PATH"] = str(CUDA_BIN) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(ENGINE / "gguf-py")
    return env


def _run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    """Run a command with stdin closed (llama-cli would otherwise wait for chat input)."""
    return subprocess.run([str(c) for c in cmd], cwd=ROOT, env=_env(), text=True,
                          encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)


def ensure_gguf() -> Path:
    """Build (if absent) checkpoint -> base GGUF -> LittleBit Q1_0 GGUF; return the last."""
    if not (CKPT_DIR / "model.safetensors").exists():
        # Init-only: SVD + joint ITQ + dual-SVID, residual path on (LittleBit-2).
        sys.path.insert(0, str(ROOT / "tools" / "dev"))
        from make_lb2_init import make_checkpoint
        make_checkpoint(str(MODEL_DIR), CKPT_DIR, BPW, use_itq=True, residual=True)
    if not BASE_GGUF.exists():
        # F32 keeps embeddings exact so they do not add error to the logit comparison.
        # Needs the `sentencepiece` package: the Qwen converter imports it eagerly.
        BASE_GGUF.parent.mkdir(parents=True, exist_ok=True)
        r = _run([sys.executable, ENGINE / "convert_hf_to_gguf.py", MODEL_DIR,
                  "--outtype", "f32", "--outfile", BASE_GGUF])
        assert r.returncode == 0, r.stdout[-2000:]
    if not LB_GGUF.exists():
        r = _run([sys.executable, ENGINE / "tools" / "littlebit" / "lb_to_gguf.py",
                  CKPT_DIR, MODEL_DIR, BASE_GGUF, LB_GGUF])
        assert r.returncode == 0, r.stdout[-2000:]
    return LB_GGUF


def run_llama_cli(gguf: Path, prompt: str, n_predict: int, ngl: int) -> subprocess.CompletedProcess:
    """`llama-cli -m gguf -p prompt -n N -ngl L` (stdin closed so it exits after one turn)."""
    return _run([BIN / ("llama-cli" + EXE), "-m", gguf, "-p", prompt, "-n", n_predict, "-ngl", ngl])


def count_generated_tokens(gguf: Path, prompt: str, n_predict: int, ngl: int) -> int:
    """Number of tokens generated, from llama-completion's perf line ("... / 8 runs").

    llama-cli's chat mode does not print a token count, so the same model is run
    through llama-completion (plain completion, no chat template).
    """
    r = _run([BIN / ("llama-completion" + EXE), "-m", gguf, "-p", prompt, "-n", n_predict,
              "-ngl", ngl, "-no-cnv"])
    assert r.returncode == 0, r.stdout[-2000:]
    m = re.search(r"eval time =\s+[\d.]+ ms /\s+(\d+) runs", r.stdout.replace("prompt eval", "PE"))
    assert m, r.stdout[-2000:]
    return int(m.group(1))


def engine_logits(gguf: Path, prompt: str, ngl: int) -> tuple[np.ndarray, list[int]]:
    """Last-position logits and token ids from `llama-debug --save-logits` (no engine changes)."""
    out = LOGITS_DIR / f"ngl{ngl}"
    r = _run([BIN / ("llama-debug" + EXE), "-m", gguf, "-p", prompt, "-ngl", ngl,
              "--save-logits", "--logits-output-dir", out])
    assert r.returncode == 0, r.stdout[-2000:]
    stem = "llamacpp-" + gguf.stem
    logits = np.fromfile(out / f"{stem}.bin", dtype=np.float32)
    tokens = np.fromfile(out / f"{stem}-tokens.bin", dtype=np.int32).tolist()
    return logits, tokens


def reference_logits(token_ids: list[int]) -> np.ndarray:
    """Last-position logits from the pure-PyTorch reference, CPU, fp32 compute."""
    import torch
    sys.path.insert(0, str(ROOT / "tools" / "dev"))
    from reference import build_model
    model = build_model(str(MODEL_DIR), CKPT_DIR).float().eval()
    with torch.no_grad():
        out = model(torch.tensor([token_ids], dtype=torch.long))
    return out.logits[0, -1].float().numpy()
