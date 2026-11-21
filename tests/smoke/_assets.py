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

import functools
import hashlib
import os
import re
import shutil
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

def _add_path(p: Path) -> None:
    """Put `p` on sys.path once (the helpers may be called many times)."""
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


PROMPT = "The capital of France is"
PROMPT_LONG = "In 1969, the first humans landed on the Moon and said"
PROMPT_ONE = "Hello"           # single token (9707): position 0, RoPE/attention trivial
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


def _digest(*parts: object) -> str:
    """sha256 over file contents (Paths) and literal values; keys the artifact cache."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, Path):
            files = sorted(p.rglob("*.py")) if p.is_dir() else [p]
            for f in files:
                h.update(f.name.encode())
                h.update(f.read_bytes())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()


def _stamp_ok(target: Path, key: str) -> bool:
    """True when `target` exists and was built with exactly `key` (so a changed tool rebuilds it)."""
    stamp = target.with_name(target.name + ".key")
    return target.exists() and stamp.exists() and stamp.read_text() == key


def _commit(tmp: Path, target: Path, key: str) -> None:
    """Atomically publish `tmp` as `target` (a crash mid-build never leaves a half file)."""
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    os.replace(tmp, target)
    target.with_name(target.name + ".key").write_text(key)


def ensure_gguf() -> Path:
    """Build (if absent or stale) checkpoint -> base GGUF -> LittleBit Q1_0 GGUF; return the last."""
    maker = ROOT / "tools" / "dev" / "make_lb2_init.py"
    ck_key = _digest(maker, MODEL_DIR / "model.safetensors", BPW, True, True, 0)
    if not _stamp_ok(CKPT_DIR, ck_key):
        # Init-only: SVD + joint ITQ + dual-SVID, residual path on (LittleBit-2).
        _add_path(ROOT / "tools" / "dev")
        from make_lb2_init import make_checkpoint
        tmp = CKPT_DIR.with_name(CKPT_DIR.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        make_checkpoint(str(MODEL_DIR), tmp, BPW, use_itq=True, residual=True)
        _commit(tmp, CKPT_DIR, ck_key)
    base_key = _digest(ENGINE / "convert_hf_to_gguf.py", ENGINE / "conversion", MODEL_DIR / "model.safetensors")
    if not _stamp_ok(BASE_GGUF, base_key):
        # F32 keeps embeddings exact so they do not add error to the logit comparison.
        # Needs the `sentencepiece` package: the Qwen converter imports it eagerly.
        BASE_GGUF.parent.mkdir(parents=True, exist_ok=True)
        tmp = BASE_GGUF.with_name(BASE_GGUF.name + ".tmp")
        r = _run([sys.executable, ENGINE / "convert_hf_to_gguf.py", MODEL_DIR,
                  "--outtype", "f32", "--outfile", tmp])
        assert r.returncode == 0, r.stdout[-2000:]
        _commit(tmp, BASE_GGUF, base_key)
    lb_key = _digest(ENGINE / "tools" / "littlebit" / "lb_to_gguf.py", ck_key, base_key)
    if not _stamp_ok(LB_GGUF, lb_key):
        tmp = LB_GGUF.with_name(LB_GGUF.name + ".tmp")
        r = _run([sys.executable, ENGINE / "tools" / "littlebit" / "lb_to_gguf.py",
                  CKPT_DIR, MODEL_DIR, BASE_GGUF, tmp])
        assert r.returncode == 0, r.stdout[-2000:]
        _commit(tmp, LB_GGUF, lb_key)
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


def engine_layer_sums(gguf: Path, prompt: str, ngl: int = 0) -> dict[int, float]:
    """Sum of every element of each layer output `l_out-N` for a single-token prompt.

    Uses llama-debug's tensor dump (`--verbose`, filtered), which prints a sum
    per tensor, so no engine change is needed.  Only the final (real) pass has
    a 1-token shape; the 2-token warmup pass is ignored.
    """
    r = _run([BIN / ("llama-debug" + EXE), "-m", gguf, "-p", prompt, "-ngl", ngl,
              "--verbose", "--tensor-filter", "^l_out-"])
    assert r.returncode == 0, r.stdout[-2000:]
    sums: dict[int, float] = {}
    for block in re.split(r"(?=common_debug_cb_eval:)", r.stdout):
        m = re.match(r"common_debug_cb_eval:\s+l_out-(\d+) = .*\{1024, 1, 1, 1\}\n", block)
        s = re.search(r"sum = (\S+)", block)
        if m and s:
            sums[int(m.group(1))] = float(s.group(1))
    return sums


@functools.lru_cache(maxsize=1)
def reference_model():
    """The reference model (fp32 compute, CPU); built once per session (it is slow)."""
    _add_path(ROOT / "tools" / "dev")
    from reference import build_model
    return build_model(str(MODEL_DIR), CKPT_DIR).float().eval()


def reference_logits(token_ids: list[int], **emulate_kwargs) -> np.ndarray:
    """Last-position logits of the reference, optionally with the Q8_0 emulation/mutations.

    No kwargs -> plain fp32 `reference.py`; `emulate_kwargs` are forwarded to
    `emulate_q8_slow_path.emulate` (quantize, kv_f16, drop_residual, permute_qk).
    """
    _add_path(ROOT / "tools" / "dev")
    import emulate_q8_slow_path as E
    model = reference_model()
    E.emulate(model, **(emulate_kwargs or dict(quantize=False, kv_f16=False)))
    return E.last_logits(model, token_ids)


def layer_sums_reference(token_ids: list[int]) -> list[float]:
    """Per-layer output sums of the Q8_0-emulated reference (matches `engine_layer_sums`)."""
    import torch
    _add_path(ROOT / "tools" / "dev")
    import emulate_q8_slow_path as E
    model = reference_model()
    E.emulate(model)
    outs: list[float] = []
    hooks = [layer.register_forward_hook(
        lambda mod, i, o: outs.append((o[0] if isinstance(o, tuple) else o).sum().item()))
        for layer in model.model.layers]
    try:
        with torch.no_grad():
            model(torch.tensor([token_ids], dtype=torch.long))
    finally:
        for h in hooks:
            h.remove()
    return outs
