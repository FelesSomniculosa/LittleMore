"""Asset and subprocess helpers for the smoke and engine-run tests.

Everything here is cached under `<repo>/.cache/` (git-ignored; never commit
weights or GGUFs).  Each `ensure_*` function is a no-op when its output exists,
so a second run only pays for the llama.cpp invocations.

Layout:
    .cache/models/Qwen3-0.6B/                      base HF model (download command: docs/development.md)
    .cache/ckpt/qwen3-0.6b-lb2-0.55/               init-only LittleBit-2 checkpoint (tools/dev/make_lb2_init.py)
    .cache/gguf/qwen3-0.6b-lb2-0.55-b1.gguf        the LittleBit GGUF (B1 signs) made by tools/convert/lb_to_gguf.py
    .cache/gguf/qwen3-0.6b-lb2-0.55-q1_0.gguf      slow-path baseline: the same file with the signs re-encoded as Q1_0
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
# Folder with llama-debug and the backend plugins: the engine's own build by default; LITTLEMORE_ENGINE_BIN points at
# another build (CI uses the top-level CMake build, where the engine programs land in build/bin).
BIN = Path(os.environ["LITTLEMORE_ENGINE_BIN"]) if os.environ.get("LITTLEMORE_ENGINE_BIN") else ENGINE / "build" / "bin"
EXE = ".exe" if os.name == "nt" else ""
# ggml-cuda.dll links CUDA 12.9 runtime DLLs; CUDA 13 is also installed here, so
# the 12.9 bin dir must be found first on PATH or the CUDA backend fails to load.
CUDA_BIN = Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin")

MODEL_DIR = CACHE / "models" / "Qwen3-0.6B"
CKPT_DIR = CACHE / "ckpt" / "qwen3-0.6b-lb2-0.55"
B1_GGUF = CACHE / "gguf" / "qwen3-0.6b-lb2-0.55-b1.gguf"
LB_GGUF = CACHE / "gguf" / "qwen3-0.6b-lb2-0.55-q1_0.gguf"   # the Q1_0 slow-path baseline
LOGITS_DIR = CACHE / "logits"
BPW = 0.55

def _add_path(p: Path) -> None:
    """Put `p` on sys.path once (the helpers may be called many times)."""
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


PROMPT = "The capital of France is"
PROMPT_LONG = "In 1969, the first humans landed on the Moon and said"
PROMPT_ONE = "Hello"           # single token (9707): position 0, RoPE/attention trivial
# 44 tokens: one prompt batch above the 8-token decode limit of the CUDA LittleBit op, so every Qwen3-0.6B linear takes
# a prefill path (engine/ggml/src/ggml-cuda/littlebit.cu): on compute capability 8.0+ the tensor-core bit-plane path of
# littlebit-bp.cu for every n > 8; the float32 GEMM prefill path only on older GPUs or with
# GGML_CUDA_LITTLEBIT_F32_PREFILL=1.  The chunked-decode limits (16 tokens for k / v, 24 for the others, all below
# 6 Mi signs, littlebit-dispatch.h) apply to that float32 fallback only; 44 tokens is above them, so the prompt takes
# the prefill path on either variant.
PROMPT_PREFILL = ("In 1969, the first humans landed on the Moon. Neil Armstrong stepped onto the surface and said that "
                  "it was one small step for a man and one giant leap for mankind, while millions watched on television")
# Token ids of PROMPT for Qwen3 (no BOS); asserted against the engine's tokenizer.
PROMPT_IDS = [785, 6722, 315, 9625, 374]


def missing_assets() -> str | None:
    """Return a human-readable skip reason, or None when everything needed exists."""
    if not (MODEL_DIR / "model.safetensors").exists():
        return f"base model not cached at {MODEL_DIR} (download it with: .venv/Scripts/hf.exe download Qwen/Qwen3-0.6B --local-dir .cache/models/Qwen3-0.6B)"
    for tool in ("llama-cli", "llama-completion", "llama-debug"):
        if not (BIN / (tool + EXE)).exists():
            return f"engine binary {tool} not built in {BIN} (see docs/engine-changes.md)"
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


def copy_gguf(src: Path, dst: Path, replace=None, extra=(), extra_kv=(), drop=None, drop_kv=()) -> None:
    """Write `dst` = a copy of GGUF `src` with metadata and tensors unchanged, except:

    `replace(tensor)` may return `(array, raw_dtype)` to swap one tensor's data (None keeps it);
    `extra` is a list of `(name, array, raw_dtype)` tensors appended; `extra_kv` a list of
    `(key, value, value_type[, sub_type])` metadata entries added (replacing a copied entry of the same key); `drop(name)` returning True leaves that tensor
    out; `drop_kv` is a collection of metadata keys left out of the copy.  Used to derive test variants.
    """
    _add_path(ENGINE / "gguf-py")
    import gguf
    reader = gguf.GGUFReader(str(src))
    writer = gguf.GGUFWriter(str(dst), reader.get_field("general.architecture").contents())
    skip = ("GGUF.", "general.architecture", "general.alignment")
    overridden = {kv[0] for kv in extra_kv}  # a metadata key given in `extra_kv` replaces the copied one
    for key, field in reader.fields.items():
        if not key.startswith(skip) and key not in overridden and key not in drop_kv:
            writer.add_key_value(key, field.contents(), field.types[0],
                                 field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
    for kv in extra_kv:
        writer.add_key_value(*kv)
    for t in reader.tensors:
        if drop is not None and drop(t.name):
            continue
        new = replace(t) if replace else None
        data, dtype = new if new else (np.asarray(t.data), t.tensor_type)
        writer.add_tensor(t.name, data, raw_dtype=dtype)
    for name, data, dtype in extra:
        writer.add_tensor(name, data, raw_dtype=dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


def b1_to_q1_0_gguf(src: Path, dst: Path) -> None:
    """Write `dst` = `src` with every B1 sign tensor re-encoded as Q1_0 (the slow-path baseline).

    The engine's slow path (the op chain in build_lb_path) takes Q1_0 factors.  Our converter only
    writes B1, so the baseline file is derived from it: B1 -> +-1.0 floats -> gguf-py's Q1_0
    quantizer.  Every block is then exactly "scale 1.0, same sign bits", so the baseline computes the
    same linear as the B1 file up to the Q8_0 activation noise of the Q1_0 kernels.  F16 tensors (the
    embedding / output head) are widened to F32, which is lossless and keeps the baseline as exact as the
    original slow-path file (whose head came from an F32 base GGUF).  Everything else is copied unchanged.
    """
    _add_path(ENGINE / "gguf-py")
    import gguf
    from gguf.quants import dequantize, quantize
    B1, Q1_0 = gguf.GGMLQuantizationType.B1, gguf.GGMLQuantizationType.Q1_0
    F16, F32 = gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32

    def convert(t):
        """Per-tensor rewrite for copy_gguf: B1 -> Q1_0, F16 -> F32, None = copy the tensor unchanged."""
        if t.tensor_type == B1:
            return quantize(dequantize(np.asarray(t.data), B1), Q1_0), Q1_0
        if t.tensor_type == F16:   # embedding/head: widen (lossless) so the CUDA head matmul is exact, as in the original slow-path file
            return np.asarray(t.data).astype(np.float32), F32
        return None
    copy_gguf(src, dst, convert)


def ensure_b1_gguf() -> Path:
    """Build (if absent or stale) checkpoint -> B1 GGUF with tools/convert/lb_to_gguf.py; return it."""
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
    conv_key = _digest(ROOT / "tools" / "convert", ck_key)
    if not _stamp_ok(B1_GGUF, conv_key):
        _add_path(ROOT)
        from tools.convert import lb_to_gguf
        B1_GGUF.parent.mkdir(parents=True, exist_ok=True)
        tmp = B1_GGUF.with_name(B1_GGUF.name + ".tmp")
        lb_to_gguf.convert(CKPT_DIR, tmp)
        _commit(tmp, B1_GGUF, conv_key)
    return B1_GGUF


DENSE_F32 = CACHE / "gguf" / "qwen3-0.6b-base-f32.gguf"
DENSE_F32_H16 = CACHE / "gguf" / "qwen3-0.6b-base-f32-head-f16.gguf"


def ensure_dense_floor_gguf() -> Path:
    """The DENSE base model as F32 with only the tied embedding / output head in F16 (cached; returns it).

    Same non-LittleBit tensors and head precision as the LittleBit GGUF, but every linear dense: running it
    gives the engine's own noise floor (CUDA kernels, F16 head, flash attention) with no LittleBit in the way.
    """
    key = _digest(ENGINE / "convert_hf_to_gguf.py", ENGINE / "conversion", MODEL_DIR / "model.safetensors", Path(__file__))
    if not _stamp_ok(DENSE_F32_H16, key):
        if not _stamp_ok(DENSE_F32, key):
            DENSE_F32.parent.mkdir(parents=True, exist_ok=True)
            tmp = DENSE_F32.with_name(DENSE_F32.name + ".tmp")
            r = _run([sys.executable, ENGINE / "convert_hf_to_gguf.py", MODEL_DIR, "--outtype", "f32", "--outfile", tmp])
            assert r.returncode == 0, r.stdout[-2000:]
            _commit(tmp, DENSE_F32, key)
        _add_path(ENGINE / "gguf-py")
        import gguf
        tmp = DENSE_F32_H16.with_name(DENSE_F32_H16.name + ".tmp")
        copy_gguf(DENSE_F32, tmp, lambda t: (np.asarray(t.data).astype(np.float16), gguf.GGMLQuantizationType.F16)
                  if t.name == "token_embd.weight" else None)
        _commit(tmp, DENSE_F32_H16, key)
    return DENSE_F32_H16


@functools.lru_cache(maxsize=1)
def dense_reference_model():
    """The dense HF Qwen3-0.6B in fp32 on CPU (transformers); the ground truth for the dense noise-floor runs."""
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype=torch.float32).eval()


def dense_reference_logits(token_ids: list[int]) -> np.ndarray:
    """Last-position logits of the dense fp32 HF model."""
    import torch
    with torch.no_grad():
        return dense_reference_model()(torch.tensor([token_ids])).logits[0, -1].numpy()


def ensure_b1_f32_head_gguf() -> Path:
    """The B1 GGUF with its F16 tensors (embedding / output head) widened to F32: a control that removes the F16 head."""
    b1 = ensure_b1_gguf()
    out = CACHE / "gguf" / "qwen3-0.6b-lb2-0.55-b1-f32head.gguf"
    key = _digest(Path(__file__), _stamp_text(b1))
    if not _stamp_ok(out, key):
        _add_path(ENGINE / "gguf-py")
        import gguf
        F16, F32 = gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32
        tmp = out.with_name(out.name + ".tmp")
        copy_gguf(b1, tmp, lambda t: (np.asarray(t.data).astype(np.float32), F32) if t.tensor_type == F16 else None)
        _commit(tmp, out, key)
    return out


def ensure_gguf() -> Path:
    """Return the Q1_0 slow-path baseline GGUF, derived from the B1 GGUF (see `b1_to_q1_0_gguf`)."""
    b1 = ensure_b1_gguf()
    key = _digest(Path(__file__), _stamp_text(b1))   # the b1 file's own .key stamp, so its 640 MiB are not hashed
    if not _stamp_ok(LB_GGUF, key):
        tmp = LB_GGUF.with_name(LB_GGUF.name + ".tmp")
        b1_to_q1_0_gguf(b1, tmp)
        _commit(tmp, LB_GGUF, key)
    return LB_GGUF


def _stamp_text(target: Path) -> str:
    """The build key of a cached artifact, so derived files rebuild when their source does."""
    return target.with_name(target.name + ".key").read_text()


# NOTE: run_llama_cli, count_generated_tokens and engine_layer_sums run with plain `-ngl N` and rely on their prompts
# (1 token or a few) being under 32 tokens: a CUDA build offloads batches of >= 32 tokens to the GPU even at `-ngl 0`
# (see engine_logits, which adds `-dev none`).  Pass `-dev none` yourself before using them with a longer prompt.
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


def engine_logits(gguf: Path, prompt: str, ngl: int, extra_args: tuple = (),
                  out_dir: Path | None = None) -> tuple[np.ndarray, list[int]]:
    """Last-position logits and token ids from `llama-debug --save-logits` (no engine changes).

    `extra_args` are appended to the command line (e.g. `("-ctk", "f32")`); `out_dir` overrides the
    default `.cache/logits/ngl<N>` (the file name inside it is derived from the GGUF's stem).
    """
    if ngl == 0 and "-dev" not in extra_args and "--device" not in extra_args:
        # A CUDA build with plain `-ngl 0` still offloads the matmuls of prompts of >= 32 tokens to the GPU (ggml's
        # op offload), so "CPU" runs would not be CPU runs.  `-dev none` removes every GPU device (measured on the
        # tiny Llama / Gemma 3 models, 37 tokens: logits differ by up to 7e-4 with and without it; prompts under
        # 32 tokens are bit-identical either way).
        extra_args = ("-dev", "none", *extra_args)
    out = out_dir or LOGITS_DIR / f"ngl{ngl}"
    # llama-debug names its files after the model's `general.name`, not the file; start from an empty
    # directory so the files found afterwards are this run's.
    out.mkdir(parents=True, exist_ok=True)   # llama-debug creates only the last path component
    for old in out.glob("llamacpp-*.bin"):
        old.unlink()
    r = _run([BIN / ("llama-debug" + EXE), "-m", gguf, "-p", prompt, "-ngl", ngl,
              "--save-logits", "--logits-output-dir", out, *extra_args])
    assert r.returncode == 0, r.stdout[-2000:]
    (tok_file,) = out.glob("llamacpp-*-tokens.bin")
    logits = np.fromfile(tok_file.with_name(tok_file.name.removesuffix("-tokens.bin") + ".bin"), dtype=np.float32)
    tokens = np.fromfile(tok_file, dtype=np.int32).tolist()
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
def _reference_model_built():
    """The reference model object, built once per session (it is slow). Use `reference_model()`."""
    _add_path(ROOT / "tools" / "dev")
    from reference import build_model
    return build_model(str(MODEL_DIR), CKPT_DIR).float().eval()


def reference_model():
    """The reference model (fp32 compute, CPU), PLAIN: any Q8_0 emulation or mutation an earlier caller applied with
    `emulate_q8_slow_path.emulate` (which patches the shared object and the RoPE function of transformers in place) is
    undone first (`unemulate`: every LBLinear runs its own `LBLinear.forward` again, as on a freshly built model), and
    `assert_plain` then refuses a model that still has any patched module `forward` or RoPE function, so a mutation
    applied some other way fails loudly instead of being used as the reference. Without the reset a test that runs
    after one using `drop_residual=True` or the default emulation got a mutated "reference" (measured:
    test_perplexity_parity failed by 111 % after test_fused_graph_qwen3_06b's residual control, by 14 % in a full
    `pytest` run; it passed when run alone)."""
    _add_path(ROOT / "tools" / "dev")
    import emulate_q8_slow_path as E
    model = E.unemulate(_reference_model_built())
    E.assert_plain(model)
    return model


def reference_logits(token_ids: list[int], **emulate_kwargs) -> np.ndarray:
    """Last-position logits of the reference, optionally with the Q8_0 emulation/mutations.

    No kwargs -> plain fp32 `reference.py` (its own `LBLinear.forward`); `emulate_kwargs` are forwarded to
    `emulate_q8_slow_path.emulate` (quantize, kv_f16, drop_residual, permute_qk).
    """
    _add_path(ROOT / "tools" / "dev")
    import emulate_q8_slow_path as E
    model = reference_model()
    if emulate_kwargs:
        E.emulate(model, **emulate_kwargs)
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
