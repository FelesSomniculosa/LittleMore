"""C. Prompt and token edge cases, and numerical correctness at every kernel dispatch boundary.

Three groups:

1. Tokenizer: odd strings (empty, special-token text, invalid UTF-8, NUL bytes, emoji / CJK / RTL / combining marks, a
   very long word, repeated tokens, long texts) through lm_tokenize / lm_detokenize of the Qwen3-0.6B model, compared
   token for token with the Hugging Face tokenizer of the same model, and round-tripped byte for byte.
2. Logits at every batch size where the CUDA LittleBit op changes kernels (docs/engine-changes.md and
   ggml-cuda/littlebit-dispatch.h: decode kernels up to 8 tokens; with the tensor-core prefill path every larger batch;
   with GGML_CUDA_LITTLEBIT_F32_PREFILL=1 the chunked decode kernels up to 16 / 24 tokens (the band of each Qwen3-0.6B
   linear) and the float32 GEMM path above), and at the 512-token physical batch boundary. For each prompt length the
   last-position logits of llama-debug are compared, all finite, against the float32 PyTorch reference
   (tools/dev/reference.py): CPU (AVX2 kernel, `-dev none`) and CUDA (both prefill paths) in the exact configuration
   (`-fa off -ctk f32 -ctv f32`, F32 output head, `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`, so only the LittleBit kernels and
   ordinary float32 rounding remain), plus CUDA in the default configuration against the established tolerance of
   tests/engine/test_fused_graph_qwen3_06b.py.
3. Through the C API: the context boundaries (n_ctx - 1, n_ctx, n_ctx + 1 prompt tokens, generation that reaches
   n_ctx), long prompts (4k, 8k tokens), greedy determinism (64 tokens, repeated runs and fresh processes) and the
   CUDA-vs-CPU divergence point of greedy text, with the logit margin at that point compared to the measured
   CUDA-vs-CPU logit difference (numeric noise, not a bug, if the top-2 margin is within that difference).
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import numpy as np
import pytest

import _rb
from conftest import ENGINE_BIN, QWEN06_CKPT, QWEN06_F32HEAD, QWEN06_HF, ROOT

EXE = ".exe" if os.name == "nt" else ""
# Prompt lengths: every dispatch boundary of the CUDA op (8/9; the chunk bands 16/17 and 24/25; 32/33, 40/41, 48/49
# are the bands of larger layers and the old limits), 64, the 128/256 powers of two, the 512 physical batch, 1 and 2.
BOUNDARY_LENGTHS = [1, 2, 7, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 64, 127, 128, 129, 255, 256, 257, 511, 512, 513]
EXACT = ("-fa", "off", "-ctk", "f32", "-ctv", "f32")
# Established tolerances (tests/engine/test_fused_graph_qwen3_06b.py): CPU exact 1e-3, GPU 0.1 in every configuration
# (max-abs over the 151936 logits of this init-only model, logit rms about 2.5). The GPU bound is that loose because the
# CUDA backend itself is not exact for prompts of several tokens: the DENSE Qwen3-0.6B (no LittleBit at all) shows the
# same floor (measured here per length, column "dense-cuda-exact"; docs/engine-changes.md, noise floor table).
CPU_EXACT_MAX_ABS = 1e-3
CUDA_EXACT_MAX_ABS = 0.1
GPU_DEFAULT_MAX_ABS = 0.1
DENSE_F32 = ROOT / ".cache" / "gguf" / "qwen3-0.6b-base-f32.gguf"


@functools.lru_cache(maxsize=1)
def hf_tokenizer():
    from transformers import AutoTokenizer
    if not (QWEN06_HF / "tokenizer.json").exists():
        pytest.skip("Qwen3-0.6B tokenizer not cached")
    return AutoTokenizer.from_pretrained(str(QWEN06_HF))


# --------------------------------------------------------------------------------------------------------------
# 1. Tokenizer
# --------------------------------------------------------------------------------------------------------------
TEXTS = {
    "empty": "",
    "one-char": "a",
    "space": " ",
    "whitespace-mix": " \t\n\r\n    x",
    "special-im_start": "<|im_start|>user\nhi<|im_end|>",
    "special-endoftext": "<|endoftext|>",
    "think-tags": "<think>\nreasoning\n</think>\n\nanswer",
    "emoji": "🙂🙃👩‍👩‍👧‍👦🇫🇷 ok",
    "cjk": "日本語の文章と中文句子와 한국어",
    "rtl": "مرحبا بالعالم שלום עולם ‏",
    "combining": "é ä ñ ẛ̣ Z͑ͫ̓ͪ̂",
    "long-word": "a" * 5000,
    "long-mixed-word": "Pneumonoultramicroscopicsilicovolcanoconiosis" * 100,
    "repeated-token": " the" * 3000,
    "digits": "3.14159265358979323846264338327950288419716939937510" * 20,
    "code": "def f(x):\n\treturn {'a': [1, 2, 3]}  # comment\n",
    "bidi-controls": "‮abc‬ ⁦x⁩",
    "zero-width": "a​b‌c‍d﻿e",
    "nfd-latin": "café naïve Ångström",   # decomposed (NFD) accents
    "nfd-hangul": "한국",       # Korean as conjoining jamo
}
# Texts whose Hugging Face tokenization differs from the engine's ONLY because the Qwen3 tokenizer.json normalizes to
# NFC first and llama.cpp does not (checked below: the engine's tokens of the NFC form equal the Hugging Face tokens).
# Reported as an open decision in the Task 20 report (normalizing would change lm_tokenize's byte-exact round trip).
NFC_SENSITIVE = {"combining", "nfd-latin", "nfd-hangul"}
RAW = {  # byte strings that are not valid UTF-8 (only the C API takes them)
    "invalid-utf8-ff": b"abc\xff\xfedef",
    "invalid-utf8-truncated": b"caf\xc3",
    "invalid-utf8-surrogate": b"\xed\xa0\x80x",
    "invalid-utf8-overlong": b"\xc0\xafx",
    "nul-bytes": b"a\x00b\x00\x00c",
    "only-nul": b"\x00",
}


def test_tokenizer_matches_hugging_face_and_round_trips(qwen06: Path) -> None:
    tok = hf_tokenizer()
    cases = {k: v.encode("utf-8") for k, v in TEXTS.items()} | RAW
    o = _rb.run_child(f"""
st, m = load({str(qwen06)!r}, {{'devices': 'none'}})
out = {{}}
for k, b in {cases!r}.items():
    import unicodedata
    try:
        nfc = unicodedata.normalize('NFC', b.decode('utf-8')).encode('utf-8')
    except UnicodeDecodeError:
        nfc = b
    s0, ids0 = tokenize(m, b, False)
    ids_nfc = tokenize(m, nfc, False)[1]
    s1, ids1 = tokenize(m, b, True)
    d0 = detokenize(m, ids0, False)[1]
    d1 = detokenize(m, ids1, True)[1]
    out[k] = dict(st=[s0, s1], ids_plain=ids0, ids_nfc=ids_nfc, ids_special=ids1, rt_plain=(d0 == b), rt_special=(d1 == b), back=d0.hex()[:80])
emit(**out)
""", timeout=300)
    assert o.clean, o.describe()
    r = o.result
    bad = []
    for k, v in TEXTS.items():
        e = r[k]
        hf_plain = tok(v, add_special_tokens=False, split_special_tokens=True)["input_ids"]
        hf_special = tok(v, add_special_tokens=False)["input_ids"]
        if k in NFC_SENSITIVE:  # see NFC_SENSITIVE: the raw tokens differ, the NFC form must match exactly
            ok = (e["st"] == ["LM_OK", "LM_OK"] and e["ids_plain"] != hf_plain and e["ids_nfc"] == hf_plain
                  and e["rt_plain"] and e["rt_special"])
            print(ascii(f"{k}: engine {len(e['ids_plain'])} tokens, Hugging Face (NFC) {len(hf_plain)}; engine on the NFC "
                        f"form == Hugging Face: {e['ids_nfc'] == hf_plain}"))
            if not ok:
                bad.append((k, "NFC", e["ids_plain"][:20], e["ids_nfc"][:20], hf_plain[:20]))
            continue
        ok = (e["st"] == ["LM_OK", "LM_OK"] and e["ids_plain"] == hf_plain and e["ids_special"] == hf_special
              and e["rt_plain"] and e["rt_special"])
        print(ascii(f"{k}: {len(hf_plain)} tokens, plain==HF {e['ids_plain'] == hf_plain}, special==HF "
                    f"{e['ids_special'] == hf_special}, round trip {e['rt_plain']}/{e['rt_special']}"))
        if not ok:
            bad.append((k, e["st"], e["ids_plain"][:20], hf_plain[:20], e["ids_special"][:20], hf_special[:20]))
    for k in RAW:  # no reference for invalid UTF-8: it must tokenize cleanly, and NUL bytes must round-trip
        e = r[k]
        print(ascii(f"{k}: {e['st']} {len(e['ids_plain'])} tokens, round trip {e['rt_plain']} -> {e['back']}"))
        if e["st"] != ["LM_OK", "LM_OK"]:
            bad.append((k, e["st"]))
        if k.startswith("nul") or k == "only-nul":
            if not e["rt_plain"]:
                bad.append((k, "NUL bytes did not round-trip", e["back"]))
    assert not bad, bad


# --------------------------------------------------------------------------------------------------------------
# 2. Logits at the kernel dispatch boundaries
# --------------------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _corpus_ids() -> list[int]:
    text = (ROOT / ".cache" / "data" / "wiki.test.raw").read_text(encoding="utf-8")[:200_000]
    return hf_tokenizer()(text, add_special_tokens=False)["input_ids"]


def prompt_of(n: int) -> tuple[str, list[int]]:
    """A natural-text prompt that the Qwen3 tokenizer splits into exactly `n` tokens (searched from the WikiText-2
    test text; the decoded prefix must re-tokenize to the same ids)."""
    tok = hf_tokenizer()
    ids = _corpus_ids()
    for start in range(0, 20_000, 97):
        cand = ids[start:start + n]
        text = tok.decode(cand)
        if tok(text, add_special_tokens=False)["input_ids"] == cand and not text.startswith(" "):
            return text, cand
    raise AssertionError(f"no {n}-token prompt found")


def debug_logits(gguf: Path, text: str, ngl: int, args: tuple, env: dict | None = None, n_ctx: int = 1024) -> tuple[np.ndarray, list[int]]:
    """Last-position logits and token ids of llama-debug (one batch of up to 512 tokens per call, as the engine's
    default physical batch)."""
    import subprocess
    exe = ENGINE_BIN / ("llama-debug" + EXE)
    if not exe.exists():
        pytest.skip(f"{exe} not built")
    out = _rb.SCRATCH / "logits" / f"{os.getpid()}"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("llamacpp-*"):
        old.unlink()
    dev = ("-dev", "none") if ngl == 0 else ()
    cmd = [str(exe), "-m", str(gguf), "-p", text, "-ngl", str(ngl), "-c", str(n_ctx), "-b", str(n_ctx), "--save-logits",
           "--logits-output-dir", str(out), *dev, *args]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
                       env=_rb.child_env(env), stdin=subprocess.DEVNULL)
    assert p.returncode == 0, (p.returncode, p.stdout[-1500:], p.stderr[-1500:])
    (tok_file,) = out.glob("llamacpp-*-tokens.bin")
    logits = np.fromfile(tok_file.with_name(tok_file.name.removesuffix("-tokens.bin") + ".bin"), dtype=np.float32)
    return logits, np.fromfile(tok_file, dtype=np.int32).tolist()


@functools.lru_cache(maxsize=1)
def _reference():
    import sys
    sys.path.insert(0, str(ROOT / "tools" / "dev"))
    from reference import build_model
    return build_model(str(QWEN06_HF), QWEN06_CKPT).float().eval()


def reference_logits(ids: list[int]) -> np.ndarray:
    import torch
    with torch.no_grad():
        return _reference()(torch.tensor([ids])).logits[0, -1].float().numpy()


@functools.lru_cache(maxsize=1)
def _dense_reference():
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(str(QWEN06_HF), torch_dtype=torch.float32).eval()


def dense_reference_logits(ids: list[int]) -> np.ndarray:
    import torch
    with torch.no_grad():
        return _dense_reference()(torch.tensor([ids])).logits[0, -1].float().numpy()


CONFIGS = {
    # name: (gguf, ngl, args, env)
    "cpu-exact": ("f32head", 0, EXACT, {}),
    "cuda-bp-exact": ("f32head", 99, EXACT, {"GGML_CUDA_CUBLAS_COMPUTE_TYPE": "f32"}),
    "cuda-f32-exact": ("f32head", 99, EXACT, {"GGML_CUDA_CUBLAS_COMPUTE_TYPE": "f32", "GGML_CUDA_LITTLEBIT_F32_PREFILL": "1"}),
    "cuda-bp-default": ("f16head", 99, (), {}),
    "cuda-f32-default": ("f16head", 99, (), {"GGML_CUDA_LITTLEBIT_F32_PREFILL": "1"}),
}


@pytest.fixture(scope="module")
def boundary_table(qwen06: Path, has_cuda: bool) -> dict:
    """Run every configuration at every boundary length once (shared by the assertions below); also written to
    .cache/robust/boundary_logits.json."""
    if not QWEN06_F32HEAD.exists():
        pytest.skip(f"{QWEN06_F32HEAD} missing (built by tests/engine/test_fused_graph_qwen3_06b.py)")
    if not (QWEN06_CKPT / "model.safetensors").exists():
        pytest.skip("Qwen3-0.6B LittleBit checkpoint not cached")
    ggufs = {"f32head": QWEN06_F32HEAD, "f16head": qwen06}
    table: dict = {}
    for n in BOUNDARY_LENGTHS:
        text, ids = prompt_of(n)
        ref = reference_logits(ids)
        row = {"n": n}
        for name, (g, ngl, args, env) in CONFIGS.items():
            if ngl and not has_cuda:
                continue
            lg, eng_ids = debug_logits(ggufs[g], text, ngl, args, env)
            assert eng_ids == ids, (n, name, eng_ids[:10], ids[:10])
            row[name] = {"finite": bool(np.isfinite(lg).all()), "max_abs": float(np.abs(lg - ref).max()),
                         "top1": int(lg.argmax()) == int(ref.argmax())}
        if has_cuda and DENSE_F32.exists():  # the engine's own CUDA floor at this length, without any LittleBit
            lg, _ = debug_logits(DENSE_F32, text, 99, EXACT, {"GGML_CUDA_CUBLAS_COMPUTE_TYPE": "f32"})
            row["dense-cuda-exact"] = {"finite": bool(np.isfinite(lg).all()),
                                       "max_abs": float(np.abs(lg - dense_reference_logits(ids)).max()), "top1": True}
        table[n] = row
        print(ascii(f"n={n}: " + ", ".join(f"{k} {v['max_abs']:.2e}{'' if v['top1'] else ' TOP1-DIFF'}"
                                         f"{'' if v['finite'] else ' NONFINITE'}" for k, v in row.items() if k != "n")))
    (_rb.SCRATCH / "boundary_logits.json").write_text(json.dumps(table, indent=1))
    return table


@pytest.mark.parametrize("n", BOUNDARY_LENGTHS)
def test_logits_at_dispatch_boundaries(boundary_table: dict, n: int) -> None:
    row = boundary_table[n]
    for name, v in row.items():
        if name == "n":
            continue
        assert v["finite"], (n, name)
        if name.startswith("dense"):
            continue  # recorded for comparison only
        limit = {"cpu-exact": CPU_EXACT_MAX_ABS, "cuda-bp-exact": CUDA_EXACT_MAX_ABS, "cuda-f32-exact": CUDA_EXACT_MAX_ABS,
                 "cuda-bp-default": GPU_DEFAULT_MAX_ABS, "cuda-f32-default": GPU_DEFAULT_MAX_ABS}[name]
        assert v["max_abs"] <= limit, (n, name, v)
        if name.endswith("exact"):
            assert v["top1"], (n, name, v)


# --------------------------------------------------------------------------------------------------------------
# 3. Through the C API: context limits, long prompts, determinism, CUDA vs CPU divergence
# --------------------------------------------------------------------------------------------------------------
def _ids_snippet(model: Path, devices: str, body: str) -> str:
    return f"st, m = load({str(model)!r}, {{'devices': {devices!r}}})\n" + body


@pytest.mark.parametrize("devices", ["none", "CUDA0"])
def test_context_boundaries(qwen06: Path, has_cuda: bool, devices: str) -> None:
    """n_ctx 64: a 63-token prompt leaves room for exactly one token; 64 and 65 tokens are LM_ERR_CONTEXT_FULL (with a
    message); a 10-token prompt with max_tokens 1000 stops at the context (54 tokens), LM_OK; a prompt longer than
    n_batch is split (129 tokens, n_batch 128) and gives the same text as with one batch."""
    if devices != "none" and not has_cuda:
        pytest.skip("needs CUDA")
    p63, p64, p65, p10, p129 = (prompt_of(n)[0] for n in (63, 64, 65, 10, 129))
    o = _rb.run_child(_ids_snippet(qwen06, devices, f"""
_, s = session(m, {{'n_ctx': 64}})
g = {{'temperature': 0.0, 'max_tokens': 1000}}
out = {{}}
for k, p in (('63', {p63!r}), ('64', {p64!r}), ('65', {p65!r}), ('10', {p10!r})):
    st, text, c = generate(s, p.encode(), g)
    out[k] = (st, len(c.pieces), err() if st != 'LM_OK' else '')
_, s1 = session(m, {{'n_ctx': 256, 'n_batch': 128}}); _, s2 = session(m, {{'n_ctx': 256}})
a = generate(s1, {p129!r}.encode(), {{'temperature': 0.0, 'max_tokens': 32}}); b = generate(s2, {p129!r}.encode(), {{'temperature': 0.0, 'max_tokens': 32}})
out['split_same'] = a[1] == b[1]
emit(**out)
"""), timeout=300)
    assert o.clean, o.describe()
    r = o.result
    print(ascii(r))
    assert r["63"][0] == "LM_OK" and r["63"][1] == 1, r
    assert r["64"][0] == "LM_ERR_CONTEXT_FULL" and "64 tokens" in r["64"][2], r
    assert r["65"][0] == "LM_ERR_CONTEXT_FULL", r
    assert r["10"][0] == "LM_OK" and r["10"][1] <= 54, r   # an end-of-generation token may come earlier
    if devices == "none":
        assert r["split_same"], r   # on CUDA another batch split may sum in another order: recorded only


@pytest.mark.parametrize("n_prompt,devices", [(4096, "CUDA0"), (8000, "CUDA0"), (4096, "none")])
def test_long_prompts(qwen06: Path, has_cuda: bool, n_prompt: int, devices: str) -> None:
    """Long prompts through the C API (n_ctx 8192): the generation works, and on the CPU the text is the same with the
    default 512-token batch and with a 2048-token batch."""
    if devices != "none" and not has_cuda:
        pytest.skip("needs CUDA")
    text, ids = prompt_of(n_prompt)
    pf = _rb.SCRATCH / f"long_{n_prompt}.txt"
    pf.write_text(text, encoding="utf-8")
    o = _rb.run_child(_ids_snippet(qwen06, devices, f"""
p = open({str(pf)!r}, encoding='utf-8').read().encode()
n = len(tokenize(m, p)[1])
_, s1 = session(m, {{'n_ctx': 8192}}); _, s2 = session(m, {{'n_ctx': 8192, 'n_batch': 2048}})
g = {{'temperature': 0.0, 'max_tokens': 16}}
a = generate(s1, p, g); b = generate(s2, p, g)
emit(n=n, st=[a[0], b[0]], pieces=len(a[2].pieces), same=(a[1] == b[1]))
"""), timeout=900)
    assert o.clean, o.describe()
    r = o.result
    print(ascii(r))
    assert r["n"] == n_prompt and r["st"] == ["LM_OK", "LM_OK"] and r["pieces"] >= 1, r
    if devices == "none":
        assert r["same"], r


def test_long_prompt_logits_cuda_vs_cpu(qwen06: Path, has_cuda: bool) -> None:
    """4096-token prompt: last-position logits finite on CUDA (both prefill paths) and on the CPU, and CUDA (exact
    configuration, F32 head, 32-bit cuBLAS) within CUDA_EXACT_MAX_ABS of the CPU."""
    if not has_cuda or not QWEN06_F32HEAD.exists():
        pytest.skip("needs CUDA and the F32-head GGUF")
    text, ids = prompt_of(4096)
    cpu, _ = debug_logits(QWEN06_F32HEAD, text, 0, EXACT, n_ctx=4352)
    out = {}
    for name in ("cuda-bp-exact", "cuda-f32-exact"):
        _, ngl, args, env = CONFIGS[name]
        lg, _ = debug_logits(QWEN06_F32HEAD, text, ngl, args, env, n_ctx=4352)
        out[name] = (bool(np.isfinite(lg).all()), float(np.abs(lg - cpu).max()), int(lg.argmax()) == int(cpu.argmax()))
    print(ascii(out))
    assert np.isfinite(cpu).all()
    for name, (fin, d, top) in out.items():
        assert fin and d <= CUDA_EXACT_MAX_ABS and top, (name, out)


DET_SNIPPET = """
st, m = load({model!r}, {{'devices': {dev!r}}})
_, s = session(m, {{'n_ctx': 256}})
texts = [generate(s, {prompt!r}.encode(), {{'temperature': 0.0, 'max_tokens': 64}})[1].hex() for _ in range(3)]
emit(texts=texts)
"""


@pytest.mark.parametrize("cfg", ["cpu", "cuda", "cuda-f32-prefill"])
def test_greedy_determinism(qwen06: Path, has_cuda: bool, cfg: str) -> None:
    """Greedy 64-token generations (33-token prompt, so the prefill path is used) are identical within a process
    (3 runs) and across two fresh processes."""
    if cfg != "cpu" and not has_cuda:
        pytest.skip("needs CUDA")
    prompt, _ = prompt_of(33)
    dev = "none" if cfg == "cpu" else "CUDA0"
    env = {"GGML_CUDA_LITTLEBIT_F32_PREFILL": "1"} if cfg == "cuda-f32-prefill" else None
    runs = []
    for _ in range(2):
        o = _rb.run_child(DET_SNIPPET.format(model=str(qwen06), dev=dev, prompt=prompt), timeout=300, env=env)
        assert o.clean, o.describe()
        runs += o.result["texts"]
    assert len(set(runs)) == 1, [bytes.fromhex(t)[:40] for t in runs]


def completion(model: Path, prompt: str, n: int, ngl: int, args: tuple = (), env: dict | None = None) -> str:
    """Greedy continuation of `prompt` by the engine's llama-completion (any engine option can be set there, unlike
    through the C API)."""
    import re
    import subprocess
    exe = ENGINE_BIN / ("llama-completion" + EXE)
    if not exe.exists():
        pytest.skip(f"{exe} not built")
    dev = ("-dev", "none") if ngl == 0 else ()
    p = subprocess.run([str(exe), "-m", str(model), "-p", prompt, "-n", str(n), "--temp", "0", "-ngl", str(ngl), "-c", "2048",
                        "--no-display-prompt", "-no-cnv", "--seed", "0", *dev, *args], capture_output=True, timeout=900,
                       env=_rb.child_env(env), stdin=subprocess.DEVNULL)
    assert p.returncode == 0, p.stderr[-1500:]
    return re.sub(rb"\x1b\[[0-9;]*m", b"", p.stdout).decode("utf-8", "replace")  # drop the ANSI colour codes


# The numeric paths compared for greedy agreement: all of them are correct implementations that differ only in
# rounding (f16 attention / KV cache, the CUDA prefill path, the CUDA kernels themselves).
VARIANTS = {
    "cpu": (0, (), None),
    "cpu-exact": (0, EXACT, None),
    "cuda": (99, (), None),
    "cuda-exact": (99, EXACT, None),
    "cuda-f32-prefill": (99, (), {"GGML_CUDA_LITTLEBIT_F32_PREFILL": "1"}),
}


def divergence(model: Path, prompt: str, n_tokens: int, variants: dict = VARIANTS) -> dict:
    """Greedy texts of every variant and, for each pair, the first character where they differ (None = identical).

    Why pairs of CPU-only variants too: if two correct CPU paths that differ only by f16 rounding in attention part
    as early as CPU and CUDA do, then the CPU-vs-CUDA divergence is the model's sensitivity to rounding (near-ties in
    a flat output distribution), not a CUDA error."""
    texts = {k: completion(model, prompt, n_tokens, ngl, args, env) for k, (ngl, args, env) in variants.items()}
    names = list(texts)
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ta, tb = texts[a], texts[b]
            k = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
            if k is None and len(ta) != len(tb):
                k = min(len(ta), len(tb))
            pairs[f"{a} vs {b}"] = k
    return {"texts": texts, "pairs": pairs}


def same_device_bound(r: dict) -> int:
    """Earliest divergence between two variants on the SAME device (None counts as the full text length)."""
    full = min(len(t) for t in r["texts"].values())
    same = ["cpu vs cpu-exact", "cuda vs cuda-exact", "cuda vs cuda-f32-prefill", "cuda-exact vs cuda-f32-prefill"]
    return min(full if r["pairs"][k] is None else r["pairs"][k] for k in same if k in r["pairs"])


def test_cuda_vs_cpu_greedy_divergence_is_numeric_noise(qwen06: Path, has_cuda: bool) -> None:
    """Greedy 64 tokens from every numeric variant (all correct; they differ only in rounding). Required: the first
    characters agree everywhere, and CPU and CUDA (default settings) agree at least as long as two variants on the
    same device agree with each other: CUDA does not part from the CPU earlier than a mere change of attention
    precision or prefill path parts CUDA from itself. That makes the CPU/CUDA divergence rounding noise amplified by
    near-ties of this init-only model's flat output distribution, not a disagreement of the kernels."""
    if not has_cuda:
        pytest.skip("needs CUDA")
    r = divergence(qwen06, "The capital of France is", 64)
    for k, v in r["pairs"].items():
        print(ascii(f"{k}: first difference at character {v}"))
    assert all(v is None or v > 0 for v in r["pairs"].values()), r["pairs"]
    cross = r["pairs"]["cpu vs cuda"]
    full = min(len(t) for t in r["texts"].values())
    assert (full if cross is None else cross) >= same_device_bound(r), r["pairs"]


def test_cpu_exact_bound_can_fail(qwen06: Path) -> None:
    """Negative control of CPU_EXACT_MAX_ABS: the CPU in its DEFAULT configuration (flash attention, f16 KV cache) is
    not exact, and the bound must say so (measured 1.3e-2 at 16 tokens in tests/engine/test_fused_graph_qwen3_06b.py)."""
    if not (QWEN06_CKPT / "model.safetensors").exists() or not QWEN06_F32HEAD.exists():
        pytest.skip("checkpoint or F32-head GGUF missing")
    text, ids = prompt_of(33)
    lg, _ = debug_logits(QWEN06_F32HEAD, text, 0, ())
    err = float(np.abs(lg - reference_logits(ids)).max())
    print(f"cpu default config at 33 tokens: {err:.2e} (bound {CPU_EXACT_MAX_ABS:.0e})")
    assert err > CPU_EXACT_MAX_ABS


@pytest.mark.parametrize("case", sorted(NFC_SENSITIVE))
def test_sdk_tokens_match_hugging_face_for_non_nfc_text(qwen06: Path, case: str) -> None:
    """The Python SDK normalizes text to NFC (as the Qwen3 tokenizer.json does), so its tokens equal Hugging Face's
    for decomposed text too; the C API's lm_tokenize (checked above) does not normalize and differs."""
    tok = hf_tokenizer()
    text = TEXTS[case]
    o = _rb.run_child(f"""
import littlemore
m = littlemore.Model({str(qwen06)!r}, devices='none', n_ctx=256)
emit(sdk=m.tokenize({text!r}), raw=tokenize(m._model, {text!r}.encode('utf-8'))[1])
m.close()
""")
    assert o.clean, o.describe()
    hf = tok(text, add_special_tokens=False)["input_ids"]
    assert o.result["sdk"] == hf, (o.result["sdk"][:20], hf[:20])
    assert o.result["raw"] != hf   # negative control: without the SDK's normalization the tokens differ
