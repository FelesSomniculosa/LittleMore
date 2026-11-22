"""H. Full end to end on the community Qwen3-8B LittleBit model (LITTLEMORE_RUN_8B=1; never downloads or converts).

The GGUF was converted in Task 17 (docs/quickstart.md, section 3). Checked here, after the Task 20 fixes:
  * the Python SDK loads it with auto-fit: every layer and the head on the GPU, below 6 GiB of GPU memory, and greedy
    generation gives the text documented in docs/quickstart.md, section 4;
  * a multi-turn chat (full history passed on each call, as documented), streaming equal to non-streaming, the Qwen3
    template equal to the Hugging Face rendering for a one-turn chat;
  * CPU only (devices="none"): the same model works without the GPU, and where CPU and CUDA greedy texts part, a mere
    change of rounding on one device parts them as early (tests/robustness/test_c_prompts.py, `divergence`);
  * a C program (tests/robustness/c/example_chat.c, the header's minimal example) chats with it;
  * the WikiText-2 perplexity with the card-comparable settings (2048-token windows, every token scored, the test split
    joined with blank lines: `-c 1025 --ppl-stride 2047 -b 1024`, docs/quickstart.md section 7) is unchanged:
    24.4937 was measured in Task 17 over all 146 windows; LITTLEMORE_8B_PPL_WINDOWS=N (at least 8) spot-checks the first
    N windows instead and then compares their running perplexity with the value recorded here.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

import _rb
from conftest import CKPT_8B, ENGINE_BIN, ROOT

EXE = ".exe" if os.name == "nt" else ""
QUICKSTART_TEXT = " Paris, and the city is the capital of the French Republic. The capital is located in the French Riviera"
PPL_ALL_WINDOWS = 24.4937          # Task 17, all 146 windows (docs/quickstart.md, section 7)
NN_RAW = ROOT / ".cache" / "data" / "wiki.test.nn.raw"


def _sdk(body: str, timeout: float = 900, env: dict | None = None) -> _rb.Outcome:
    return _rb.run_child(f"import sys\nsys.path.insert(0, {str(ROOT / 'python')!r})\nimport littlemore\n" + body,
                         timeout=timeout, env=env)


def test_sdk_autofit_generation_and_gpu_memory(gguf_8b: Path) -> None:
    o = _sdk(f"""
import subprocess
def used():
    return int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()[0])
before = used()
with littlemore.Model({str(gguf_8b)!r}) as m:
    info = m.info()['model']
    text = m.generate('The capital of France is', max_tokens=32, temperature=0)
    emit(n_gpu_layers=info['n_gpu_layers'], n_layer=info['n_layer'], devices=info['gpu_devices'], text=text,
         gpu_mib=used(), before_mib=before)
""")
    assert o.clean, o.describe()
    r = o.result
    print(ascii(r))
    assert r["n_gpu_layers"] == r["n_layer"] + 1 == 37 and r["devices"] == "CUDA0", r
    assert r["gpu_mib"] < 6 * 1024, r
    assert r["text"].startswith(QUICKSTART_TEXT), r["text"]


def test_multi_turn_chat_streaming_and_template(gguf_8b: Path) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(CKPT_8B))
    one = [{"role": "user", "content": "What is the capital of France?"}]
    hf = tok.apply_chat_template(one, tokenize=False, add_generation_prompt=True)
    o = _sdk(f"""
with littlemore.Model({str(gguf_8b)!r}) as m:
    g = dict(temperature=0, max_tokens=48)
    hist = [{{'role': 'system', 'content': 'You are a helpful assistant. Answer briefly.'}}]
    answers = []
    for q in ('Name a large city in France.', 'What river flows through it?', 'And one famous building there?'):
        hist.append({{'role': 'user', 'content': q}})
        a = m.chat(hist, **g)
        answers.append(a)
        hist.append({{'role': 'assistant', 'content': a}})
    streamed = ''.join(m.chat(hist[:-1], stream=True, **g))
    one_chat = m.chat({one!r}, **g)
    one_gen = m.generate({hf!r}, **g)
    emit(answers=answers, stream_same=(streamed == answers[-1]), template_same=(one_chat == one_gen), one=one_chat)
""")
    assert o.clean, o.describe()
    r = o.result
    print(ascii(r))
    assert all(len(a) > 0 for a in r["answers"]) and r["stream_same"] and r["template_same"], r


def test_cpu_only_and_divergence(gguf_8b: Path) -> None:
    o = _sdk(f"""
import time
t = time.time()
with littlemore.Model({str(gguf_8b)!r}, devices='none') as m:
    info = m.info()['model']
    text = m.generate('The capital of France is', max_tokens=16, temperature=0)
emit(n_gpu_layers=info['n_gpu_layers'], devices=info['gpu_devices'], text=text, s=round(time.time() - t, 1))
""")
    assert o.clean, o.describe()
    r = o.result
    print(ascii(r))
    assert r["n_gpu_layers"] == 0 and r["devices"] == "" and len(r["text"]) > 0, r
    from test_c_prompts import divergence, same_device_bound
    d = divergence(gguf_8b, "The capital of France is", 64)
    for k, v in d["pairs"].items():
        print(ascii(f"8B {k}: first difference at character {v}"))
    for k, t in d["texts"].items():
        print(ascii(f"8B {k}: {t[:100]!r}"))
    cross = d["pairs"]["cpu vs cuda"]
    full = min(len(t) for t in d["texts"].values())
    assert (full if cross is None else cross) >= min(same_device_bound(d), full), d["pairs"]


def test_c_api_example(gguf_8b: Path) -> None:
    exe = ROOT / "build" / "bin" / ("example_chat" + EXE)
    if not exe.exists():
        pytest.skip("example_chat not built (cmake --build build)")
    p = subprocess.run([str(exe), str(gguf_8b), "What is the capital of France?", "24"], capture_output=True, timeout=600,
                       env=_rb.child_env({"LITTLEMORE_LOG": "error"}))
    out, err = p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")
    print(ascii(f"exit {p.returncode}; stderr {err.strip()[-300:]!r}; answer {out.strip()[:200]!r}"))
    assert p.returncode == 0 and "37 of 37 layers on the GPU (CUDA0)" in err and out.strip(), (p.returncode, err[-800:])


def test_perplexity_unchanged(gguf_8b: Path) -> None:
    exe = ENGINE_BIN / ("llama-perplexity" + EXE)
    if not exe.exists() or not NN_RAW.exists():
        pytest.skip("llama-perplexity or .cache/data/wiki.test.nn.raw missing (tests/e2e/test_community_8b.py builds it)")
    windows = int(os.environ.get("LITTLEMORE_8B_PPL_WINDOWS", "146"))
    assert windows >= 8
    cmd = [str(exe), "-m", str(gguf_8b), "-f", str(NN_RAW), "-c", "1025", "--ppl-stride", "2047", "-b", "1024", "-ngl", "99"]
    if windows < 146:
        cmd += ["--chunks", str(windows)]
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600,
                       env=_rb.child_env(), stdin=subprocess.DEVNULL)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-2000:]
    running = [float(x) for x in re.findall(r"\[\d+\]([0-9.]+)", out)]
    final = re.search(r"Final estimate: PPL = ([0-9.]+)", out)
    print(f"{len(running)} windows in {time.time() - t:.0f} s; first 8 running values {running[:8]}; "
          f"final {final.group(1) if final else None}")
    assert len(running) >= min(windows, 146) and all(v == v for v in running)
    if windows >= 146:
        assert final and abs(float(final.group(1)) - PPL_ALL_WINDOWS) <= 5e-4, (final.group(1) if final else out[-1500:])
