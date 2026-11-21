"""Noise floor of the engine's CPU/CUDA/F16-head/KV paths with NO LittleBit involved (Task 8 review, item 3).

Runs the DENSE Qwen3-0.6B (HF fp32 reference vs llama.cpp on F16, F32 and "F32 + F16 head" GGUFs) and the LittleBit B1 GGUF
on the same token ids and settings, and prints max-abs / rms for each pair.  Prints a markdown table.
Usage: .venv/Scripts/python.exe tools/dev/dense_floor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools" / "dev", ROOT / "tests" / "smoke"):
    sys.path.insert(0, str(p))
import numpy as np
import torch
import _assets as A

F16 = A.CACHE / "gguf" / "qwen3-0.6b-base-f16.gguf"
F32 = A.CACHE / "gguf" / "qwen3-0.6b-base-f32.gguf"
F32H16 = A.CACHE / "gguf" / "qwen3-0.6b-base-f32-head-f16.gguf"
CONFIGS = {"default": (), "exact": ("-fa", "off", "-ctk", "f32", "-ctv", "f32")}


def dense_reference(ids):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(A.MODEL_DIR), dtype=torch.float32).eval()
    with torch.no_grad():
        return m(torch.tensor([ids])).logits[0, -1].numpy()


def stats(a, b):
    d = a - b
    return float(np.abs(d).max()), float(np.sqrt(np.mean(d * d)))


def main():
    if not F32H16.exists():   # F32 dense with only the (tied) embedding / head in F16, like the LittleBit file
        import gguf
        A.copy_gguf(F32, F32H16, lambda t: (np.asarray(t.data).astype(np.float16), gguf.GGMLQuantizationType.F16)
                    if t.name == "token_embd.weight" else None)
    b1 = A.ensure_b1_gguf()
    rows = []
    scale_dense, scale_lb = {}, {}
    for pname, prompt in (("1tok", A.PROMPT_ONE), ("16tok", A.PROMPT_LONG)):
        ids = A.engine_logits(b1, prompt, 0)[1]
        ref_dense = dense_reference(ids)
        ref_lb = A.reference_logits(ids)
        scale_dense[pname], scale_lb[pname] = float(np.sqrt(np.mean(ref_dense ** 2))), float(np.sqrt(np.mean(ref_lb ** 2)))
        lb_cpu_exact = A.engine_logits(b1, prompt, 0, CONFIGS["exact"])[0]
        for ngl in (0, 99):
            for cname, extra in CONFIGS.items():
                for label, g in (("dense F16", F16), ("dense F32", F32), ("dense F32+F16 head", F32H16)):
                    lg, i2 = A.engine_logits(g, prompt, ngl, extra)
                    assert i2 == ids
                    rows.append((pname, ngl, cname, label + " vs dense HF fp32", *stats(lg, ref_dense)))
                lg = A.engine_logits(b1, prompt, ngl, extra)[0]
                rows.append((pname, ngl, cname, "LittleBit B1 vs LittleBit reference", *stats(lg, ref_lb)))
                rows.append((pname, ngl, cname, "LittleBit B1 vs LittleBit CPU-exact run", *stats(lg, lb_cpu_exact)))
    print("| prompt | ngl | config | pair | max-abs | rms |\n|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.2e} | {r[5]:.2e} |")


if __name__ == "__main__":
    main()
