"""Rebuild a Hugging Face checkpoint directory from a quantized Qwen3 GGUF (dev tool, speed benchmarks only).

What
    Reads a Qwen3 GGUF (any tensor types gguf-py can dequantize, e.g. Q4_K / Q6_K of a Q4_K_M file), dequantizes every
    tensor to float32, maps the GGUF names back to the Hugging Face names and writes `model.safetensors` (bfloat16, the
    dtype of the original release) next to a `config.json` and tokenizer files copied from `--side-files`.  The result
    is a normal HF directory that `make_lb2_init.py`, llama.cpp's `convert_hf_to_gguf.py` and transformers can read.
Why
    The speed gate is measured on Qwen3-0.6B and Qwen3-1.7B.  When the original bf16 weights of a model are not
    available (a 4 GB download that needs approval), a quantized GGUF already on disk gives a model with exactly the
    same architecture and tensor shapes, and only slightly different weight VALUES.  The speed of every kernel involved
    (the dense bf16 / F16 mat-vec, the LittleBit op) depends on the shapes and types only, never on the values, so such
    a "proxy" checkpoint is valid for speed measurements.  It is NOT valid for accuracy or perplexity claims about the
    original model: the weights carry the quantization error of the source file.  bench/README.md describes the
    control that checks the "values do not matter for speed" assumption on Qwen3-0.6B (real vs proxy weights).
Scope
    Qwen3 dense only (`general.architecture == "qwen3"`).  llama.cpp's Qwen3 converter writes the attention weights
    without any row permutation (unlike Llama's RoPE permutation), so the mapping below is a pure renaming; a GGUF of
    another architecture is refused.  A tied output head (no `output.weight`) stays tied (`tie_word_embeddings`).

Usage: python tools/dev/dequant_gguf_to_hf.py <in.gguf> <out_dir> --side-files <dir with config.json + tokenizer>
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "engine" / "gguf-py") not in sys.path:
    sys.path.insert(0, str(_REPO / "engine" / "gguf-py"))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

# GGUF tensor name (layer index abstracted) -> Hugging Face name, for llama.cpp's Qwen3 converter
_GLOBAL = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
_LAYER = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
_SIDE_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt", "LICENSE")


def hf_name(gguf_name: str) -> str:
    """The Hugging Face tensor name of a GGUF tensor of llama.cpp's Qwen3 layout; KeyError for anything else."""
    if gguf_name in _GLOBAL:
        return _GLOBAL[gguf_name]
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", gguf_name)
    if not m or m.group(2) not in _LAYER:
        raise KeyError(f"no Hugging Face name known for GGUF tensor {gguf_name!r}")
    return f"model.layers.{m.group(1)}.{_LAYER[m.group(2)]}"


def dequant_gguf_to_hf(src: Path, out_dir: Path, side_files: Path) -> dict[str, tuple[int, ...]]:
    """Write `out_dir` (model.safetensors in bf16 + config/tokenizer copied from `side_files`); returns name -> shape.

    Every tensor of the GGUF must map to a known HF name (no silent drops), and the hidden size / layer count of the
    GGUF must match `side_files/config.json` (so a config of another model size cannot slip in).
    """
    from safetensors.torch import save_file

    reader = gguf.GGUFReader(str(src))
    arch = reader.fields["general.architecture"].contents()
    if arch != "qwen3":
        raise ValueError(f"{src}: architecture {arch!r}; only qwen3 is supported (no permutation is undone)")
    cfg = json.loads((side_files / "config.json").read_text(encoding="utf-8"))
    for key, gkey in (("hidden_size", "qwen3.embedding_length"), ("num_hidden_layers", "qwen3.block_count"),
                      ("intermediate_size", "qwen3.feed_forward_length")):
        if int(cfg[key]) != int(reader.fields[gkey].contents()):
            raise ValueError(f"config.json {key}={cfg[key]} but the GGUF says {reader.fields[gkey].contents()}")
    tensors: dict[str, torch.Tensor] = {}
    for t in reader.tensors:
        name = hf_name(t.name)
        # gguf-py returns the numpy shape (reversed ggml ne), i.e. (out, in) for a weight: the HF layout
        w = dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)
        tensors[name] = torch.from_numpy(np.ascontiguousarray(w)).to(torch.bfloat16)
    if "lm_head.weight" not in tensors and not cfg.get("tie_word_embeddings", False):
        raise ValueError("the GGUF has no output.weight (tied head) but config.json says tie_word_embeddings=false")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    for fn in _SIDE_FILES:
        if (side_files / fn).exists():
            shutil.copy2(side_files / fn, out_dir / fn)
    (out_dir / "PROXY_WEIGHTS.txt").write_text(
        f"Weights dequantized from {src.name} ({len(tensors)} tensors); valid for speed measurements only.\n",
        encoding="utf-8")
    return {k: tuple(v.shape) for k, v in tensors.items()}


def main(argv: list[str] | None = None) -> int:
    """Command-line entry (see the module docstring)."""
    ap = argparse.ArgumentParser(description="Dequantize a Qwen3 GGUF into a Hugging Face directory (speed proxy).")
    ap.add_argument("src", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--side-files", type=Path, required=True, help="directory with config.json and tokenizer files")
    a = ap.parse_args(argv)
    shapes = dequant_gguf_to_hf(a.src, a.out_dir, a.side_files)
    print(f"wrote {a.out_dir}: {len(shapes)} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
