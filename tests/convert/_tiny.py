"""Offline builders for tiny random HF models and their init-only LittleBit checkpoints.

Used by tests/convert.  Nothing here touches the network: models are created from
tiny configs with random weights and saved to a directory, then turned into a
Samsung-format checkpoint with tools/dev/make_lb2_init.make_checkpoint.

Tokenizers (the engine's convert_hf_to_gguf needs one to write the GGUF vocabulary):
  * Qwen3 / Qwen3.5: the Qwen3-0.6B tokenizer cached under .cache/models/Qwen3-0.6B
    (the converter recognises a BPE pre-tokenizer only by hashing a known one, so a
    home-made BPE would be rejected).  The model's vocab_size must cover it
    (151936), which makes these tiny models ~80 MB each; the tests skip when the
    cache is absent.
  * Llama / Gemma 3: a tiny SentencePiece model trained here on a few sentences
    (vocab 300), so these two families need no downloads at all.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
QWEN_TOK_DIR = ROOT / ".cache" / "models" / "Qwen3-0.6B"
QWEN_TOK_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
QWEN_VOCAB = 151936
SPM_VOCAB = 300

FAMILIES = ("qwen3", "llama", "gemma3", "qwen35")


def qwen_tokenizer_available() -> bool:
    """True when the cached Qwen3-0.6B tokenizer files exist."""
    return all((QWEN_TOK_DIR / f).exists() for f in QWEN_TOK_FILES)


def _train_spm(out_dir: Path) -> None:
    """Train a ~300-piece SentencePiece model into `out_dir/tokenizer.model`."""
    import sentencepiece as spm
    corpus = out_dir / "_corpus.txt"
    words = ("the quick brown fox jumps over the lazy dog while a small model reads "
             "bits of signs and scales to answer questions about one bit weights").split()
    corpus.write_text("\n".join(" ".join(words[i:] + words[:i]) for i in range(len(words))),
                      encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(out_dir / "tokenizer"), vocab_size=SPM_VOCAB,
        model_type="bpe", byte_fallback=True, hard_vocab_limit=False,
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1, minloglevel=2)
    corpus.unlink()
    (out_dir / "tokenizer.vocab").unlink(missing_ok=True)
    (out_dir / "tokenizer_config.json").write_text(
        '{"tokenizer_class": "LlamaTokenizer", "bos_token": "<s>", "eos_token": "</s>", '
        '"unk_token": "<unk>", "add_bos_token": true}', encoding="utf-8")


def _config(family: str, **over):
    """Tiny config of `family`; `over` overrides any field (e.g. attention_bias=True)."""
    import transformers as T
    common = dict(hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                  max_position_embeddings=128, tie_word_embeddings=True)
    if family == "qwen3":
        kw = dict(common, vocab_size=QWEN_VOCAB, bos_token_id=151643, eos_token_id=151645)
        kw.update(over)
        return T.Qwen3Config(**kw)
    if family == "llama":
        kw = dict(common, vocab_size=SPM_VOCAB, bos_token_id=1, eos_token_id=2)
        kw.pop("head_dim")
        kw.update(over)
        return T.LlamaConfig(**kw)
    if family == "gemma3":
        kw = dict(common, vocab_size=SPM_VOCAB, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                  sliding_window=16, query_pre_attn_scalar=64,
                  layer_types=["sliding_attention", "full_attention"])
        kw.update(over)
        return T.Gemma3TextConfig(**kw)
    if family == "qwen35":
        kw = dict(common, vocab_size=QWEN_VOCAB, bos_token_id=151643, eos_token_id=151645,
                  layer_types=["linear_attention", "full_attention"],
                  linear_conv_kernel_dim=4, linear_key_head_dim=32, linear_value_head_dim=32,
                  linear_num_key_heads=2, linear_num_value_heads=4)
        kw.update(over)
        return T.Qwen3_5TextConfig(**kw)
    raise ValueError(family)


def make_tiny_model(family: str, out_dir: Path, seed: int = 0, **over) -> Path:
    """Create a random tiny `family` model + tokenizer in `out_dir` (HF layout, safetensors)."""
    import transformers as T
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    cfg = _config(family, **over)
    model = T.AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    model.save_pretrained(out_dir, safe_serialization=True)
    if family in ("qwen3", "qwen35"):
        for f in QWEN_TOK_FILES:
            shutil.copy2(QWEN_TOK_DIR / f, out_dir / f)
    else:
        _train_spm(out_dir)
    return out_dir


def make_tiny_checkpoint(family: str, tmp: Path, bpw: float = 2.0, residual: bool = True,
                         kv_factor: float = 1.0, seed: int = 0, **over) -> Path:
    """Tiny HF model -> Samsung-format init-only checkpoint (`tmp/ckpt`); returns its dir."""
    from make_lb2_init import make_checkpoint
    hf = make_tiny_model(family, Path(tmp) / "hf", seed=seed, **over)
    ck = Path(tmp) / "ckpt"
    make_checkpoint(str(hf), ck, bpw, use_itq=False, residual=residual, kv_factor=kv_factor, seed=seed)
    return ck
