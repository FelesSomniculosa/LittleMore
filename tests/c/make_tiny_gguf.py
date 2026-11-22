"""Build the tiny LittleBit GGUF that the C API test (tests/c/test_api.c) loads.

The C test cannot build a model itself, so CTest runs this script first (a test fixture, see the top-level
CMakeLists.txt) and passes the resulting file to the test. The model is the tiny random Llama of
tests/convert/_tiny.py (2 layers, hidden size 256, a 300-piece SentencePiece vocabulary trained on a few
sentences, so nothing is downloaded), turned into an init-only LittleBit-2 checkpoint (residual path on) and
converted to a B1 GGUF with tools/convert/lb_to_gguf.py. The weights are random: the text it generates is
gibberish; the test checks plumbing (loading, templates, streaming, determinism, auto-fit), not quality.

One change to the tiny model: its tokenizer_config.json gets the ChatML chat template, so the GGUF carries a
`tokenizer.chat_template` and the test can check that lm_chat applies the template stored in the model (the
template is plain ChatML, which llama.cpp recognises by its "<|im_start|>" marker).

The output is cached: it is rebuilt only when it is missing or when this script, the tiny-model builder, the
checkpoint maker or the converter changed (a `.key` stamp next to it holds a hash of those sources).

Usage:  python tests/c/make_tiny_gguf.py <output.gguf>
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools" / "dev", ROOT / "tests" / "convert", ROOT / "tests" / "smoke"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import _assets as A  # noqa: E402  (path set up above)

# ChatML, the template format of Qwen and many other models: "<|im_start|>{role}\n{content}<|im_end|>\n" per
# message, then "<|im_start|>assistant\n" to open the answer.
CHATML = ("{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + "
          "'<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
SEED = 0
BPW = 2.0


def build(out: Path) -> Path:
    """Create (or reuse) the tiny GGUF at `out`; returns `out`."""
    key = A._digest(Path(__file__), ROOT / "tests" / "convert" / "_tiny.py", ROOT / "tools" / "dev" / "make_lb2_init.py",
                    ROOT / "tools" / "convert", SEED, BPW, CHATML)
    if A._stamp_ok(out, key):
        return out
    import _tiny
    from make_lb2_init import make_checkpoint
    from tools.convert import lb_to_gguf

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        hf = _tiny.make_tiny_model("llama", tmp / "hf", seed=SEED)
        cfg_path = hf / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["chat_template"] = CHATML
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        ck = tmp / "ckpt"
        make_checkpoint(str(hf), ck, BPW, use_itq=False, residual=True, seed=SEED)
        gguf_tmp = tmp / "model.gguf"
        lb_to_gguf.convert(ck, gguf_tmp)
        part = out.with_name(out.name + ".tmp")
        shutil.copyfile(gguf_tmp, part)
        A._commit(part, out, key)
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    path = build(Path(argv[1]).resolve())
    print(f"tiny GGUF ready: {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
