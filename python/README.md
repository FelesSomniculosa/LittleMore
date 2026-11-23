# LittleMore for Python

Run LittleBit (sub-1-bit) language models, and ordinary GGUF models, from Python or the terminal. The package is a
thin `ctypes` layer over the LittleMore C library: installing it needs no compiler. Wheels bundle the library and its
compute-backend plugins (CPU variants, CUDA, Vulkan).

## Quickstart

A walk-through from installing to running an 8B LittleBit model on a 6 GB GPU (download, conversion, measured speed
and quality): [docs/quickstart.md](../docs/quickstart.md).

```python
import littlemore

with littlemore.Model("model.gguf") as model:                 # n_gpu_layers="auto": fit as many layers on the GPU as fit
    print(model.chat([{"role": "user", "content": "Hello!"}]))

    for piece in model.chat([{"role": "user", "content": "Tell me a story"}], stream=True):
        print(piece, end="", flush=True)                      # a generator of text pieces

    print(model.generate("Once upon a time", max_tokens=50))  # raw continuation, no chat template
    print(model.info())                                       # library version, backends, layers on the GPU, ...
```

```
littlemore run model.gguf                       # chat in the terminal (/exit, /clear)
littlemore run model.gguf --prompt "Hello"      # one answer, for scripts
littlemore run model.gguf --cpu-only            # no GPU at all
littlemore bench model.gguf                     # load time, first-piece latency, generation speed
littlemore convert --repo CHECKOUT CHECKPOINT_DIR -o model.gguf  # LittleBit checkpoint -> GGUF (source checkout only, see below)
```

## What to know

- **Multi-turn chat**: every `chat()` call starts from scratch. Pass the whole history, including earlier assistant
  answers, each time (`littlemore run` does).
- **Sampling** is deterministic by default (the same input gives the same text, even at `temperature > 0`, because the
  default seed is 0). Keyword arguments: `temperature`, `top_k`, `top_p`, `seed`, `max_tokens`. Use
  `seed="random"` (or `littlemore.SEED_RANDOM`) for varied answers.
- **GPU use**: `n_gpu_layers="auto"` (default, `-1`) or a number. `n_gpu_layers=0` keeps the weights on the CPU, but the GPU
  still processes large prompt batches (0.7 to 1 GiB of GPU memory, much faster prompts; generation stays on the CPU).
  That is intended. For no GPU use at all pass `devices="none"` (CLI: `--cpu-only`). `devices="CUDA0"` picks a device.
  See `Model.info()["backends"]` for the device names.
- **Streaming** runs the generation in a worker thread, so the generator can be consumed from any thread. **A stream you
  stop reading early must be closed** (`stream.close()`, or `with contextlib.closing(stream) as s:`; dropping the last
  reference also closes it): until then it owns the model's session. While a stream is open, `tokenize`, `detokenize` and
  `info` keep working; a new `chat` / `generate` from the same thread raises `LittleMoreError` (status name
  `STREAM_OPEN`) instead of deadlocking; other threads wait until the stream ends or is closed. Several threads can call
  `chat` on one model; the calls run one after another. Closing the model, or leaving the program, stops open streams.
- **Text decoding** never raises: pieces are decoded incrementally as UTF-8 and invalid bytes become U+FFFD.
- **Unicode normalization** (`normalize`): by default (`Model(..., normalize="nfc")`) the SDK normalizes every text it
  passes to the library (`generate` prompts, `chat` roles and contents, `tokenize`) to NFC. A GGUF file does not record
  a tokenizer's normalizer, so the SDK cannot choose per model: **for a model whose tokenizer has no normalizer
  (Llama-3 style byte-level BPE) pass `normalize=None`**, which hands the text's UTF-8 bytes to the library exactly as
  given. `generate`, `chat` and `tokenize` take the same keyword to override the Model's setting for one call
  (`model.tokenize(text, normalize=None)`); any other value than `"nfc"` or `None` is a `ValueError`. What NFC means,
  checked on the tokenizers available here:
  - Qwen3 (every Qwen3 / Qwen3.5 `tokenizer.json` here declares `"normalizer": {"type": "NFC"}`): with the default the
    SDK's tokens equal the Hugging Face tokens for decomposed text too; without it a decomposed text ("e" + U+0301
    instead of "é", Korean as conjoining jamo) gets other and more tokens (16 instead of 8 for "café naïve Ångström"
    in NFD).
  - SentencePiece models trained with SentencePiece's default `nmt_nfkc` normalization (the tiny test models): the
    reference tokenizer gives the same tokens for the NFD and NFC forms, so NFC input moves the engine towards it.
  - A tokenizer **without** a normalizer: its reference tokens of a decomposed text are those of the decomposed bytes,
    so the default NFC differs from it on non-NFC input and `normalize=None` matches it. **Verified only by
    simulation**: no such tokenizer is available locally, so the Qwen3 vocabulary was used with its normalizer
    removed; on NFD input that reference gives 16 tokens for "café naïve Ångström" (8 with NFC) and 8 for "한국어" as
    jamo (3 with NFC). NFC input is unaffected either way.
  With the default, `detokenize(tokenize(text))` returns the NFC form of `text`. The C API itself does **not**
  normalize: it passes bytes on unchanged (docs/c-api.md, "Tokenizer and Unicode normalization").
- **Chat with a reasoning model** (Qwen3): `<think>...</think>` blocks of earlier assistant turns are removed by the
  library, as Qwen3's own chat template does, so the history may contain the model's full earlier answers.
- **Errors** are `littlemore.LittleMoreError` with `.status`, `.status_name` (e.g. `LM_ERR_MODEL_LOAD`) and the
  library's own message.
- **Closing**: `with Model(...)`, or `model.close()` (safe to repeat). Using a closed model raises `LittleMoreError`.

## CUDA on Windows

The CUDA backend needs the CUDA 12 runtime DLLs (`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`). The wheel does
not bundle them (cublasLt alone is 640 MiB, above PyPI's wheel size limit); you need either a CUDA 12 toolkit or
`pip install "littlemore[cuda]"` (NVIDIA's own runtime wheels), plus a recent NVIDIA driver (the extra is not tested yet). Measured with a PE import table dump: `ggml-cuda.dll` imports
`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll` and the driver's `nvcuda.dll` dynamically (it does not link the CUDA
runtime statically), and the plugins need the Microsoft Visual C++ Redistributable (`VCRUNTIME140`, `MSVCP140`, `VCOMP140`). Python 3.8+ does not search
`PATH` for DLL dependencies, so the SDK registers the folders itself with `os.add_dll_directory()` before loading the
library: `LITTLEMORE_DLL_DIR`, `CUDA_PATH*`, PATH folders that hold `cudart64_*.dll`, the `nvidia-*-cu12` wheels' folders
and the default toolkit install folders. If `info()["backends"]` shows no `CUDA0`, set `LITTLEMORE_DLL_DIR` to the folder
with the runtime DLLs. Without CUDA the CPU and Vulkan backends still work.

## Building a wheel

From a source checkout, after building the C library (`docs/development.md`):

```
python -m build --wheel --outdir dist python
```

The result is `dist/littlemore-<version>-py3-none-<platform>.whl`: platform specific, independent of the Python version.
On Linux the tag is `linux_x86_64`, which **PyPI does not accept** (it needs a `manylinux` tag, made by running
`auditwheel repair` on a build from a manylinux container; not done yet): such a wheel is for local use and CI tests
only. The libraries in the wheel find each other through an origin-relative RPATH (`$ORIGIN`, set in `CMakeLists.txt`);
that has not been run on Linux by the maintainer yet, the CI job "Install the wheel and import it" is its first test.
It carries `LICENSE` (MIT) and `THIRD_PARTY_NOTICES` (the llama.cpp / ggml licence for the bundled binaries) in its
`dist-info/licenses/` folder.
The files come from `build/bin` (override with `LITTLEMORE_BIN_DIR`).
In a source checkout the SDK uses the newer of `littlemore/_bin` and `build/bin` and warns once if the two differ. Install it with `pip install dist/littlemore-*.whl`.

## `littlemore convert`

`convert` runs the converter of the source repository (`tools/convert/lb_to_gguf.py`) and passes all other options
through. You must name the checkout explicitly, with `--repo PATH` or the environment variable `LITTLEMORE_REPO`: the
converter is code that gets executed, so it is never looked up in the current directory. The folder must look like the
LittleMore repository, and the path used is printed. It needs PyTorch, transformers and the GGUF writer, which the
wheel does not carry, so use the development environment of that checkout:

```
littlemore convert --repo /path/to/LittleMore CHECKPOINT_DIR -o model.gguf
```

## Tests

```
python -m pytest tests/python -q              # fast checks (need the built library)
python -m pytest tests/python -m smoke -q     # also loads a tiny model, builds a tiny checkpoint, runs the CLI
```
