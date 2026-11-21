# Development guide

How to build, test and run LittleMore end to end. LittleMore runs LittleBit
(sub-1-bit) language models through a patched llama.cpp; see
[engine-changes.md](engine-changes.md) for what the patch is.

All commands are run from the repository root in PowerShell, using the project
virtual environment (`.venv`) so no activation is needed.

## 1. Prerequisites

Install the compilers, CUDA 12.9, Vulkan SDK, Python 3.13 and the Python
packages as described in [setup-windows.md](setup-windows.md), then check:

```powershell
.venv\Scripts\python.exe tools/dev/check_env.py
```

It must print `OK: environment ready.` and exit with code 0.

## 2. Build the engine

The engine is a separate git clone in `engine/` (see
[engine-changes.md](engine-changes.md) for how to create it). Build it from a
shell that has the Microsoft C++ compiler on its path:

1. Open "Developer PowerShell for VS 2022" from the Start menu, **or** open a
   normal `cmd` window and run
   `"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"`
   (this sets up the 64-bit compiler for that window only).
2. Make sure CUDA 12.9 comes first on `PATH` (CUDA 13 may also be installed).
3. In that shell:

```
cd engine
cmake -G Ninja -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON ^
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9" ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe" ^
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j 6
```

(`^` continues a line in `cmd`; in PowerShell use a backtick instead.
`89` is the CUDA architecture of an RTX 40-series GPU; change it for other
GPUs. `-j 6` limits parallel jobs to keep RAM use low.) The programs
(`llama-cli`, `llama-completion`, `llama-debug`, ...) and the backend plugin
DLLs end up in `engine/build/bin`.

## 3. Unit tests (fast, no engine needed)

```powershell
.venv\Scripts\python.exe -m pytest tests/dev -q
```

These test the PyTorch reference (`tools/dev/reference.py`) and the checkpoint
maker (`tools/dev/make_lb2_init.py`) on tiny random models. They take under a
minute and need no downloads.

## 4. Smoke test (slow, needs the engine and a model)

The smoke test runs the whole chain: base model, init-only LittleBit-2
checkpoint, GGUF conversion, then llama.cpp on GPU and CPU, and compares the
engine's numbers with the PyTorch reference. First download the base model
once (about 1.5 GB):

```powershell
.venv\Scripts\hf.exe download Qwen/Qwen3-0.6B --local-dir .cache/models/Qwen3-0.6B
```

Then:

```powershell
.venv\Scripts\python.exe -m pytest -m smoke -v -s
```

The first run converts the model and takes several minutes; later runs reuse the
cache. If the model or the engine binaries are missing, the tests are skipped
with a message saying what is missing. The generated text is meaningless on
purpose: the checkpoint is not trained, only plumbing and numbers are tested.

### Where files are cached

Everything generated lives under `.cache/` (ignored by git, safe to delete):

| Path | Content |
|---|---|
| `.cache/models/Qwen3-0.6B/` | the downloaded base model |
| `.cache/ckpt/qwen3-0.6b-lb2-0.55/` | init-only LittleBit-2 checkpoint at 0.55 bits per weight |
| `.cache/gguf/` | the plain F32 GGUF and the LittleBit (Q1_0) GGUF |
| `.cache/logits/` | logits saved by `llama-debug` |

Each generated file has a `.key` stamp; it is rebuilt automatically when the
tool sources that produced it change.

## 5. Tools in `tools/dev`

| File | Purpose |
|---|---|
| `check_env.py` | checks that all build tools and a CUDA-enabled PyTorch are installed (exit 0 = ready) |
| `reference.py` | plain-PyTorch LittleBit model: loads a checkpoint and runs it; the ground truth for all other tests |
| `make_lb2_init.py` | makes an init-only LittleBit-2 checkpoint from any Hugging Face causal language model (no training) |
| `emulate_q8_slow_path.py` | the reference plus the int8 activation noise that llama.cpp adds, so engine and reference can be compared fairly |

Example, making a checkpoint by hand:

```powershell
.venv\Scripts\python.exe tools/dev/make_lb2_init.py .cache/models/Qwen3-0.6B out/ckpt --bpw 0.55
```

## Glossary

- **LittleBit / LittleBit-2**: methods that store a weight matrix as low-rank
  sign matrices plus scale vectors (below 1 bit per weight); LittleBit-2 adds a
  rotation and a residual path. See [engine-changes.md](engine-changes.md).
- **bpw**: bits per weight, the storage budget per original parameter.
- **GGUF**: llama.cpp's single-file model format.
- **SVD** (singular value decomposition): splits a matrix `W` into
  `U S V^T`; keeping only the largest `r` singular values gives the best rank-`r`
  approximation.
- **Joint-ITQ** (joint iterative quantization): finds an orthogonal rotation of
  the SVD factors so that taking only their signs loses as little as possible.
- **Dual-SVID**: approximates the magnitudes of the sign factors by simple
  scale vectors (a rank-1 fit), so that sign matrices times scales reproduce the
  factors.
- **QAT** (quantization-aware training): training that adapts the model to its
  low-bit form. The checkpoints made here skip it ("init-only"), so they
  produce gibberish text but are exact for testing the software.
- **GQA** (grouped-query attention): attention where several query heads share
  one key/value head, so the `k_proj` and `v_proj` matrices are smaller.
- **RoPE** (rotary position embedding): the way queries and keys are rotated
  by their position in the text.
- **rank**: the inner size `r` of the two sign matrices (`U` is out x r, `V` is
  r x in); higher rank means a closer approximation and more storage.
- **teeth**: project slang. A check "has teeth" if it actually fails when the
  code under test is broken. This is usually proved with a mutation check: break
  the code (or the reference) on purpose and confirm the test fails.
- **mutation check**: deliberately introducing a known fault to see that a
  test detects it.
- **smoke test**: a slow end-to-end test that checks the whole chain works,
  not every detail.
