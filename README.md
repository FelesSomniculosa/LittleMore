# LittleMore

Run **LittleBit** language models (weights compressed to well below 1 bit each) locally, from a C library or from
Python, on a patched [llama.cpp](https://github.com/ggml-org/llama.cpp).

> **Status: early (phase 1).** CUDA and x86 CPU work and are tested on one machine (Windows 11, RTX 4050 Laptop 6 GB).
> Nothing is published on PyPI yet. See [Limitations](#limitations-read-before-you-rely-on-it).

> **Model licence warning.** The LittleBit checkpoints that exist today are **non-commercial** (CC BY-NC 4.0, plus the
> licence of the base model). LittleMore is MIT, but it does not change the licence of a model you run, and it hosts no
> models. Read [docs/licensing.md](docs/licensing.md) before using a model in a product.

## What it is and why

LittleBit ([paper](https://arxiv.org/abs/2506.13771); LittleBit-2: [paper](https://arxiv.org/abs/2603.00042)) stores
each weight matrix as two low-rank matrices of +1/-1 signs plus a few small scale vectors. That brings a model to
0.1 to 1 bit per weight, so an 8B-parameter model fits in about 3 GB. The published reference code expands the weights
back to full size to run them, and existing community llama.cpp ports run slower than the uncompressed model. LittleMore
aims to make the small size actually usable at run time.

The four promises of the project:

1. **Small memory.** Signs are stored at exactly 1 bit each (a new ggml type, `GGML_TYPE_B1`).
2. **Faster than the uncompressed model, for token generation.** One fused op (`GGML_OP_LITTLEBIT`) computes a whole
   LittleBit linear layer, with CPU (scalar, AVX2) and CUDA kernels. The speed gate covers *token generation* on CUDA for
   Qwen3-0.6B and 1.7B, measured on one RTX 4050 Laptop GPU; see [Measured speed](#measured-speed). It does **not** cover
   prompt processing: on the GPU, the 1.7B model's prompt speed is about 0.84-0.87x that of the F16 model
   ([bench/results/phase1_13c.md](bench/results/phase1_13c.md)).
3. **Automatic hardware use.** The backends (CPU variants, CUDA, Vulkan) are runtime-loaded plugins, and the library
   fits as many layers as possible into the free GPU memory by itself.
4. **A simple library API.** A small C API (`include/littlemore.h`) and a Python SDK / CLI on top of it.

**Silent wrong output is treated as the worst bug:** the converter checks every layer it writes against an independent
float64 evaluation, every kernel is compared with a reference, and the engine's perplexity is compared with a separate
PyTorch implementation (see [docs/development.md](docs/development.md), sections 6 to 7c).

### Architecture in one picture

```
 LittleBit checkpoint (Samsung format, you download it)
        |  littlemore convert     tools/convert: checks every layer after writing
        v
     model.gguf  (B1 signs + scales)
        |
        v
 +-----------------------------------------------+
 | littlemore C library      include/, src/      |   <- also used by the Python SDK (ctypes) and CLI
 +-----------------------------------------------+
 | engine/  patched llama.cpp (git submodule)    |
 |   GGML_TYPE_B1, GGML_OP_LITTLEBIT, graph hook |
 |   backend plugins: CPU variants, CUDA, Vulkan |
 +-----------------------------------------------+
```

What the engine patch changes is documented in [docs/engine-changes.md](docs/engine-changes.md).

### What is in phase 1, and what is not

In phase 1: the engine changes (B1 type, fused op, CPU and CUDA kernels, graph hook), the converter, the C API, the
Python SDK and CLI, tests and the speed benchmark with its gate, model families Qwen3, Llama, Gemma 3 and Qwen3.5
(converter and graph tested on tiny random models of each; real weights tested on Qwen3 only).

Non-goals of the project: training or compressing models (you bring a trained checkpoint), hosting or curating models,
Kotlin/Swift wrappers, a Node SDK, a GUI, MoE models, multi-GPU splitting and upstreaming to llama.cpp.

## Install

LittleMore is not on PyPI yet. Build the C library, then build and install the wheel from a source checkout (full
steps, including what the wheel needs on the machine, are in [docs/quickstart.md](docs/quickstart.md)):

```
git clone --recurse-submodules <this repository>
# build the C library (see "Build and test" below), then
python -m build --wheel --outdir dist python
pip install dist/littlemore-*.whl
```

The wheel bundles the library and the backend plugins but **not** the NVIDIA CUDA runtime DLLs. For the GPU you need
an NVIDIA driver plus CUDA 12 runtime libraries (a toolkit, or `pip install "littlemore[cuda]"`, not tested yet);
without them it still runs on the CPU.

## Quickstart: convert and run

The walk-through with every measurement is [docs/quickstart.md](docs/quickstart.md). In short:

```
# 1. download a LittleBit checkpoint (mind its licence!), e.g. the community Qwen3-8B 0.55 bpw one
hf download LittleBitLLM/littlebit-qwen3-8b-0.55bpw --local-dir model-dir --exclude "logs/*" "runs/*"

# 2. convert it to GGUF (the converter also validates every layer; needs PyTorch, see docs/setup-windows.md)
littlemore convert --repo /path/to/this/checkout model-dir -o model.gguf

# 3. run it
littlemore run model.gguf --prompt "Hello"
littlemore bench model.gguf
```

```python
import littlemore

with littlemore.Model("model.gguf") as model:      # puts as many layers on the GPU as fit
    print(model.generate("The capital of France is", max_tokens=32, temperature=0))
```

The exact download command, a pinned revision and the expected output are in the quickstart. The Python API is in
[python/README.md](python/README.md), the C API in [docs/c-api.md](docs/c-api.md).

## Measured speed

All numbers are from files committed in this repository, measured with `llama-bench` on one laptop (RTX 4050 Laptop
GPU 6 GB, i7-13620H, Windows 11, CUDA 12.9); they are not guarantees for other hardware.

| Result | Number | Source |
|---|---|---|
| Qwen3-0.6B, generation, LittleBit 0.55 bpw vs bf16 on CUDA | 212.9 vs 126.9 tokens/s (1.68x) | [bench/results/phase1.md](bench/results/phase1.md) |
| Qwen3-1.7B, same | 142.9 vs 51.1 tokens/s (2.80x) | [bench/results/phase1.md](bench/results/phase1.md) |
| Community Qwen3-8B 0.55 bpw, all layers on the GPU | 63.5 tokens/s generation, 1223 tokens/s for a 512-token prompt | [docs/quickstart.md](docs/quickstart.md), section 6 (`bench/results/community_8b_*.json`) |
| Same 8B model, CPU only (i7-13620H) | 13.4 tokens/s generation | [docs/quickstart.md](docs/quickstart.md), section 6 |

The speed gate ("LittleBit 0.55 bpw generates faster than bf16 on CUDA for Qwen3-0.6B and 1.7B") passes on the original
weights of both models; the setup, repetitions and caveats are in [bench/results/phase1.md](bench/results/phase1.md)
and [bench/README.md](bench/README.md). The 0.6B and 1.7B rows use init-only checkpoints (made without training, so
they produce meaningless text): they measure speed, not quality.

### Quality, honestly

The community 8B checkpoint is early and **far below the original model**: WikiText-2 perplexity 24.49 here (24.48 on
its model card) against 9.73 for the original Qwen3-8B, and its continuations drift or repeat quickly
([docs/quickstart.md](docs/quickstart.md), section 7). Use it to try sub-1-bit inference, not for real tasks. What the
repository does verify is that the engine computes the same function as the PyTorch reference, within 0.1 % perplexity
on the GPU and 0.0002 % on the CPU for that checkpoint (same section).

## Build and test

You need CMake 3.22 or newer, a C++17 compiler, Python 3.10+ and, for the GPU backends, the CUDA 12 toolkit. On Windows
the whole setup is in [docs/setup-windows.md](docs/setup-windows.md); the commands are in
[docs/development.md](docs/development.md). A CPU-only build, which is what [CI](.github/workflows/ci.yml) does, is:

```
git submodule update --init --recursive
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
      -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_CURL=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R "test-utf8|test-api|littlemore-tiny-gguf|test-b1|test-gguf|test-alloc"
build/bin/test-backend-ops test -o LITTLEBIT -b CPU
python -m pytest -m "not smoke and not engine_run" -q       # tests that need no GPU and no model download
LITTLEMORE_ENGINE_BIN=build/bin python -m pytest tests/engine/test_fused_graph_families.py -m engine_run -k "llama or gemma" -q   # engine vs PyTorch reference, tiny models (needs the llama-debug target)
```

(`-m "not smoke ..."` leaves out the slow end-to-end tests that need the engine binaries and downloaded models; they are
described in [docs/development.md](docs/development.md).) CI runs this on Windows and Linux x64, plus a CUDA
compile-only job, and builds the wheel.

## Repository layout

| Path | Content |
|---|---|
| `include/`, `src/` | the C API (`littlemore.h`) and its implementation |
| `engine/` | git submodule: the patched llama.cpp (branch `littlemore`), see [docs/engine-changes.md](docs/engine-changes.md) |
| `python/` | the Python SDK and the `littlemore` command line (`ctypes` over the C library), wheel packaging |
| `tools/convert/` | LittleBit checkpoint to GGUF converter, with read-back validation |
| `tools/dev/` | developer tools: the PyTorch reference model, test-checkpoint maker, environment check, studies |
| `tests/` | unit, converter, engine, end-to-end, benchmark-logic and C API tests |
| `bench/` | speed benchmark harness, plans and the committed results |
| `docs/` | quickstart, development guide, engine changes, C API, environment variables, licensing |
| `.github/workflows/` | CI |

## Limitations (read before you rely on it)

- **Platforms.** Phase 1 targets **CUDA (NVIDIA)** and **x86 CPU**, tested on a single Windows 11 machine. Linux and
  macOS are **untested** on real hardware (CI builds Linux, but a green build is not a test of GPU behaviour).
- **Vulkan** devices are detected and a model can be loaded with them, but there is **no Vulkan kernel yet**: the fused
  op and the LittleBit weights stay on the CPU (docs/engine-changes.md, "Vulkan kernel" is an open item), and Vulkan
  speed is **not benchmarked against the gate**.
- **Not yet:** ARM CPU kernels (NEON), Metal, Android and iOS packages. They are planned, not done.
- **CUDA runtime** libraries are not bundled in the wheel; the `littlemore[cuda]` route is not tested yet.
- **Checkpoints** are non-commercial today and early in quality (see above).
- Measured numbers come from one laptop GPU; other GPUs, drivers and CPUs will differ.
- Phase 1 has not yet had a public CI run; the CUDA compile-only jobs were written without being able to run GitHub
  Actions locally.

## Licence

LittleMore's code is [MIT](LICENSE). The engine is llama.cpp/ggml (MIT). Models have their own licences, currently
non-commercial. Details: [docs/licensing.md](docs/licensing.md).
