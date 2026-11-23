# LittleMore

Run **LittleBit** language models (weights compressed to well below 1 bit each) locally, from a C library or from
Python, on a patched [llama.cpp](https://github.com/ggml-org/llama.cpp).

> **Status: early (phase 1).** NVIDIA graphics cards (CUDA) and normal 64-bit processors (x86 CPU) work. Everything has
> been tested on **one** computer only (Windows 11, one NVIDIA laptop GPU). Nothing is published on PyPI yet. See
> [Limitations](#limitations-read-before-you-rely-on-it).

> **Model licence warning.** The LittleBit checkpoints that exist today are **non-commercial** (CC BY-NC 4.0, plus the
> licence of the base model). LittleMore is MIT, but it does not change the licence of a model you run, and it hosts no
> models. Read [docs/licensing.md](docs/licensing.md) before using a model in a product.

---

# Part 1: The short version (no jargon)

This part is for everyone. [Part 2](#part-2-for-developers) is for developers.

## Why this project is useful

AI chat models are big. A good one can need 16 GB or more of memory just to be loaded, which is more than many
computers have to spare. People end up renting a server, sending their text to a cloud service, or settling for a much
smaller and weaker model.

A compression method called **LittleBit** squeezes a model to a fraction of its normal size. The Qwen3 8B model, for
example, shrinks from about 16 GB to about 3 GB. Until now, though, the programs that could open these squeezed models
were slow. Some were even slower than the full-size model, so you saved space and lost speed.

LittleMore is the tool that makes the small size **actually usable**:

- **It needs far less memory.** A model that normally needs a very large amount of memory needs a fraction of it.
  The smaller the machine, the more that matters.
- **It stays on your machine.** Nothing is sent to a cloud service, so your text stays private and it works offline.
- **It is fast.** In our tests, the small models answered faster than the full-size versions (numbers below).
- **It is easy to try.** One command converts a downloaded model, and a few lines of Python or C code run it.

## How does it compare?

### Memory: how much space does the model need?

Memory needs follow the size of the model, so this part does not depend on which computer you have. The same percentages
apply whether the memory is on a graphics card or in the main memory of the computer.

For the Qwen3 8B model, three versions:

| | Full-size version | Usual "4-bit" version | **LittleMore (LittleBit)** |
|---|---|---|---|
| Size of the model | about 16 GB | about 5.0 GB | **about 3.0 GB** |
| Compared with the usual 4-bit version | - | baseline | **about 39% less** |
| Compared with the full-size version | baseline | - | **about 81% less** |
| Memory used while chatting (our test) | not measured | about 5.3 GB | **about 2.7 GB (about 49% less)** |

In plain terms: **whatever amount of memory your computer has, LittleMore's 8B model needs roughly 40% to 50% less than
the usual 4-bit version, and about 80% less than the full-size one.**

### Speed: how fast does it write?

"Speed" means how many word pieces (called tokens) the model writes per second while answering. More is better. These
numbers are from **one test laptop** (an NVIDIA RTX 4050 laptop graphics card, an Intel i7-13620H processor,
Windows 11). A different computer will give different numbers; what carries over is the *comparison*, not the exact
figure.

| Model | Full-size version | With LittleMore | Result |
|---|---|---|---|
| Qwen3 0.6B (small) | 127 per second | **213 per second** | about 1.7 times faster |
| Qwen3 1.7B (medium) | 51 per second | **143 per second** | about 2.8 times faster |
| Qwen3 8B (large) | not measured (too large for our test laptop's graphics card) | **63 per second** | - |

For the 8B model, a quick informal try on the same laptop showed the usual 4-bit version writing at a little over a
third of LittleMore's speed (about 23 against 63 per second, roughly 2.7 times slower). That was a casual run, not a proper benchmark; a careful head-to-head is
planned.

### Electricity: how much power does it use?

This is an **estimate**, not a measurement. Writing text is limited by how much data has to be read from memory for
every word. A model that is about 40% smaller, and that writes words noticeably faster, should need clearly less energy
per word. Our estimate is **roughly 40% to 60% less energy per word** than the usual 4-bit version. We have not yet
measured power with a meter, so treat it as "roughly half, give or take", not as a promise. The size of the saving on
your computer will depend on your hardware.

### Things to know before you rely on these numbers

- **Reading a long question is not faster.** On the medium model, the "reading your prompt" step is about 15% slower
  than the full-size model. The speed-up is for the answering part.
- **Quality is not the same as the original.** The only large LittleBit model available today (the 8B) is an early
  community version. It runs well, but its answers are clearly worse than the original Qwen3 8B, so use it to try the
  technology, not for real work. The 0.6B and 1.7B models in the table were made only to measure speed, so their text is
  meaningless. What we *did* verify is that LittleMore computes the same thing as the reference program.
- **Only one test computer.** Speed and power on other computers will differ. The detailed table and the method are in
  [Measured speed](#measured-speed).

## What has been built

- **A fast engine** based on the well-known llama.cpp program, with new parts that run the squeezed models quickly on a
  graphics card and on the normal processor.
- **A converter** that turns a downloaded LittleBit model into a file LittleMore can open, and checks every layer on the
  way so a broken conversion cannot go unnoticed.
- **A library for programmers** (written in C, so almost any language can use it) for loading a model, chatting,
  streaming the answer word by word, and more.
- **A Python package and a command line**, for people who just want to type a command and chat.
- **A large set of automated tests**, including "try to break it" tests with damaged model files, extreme settings, and
  memory running out. They found and fixed real bugs; see [docs/development.md](docs/development.md).

## What is supported today

| | Status |
|---|---|
| NVIDIA graphics cards | Works; tested on one laptop GPU |
| Normal processors (Intel/AMD, 64-bit) | Works; tested on one processor |
| Windows 11 | Works and tested |
| Model family Qwen3 | Tested with real model weights |
| Model families Llama, Gemma 3, Qwen3.5 | Built and tested on tiny practice models, not yet on real weights |
| Linux | Builds, but **not yet tested** on a real machine |
| AMD and Intel graphics cards (Vulkan) | Detected, but they do not make anything faster yet |
| Apple Macs, iPhones, Android phones, other ARM chips | **Not yet** |

## What is left to build

- **More kinds of hardware:** speed-ups for AMD and Intel graphics cards, for phones and other ARM chips (Android), and
  for Apple devices (Mac, iPhone, iPad).
- **Real-world testing:** run it on Linux, on Macs, on other graphics cards and processors, and on GitHub's automatic
  build system. None of this has been possible yet.
- **Better models:** LittleBit models need training, which this project does not do. As better ones are published, they
  should work with LittleMore.
- **Real weights for more families:** confirm Llama, Gemma 3 and Qwen3.5 on real models, not only tiny practice ones.
- **A fair head-to-head** with the usual 4-bit version of the same model, including a proper power measurement.
- **Publishing:** an easy `pip install`, once the packaging and test runs are done.

## How you can help take it further

If you are a developer and want to push this forward, the most useful things are:

1. **Try it on your hardware and report numbers.** A different graphics card, a Linux machine or a Mac gives us data we
   cannot get alone. Even "it did not work, here is the error" helps.
2. **Add support for another kind of chip.** The biggest gaps are AMD and Intel graphics cards (Vulkan), phones (ARM and
   Android) and Apple devices (Metal). The existing NVIDIA and processor code is a worked example to follow.
3. **Train or publish better models.** Quality is the weakest point, and it depends on better LittleBit models.
4. **Test more model families with real weights** (Llama, Gemma 3, Qwen3.5) and tell us what breaks.
5. **Measure power.** A proper electricity comparison on real hardware would turn our estimate into a fact.
6. **Improve the guides.** Clearer instructions, other languages and example apps lower the barrier for newcomers.
7. **Break it.** Try odd inputs and odd settings. Every bug found makes it safer for everyone.

---

# Part 2: For developers

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

All numbers are from files committed in this repository, measured with `llama-bench` on **one test laptop** (RTX 4050
Laptop GPU 6 GB, i7-13620H, Windows 11, CUDA 12.9); they are not guarantees for other hardware.

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

Memory and energy: the model-file sizes (about 3.0 GB for the community 8B GGUF, 5.0 GB for the Q4_K_M GGUF of the same
model) are exact. The GPU memory seen with `nvidia-smi` while generating (about 2.7 GB vs about 5.3 GB) was measured
once and is not a repeated benchmark. The electricity figure in Part 1 is an estimate; no power measurement has been
made yet.

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
- Measured numbers come from one laptop; other GPUs, drivers and CPUs will differ.
- Phase 1 has not yet had a public CI run; the CUDA compile-only jobs were written without being able to run GitHub
  Actions locally.

## Licence

LittleMore's code is [MIT](LICENSE). The engine is llama.cpp/ggml (MIT). Models have their own licences, currently
non-commercial. Details: [docs/licensing.md](docs/licensing.md).
