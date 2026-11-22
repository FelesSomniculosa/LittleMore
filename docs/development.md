# Development guide

How to build, test and run LittleMore end to end. All environment variables and build knobs are listed in
[environment-variables.md](environment-variables.md). LittleMore runs LittleBit
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

## 2b. Build and test the C API

The C API (`include/littlemore.h`, the `littlemore` shared library) has its own top-level CMake build, which
builds the engine again with `add_subdirectory` into `build/` and runs its C test with CTest
(`test-api-cpu`, `test-api-cuda`). Commands, design and measurements: [c-api.md](c-api.md).

The Python SDK and the `littlemore` command line (`python/`, a `ctypes` layer over that library) are described in
[python/README.md](../python/README.md): usage, building a wheel, and how the CUDA runtime DLLs are found. Its tests
are `tests/python/test_sdk.py` (the fast ones run with the default `pytest`; the ones that load the tiny model carry
the `smoke` marker, like the other end-to-end tests):

```
.venv\Scripts\python.exe -m pytest tests/python -q            # fast part
.venv\Scripts\python.exe -m pytest tests/python -m smoke -q     # with the tiny model (needs build/bin)
```

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
| `.cache/gguf/` | the LittleBit B1 GGUF (made by `tools/convert/lb_to_gguf.py`) and the Q1_0 slow-path baseline derived from it |
| `.cache/logits/` | logits saved by `llama-debug` |

Each generated file has a `.key` stamp; it is rebuilt automatically when the
tool sources that produced it change.

## 5. Tools in `tools/dev`

| File | Purpose |
|---|---|
| `check_env.py` | checks that all build tools and a CUDA-enabled PyTorch are installed (exit 0 = ready) |
| `reference.py` | plain-PyTorch LittleBit model: loads a checkpoint and runs it; the ground truth for all other tests |
| `make_lb2_init.py` | makes an init-only LittleBit-2 checkpoint from any Hugging Face causal language model (no training) |
| `dense_floor.py` | runs the dense Qwen3-0.6B and the LittleBit file through llama.cpp (CPU / GPU, default / exact) against their fp32 references and prints the noise-floor table of docs/engine-changes.md |
| `emulate_q8_slow_path.py` | the reference plus the int8 activation noise that llama.cpp adds, so engine and reference can be compared fairly |
| `dequant_gguf_to_hf.py` | rebuilds a Hugging Face directory from a quantized Qwen3 GGUF (dequantized weights): a "proxy" base model with the right shapes for SPEED measurements when the original weights are not downloaded (not for accuracy) |
| `prefill_guard_study.py` / `.sh` | measures the model-level GPU error of the Qwen3-0.6B LittleBit file on 5 long prompts, on both CUDA paths and on deliberately broken builds (the evidence for the coarse prefill guards; see docs/engine-changes.md) |

Example, making a checkpoint by hand:

```powershell
.venv\Scripts\python.exe tools/dev/make_lb2_init.py .cache/models/Qwen3-0.6B out/ckpt --bpw 0.55
```

## 6. Converter in `tools/convert`

| File | Purpose |
|---|---|
| `lb_to_gguf.py` | converts a Samsung-format LittleBit checkpoint directory to a GGUF whose sign matrices are `B1` (exactly 1 bit per sign); every linear is validated after writing |
| `validate.py` | re-decodes the written GGUF tensors and compares each linear with `reference.py` |

```powershell
.venv\Scripts\python.exe -m tools.convert.lb_to_gguf .cache/ckpt/qwen3-0.6b-lb2-0.55 -o out/model.gguf
.venv\Scripts\python.exe -m tools.convert.lb_to_gguf .cache/ckpt/qwen3-0.6b-lb2-0.55 -o out/model-q8.gguf --head-type q8_0
.venv\Scripts\python.exe -m pytest tests/convert -q
```

`--head-type f16|q8_0|q6_k` (default `f16`) sets the type of the token embedding and the output head, the largest
tensors of a small model; Q8_0 / Q6_K shrink them 1.9x / 2.4x and make generation faster (measured in
[bench/results/phase1.md](../bench/results/phase1.md), with the perplexity change). Everything else in the file is
unchanged; correctness is measured with the F16 head. The Q6_K quantizer is a port of ggml's C reference quantizer into
the engine's gguf-py, bit-identical to it (`tests/convert/test_head_type.py`). With validation on, the converter also
dequantizes the written head and compares it with the checkpoint's (relative RMS error at most 1.1e-2 for Q8_0, 3.8e-2
for Q6_K; measured 5.35e-3 / 1.77e-2 on Qwen3-0.6B), so a wrong or corrupted head fails the conversion.

Input widths (hidden size, intermediate size) must be multiples of 128, because
a `B1` row is whole 128-sign blocks; the converter refuses anything else. The
rank may be anything (it is padded to a multiple of 128 with zero scales, which
is exact). A checkpoint without `littlebit_config.json` is refused with an
explanation; pass `--assume-no-littlebit-config` to skip only the residual
cross-check against that file. Gemma 3 is tested on the generic path only
(identity permutations); Llama and Qwen3.5 exercise real permutations.

These GGUFs run in the engine (fused `ggml_littlebit` op: CPU kernel, and CUDA kernels for
any batch size; with `-ngl 99` on CUDA the LittleBit weights live on the GPU, on a Vulkan
device they stay in host memory and the op runs on the CPU). Linears with a residual path
also get `lb_s_pkr`, the three residual scale vectors packed in the order the engine's op
reads them (see [engine-changes.md](engine-changes.md)). The converter
tests build tiny random models offline; the Qwen families need the cached
Qwen3-0.6B tokenizer (they skip without it), Llama and Gemma 3 need nothing.

## 7. Engine tests

```powershell
.venv\Scripts\python.exe -m pytest tests/engine -m engine_run -q -s     # fused graph per model family + Qwen3-0.6B
.venv\Scripts\python.exe -m pytest tests/smoke -m smoke -q -W error     # slow-path baseline (Q1_0 chain)
```

`engine_run` tests build tiny random models of Qwen3, Llama, Gemma 3 and Qwen3.5, convert them, run
`llama-debug` and compare the last-position logits with `reference.py`; the Qwen3-0.6B file does the same on the real
init-only checkpoint at `-ngl 0` and `-ngl 99`. The engine's own tests: from `engine/build`,
`ctest -R "test-b1|test-gguf|test-alloc|test-littlebit"` (needs the MSVC environment on `PATH`).
On a GPU with the tensor-core prefill path (compute capability 8.0+, the CUDA backend reports `LITTLEBIT_BP_PREFILL`),
run `bin\test-littlebit-ref.exe` and `bin\test-backend-ops.exe test -o LITTLEBIT -b CUDA0` a second time with
`GGML_CUDA_LITTLEBIT_F32_PREFILL=1` set (ctest does this for test-littlebit-ref: `test-littlebit-ref-f32-prefill`):
that forces the float32 GEMM prefill path and its chunk bands, which the
default run no longer reaches on such a GPU.
`test-littlebit-ref` also compares the CPU's AVX2 kernel of the fused op with its scalar reference kernel (they must be
bit-identical when neither kernel is FP-contracted, i.e. no a*b + c fused into an FMA by the compiler; asserted for MSVC
builds), and `test-backend-ops test -o LITTLEBIT -b CPU` does the same through the test harness; see "CPU: AVX2
kernel" in [engine-changes.md](engine-changes.md#cpu-avx2-kernel-of-ggml_op_littlebit) for the CPU speed measurements and
`engine/tools/littlebit-bench/cpu_bench.cpp` for the benchmark program.

## 7b. Perplexity parity (engine vs reference)

```powershell
.venv\Scripts\python.exe -m pytest tests/e2e -m smoke -v -s          # about 3.5 minutes; LITTLEMORE_PPL_CHUNKS=N sets the chunk count
```

`tests/e2e/test_perplexity_parity.py` scores the first 32 chunks of 512 tokens (16384 tokens, 8160 scored) of the
WikiText-2 raw test split with `tools/dev/reference.py` (CPU, fp32, factorized) and with `llama-perplexity` on the
init-only Qwen3-0.6B LittleBit 0.55 GGUF, and requires the two perplexities to agree within 1 %: on CUDA, on the CPU
(AVX2 kernel; `-ngl 0 -dev none`, because plain `-ngl 0` still offloads prefill matmuls to the GPU), and, within 0.1 %,
on the CPU with the f16 roundings of attention removed (`-fa off -ctk f32 -ctv f32`). A second test compares the whole
token-id stream of the engine's tokenizer with the Hugging Face one, and a mutation test shows that a 2 % change of one
scale vector, or a one-token shift of the reference's input, breaks the bound.

The dataset is pinned: `Salesforce/wikitext`, config `wikitext-2-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, file `test-00000-of-00001.parquet` (0.7 MB, sha256 `5f1bea06...ab246d91`);
`.cache/data/wiki.test.raw` is its `text` column joined with `""` (1287656 bytes, sha256 `bbf94c53...964574a`, rebuilt
by the test from the parquet if missing). The test skips with a reason when the dataset, base model or binaries are absent.

What it proves: the whole engine (converter, CPU and CUDA fused op, graph, attention, head) computes the same function as
the independent PyTorch reference over thousands of tokens. Limits: the weights are untrained (init only), so the
perplexity is in the tens of millions and says nothing about quality; parity is the point. Only a subset of the test
split is scored (the first 32 of 583 chunks), and a single sequence length (512). The 1 % bound catches roughly a 0.3 %
error on one scale vector (a 2 % error moved the perplexity by 6.3 %, measured on 2 chunks of the CUDA run).

What the per-chunk numbers show (measured, 64 chunks, CUDA; `std` = spread of the per-chunk NLL difference to the
reference, in nats):

| CUDA configuration | std | mean gap, 32 chunks | mean gap, 64 chunks |
|---|---|---|---|
| default (tensor-core prefill, f16 attention) | 0.0156 | -4.5e-4 | +1.27e-3 |
| `GGML_CUDA_LITTLEBIT_F32_PREFILL=1` | 0.0121 | -6.4e-4 | +1.4e-4 |
| `-fa off -ctk f32 -ctv f32` | 0.0101 | +8.7e-4 | -2.7e-4 |
| both | 0.0099 | -1.1e-3 | -7.9e-4 |

On the CPU the same per-chunk gap is 5e-5 with exact attention settings, and the dense Qwen3-0.6B F32 GGUF on CUDA
differs from its HF reference by only 1.2e-4 nats per chunk (CPU: 1.0e-4). So the per-chunk scatter (about 1 %, up to a
few percent of the perplexity of single chunks) is specific to the LittleBit model on CUDA; neither the f16 attention nor
the tensor-core prefill path, alone or together, removes it (they reduce it by up to a third). Its cause is not decided
(hypotheses: rounding in the CUDA LittleBit kernels or the float32 GEMM path, amplified by the very flat output
distribution of an untrained model; not tested). The standard error of the mean per-chunk gap is 0.0156/sqrt(64) = 2e-3
over 64 chunks (2.8e-3 over 32), so the measured mean gaps of the default CUDA run (-4.5e-4 over 32 chunks, +1.27e-3 over
64 chunks) are both within one standard error of zero, i.e. consistent with zero; no trend with the chunk count is
claimed. On the CPU the default-settings perplexity gap is 1.16e-3 over 32 chunks (3.4e-3 over the first 8 chunks); it
is the f16 rounding of Q/K/V in attention: with `-fa off -ctk f32 -ctv f32` it is 1.8e-5 over 32 chunks (2.7e-5 over 8).

Gotcha (`-ngl 0` is not a CPU run on a CUDA build): llama.cpp offloads the matmuls of every batch of 32 tokens or
more to the GPU ("op offload") even with `-ngl 0`. A "CPU" run must pass `-dev none` (or `--no-op-offload`). The test
asserts it from the `-v` log (no `CUDA0` line, a `CPU compute buffer` line); `tests/smoke/_assets.engine_logits` adds
`-dev none` itself for `ngl == 0`; `bench/run_bench.py` already does for its `cpu` backend. Found by the perplexity test:
at 64 tokens the logits differ from the reference by 2.9e-2 rms with offload and 8.6e-5 without; the tiny Llama and
Gemma 3 models of `tests/engine/test_fused_graph_families.py` (37-token prompts) differed by up to 7e-4 (bitwise
different logits) between the two.

Test selection: a plain `pytest` also runs the smoke-marked tests, including this file (about 3.3 minutes extra when the
assets exist; they skip otherwise). `pytest -m "not smoke"` is the fast subset.

## 8. Speed benchmarks and the speed gate

```powershell
.venv\Scripts\python.exe -m pytest tests/bench -q                                   # the gate rule (pure logic)
.venv\Scripts\python.exe -m bench.run_bench --plan bench/plans/phase1.json --rounds 3 --out bench/results/phase1.json
```

`bench/run_bench.py` wraps `llama-bench` (warm-up, repetitions, interleaved rounds, GPU memory and clock); the gate is
"LittleBit 0.55 bpw generates faster than bf16 on CUDA for Qwen3-0.6B and Qwen3-1.7B" (Phase 1 status: PASS on
the original weights of both models). See [bench/README.md](../bench/README.md)
and the results in [bench/results/phase1.md](../bench/results/phase1.md). Prompt processing with the tensor-core
prefill path (`bench/plans/phase1_13b.json` measures LittleBit with and without it, `"env"` in a plan entry):
[bench/results/phase1_13b.md](../bench/results/phase1_13b.md).

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
- **init-only**: a checkpoint made with no QAT at all (only the SVD / ITQ / SVID
  initialisation), so it is exact for testing but produces gibberish text.
- **Samsung-format checkpoint**: the file layout of the original LittleBit
  release: bit-packed sign matrices (`*_packed`, where a 0 bit means +1) plus
  scale vectors. This project converts it to GGUF.
- **B1**: the engine's own ggml storage type for a sign matrix: 128 signs in 16
  bytes, exactly 1 bit per sign and no scale (a 1 bit means +1).
- **Q1_0**: an upstream ggml type that also stores 1 bit per sign but adds a
  16-bit scale per 128 signs (1.125 bits per weight).
- **Q8_0**: a ggml type with 8-bit integers and one scale per block of 32 values;
  llama.cpp also uses it to quantize activations before an integer dot product.
- **slow path**: the older way of running a LittleBit linear: a chain of five
  ordinary ops per path (multiply by a scale, matrix-multiply, ...) on Q1_0
  weights. It rounds the activations to Q8_0, so it is noisy, and it is kept as a
  baseline to compare the fused op with.
- **fused op**: `GGML_OP_LITTLEBIT`, one op that computes a whole LittleBit
  linear from the B1 signs and the scales, in float32, with no activation
  rounding.
- **residual path**: the second, smaller pair of sign matrices of LittleBit-2,
  which approximates what the first pair got wrong; its output is added to the
  first one.
- **latent unit**: one of the `r` inner rows between the two sign matrices (an
  index of the rank dimension); the scale vector `s_mid` has one entry per unit.
- **`s_mid`**: the scale applied between the two sign matrices. The checkpoint
  stores two scales there (`v1` and `u2`); the converter multiplies them into one
  vector (`s_mid = v1 * u2`) because nothing sits between them.
- **backend**: a part of ggml that runs ops on one kind of hardware (CPU, CUDA,
  Vulkan).
- **`supports_op`**: the function through which each backend tells the scheduler
  whether it can run a given op; the scheduler sends refused ops to another backend.
- **`-ngl` / GPU offload**: the llama.cpp option `-ngl N` puts N layers (99 =
  all) on the GPU. LittleBit weights go with their layer when its device can run the
  fused op (CUDA); otherwise (Vulkan) they stay on the CPU.
- **flash attention** (`-fa`): a fused attention implementation; it keeps
  intermediate values in f16, so switching it off (`-fa off`) removes that
  rounding.
- **KV cache** (f16 KV): the stored keys and values of earlier tokens; by default
  stored as 16-bit floats, `-ctk f32 -ctv f32` stores them as 32-bit.
- **`llama-debug`**: an engine program that loads a model, runs a prompt and
  prints/saves the logits; the tests use it to get the engine's numbers.
- **ctest**: CMake's test runner, used for the engine's own C++ tests.
- **MSVC**: the Microsoft C++ compiler used to build the engine on Windows.
- **NMSE** (normalised mean squared error): the squared error divided by the
  squared size of the reference; a scale-free measure of how far two outputs are.
- **DeltaNet**: a linear-attention (recurrent) layer type used by Qwen3.5 next to
  ordinary attention layers.
- **sliding-window attention**: attention in which a token only looks at the
  most recent N tokens; Gemma 3 uses it in 5 of every 6 layers.
