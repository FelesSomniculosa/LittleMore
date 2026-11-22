# Environment variables and build knobs

Every environment variable, compile-time constant and backend feature flag that LittleMore defines or reads, in one
place. Variables that belong to llama.cpp / ggml itself (for example `GGML_CPU_DISABLE_FUSION`) are not listed unless a
LittleMore test sets them. How to build and test: [development.md](development.md); what the engine patch is:
[engine-changes.md](engine-changes.md).

## Runtime environment variables

| Name | Read by | Values | What it does | When to use it |
|---|---|---|---|---|
| `GGML_CUDA_LITTLEBIT_F32_PREFILL` | the CUDA backend (`ggml-cuda/littlebit-bp.cu`, `ggml_cuda_littlebit_bp_device`), once per process, on the first call | unset or `0`: default; any other integer (`1`): force the float32 path (`atoi` of the value is non-zero) | Disables the tensor-core bit-plane prefill path, so batches above 8 tokens take the float32 GEMM prefill path (and its chunked-decode bands) even on a GPU of compute capability 8.0+. The setting is not re-read while the process runs. | Testing and comparing the two prefill paths: `test-littlebit-ref` and `test-backend-ops -o LITTLEBIT -b CUDA0` a second time (ctest does this for `test-littlebit-ref` as `test-littlebit-ref-f32-prefill`), and speed comparisons through the `"env"` field of a bench plan (`bench/plans/phase1_13b.json`). Not needed in normal use. |
| `LB_BENCH_SKIP_F16` | `engine/tools/littlebit-bench/prefill_bench.cpp` | `1` | Times only the `GGML_OP_LITTLEBIT` op and skips the F16 `MUL_MAT` comparison. | Measuring chunk-limit crossovers or op-level changes where the F16 numbers are not needed (`tools/dev/chunk_crossover_study.sh` does this). |
| `LITTLEMORE_PPL_CHUNKS` | `tests/e2e/test_perplexity_parity.py` | positive integer (default: the test's `DEFAULT_CHUNKS`, 32) | Number of 512-token WikiText-2 chunks scored by the perplexity parity test. | A quicker run with fewer chunks, or a tighter estimate with more. |
| `GGML_SCHED_DEBUG` | set to `2` by `tests/engine/test_fused_graph_qwen3_06b.py` on the child process (a ggml variable, not a LittleMore one) | `2` | Makes the ggml scheduler print which backend every graph node runs on. | Used by the placement guard (every `LITTLEBIT` node must be on CUDA0 at `-ngl 99`); set it yourself to see where nodes run. |

The benchmark harness passes `"env": {"NAME": "value"}` of a plan entry to llama-bench and records it in the results
(see [bench/README.md](../bench/README.md)); `bench/run_bench.py` and the smoke tests also put the CUDA 12.9 `bin`
directory first on `PATH` for the programs they start, so the engine's CUDA DLLs are found.

## Compile-time knobs

| Name | Where | Default | What it does | When to change it |
|---|---|---|---|---|
| `LB_BP_BITS` | C++ preprocessor constant in `engine/ggml/src/ggml-cuda/littlebit-bp.cu` (`-DLB_BP_BITS=<n>`, for example through `-DCMAKE_CUDA_FLAGS=-DLB_BP_BITS=24` at configure time) | 28; allowed 15 to 28 (a `static_assert` rejects other values) | Number of bits of the fixed-point activations in the tensor-core prefill path. Fewer bits mean fewer tensor-core instructions and a faster prefill, at the cost of accuracy. | Only for experiments. The measured trade-off (24 bits: +5 % to +12 % prefill speed but about 4x the op error; 20 bits: fails the op-level tolerance `LB_BP_TOL`) is in the table "Options that trade accuracy" in [engine-changes.md](engine-changes.md). The default is the only exact setting. |
| `GGML_CUDA`, `CMAKE_CUDA_ARCHITECTURES`, `GGML_VULKAN`, `GGML_BACKEND_DL`, `GGML_CPU_ALL_VARIANTS` | CMake options of the engine build, as in [development.md](development.md) section 2 | see that command | Select the backends (CUDA, Vulkan), the GPU architecture (`89` for the RTX 4050 Laptop GPU of this project), loadable backends and one CPU backend per instruction set (the AVX2 LittleBit kernel is compiled into the AVX2 variants only). | When building for another GPU or without a backend. The tensor-core prefill path needs an architecture of compute capability 8.0 or higher in the build; older ones fall back to the float32 path. |

## Backend feature flags (read-only, reported by the backends)

These are not set by the user; the engine reports them in the backend's feature list (`ggml_backend_get_features`) and tests read them to decide what a run covers.

| Name | Reported by | Meaning |
|---|---|---|
| `LITTLEBIT_AVX2` | the CPU backend variants that include the AVX2 kernel of `GGML_OP_LITTLEBIT` | `GGML_OP_LITTLEBIT` runs on its AVX2 kernel in this CPU variant (unless the backend is set to its reference implementations); `test-littlebit-ref` and `test-backend-ops` then compare it with the scalar reference kernel, and `cpu_bench` reports it. |
| `LITTLEBIT_BP_PREFILL` | the CUDA backend, when every CUDA device has compute capability 8.0+ and `GGML_CUDA_LITTLEBIT_F32_PREFILL` is not set | The tensor-core bit-plane prefill path is active, so `test-littlebit-ref` tests it against its tighter bound (`LB_BP_TOL`) and covers the dispatch boundary at 9 tokens instead of the float32 chunk bands. |
