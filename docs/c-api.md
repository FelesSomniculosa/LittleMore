# The LittleMore C API (`littlemore.h`)

`include/littlemore.h` is the whole public interface of the `littlemore` shared library: load a GGUF model
(LittleBit or ordinary), open sessions, chat or generate with streaming, tokenize, and list the compute
backends. It is plain C (no C++ or engine types), so it can be called from C, Python `ctypes`, C#, Rust and
others. Every function and struct has a doc comment in the header; this page is the overview.

## Build

From a shell with the MSVC compiler and CUDA 12.9 first on `PATH` (see [development.md](development.md),
section 2), at the repository root:

```
cmake -G Ninja -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_VULKAN=ON ^
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9" ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe" ^
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 6
ctest --test-dir build --output-on-failure
```

The top-level `CMakeLists.txt` adds the engine (`engine/`) with `add_subdirectory`, so this is a second,
independent build of the engine next to `engine/build` (the first full build took 9.4 minutes with `-j 8` on
the 16-thread development laptop, CUDA and Vulkan on). It builds only what the library needs: `llama`, `ggml`, `ggml-base` and the backend plugins; the
engine's tools and tests are still built from `engine/` as before. Everything lands in `build/bin`:
`littlemore.dll`, `llama.dll`, `ggml*.dll`, the plugins (`ggml-cpu-*.dll`, `ggml-cuda.dll`,
`ggml-vulkan.dll`) and `test_api.exe`. Ship them together.

Defaults set by the top-level build (each can be overridden on the command line): `BUILD_SHARED_LIBS=ON`,
`GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON` (x86-64), engine tools/tests/examples/`common` off.

## Backends are plugins

With `GGML_BACKEND_DL` every ggml backend is a separate DLL loaded at run time. At the first call that needs
them (`lm_model_load`, `lm_backends_info`), the library loads every plugin found **in its own directory**
(`ggml_backend_load_all_from_path`), not in the executable's directory, so it works inside `python.exe` or
any other host. The best CPU variant for the running CPU is chosen; the CUDA plugin loads only when its
runtime DLLs and an NVIDIA driver are present. `lm_backends_info()` lists what was loaded, for example:

```
CUDA0: NVIDIA GeForce RTX 4050 Laptop GPU [GPU, free 5.0 GiB of 6.0 GiB]
Vulkan0: NVIDIA GeForce RTX 4050 Laptop GPU [GPU, free 5.0 GiB of 5.8 GiB]
Vulkan1: Intel(R) UHD Graphics [iGPU, free 7.2 GiB of 7.9 GiB]
CPU: 13th Gen Intel(R) Core(TM) i7-13620H [CPU]
```

Devices (`lm_model_options.devices`): `NULL` = every GPU without duplicates (the same rule as llama.cpp: one
physical GPU seen by CUDA and Vulkan is used once, through the first backend, CUDA; an integrated GPU only
when there is no discrete one); `"none"` = CPU only; `"CUDA0"` or `"CUDA0,CUDA1"` = exactly those.

## Usage rules

- **Versions**: `LM_API_VERSION` (header) must equal `lm_api_version()` (library). `lm_version()` is the
  semantic version.
- **Option structs** start with `struct_size`; fill them with the `lm_*_default()` functions. An older
  library refuses a newer caller's struct only if a field it does not know is non-zero (`LM_ERR_UNSUPPORTED`).
- **Errors**: every fallible call returns an `lm_status` (0 = `LM_OK`). After a failure `lm_last_error()`
  (thread-local) holds a message that starts with the failing function's name and, where the engine logged
  why, ends with the engine's own error lines.
- **Threads**: a model can be shared by many sessions and threads; calls on one session are serialised by a
  lock inside it; different sessions run in parallel.
- **Sessions are stateless between calls**: each `lm_chat` / `lm_generate` starts from an empty KV cache and
  reads its whole prompt (for multi-turn chat pass the full history). Temperature `<= 0` is greedy and
  deterministic; a fixed `seed` makes sampling reproducible.
- **Chat templates**: `lm_chat` uses the template stored in the GGUF through `llama_chat_apply_template`,
  which recognises llama.cpp's built-in template families (ChatML when the file has none).
- **Streaming**: the callback gets UTF-8 pieces that never split a character; returning non-zero stops.

## Auto-fit of GPU layers (`n_gpu_layers = -1`)

Before loading, the library reads the GGUF tensor table and estimates, per block, its weight bytes plus its
f16 KV cache at `fit_n_ctx` tokens (default 4096, capped at the training context); the output head is the
"extra layer" llama.cpp offloads first. It keeps a reserve per GPU for compute buffers and runtime overhead
(256 MiB + the logits of a 512-token batch + the attention scores of a 512-token batch without flash
attention), then offloads as many layers as fit into the free memory the devices report, or into
`vram_budget` when that is set. If not even the head fits, the model runs with 0 GPU layers.
`lm_model_get_info()` reports the decision (`n_gpu_layers`, budget, reserve, layer and head estimates).

LittleBit B1 weights are not moved by the fit: it only chooses `n_gpu_layers`, and the engine still places
each B1 linear on the first device of its layer that can run the fused LittleBit op (CUDA; on Vulkan they
stay on the CPU, so the fit's estimate is then conservative). Checked with `GGML_SCHED_DEBUG=2` on CUDA0:
all 126 LITTLEBIT nodes of the tiny test model run on CUDA0.

Estimate against measured use (RTX 4050 Laptop, 6 GiB, CUDA0 only, session `n_ctx` 4096, 601-token prompt;
"used" = free memory before loading minus free memory after generating):

| Model | n_gpu_layers | Estimate | Used |
|---|---|---|---|
| Qwen3-0.6B LittleBit 0.55 (B1, f16 head) | 29 (all) | 1468 MiB | 1114 MiB |
| Qwen3-1.7B LittleBit 0.55 (B1, f16 head) | 29 (all) | 1852 MiB | 1504 MiB |
| Qwen3-0.6B f16 | 29 (all) | 2266 MiB | 1926 MiB |
| Qwen3-0.6B f32 | 29 (all) | 3402 MiB | 3058 MiB |
| Qwen3-1.7B f16 | 29 (all) | 4411 MiB | 4082 MiB |
| Qwen3-1.7B f16, `vram_budget` 2.5 GiB | 12 | 2506 MiB | 2172 MiB |

The estimate is 330 to 350 MiB above the real use in every case (on the safe side).

Known limits, measured:

- **0 GPU layers is not "no GPU"**: as in llama.cpp, the engine still sends large prompt batches to a GPU
  of the model's device list. Measured with a 601-token prompt and 0 layers: 616 MiB (0.6B LittleBit) to
  944 MiB (1.7B f16) of GPU memory. Use `devices = "none"` for a run that touches no GPU.
- **Windows (WDDM) does not report other processes' GPU memory**: while another process held 3.6 GiB of the
  6 GiB (allocated and written), `cudaMemGetInfo` in a new process still reported 5.0 GiB free, the fit
  offloaded all of Qwen3-1.7B f16, and generation succeeded with the GPU memory oversubscribed (the driver
  pages it to system memory). Set `vram_budget` when other programs use the GPU.

## Tests

`tests/c/test_api.c` (compiled as C) runs twice under CTest: `test-api-cpu` (`devices = "none"`) and
`test-api-cuda` (`devices = "CUDA0"`; skipped with exit code 77 when there is no CUDA device). A fixture
(`littlemore-tiny-gguf`) first builds the tiny model with `tests/c/make_tiny_gguf.py` (a random 2-layer
Llama with a 300-piece vocabulary and the ChatML template, converted to a LittleBit B1 GGUF, 0.7 MB, cached
in `.cache/gguf/`). It checks NULL arguments and struct sizes, the error messages, backend listing, the
auto-fit (all layers on CUDA, 0 with a 1-byte budget, head + one block with a budget for exactly that),
tokenize/detokenize, streaming, template application, stopping, determinism, context overflow, two threads
on one session, and memory growth over 20 load/chat/free cycles.

Mutation checks (each made the test fail, then was reverted): not freeing the model (private memory grew
17 MiB on the CPU, 7.5 MiB on CUDA, limit 4 MiB) or the session (more than 480 MiB); not storing error
messages (26 failed checks); an auto-fit that ignores the budget (3 failed checks); `lm_chat` without the
template (2 failed checks); no session lock (decode errors, garbled text or a crash).

Logging: the library prints engine errors to stderr; `LITTLEMORE_LOG` changes that (see
[environment-variables.md](environment-variables.md)).
