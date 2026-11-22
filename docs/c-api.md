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
`GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON` (x86-64), engine tools/tests/examples/`common` off. CMake 3.22
or newer is required (the tests use the `ENVIRONMENT_MODIFICATION` test property).

Install: `cmake --install build --prefix <dir>` puts `littlemore` (and `llama`, `ggml`, `ggml-base`, every
backend plugin) in `<dir>/bin`, the header in `<dir>/include`, the import library in `<dir>/lib` and a CMake
package in `<dir>/lib/cmake/littlemore`, so another project can use
`find_package(littlemore)` and link `littlemore::littlemore`. The library is installed next to the plugins on
every platform because it loads them from its own directory (only the Windows layout is verified). The shared
library's SOVERSION is `LM_API_VERSION`.

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
- **Platforms**: 64-bit only. The structs have no hidden padding there (explicit `reserved` fields, checked by
  compile-time asserts in the library and in the C test), so `sizeof` is the end of the last field.
- **Option structs** start with `struct_size`; fill them with the `lm_*_default()` functions. Append-only rule:
  fields are only added at the end; `reserved` fields must be 0. An older caller's `struct_size` must end on a
  field boundary (it is read up to there, defaults for the rest); a newer caller's larger struct is refused
  (`LM_ERR_UNSUPPORTED`) only if a byte after the last field this library knows is non-zero. `lm_message` is
  used in arrays, so it has no `struct_size` and never changes.
- **FFI**: `lm_status` is a 32-bit C enum (bind it as `int32`); every function and the callback type use
  `LM_CALL` (`__cdecl` on Windows, the default elsewhere).
- **Errors**: every fallible call returns an `lm_status` (0 = `LM_OK`). After a failure `lm_last_error()`
  (thread-local) holds a message that starts with the failing function's name and, where the engine logged
  why, ends with the engine's own error lines.
- **Threads**: a model can be shared by many sessions and threads; calls on one session are serialised by a
  lock inside it; different sessions run in parallel. `lm_model_free` refuses (`LM_ERR_BUSY`) while a session of
  the model is alive. `lm_session_options.n_threads` is at most `LM_MAX_THREADS` (512, the engine's limit: the CPU
  backend starts one OS thread per requested thread for every batch; before this limit 5000 started 5000 threads per
  token and `INT32_MAX` crashed the process) and `n_ctx` at most `INT32_MAX`; larger values are `LM_ERR_INVALID_ARG`.
- **Freed handles**: a model or session that was freed must not be passed again (double free, use after free). As
  with `free()`, the library cannot detect it (measured: a second `lm_model_free` of the same pointer is an access
  violation). The Python SDK never does it: `close()` is idempotent and a closed model raises `LittleMoreError`.
- **Inside a token callback** (it runs on the generating thread, inside the call): `lm_tokenize`,
  `lm_detokenize`, `lm_model_get_info`, `lm_backends_info` and calls on other sessions are allowed; `lm_chat`,
  `lm_generate` and `lm_session_free` on the generating session return `LM_ERR_REENTRANT` and change nothing
  (the outer generation continues); `lm_model_free` returns `LM_ERR_BUSY`.
- **Sessions are stateless between calls**: each `lm_chat` / `lm_generate` starts from an empty KV cache and
  reads its whole prompt (for multi-turn chat pass the full history). Temperature `<= 0` is greedy and
  deterministic; a fixed `seed` makes sampling reproducible. The default seed is 0, so with default settings
  every call gives the same text even at temperature 0.8; pass `LM_SEED_RANDOM` (or your own seed per call)
  for varied answers. Prompt plus generated tokens never exceed the session's `n_ctx` (a generation that
  reaches it stops with `LM_OK`).
- **Chat templates**: `lm_chat` uses the template stored in the GGUF through `llama_chat_apply_template`,
  which recognises llama.cpp's built-in template families (ChatML when the file has none). The formatted
  conversation is tokenized with special-token parsing on, as llama.cpp's own chat programs do, so text such as
  `<|im_end|>` inside a message becomes the real control token (it can end a turn early or forge one): escape
  it in untrusted input. Tokenizing message contents separately with special tokens off would prevent this but
  can change the tokens at the boundaries between template text and content, so it is not done.
  Known difference from a model's own Jinja template (measured on Qwen3, `tests/robustness/test_d_misuse.py`): for
  one-turn and plain multi-turn conversations the formatted prompt is token-identical to the Hugging Face
  `apply_chat_template` rendering, but Qwen3's template removes the `<think>...</think>` block of earlier assistant
  turns and llama.cpp's built-in ChatML formatting keeps it, so a history that contains reasoning gives the model a
  different prompt. Strip earlier reasoning from the history yourself if that matters.
- **Tokenizer and Unicode normalization**: the Qwen3 tokenizer (`tokenizer.json`) normalizes text to NFC before
  tokenizing; llama.cpp's tokenizer does not. Text in decomposed form (NFD, e.g. "e" + U+0301 instead of "é", or Korean
  as conjoining jamo) therefore gets more and different tokens than the Hugging Face tokenizer gives it (measured:
  16 instead of 8 tokens for "café naïve Ångström" in NFD); NFC text is token-identical. `lm_tokenize` /
  `lm_detokenize` keep their byte-exact round trip either way. Normalize input to NFC (Python:
  `unicodedata.normalize("NFC", text)`) when the tokens must match the reference tokenizer.
- **Streaming**: the callback gets the generated bytes unchanged. The only processing is that an unfinished
  UTF-8 sequence at the end of a piece is held back until the next token completes it, so whenever the model
  produces valid UTF-8 each piece is valid UTF-8 on its own; invalid sequences pass through as they are, and
  anything still held back at the end is delivered. Returning non-zero stops.

## Auto-fit of GPU layers (`n_gpu_layers = -1`)

Before loading, the library reads the GGUF tensor table and estimates, per block, its weight bytes plus its
f16 KV cache at `fit_n_ctx` tokens (default 4096, capped at the training context); the output head is the
"extra layer" llama.cpp offloads first. It keeps a reserve per GPU for compute buffers and runtime overhead
(256 MiB + the logits of a 512-token batch + the attention scores of a 512-token batch without flash
attention; this assumes the session's default batch, `n_batch` = `n_ubatch` = 512), then offloads as many
layers as fit into the free memory the devices report, summed over all of them (the reserve is counted once
per device; the engine then splits the layers over the devices by their free memory), or into
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

- **Python hosts**: see "Python / ctypes on Windows" below (CUDA runtime DLLs and `os.add_dll_directory`).
- **0 GPU layers still uses the GPU (intended), `devices = "none"` is the CPU-only switch.** With a GPU in the
  device list, the engine runs large prompt batches on the GPU even when no layer is offloaded (explicit
  `n_gpu_layers = 0`, or an auto-fit that fits nothing), as llama.cpp does: it costs GPU memory and makes
  prompts much faster. `devices = "none"` uses no GPU: the model allocates nothing on it (no weights, cache or
  compute buffers). Loading the backends still creates a driver context on the GPU, about 73 MiB in `nvidia-smi`
  (measured on the RTX 4050 Laptop with no model loaded at all). The CPU-mode CTest runs assert
  this from the engine's allocation log (no GPU buffer, no node on a GPU), and a control run in CUDA mode shows
  the same check detects GPU use. Measured on one RTX 4050 Laptop GPU (6 GiB), 0 GPU layers, GPU kept vs
  `"none"` (3 interleaved rounds, median of 15; llama-bench with 4 threads and batch 512, first-token time
  through the C API):

  | Model | GPU memory | prompt 512 tok/s | first token, 512-token prompt | generation tok/s |
  |---|---|---|---|---|
  | Qwen3-0.6B LittleBit 0.55 | 723 MiB vs 0 | 5936 vs 354 | 120 vs 1449 ms | 87.7 vs 85.7 |
  | Qwen3-1.7B LittleBit 0.55 | 1033 MiB vs 0 | 2824 vs 123 | 202 vs 4192 ms | 34.7 vs 34.2 |
  | Qwen3-0.6B f16 | 749 MiB vs 0 | 3456 vs 195 | 179 vs 2630 ms | 41.9 vs 42.3 |
  | Qwen3-1.7B f16 | 1053 MiB vs 0 | 1378 vs 65 | 443 vs 8046 ms | 15.2 vs 15.1 |

  Shorter prompts gain less (128 tokens: 3.6x to 10.9x; 32 tokens: 1.7x to 3.2x for LittleBit, none for f16).
  The header describes this in one place, at `lm_model_options.devices` ("GPU use at 0 GPU layers").
- **Windows (WDDM) does not report other processes' GPU memory**: while another process held 3.6 GiB of the
  6 GiB (allocated and written), `cudaMemGetInfo` in a new process still reported 5.0 GiB free, the fit
  offloaded all of Qwen3-1.7B f16, and generation succeeded with the GPU memory oversubscribed (the driver
  pages it to system memory). Set `vram_budget` when other programs use the GPU. Re-measured with the community
  Qwen3-8B model while another process held 3.0, 5.0 and 5.8 GiB (`tests/robustness/test_e_resources.py`): the fit
  still saw 5080 MiB, put all 37 layers on the GPU and generation worked every time (oversubscribed, no error); with
  `vram_budget` set to the free memory nvidia-smi showed, the fit chose 23 layers (3.0 GiB held) or 0 layers.

## Python / ctypes on Windows

Python 3.8 and later do not search `PATH` for the DLLs that a loaded DLL depends on. Loaded with `ctypes` from
`python.exe`, `littlemore.dll` finds all plugins in its own directory, but the CUDA plugin cannot load its CUDA
runtime DLLs even with the CUDA 12.9 `bin` directory first on `PATH` (measured: no CUDA0 device, only CPU and
Vulkan). Call `os.add_dll_directory(<directory with the CUDA runtime DLLs>)` (the toolkit's `bin`, or wherever a
package bundles them) before loading the library; then CUDA0 appears. The Python SDK ([python/README.md](../python/README.md))
does this for you.

## Tests

`tests/c/test_api.c` (compiled as C) runs three times under CTest: `test-api-cpu` (`devices = "none"`),
`test-api-cuda` (`devices = "CUDA0"`; skipped with exit code 77 when there is no CUDA device) and
`test-api-plugin-dir` (the CPU run again, from a copy of the executable in `build/elsewhere`, with that
directory as the working directory: no plugin is next to the executable or in the current directory, so it
passes only if the library loads the plugins from its own directory). `test-utf8` (`tests/c/test_utf8.c`) is a
unit test of the streaming boundary rule (`src/utf8_boundary.h`). `test-api-null` runs
`tests/robustness/c/null_harness.c`, generated from this header by `tests/robustness/gen_null_harness.py`: NULL in every
pointer parameter of every exported function must give `LM_ERR_INVALID_ARG` (or `LM_OK` where the header allows NULL),
never a crash (79 checks). `test-example-chat` builds and runs the header's minimal example
(`tests/robustness/c/example_chat.c`) on the tiny model. A fixture (`littlemore-tiny-gguf`) first
builds the tiny model with `tests/c/make_tiny_gguf.py` (a random 2-layer Llama with a 300-piece vocabulary and
the ChatML template, converted to a LittleBit B1 GGUF, 0.7 MB, cached in `.cache/gguf/`). The API test checks
the struct layouts at compile time, NULL arguments, struct sizes (field boundaries, unknown and reserved
fields), the error messages (each must name the failing call) and that they are per thread, backend listing,
the auto-fit (all layers on CUDA, 0 with a 1-byte budget, head + one block with a budget for exactly that),
an older smaller `lm_model_info`, tokenize/detokenize (including the `special` flag), streaming, template
application, stopping, determinism, context overflow and a generation that fills the context, re-entrant
calls from a token callback, two threads on one session, and memory growth over 20 load/chat/free cycles.
Each API test has a 120-second timeout, so a deadlock fails instead of hanging. The CPU-mode runs also print the engine's info log
(`LITTLEMORE_LOG=info`) and fail if it shows any GPU buffer or any node scheduled on a GPU (so `"none"` really
touches no GPU); `test-api-gpu-log-control` (CUDA mode, same log) must show such lines, proving the check works.

Mutation checks (each made the test fail, then was reverted): not freeing the model (private memory grew
17 MiB on the CPU, 7.5 MiB on CUDA, limit 4 MiB) or the session (more than 480 MiB); not storing error
messages (26 failed checks); an auto-fit that ignores the budget (3 failed checks); `lm_chat` without the
template (2 failed checks); no session lock (decode errors, garbled text or a crash).

Logging: the library prints engine errors to stderr; `LITTLEMORE_LOG` changes that (see
[environment-variables.md](environment-variables.md)).
