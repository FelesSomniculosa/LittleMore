# Engine changes (llama.cpp fork)

LittleMore runs Samsung LittleBit (sub-1-bit) models through a patched llama.cpp.
The engine lives in `engine/`, which is its **own git repository** (a plain
local clone of upstream) and is listed in the outer repo's `.gitignore`, so the
outer repo does not track it. The pin is recorded here instead.

## Pin

| Item | Value |
|---|---|
| Upstream | https://github.com/ggml-org/llama.cpp |
| Upstream tag | `b12606` |
| Upstream commit | `e412c6f468b7b9e0f16f2540244236f8ad68e9ae` |
| Engine branch | `littlemore` (upstream tag + the single commit below) |
| Engine HEAD | `0f918ca03` (LittleBit cherry-pick) |

Reproduce:

```
git clone --branch b12606 https://github.com/ggml-org/llama.cpp engine
cd engine
git checkout -b littlemore
git remote add cybertiron https://github.com/Cybertiron/llama.cpp-littlebit
git fetch cybertiron littlebit
git cherry-pick -x 4ce291b          # then drop the promo files, see below
```

## What was cherry-picked

Only commit `4ce291b` "Add LittleBit (sub-1-bit binary low-rank) linear support"
from https://github.com/Cybertiron/llama.cpp-littlebit (branch `littlebit`),
MIT licensed (same as llama.cpp). The fork's other commits (KV-cache
q3_K/q2_K/KVarN, README, FUNDING) were NOT taken.

Diff review before applying: no network calls, telemetry, subprocess/eval,
binary blobs or obfuscation. Python converter only reads local files.

What it adds:
- `src/llama-model.h`: `llama_lb_linear` struct and `model.lb_linears` map.
- `src/llama-model.cpp`: a 2-D `blk.N.<x>.weight` that is missing from the file
  while `blk.N.<x>.lb_v` exists is loaded as a LittleBit linear
  (tensors `lb_v`, `lb_u`, `lb_s_v2`, `lb_s_mid`, `lb_s_u1`, plus optional
  residual path with an `r` suffix: `lb_vr`, `lb_ur`, `lb_s_v2r`, ...).
- `src/llama-model-loader.cpp`: `lb_s_*` tensors are placed as `GGML_OP_MUL`.
- `src/llama-graph.cpp`: `build_lb_path` computes
  `mul -> mul_mat -> mul -> mul_mat -> mul`; `build_lora_mm` uses it for
  registered linears, so any architecture going through `build_lora_mm` works.
- `tools/littlebit/lb_to_gguf.py` (+ README): checkpoint-to-GGUF converter.

Local edits on top of the cherry-pick:
- Dropped promotional files/hunks: `README.md` banner, `.github/FUNDING.yml`,
  donation links in `tools/littlebit/README.md`.
- Added explanatory comments to the LittleBit hunks.

## Conflict resolutions

None needed: the cherry-pick applied cleanly onto `b12606` (git auto-merged
`llama-graph.cpp` and `llama-model.cpp`). Surrounding upstream code
(`build_lora_mm`, `create_tensor`) still matched the hunks' expectations.

## Build

Run from an MSVC x64 environment (`vcvars64.bat`), CUDA 12.9 first on PATH:

```
cmake -G Ninja -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON ^
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9" ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe" ^
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j 6
```

(`-DCMAKE_CUDA_COMPILER` pins nvcc 12.9 since CUDA 13.x is also installed;
`-j 6` keeps RAM use manageable on a 16 GB machine.) Binaries and backend
plugins (`ggml-cuda.dll`, `ggml-vulkan.dll`, `ggml-cpu-*.dll`) land in
`engine/build/bin`.

## Known gaps (TODO for Task 8)

- [ ] LoRA adapters are bypassed on LittleBit linears (`build_lora_mm` returns early).
- [ ] The Hadamard input transform (`build_hadamard_input`) is skipped for LittleBit linears.
- [ ] Split-mode / tensor-split regexes do not match `lb_*` tensor names.
- [ ] `lb_to_gguf.py` silently skips linears it does not recognise; it should warn or fail.
