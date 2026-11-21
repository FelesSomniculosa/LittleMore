# Engine changes (llama.cpp fork)

LittleMore runs Samsung LittleBit (sub-1-bit) models through a patched llama.cpp.
The engine lives in `engine/`, which is its **own git repository** (a plain
local clone of upstream) and is listed in the outer repo's `.gitignore`, so the
outer repo does not track it. The pin is recorded here instead.

## What is LittleBit

LittleBit stores each weight matrix of a language model as two matrices of
+1/-1 signs (1 bit per entry) of low rank, plus a few small scale vectors, which
brings a model below 1 bit per weight.  A second "residual" path (LittleBit-2)
repeats the idea on the error left by the first path.  Papers: LittleBit,
https://arxiv.org/abs/2506.13771 ; LittleBit-2, https://arxiv.org/abs/2603.00042 .
Other terms in this project are explained in the glossary of
[development.md](development.md#glossary).

## Glossary

- **cherry-pick**: copy one commit from another git branch or repository onto the current branch.
- **pin**: fix the exact upstream version (tag and commit) the engine is based on, so builds are reproducible.
- **GGUF**: the single-file model format used by llama.cpp (tensors plus metadata).
- **Q1_0**: a ggml storage type for weights with 1 bit per value (a sign) and a scale per block.
- **Q8_0**: a ggml storage type with 8-bit integers and one scale per block of 32 values; llama.cpp also uses it to quantize activations before an integer dot product.
- **backend plugin**: a separate library (`ggml-cuda.dll`, `ggml-vulkan.dll`, `ggml-cpu-*.dll`) that llama.cpp loads at run time to compute on GPU or CPU (built with `GGML_BACKEND_DL`).
- **Hadamard transform**: a fixed orthogonal mixing of the input values that some quantisation schemes apply before a matrix product.
- **build_lora_mm**: llama.cpp's graph helper that builds the matrix multiplication for a linear layer (including LoRA adapters); LittleBit hooks in here.
- **tensor split**: distributing the tensors of a model over several GPUs or devices.

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

## GGML_TYPE_B1 (pure 1-bit sign weights)

Added in the engine (`littlemore` branch): a new ggml type for the +/-1 LittleBit
factors, exactly 1.0 bits per weight with **no scale** (Q1_0 costs 1.125 bpw
because of its fp16 scale; LittleBit keeps all scales in separate F32 vectors).

- `block_b1 { uint8_t qs[16]; }`, `QK_B1 = 128`. Element `i` lives in byte `i/8`,
  bit `i%8` (LSB first). **bit 1 = +1, bit 0 = -1** - the opposite polarity of
  Samsung's checkpoint packing, so the converter flips the bits.
- `quantize_row_b1_ref` (x >= 0 gives +1, so 0.0 is +1) and `dequantize_row_b1`.
- Enum value **`GGML_TYPE_B1 = 44`**, `GGML_TYPE_COUNT = 45`; mirrored in gguf-py
  (`GGMLQuantizationType.B1`, `GGML_QUANT_SIZES`, `quants.B1`).
- **Rebase hazard:** upstream's last type is `F8_E4M3 = 43`. If a future upstream
  claims 44, renumber `GGML_TYPE_B1` in `ggml.h` and `gguf-py/gguf/constants.py`
  and record the new number here (files written with the old number would be misread).
- CPU: `to_float` only. There is **no `vec_dot`** and no `from_float`.
- **Rule: no backend may claim B1 for MUL_MAT / MUL_MAT_ID / GET_ROWS / CPY / SET_ROWS
  (or any other op) until a kernel exists; the fused LITTLEBIT op (Tasks 7, 10-12)
  consumes B1 directly** and must add its own explicit exception in each backend's
  `supports_op`. Tasks 10/11 must not claim B1 mul_mat.
- Audit of `supports_op` (checked by reading code and by running
  `ggml_backend_dev_supports_op` on CPU, CUDA0, Vulkan0, Vulkan1 in `test-b1`):
  - CPU: MUL_MAT_ID / CPY / SET_ROWS (B1 destination) / OUT_PROD / FLASH_ATTN_EXT
    would have reached a null `vec_dot` / `from_float`, and GET_ROWS / OUT_PROD
    would have hit `GGML_ABORT` (no B1 case in `ops.cpp` switches). Fixed with one
    guard at the top of `ggml_backend_cpu_device_supports_op`: any op with a B1
    source or result (except view-like ops) is refused. Repack/extra-buffer paths are
    type-gated (`ggml_repack_get_optimal_repack_type`) and cannot see B1.
  - CUDA / Vulkan: GET_ROWS, SET_ROWS, CPY and MUL_MAT-with-B1-weight are
    whitelists and already answered false. **But MUL_MAT with B1 as src1 (the
    activation) answered TRUE on CUDA0, Vulkan0 and Vulkan1** (measured), which
    would compute garbage. Fixed with the same small guard at the top of both
    backends' `supports_op`.
- Tests: `engine/tests/test-b1.cpp` (ctest `test-b1`), `tests/engine/test_gguf_b1.py`.

## GGML_OP_LITTLEBIT (fused LittleBit linear, CPU reference)

One op that computes a whole LittleBit linear straight from the B1 sign matrices and F32
activations (no activation quantization, unlike the 11-op slow path in `build_lb_path`).
Added in the engine (`littlebit` work on branch `littlemore`); the graph builder does not use it yet.

- **API:** `ggml_littlebit(ctx, x, V, U, s_v2, s_mid, s_u1, Vr, Ur, s_v2r, s_midr, s_u1r)`; the five
  residual arguments are all NULL (single-path) or all set. Also `ggml_littlebit_packed(...)` taking
  the residual scales pre-packed (see below).
- **Shapes** (ggml order, `ne[0]` innermost): `x` F32 `[d_in, n]`; `V` B1 `[d_in, r]` (row j = signs of
  latent unit j); `U` B1 `[r, d_out]` (row i = signs of output i); `s_v2` `[d_in]`; `s_mid` `[r]`;
  `s_u1` `[d_out]`; result F32 `[d_out, n]`. The residual path has its own rank `r2`.
- **Padding rule:** `d_in` and `d_out` are multiples of 128; the ranks `r`, `r2` are padded up to a
  multiple of 128. Padded latent units carry `s_mid = 0` (the converter guarantees it) so they add
  exactly nothing; their V rows / U columns may hold arbitrary +-1. The kernels do not special-case padding.
- **Math** per token and path: `xs = x*s_v2`; `t_j = s_mid_j * sum_k sgn(V[j,k]) xs_k`;
  `y_i = s_u1_i * sum_j sgn(U[i,j]) t_j`; result = primary + residual. `sgn(bit 1) = +1`,
  `sgn(bit 0) = -1`, as in `dequantize_row_b1`.
- **Accumulation (CPU reference):** float32; stage 1 sums k ascending, stage 2 sums j ascending over all
  padded units, then primary + residual. Each output element is computed by one thread in a fixed order,
  so results are bit-identical for any thread count. Other backends agree to float rounding only.
- **Two-stage structure:** stage 1 writes `t` (`r + r2` floats per token, in the CPU work buffer), a
  barrier, stage 2 reads it. GPU kernels mirror this; the CPU kernel is the reference they are tested against.
- **`GGML_MAX_SRC` decision:** it is 10 and the 12-argument form needs 11 source tensors. The three
  residual scale vectors are therefore packed into one F32 tensor `[s_v2r | s_midr | s_u1r]` (length
  `d_in + r2 + d_out`), built inside `ggml_littlebit` with two `GGML_OP_CONCAT` nodes. Raising
  `GGML_MAX_SRC` was rejected: it is used by about 30 files (RPC wire format, backend-meta, CUDA graph
  caches, Hexagon/ET/CANN structs) and changes `sizeof(ggml_tensor)`. A loader that packs the scales
  once at load time can call `ggml_littlebit_packed` and avoid the per-call concats. Source slots:
  `0 x, 1 V, 2 U, 3 s_v2, 4 s_mid, 5 s_u1, 6 Vr, 7 Ur, 8 s_pack_r`.
- **Op number / rebase hazard:** `GGML_OP_LITTLEBIT` is the last entry before `GGML_OP_COUNT`
  (`GGML_OP_COUNT` 100 -> 101; the two static_asserts in `ggml.c` were bumped). The op number is not
  stored in GGUF files.
- **How backends opt in:** answer true for `GGML_OP_LITTLEBIT` in `supports_op`. Every backend already
  refuses any op with a B1 source (Task 6 guard), so each needs an explicit exception placed before that
  guard. Today only the CPU backend has one (`ggml-cpu.cpp`); CUDA and Vulkan still answer false, and
  the scheduler then runs the op on the CPU backend. Asserted in `tests/test-littlebit-ref.cpp`.
- **Tests:** `test-backend-ops test -o LITTLEBIT` (each backend vs CPU, plus the CPU result vs an
  independent double-precision reference in `tests/littlebit-ref.h`, tolerance 1e-4 of max|y|);
  `tests/test-littlebit-ref.cpp` (ctest `test-littlebit-ref`: reference grid, mutation checks,
  thread determinism, supports_op).

## Known gaps / TODO

- [ ] LoRA adapters are bypassed on LittleBit linears (`build_lora_mm` returns early).
- [ ] The Hadamard input transform (`build_hadamard_input`) is skipped for LittleBit linears.
- [ ] Split-mode / tensor-split regexes do not match `lb_*` tensor names.
- [ ] `lb_to_gguf.py` silently skips linears it does not recognise; it should warn or fail.
