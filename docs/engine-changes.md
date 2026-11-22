# Engine changes (llama.cpp fork)

LittleMore runs Samsung LittleBit (sub-1-bit) models through a patched llama.cpp.
The engine lives in `engine/`, which is its **own git repository** (a plain
local clone of upstream) and is listed in the outer repo's `.gitignore`, so the
outer repo does not track it. The pin is recorded here instead.

## What this fork adds and where to find it

The engine is upstream llama.cpp plus these additions (details in the sections below):

| Addition | Where it is described |
|---|---|
| `GGML_TYPE_B1`: a ggml type that stores +1/-1 signs at exactly 1 bit each | [GGML_TYPE_B1](#ggml_type_b1-pure-1-bit-sign-weights) |
| `GGML_OP_LITTLEBIT`: one fused op that computes a whole LittleBit linear (CPU kernels: scalar reference and AVX2; CUDA kernels for any batch: decode kernels for 1..8 tokens; above, an exact tensor-core prefill path on 1-bit bit-plane products (compute capability 8.0+), else the decode kernels on 8-token chunks and a float32 GEMM prefill path) | [GGML_OP_LITTLEBIT](#ggml_op_littlebit-fused-littlebit-linear-cpu-reference), [CPU: AVX2 kernel](#cpu-avx2-kernel-of-ggml_op_littlebit), [CUDA backend](#cuda-backend-of-ggml_op_littlebit-decode-18-tokens), [CUDA batches above 8 tokens](#cuda-batches-above-8-tokens-chunked-decode-and-prefill-path), [CUDA tensor-core prefill path](#cuda-tensor-core-prefill-path-exact-bit-plane-products-task-13b) |
| Loader and graph hook: a linear whose `weight` is missing but whose `lb_v` factors exist is loaded as a LittleBit linear, and `build_lora_mm` builds the fused op for it | [LittleBit in the graph](#littlebit-in-the-graph) |
| Per-backend rules: no backend may claim a B1 tensor except for the fused op; the LittleBit tensors of a layer live on the layer's device when that device can run the op (CUDA), else on the CPU (Vulkan) | the "Rule" bullet and the `supports_op` audit under GGML_TYPE_B1; "Loader" under LittleBit in the graph |
| Pre-packed residual scales `lb_s_pkr` in the GGUF (no per-call CONCAT nodes) | "Loader" under LittleBit in the graph |
| Model-level speed on the GPU (informal) | [Qwen3-0.6B on the GPU](#qwen3-06b-on-the-gpu-informal-speed-read) |
| Test coverage and measured accuracy, including the GPU noise floor | the end of "LittleBit in the graph", then "GPU offload error" |

The rest of this file records how the fork was made (pin, cherry-pick) and what was measured.

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
| Engine HEAD | see `git -C engine log` (cherry-pick `0f918ca03`, then B1, LITTLEBIT op, graph hook) |

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
- `tools/littlebit/lb_to_gguf.py` (+ README): checkpoint-to-GGUF converter. **Since removed**:
  `tools/convert/lb_to_gguf.py` in the outer repo replaces it (B1 output, validation).

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
  (or any other op) until a kernel exists; the fused LITTLEBIT op (CPU kernel, CUDA kernels)
  consumes B1 directly** and must add its own explicit exception in each backend's
  `supports_op`. A GPU kernel for the fused op must not claim B1 mul_mat (the CUDA one does not: its exception
  covers `GGML_OP_LITTLEBIT` only).
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
activations (no activation quantization, unlike the slow path, a chain of 5 ops per path in `build_lb_path`).
Added in the engine (`littlebit` work on branch `littlemore`); the graph builder uses it for B1 linears
(see "LittleBit in the graph" below).

- **API:** `ggml_littlebit(ctx, x, V, U, s_v2, s_mid, s_u1, Vr, Ur, s_v2r, s_midr, s_u1r)`; the five
  residual arguments are all NULL (single-path) or all set. Also `ggml_littlebit_packed(...)` taking
  the residual scales pre-packed (see below).
- **Shapes** (ggml order, `ne[0]` innermost): `x` F32 `[d_in, n]`; `V` B1 `[d_in, r]` (row j = signs of
  latent unit j); `U` B1 `[r, d_out]` (row i = signs of output i); `s_v2` `[d_in]`; `s_mid` `[r]`;
  `s_u1` `[d_out]`; result F32 `[d_out, n]`. The residual path has its own rank `r2`.
- **Padding rule:** `d_in` is a multiple of 128 and `d_out` any positive number (relaxed when the op was wired into the graph: the
  Qwen3.5 DeltaNet gate projections `in_proj_a/b` have `d_out` = number of value heads, e.g. 16; the CPU kernel
  never needed it, **future GPU kernels must handle a ragged `d_out`**); the ranks `r`, `r2` are padded up to a
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
  caches, Hexagon/ET/CANN structs) and changes `sizeof(ggml_tensor)`. The model loader calls
  `ggml_littlebit_packed` with the packed vector stored in the GGUF (`lb_s_pkr`, see "Loader") and so avoids the
  per-call concats; files without it keep the 12-argument form. Source slots:
  `0 x, 1 V, 2 U, 3 s_v2, 4 s_mid, 5 s_u1, 6 Vr, 7 Ur, 8 s_pack_r`.
- **Op number / rebase hazard:** `GGML_OP_LITTLEBIT` is the last entry before `GGML_OP_COUNT`
  (`GGML_OP_COUNT` 100 -> 101; the two static_asserts in `ggml.c` were bumped). The op number is not
  stored in GGUF files.
- **How backends opt in:** answer true for `GGML_OP_LITTLEBIT` in `supports_op`. Every backend already
  refuses any op with a B1 source (the guard described above), so each needs an explicit exception placed before that
  guard. The CPU backend (`ggml-cpu.cpp`) takes any batch, but not with the weights in one of its extra buffer types
  (repack, AMX, ...: they hold only the types they re-lay out; uploading B1 into the repack buffer would dereference
  null traits); CUDA takes any batch (see the two CUDA sections below); Vulkan still answers false. Where the answer is
  false the scheduler runs the op on the CPU backend. Asserted per device, batch size and CPU buffer type in
  `tests/test-littlebit-ref.cpp`.
- **Tests:** `test-backend-ops test -o LITTLEBIT` (446 cases with `-b CPU`): each backend against the CPU (NMSE, see
  the CUDA section for the normalization), the backend output and, once per case, the CPU kernel against an
  independent double-precision reference in `tests/littlebit-ref.h` (tolerance 1e-4 relative to max(max|y|, the expected
  output scale)), and a determinism check. The grid has ragged `d_out` (1, 4, 16, 37, 100; `d_in` stays a multiple of
  128) with and without residual, equal and different residual ranks, n in {1, 4, 8, 9, 32, 512}, every n from 1 to 8
  on shapes that take the multi-tile paths of the CUDA decode kernels, and d_in up to 10368; for the CUDA paths above
  8 tokens also n in {16, 33, 40, 48, 49, 64, 2048} on shapes with several prefill tiles. `tests/test-littlebit-ref.cpp`
  (ctest `test-littlebit-ref`): CPU reference grid (worst relative error 1.8e-5 over 438 cases), mutation checks of the
  reference, thread counts 1/3/4/8 (bit-identical; 6e-7 for the ragged-by-thread cases), supports_op per device and
  batch size, the CUDA grid (see the CUDA section), the CPU's AVX2 kernel against this reference kernel (next section).
  With `-b CPU` the backend under test is the CPU's default kernel (AVX2 on an AVX2 machine) and the comparison
  backend runs the reference kernel (the harness sets `ggml_backend_cpu_set_use_ref` on it).

## CPU: AVX2 kernel of GGML_OP_LITTLEBIT

`ggml/src/ggml-cpu/littlebit-x86.cpp` (the file header documents math, bit layout, blocking, work split and work
buffer). It runs in the CPU backend variants built with AVX2 (`haswell`, `alderlake` and the AVX-512 variants, whose
`/arch:AVX512` also defines `__AVX2__`); the scalar kernel in `ops.cpp` stays the reference and the fallback.

- **Dispatch:** at the top of `ggml_compute_forward_littlebit`: the AVX2 kernel when the variant was compiled with AVX2
  and the backend is not set to its reference implementations (`params->use_ref`, the upstream switch
  `ggml_backend_cpu_set_use_ref`); otherwise, and for shapes the AVX2 kernel does not take (rows that are not whole
  32-sign words, or row strides too large for 32-bit gather offsets; neither occurs with B1), the scalar kernel. Which
  variant runs is chosen at load time from the CPU's features (`GGML_BACKEND_DL` + `GGML_CPU_ALL_VARIANTS`;
  `ggml-cpu/arch/x86/cpu-feats.cpp` scores an AVX2 variant 0 on a CPU without AVX2, so it is never loaded there), so
  the AVX2 code can only run on a CPU that has AVX2. The decision is the same for all threads of the op (both kernels
  contain a barrier). A variant with the kernel lists the feature `LITTLEBIT_AVX2` in `ggml_backend_get_features`
  (shown in llama.cpp's system info).
- **How:** the 8 float lanes of a vector are 8 different rows (8 latent units in stage 1, 8 outputs in stage 2). One
  gather loads the same 32-sign word of the 8 rows; for each of its 32 columns in ascending order the sign bit is moved
  to bit 31 (`vpsllvd` by 31 - c, `vpand` with 0x80000000) and XORed onto the broadcast activation, which is an exact
  negation where the sign is -1, then added. Up to 4 row groups (1 token) or up to 8 tokens per pass share the sign
  masks and the broadcasts (block shapes G x T: 1..4 x 1, 1..2 x 2..4, 1 x 5..8). Stage 1 uses `x * s_v2` of the
  current token tile, computed once per thread and tile into the work buffer (`ggml_littlebit_avx2_work_size`, called
  by `ggml_graph_plan`). The ragged last group of stage 2 (`d_out` not a multiple of 8) uses masked gathers, loads and
  stores, so rows past `d_out` are never read or written.
- **Numerics: bit-identical to the reference kernel when neither kernel is FP-contracted.** Each lane performs exactly the float operations of the
  reference for its row, in the same order (start at +0.0f, add +a or -a for k ascending, then the scale; stage 2
  `s_u1 * primary`, then `+ s_u1r * residual`); no reassociation, no FMA, activations stay float32. Measured:
  `test-littlebit-ref` (7) 110 of 110 cases bit-identical (NMSE 0, bound 1e-6), d_in 128..5248, ragged d_out, n = 1..9,
  16, 17, 33, 512, on 1, 3, 4, 7 and 16 threads; every number printed by `test-littlebit-ref` (CPU grid against the double
  reference, CUDA against the CPU kernel) is the same as before the change; `llama-debug` logits of Qwen3-0.6B at
  `-ngl 0 -dev none` are byte-identical before and after for 1, 16 and 38 tokens, default and `-fa off -ctk f32 -ctv f32`.
  The bit-identity relies on the compiler not contracting `a * b + c` into an FMA differently in the two kernels:
  MSVC does not contract by default, so `test-littlebit-ref` asserts identity for MSVC builds (`_MSC_VER` and not
  `__clang__`: clang-cl defines `_MSC_VER` but contracts like Clang) and asserts only NMSE <= 1e-6 for other compilers
  (GCC and Clang contract by default in GNU mode; not built here).
- **Tests:** `test-littlebit-ref` section (7): requires `LITTLEBIT_AVX2` whenever the variant reports AVX2 (otherwise the
  comparison would compare the scalar kernel with itself), then the AVX2 kernel against the reference kernel as above.
  RED before the kernel existed: the feature check failed (`AVX2 yes, LITTLEBIT_AVX2 no`, 1 failed check).
  The kernel really runs: the CPU backend counts the ops its AVX2 kernel computed (thread 0 adds one per op, a relaxed
  atomic; read through the registry as `ggml_backend_cpu_littlebit_avx2_calls`), and (7) requires +1 for every run on an
  AVX2 backend and +0 for every run on the reference backend. 10 more cases in (7) use misaligned views (x, `s_v2` and the
  packed residual scales at a 4-byte offset, `lb_run_op_unaligned`): now 120 of 120 bit-identical, counted 120 of 120.
  `test-backend-ops test -o LITTLEBIT -b CPU`: 446/446 (AVX2 kernel against the reference kernel with NMSE bound 1e-10,
  against the double reference, determinism), 3 runs. Thread counts 1/3/4/8 still bit-identical (section 3 / 3b, now
  on the AVX2 kernel). Fallback: the test programs run with only `ggml-cpu-sse42.dll` or only `ggml-cpu-x64.dll` next to
  them (no AVX2: scalar kernel) pass, as with only `ggml-cpu-haswell.dll` (AVX2 without AVX-VNNI: AVX2 kernel, 110/110
  bit-identical). The AVX-512 variants compile the same AVX2 code but cannot run on the test CPU (no AVX-512).
- **Mutations** (each applied to `littlebit-x86.cpp`, the `alderlake` variant rebuilt, both test programs run): V
  polarity (stage-1 result negated): all 110 AVX2-vs-reference cases fail (NMSE 1.2 .. 4), 682 failed checks; primary U
  polarity: 110 cases (NMSE 1.2e-2 .. 5.1); MSB-first bit order: 110 (NMSE 0.29 .. 2.7); `s_mid` skipped: 110 (0.67 ..
  43); residual dropped in stage 2: the 85 cases with a residual (NMSE 1.5e-2 .. 1.1); last row of the ragged group not
  stored: the 61 cases with `d_out` not a multiple of 8 (NMSE 6.9e-4 ..); in a block of several row groups the odd
  groups reading the even group's rows: 60 cases (3.6e-3 .. 1.8); every token of a stage-1 tile using the first token's
  activations: the 101 cases with n >= 2 (0.32 .. 3.7); every token of a stage-2 tile using the first token's residual
  t: 78 cases (1.7e-2 .. 3.3). `test-backend-ops -b CPU` failed on each (its CPU-vs-double-reference check exits on the
  first wrong case). Flipping the sign masks of BOTH stages (`vpandn`) changed nothing, bit for bit: that is the
  mathematical no-op already documented for the reference mutations (flipping V and U together cancels exactly).
  Two more, later: a reorder-only mutation (the 32 columns of a word walked 31 -> 0 instead of 0 -> 31: same values,
  other summation order) fails exactly the 120 identity checks of (7) and nothing else (NMSE up to 8.7e-13, inside the
  1e-6 bound; `test-backend-ops -b CPU` passes 446/446, as its NMSE bound 1e-10 cannot see a reordering), so the
  memcmp is what guards "same order"; and a silent fallback (the dispatch never calling the AVX2 kernel) fails the
  counter checks of (7) (121 failed checks, "counted in 0 of 120 cases") while every comparison still passes
  (scalar against scalar) and `test-backend-ops -b CPU` passes 446/446: the counter is what catches it.
- **AVX-512:** not written. The test CPU (i7-13620H, Raptor Lake) has no AVX-512, so such a path could not be run or
  tested here; the AVX-512 variants use the AVX2 kernel.

### Measured speed (i7-13620H laptop, 6 P-cores + 4 E-cores, 16 threads; plugged in, informal)

Op level, `cpu_bench.exe 6 <avx2|ref> 64 1,8` (`engine/tools/littlebit-bench`, 6 threads, every op with its own weights,
64 MB per graph, i.e. cold caches like a model at decode time; G/s = sign multiply-adds per second, one per sign bit and
token; ranges over 3 AVX2 runs and 2 reference runs, alternated):

| d_in x d_out | r + r2 | n | reference kernel | AVX2 kernel | speed-up |
|---|---|---|---|---|---|
| 1024 x 2048 (Qwen3-0.6B q) | 256 + 256 | 1 | 774 .. 790 us (2.0 G/s) | 23.9 .. 25.2 us (62 .. 66 G/s) | 31x |
| 1024 x 1024 (k, v) | 128 + 128 | 1 | 260 .. 262 us | 9.6 .. 10.0 us (52 .. 55 G/s) | 27x |
| 1024 x 3072 (gate, up) | 256 + 256 | 1 | 956 .. 991 us | 31.1 .. 32.0 us | 30x |
| 3072 x 1024 (down) | 256 + 256 | 1 | 1025 .. 1047 us | 30.7 .. 34.1 us | 32x |
| 1024 x 2048 | 256 + 256 | 8 | 1939 .. 1987 us (6.4 G/s) | 88.6 .. 103 us (122 .. 142 G/s) | 20x |
| 4096 x 4096 | 640 + 512 | 1 | 4566 .. 4731 us | 124 .. 150 us (63 .. 76 G/s) | 33x |
| 5120 x 13824 | 1152 + 1024 | 1 | 19.2 .. 19.5 ms | 525 .. 604 us (68 .. 79 G/s) | 34x |
| 13824 x 5120 | 1152 + 1024 | 8 | 82.1 .. 82.4 ms (4.0 G/s) | 2.18 .. 2.45 ms (135 .. 151 G/s) | 36x |

`test-backend-ops perf -o LITTLEBIT -b CPU` (8 threads, one op repeated, weights in cache), reference -> AVX2:
25x to 30x with 1 token (5120 x 13824: 16.5 ms -> 579 us), 18x to 32x with 8 tokens, 9.9x to 13.5x with 128 and 512 tokens
(5120 x 13824, 512 tokens: 1.61 s -> 141 ms, 0.51 dense-equivalent TFLOPS, 2 n d_in d_out / time).

What bounds it: with one token on 6 threads it reads about 9-10 GB/s of weights, while the same machine streams the
dense F16 model at about 70 GB/s (1.40 GiB per token at 46.5 tokens/s, below), so the kernel is not bandwidth bound
but consistent with instruction bound (not verified): 16 vector instructions per 32 sign-adds with one token, about 11 G sign-adds/s per thread
measured (clocks and port use were not measured, so the gap to the instruction ceiling is not attributed). With 8
tokens the sign masks and gathers are shared and the rate about doubles. Keeping the accumulators in registers
mattered (see the comment at `lb_block_impl`: plain loops left them in memory under MSVC; one thread, 8 tokens: 10.5 ..
15.1 against 22.6 .. 28.3 G/s).

Model level, Qwen3-0.6B LittleBit B1 0.55 bpw (`qwen3-0.6b-lb2-0.55-b1.gguf`), `llama-bench -ngl 0 -dev none -t 6
-p 128 -n 32 -r 5`, the engine before (scalar kernel) and after this change in alternating runs, 3 rounds each:

| engine | pp128 (tokens/s) | tg32 (tokens/s) |
|---|---|---|
| before (scalar reference kernel) | 40.4 +- 0.9, 36.7 +- 1.0, 36.2 +- 1.3 | 7.37 +- 0.08, 7.24 +- 0.04, 7.17 +- 0.08 |
| after (AVX2 kernel) | 420 +- 33, 391 +- 32, 360 +- 14 | 94.4 +- 4.5, 95.8 +- 1.7, 92.8 +- 2.0 |

That is 12.9x faster decoding and about 10x faster prompt processing. Same settings, for context: dense F16 Qwen3-0.6B
tg32 46.5 +- 1.1, pp128 264 +- 6; the Q1_0 slow path with the same F16 head tg32 91.9 +- 1.6, pp128 369 +- 19 (the fused
op is now as fast as the int8 chain on the CPU, without its activation rounding). After, other settings: `-t 4` tg32
84.9, `-t 8` 102.6, `-t 12` 96.0 (single runs); `-t 6 -p 512 -n 128 -r 3`: pp512 354 +- 8, tg128 93.2 +- 2.1. Where
decode time goes now, estimated from the op-level times above (28 layers x q, k, v, o, gate, up, down): the 196
LittleBit linears take about 4.6 ms of the 10.7 ms per token; the rest (the F16 output head, 311 MB per token,
attention, norms) was not broken down.

Reproduce: `test-littlebit-ref.exe`, `test-backend-ops.exe test -o LITTLEBIT -b CPU`, `test-backend-ops.exe perf -o
LITTLEBIT -b CPU` from `engine/build/bin`; `cpu_bench.exe` as in `engine/tools/littlebit-bench/README.md`; the model rows
with `llama-bench.exe -m <gguf> -ngl 0 -dev none -t 6 -p 128 -n 32 -r 5`. A laptop's CPU clocks depend on power mode,
temperature and load, and the scheduler decides which threads run on P- or E-cores: compare kernels in alternating
runs, as above.

## CUDA backend of GGML_OP_LITTLEBIT (decode, 1..8 tokens)

`ggml/src/ggml-cuda/littlebit.cu` (+ `littlebit.cuh`), registered in `ggml-cuda.cu` (dispatch in
`ggml_cuda_compute_forward`, a `supports_op` case). The file header documents the math, the bit order and the kernels.

- **Supported:** `x` F32 contiguous with any number of tokens (`ne[1]*ne[2]*ne[3]`; this section is the 1..8-token
  decode kernels, larger batches in the next section), `V`/`U` (and `Vr`/`Ur`) B1 contiguous 2-D, all scales F32
  contiguous, `d_in`, `r`, `r2` multiples of 128, any `d_out`. Anything else answers false in `supports_op`, and the
  scheduler runs the op on the CPU. Models use it since the loader places the LittleBit tensors of a CUDA layer on the
  GPU (see "Loader").
- **Structure:** two launches per call, mirroring the CPU kernel's two stages. V stage: one row index space for the
  latent units of both paths `[0, r + r2)`, result `t` (F32 `[n][r + r2]`) in a scratch buffer from the CUDA context
  pool (no allocation and no host synchronization per call, CUDA-graph safe). U stage: each output row sums its
  primary and residual segment and applies `s_u1` / `s_u1r`. A warp owns R output rows and all tokens (R x n
  accumulators in registers, so each weight bit is read once per call and each activation load feeds R rows); its 32
  lanes split the summed dimension in 32-bit (V) or 8-bit (U) chunks with coalesced loads; the activation (`x * s_v2`
  in the V stage) is staged per block in shared memory with 4 floats of padding per 32 columns, which makes the
  16-byte loads conflict-free; lane partial sums are added by a warp butterfly. Signs are applied with one LOP3
  instruction that flips the float sign bit (one token) or builds +-1.0f for an FMA (several tokens), exact in both
  cases. Deterministic (no atomics; bit-identical repeated runs, asserted).
- **Accuracy:** float32 throughout, in a different summation order than the CPU kernel. Against the independent double
  reference: worst 9.1e-7 relative to the output scale over 350 cases (the CPU kernel: 1.8e-5 over 438 cases).
  Against the CPU kernel: NMSE median 1.9e-13, max 1.5e-12 over 1075 harness case runs (bound 1e-10). ggml-cuda is
  built with `-use_fast_math`, so subnormal intermediate values are flushed to zero (the CPU keeps them).
- **Tests:** `test-backend-ops test -o LITTLEBIT -b CUDA0`: all 446 cases pass, the 241 with n from 1 to 8 here (ragged `d_out`
  1/4/16/37/100, with and without residual, equal and different residual ranks, every batch size 1..8 on shapes that
  take the multi-tile paths: a V stage with a short last activation tile (`d_in` 1152 from 6 tokens, 5248 from 2,
  10368 with 1), a U stage with several tiles (r 1408, r2 1792 from 6 tokens)), each compared with the CPU backend AND
  with the double reference (harness hook `check_output`), plus a determinism check (the other 205 cases, n > 8, are
  in the next section). The NMSE of the op output is normalized by max(sum y^2, sum E[y^2])
  (`lb_expected_sq` in `tests/littlebit-ref.h`; the reference checks use the same scale): with one output element the
  plain NMSE is the squared relative error of a single sum that may cancel to almost 0 by chance (measured up to 4e-11,
  against 5e-13 at the 90th percentile), which would make any fixed bound fail at random. `test-littlebit-ref` checks
  per device which batch sizes are accepted, requires that at least one CUDA device was checked, runs a CUDA grid of
  350 cases (the shapes above, n = 1..8) against the reference (606 more for n > 8, next section), and 6 cases with `x`, `s_v2` and the packed residual
  scales as views at a 4-byte offset (asserted misaligned after allocation), which makes the kernel stage them with
  scalar instead of 16-byte loads. Mutations of the kernel, each failing every case it affects and `test-littlebit-ref`:
  V polarity flipped, residual path dropped, `s_mid` skipped, MSB-first bit order (all 215 then-existing n <= 8 cases or,
  for the dropped residual, all 134 cases with a residual; smallest NMSE 5.6e-3, medians 0.46 to 4); the short last
  activation tile skipped (exactly the 15 multi-tile cases with a short last tile fail, NMSE 4.8e-3 .. 0.34); outputs
  of tokens permuted for n >= 5 (all 85 cases with n >= 5 fail, NMSE 0.77 .. 2.7); the column scale dropped in the
  scalar staging path (only the 6 misaligned cases of `test-littlebit-ref` exercise it; all 6 fail, relative error
  about 1).

### Measured performance (RTX 4050 Laptop GPU, sm_89, 20 SMs, 24 MB L2, 192 GB/s)

Ranks at about 0.55 bits per weight (the `test-backend-ops perf` cases). "harness" = `test-backend-ops perf`: the same
op repeated in one graph, so its small weights stay in the 24 MB L2. "cold" = one graph of many ops with their own
weights (96 MB or more per graph, like the layers of a model), `engine/tools/littlebit-bench/cold_bench.cpp` on the
public ggml API. Bandwidth = weight bytes (signs + scales) / time. The F16 mat-vec is `MUL_MAT` with F16 weights of the same
shape on the same GPU (cold; the harness gives the same within 5% from 4096 x 4096 up). CPU = the scalar reference
kernel in the harness (measured before the AVX2 kernel existed, which is 25x to 30x faster with one token, see "CPU: AVX2 kernel").

| d_in x d_out | r + r2 (padded) | n | weights | LittleBit harness | cold | cold GB/s | F16 mat-vec | speed-up | CPU kernel |
|---|---|---|---|---|---|---|---|---|---|
| 1024 x 1024 | 256 + 128 | 1 | 0.12 MB | 6.7 us | 7.8 us | 15 | 14.3 us | 1.8x | 295 us |
| 1024 x 1024 | 256 + 128 | 8 | 0.12 MB | 15.3 us | 15.2 us | 8 | 13.7 us | 0.9x | 888 us |
| 2048 x 2048 | 384 + 256 | 1 | 0.36 MB | 9.6 us | 10.3 us | 35 | 47.6 us | 4.6x | 989 us |
| 2048 x 2048 | 384 + 256 | 8 | 0.36 MB | 28.3 us | 29.9 us | 12 | 50.0 us | 1.7x | 2.9 ms |
| 4096 x 4096 | 640 + 512 | 1 | 1.25 MB | 16.7 us | 20.5 us | 61 | 184 us | 9.0x | 3.6 ms |
| 4096 x 4096 | 640 + 512 | 8 | 1.25 MB | 70.2 us | 75.7 us | 17 | 191 us | 2.5x | 11.3 ms |
| 5120 x 5120 | 768 + 640 | 1 | 1.89 MB | 22.8 us | 27.9 us | 68 | 289 us | 10.3x | 5.5 ms |
| 5120 x 5120 | 768 + 640 | 8 | 1.89 MB | 102 us | 111 us | 17 | 292 us | 2.6x | 20.4 ms |
| 5120 x 13824 | 1152 + 1024 | 1 | 5.31 MB | 54.1 us | 71.1 us | 75 | 809 us | 11.4x | 15.3 ms |
| 5120 x 13824 | 1152 + 1024 | 8 | 5.31 MB | 261 us | 290 us | 18 | 800 us | 2.8x | 52.8 ms |
| 13824 x 5120 | 1152 + 1024 | 1 | 5.31 MB | 54.4 us | 72.5 us | 73 | 808 us | 11.2x | 15.7 ms |
| 13824 x 5120 | 1152 + 1024 | 8 | 5.31 MB | 237 us | 263 us | 20 | 827 us | 3.1x | 88.6 ms |

What bounds it (micro-benchmarks on the same GPU, programs in `engine/tools/littlebit-bench`, see its README; at the
sustained boost clock, 2040-2340 MHz on this laptop GPU, which idles at 1605 MHz: the CUDA programs warm the GPU up for
3 s first, without that they report about 1.6x lower compute ceilings; all these numbers depend on the GPU, its clock
and power limit):

| quantity | value | program |
|---|---|---|
| DRAM read bandwidth | up to 187 GB/s (128 to 188 between runs) | `read_bandwidth.cu` |
| L2 read bandwidth (16 MB buffer) | about 800 to 890 GB/s | `read_bandwidth.cu` |
| inner-loop ceiling, signs and activations on chip | 180 / 127 / 89 / 50 GB/s of sign bits for 1 / 2 / 4 / 8 tokens | `inner_ceiling.cu` |
| sign trick (shift, LOP3, add) vs predicated add with `2A - S`, 1 token | 188 vs 296 GB/s of sign bits | `sign_variants.cu` |
| tensor cores: `mma.m16n8k16` f16 / bf16, f32 accumulate | 5.4 G per second | `mma_rate.cu` |
| cold-cache op times (table above) | | `cold_bench.cpp` |

So with one token neither ceiling is reached: the large shapes run at 75 to 98 GB/s, about half of both the DRAM
bandwidth and the instruction ceiling. Part of the gap is the measured fixed latency of the two dependent launches
(about 3 us each; the smallest shape is almost only that); the rest is not attributed yet (likely latency inside the
kernel: the staging barrier per activation tile with 8 to 16 warps per SM), so the kernel has room left. With 8 tokens
the op runs at about 20 GB/s against an instruction ceiling of 50 GB/s. The `2A - S` form would raise the one-token
ceiling to about 296 GB/s but changes the numerics (cancellation when |t| is much smaller than the sum of the
activations), so it is not used. Tensor cores: float32-exact results need 3 bf16 MMAs per tile, about 58 GB/s-equivalent
with 8 tokens at the measured rate before building the sign fragments, not clearly better than the CUDA cores, and with
one token 7/8 of each MMA would be wasted. With 8 tokens the 1024 x 1024 case is slower than the F16 mat-vec, which
then runs on tensor cores.

Reproduce (from `engine/build`, CUDA 12.9 `bin` on `PATH`):

```
bin\test-backend-ops.exe test -o LITTLEBIT -b CUDA0
bin\test-littlebit-ref.exe
bin\test-backend-ops.exe perf -o LITTLEBIT -b CUDA0
bin\test-backend-ops.exe perf -o LITTLEBIT -b CPU
bin\test-backend-ops.exe perf -o MUL_MAT -b CUDA0 -p "type_a=f16,type_b=f32,m=(1024|2048|4096|5120|13824),n=(1|8),k=(1024|2048|4096|5120|13824),bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1"
```

The "cold" column and the ceilings: build and run the programs in `engine/tools/littlebit-bench` as its README says
(`cold_bench.exe CUDA0 96` from `build/bin` for the cold column and the F16 mat-vec column; `read_bandwidth.exe`,
`inner_ceiling.exe`, `sign_variants.exe`, `mma_rate.exe` for the table above). Re-run on 2026-11-21 after committing:
cold 5120 x 13824 one token 69.7 us (76 GB/s) against 803 us for F16, the other LittleBit rows within 2% and the F16 rows within 6% of the table.

### CUDA: batches above 8 tokens (chunked decode and prefill path)

Same file (`littlebit.cu`, from "Prefill path" on; the file header documents math, tiles, scratch and numerics).
**This section describes the fallback** (GPUs below compute capability 8.0, HIP, MUSA, or
`GGML_CUDA_LITTLEBIT_F32_PREFILL=1`): on newer NVIDIA GPUs every batch above 8 tokens takes the tensor-core prefill
path of the next section, about 2x to 4x faster per layer at 512 tokens (2.7x to 3.2x for whole models, see the tables there) and more accurate. Dispatch by batch size `n` in
`ggml_cuda_op_littlebit` on the fallback:

| n | path |
|---|---|
| 1..8 | the decode kernels, one launch pair (previous section) |
| 9..16, 9..24, 9..40 or 9..48, by the layer's number of signs S = (r + r2) (d_in + d_out): below 1 Mi, 1..6 Mi, 6..8 Mi, from 8 Mi | the decode kernels on consecutive 8-token chunks (unchanged kernels; every token's result is bit-identical to a smaller call) |
| above | prefill path |

- **Prefill path:** the signs are expanded to floats that already carry the inner scales, `A[j][k] = sgn(V[j,k]) * s_v2[k]`
  (rows of `[V ; Vr]`, residual rows with `s_v2r`) and `B[i][j] = sgn(U[i,j]) * s_mid[j]` (row `i` = U row then Ur row,
  `s_midr`); a sign times a scale is an exact float. Then cuBLAS GEMMs: `t = A . x` for all `r + r2` latent units at
  once, `P = B[:, 0:r] . t[0:r]` straight into the output, `Q = B[:, r:] . t[r:]`, and an epilogue
  `y = s_u1 * P + s_u1r * Q` (fixed order). Padded units have `s_mid = 0`, so `B = +-0` there: exact zero contribution.
- **Why dequantize-then-GEMM:** with many tokens the work, n (r + r2) (d_in + d_out) multiply-adds, is compute bound.
  The decode kernels need an instruction per weight bit and token (about 2 TFLOPS of this work at 8 tokens on the RTX
  4050 Laptop GPU) and cost one call per 8 tokens; cuBLAS SGEMM sustains about 3.5-4 TFLOPS there under the power cap
  and the expansion (4 bytes written and read per sign) is paid once per call. Nsight Systems, 5120 x 13824 at 512
  tokens: SGEMM kernels 88 % of the GPU time, expansion 5 %, epilogue 4 %.
- **Why float32, no TF32 / f16:** ggml-cuda's cuBLAS handle runs with `CUBLAS_TF32_TENSOR_OP_MATH`, so a plain
  `CUBLAS_COMPUTE_32F` GEMM uses TF32 (measured: identical results to `CUBLAS_COMPUTE_32F_FAST_TF32`, worst error against
  the double reference 8.1e-4 in `test-littlebit-ref`, 8x its 1e-4 tolerance, NMSE against the CPU 3.7e-7). The path
  uses `CUBLAS_COMPUTE_32F_PEDANTIC` (all float32 whatever the handle mode): worst 6.4e-7. TF32 would be about 1.7x
  faster at 512 tokens; `CUBLAS_DEFAULT_MATH` + `CUBLAS_COMPUTE_32F` measured as fast as PEDANTIC. The activations
  x * s_v2 are not rounded to f16 / bf16 for the same reason.
  (Side finding, code reading only: the same handle mode means ggml's own F32 `MUL_MAT` through `cublasSgemm` runs in
  TF32, and F16 weights go through `CUBLAS_COMPUTE_16F`, i.e. f16 accumulation, in `ggml_cuda_mul_mat_cublas_impl`.
  Both are plausible sources of the dense multi-token GPU floor below; not tested.)
- **Scratch:** per tile at most `LB_PREFILL_TILE_BYTES` = 16 MiB of expanded signs (tiles are multiples of 128 rows),
  at most 16 MiB of residual GEMM output (`Q`, which also bounds the output tile by the batch size), and the latent
  values `t`, n (r + r2) floats (like the CPU kernel's work buffer); all from the CUDA pool, no cudaMalloc and no host
  synchronization per call. The tile size was measured at 4, 8, 16, 32 and 64 MiB on 4096 x 4096 and 5120 x 13824 /
  13824 x 5120 at 128 and 512 tokens: no consistent difference beyond the 10 % run-to-run spread, so the L2-sized 16 MiB
  was kept.
- **Chunk limits, measured** (re-tuned after the first float32-prefill version; the rule lives in `ggml-cuda/littlebit-dispatch.h`, shared with
  `test-littlebit-ref`). Method: two temporary builds with the path forced (always chunked / always prefill above 8
  tokens), `prefill_bench` with `LB_BENCH_SKIP_F16=1` (LITTLEBIT only, 1.5 s warm-up and >= 1 s timing per case), n = 9,
  12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128, on the linears of Qwen3-0.6B / 1.7B at 0.55 bits per weight and the
  built-in larger shapes; one run per mode, so single values carry the usual ~10 % spread. Time per op in us, chunked /
  prefill:

  | layer (d_in x d_out, r + r2 padded) | S (Mi signs) | n = 16 | 20 | 24 | 32 | 48 | 64 | 96 | 128 | first n from which prefill stays faster |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 1024 x 1024, 128 + 128 (0.6B k, v) | 0.5 | 29 / 34 | 39 / 27 | 44 / 28 | 57 / 30 | 91 / 43 | 115 / 47 | 187 / 56 | 286 / 62 | 20 |
  | 1024 x 2048, 256 + 256 (0.6B q) | 1.5 | 46 / 94 | 64 / 61 | 70 / 62 | 131 / 64 | 189 / 85 | 212 / 90 | 283 / 129 | 365 / 157 | 20 |
  | 1024 x 3072, 256 + 256 (0.6B gate, up) | 2.0 | 60 / 125 | 75 / 77 | 87 / 80 | 156 / 82 | 236 / 117 | 286 / 124 | 458 / 171 | 613 / 182 | 24 |
  | 3072 x 1024, 256 + 256 (0.6B down) | 2.0 | 59 / 77 | 68 / 67 | 82 / 68 | 92 / 70 | 131 / 107 | 172 / 99 | 259 / 149 | 342 / 165 | 20 |
  | 2048 x 2048, 384 + 384 (1.7B q, o) | 3.0 | 74 / 167 | 93 / 99 | 109 / 101 | 144 / 103 | 215 / 137 | 290 / 143 | 432 / 230 | 573 / 293 | 24 |
  | 2048 x 6144, 512 + 512 (1.7B gate, up) | 8.0 | 156 / 451 | 191 / 266 | 241 / 273 | 401 / 279 | 597 / 397 | 793 / 411 | 1185 / 596 | 1575 / 746 | 28 |
  | 6144 x 2048, 512 + 512 (1.7B down) | 8.0 | 164 / 382 | 192 / 244 | 204 / 251 | 288 / 254 | 435 / 331 | 545 / 343 | 760 / 584 | 1016 / 649 | 32 |
  | 4096 x 4096, 640 + 512 | 9.0 | 150 / 508 | 189 / 316 | 222 / 324 | 294 / 330 | 436 / 482 | 580 / 502 | 867 / 691 | 1152 / 785 | 56 |
  | 5120 x 5120, 768 + 640 | 13.8 | 220 / 760 | 277 / 460 | 325 / 470 | 430 / 479 | 638 / 628 | 832 / 656 | 1244 / 997 | 1668 / 1131 | 48 |
  | 13824 x 5120, 1152 + 1024 | 39.3 | 520 / 1464 | 659 / 1230 | 768 / 1258 | 1019 / 1272 | 1481 / 1743 | 1952 / 1818 | 2919 / 3062 | 3882 / 3084 | 128 |
  | 5120 x 13824, 1152 + 1024 | 39.3 | 558 / 2161 | 733 / 1339 | 822 / 1367 | 1095 / 1403 | 1600 / 2158 | 2123 / 2256 | 3330 / 3253 | 4235 / 4718 | none up to 128 |

  The prefill path's time is not monotonic in n: cuBLAS is slower at 16 tokens than at 20 on every shape, and steps up at
  40 and at 80 tokens (kernel choices of cuBLAS for these GEMM shapes; not investigated further), so no single limit is
  right for every layer. The old rule (32 tokens, 48 from 8 Mi signs) was too high for the small layers (up to 74 us,
  i.e. 1.9x, too slow at 32 tokens on 1024 x 3072). A first re-tune by sign count alone (16 / 24 / 48 / 96 tokens for
  <6 / 6..12 / 12..24 / >=24 Mi) fixed that but made 4096 x 4096 up to 30 % slower at 25-56 tokens than the old rule.

  **Second re-tune (22 layer shapes)** (`tools/dev/chunk_crossover_study.sh` / `.py`, raw data in
  `docs/data/chunk_crossover_2026-11-22_{chunk,gemm}.txt`; 22 layer shapes at 0.55 bits per weight: every linear of
  Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B and Llama-3-8B plus the built-in larger shapes, n = 9, 12, 16, 20, 24, 28, 32, 40, 48,
  56, 64, 72, 80, 96, 112, 128, one run per path). Goal: no shape slower than with the old rule by more than 5 % at any
  measured n, keeping the gain on the small layers. Per shape, the limits that meet the goal:

  | S (Mi signs) | shapes | limits within 5 % of the old rule |
  |---|---|---|
  | 0.5 .. 0.8 | Qwen3-0.6B k, v | 16 .. 32 |
  | 1.5 .. 3.0 | the other Qwen3-0.6B linears, Qwen3-1.7B q, k, v, o, Qwen3-4B / Llama-3-8B k, v (4096 x 1024: 24 .. 32) | 16, 20 or 24 .. 32 |
  | 6.5 | Qwen3-4B q (2560 x 4096: 24 .. 40), o (4096 x 2560: 32 .. 56) | 32 .. 40 |
  | 8 .. 39 | Qwen3-1.7B / 4B, Llama-3-8B FFN, 4096 x 4096 (48 .. 56), 5120 x 5120, 5120 x 13824 ... | 48 (some shapes also 40 or up to 80) |

  New rule (in `littlebit-dispatch.h`): 16 tokens below 1 Mi signs, 24 for 1..6 Mi, 40 for 6..8 Mi, 48 from 8 Mi. Result
  on the 22 measured shapes: never slower than the old rule at any measured n (worst ratio 1.00); against the faster of
  the two paths at most 1.11x for the layers below 6 Mi (the old rule: up to 1.91x), and for the large layers the same
  as the old rule (up to 1.35x at 72 tokens on 9728 x 2560, where the chunked kernels stay faster up to 112 tokens). A
  later switch for the large layers cannot be had without a trade-off: 2560 x 4096 (6.5 Mi) needs the switch by 40
  tokens and 4096 x 4096 (9 Mi) not before 48; 14336 x 4096 (31.5 Mi) is 13 % slower than the old rule at 64 tokens
  with a 96-token limit. A second feature was looked at: the d_out / d_in aspect ratio does not separate the shapes
  (square 4096 x 4096 switches late, square 2048 x 2048 early; wide and tall shapes of the same sign count behave
  alike), the sign count with these band edges does. Single runs carry the usual ~10 % spread, and the limits are
  measured on this one GPU and are GPU specific (the header says so); only speed depends on them. Boundaries tested:
  `test-littlebit-ref` (5c) covers one shape per band (T + 1 = 17, 25, 41, 49 on the prefill path against T on the
  chunked path), and `test-backend-ops` has cases at 16 / 17, 24 / 25, 40 / 41 and 48 / 49 tokens.
- **Determinism and CUDA graphs:** every output element has one fixed sequence of operations (cuBLAS is bit-reproducible
  for a given toolkit, GPU and shape; ggml never enables cuBLAS atomics); repeated runs are bit-identical (harness check).
  The path is plain stream work (kernels and cuBLAS calls; cuBLAS has its workspace set by ggml-cuda, so it allocates
  nothing during capture): llama.cpp captures prompt-processing graphs too (Nsight Systems on `llama-bench -p 512`:
  2 `cudaGraphInstantiate`, 6 `cudaGraphLaunch`), and the harness's determinism re-runs, which replay captured graphs
  (51 instantiations, 222 graph launches for the 70 cases of `-p "...n=33|n=512"`), are bit-identical to the first,
  direct run.
- **Accuracy:** against the double reference worst 6.4e-7 relative to the output scale over 606 cases with 9..2048 tokens
  (`test-littlebit-ref`); against the CPU kernel (harness NMSE, normalization as above, 3 runs): chunked path median
  2.3e-13, max 2.3e-12 (354 case runs); prefill path median 2.2e-13, max 2.5e-12 (252 case runs); the decode kernels in
  the same runs median 9.6e-14, max 6.1e-12. The bound 1e-10 is unchanged (16x above the largest value). Float32
  accumulation over K terms has a relative rounding error of order 1e-7 sqrt(K) per output (K up to d_in = 10368 here),
  i.e. NMSE of order 1e-13..1e-12, which is what is measured. Dispatch boundaries on the same tokens
  (`test_cuda_dispatch_boundaries`, 9 shapes, at least one per band of the rule): 9 tokens (8 + 1 chunks) bit-identical to
  an 8-token and a 1-token call; T + 1 tokens on the prefill path against T on the chunked path (T = 16, 24, 40, 48):
  worst NMSE 3.8e-13, worst max difference 6.1e-7 of the output scale.
- **Tests:** `test-backend-ops -o LITTLEBIT -b CUDA0`: the 210 cases with n > 8 pass (451 in all; with the current
  rule: n = 9, 16 on small layers, 24 on a 1.3 Mi-sign layer, 40 on a 6 Mi-sign layer, 40 / 48 on a 13.75 Mi-sign layer:
  chunks; 17, 25, 32, 33, 41, 49, 64, 512, 2048: prefill; tiles: d_in 5248 with 640 + 640 latent rows (latent
  tiles of 768 + 512 rows, the first holding primary and residual rows), 1408 rows without residual, d_out 4100 (output
  tiles 3200 + 900 or 2944 + 1156) and 2048 tokens with r + r2 = 256 (output tiles bounded by the batch: 2048 + 2048 +
  404)). `test-littlebit-ref`: 606 CUDA cases with 9..2048 tokens, the dispatch-boundary checks, 4 of the 10 misaligned
  view cases on these paths. Before the change all of them were refused (RED: `test-littlebit-ref` aborted on the first
  9-token case; `test-backend-ops` reported 196 cases "not supported").
- **Mutations** (each applied to the kernel, rebuilt, both test programs run): output-tile expansion without `s_mid`:
  all 84 harness cases with n > 32 fail (NMSE 2.6 .. 60), 225 failed checks in `test-littlebit-ref`; output tile offset
  ignored: exactly the 3 cases with several output tiles fail (NMSE 0.43 .. 1.1); latent tile written at column 0:
  exactly the 4 cases with several latent tiles (NMSE 1.3 .. 1.6); residual dropped in the prefill epilogue: all 52
  cases with n > 32 and a residual (NMSE 1.4e-2 .. 1.0); every chunk reading the first chunk's activations: all 118
  cases with 9..32 tokens (NMSE 4.4e-4 .. 1.7; the mutation pass ran before the 40 / 48 / 49-token cases of the large layer were added, on 443 harness cases).

Measured, `prefill_bench` (own weights per op, every case warmed for 1.5 s; 0.55 bits-per-weight ranks; time per op in
us; F16 = `MUL_MAT` with F16 weights of the same shape, cuBLAS f16 tensor cores; "TFLOPS dense-eq." = 2 n d_in d_out /
time of the LittleBit op):

| d_in x d_out | n = 9: LB / F16 | 32 | 128 | 512 | 512: LB TFLOPS (work / dense-eq.) | 512: F16 TFLOPS |
|---|---|---|---|---|---|---|
| 1024 x 1024 | 20.9 / 14.4 | 58.1 / 17.1 | 88.2 / 21.8 | 262 / 62.8 | 3.1 / 4.1 | 17.1 |
| 2048 x 2048 | 38.7 / 49.4 | 116 / 55.2 | 249 / 66.2 | 754 / 216 | 3.6 / 5.7 | 19.9 |
| 4096 x 4096 | 93.3 / 192 | 290 / 211 | 782 / 255 | 2663 / 773 | 3.6 / 6.5 | 22.2 |
| 5120 x 5120 | 137 / 292 | 428 / 323 | 1129 / 419 | 3893 / 1200 | 3.8 / 6.9 | 22.4 |
| 5120 x 13824 | 344 / 776 | 1069 / 863 | 3851 / 1194 | 11264 / 3345 | 3.8 / 6.4 | 21.7 |
| 13824 x 5120 | 326 / 775 | 1012 / 830 | 3115 / 972 | 11219 / 2617 | 3.8 / 6.5 | 27.7 |

So up to 8-32 tokens a LittleBit layer is faster than F16 on the large shapes (the F16 kernels then read all weights per
call), from 128 tokens on it is 3x to 4x slower: float32 on the CUDA cores (about 3.8 TFLOPS of real work) against f16
tensor cores with f16 accumulation (about 20-28 TFLOPS), on about 58 % of the dense work at 0.55 bits per weight.
`test-backend-ops perf -o LITTLEBIT -b CUDA0` gives the same within about 10 % (e.g. 5120 x 13824, 512 tokens: 11.1 ms);
its first timings per case can run at a low clock because the harness spends seconds on the CPU between LittleBit
cases (the GPU then drops to 210-480 MHz; the CPU reference check is now skipped in perf mode to shorten that gap).

Reproduce (from `engine/build`, CUDA 12.9 `bin` on `PATH`): `bin\test-littlebit-ref.exe`,
`bin\test-backend-ops.exe test -o LITTLEBIT -b CUDA0`, `bin\test-backend-ops.exe perf -o LITTLEBIT -b CUDA0 -p "n=(32|33|128|512)$"`,
and `prefill_bench.exe CUDA0 64 9,16,32,33,64,128,512` built as in `engine/tools/littlebit-bench/README.md`.

### CUDA: tensor-core prefill path (exact bit-plane products)

File `ggml/src/ggml-cuda/littlebit-bp.cu` (its header documents the math, kernels, layouts and the error bound). On
NVIDIA GPUs from compute capability 8.0 every batch above 8 tokens takes this path instead of the chunked decode
kernels and the float32 GEMMs above; those remain for older GPUs, HIP, MUSA, and when `GGML_CUDA_LITTLEBIT_F32_PREFILL=1`
is set (used by tests and by the benchmark plan to measure both paths in the same run). The CUDA backend reports the
feature `LITTLEBIT_BP_PREFILL` when every device takes it, so tests know the dispatch rule.

**Why (profile first).** Nsight Systems on `llama-bench -p 512 -n 0` (6 passes, `--cuda-graph-trace=node`), per pass of
512 tokens, before this change:

| | LittleBit linears | of which float32 SGEMM | epilogue | sign expansion | rest of the model | F16 linears (dense model) |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 124 ms | 106 ms (86 %) | 11.6 ms | 6.0 ms | 22 ms | 22 ms (19.4 GEMM + 2.8 type conversions) |
| Qwen3-1.7B | 334 ms | 296 ms (89 %) | 22.4 ms | 16.1 ms | 30 ms | 88 ms |

So the time is in the float32 GEMMs on the CUDA cores (about 3.3-3.8 TFLOPS, against 18-23 TFLOPS of the f16 tensor
cores for the dense model); fusing the epilogue or caching the expanded signs (two further options considered: fusing the epilogue, caching the expanded signs) could save
at most the 12-15 % outside the GEMMs. The lever is to move the products onto tensor cores without rounding the
activations.

**How.** Signs are 1-bit, and the tensor cores have an exact 1-bit product: `mma.sync.m16n8k256 .b1 .and.popc`
(engine/tools/littlebit-bench/mma_rate.cu: 10.1-11.7 G instructions per second on the RTX 4050 Laptop GPU, 330-380 T bit
products per second; bf16 with f32 accumulation: 5.4 G/s of m16n8k16, 22 TFLOPS; int8: 9.7 G/s of m16n8k32). Per token
and chunk of 256 columns the activations are written as 28-bit integers relative to the chunk maximum (the only
rounding: at most 2^-27 of the chunk maximum per value, values within 8x of the maximum exact) and split into 28 bit
planes; with sgn = 2 w - 1, sum sgn q = 2 sum w q - sum q, and sum w q = sum over planes of 2^p popc(w AND plane), all
exact int32 (Horner's rule in the accumulator, two groups of 14 planes). Each chunk then costs two float roundings
(the exact chunk sum to float, the add to the accumulator), against one rounding per column in the float32 paths. The
latent kernel's 256 rows are exactly one chunk of t, so it quantizes t into planes in its epilogue; t never goes to
memory. Kernels (as first built, engine commit `eaf9b01e6`; since the speed levers the planes of x are built inside the latent kernel, see "CUDA: prefill speed levers"): planes of x (one warp per token and chunk, a warp-wide 32 x 32 bit transpose), latent stage (3-stage
cp.async ring, weight bits read straight from global memory one chunk ahead), output stage (t planes of 8-16 tokens
resident in shared memory, warps loop over 32-row tiles; ranks above 512 in windows that add to y). Planes are
bounded to 12 MiB per token chunk (they stay in L2; scratch is bounded). Deterministic (integer sums; float adds in
chunk order), CUDA-graph safe (plain launches), no atomics.

**Accuracy (lossless).** `test-littlebit-ref`, 607 CUDA cases with 9..2048 tokens: worst error against the double
reference 1.81e-7 (the float32 GEMM path, same cases, `GGML_CUDA_LITTLEBIT_F32_PREFILL=1`: 6.93e-7; tolerance 1e-4),
worst NMSE against the CPU kernel 2.25e-12 (2.35e-12; bound 1e-10; this metric is dominated by the CPU kernel's own
float32 rounding); 9 tokens (this path) against 8 (decode kernels) on the same tokens: worst NMSE 7.9e-14 (the float32
path's boundary: 3.8e-13). `test-backend-ops -o LITTLEBIT -b CUDA0`: 452/452 (also 452/452 with the float32 path
forced). Model level, `tests/engine/test_fused_graph_qwen3_06b.py` (coarse guards, see "GPU offload error"): 44-token
prompt, default configuration 2.95e-2 (GPU / pure CPU 1.34, bound 2.0; before: 3.09e-2, 1.40), exact configuration
2.22e-2 (dense floor ratio 1.80, bound 2.0; before: 1.94e-2, 1.58); 16-token prompt (now this path, before the chunked
decode kernels): 3.92e-2 / 4.27e-2. These model-level values move by tens of percent with any change of summation
order on this init-only model (measured over several correct paths and prompts: 0.019 .. 0.189 between correct paths and prompts), so they are
within the noise of correct code; the precise detectors are the op-level bounds above. Plainly: the exact-config
guard ratio rose from 1.58 to 1.80 (limit 2.0), so its margin shrank from 21 % to 10 %; it can trip on unrelated
correct changes, and must not be loosened for that, but read against the op-level tests (the ratio does not track op
error: see the P = 24 row below).

New test teeth: `LB_BP_TOL = 4e-7` (worst error against the double reference of the tensor-core cases; measured
1.8e-7; 24- and 20-bit builds give 7.1e-7 and 1.0e-5 and fail it) and a case with two token chunks and two output windows (5248 x 37, 640 + 640 latent units, 512 tokens) in
both test programs. Mutations (each built and run, source restored and compared; the harness had 451 cases then, 452 with the case added afterwards): top plane not negated (harness
241/451 pass, error 1.3e4), wrong low-part sum (241/451, 2.2e-2), later output windows overwriting y (351/451, 1.24),
token-chunk offset of x ignored (caught only by the new case in test-littlebit-ref, error 1.57: the harness had no
case with two token chunks, now added), residual path using s_mid (314/451, inf), 20-bit activations (passes TOL and
the NMSE bound, error 1.0e-5, NMSE 8.7e-11; fails `LB_BP_TOL`).

**Dispatch rule.** With this path the decode kernels keep 1..8 tokens and every larger batch takes it: measured with
`prefill_bench` (n = 9 .. 48, LITTLEBIT only), this path against the chunked decode kernels, us per op:

| layer | 9 | 12 | 16 | 20 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|---|
| 1024 x 1024 (0.6B k, v) | 14.6 / 20.4 | 14.9 / 24.6 | 14.9 / 28.6 | 15.2 / 38.4 | 15.4 / 42.6 | 15.5 / 56.3 | 16.3 / 84.6 |
| 1024 x 2048 (0.6B q) | 16.8 / 30.8 | 17.1 / 37.3 | 17.3 / 45.7 | 18.9 / 61.6 | 20.4 / 70.4 | 20.5 / 92.3 | 25.1 / 137.6 |
| 1024 x 3072 (0.6B gate, up) | 19.0 / 40.6 | 19.7 / 47.4 | 19.8 / 58.5 | 24.0 / 73.3 | 24.6 / 85.5 | 27.2 / 112.5 | 28.8 / 167.9 |
| 2048 x 2048 (1.7B q, o) | 25.4 / 48.9 | 26.2 / 60.0 | 26.5 / 74.9 | 31.0 / 94.8 | 31.8 / 110.7 | 33.7 / 145.0 | 44.7 / 217.9 |
| 2048 x 6144 (1.7B gate, up) | 41.1 / 96.1 | 41.7 / 119.5 | 42.0 / 158.7 | 46.1 / 193.4 | 48.0 / 232.0 | 58.4 / 304.1 | 82.2 / 453.5 |

(1.4x faster at 9 tokens on the smallest layer to 5.5x at 48 tokens on the largest; one run each.)

**Speed, op level** (`prefill_bench`, own weights per op, 1.5 s warm-up per case; LittleBit with this path / F16
`MUL_MAT`, us per op; the float32 GEMM path at 512 tokens in brackets, from the table above):

| d_in x d_out | n = 9 | 32 | 128 | 512 |
|---|---|---|---|---|
| 1024 x 1024 | 16.6 / 14.8 | 17.1 / 18.4 | 31.2 / 23.0 | 95.1 / 65.1 (262) |
| 2048 x 2048 | 25.4 / 49.4 | 29.7 / 55.8 | 78.9 / 71.5 | 277 / 219 (754) |
| 4096 x 4096 | 50.8 / 193 | 75.5 / 226 | 227 / 276 | 931 / 843 (2663) |
| 5120 x 5120 | 65.7 / 292 | 99.7 / 328 | 389 / 438 | 1269 / 1292 (3893) |
| 5120 x 13824 | 141 / 773 | 249 / 938 | 845 / 1228 | 3416 / 3422 (11264) |
| 13824 x 5120 | 162 / 837 | 259 / 832 | 904 / 1010 | 3636 / 2829 (11219) |

The Qwen3 layer shapes at 512 tokens (same tool; float32 path measured in the same session before the change):

| layer (padded ranks) | float32 GEMM path | tensor-core path | F16 `MUL_MAT` |
|---|---|---|---|
| 0.6B q 1024 x 2048 (256 + 256) | 536 | 157 | 141 |
| 0.6B k, v 1024 x 1024 (128 + 128) | 202 | 100 | 67 |
| 0.6B o 2048 x 1024 (256 + 256) | 535 | 195 | 106 |
| 0.6B gate, up 1024 x 3072 (256 + 256) | 758 | 190 | 159 |
| 0.6B down 3072 x 1024 (256 + 256) | 686 | 238 | 159 |
| 1.7B q, o 2048 x 2048 (384 + 384) | 927 | 354 | 226 |
| 1.7B k, v 2048 x 1024 (256 + 256) | 534 | 188 | 107 |
| 1.7B gate, up 2048 x 6144 (512 + 512) | 2432 | 694 | 591 |
| 1.7B down 6144 x 2048 (512 + 512) | 2239 | 781 | 647 |

Where the time goes now (Nsight Systems on `prefill_bench`, share of the op): 0.6B q: output stage 57 %, latent 29 %,
planes of x 14 %; 0.6B o: latent 50 %, output 27 %, planes 23 %; 1.7B gate: output 65 %, latent 27 %, planes 8 %; 1.7B
down: latent 62 %, output 20 %, planes 18 %. The inner loop alone (planes in shared memory, no global traffic) runs at
about 90 % of the instruction's peak; the kernels reach 40-80 % of it (5-9 G instructions per second), limited by
latency at block start / end, global loads of the weight bits and the plane traffic. Not pursued further in this task.

**Model level** (first tensor-core build; superseded by the later build below, see "Speed-lever result"; gate harness, 3 interleaved rounds, [bench/results/phase1_13b.md](../bench/results/phase1_13b.md)): pp512 Qwen3-0.6B 9628 t/s (float32 path in the same rounds 3051, F16 11643, bf16 9256): 3.2x faster, 0.83x F16; Qwen3-1.7B 4118 (1531, F16 3872, bf16 3722): 2.7x faster. The 1.06x over F16 on the mean is a power-cap artifact: the F16 model's clock falls to 705-825 MHz within its 10 repetitions (LittleBit: 1395 MHz minimum). At equal clocks (`bench/clock_matched.py`, first timed repetition, both at 2325-2340 MHz) LittleBit is about 0.8-0.9x F16 on Qwen3-1.7B (0.82 .. 0.97) and below F16 on Qwen3-0.6B in this build (0.98 in the one round at equal clock, 0.82-0.94 against an F16 at lower clocks (rounds 0, 2, 3, 5; round 4 is left out: its F16 first repetition, 8338 t/s, is an outlier against 10961-12211 in the other rounds, and the ratio would be 1.22; in rounds 0, 2 and 4 the F16 run has no clock sample inside its first repetition (rep1_clock_mhz null), so "F16 at a lower clock" there rests on the clock at the start of the repetition only)); see bench/results/phase1_13b.md. tg128 unchanged (219.6 / 143.3). VRAM 801 vs 821 MiB and 1191 vs 1223 MiB (smaller scratch).

**Options that trade accuracy (not adopted; measured for the decision).** The number of activation bits P is a
compile-time constant (`LB_BP_BITS`); fewer bits mean fewer instructions:

| option | pp512 Qwen3-0.6B (3 interleaved rounds) | pp512 Qwen3-1.7B | op: worst error vs double reference | op: worst NMSE vs CPU | model level (44 tokens, exact / default config) | VRAM |
|---|---|---|---|---|---|---|
| P = 28 (adopted) | 8584 (9048, 8339, 8365) | 3443 (3498, 3455, 3376) | 1.8e-7 | 2.3e-12 | 2.22e-2 (ratio 1.80) / 2.95e-2 (1.34): pass | same |
| P = 24 | 9014 (+5.0 %) | 3866 (+12.3 %) | 7.1e-7 (4x worse; = the float32 GEMM path's 6.9e-7) | 3.2e-12 | 2.83e-2 (2.30: FAILS the 2.0 guard) / 2.44e-2 (1.11) | same |
| P = 20 | 9667 (+12.6 %) | 4054 (+17.8 %) | 1.0e-5 (57x worse; fails `LB_BP_TOL`) | 8.7e-11 | 2.80e-2 (2.27: FAILS) / 3.64e-2 (1.66) | same |
| TF32 GEMMs (float32 path) | about 1.7x the float32 path, i.e. slower than P = 28 | | 8.1e-4 (8x over the 1e-4 tolerance) | 3.7e-7 | not measured | same |

(pp512 here: plain `llama-bench -p 512 -n 0 -r 5`, not the gate protocol; the model-level values are single runs of the
coarse guard.) "Fails the guard" is not evidence of real harm by itself: P = 24 has the float32 GEMM path's op error
(7.1e-7 vs 6.9e-7) yet scores 2.30 where the float32 path scores 1.58, so the guard ratio does not track op error;
the op-level columns are the measure of accuracy. P = 24 and P = 20 now also fail `LB_BP_TOL` (4e-7). Split-precision bf16 / 3xTF32 GEMMs were not built: their measured instruction rates (bf16 f32-acc
22 TFLOPS with 3 products per exact value, TF32 11.8 TFLOPS with 2) put them below the exact 1-bit path even at their
peak.

### CUDA: prefill speed levers

Follow-up to the section above: lossless changes only (op error, determinism and fallbacks unchanged), each measured
against the previous version with `prefill_bench` in interleaved A/B rounds (variant DLL copies, 3-4 rounds, median).
Because this laptop GPU runs at its power cap (45 W, SM clock 1.5-2.0 GHz under this load, varying by run), the
comparison is in **cycles**: time per op x the median SM clock sampled every 100 ms by `nvidia-smi` while the GPU draws
more than 25 W. Repeated identical configurations differ by up to about 4 % in this metric.

**Where the time was** (Nsight Systems inside `llama-bench -p 512` on Qwen3-1.7B, per layer, before the levers but
with the planes already built in the latent kernel): latent kernels 1.9 ms, output kernels 1.6 ms, although the
output stage has more tensor-core instructions (8.2 M against 6.9 M): the latent kernel ran at 3.2-4.0 G instructions
per second, the output kernel at 3.7-5.9 G (9.5 G peak at 2.19 GHz, `mma_rate.cu`). Gaps between kernels: 0.16 ms
per 512-token pass, i.e. launches are not a cost. Diagnostic builds of the latent kernel (wrong results, timing only):
without the per-chunk plane build 6-25 % fewer cycles; without the weight loads or the per-chunk barrier as well,
little more. So the build of the planes, not the loads or the barrier, was the latent stage's overhead.

Adopted (engine commits `20e96284a`, `9f531c3f5`; test case `2d9b9e98b`):

1. **Planes of x built inside the latent kernel.** The block's 8 warps each build one token's planes of the next
   chunk into a 2-slot shared-memory ring; the separate planes kernel, its launch and its global scratch (2 paths x 3.5
   bytes per input value) are gone. Same arithmetic, bit-identical results.
2. **One basic block for products, build and loads; `mma` asm no longer `volatile`.** The compiler can then put the
   build's shuffles and integer work between the dependent instructions of the Horner chains. The last chunk (nothing
   to build) is peeled off. Plainly: its speed effect (0.94-0.98 cycles, about 3 %) is **below the 5 % rule and within
   the noise**; it is kept as part of lever 1, which was adopted for removing a kernel launch per call and the global
   planes-of-x scratch (bit-identical results), not for a speed gain of this step.
3. **Output-stage row split.** About 4 blocks per resident block slot (at least one row tile per warp) instead of the
   former "fewest waves x tiles" rule, which gave e.g. 32 large blocks on 20 SMs (12 SMs doing twice the work of 8).

Cycles per op relative to the previous step (prefill_bench, 512 tokens):

| layer (padded ranks) | 1. build in latent (vs `eaf9b01e6`) | 2. one basic block | 3. row split |
|---|---|---|---|
| 0.6B q 1024 x 2048 (256 + 256) | 0.94 | 0.98 | 0.91 |
| 0.6B k, v 1024 x 1024 (128 + 128) | 0.94 (time) | 0.98 | 0.92 |
| 0.6B o / 1.7B k, v 2048 x 1024 (256 + 256) | 0.94 | 0.95 | 0.95 |
| 0.6B gate, up 1024 x 3072 (256 + 256) | 1.01 (time) | | 0.87 |
| 0.6B down 3072 x 1024 (256 + 256) | 0.91 | | 0.97 |
| 1.7B q, o 2048 x 2048 (384 + 384) | 0.98 | 0.97 | 0.91 |
| 1.7B gate, up 2048 x 6144 (512 + 512) | 0.92 | 0.97 | 1.04 (same split as before: noise) |
| 1.7B down 6144 x 2048 (512 + 512) | 0.96 | 0.94 | 0.97 |

(blank: not measured in that round; "time": first round, before the cycle metric, us per op.) End to end against
the tensor-core build `eaf9b01e6`: see the table "Speed-lever result" below.

Tried and **not adopted** (gain below 5 % or a loss; sources restored):

| lever | result (cycles vs the version it was built on) |
|---|---|
| latent block of 16 warps = 512 rows (planes built once per path instead of once per 256 rows; one block per SM) | 1.06-1.12x slower on the 1.7B layers (fewer blocks per SM hide less latency) |
| 4 Horner chains of 7 planes instead of 2 of 14 (more independent instructions per warp; bit-identical) | latent only: 0.99-1.02; latent and output: 1.00-1.04 (output kernels then spill) |
| latent kernel at 3 blocks per SM (`__launch_bounds__(256, 3)`, 80 registers, no spills) | 0.97-1.00 |
| sharing the planes of x between q / k / v and gate / up (lever 1 of the plan) | not possible losslessly: the planes are those of x * s_v2, and s_v2 (a per-input-column scale) is different for every linear; sharing them would change the arithmetic |

Mutations of this round (each built and run, source restored; test-littlebit-ref / test-backend-ops CUDA0):
path 1 using s_v2 instead of s_v2r: fails, 314/452, error 2.2; the build written into the slot being read: fails,
375/452; the peeled last chunk dropped: fails, 241/452, error 1.08; the prefetch of x one chunk early: fails,
376/453, error 1.93; the token-chunk offset of x ignored: first **passed every test** (with only the planes of t in
the 12 MiB budget the 5248 x 37 case no longer splits into two token chunks), so a case with two token chunks was
added (256 x 37, 1024 + 1024 latent units, 2048 tokens, both test programs): now fails, 452/453, error 1.56. A ragged case followed (review fix: 640 x 37, 1152 + 1152 latent units, 2050 tokens: token chunks of 1040 +
1010 with a partial token tile, 2.5 chunks of x, 4.5 chunks of t per path, three output windows; engine `b564b699b`):
it passes, and the same mutation fails both new cases (452/454, errors 1.56 and 1.27).

**Speed-lever result** (bench/results/phase1_13c.md, build `2d9b9e98b`). Op level, cycles against the tensor-core build `eaf9b01e6` (3 interleaved rounds,
512 tokens): Qwen3-0.6B layers 0.81-0.85 (one whole layer 0.82), Qwen3-1.7B q / o 0.92, k / v 0.85, gate / up 0.99,
down 0.87 (one layer 0.93). Model level (`bench/clock_matched.py`, pp512 first repetition; one rule for all runs: only runs whose first
repetition is entirely at 2340 MHz, 5 of 24 runs excluded and listed in the results file): Qwen3-0.6B LittleBit
12000-12127 -> 12985-13541 t/s, i.e. 0.93-0.94 -> 1.01-1.05 against the median F16 run (12854; F16 itself spreads
9050-14486 at that clock): parity within the noise. Qwen3-1.7B 4350 (one run) -> 3775 / 4871 / 4756, i.e. 0.78 ->
0.68 / 0.87 / 0.85 against the F16 median 5576; the op-level 0.93x cycles per layer bounds the gain to about 7.5 %
(0.78 -> 0.84), and the observed 0.85-0.87 is not distinguishable from that bound: about 0.84-0.87, still below F16.
VRAM, measured (peak `nvidia-smi` memory during `llama-bench -p 512`, 2 rounds each): 801 -> 791 MiB (0.6B), 1191 ->
1181 MiB (1.7B); the scratch no longer holds the planes of x (computed for 1.7B down at 512 tokens: 13.1 MB -> 3.7
MB). Accuracy and
determinism unchanged: every change is bit-identical (test-littlebit-ref worst 1.806e-7, NMSE 2.251e-12, as before).
Not tried (lever 3 of the plan): cp.async prefetch of the weight bits (the diagnostic build without any per-chunk weight
loads saved little, and the loads are already issued one chunk / one row tile ahead), persistent or stream-K
scheduling (the row split, a cheaper form of the same balancing, took most of what imbalance cost), smaller tiles for
small shapes (the m16n8k256 instruction fixes 8 tokens and 16 rows per tile; smaller blocks were not built). Not
done: merging the q / k / v and gate / up ops into shared launches (graph-level reordering plus multi-op
kernels); the in-model trace shows no launch gaps (0.16 ms per pass), so its gain would come only from fuller grids
on the small k / v layers. Where the 1.7B time is now: the latent stage (per-chunk plane build, repeated for each 256
latent rows) and the output stage of gate / up, both at roughly 60-75 % of the instruction peak per cycle.

## LittleBit in the graph

`build_lora_mm` (src/llama-graph.cpp) finds a LittleBit linear through `model.lb_linears` (keyed by the pointer
`create_tensor` returned for the "weight") and then:

- **B1 factors** (what `tools/convert/lb_to_gguf.py` writes): one `ggml_littlebit_packed(...)` node when the file carries
  the packed residual scales `lb_s_pkr` (or has no residual path), else one `ggml_littlebit(...)` node (12-argument form,
  older files: the residual scales are packed by two CONCAT nodes inside, per call). F32 activations, no activation
  quantization. The input is cast / made contiguous first if needed.
- **Q1_0 factors** (the legacy form of Cybertiron's converter): the old chain `build_lb_path`, kept as a documented
  slow baseline. It multiplies Q1_0 weights by Q8_0-quantized activations, so it carries int8 activation noise
  (measured 50x to 100x the error of the fused path, see below). The test helper
  `tests/smoke/_assets.py::b1_to_q1_0_gguf` derives such a file from a B1 GGUF (the converter only writes B1).
- `w_s` (the optional per-weight scale) is applied after both forms.
- Biases are not part of `build_lora_mm`: callers add them afterwards, see the bias table.

### Loader

- `create_tensor` accepts `lb_v` of type B1 or Q1_0 only.
- **Placement of B1 linears (capability probe):** all tensors of one linear (signs and F32 scales) go into ONE buffer
  type, chosen by `llama_lb_select_buft` (`src/llama-model.cpp`): the first entry of the layer's buffer-type list (the
  layer's device, its host / extra buffer types, then the CPU fallbacks; the same list the loader uses for the layer's
  other weights) whose device answers `supports_op` true for a `GGML_OP_LITTLEBIT` node built on a scratch context with
  the linear's real shapes and types, its weights marked as living in that buffer type (zero-size dummy buffer, as the
  loader's MUL_MAT probe does), for 1 and for 512 tokens. The choice is passed to the loader as `buft_fixed`
  (`llama_model_loader::create_tensor`): no per-tensor op check, no `--override-tensor` (as before, overrides do not
  apply to LittleBit tensors), the mmap / host-buffer rule still applies. Why a probe and no device-name rule: the
  weights are only useful where the op can run, and the backends already say where that is; a backend that gains a
  kernel gets the weights without a loader change. Results: a CUDA layer keeps its LittleBit tensors on the GPU; a
  Vulkan layer (no kernel) falls through Vulkan, Vulkan_Host and CPU_REPACK (refused by the CPU, see "How backends opt
  in") to the plain CPU buffer type; Q1_0 factors and non-LittleBit models are unaffected (the probe runs for B1 only).
  Verified on Qwen3-0.6B (0.55 bpw, B1, `GGML_SCHED_DEBUG=2 llama-completion -v`):
  - `-ngl 99` (CUDA): 2 splits per graph (CPU, nothing to do; CUDA0 with the 9 usual graph inputs), every one of the
    196 `LITTLEBIT` nodes and all their weight sources on `CUDA0`, no `CUDA0#...` weight copies;
  - `-dev Vulkan0 -ngl 99`: the loader logs "tensor 'blk.0.attn_q.lb_v' (b1) (and 1567 others) cannot be used with
    preferred buffer type Vulkan0, using CPU instead" (196 x 8 tensors), every `LITTLEBIT` node and its weights on `CPU`,
    only the activation is copied (`CPU#attn_norm-0#0`); 226 splits per graph. Logits against the fp32 reference
    (`-fa off`, f32 KV): 8.9e-5 (1 token), 2.3e-2 (16), 2.9e-2 (42 tokens), top-1 equal; the same order as CUDA;
  - `-ngl 0`: before (engine `1f1f0e663`, CPU pin) vs after, same file and prompt, `llama-debug` logits: bit-identical
    for 1 and 16 tokens (default and `-fa off` f32 KV); with 42 tokens they differ (2e-2 .. 2.9e-2) because llama.cpp's
    op offload now also moves the prompt's `LITTLEBIT` nodes to CUDA0 (CUDA's `offload_op` takes batches of 32 tokens or
    more, which CUDA now supports; their weights are copied per call, as for dense weights in CPU layers; decode nodes
    stay on the CPU). With `--no-op-offload`, or `-dev none` (no GPU at all), the result is bit-identical to before for
    every prompt. Cost of that weight copy (`llama-bench -ngl 0 -t 6 -p 32,64,512 -n 0 -nopo 0,1 -r 5`, Qwen3-0.6B
    B1): every prompt batch of 32 tokens or more copies each layer's LittleBit weights (35 MiB of signs plus the scales
    for the whole model) to the GPU, and it still pays: pp32 565 +- 76 vs 480 +- 47 tokens/s without op offload, pp64 1070
    vs 421, pp512 2640 vs 442.
  - Placement guard: `tests/engine/test_fused_graph_qwen3_06b.py::test_littlebit_nodes_and_weights_on_cuda_at_ngl99`
    parses `GGML_SCHED_DEBUG=2` output of `llama-completion -ngl 99`: every `LITTLEBIT` node (196 per graph, 4508 in the
    printed graphs) is on CUDA0 with all of its sources (activation and 8 weight tensors) on CUDA0 and none of them a
    per-call copy; the same check flags every node of a `-dev Vulkan0 -ngl 99` run (nodes and weights on the CPU), so it
    can fail.
- **Pre-packed residual scales:** the converter also writes `blk.N.<x>.lb_s_pkr` = [`lb_s_v2r` | `lb_s_midr` | `lb_s_u1r`]
  (F32, n_in + r2 + n_out; validated on its own against the float64 reference decode and byte-compared with the three
  vectors). A B1 linear whose file has it loads it into `llama_lb_linear::s_pkr` and skips the three separate vectors
  (new loader flag `TENSOR_SKIP_QUIET`: skipped like `TENSOR_SKIP`, logged at debug level), so the graph calls
  `ggml_littlebit_packed` and needs no CONCAT nodes; the Q1_0 chain does the reverse (reads the three, skips
  `lb_s_pkr`). Files without `lb_s_pkr` load as before. Measured on Qwen3-0.6B (`GGML_SCHED_DEBUG=2`, one decode graph):
  651 nodes and 0 CONCAT with `lb_s_pkr`, 1043 nodes and 392 CONCAT (2 per linear) without; logits bit-identical
  between the two files at `-ngl 0` and `-ngl 99` (guarded per model family by
  `test_fused_graph_families.py::test_old_format_without_packed_scales_is_bit_identical`: the tiny model of each family
  converted, `lb_s_pkr` stripped, logits bit-identical at `-ngl 0` and `-ngl 99`; a copy whose `lb_s_pkr` has its last
  segment doubled moves the logits, so the packed tensor is really read). Cost: 2.7 MB more file for Qwen3-0.6B (670.7 -> 673.4 MB), no extra
  memory. Chosen over packing at load time, which would keep the three vectors in memory as well (they must be read to
  build the packed copy).
- The scale vectors are loaded as `GGML_OP_MUL` operands (they are broadcast-multiplied into the activations), not as
  the MUL_MAT weight their tensor id would suggest.
- **Coverage check** (new, in `llama_model_base::load_tensors`, before the generic "wrong number of tensors"): every
  `*.lb_v` in the file must have been consumed by the architecture, and a `<linear>.bias` next to it must have been
  consumed too, otherwise the load fails naming the tensor (a linear nobody loads would otherwise run without its
  weight; an ignored bias gives wrong output). Tested in `tests/engine/test_fused_graph_families.py`.
- LoRA: adapters are not applied to LittleBit linears (the dense weight never exists). The adapter loader
  (`llama-adapter.cpp`) used to reject such an adapter with "does not exist in base model"; it now skips those
  tensors and logs one warning ("LoRA adapters are not supported on LittleBit linears ..."). The warning is guarded by a function-level
  `static std::once_flag`, so only the FIRST adapter loaded in a process warns; later adapters are skipped silently. The skip is the only place the case
  can happen: `get_weight(w)` in `build_lora_mm` never finds a delta for a LittleBit linear. Tested (warning printed
  once, logits unchanged).

### Hadamard / precision policy

`build_hadamard_input` and `prec_policy` are driven by GGUF metadata (`prism.hadamard.*`, the tensor-extra precision
arrays) naming dense weights `blk.N.<x>.weight`; nothing else enables them. Our converter writes none of these
keys, so for all four families they do not apply to a LittleBit GGUF. A file that did carry such metadata for a
LittleBit linear fails loudly: the Hadamard loader looks the name up with `get_tensor` and throws "prism.hadamard
weight not found" (the tensor is called `...lb_v`); `prec_policy` ignores names it cannot resolve. The Hadamard loader also accepts only the
architectures Llama, Qwen3, Qwen3 MoE, Qwen3.5 (+MoE) and Qwen3-Next, so Gemma 3 with such metadata is refused outright. Hadamard path tested
(`test_hadamard_metadata_on_littlebit_linear_fails_load`: a LittleBit Llama GGUF with `prism.hadamard.*` metadata naming `blk.0.attn_q.weight` fails
to load with that error); `prec_policy` found by reading `llama_prec_policy::load` only.

### Coverage: every linear of a transformer block goes through `build_lora_mm`

No `ggml_mul_mat` call on a layer weight remains in `models/{qwen3,llama,gemma3,qwen35,delta-net-base}.cpp`
(0 occurrences); every linear below reaches `build_lora_mm` (read in `llama-graph.cpp` and the builders):

| Family | HF linear | engine tensor | path |
|---|---|---|---|
| Qwen3, Llama, Gemma 3 | q_proj, k_proj, v_proj | attn_q/k/v | `build_qkv` -> `build_lora_mm` |
| | o_proj | attn_output | `build_attn` -> `build_lora_mm(wo, ...)` |
| | gate_proj, up_proj, down_proj | ffn_gate/up/down | `build_ffn` -> `build_lora_mm` |
| Qwen3.5 (full attention layers) | q_proj (includes the gate), k_proj, v_proj | attn_q/k/v | `build_qkv` |
| | o_proj | attn_output | `build_lora_mm(wo)` in `build_layer_attn` |
| Qwen3.5 (DeltaNet layers) | in_proj_qkv, in_proj_z | attn_qkv, attn_gate | `build_qkvz` -> `build_lora_mm` |
| | in_proj_b, in_proj_a | ssm_beta, ssm_alpha | `build_lora_mm` in `build_layer_attn_linear` |
| | out_proj | ssm_out | `build_lora_mm` |
| | gate/up/down | ffn_* | `build_ffn` |

Not LittleBit linears: embeddings and the output head (the checkpoint keeps them dense, the converter writes F16, or
Q8_0 / Q6_K with `--head-type`),
norms, the DeltaNet conv / `dt_bias` / `A_log`. Merged tensors (`wqkv`, `ssm_beta_alpha_cat`, `wqkv_z`) also go through
`build_lora_mm`; the converter does not write them. Qwen3.5's MTP head is not converted. Anything a GGUF carries that
the builder does not request fails the load (see the coverage check).

### Biases, per family (HF `attention_bias` / `mlp_bias` switches)

| Family | bias on converted linears in HF | engine consumes it? |
|---|---|---|
| Qwen3 | q,k,v,o when `attention_bias` | q/k/v: yes (`create_tensor_qkv` loads `wq_b..`, `build_qkv` adds them). o: **no before the fused-op change**, now `wo_b` is loaded and `build_attn` adds it |
| Llama | q,k,v,o (`attention_bias`), gate,up,down (`mlp_bias`) | yes, all (`wo_b`, `ffn_*_b`) |
| Gemma 3 | q,k,v,o when `attention_bias` | q/k/v yes; o: **no before the fused-op change**, now loaded and applied (`build_attn` was called with a literal `NULL`) |
| Qwen3.5 | q,k,v,o when `attention_bias`; DeltaNet projections and MLP have none | q/k/v yes; o: added with the fused-op change (`build_layer_attn` adds `wo_b`) |

Tested with random biases on every family (`test_biases_are_applied`): the fused graph matches a reference that adds
the same biases (max-abs 7e-6 .. 3.5e-4) and is far from one that ignores them. A bias on a linear whose builder has
no bias slot (e.g. Qwen3 FFN) fails the load. Note `reference.py` itself drops the bias of a replaced Linear (it
warns); the tests wrap it.

### Measured: fused graph vs the fp32 reference (CPU, tiny models, same token ids)

| Family | tokens | max-abs | rms | corr | top-1 | slow path max-abs |
|---|---|---|---|---|---|---|
| Qwen3 | 5 | 4.0e-4 | 8.3e-5 | 0.99999997 | same | 2.1e-2 |
| Llama | 37 | 1.9e-4 | 6.3e-5 | 0.99999999 | same | 1.1e-2 |
| Gemma 3 (6 layers, 5:1 pattern) | 37 | 7.0e-4 | 2.0e-4 | 0.99999981 | same | 3.4e-2 |
| Qwen3.5 (DeltaNet + attention) | 5 | 2.5e-4 | 5.3e-5 | 0.99999999 | same | 1.8e-2 |

Logit rms is about 0.33. Qwen3-0.6B (init-only, 0.55 bpw, 28 layers), see the table in
`tests/engine/test_fused_graph_qwen3_06b.py`: 1.3e-2 on CPU in the default configuration (flash attention, f16 KV
cache), **1e-4 with `-fa off -ctk f32 -ctv f32`**, 1.4e-2 .. 5.5e-2 with `-ngl 99` (the whole graph on CUDA, the
LittleBit linears on the CUDA kernels; 1.5e-2 .. 6e-2 while they still ran on the CPU), against 0.73 .. 1.6 for the
slow path (these figures are the output of the tests in that file, run with `-s`).
Speed (CPU, informal and machine-dependent): `test_cpu_speed_fused_vs_chain_is_reported` measured the CPU reference kernel
at 10 tokens/s against 89 tokens/s for the Q1_0 chain in one run, and 9.2 against 50.6 when repeated on 2026-11-21 (32 tokens
from "Hello"); either way the scalar reference kernel is several times slower than the slow path, and it is not tuned.
With the AVX2 kernel (see "CPU: AVX2 kernel") the fused op decodes at about 93-96 tokens/s (`llama-bench -t 6`), as fast
as the Q1_0 chain with the same F16 head (91.9) and 2x the dense F16 model (46.5).

**Gemma 3 layer pattern:** the engine reads the per-layer sliding-window pattern from the array key
`gemma3.attention.sliding_window_pattern` (bool per layer, true = sliding window; `load_swa_pattern`) and falls back to the real
Gemma 3 layout (every 6th layer global) when the key is absent. llama.cpp's own `Gemma3Model` converter writes the key
for no config, so a model whose `layer_types` is not 5:1 (a tiny or custom model) ran silently wrong: the two-layer
`[sliding, full]` test model was read as `[sliding, sliding]` and was off by 4.4e-1 in the logits for fused and slow path alike (1 token exact, 2 tokens 1e-2, 37 tokens 4e-1).
`tools/convert/lb_to_gguf.py` now writes the array from `layer_types` for every Gemma 3 conversion (an integer
`sliding_window_pattern` other than 1 / 6 is written as the integer; an unknown layer type or a list of the wrong
length is a `ConversionError`; configs without `layer_types` keep the engine default). Verified in llama.cpp against the reference
(37 tokens, window 16, max-abs): `[sliding, full]` 3.8e-4, `[sliding, full] * 2` 4.6e-4, `[full, sliding, sliding]` 5.3e-4,
four sliding layers 4.0e-4, the 6-layer 5:1 pattern 7.0e-4
(`tests/engine/test_fused_graph_families.py::test_gemma3_layer_patterns`, `tests/convert/test_gemma_pattern.py`).

### GPU offload error: how much is the CUDA noise floor

`tools/dev/dense_floor.py` runs the DENSE Qwen3-0.6B (no LittleBit) through the same llama.cpp paths against its own fp32 HF
(transformers, CPU) reference, next to the LittleBit file, on the same token ids and settings. max-abs over the vocabulary
(`default` = flash attention + f16 KV, `exact` = `-fa off -ctk f32 -ctv f32`):

Columns: "dense F16" and "dense F32" are the original (unquantized) Qwen3-0.6B written as an F16 or an F32 GGUF; "dense F32 + F16 head" is the F32
file with only the tied token-embedding / output-head tensor stored as F16, which is how the LittleBit file stores its head, so it separates the
effect of the F16 head from everything else. All dense columns are compared with the dense fp32 HF reference; the last column compares the
LittleBit B1 file with the LittleBit reference (`reference.py`).

| prompt | ngl | config | dense F16 | dense F32 | dense F32 + F16 head | LittleBit B1 vs its reference |
|---|---|---|---|---|---|---|
| 1 token | 0 | default | 5.3e-4 | 5.4e-4 | 5.4e-4 | 1.35e-2 |
| 1 token | 0 | exact | 2.7e-5 | 1.5e-5 | 1.5e-5 | 1.5e-4 |
| 1 token | 99 | default | 1.1e-2 | 4.9e-4 | 4.9e-4 | 1.45e-2 |
| 1 token | 99 | exact | 1.1e-2 | 1.7e-5 | 1.7e-5 | 4.9e-3 |
| 16 tokens | 0 | default | 7.1e-3 | 7.3e-3 | 8.6e-3 | 1.34e-2 |
| 16 tokens | 0 | exact | 3.7e-5 | 4.3e-5 | 4.4e-5 | 9.8e-5 |
| 16 tokens | 99 | default | 2.0e-2 | 2.5e-2 | 2.5e-2 | 3.7e-2 |
| 16 tokens | 99 | exact | 2.9e-2 | 3.9e-2 | 3.9e-2 | 6.0e-2 |

The LittleBit column was measured while the LittleBit op still ran on the CPU inside the offloaded graph. With the op on
the CUDA kernels (this engine) the `-ngl 99` LittleBit values are 1.38e-2 (1 token, default), 4.8e-3 (1 token, exact),
3.14e-2 (16, default) and 5.53e-2 (16, exact), i.e. 1.25x and 1.43x the dense floor at 16 tokens (`test_gpu_error_is_within_dense_floor_factor`).

Control for the 1-token GPU case: the LittleBit file with its F16 tensors widened to F32 gives 1.1e-4 (exact; 2.6e-5 with the op on CUDA) and 1.3e-2 (default),
against 4.9e-3 and 1.45e-2 with the F16 head. The LittleBit run at `-ngl 99` differs from the same run at `-ngl 0` by exactly the same amount
as it differs from the reference (the CPU-exact run is the reference to 1e-4).

Conclusions, with what is and is not shown:
- CPU: the fused path is exact to 1e-4 with `-fa off` + f32 KV; the 1.3e-2 default error is flash attention's f16 path
  (the dense model has 5e-4 at 1 token, but the init-only LittleBit model is much more sensitive to it; not further isolated).
- GPU, 1 token, exact config: the 4.9e-3 is the F16 output head on CUDA (F32 head: 1.1e-4). The dense F32+F16-head control does
  **not** reproduce it (1.7e-5), so it depends on the LittleBit model's hidden state; the head itself is shown to be the carrier, the reason it
  matters only for this model is not isolated.
- GPU, 16 tokens: the floor exists without any LittleBit: dense F32 on CUDA is 2.5e-2 / 3.9e-2 against 1.7e-5 at one token, so it comes with several
  tokens (batched CUDA matmuls; TF32 / f16 tensor-core use is the likely cause but was NOT tested). LittleBit is 1.46x / 1.56x the dense
  floor by max-abs (rms 1.47x / 1.20x). **About 1.5x is not explained by the dense control** (unattributed residual). The LittleBit model has other activation
  magnitudes than the dense one; that was not isolated.
- With the CUDA kernels for `GGML_OP_LITTLEBIT` (float32, 1e-6 relative to the double reference per op) the 16-token
  ratio to the dense floor went down (1.46 -> 1.25 default, 1.56 -> 1.43 exact) and the 1-token F32-head run from 1.1e-4
  to 2.6e-5; the remaining excess over the dense floor is still not attributed.

Regression guards in `tests/engine/test_fused_graph_qwen3_06b.py`: LittleBit-vs-reference <= 2.0x dense-vs-dense-reference at 16 tokens, `-ngl 99`
(measured 1.25 / 1.43 with the op on CUDA); 1 token `-ngl 99` exact: F16 head <= 1e-2 (4.8e-3), F32 head <= 2.5e-4 (2.6e-5; the limit was 1e-3
while the op ran on the CPU and measured 1.1e-4).

Prefill on the GPU (`PROMPT_PREFILL`, 44 tokens, so every Qwen3-0.6B linear of the prompt batch takes the
prefill path): exact config 1.94e-2 against the dense floor 1.23e-2, 1.58x (limit 2.0). Default config 3.09e-2 against
1.49e-2, 2.08x: above 2, diagnosed rather than loosened. On the pure CPU (`-dev none`, no GPU at all) the same 44
tokens give LittleBit 2.20e-2 against dense 8.31e-3 in the default config (2.65x) and 1.29e-4 in the exact one, so the
init-only LittleBit model amplifies some rounding far more than the dense model, independent of the GPU. Isolated
on the pure CPU (5 prompts of 22..62 tokens): the carrier is f16 rounding of keys / values in
attention. Flash attention gives identical errors with an f16 or an f32 KV cache; with flash attention off, the f16
KV cache alone gives the same level (LittleBit 2.4e-2..1.1e-1, dense 5.5e-3..8.6e-3); with f32 KV and no flash
attention LittleBit is at 1.3e-4..5.2e-4 (dense 1.7e-5..2.5e-5). The prefill path is not the carrier: with the CUDA op forced onto the chunked decode kernels (temporary build)
the run gives 2.52e-2 (default) and 2.91e-2 (exact), the same level in a different rounding order. The default-config
prefill guard is therefore LittleBit-on-GPU <= 2.0x LittleBit-on-the-pure-CPU in the same configuration (measured
1.40). Side finding: at `-ngl 0` a prompt of 32 tokens or more is not a CPU-only run (op offload moves batched ops,
including LittleBit, to CUDA0): 44 tokens at `-ngl 0` exact give 2.46e-2, not the 1.3e-4 of `-dev none`.

**These model-level GPU guards are coarse; the dense-floor factor is not a sound detector in general**. Measured on 5 prompts of 22..62 tokens (all above the 16-token chunk limit), each on the prefill path and on
the chunked decode kernels (temporary forced build), max-abs against the fp32 references, `-ngl 99`:

| run (5 prompts) | LittleBit default | LittleBit exact | / dense floor, default | / dense floor, exact | GPU / pure CPU, default |
|---|---|---|---|---|---|
| prefill path (as committed) | 0.031 .. 0.191 | 0.019 .. 0.189 | 2.08 .. 19.8 | 1.58 .. 25.1 | 1.40 .. 1.80 |
| chunked decode kernels (forced) | 0.025 .. 0.202 | 0.029 .. 0.170 | 1.69 .. 21.0 | 2.36 .. 22.5 | 1.14 .. 2.42 |
| mutation: residual term x 1.001 | 0.105 .. 0.439 | 0.075 .. 0.494 | 9.3 .. 45.6 | 7.8 .. 65.4 | 3.44 .. 7.81 |
| mutation: residual term x 1.01 | 0.98 .. 3.09 | 0.98 .. 3.12 | 86 .. 269 | 102 .. 342 | 21.6 .. 74.3 |
| mutation: residual term x 1.1 | 7.7 .. 22.3 | 7.7 .. 22.4 | 681 .. 1430 | 801 .. 1527 | 87 .. 606 |
| mutation: residual dropped | 20.5 .. 25.5 | 20.1 .. 25.5 | 1468 .. 2126 | 1739 .. 2666 | 171 .. 991 |

(Mutations: in the prefill epilogue, `y = s_u1 * P + m * s_u1r * Q` with m = 1.001 / 1.01 / 1.1, or the residual term
removed; one rebuild of ggml-cuda each. Measured with the earlier chunk limits (16 tokens for every Qwen3-0.6B
linear); with the current limits (24 for all but k / v) the 22-token prompt would run those linears on the chunked
kernels, the
other four prompts still on the prefill path.) So the 2.0x dense factor fails correct code on most prompts (it holds
only where the test applies it: the 16-token prompt in both configurations, the 44-token prompt in the exact one), the GPU / pure-CPU ratio has correct runs up to 2.42, and the absolute
errors of correct runs (up to 0.20) overlap with the smallest mutation (x 1.001: from 0.075). On the test's own prompt
both guards catch every mutation, including x 1.001. The op-level tests (`test-littlebit-ref`, `test-backend-ops`,
NMSE <= 1e-10) remain the precise detectors of the kernels; the model-level guards only catch gross integration errors.
Decision (recorded from these figures): keep the committed guards ("B1"): default config
LittleBit-GPU <= 2.0x LittleBit-pure-CPU, exact config LittleBit <= 2.0x the dense floor. Their measured false-failure
rates on the 10 correct runs of the study: 1/10 for the default-config guard (2.42: forced chunked path, 48-token
prompt; the committed prefill path 0/5) and 9/10 for the exact-config dense factor (it holds only on the test's
prompt); on the test's prompt both catch every mutation down to x1.001. Reproduce: `tools/dev/prefill_guard_study.sh`
(temporary builds) and `tools/dev/prefill_guard_study.py` (measure / cpu / table); raw data:
`docs/data/prefill_guard_study_2026-11-22.jsonl` and `docs/data/prefill_guard_cpu_2026-11-22.jsonl`.

## Qwen3-0.6B on the GPU (informal speed read)

(An early informal read. The formal speed gate, both Qwen3 sizes, both head settings, bf16 / F16 / Q4_K_M baselines and
the harness: [bench/results/phase1.md](../bench/results/phase1.md). Gate PASS on the original weights: tg128
LittleBit 0.55 vs bf16 212.9 vs 126.9 tokens/s on Qwen3-0.6B and 142.9 vs 51.1 on Qwen3-1.7B, in a run at the GPU's
power cap; the numbers in this section are that early informal read, measured cooler.)

`llama-bench -ngl 99 -p 512 -n 128 -r 3` (default settings: flash attention auto, f16 KV), RTX 4050 Laptop GPU warmed
for 3 s before each model (`warm.exe`, an FMA loop), on 2026-11-21; peak VRAM = largest `memory.used` of nvidia-smi
sampled every 200 ms during the run (whole GPU, includes the CUDA context). Same tokenizer and head for all files:
the embedding and the output head are F16 (311 MB each, not LittleBit). Informal: one machine, clocks vary by about
5 % between runs; the formal speed gate is measured later with its own harness.

| file / engine | size | pp512 (tokens/s) | tg128 (tokens/s) | peak VRAM |
|---|---|---|---|---|
| LittleBit B1, 0.55 bpw, this engine | 636 MiB | 4212 +- 215 | **227 +- 11** | 821 MiB |
| dense F16 (`qwen3-0.6b-base-f16.gguf`) | 1.40 GiB | 12271 +- 1014 | 124.5 +- 6.3 | 1595 MiB |
| Q1_0 chain (B1 factors re-encoded as Q1_0, same F16 head), this engine | 638 MiB | 9341 +- 304 | 131.3 +- 2.6 | 823 MiB |
| LittleBit B1, engine before this change (LittleBit tensors pinned to the CPU) | 634 MiB | 45.7 +- 0.3 | 6.44 +- 0.03 | 751 MiB |

- Decode: LittleBit B1 generates 1.82x faster than the dense F16 model and 1.73x faster than the Q1_0 chain.
- Prompt processing: 2.9x slower than F16 and 2.2x slower than the Q1_0 chain (whose CUDA path runs int8 MMQ on Q8_0
  activations; the fused op stays float32, see the prefill section). Superseded by the tensor-core prefill path (see its section and [bench/results/phase1_13b.md](../bench/results/phase1_13b.md)).
- Where decode time goes (Nsight Systems, `llama-bench -p 0 -n 128 -r 1`, 129 decode steps, kernel time per token
  4.26 ms of 4.4 ms wall, i.e. the GPU is about 97 % busy; decode runs as CUDA graphs, 256 `cudaGraphLaunch`):
  the F16 output head (`mul_mat_vec_f`, 311 MB per token) 1.84 ms = 43 %, at about 169 GB/s; the 196 LittleBit linears
  1.81 ms = 42 % in 392 kernel launches (4.6 us per launch on average; their 42 MB of signs and scales per token amount
  to only about 23 GB/s, so they are bound by the fixed cost per launch of these small 1024..3072-wide shapes, not by
  bytes); flash attention 5 %; norms, RoPE, residual adds, SiLU, KV writes 10 % (the residual adds are separate kernels
  here, 56 per token, while the dense model fuses them into its mat-vec). The dense F16 model: 7.7 ms of kernel time
  per token, 93 % in its mat-vecs.
- Next gains, from these numbers: the head (43 %: a Q8_0 / Q6_K head would cut its bytes 2-2.7x); per-launch cost of
  the LittleBit op (one launch instead of two per linear, merging the q/k/v and the gate/up linears that share an input
  into one op, fusing the residual add); for the 0.6B model the CONCAT nodes are already gone (`lb_s_pkr`).

Reproduce: `llama-bench.exe -m <gguf> -ngl 99 -p 512 -n 128 -r 3` from `engine/build/bin`; the Q1_0 file with F16 head
is `tests/smoke/_assets.copy_gguf(b1, dst, B1 -> Q1_0)` on the B1 file (the cached slow-path baseline widens its head
to F32 and is not comparable for speed); the profile: `nsys profile -t cuda --cuda-graph-trace=node llama-bench.exe ...`
then `nsys stats -r cuda_gpu_kern_sum`.

## Known gaps / TODO

- [x] LoRA adapters on LittleBit linears: skipped at adapter load with one warning (not applied; there is no dense weight to add a delta to).
- [x] The Hadamard input transform / precision policy: not applicable (metadata driven), fails loudly if metadata names a LittleBit linear.
- [x] Layer split over several GPUs: LittleBit tensors follow their layer's device (placement probe per layer). Row
  split mode does not exist for CUDA in this llama.cpp version; `--override-tensor` does not apply to LittleBit tensors.
- [x] `lb_to_gguf.py` silently skipped unknown linears: the engine copy is deleted; `tools/convert/lb_to_gguf.py` validates every linear.
- [x] CUDA kernel for `GGML_OP_LITTLEBIT`, 1..8 tokens, ragged `d_out` handled (see "CUDA backend").
- [x] Use it in models: LittleBit tensors live on the GPU of an offloaded CUDA layer; CUDA paths for any batch.
- [x] Fast CPU kernel: AVX2, bit-identical to the scalar reference (see "CPU: AVX2 kernel").
- [ ] AVX-512 CPU path (the AVX-512 variants use the AVX2 kernel; no AVX-512 CPU to test on).
- [ ] Vulkan kernel (Vulkan layers run the op on the CPU).
- [ ] Prompt processing is 3x to 4x slower than F16 per large layer (float32 on CUDA cores vs f16 tensor cores).
- [x] Speed gate: LittleBit 0.55 generates faster than bf16 on CUDA for Qwen3-0.6B (1.68x) and Qwen3-1.7B
  (2.80x), original weights ([bench/results/phase1.md](../bench/results/phase1.md)).
- [x] Qwen3-1.7B rows of the gate re-run on the original weights (real vs proxy within 0.2 %).
