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
- `tools/littlebit/lb_to_gguf.py` (+ README): checkpoint-to-GGUF converter. **Removed again in Task 8**:
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
Added in the engine (`littlebit` work on branch `littlemore`); the graph builder uses it for B1 linears since Task 8
(see "LittleBit in the graph" below).

- **API:** `ggml_littlebit(ctx, x, V, U, s_v2, s_mid, s_u1, Vr, Ur, s_v2r, s_midr, s_u1r)`; the five
  residual arguments are all NULL (single-path) or all set. Also `ggml_littlebit_packed(...)` taking
  the residual scales pre-packed (see below).
- **Shapes** (ggml order, `ne[0]` innermost): `x` F32 `[d_in, n]`; `V` B1 `[d_in, r]` (row j = signs of
  latent unit j); `U` B1 `[r, d_out]` (row i = signs of output i); `s_v2` `[d_in]`; `s_mid` `[r]`;
  `s_u1` `[d_out]`; result F32 `[d_out, n]`. The residual path has its own rank `r2`.
- **Padding rule:** `d_in` is a multiple of 128 and `d_out` any positive number (relaxed in Task 8: the
  Qwen3.5 DeltaNet gate projections `in_proj_a/b` have `d_out` = number of value heads, e.g. 16; the CPU kernel
  never needed it, **the GPU kernels of Tasks 10-12 must handle a ragged `d_out`**); the ranks `r`, `r2` are padded up to a
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

## LittleBit in the graph (Task 8)

`build_lora_mm` (src/llama-graph.cpp) finds a LittleBit linear through `model.lb_linears` (keyed by the pointer
`create_tensor` returned for the "weight") and then:

- **B1 factors** (what `tools/convert/lb_to_gguf.py` writes): one `ggml_littlebit(...)` node (12-argument form; the
  residual scales are packed by two CONCAT nodes inside, `ggml_littlebit_packed` is deferred to Task 13). F32
  activations, no activation quantization. The input is cast / made contiguous first if needed.
- **Q1_0 factors** (the legacy form of Cybertiron's converter): the old chain `build_lb_path`, kept as a documented
  slow baseline. It multiplies Q1_0 weights by Q8_0-quantized activations, so it carries int8 activation noise
  (measured 50x to 100x the error of the fused path, see below). The test helper
  `tests/smoke/_assets.py::b1_to_q1_0_gguf` derives such a file from a B1 GGUF (the converter only writes B1).
- `w_s` (the optional per-weight scale) is applied after both forms.
- Biases are not part of `build_lora_mm`: callers add them afterwards, see the bias table.

### Loader

- `create_tensor` accepts `lb_v` of type B1 or Q1_0 only. For B1, all tensors of the linear (signs and the F32
  scale vectors) are placed with the new flag `TENSOR_PIN_CPU`: plain CPU buffer type, ignoring layer offload and
  `--override-tensor`. Reason: `GGML_OP_LITTLEBIT` has a CPU kernel only; weights on the GPU would be copied
  across devices on every call. Verified with `GGML_SCHED_DEBUG=1 llama-debug -ngl 99` on a tiny Llama: the
  embedding, norms, RoPE, flash attention and the output head run on CUDA0, and every `LITTLEBIT` node reads
  its weights in place with a `CPU#...` copy of the activation as input, i.e. the node runs on the CPU backend. Task 10+ must
  drop `TENSOR_PIN_CPU` for a backend that gets a kernel.
- The scale vectors are loaded as `GGML_OP_MUL` operands (they are broadcast-multiplied into the activations), not as
  the MUL_MAT weight their tensor id would suggest.
- **Coverage check** (new, in `llama_model_base::load_tensors`, before the generic "wrong number of tensors"): every
  `*.lb_v` in the file must have been consumed by the architecture, and a `<linear>.bias` next to it must have been
  consumed too, otherwise the load fails naming the tensor (a linear nobody loads would otherwise run without its
  weight; an ignored bias gives wrong output). Tested in `tests/engine/test_fused_graph_families.py`.
- LoRA: adapters are not applied to LittleBit linears (the dense weight never exists). The adapter loader
  (`llama-adapter.cpp`) used to reject such an adapter with "does not exist in base model"; it now skips those
  tensors and logs one warning ("LoRA adapters are not supported on LittleBit linears ..."). The skip is the only place the case
  can happen: `get_weight(w)` in `build_lora_mm` never finds a delta for a LittleBit linear. Tested (warning printed
  once, logits unchanged).

### Hadamard / precision policy

`build_hadamard_input` and `prec_policy` are driven by GGUF metadata (`prism.hadamard.*`, the tensor-extra precision
arrays) naming dense weights `blk.N.<x>.weight`; nothing else enables them. Our converter writes none of these
keys, so for all four families they do not apply to a LittleBit GGUF. A file that did carry such metadata for a
LittleBit linear fails loudly: the Hadamard loader looks the name up with `get_tensor` and throws "prism.hadamard
weight not found" (the tensor is called `...lb_v`); `prec_policy` ignores names it cannot resolve. (Found by reading
`llama-model.cpp`, the Hadamard loader and `llama_prec_policy::load`; not exercised by a test.)

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

Not LittleBit linears: embeddings and the output head (the checkpoint keeps them dense, the converter writes F16),
norms, the DeltaNet conv / `dt_bias` / `A_log`. Merged tensors (`wqkv`, `ssm_beta_alpha_cat`, `wqkv_z`) also go through
`build_lora_mm`; the converter does not write them. Qwen3.5's MTP head is not converted. Anything a GGUF carries that
the builder does not request fails the load (see the coverage check).

### Biases, per family (HF `attention_bias` / `mlp_bias` switches)

| Family | bias on converted linears in HF | engine consumes it? |
|---|---|---|
| Qwen3 | q,k,v,o when `attention_bias` | q/k/v: yes (`create_tensor_qkv` loads `wq_b..`, `build_qkv` adds them). o: **no before Task 8**, now `wo_b` is loaded and `build_attn` adds it |
| Llama | q,k,v,o (`attention_bias`), gate,up,down (`mlp_bias`) | yes, all (`wo_b`, `ffn_*_b`) |
| Gemma 3 | q,k,v,o when `attention_bias` | q/k/v yes; o: **no before Task 8**, now loaded and applied (`build_attn` was called with a literal `NULL`) |
| Qwen3.5 | q,k,v,o when `attention_bias`; DeltaNet projections and MLP have none | q/k/v yes; o: added in Task 8 (`build_layer_attn` adds `wo_b`) |

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
cache), **1e-4 with `-fa off -ctk f32 -ctv f32`**, 1.5e-2 .. 6e-2 with `-ngl 99` (the CUDA half of the graph), against
0.73 .. 1.6 for the slow path. Speed (CPU, informational): the CPU reference kernel generates 10 tokens/s against 89
tokens/s for the Q1_0 chain; the kernel is a scalar reference, tuning is for Tasks 10+.

**Gemma 3 note:** the engine's Gemma 3 builder hard-codes the layer pattern (5 sliding-window layers, then 1 global;
rope base 1e4 / 1e6); the GGUF has no per-layer list. A tiny model with `layer_types=[sliding, full]` is therefore
read as two sliding layers and is wrong (4e-1 logit error for fused and slow path alike, no LittleBit involvement).
The real Gemma 3 pattern is fine. The tests use six layers.

## Known gaps / TODO

- [x] LoRA adapters on LittleBit linears: skipped at adapter load with one warning (not applied; there is no dense weight to add a delta to).
- [x] The Hadamard input transform / precision policy: not applicable (metadata driven), fails loudly if metadata names a LittleBit linear.
- [ ] Split-mode / tensor-split regexes do not match `lb_*` tensor names (moot while the LittleBit tensors are pinned to the CPU).
- [x] `lb_to_gguf.py` silently skipped unknown linears: the engine copy is deleted; `tools/convert/lb_to_gguf.py` validates every linear.
- [ ] GPU kernels for `GGML_OP_LITTLEBIT` (Tasks 10-12): remove `TENSOR_PIN_CPU`, handle ragged `d_out`.
