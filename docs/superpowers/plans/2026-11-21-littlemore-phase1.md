# LittleMore Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Samsung LittleBit / LittleBit-2 checkpoints through a llama.cpp fork with a fused 2-kernel op and a true 1-bit weight type (CUDA + x86 CPU), exposed through a C library and a Python SDK.

**Architecture:** `engine/` is a fork of upstream llama.cpp (cherry-picking only Cybertiron commit `4ce291b`) gaining `GGML_TYPE_B1` and `GGML_OP_LITTLEBIT`. A converter turns Samsung-format checkpoints into GGUF. `littlemore.h` wraps llama.cpp; the Python SDK is ctypes over it. A lean PyTorch reference is the ground truth for every correctness test.

**Tech Stack:** C/C++17, CUDA 12.x (sm_89 local), CMake+Ninja, Python 3.13 (torch, safetensors, numpy, pytest), GitHub Actions.

**Spec:** `docs/specs/2026-11-21-littlemore-design.md` (design starts at "# DESIGN: LittleMore"; background research above it).

## Global Constraints
- Math, per linear: `y = u1⊙(Ub·(mid⊙(Vbᵀ·(v2⊙x)))) + residual`, `mid = v1⊙u2`; `Ub, Vb ∈ {±1}`; residual has the identical form with `_R` tensors.
- Samsung checkpoint format: `<name>.U_packed|V_packed` int32, **+1→bit 0, −1→bit 1, LSB-first, 32 columns per word**; `U_shape`/`V_shape`; scales `u1(1,out) u2(1,r) v1(1,r) v2(1,in)`; `littlebit_config.json`. Embeddings, norms, lm_head stay FP16.
- `GGML_TYPE_B1`: 128 signs per 16-byte block, exactly 1.0 bpw, **bit 1 = +1** (opposite of Samsung's packing; the converter flips it), no scale. Enum value placed after upstream's last type.
- Rank padded to a multiple of 128; padded `mid` entries are 0 (exact).
- Licences: our code MIT; Samsung code (CC BY-NC) is never vendored, reference math is re-implemented from the papers.
- No prebuilt third-party binaries; everything built from source.
- Hard gate: on CUDA, LittleBit 0.55 bpw token generation beats f16/bf16 for Qwen3-0.6B and 1.7B on the RTX 4050 (6 GB).
- Correctness: backend fused op vs CPU reference via `test-backend-ops`; per-layer converter validation; end-to-end perplexity within ~1% of the PyTorch reference.
- Public-friendly comments everywhere: file header (what/why) on every file; doc comment on every public function/kernel/struct (kernels explain math, packed-bit order, scale folding); inline "why" comments for tricks (sign-bit XOR, rank padding, bit order); Python docstrings + type hints; comment intent, not the obvious.
- Ruflo: route each task with `hooks_route`, store each milestone's decisions with `memory_store` (namespace `littlemore`), run independent tasks as swarm agents where files don't overlap.
- Creating the GitHub repo, forking llama.cpp on GitHub, any push, and PyPI publishing require asking the user first.

## Review Focus
1. Rank not a multiple of 128 or 8, and very low rank (r=8): padded results must equal the unpadded reference exactly.
2. Checkpoint with no residual tensors (v1 single path / `--residual False`): converter and op must work with the residual absent.
3. GQA models where k_proj/v_proj rank differs from others (`kv_factor`): per-tensor rank, never a global one.
4. Corrupt or mismatched inputs (shape disagrees with packed words, missing `littlebit_config.json`, duplicate shard keys): fail loudly with a clear message, never write a partial GGUF.
5. Batch sizes at the decode/prefill boundary (n_tokens 1, 8, 9, 512) and a GPU with too little free VRAM: results identical across the switch; auto-fit falls back to CPU layers instead of crashing.

---

## Milestone M0 — Toolchain, engine base, slow path

### Task 1: Toolchain
**Files:** Create `docs/setup-windows.md`, `tools/dev/check_env.py`
**Interfaces:** Produces: `python tools/dev/check_env.py` exits 0 only if nvcc ≥12, cmake, ninja, vulkaninfo, VS Build Tools, and `torch.cuda.is_available()` are all present.
- [ ] **Step 1:** Write `check_env.py` (prints each tool + version, exit 1 on any missing); run it, expect FAIL listing nvcc/cmake/ninja/vulkan/torch.
- [ ] **Step 2:** Install CUDA Toolkit 12.x, CMake, Ninja, Vulkan SDK (ask the user before installing), create `.venv` with `torch` (CUDA build), `safetensors`, `numpy`, `pytest`, `transformers`, `gguf`.
- [ ] **Step 3:** Run `check_env.py`, expect exit 0. Document the exact steps in `docs/setup-windows.md`.
- [ ] **Step 4:** Commit `chore: environment check and Windows setup doc`.

### Task 2: Engine fork + cherry-pick + build
**Files:** Create `engine/` (git submodule path; local clone for now), `docs/engine-changes.md`
**Interfaces:** Produces: `engine/build/bin/llama-cli`, `llama-bench`, `llama-perplexity`, `test-backend-ops`, built with CPU, CUDA (sm_89) and Vulkan as `GGML_BACKEND_DL` plugins; branch `littlemore` containing Cybertiron `4ce291b` only.
- [ ] **Step 1:** Clone upstream llama.cpp at the latest tagged release into `engine/`; branch `littlemore`; record the tag in `docs/engine-changes.md`.
- [ ] **Step 2:** Add Cybertiron/llama.cpp-littlebit as a remote; read the `4ce291b` diff first (no network calls or telemetry); `git cherry-pick 4ce291b`; resolve conflicts. Do NOT take the KV-cache commits.
- [ ] **Step 3:** Configure `cmake -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON` and build. Expected: all targets build; `llama-cli --version` runs.
- [ ] **Step 4:** Commit in the submodule; commit the pointer in the main repo.

### Task 3: Lean PyTorch reference
**Files:** Create `tools/dev/reference.py`, `tests/dev/test_reference.py`
**Interfaces:** Produces: `unpack_signs(packed: Tensor[int32], shape: tuple[int,int]) -> Tensor[±1 float]`; `class LBLinear(nn.Module)` with `forward(x)` computing the factorized chain **without building W**; `load_checkpoint(dir: Path) -> dict[str, LBLinearParams]`; `build_model(base_hf_id: str, ckpt_dir: Path) -> nn.Module` swapping each target `nn.Linear` for `LBLinear`.
- [ ] **Step 1:** Tests: `test_unpack_matches_spec` (hand-built word with known bits: bit0=0 → +1, bit1=1 → −1, LSB-first); `test_forward_matches_dense` (random factors; `LBLinear(x)` equals `x @ W_dense.T` with `W_dense` built from the formula, rtol 1e-5); `test_residual_optional` (residual params absent works and equals primary only); `test_padded_rank_exact`.
- [ ] **Step 2:** Run, expect FAIL (missing module).
- [ ] **Step 3:** Implement per the signatures; docstrings explain the bit order and why W is never materialized.
- [ ] **Step 4:** Run, expect PASS. Commit.

### Task 4: Init-only LittleBit-2 maker
**Files:** Create `tools/dev/make_lb2_init.py`, `tests/dev/test_make_lb2_init.py`
**Interfaces:** Consumes: `unpack_signs` (Task 3). Produces: `joint_itq(Z: Tensor, n_iter: int = 50, seed: int = 0) -> Tensor` (orthogonal R); `dual_svid(U: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]` returning `(Ub, Vb, h, g, l)`; `rank_for_bits(d_out: int, d_in: int, bpw: float, residual: bool = True) -> int` (`r = ⌊(B·N − 32(d_in+d_out)) / (2(d_in+d_out+16))⌋`, rounded down to a multiple of 8, minimum 8); `make_checkpoint(hf_id: str, out_dir: Path, bpw: float, use_itq: bool = True) -> None` writing Samsung-format safetensors + `littlebit_config.json`.
- [ ] **Step 1:** Tests: `test_joint_itq_orthogonal` (RᵀR≈I); `test_joint_itq_reduces_distortion` (mean λ = 1−(1/r)(‖u‖₁/‖u‖₂)² on spiky SVD factors drops vs. no rotation); `test_rotation_preserves_product` (ÛR(V̂R)ᵀ = ÛV̂ᵀ); `test_rank_for_bits_never_exceeds_budget` (actual bpw ≤ target for 4096×4096 @0.55, 4096×11008 @0.1); `test_roundtrip_pack_unpack`.
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Implement (algorithm from design §2: √Σ split, Z=[Û;V̂] concatenated, random orthogonal R, loop `B=sign(ZR); ΦΩΨᵀ=SVD(BᵀZ); R=ΨΦᵀ`; Dual-SVID rank-1 SVD of |Ũ|,|Ṽ|; residual path on W−Ŵ_pri). Pack with Samsung's bit order. Use exact `torch.linalg.svd`.
- [ ] **Step 4:** Run, expect PASS. Commit.

### Task 5: M0 smoke test on the slow path
**Files:** Create `tests/smoke/test_m0_slow_path.py`
- [ ] **Step 1:** Generate init-only Qwen3-0.6B at 0.55 bpw with Task 4; convert with Cybertiron's `tools/littlebit/lb_to_gguf.py` (Q1_0 path); run `llama-cli -m out.gguf -p "Hello" -n 8 -ngl 99`.
- [ ] **Step 2:** Test asserts: command exits 0, 8 tokens produced, and logits for a fixed prompt match `reference.py` within max-abs-diff 1e-2 (record the measured value in the test as the baseline).
- [ ] **Step 3:** Run, expect PASS. Record milestone decisions with Ruflo `memory_store` (key `m0-done`). Commit.

---

## Milestone M1 — B1 type, converter, CPU fused op

### Task 6: `GGML_TYPE_B1`
**Files:** Modify `engine/ggml/include/ggml.h`, `engine/ggml/src/ggml-common.h`, `ggml-quants.{c,h}`, `ggml.c` (type traits); Test `engine/tests/test-b1.cpp`
**Interfaces:** Produces: `block_b1 { uint8_t qs[16]; }`; `quantize_row_b1_ref(const float*, block_b1*, int64_t k)`; `dequantize_row_b1(const block_b1*, float*, int64_t k)` (bit 1 → +1.0f, bit 0 → −1.0f).
- [ ] **Step 1:** Test: round-trip of random ±1 vectors (k = 128·n) is exact; a known byte pattern decodes to the expected signs; type size is exactly 16 bytes per 128 elements.
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Implement and register type traits (CPU `to_float` only; no `vec_dot`). The header comment documents the bit order and the rebase hazard of the enum value.
- [ ] **Step 4:** PASS. Commit.

### Task 7: Fused op `GGML_OP_LITTLEBIT` — CPU reference
**Files:** Modify `engine/ggml/include/ggml.h`, `ggml.c`, `ggml-cpu/ops.{h,cpp}`, `ggml-cpu/ggml-cpu.c`; Test `engine/tests/test-backend-ops.cpp` (new case)
**Interfaces:** Produces: `struct ggml_tensor * ggml_littlebit(ctx, x, V, U, s_v2, s_mid, s_u1, Vr, Ur, s_v2r, s_midr, s_u1r)` (residual args may be NULL; V, U are `GGML_TYPE_B1`, ranks padded to 128; x F32 `[d_in, n]`, result F32 `[d_out, n]`).
- [ ] **Step 1:** Add `test-backend-ops` case `LITTLEBIT` over ranks {8, 128, 136, 546→padded 640}, d_in/d_out multiples of 128, with/without residual, n_tokens {1,4,8,9,32,512}; each backend is compared to CPU. The CPU result is itself checked against an independent NumPy computation in the harness.
- [ ] **Step 2:** Run `test-backend-ops -o LITTLEBIT`, expect FAIL (op unknown).
- [ ] **Step 3:** Implement op creation + scalar CPU kernel computing the V-stage into a scratch buffer, then the U-stage (documented as the reference two-stage structure the GPU kernels mirror).
- [ ] **Step 4:** Run, expect PASS on CPU. Commit.

### Task 8: Graph hook → fused op, gap fixes
**Files:** Modify `engine/src/llama-graph.cpp` (`build_lb_path` / `build_lora_mm`), `engine/src/llama-model-loader.cpp`
**Interfaces:** Consumes: `ggml_littlebit`. Produces: LittleBit layers built with one op; a logged warning when a LoRA adapter targets a LittleBit linear; tensors typed `GGML_TYPE_B1`.
- [ ] **Step 1:** Test (`tests/smoke/test_graph_fused.py`): logits from the fused graph equal the Task 5 slow-path logits within 1e-3 on init-only Qwen3-0.6B.
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Replace the 11-op chain with one `ggml_littlebit` call; add the LoRA-bypass warning; determine (by reading the code) whether `build_hadamard_input` applies to each of the 4 families and apply it in the LittleBit branch if so. Record the finding in `docs/engine-changes.md`.
- [ ] **Step 4:** PASS.
- [ ] **Step 5:** For Qwen3, Llama, Gemma 3 and Qwen3.5/3.8, list every linear and confirm it goes through `build_lora_mm`; patch any that don't (record the list in `docs/engine-changes.md`). Commit.

### Task 9: Converter `littlemore convert`
**Files:** Create `tools/convert/lb_to_gguf.py`, `tools/convert/validate.py`, `tests/convert/test_convert.py`
**Interfaces:** Consumes: `unpack_signs`, `LBLinear`. Produces: `convert(ckpt_dir: Path, out: Path, head_type: str = "f16", validate: bool = True) -> None`; `validate_layer(ckpt_params, gguf_tensors, n_samples: int = 4) -> float` (max relative error); `class ConversionError(Exception)`.
- [ ] **Step 1:** Tests on tiny synthetic checkpoints from Task 4 (hidden size 256): `test_each_layer_matches_reference`; `test_no_residual_checkpoint`; `test_rank_not_multiple_of_128_pads_exactly`; `test_kv_factor_per_tensor_rank`; `test_missing_config_raises`; `test_shape_mismatch_raises`; `test_duplicate_shard_key_raises`; `test_no_partial_file_on_failure` (output absent after a `ConversionError`).
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Start from Cybertiron's `lb_to_gguf.py` (MIT; keep attribution): unpack Samsung bits, **flip polarity** to B1 (bit 1 = +1), pad rank to 128, write tensor names `lb_v/lb_u/lb_s_v2/lb_s_mid/lb_s_u1` (+`r` suffix for the residual), keep the row/column permutation probing through llama.cpp's `convert_hf_to_gguf`, fail on silently skipped linears, write to a temp file and rename only after validation passes. Comment every bit manipulation with the reason.
- [ ] **Step 4:** Run, expect PASS.
- [ ] **Step 5:** Add the other families one at a time (Llama, Gemma 3, Qwen3.5/3.8), one parametrized test each on a tiny synthetic model. Commit.

---

## Milestone M2 — CUDA + AVX2, speed gates

### Task 10: CUDA fused op, decode path
**Files:** Create `engine/ggml/src/ggml-cuda/littlebit.cu`, `littlebit.cuh`; Modify `ggml-cuda.cu` (`supports_op`, dispatch)
**Interfaces:** Consumes: `GGML_OP_LITTLEBIT`. Produces: `void ggml_cuda_op_littlebit(ggml_backend_cuda_context & ctx, ggml_tensor * dst)` for n_tokens ≤ 8, using two launches (V-stage; U-stage with `u1` epilogue and the residual summed).
- [ ] **Step 1:** Run `test-backend-ops -o LITTLEBIT -b CUDA0` for n_tokens ≤ 8 (cases from Task 7), tolerance NMSE ≤ 1e-6 vs CPU; expect FAIL (unsupported).
- [ ] **Step 2:** Implement: the V-stage loads `x⊙v2` into shared memory, warps dot it against packed sign rows using XOR on the float sign bit, `mid` is applied in the epilogue, and the concatenated primary+residual latent `t` goes to scratch; the U-stage dots `t` against U/Ur sign bits with the `u1`/`u1r` epilogue. The file header explains the math and layout.
- [ ] **Step 3:** PASS on all shapes including non-multiple ranks. Commit.

### Task 11: CUDA prefill path
**Files:** Modify `littlebit.cu`
- [ ] **Step 1:** Run the CUDA LITTLEBIT test with n_tokens {9, 32, 512}; expect FAIL.
- [ ] **Step 2:** For n_tokens > 8, decode B1 tiles to f16 and use a tiled/mma GEMM for both stages. Correct first.
- [ ] **Step 3:** PASS; assert results at n_tokens 8 vs 9 agree (NMSE ≤ 1e-6). Commit.

### Task 12: AVX2 (and AVX-512 if present) CPU kernels
**Files:** Create `engine/ggml/src/ggml-cpu/littlebit-x86.cpp`
- [ ] **Step 1:** Run the CPU LITTLEBIT test with AVX2 forced against the scalar reference; expect FAIL (not wired).
- [ ] **Step 2:** Implement sign flip via XOR on the float sign bit with `_mm256` ops; runtime dispatch by CPU features; keep the scalar fallback.
- [ ] **Step 3:** PASS (NMSE ≤ 1e-6). Commit.

### Task 13: Head/embedding option, benchmark harness, speed gates
**Files:** Create `bench/run_bench.py`, `bench/README.md`, `tests/bench/test_check_gate.py`; Modify `tools/convert/lb_to_gguf.py` (`--head-type`)
**Interfaces:** Produces: `run_bench(model_gguf: Path, backend: str, n_gpu_layers: int) -> dict` with keys `pp512`, `tg128`, `vram_mb`; `check_gate(results: dict) -> bool`.
- [ ] **Step 1:** Test: `check_gate` passes only when LittleBit tg128 > bf16 tg128 for both 0.6B and 1.7B (mocked numbers, pure logic); run, expect FAIL.
- [ ] **Step 2:** Implement `--head-type q8_0|q6_k|f16` (report the perplexity delta) and the harness wrapping `llama-bench`.
- [ ] **Step 3:** Run for Qwen3-0.6B/1.7B init-only at 0.55 and 0.1 bpw versus bf16 and Q4_K_M, with f16 and q8_0 head; write `bench/results/phase1.md`. **Gate:** CUDA LittleBit 0.55 tg128 > bf16 on both. If missed, implement the deferred QKV / gate-up V-stage merge and re-measure. Store results with Ruflo `memory_store`. Commit.

### Task 14: Perplexity parity
**Files:** Create `tests/e2e/test_perplexity_parity.py`
- [ ] **Step 1:** On init-only Qwen3-0.6B, compute WikiText-2 perplexity with `reference.py` (CPU, factorized) and with `llama-perplexity` (CUDA); assert relative difference ≤ 1%.
- [ ] **Step 2:** Run; expect PASS (investigate any mismatch before continuing). Commit.

---

## Milestone M3 — C API, SDK, CI, docs

### Task 15: C API `littlemore.h`
**Files:** Create `include/littlemore.h`, `src/littlemore.cpp`, `CMakeLists.txt`, `tests/c/test_api.c`
**Interfaces:** Produces (ABI-stable C, versioned by `LM_API_VERSION`): opaque `lm_model`, `lm_session`; `lm_status lm_model_load(const char * path, const lm_model_options * opts, lm_model ** out)`; `lm_status lm_session_new(lm_model *, const lm_session_options *, lm_session ** out)`; `lm_status lm_chat(lm_session *, const lm_message * msgs, size_t n, const lm_sampling * s, lm_token_cb cb, void * user)`; `lm_generate(...)`; `lm_tokenize` / `lm_detokenize`; `lm_backends_info(char * buf, size_t cap)`; `const char * lm_last_error(void)`; `lm_*_free`. `n_gpu_layers = -1` means auto-fit to free VRAM; ggml backends load as runtime plugins (`GGML_BACKEND_DL`).
- [ ] **Step 1:** C test: load a tiny GGUF, chat with a callback, check ≥ 1 token streamed, a bad path returns an error with a non-empty message, backend info lists CPU; run, expect FAIL.
- [ ] **Step 2:** Implement over `llama.h`; auto-fit estimates per-layer size and falls back to CPU layers when VRAM is short (test with a forced tiny VRAM budget). Doc comment on every public symbol.
- [ ] **Step 3:** PASS. Commit.

### Task 16: Python SDK + CLI
**Files:** Create `python/littlemore/{__init__,_lib,model,cli}.py`, `python/pyproject.toml`, `tests/python/test_sdk.py`
**Interfaces:** Produces: `class Model(path: str | Path, n_gpu_layers: int | str = "auto")` with `.chat(messages: list[dict], stream: bool = False)`, `.generate(prompt: str, stream: bool = False)`, `.info() -> dict`; CLI `littlemore convert|run|bench`.
- [ ] **Step 1:** Tests: `test_generate_stream_yields_tokens`, `test_chat_applies_template`, `test_info_lists_backends`, `test_bad_path_raises_littlemore_error`, `test_cli_convert_then_run` (tiny synthetic); run, expect FAIL.
- [ ] **Step 2:** Implement ctypes bindings (no compiler needed at install) and the wheel config bundling the backend plugins.
- [ ] **Step 3:** PASS. Commit.

### Task 17: Real-weights end-to-end
**Files:** Create `tests/e2e/test_community_8b.py` (marked slow/manual), `docs/quickstart.md`
- [ ] **Step 1:** Download `LittleBitLLM/littlebit-qwen3-8b-0.55bpw` (ask the user first; it is large), convert, run `littlemore run` with auto-fit on the 4050; assert it loads within 6 GB and generates 32 tokens; report perplexity against the HF card (24.48) as information, not a gate.
- [ ] **Step 2:** Write `docs/quickstart.md` (install, convert, run, Python example). Commit.

### Task 18: CI and public-repo readiness
**Files:** Create `.github/workflows/ci.yml`, `README.md`, `LICENSE` (MIT), `docs/licensing.md`
- [ ] **Step 1:** Workflow: Windows + Linux x64 builds (CUDA compile-only), CPU tests, converter tests, wheel build, `test-backend-ops` on CPU.
- [ ] **Step 2:** README explains the project, the non-commercial checkpoint caveat, build/test, and layout; `docs/licensing.md` covers the MIT / CC BY-NC split.
- [ ] **Step 3:** **Ask the user** before creating the GitHub repo, forking llama.cpp there, or pushing. Then create/push and confirm CI is green. Commit.

### Task 19: Public-comment review (also run at the end of M0, M1 and M2)
- [ ] **Step 1:** A fresh reviewer subagent audits every source file against the Global Constraints comment rules (headers, doc comments, "why" comments, docstrings/type hints) and lists gaps.
- [ ] **Step 2:** Fix the gaps; re-run all tests.
- [ ] **Step 3:** Store a Phase 1 summary with Ruflo `memory_store`; commit.
