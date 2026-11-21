# LittleMore: LittleBit-2 runtime and developer toolkit (research + design)

## Context
Samsung's LittleBit-2 (ICML 2026) compresses LLMs to 0.1–1 bit per weight, so very large models fit in very little memory. Nobody ships a runtime that delivers that in practice:
- Samsung's code expands the weights back to full size.
- The community llama.cpp ports run 2–2.5× slower than uncompressed models.

The user wants **LittleMore**: a runtime plus developer tools for running LittleBit models locally, without hassle, on desktop and mobile. It will live in `c:\Vibe Coding\LittleMore` (currently empty) and become a public GitHub repo.

The research comes first (sections 1–5c). The design follows (from "# DESIGN").

## Sources
- LittleBit-2 paper: arXiv 2603.00042 (v2, May 2026). Banseok Lee and Youngmin Kim, Samsung Research. ICML 2026. License CC BY-NC-ND 4.0.
- LittleBit (v1) paper: arXiv 2506.13771 (v5), NeurIPS 2025.
- Code for both: github.com/SamsungLabs/LittleBit. LittleBit-2 is the `--use_itq True` flag. License CC BY-NC 4.0, so **non-commercial only**.
- Samsung Research blog post on LittleBit-2.

## 1. Core idea (LittleBit, v1)
Each linear weight `W (d_out×d_in)` is replaced by a **binarized low-rank factorization** with three FP16 scale vectors:

```
Ŵ = diag(h) · sign(U) · diag(ℓ) · sign(V)ᵀ · diag(g)        U: d_out×r, V: d_in×r
Ŵ_total = Ŵ_pri + Ŵ_res      (a residual path with the same structure fits W − Ŵ_pri)
```

- The bit budget is set by the rank r, not by the matrix size. That is how it gets **below 1 bit per weight** (0.1 to 1.0 BPW).
- Bits for the residual (two-path) setup: `M = 2r(d_in+d_out+16) + 32(d_in+d_out)`. Rank for a budget B: `r = ⌊(B·N − 32(d_in+d_out)) / (2(d_in+d_out+16))⌋`, where N = d_in·d_out. In code, r is rounded down to a multiple of 8, with a minimum of 8.
- **The weight is never materialized at inference.** The forward pass is `Y = ((((X ⊙ g) · sign(V)) ⊙ ℓ) · sign(U)ᵀ) ⊙ h`, which is two binary matmuls plus element-wise scaling.
- **Dual-SVID init:**
  1. Run a truncated SVD and split √Σ into each side: `Û = U_r√Σ`, `V̂ = V_r√Σ`.
  2. Take the signs as the binary factors.
  3. Take a rank-1 SVD of `|Û|` to get `(h, ℓ_u)`, and of `|V̂|` to get `(g, ℓ_v)`. Set `ℓ = ℓ_u ⊙ ℓ_v`.
  4. Fit the residual path the same way on `W − Ŵ_pri`.
- **QAT (required, this is the expensive part):**
  - The FP16 teacher distills into the student.
  - Loss = KL(teacher‖student logits, no temperature) + 10 × Σ MSE(hidden_states[1:]).
  - Sign uses SmoothSign: forward is `sign` (with 0 mapped to +1), backward is `100·(1−tanh²(100x))`.
  - Data is C4 + WikiText-2 at sequence length 2048. Adam, lr 4e-5, 5 epochs, cosine schedule with 2% warmup, batch 4 per GPU. The paper uses DeepSpeed ZeRO-3 on 4 GPUs.
  - Embeddings, LayerNorm and lm_head stay FP16.
  - For GQA models, `kv_factor` scales the k_proj/v_proj rank.

## 2. What LittleBit-2 adds
The architecture and inference cost are unchanged. **Only the initialization changes.**

- **Diagnosis ("latent geometry misalignment"):**
  - SVD factors are spiky and axis-aligned. For Llama-2 7B L15 q_proj, kurtosis is about 16.8.
  - So `sign()` throws away a lot of energy.
  - Per-row distortion is `λ(u) = 1 − (1/r)(‖u‖₁/‖u‖₂)²`. The SVD init has λ_max ≈ 0.88.
- **Internal Latent Rotation:**
  - Since `ÛV̂ᵀ = (ÛR)(V̂R)ᵀ` for any orthogonal R, the rotation is absorbed into the factors offline.
  - Unlike QuaRot/SpinQuant, this needs no online Hadamard transform.
  - A random R gives a Gaussian-like spread with E[λ] = 1 − 2/π ≈ 0.36.
- **Joint-ITQ (Gong et al.'s ITQ applied to both factors together):**
  - Objective: `min ‖B − ZR‖²_F`, where `Z = [Û; V̂]`.
  - Start from a random orthogonal R, then repeat T = 50 times:
    - `B = sign(ZR)`
    - `ΦΩΨᵀ = SVD(BᵀZ)`
    - `R = ΨΦᵀ`
  - Then set `U ← ÛR`, `V ← V̂R` and run Dual-SVID on the rotated factors.
  - Mean λ drops to 0.30. This adds about 3 s per 4096² layer.
  - The repo's `_compute_itq_rotation` is about 20 lines.
- **Theory ("spectral break-even"):**
  - Singular values follow `σ_k ∝ k^−γ`.
  - A binary factorization at 16× the rank beats a tiny-rank FP16 factorization when the tail energy it recovers exceeds the binarization cost Λ.
  - The break-even γ* moves 0.36 → 0.41 (random rotation) → 0.51 (Joint-ITQ).
  - Llama-2 layers have a median γ of about 0.27, so most layers favor the binary approach.
- **Training effect:** QAT converges faster, with about 3.8× fewer sign flips early on.

## 3. Results (WikiText-2 PPL / 5-task zero-shot average)
| Model | FP16 | OneBit 1.0 | LittleBit 1.0 | **LB-2 1.0** | LB 0.55 | **LB-2 0.55** | LB 0.1 | **LB-2 0.1** |
|---|---|---|---|---|---|---|---|---|
| Llama-3 8B | 6.10 / 72.97 | 13.09 / 52.23 | 16.30 / 47.11 | **11.53 / 57.33** | 18.47 / 44.59 | **14.01 / 51.33** | 26.11 / 41.80 | **23.74 / 41.64** |
| Llama-2 7B | 5.47 / 68.88 | 8.36 / 54.72 | 9.08 / 52.81 | **8.27 / 55.79** | 10.47 / 48.49 | **9.65 / 50.97** | 15.92 / 41.33 | **14.70 / 42.06** |
| Llama-2 13B | 4.88 / 71.87 | 7.41 / 58.47 | 8.18 / 51.49 | **7.37 / 60.53** | 9.24 / 51.76 | **8.56 / 55.64** | 15.09 / 40.99 | **13.94 / 42.18** |

- Gemma-3 27B at 0.1 BPW: PPL 16.38 (FP16 4.58). Tiny-rank FP16 at the same budget gives 37.53.
- EfficientQAT at 2 bits is still better: Llama-3 8B PPL 9.80, 2.0 GB body.
- Memory, model body only:
  - Llama-2 7B: 13 GB → 0.8 GB at 1.0 BPW, 0.1 GB at 0.1 BPW. The total is 0.6 GB because embeddings and head stay FP16.
  - Llama-2 70B: 138 GB → 1.98 GB at 0.1 BPW.
- Speed, from the v1 paper:
  - Custom 1-bit GEMV CUDA kernel: 11.6× faster than cuBLAS FP16 on a 70B MLP layer at 0.1 BPW (A100).
  - Llama-2 7B decoding: 82.6 → 203 tok/s end to end.

## 4. Caveats found while reading the code and papers
1. **No pretrained checkpoints are published.** To use it, you have to run QAT yourself: a full FP16 teacher plus multi-GPU training over many hours.
2. **The public repo has no 1-bit inference path.** Checkpoints are saved bit-packed (int32, 32 signs per word), but loading unpacks them to dense bf16 ±1 and runs plain `@` matmul. So the released code **does not actually save runtime memory**, and the speedup kernel is not released.
3. Quality at 0.1 BPW is weak: PPL 15 to 24, and the paper notes heavy hallucination. At 0.55 to 1.0 BPW it is usable but well below FP16.
4. The license is non-commercial.
5. Small repo bugs: `split_dim_used` references an unset attribute, `save_artifacts` is defined twice, and the eff-bit accounting undercounts the latent scale.

## 5b. Existing runtimes (checked 2026-11-21)
- **Samsung (official):** no runtime and no kernel.
  - Issue #18: the speed numbers came from an internal custom CUDA kernel.
  - Issue #1: the trained checkpoints can't be released "due to internal policies."
  - The repo bug that flipped about 50% of signs when reloading packed checkpoints was fixed in PR #22 (Oct 2026).
- **Cybertiron/llama.cpp-littlebit:** a llama.cpp fork, MIT.
  - Stores the binary factors as ggml `Q1_0` and runs the factor chain without expanding weights to dense (per its README).
  - CUDA + CPU. Converter at `tools/littlebit/lb_to_gguf.py`.
  - Self-reported on an RTX 3090: Qwen3-0.6B PPL 45.6 (bf16 18.5), 0.36 GiB, 149 vs 351 tok/s. So it is **slower than bf16**.
  - Release is 1 day old, 0 stars. The prebuilt Windows binary zip has no checksums. The README's "27B included" claim is overstated.
- **GeniusDevam/littlebit-gguf:** a patch on pinned llama.cpp, MIT, source only.
  - Reuses the `Q1_0` kernel and folds `h` and `ℓ` into the block scales. CPU only.
  - Benchmarks use untrained synthetic checkpoints: Qwen2.5-0.5B at 0.1 BPW gives 30 gen tok/s on an i7-8550U, vs 66 for dense Q1_0. Also slower.
  - Its sister repo **littlebit.cpp** has docs only, no code.
- **littlebit-vllm (PyPI):** not verified (PyPI blocked the fetch).
  - Used by the HF model `dellathefella/Qwen3.8-27B-LittleBit-2-0.55bpw`. That model is initialization-only with no QAT, and its factors are stored as bf16, not bit-packed.
- **mgiachetti/little-bit:** a PyTorch training reimplementation, MIT. No packed inference.
  - Qwen3-0.6B at 0.55 BPW reaches PPL 30.6 vs 21.0 after about 72 hours on one L40S.
- **Checkpoints:** none official.
  - Community `LittleBitLLM/littlebit-qwen3-8b-0.55bpw` is an early checkpoint at 0.5 epoch: PPL 24.5 vs 9.7.
  - `Haverbex/Qwen3.8-27B-LittleBit-*` cards say task performance was not preserved.
- **Conclusion:** no one has a fast, real 1-bit runtime yet. The existing ones fit models into small memory but run slower than FP16/bf16, because they chain generic Q1_0 matmuls instead of a fused low-rank binary kernel. The 11.6× kernel has never been published.

## 5c. Cybertiron/llama.cpp-littlebit source review
Based on the verbatim diff `4783b11...littlebit` and the full `tools/littlebit/lb_to_gguf.py`.

**LittleBit runtime: correct and small.**
- `build_lb_path` in `src/llama-graph.cpp` is `mul(v2) → mul_mat(Q1_0 V) → mul(v1⊙u2) → mul_mat(Q1_0 U) → mul(u1)`. The residual path is the same chain, joined with `ggml_add`.
- It hooks into `build_lora_mm` and returns early. That adds 11 ggml ops per linear layer and **no new kernels**.
- Detection is per tensor: a 2-D `blk.N.<x>.lb_v` with no `.weight`. Tensor names are `lb_v/lb_u/lb_s_v2/lb_s_mid/lb_s_u1`, plus an `r` suffix for the residual. No new metadata keys are added.
- Tested only on Qwen3-0.6B and qwen35 (first 4 layers). Other `build_lora_mm` architectures are probably eligible. MoE (3-D) is not supported.

**Converter: correct bit handling.**
- Unpacks LSB-first, with bit0 meaning +1.
- Q1_0 encodes bit1 as +1 with `d = 1.0`, so 1.125 bpw. That is 12.5% overhead.
- The rank is padded to a multiple of 128. This is exact, because the padded `mid` entries are 0.
- Row/column index probes go through the real `convert_hf_to_gguf` class, which handles the RoPE Q/K permutation and GQA.
- There is no numeric check of the GGUF output against the dense product.
- It is brittle: it depends on converter internals pinned to 4783b11.

**Bugs and risks:**
- LittleBit linears silently skip the LoRA adapter loop, `build_hadamard_input` and `prec_policy`. Hadamard handling may matter for qwen3/qwen35; this is unverified.
- Multi-GPU split regexes don't match `lb_*`.
- Converter: a linear can be silently dropped, and shard keys are last-wins.

**Why it's slow:** each linear launches about 11 small kernels with F32 intermediates, which makes it launch-bound. Example for a 5120² layer at r = 1024 on a 3090:
- bf16 reads ~52 MB, about 56 µs.
- LittleBit reads ~3 MB, about 2 µs, but the launches cost ~35–55 µs.

**Fix (the gap worth filling):** a fused op with **2 launches per linear**.
- Kernel 1: `v2` in the prologue, ±1 bit expansion of V, `mid` in the epilogue.
- Kernel 2: U with `u1` in the epilogue, summing the residual into the same output.
- Also needed:
  - a pure 1-bit type without the fp16 `d`
  - optional int8 activations with dp4a
  - CUDA-graph capture
  - `supports_op` for CUDA and CPU

**Branch scope:** 22 files changed, more than the 4 the README says.
- It also adds non-upstream KV-cache types (`GGML_TYPE_Q3_0 = 44`, `Q2_0S = 45`, KVarN ported from Anbeeld/beellama.cpp, license unchecked). These are broken on 0.6B: PPL 2951 and 34905.
- `FUNDING.yml` was added. Commits are co-authored by "claude".

**Trust:** no network, eval, subprocess, obfuscation or telemetry in the converter. Build from source. The prebuilt binaries were not inspected.

**As a base:** the LittleBit commit `4ce291b` alone is a clean, reusable starting point. Cherry-pick it onto upstream llama.cpp and leave out the KV-cache commits.

---

# DESIGN: LittleMore

## Intent and why
LittleMore is a toolkit developers use to **run AI models locally, on desktop and mobile, with no hassle**. It builds on the LittleBit-2 promise: models compressed below 1 bit per weight, so large models fit on ordinary laptops and phones.

The gap it fills: today's options either don't save memory at runtime (Samsung's code) or are slower than the uncompressed model (the community llama.cpp ports, about 11 tiny kernels per layer).

LittleMore delivers four things:
- small memory
- faster-than-uncompressed speed
- automatic use of whatever hardware is present
- a simple library API, so developers embed models in their own apps

Size estimates (embeddings and head at 16-bit):
- Qwen3-8B at 0.55 bits: about 3 GB
- ~27B at 0.55: about 4.5 GB
- 70B at 0.1: about 4 GB

## Users and what "no hassle" means
- **Who:** app developers embedding a local model through a **C library** or the **Python SDK**.
- **Install:**
  - `pip install littlemore` (prebuilt wheels with nothing to compile)
  - or download a prebuilt C library package for Windows, Linux, macOS, Android or iOS
- **Hardware:** the library detects the available hardware at runtime. One package covers CUDA, Vulkan, Metal or CPU, and it automatically fits as many layers as possible into the GPU's free memory.
- **Models:** developers bring a LittleBit / LittleBit-2 checkpoint in Samsung's format and run one command, `littlemore convert <ckpt_dir> -o model.gguf`. The command validates the result numerically. **We do not host models** (user's choice).

## Goals
1. **Engine:** a fused LittleBit operation plus a true 1-bit weight format inside llama.cpp, with fast kernels for:
   - **CUDA** (NVIDIA)
   - **x86 CPU** (AVX2, optional AVX-512)
   - **Vulkan** (AMD, Intel and NVIDIA desktop GPUs, and Android GPUs)
   - **ARM CPU** (NEON, for phones and ARM PCs)
   - **Metal** (Apple Silicon Macs, iPhone and iPad)
2. **C library `littlemore.h`:** a small, stable, high-level API.
   - Functions: load, chat with template, generate with streaming callback, tokenize, backend/device info, free.
   - Full llama.cpp `llama.h` access underneath for power users.
3. **Python SDK:** `littlemore` on PyPI (publishing gets confirmed when we reach it).
   - API: `Model(path)`, `.generate(..., stream=True)`, `.chat(messages)`
   - CLI: `littlemore convert`, `littlemore run`
4. **Mobile packages:** Android (`.so` per ABI, packaged as an AAR with a C header) and iOS (XCFramework). App developers call the C API directly.
5. **Model families at launch:** Qwen3 dense, Llama 2/3, Gemma 3, Qwen3.5/3.8 hybrid.
6. **Correct:**
   - every backend's fused op matches the CPU reference within thresholds (llama.cpp `test-backend-ops`)
   - conversions are validated layer by layer
   - end-to-end perplexity is within about 1% of our PyTorch reference
7. **Fast:**
   - **Hard gate:** on each fast backend, LittleBit at 0.55 bits beats that backend's f16/bf16 speed on token generation for Qwen3-0.6B and 1.7B. On CUDA this is measured on the 4050.
   - **Target:** on par with Q4_K_M.

## Non-goals
- Training, QAT or compression of models. Developers bring trained checkpoints (user chose "convert only").
- Hosting or curating models.
- Kotlin/Swift wrappers. These are deferred to a later phase; mobile launches with the C API only.
- Node/JS SDK and a GUI app.
- MoE models, multi-GPU splitting, upstreaming to llama.cpp.
- Cybertiron's experimental KV-cache formats (q3_K/q2_K/KVarN).

## Anti-goals
- **Silently wrong output.** Every conversion and every kernel is checked against a reference, and mismatches fail loudly.
- **Hassle:** needing to compile, choose backends or tune GPU layers by hand.
- **Slower than the uncompressed model** on a backend we call "fast".
- **Prebuilt third-party binaries.** Everything is built from source in our CI.

## Constraints
- **Test hardware:** only the user's laptop (Windows 11, RTX 4050 Laptop 6 GB sm_89, i7-13620H, 16 GB RAM). Coverage:
  - CUDA, x86 CPU, the C API, Python and the converter are fully testable locally.
  - Vulkan runs on the 4050, so it can be checked for correctness locally.
  - **ARM NEON and Metal are tested on GitHub-hosted runners** (ubuntu-24.04-arm, macos arm64), which are free for public repos.
  - Android and iOS packages are built in CI. On-device speed stays unproven until someone with a device runs the bundled bench tool. That gap is documented.
- **Repo:** a **public GitHub repo**, MIT licence. Creating the repo and every push get confirmed with the user at the time. The user's GitHub account name will be asked then.
- **Licences:**
  - llama.cpp and Cybertiron's LittleBit commit are MIT.
  - Samsung's code is CC BY-NC. We never vendor it, and our reference math is reimplemented from the papers.
  - LittleBit checkpoints are non-commercial. The docs say so.
- **Toolchain to install locally:** CUDA Toolkit 12.x, CMake, Ninja, Vulkan SDK, Python venv (torch, safetensors, numpy). VS 2022 Build Tools are already present.

## Approaches (decided)
- **A. Extend llama.cpp — CHOSEN by user.** We get the tokenizer, chat templates, sampling, KV cache, GGUF, model families, runtime backend loading, and existing CUDA/Vulkan/Metal/CPU/Android/iOS build support.
- B. Standalone engine. Rejected: months of rebuilding basics, and every platform port done by hand.
- C. PyTorch + CUDA extension. Rejected: not a runtime, too heavy on memory, and no mobile.

**Build order — CHOSEN by user: desktop first.**
1. Engine + CUDA + x86 CPU + C API + Python SDK + converter for all 4 families.
2. Vulkan + ARM NEON + Android packages.
3. Metal + macOS/iOS packages.

## The design

### Repo layout (my call)
- `LittleMore/` (public repo):
  - `engine/`: a git submodule. It points to our GitHub fork of llama.cpp, branch `littlemore`, pinned to a recent upstream release, holding all engine commits.
  - `include/littlemore.h` and `src/`: the high-level C API.
  - `python/`: the SDK (ctypes over `littlemore.h`, so installing needs no compiler) and the CLI.
  - `tools/convert/`: the converter.
  - `tools/dev/`: dev-only tools, namely the init-only checkpoint maker and the PyTorch reference.
  - `tests/`, `bench/`, `docs/` (this design goes into `docs/specs/`).
  - `.github/workflows/`: CI.

### 1. Engine base
- Fork upstream llama.cpp and cherry-pick only Cybertiron's `4ce291b`. It provides the `build_lb_path` hook in `build_lora_mm`, `lb_*` tensor detection, and the converter seed.
- Fix that commit's gaps:
  - Warn when LoRA adapters are bypassed on LittleBit layers.
  - Handle `build_hadamard_input` for the 4 families.
  - Make the converter fail on silently skipped linears and on duplicate shard keys.
- Verify that every linear layer in Qwen3, Llama, Gemma 3 and Qwen3.5/3.8 (including the DeltaNet projections) goes through `build_lora_mm`, and patch any that don't.

### 2. Weight type `GGML_TYPE_B1`
- Pure sign bits: 128 signs per 16-byte block, exactly 1.0 bpw, bit 1 meaning +1, with no scale.
- CPU `to_float` serves the fallback and debugging.
- The enum value goes after upstream's last type, documented as a rebase hazard.

### 3. Fused op `GGML_OP_LITTLEBIT`
- Signature: `ggml_littlebit(ctx, x, V, U, s_v2, s_mid, s_u1, [Vr, Ur, s_v2r, s_midr, s_u1r])`
- It computes `y = u1⊙(Ubᵀ·(mid⊙(Vbᵀ·(v2⊙x)))) + residual`, replacing the 11-op chain.
- **Two-stage structure on every backend:**
  - **V-stage:** v2 is applied as input is loaded. Dot products run against packed sign rows, flipping signs by XOR on the float sign bit. Primary and residual ranks are concatenated, and `mid` goes in the epilogue.
  - **U-stage:** `u1`/`u1r` go in the epilogue, and both paths are summed.
  - Result: 2 launches per linear for small batches (decode).
- **Large batches (prefill):** decode B1 tiles to f16 and feed the backend's matrix units or GEMM. Correct first, optimize later.
- **Backends:**
  - **Phase 1:** CPU scalar reference, CPU AVX2 (and AVX-512 where present), and CUDA (decode kernels plus tiled prefill; CUDA graphs on).
  - **Phase 2:** Vulkan compute shaders and ARM NEON.
  - **Phase 3:** Metal shaders.
- Every backend registers `supports_op`. If a backend lacks it, ggml falls back to the CPU path, so the code stays correct even where it's slow.
- **Optional, after the gates pass:** merge the QKV and gate/up V-stages, since they share their input.

### 4. Head and embeddings
- These stay 16-bit by default. That is what correctness is measured on.
- The converter flag `--head-type q8_0|q6_k` shrinks them. That matters for speed on small models, where the head dominates the data read per token.
- Benchmarks report both settings.

### 5. Converter (`littlemore convert`)
- **Input:** a Samsung-format checkpoint directory:
  - safetensors with `U_packed`/`V_packed`/`*_shape` and `u1`/`u2`/`v1`/`v2`, including the `_R` residual tensors
  - `littlebit_config.json`
  - config.json and tokenizer. If the tokenizer is missing, it is fetched from the base model id in the config, after asking.
- **Process:**
  - It extends Cybertiron's converter to emit B1 and the new tensor names.
  - Row and column permutations are found by probing through llama.cpp's `convert_hf_to_gguf` for each family.
  - It pads ranks to multiples of 128. This is exact, because the padded `mid` values are 0.
- **Validation (always on):**
  - For each linear, random inputs go through our PyTorch reference (from the checkpoint) and through a NumPy decode of the written GGUF. A mismatch aborts.
  - After writing, a short logits comparison runs on CPU.

### 6. Dev-only tools (not shipped)
- **Lean PyTorch reference** (`tools/dev/reference.py`): runs a LittleBit model in factorized form without expanding the weights, so an 8B model fits in 16 GB RAM. This is the ground truth for perplexity and logits checks. Samsung's `eval.py` doesn't fit on this machine.
- **Init-only LittleBit-2 maker** (`tools/dev/make_lb2_init.py`):
  - Steps: SVD (√Σ split), then Joint-ITQ (T = 50), then Dual-SVID, plus the residual, then rank from bits. Reimplemented from the paper.
  - It outputs Samsung's packed format.
  - It produces small test models for every family, for example Qwen3-0.6B/1.7B, Llama-3.2-1B, Gemma-3-1B and the smallest Qwen3.5.
  - These models produce gibberish text, but they are exact for correctness and speed testing.
- **Real-weights test:** `LittleBitLLM/littlebit-qwen3-8b-0.55bpw`, a community LittleBit v1 checkpoint. The runtime is the same for v1 and v2.

### 7. C API `littlemore.h` (shape, my call)
- Opaque types: `lm_model`, `lm_session`.
- Model and session:
  - `lm_model_load(path, &opts)`. Options: auto device, auto-fit GPU layers (`n_gpu_layers=-1` means fit to free VRAM), context size, threads.
  - `lm_session_new` / `lm_session_free`
- Generation:
  - `lm_chat(session, messages, n, &sampling, on_token_cb, user)`, which applies the model's chat template.
  - `lm_generate(session, prompt, ...)`
  - `lm_tokenize`, `lm_detokenize`
- Info and errors: `lm_backends_info`, `lm_last_error`.
- Stability: ABI-stable C, versioned, and thread-safe per session.
- Packaging: built as one shared library, with ggml backends as runtime-loaded plugins (llama.cpp `GGML_BACKEND_DL`). One package works on machines with or without a GPU.

### 8. Python SDK
- Implementation: ctypes over the C API.
- Wheels:
  - win_amd64 and manylinux x86_64, each with CPU + Vulkan + CUDA backend plugins (CUDA runtime DLLs bundled; the CUDA plugin loads only if an NVIDIA driver is present)
  - macOS arm64 with Metal (phase 3)
- API: `littlemore.Model(path, n_gpu_layers="auto")`, `.chat(messages, stream=True)`, `.generate(prompt)`, `.info()`
- CLI: `littlemore convert`, `littlemore run model.gguf` (terminal chat), `littlemore bench`.

### 9. CI (GitHub Actions, public repo)
- **Phase 1:** Windows x64 and Linux x64 builds; CPU tests; converter tests on tiny synthetic models; wheel builds. The CUDA build is compile-only in CI, and the CUDA tests run locally on the 4050.
- **Phase 2:** ubuntu-24.04-arm (NEON tests), Linux Vulkan builds (with lavapipe software tests), and Android NDK builds producing the AAR.
- **Phase 3:** macos arm64 (Metal tests), iOS XCFramework builds.
- Releases attach the C library packages and the wheels.

### Milestones (phase 1, the next plan)
- **M0:** toolchain installed. Fork plus cherry-pick builds on Windows with CUDA, CPU and Vulkan. An init-only Qwen3-0.6B runs on the existing slow path.
- **M1:**
  - the PyTorch reference and the init-only maker for all 4 families
  - the B1 type
  - the converter with validation
  - the CPU reference fused op, with `test-backend-ops` cases
- **M2:** CUDA fused kernels (decode and prefill) and AVX2. Speed gates measured on the 4050.
- **M3:** the `littlemore.h` C API, auto-fit and runtime backend loading, the Python SDK and CLI, CI for Windows and Linux, docs and quickstart. The community 8B model chats on the 4050 through Python.
- **Phases 2 and 3:** each gets its own plan once phase 1 lands.

## Project conventions (user instructions)
- **Ruflo ("Rufus" in the user's message, assumed to mean Ruflo):**
  - Initialize it as the first implementation step (`hooks_init`, `memory_store`, then `swarm_init` and `agent_spawn` for multi-file work). This is deferred until plan mode ends.
  - Use it throughout the project:
    - route tasks with `hooks_route`
    - store key decisions and findings in `memory_store`
    - run the independent backend work (CUDA, CPU, Vulkan, Metal, SDK) as swarm/agent tasks
- **Code comments for a public audience:**
  - Every public file gets a header comment saying what it is and why it exists.
  - Every public function, kernel and struct gets a doc comment. For kernels, it explains the math and the memory layout (the packed-bit order, the scale folding).
  - Non-obvious tricks (the sign-bit XOR, rank padding, bit order) get an inline "why" comment.
  - Comments explain intent, not restate code.
  - Python gets docstrings and type hints. READMEs and the `docs/` folder explain how to build, test and use the project.
  - Reviewers check comment quality at the end of each milestone.

## Calls I made (user can override)
- Head/embedding quantization is a converter flag; correctness is measured at 16-bit.
- Speed gate is on Qwen3-0.6B and 1.7B (bf16 4B doesn't fit in 6 GB).
- A lean PyTorch reference serves as ground truth.
- Python binding uses ctypes; repo layout and C API shape as above.
- Mobile on-device speed stays unproven until device testing; ARM and Metal are proven on CI runners.

## Left to the builder
- Kernel tiling, warp/workgroup mapping, exact error thresholds.
- The exact llama.cpp release to pin.
- Whether to keep Q1_0 as a fallback format.
- The prefill kernel strategy within "correct first".
