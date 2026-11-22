#!/bin/bash
# chunk_crossover_study.sh: time the two CUDA paths of GGML_OP_LITTLEBIT (chunked decode kernels vs prefill path) per
# layer shape and batch size, to choose the chunk limits of engine/ggml/src/ggml-cuda/littlebit-dispatch.h.
#
# What: two temporary builds of ggml-cuda, one whose dispatch always takes the chunked decode kernels and one that always
# takes the prefill path above 8 tokens; for each, engine/tools/littlebit-bench/prefill_bench (LITTLEBIT only,
# LB_BENCH_SKIP_F16=1, own weights per op, 1.5 s warm-up and >= 1 s timing per case) on the 0.55-bits-per-weight layer
# shapes of Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B and Llama-3-8B (padded ranks r = r2, rank_for_bits of make_lb2_init.py)
# and the built-in larger shapes of prefill_bench.  The original header is restored and ggml-cuda rebuilt at the end.
# Why: the limits are a speed-only choice (both paths are float32 and agree to rounding) that depends on the GPU; this
# script is how they were measured (docs/engine-changes.md, "Chunk limits").
#
# Usage (Git Bash on Windows, from the repository root):
#   BUILD_CUDA=<.bat: MSVC + CUDA 12.9 env, cmake --build engine/build -j 6 --target ggml-cuda> \
#   PREFILL_BENCH=<prefill_bench.exe built as in engine/tools/littlebit-bench/README.md> \
#   OUT=<prefix> bash tools/dev/chunk_crossover_study.sh
# writes <prefix>_chunk.txt and <prefix>_gemm.txt; analyse with tools/dev/chunk_crossover_study.py.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
H="$ROOT/engine/ggml/src/ggml-cuda/littlebit-dispatch.h"
: "${BUILD_CUDA:?}"; : "${PREFILL_BENCH:?}"; : "${OUT:?}"
NS=${NS:-9,12,16,20,24,28,32,40,48,56,64,72,80,96,112,128}
TMP="$(mktemp -d)"; cp "$H" "$TMP/h"
export LB_BENCH_SKIP_F16=1
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin:$PATH"
SHAPES=(
  "1024 2048 256 256" "1024 1024 128 128" "2048 1024 256 256" "1024 3072 256 256" "3072 1024 256 256"   # Qwen3-0.6B
  "2048 2048 384 384" "2048 6144 512 512" "6144 2048 512 512"         # Qwen3-1.7B (k, v: 2048 x 1024 as 0.6B o)
  "2560 4096 512 512" "2560 1024 256 256" "4096 2560 512 512" "2560 9728 640 640" "9728 2560 640 640"   # Qwen3-4B
  "4096 4096 640 640" "4096 1024 256 256" "4096 14336 896 896" "14336 4096 896 896"                    # Llama-3-8B
)
for mode in chunk gemm; do
  cp "$TMP/h" "$H"
  v=$([ $mode = chunk ] && echo 1000000 || echo 8)
  sed -i "/^static constexpr int ggml_cuda_littlebit_chunked_max_tokens(/a\    if (d_in >= 0) { return $v; }  // TEMPORARY" "$H"
  cmd //c "$BUILD_CUDA" > "$TMP/build_$mode.log" 2>&1 || { echo "build $mode failed ($TMP/build_$mode.log)"; cp "$TMP/h" "$H"; exit 1; }
  : > "${OUT}_$mode.txt"
  (cd "$ROOT/engine/build/bin" && for sh in "${SHAPES[@]}"; do "$PREFILL_BENCH" CUDA0 64 "$NS" $sh; done \
     && "$PREFILL_BENCH" CUDA0 64 "$NS") >> "${OUT}_$mode.txt" 2>&1
done
cp "$TMP/h" "$H"
cmd //c "$BUILD_CUDA" > "$TMP/build_restore.log" 2>&1; echo "restore build exit $?"
cmp "$H" "$TMP/h" && echo "header restored"
