#!/bin/bash
# prefill_guard_study.sh: makes the temporary engine builds of tools/dev/prefill_guard_study.py and measures each.
#
# What: (1) the build as committed (prefill path, plus the dense and pure-CPU controls), (2) a build whose CUDA LittleBit
# op always takes the chunked decode kernels (the other legitimate path), (3) four builds with a deliberately broken
# prefill epilogue: the residual term y = s_u1 * P + s_u1r * Q multiplied by 1.001, 1.01, 1.1, and removed.  Each
# temporary build only edits engine/ggml/src/ggml-cuda (littlebit-dispatch.h or littlebit.cu); the original files are
# restored and ggml-cuda is rebuilt at the end (also on a failure), and the script checks they are byte-identical.
# Why: the table "These model-level GPU guards are coarse" in docs/engine-changes.md, i.e. the spread of correct runs and
# the size of mutations at model level, is the evidence for the thresholds of the prefill guards.
#
# Usage (Git Bash on Windows, from the repository root):
#   BUILD_CUDA="<a .bat that sets up MSVC + CUDA 12.9 and runs: cmake --build engine/build -j 6 --target ggml-cuda>" \
#   OUT=<results.jsonl> bash tools/dev/prefill_guard_study.sh
# then: python tools/dev/prefill_guard_study.py table <results.jsonl>
# Takes about an hour (six ggml-cuda builds, ~50 llama-debug runs, fp32 references on the CPU).
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
H="$ROOT/engine/ggml/src/ggml-cuda/littlebit-dispatch.h"
CU="$ROOT/engine/ggml/src/ggml-cuda/littlebit.cu"
PY="$ROOT/.venv/Scripts/python.exe"
TMP="$(mktemp -d)"
OUT="${OUT:?set OUT to the results file}"
: "${BUILD_CUDA:?set BUILD_CUDA to the build script}"
cp "$H" "$TMP/h"; cp "$CU" "$TMP/cu"
restore() { cp "$TMP/h" "$H"; cp "$TMP/cu" "$CU"; }
build() { cmd //c "$BUILD_CUDA" > "$TMP/build_$1.log" 2>&1 || { echo "build $1 failed (log $TMP/build_$1.log)"; restore; cmd //c "$BUILD_CUDA" > /dev/null 2>&1; exit 1; }; }
measure() { (cd "$ROOT" && "$PY" tools/dev/prefill_guard_study.py measure "$@" "$OUT" | grep '^{'); }
: > "$OUT"
measure prefill --full
# the other legitimate path: every LittleBit op on the chunked decode kernels
sed -i "/^static constexpr int ggml_cuda_littlebit_chunked_max_tokens(/a\    if (d_in >= 0) { return 1000000; }  // TEMPORARY" "$H"
build chunked; measure chunked; restore
# mutations of the prefill epilogue
for m in 1.001 1.01 1.1; do
  sed -i "s/            v = v + s_u1r\[i\] \* Q\[idx\];/            v = v + ${m}f * s_u1r[i] * Q[idx];  \/\/ MUTATION/" "$CU"
  grep -q MUTATION "$CU" || { echo "mutation $m not applied"; restore; exit 1; }
  build "mut_$m"; measure "mut_s_u1r_x$m"; restore
done
sed -i "s/        if (Q) {/        if (Q \&\& false) {  \/\/ MUTATION/" "$CU"
grep -q MUTATION "$CU" || { echo "drop mutation not applied"; restore; exit 1; }
build mut_drop; measure mut_drop_residual; restore
build restored
cmp "$H" "$TMP/h" && cmp "$CU" "$TMP/cu" && echo "sources restored"
