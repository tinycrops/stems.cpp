#!/usr/bin/env bash
# Build stems.cpp for one backend into its own build dir (backends coexist for A/B).
#
# Usage: ./build.sh [cpu|cuda|vulkan|metal|all]   (default: cpu)
set -euo pipefail

BACKEND="${1:-cpu}"
EXTRA=("${@:2}")

case "$BACKEND" in
  cpu)    DIR=build         ; FLAGS=(-DGGML_CUDA=OFF) ;;
  cuda)   DIR=build-cuda    ; FLAGS=(-DSTEMS_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native) ;;
  vulkan) DIR=build-vulkan  ; FLAGS=(-DSTEMS_VULKAN=ON) ;;
  metal)  DIR=build-metal   ; FLAGS=(-DSTEMS_METAL=ON) ;;
  all)    DIR=build-all     ; FLAGS=(-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DSTEMS_CUDA=ON -DSTEMS_VULKAN=ON) ;;
  *) echo "unknown backend: $BACKEND  (cpu|cuda|vulkan|metal|all)" >&2; exit 1 ;;
esac

echo "[stems] configuring $BACKEND -> $DIR/"
cmake -S . -B "$DIR" -DCMAKE_BUILD_TYPE=Release "${FLAGS[@]}" "${EXTRA[@]}"
# Bounded: an unbounded -j on the CUDA kernels exhausted 32 GB and took the box down.
JOBS="${JOBS:-4}"
cmake --build "$DIR" --config Release --parallel "$JOBS"
echo "[stems] built $BACKEND -> $DIR/bin/"

# Backends live in separate dirs, so rebuilding one leaves the others on old code. A stale
# build-cuda/ once shipped stems that were mostly noise; say so rather than let it happen quietly.
newest_src=$(find src tools -type f -printf '%T@\n' | sort -n | tail -1)
for other in build build-*; do
  [ "$other" = "$DIR" ] && continue
  bin="$other/bin/stems-split"
  [ -x "$bin" ] || continue
  if awk -v a="$(stat -c %Y "$bin")" -v b="$newest_src" 'BEGIN{exit !(a < b)}'; then
    echo "[stems] WARNING: $other/ is older than the sources; rebuild it before use" >&2
  fi
done
