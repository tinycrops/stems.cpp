#!/usr/bin/env bash
# Fetch HTDemucs GGUFs into models/.  Usage: ./models.sh [htdemucs|htdemucs_6s|htdemucs_ft|all]
#
# The GGUFs are published as release assets of this repo: a straight re-encoding of
# facebookresearch/demucs' MIT-licensed checkpoints (tools/convert_htdemucs.py). To build them
# yourself instead: pip install demucs gguf && python tools/convert_htdemucs.py htdemucs models/htdemucs-f32.gguf
set -euo pipefail
BASE="${STEMS_MODELS_URL:-https://github.com/tinycrops/stems.cpp/releases/download/models-v1}"
WHICH="${1:-htdemucs}"
[ "$WHICH" = all ] && WHICH="htdemucs htdemucs_6s htdemucs_ft"
mkdir -p models
for m in $WHICH; do
  f="models/$m-f32.gguf"
  if [ -s "$f" ]; then echo "[stems] $f already present"; continue; fi
  echo "[stems] downloading $m"
  curl -fL --retry 3 -C - -o "$f.part" "$BASE/$m-f32.gguf" && mv "$f.part" "$f"
done
