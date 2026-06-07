#!/bin/bash
# OPTIONAL: clone the original baseline repositories for cross-checking.
#
# The default Table-3 pipeline uses the in-repo reimplementations in
# baselines/models.py (DeepLOB, TLOB) trained under the unified protocol — this is
# more robust than the external repos (which often pin old torch / break on新 CUDA).
# Clone these only if you want to validate our reimplementations against originals.
set -e

EXT="$(cd "$(dirname "$0")/.." && pwd)/baselines/external"
mkdir -p "$EXT"
cd "$EXT"

echo "Cloning DeepLOB (Zhang et al., 2019)..."
git clone --depth 1 \
  https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books \
  deeplob || echo "  (already present or offline)"

echo "Cloning TLOB (Berti & Kasneci, 2025)..."
git clone --depth 1 https://github.com/LeonardoBerti00/TLOB tlob \
  || echo "  (already present or offline)"

echo "Done. External repos in $EXT"
echo "Note: Dense-transformer and FlashLOB are our own models — no external repo."
