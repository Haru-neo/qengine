#!/usr/bin/env bash
# All-in-one: build qengine for the 4090 (Ada sm_89) + run a FAITHFUL KV extract.
# Run this ON THE 4090 MACHINE — native Linux, or inside WSL2 (Ubuntu) on Windows.
#   (WSL2 needs the NVIDIA Windows driver + `nvidia-smi` working inside WSL.)
#
# Prereqs on the 4090 machine:
#   - CUDA toolkit + nvcc + cmake + a C++17 compiler
#   - qengine source present, INCLUDING the gpu_loader.h QENGINE_UM_OFFLOAD change
#     (either git clone github.com/Haru-neo/qengine after that change is pushed,
#      or copy the patched src/gpu_loader.h over a fresh clone)
#   - the Coder model gguf + a corpus file (space-separated token ids, 1 line/seq)
#
# Edit these paths for the 4090 machine, then: bash setup_extract_4090.sh
QENGINE_DIR=${QENGINE_DIR:-$HOME/qwen-engine}
MODEL=${MODEL:-$HOME/models/Qwopus3.6-27B-Coder-Q8_0.gguf}
CORPUS=${CORPUS:-$HOME/corpus.txt}
OUT=${OUT:-$HOME/kvgen_out}
CUDA_ARCH=${CUDA_ARCH:-89}          # 4090 = Ada = sm_89

set -e
echo "=== [1/3] sanity ==="
command -v nvcc >/dev/null || { echo "!! nvcc not found (install CUDA toolkit)"; exit 1; }
nvidia-smi -L | head -1 || { echo "!! no GPU visible (WSL2: need NVIDIA Windows driver)"; exit 1; }
[ -f "$QENGINE_DIR/src/gpu_loader.h" ] || { echo "!! qengine source not at $QENGINE_DIR"; exit 1; }
grep -q 'QENGINE_UM_OFFLOAD' "$QENGINE_DIR/src/gpu_loader.h" || { echo "!! gpu_loader.h missing the UM-offload patch — copy the patched file"; exit 1; }
[ -f "$MODEL" ]  || { echo "!! model not found: $MODEL"; exit 1; }
[ -f "$CORPUS" ] || { echo "!! corpus not found: $CORPUS"; exit 1; }

echo "=== [2/3] build qengine for sm_$CUDA_ARCH ==="
mkdir -p "$QENGINE_DIR/build"
cd "$QENGINE_DIR/build"
cmake -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH ..
make -j"$(nproc)"
BIN="$QENGINE_DIR/build/qwen-engine"
[ -x "$BIN" ] || { echo "!! build failed"; exit 1; }

echo "=== [3/3] extract (single GPU + UM offload + TQ3 + dense) ==="
mkdir -p "$OUT"
# single GPU; UM offload fits 28GB Q8 on 24GB; DENSE (no MINF) + TQ3 = deployment-faithful
CUDA_VISIBLE_DEVICES=0 QENGINE_UM_OFFLOAD=1 \
DFLASH_EXTRACT_KV=1 DFLASH_EXTRACT_PIPELINE=0 MTP_TQ=1 \
  "$BIN" "$MODEL" --mode dflash-extract \
  --dflash-corpus "$CORPUS" --dflash-out "$OUT" --dflash-chunk-bytes 18000000000
echo "=== done -> $OUT ==="
ls -la "$OUT" | head
