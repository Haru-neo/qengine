#!/usr/bin/env bash
# Launch 27B chat for benchmarking. Args: LOGFILE [EXTRA_ENV...]
# SPARSE env: set MINF_SPARSE_ATTN externally to override.
ROOT=/home/paru/qwen-engine
LOG=${1:-main_27b_prodB.log}
cd "$ROOT"
pkill -9 -f "build/qwen-engine" 2>/dev/null || true
sleep 3
echo 9717 | sudo -S sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
rm -f "$LOG"

export CUDA_VISIBLE_DEVICES=0,1,2
export MTP_TQ=1 MLP_GATEUP_FUSED=1 MLP_GATEUP_FUSED_KERNEL=1
# TRUE_DENSE=1 fully unsets sparse env (code only checks presence, not value!)
if [ "${TRUE_DENSE:-0}" = "1" ]; then
  unset MINF_SPARSE_ATTN MINF_PROFILE_PATH
  echo "[launch] TRUE DENSE — sparse env unset"
else
  export MINF_SPARSE_ATTN=1
  export MINF_BUDGET=${MINF_BUDGET:-0.10}
  export MINF_PROFILE_PATH=$ROOT/profiles/27B_block_sparse.bin
fi

nohup ./build/qwen-engine /home/paru/models/gguf/Qwopus3.6-27B-v2-MTP-Q8_0.gguf \
  --serve 8000 --slot-caps "262144,65536" \
  >> "$LOG" 2>&1 &
echo "launched PID=$! log=$LOG SPARSE=$MINF_SPARSE_ATTN"
