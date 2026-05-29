#!/usr/bin/env bash
# Launch qengine chat (27B) + embedding (4B) + reranker (4B) as 3 processes.
# Chat sits on GPUs 0/1/2 and acts as the reverse proxy for /v1/embeddings
# (-> :8001) and /v1/rerank (-> :8002). Embed + rerank share GPU 3.
#
# Clients only need to know about :8000 — embeddings + rerank get auto-routed.

set -e

ROOT=/home/paru/qwen-engine
BIN=$ROOT/build/qwen-engine
LOG_DIR=${LOG_DIR:-$ROOT}
GGUF_DIR=${GGUF_DIR:-/home/paru/models/gguf}

CHAT_MODEL=${CHAT_MODEL:-$GGUF_DIR/Qwopus3.6-27B-v2-MTP-Q8_0.gguf}
CHAT_MMPROJ=${CHAT_MMPROJ:-$GGUF_DIR/Qwopus3.6-27B-v2-mmproj.gguf}
EMBED_MODEL=${EMBED_MODEL:-$GGUF_DIR/Qwen3-Embedding-4B-Q8_0.gguf}
RERANK_MODEL=${RERANK_MODEL:-$GGUF_DIR/Qwen3-Reranker-4B-Q8_0.gguf}

CHAT_PORT=${CHAT_PORT:-8000}
EMBED_PORT=${EMBED_PORT:-8001}
RERANK_PORT=${RERANK_PORT:-8002}

# Kill any previous engine binaries. Match the executable path specifically
# so the pkill doesn't also kill this script (whose own path contains
# "qwen-engine"). pkill exits 1 when nothing matched, so swallow that.
pkill -9 -f "build/qwen-engine" 2>/dev/null || true
sleep 2

# Drop page cache before loading the 28 GB chat GGUF.
echo 9717 | sudo -S sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true

cd "$ROOT"

# Chat server (GPUs 0,1,2)
rm -f "$LOG_DIR/main_27b.log"
CUDA_VISIBLE_DEVICES=0,1,2 \
MTP_TQ=1 MLP_GATEUP_FUSED=1 MLP_GATEUP_FUSED_KERNEL=1 \
MINF_SPARSE_ATTN=1 MINF_BUDGET=0.10 \
MINF_PROFILE_PATH=$ROOT/profiles/27B_block_sparse.bin \
nohup "$BIN" "$CHAT_MODEL" \
  --serve $CHAT_PORT --slot-caps "262144,65536" \
  --vision-mmproj "$CHAT_MMPROJ" \
  --proxy-embed 127.0.0.1:$EMBED_PORT \
  --proxy-rerank 127.0.0.1:$RERANK_PORT \
  >> "$LOG_DIR/main_27b.log" 2>&1 &
CHAT_PID=$!

# Embedding server (GPU 3)
CUDA_VISIBLE_DEVICES=3 nohup "$BIN" "$EMBED_MODEL" \
  --serve $EMBED_PORT --mode embed \
  > /tmp/embed.log 2>&1 &
EMBED_PID=$!

# Reranker server (GPU 3, shared). Qwen3-Reranker-4B Q8_0 built from the
# official safetensors with cls.output kept F16; uses the classifier head.
CUDA_VISIBLE_DEVICES=3 nohup "$BIN" "$RERANK_MODEL" \
  --serve $RERANK_PORT --mode rerank \
  > /tmp/rerank.log 2>&1 &
RERANK_PID=$!

echo "chat   PID=$CHAT_PID  port=$CHAT_PORT  (GPUs 0,1,2)"
echo "embed  PID=$EMBED_PID port=$EMBED_PORT (GPU 3)"
echo "rerank PID=$RERANK_PID port=$RERANK_PORT (GPU 3)"
echo
echo "Clients hit only :$CHAT_PORT — /v1/embeddings → :$EMBED_PORT,"
echo "/v1/rerank → :$RERANK_PORT, both auto-proxied by the chat server."
echo "Chat ~110 s to load; embed + rerank sidecars ~30 s each."
