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

# Lock all 4 GPUs to the 1380 MHz BIOS-max clock + persistence mode.
# The CMP 100-210 defaults to its 1147 MHz application clock and does NOT
# auto-boost under load, leaving ~15-20% on the table for clock-bound paths
# (Q8_0 DP4A / HFMA2 GEMM — exactly what inference uses). Locking to 1380 is a
# free, consistent gain (verified 2026-06-06: FFMA 7.3->8.4, HFMA2 15.5->17.7,
# steady). The tensor/FP64 throttle is fixed-cycle and unaffected, but we don't
# use those paths. Reversible with: sudo nvidia-smi -rgc
echo 9717 | sudo -S nvidia-smi -pm 1 >/dev/null 2>&1 || true
echo 9717 | sudo -S nvidia-smi -lgc 1380,1380 >/dev/null 2>&1 || true

cd "$ROOT"

# Chat server (GPUs 0,1,2) — DFlash speculative decode.
# Retrained drafter (AL~3.1) + lossless block-sparse verify beats the
# old MTP path, especially on long context: MTP gen falls off hard as KV grows
# (measured 24 t/s @0.5K -> 7.4 t/s @17K), DFlash stays flat (~21 t/s @100K).
# DFLASH_BUDGET=8: 2026-06-09 sweep (FA-tiled drafter build) — short/medium
# context verify is weight-read-bound so slots are ~free up to 8:
# b4 19.5s -> b8 17.6s on a 601-tok short gen (AL 2.58 -> 3.54); b12 regresses.
# Past DFLASH_BUDGET_LONG_CTX (default 32768) the engine auto-drops to
# DFLASH_BUDGET_LONG (default 4), preserving the old long-context tuning.
# PP_LAYER_BOUNDS=17,40 thins GPU0 to make room for the 1.97GB drafter (else OOM).
# DFlash auto-disables the MTP paths and skips loading the MTP head.
CHAT_DRAFTER=${CHAT_DRAFTER:-/home/paru/ue_training/dflash_train/trained_new/drafter.safetensors}
rm -f "$LOG_DIR/main_27b.log"
CUDA_VISIBLE_DEVICES=0,1,2 \
MTP_TQ=1 MLP_GATEUP_FUSED=1 MLP_GATEUP_FUSED_KERNEL=1 \
DFLASH=1 DFLASH_DRAFT_PATH="$CHAT_DRAFTER" \
DFLASH_VERIFY_BLOCKSPARSE=1 DFLASH_BUDGET=8 \
PP_LAYER_BOUNDS=17,40 \
# Single slot is REQUIRED for DFlash (slot-0-only capture buffer); a multi-entry
# --slot-caps would force num_slots>1 and silently disable DFlash. --max-seq sets
# the 256K KV cap for the one slot.
# (NOTE: comments must NOT sit between the env line-continuations and nohup —
#  a backslash-continued assignment line that ends in a comment becomes an
#  assignment-only statement, and NONE of the env reaches the server. That
#  exact bug shipped the chat server onto 4 GPUs with fp16 KV on 2026-06-10.)
MINF_SPARSE_ATTN=1 MINF_BUDGET=0.10 \
MINF_PROFILE_PATH=$ROOT/profiles/27B_block_sparse.bin \
nohup "$BIN" "$CHAT_MODEL" \
  --serve $CHAT_PORT --slots 1 --max-seq 262144 \
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

# Stagger: embed + rerank share GPU 3. Bringing both CUDA contexts up at the
# SAME instant intermittently corrupts the reranker's context (it dies on its
# first forward, and the chat proxy then silently falls back to embedding-
# cosine rerank, which looks like "instruction ignored / scores ~0.001").
# Wait for embed to finish loading + start listening before launching rerank.
echo "Waiting for embed (:$EMBED_PORT) to come up before launching rerank..."
for _ in $(seq 1 60); do
  if grep -q "listening on :$EMBED_PORT" /tmp/embed.log 2>/dev/null; then break; fi
  if ! kill -0 "$EMBED_PID" 2>/dev/null; then
    echo "WARNING: embed died during load — see /tmp/embed.log"; break
  fi
  sleep 2
done

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
