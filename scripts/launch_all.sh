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
KVGEN_MODEL=${KVGEN_MODEL:-$GGUF_DIR/Qwen3.5-0.8B-kvgen-Q8_0.gguf}
KVGEN_HEADS_FILE=${KVGEN_HEADS_FILE:-/home/paru/models/kvgen_heads_v1.bin}
SPEC_LORA_FILE=${SPEC_LORA_FILE:-/home/paru/models/spec_lora_v2.bin}

CHAT_PORT=${CHAT_PORT:-8000}
EMBED_PORT=${EMBED_PORT:-8001}
RERANK_PORT=${RERANK_PORT:-8002}
KVGEN_PORT=${KVGEN_PORT:-8011}
# kvgen (0.8B speculative-prefill KV generator) shares GPU 3 with embed+rerank.
# All three CUDA contexts + their KV caches must fit in GPU 3's 16 GB. The
# kvgen KV cache is the size driver; with MTP_TQ=1 (3-bit TurboQuant KV, set in
# the kvgen launch below) the cache is 4.9x smaller, so the FULL 256K context
# fits alongside embed+rerank: at KVGEN_MAX_SEQ=262144 the kvgen footprint is
# ~2.6 GB (KV 654 MB) and GPU 3 still holds ~1.4 GB free (measured 2026-06-13).
# fp16 KV would need 3.2 GB and NOT fit (only ~0.3 GB margin). Everything is
# pre-allocated at init (zero runtime balloon). SPEC_PREFILL_MAX_LEN tracks this,
# so 256K prompts get speculative prefill (kvgen ~225s + tail ~56s = ~280s vs a
# ~730s full prefill). The back-to-back-256K crash was fixed in 5dc13eb (chunked
# kv-inject scratch + /dev/shm unlink), so consecutive 256K requests are safe.
KVGEN_MAX_SEQ=${KVGEN_MAX_SEQ:-262144}

# Kill any previous engine binaries. Match the executable path specifically
# so the pkill doesn't also kill this script (whose own path contains
# "qwen-engine"). pkill exits 1 when nothing matched, so swallow that.
pkill -9 -f "build/qwen-engine" 2>/dev/null || true
sleep 2

# Drop page cache before loading the 28 GB chat GGUF.
echo 9717 | sudo -S sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true

# NOTE: GPU clock lock (nvidia-smi -lgc 1380,1380) was intentionally REMOVED
# per the user's decision. Do NOT re-add it here. (The 140W power limit is
# applied separately and permanently via the gpu-power-limit.service systemd
# unit, so nothing clock/power related needs to run from this script.)

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
# Single slot is REQUIRED for DFlash (slot-0-only capture buffer); a multi-entry
# --slot-caps would force num_slots>1 and silently disable DFlash. --max-seq sets
# the 256K KV cap for the one slot.
# (NOTE: NO comment lines may sit anywhere inside the backslash-continued
#  env block below — a continued assignment line that ends in a comment
#  becomes an assignment-only statement and NONE of the env reaches the
#  server. That exact bug shipped the chat onto 4 GPUs / fp16 KV / no DFLASH
#  on 2026-06-10, twice.)
# Auto spec-prefill: for prompts in [32768, KVGEN_MAX_SEQ] the chat server calls
# the kvgen sidecar (:8011) to predict the head's KV, injects it, and real-
# prefills only the recent tail — big TTFT win on long prompts. SPEC_LORA is
# merged at load for inject-region fidelity (per-request MINF auto-off inside
# the engine). Shorter prompts are untouched (normal full prefill). The server
# falls back to full prefill if kvgen is down. (Env knobs must stay INSIDE the
# backslash block below — no comments between continuations, see the note there.)
CUDA_VISIBLE_DEVICES=0,1,2 \
MTP_TQ=1 MLP_GATEUP_FUSED=1 MLP_GATEUP_FUSED_KERNEL=1 \
DFLASH=1 DFLASH_DRAFT_PATH="$CHAT_DRAFTER" \
DFLASH_VERIFY_BLOCKSPARSE=1 DFLASH_BUDGET=8 \
PP_LAYER_BOUNDS=17,40 \
MINF_SPARSE_ATTN=1 MINF_BUDGET=0.10 \
MINF_PROFILE_PATH=$ROOT/profiles/27B_block_sparse.bin \
SPEC_LORA=$SPEC_LORA_FILE \
SPEC_PREFILL_AUTO=1 SPEC_PREFILL_KVGEN=127.0.0.1:$KVGEN_PORT \
SPEC_PREFILL_MIN_LEN=32768 SPEC_PREFILL_MAX_LEN=$KVGEN_MAX_SEQ SPEC_PREFILL_KEEP=3072 \
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

# Speculative-prefill KV generator (GPU 3, shared). Stagger again: wait for
# rerank to finish initialising its CUDA context before bringing kvgen's up, for
# the same context-corruption reason embed→rerank is staggered. kvgen produces
# predicted KV for the 27B (POST :8011/kvgen); it is NOT auto-wired into the chat
# server — clients that want spec-prefill call it explicitly.
echo "Waiting for rerank (:$RERANK_PORT) to come up before launching kvgen..."
for _ in $(seq 1 60); do
  if grep -q "listening on :$RERANK_PORT" /tmp/rerank.log 2>/dev/null; then break; fi
  if ! kill -0 "$RERANK_PID" 2>/dev/null; then
    echo "WARNING: rerank died during load — see /tmp/rerank.log"; break
  fi
  sleep 2
done
CUDA_VISIBLE_DEVICES=3 \
KVGEN_HEADS="$KVGEN_HEADS_FILE" KVGEN_MAX_SEQ=$KVGEN_MAX_SEQ \
MTP_TQ=1 \
MINF_SPARSE_ATTN=1 MINF_BUDGET=0.10 MINF_UNIFORM_MS=1 \
nohup "$BIN" "$KVGEN_MODEL" \
  --serve $KVGEN_PORT --mode kvgen \
  > /tmp/kvgen.log 2>&1 &
KVGEN_PID=$!

echo "chat   PID=$CHAT_PID  port=$CHAT_PORT  (GPUs 0,1,2)"
echo "embed  PID=$EMBED_PID port=$EMBED_PORT (GPU 3)"
echo "rerank PID=$RERANK_PID port=$RERANK_PORT (GPU 3)"
echo "kvgen  PID=$KVGEN_PID port=$KVGEN_PORT (GPU 3, max_seq=$KVGEN_MAX_SEQ)"
echo
echo "Clients hit only :$CHAT_PORT — /v1/embeddings → :$EMBED_PORT,"
echo "/v1/rerank → :$RERANK_PORT, both auto-proxied by the chat server."
echo "kvgen spec-prefill is a direct endpoint: POST :$KVGEN_PORT/kvgen."
echo "Chat ~110 s to load; embed + rerank + kvgen sidecars ~10-30 s each."
