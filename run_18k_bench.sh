#!/bin/bash
# Run a single 18K-prompt 128-tok bench against port 8000.
# Server-side log lines (prefill/gen + PROFILE_ATTN/GEN breakdown) print to main_27b.log.
PORT=${1:-8000}
TOKS=${2:-128}
WORDS=$((4096))   # produces 18441-token prompt

pmt=""
for ((i=0; i<2048; i++)); do pmt+="the quick brown fox jumps over the lazy dog "; done

body=$(jq -nc --arg p "$pmt" --argjson n "$TOKS" \
  '{model:"qwen", messages:[{role:"user",content:$p}], max_tokens:$n, temperature:0, stream:false}')

t0=$(date +%s.%N)
curl -s -X POST -H 'Content-Type: application/json' -d "$body" \
  "http://localhost:$PORT/v1/chat/completions" > /tmp/resp_18k.json
t1=$(date +%s.%N)

awk -v t0=$t0 -v t1=$t1 'BEGIN{printf "client total = %.2fs\n", t1-t0}'
echo "--- server log tail ---"
tail -10 main_27b.log
