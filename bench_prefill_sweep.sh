#!/bin/bash
# Sweep prefill across context lengths against :8000, max_tokens=1.
# Unique per-length prefix defeats prefix-cache so every run is a full prefill.
# Parses server-side [API] prefill log line. Usage: ./bench_prefill_sweep.sh "1000 4000 ..." LOGFILE
PORT=8000
LENS=${1:-"1000 4000 8000 18000"}
LOG=${2:-/home/paru/qwen-engine/main_27b_prodB.log}

for toks in $LENS; do
  words=$(( toks * 9 / 11 ))
  reps=$(( words / 9 + 1 ))
  # Unique nonce per length so no two prompts share a cacheable prefix.
  pmt="benchmark run identifier $toks $RANDOM unique salt $$ . "
  for ((i=0; i<reps; i++)); do pmt+="the quick brown fox jumps over the lazy dog "; done

  body=$(jq -nc --arg p "$pmt" '{model:"qwen", messages:[{role:"user",content:$p}], max_tokens:1, temperature:0, stream:false}')

  echo "=== SWEEP target=$toks ===" >> "$LOG"
  t0=$(date +%s.%N)
  curl -s -X POST -H 'Content-Type: application/json' -d "$body" \
    "http://localhost:$PORT/v1/chat/completions" > /tmp/resp_sweep.json
  t1=$(date +%s.%N)
  awk -v t=$toks -v t0=$t0 -v t1=$t1 'BEGIN{printf "target=%d  client_total=%.2fs\n", t, t1-t0}'
  sleep 1
done
echo "--- server [API] prefill lines ---"
grep "\[API\] prefill" "$LOG" | tail -$(echo $LENS | wc -w)
