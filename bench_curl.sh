#!/bin/bash
# Bench by curl: send prompts of varying length, capture client-side TTFT+gen, server prints prefill/gen via main log
PORT=${1:-8001}
LABEL=${2:-9b}
OUTLOG=/home/paru/qwen-engine/bench_qengine_${LABEL}.log

make_prompt() {
  local n_words=$1
  local p=""
  for ((i=0; i<n_words; i++)); do p+="the quick brown fox jumps over the lazy dog "; done
  echo -n "$p"
}

bench_one() {
  local target=$1
  local pmt
  pmt=$(make_prompt $((target/2)))
  local body
  body=$(jq -nc --arg p "$pmt" '{model:"qwen",messages:[{role:"user",content:$p}],max_tokens:128,temperature:0,stream:true}')
  # measure ttft via stream
  local t0 t_first=""
  t0=$(date +%s.%N)
  while IFS= read -r line; do
    if [[ -z "$t_first" && "$line" == data:* && "$line" != *"[DONE]"* ]]; then
      t_first=$(date +%s.%N)
    fi
  done < <(curl -s -N -X POST -H 'Content-Type: application/json' -d "$body" "http://localhost:$PORT/v1/chat/completions")
  local t_end=$(date +%s.%N)
  if [[ -n "$t_first" ]]; then
    awk -v t0=$t0 -v tf=$t_first -v te=$t_end "BEGIN{printf \"target_words=$target ttft=%.3fs total=%.3fs\\n\", tf-t0, te-t0}"
  fi
}

echo "=== bench $LABEL @ port $PORT ===" >&2
for n in 64 256 1024 4096; do
  echo "--- $n words ---" >&2
  bench_one "$n"
  sleep 1
done
