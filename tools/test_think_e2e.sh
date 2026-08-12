#!/usr/bin/env bash
# End-to-end check of the reasoning_content split against a live server.
# Companion to tools/test_think_split.cpp, which unit-tests the same split.
#
#   MINF_ENABLE_JSON_MODE=1 CUDA_VISIBLE_DEVICES=1 \
#     ./build/qwen-engine <model>.gguf --serve 8009 --max-seq 4096 --slots 1 &
#   tools/test_think_e2e.sh 8009
#
# (the env var is only needed for case 5; everything else passes without it)
PORT=${1:-8009}
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT
URL="http://localhost:$PORT/v1/chat/completions"
fail=0
ok()   { printf '  ok   %s\n' "$1"; }
bad()  { printf '  FAIL %s -- %s\n' "$1" "$2"; fail=$((fail+1)); }

post() { curl -s --max-time 300 "$URL" -H 'Content-Type: application/json' -d "$1"; }

echo "== 1. enable_thinking:true, non-streaming =="
r=$(post '{"model":"qwen","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":300,"chat_template_kwargs":{"enable_thinking":true}}')
echo "$r" > "$TMP/t1.json"
rc=$(echo "$r" | jq -r '.choices[0].message.reasoning_content // ""')
ct=$(echo "$r" | jq -r '.choices[0].message.content // ""')
[ -n "$rc" ] && ok "reasoning_content present (${#rc} chars)" || bad "reasoning_content" "empty"
case "$ct" in *"</think>"*|*"<think>"*) bad "content tag-free" "content=[$ct]";; *) ok "content tag-free (${#ct} chars)";; esac
case "$rc" in *"</think>"*|*"<think>"*) bad "reasoning tag-free" "has a tag";; *) ok "reasoning tag-free";; esac
[ -n "$ct" ] && ok "content non-empty" || bad "content" "empty"

echo "== 2. enable_thinking:true, streaming =="
s=$(curl -s -N --max-time 300 "$URL" -H 'Content-Type: application/json' \
    -d '{"model":"qwen","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":300,"stream":true,"chat_template_kwargs":{"enable_thinking":true}}')
echo "$s" > "$TMP/t2.sse"
srd=$(echo "$s" | sed -n 's/^data: //p' | grep -v '^\[DONE\]' | jq -rj '.choices[0].delta.reasoning_content // ""' 2>/dev/null)
scd=$(echo "$s" | sed -n 's/^data: //p' | grep -v '^\[DONE\]' | jq -rj '.choices[0].delta.content // ""' 2>/dev/null)
[ -n "$srd" ] && ok "delta.reasoning_content present (${#srd} chars)" || bad "delta.reasoning_content" "none"
case "$scd" in *"</think>"*|*"<think>"*) bad "stream content tag-free" "content=[$scd]";; *) ok "stream content tag-free (${#scd} chars)";; esac
[ -n "$scd" ] && ok "stream content non-empty" || bad "stream content" "empty"

echo "== 3. enable_thinking:false =="
r=$(post '{"model":"qwen","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}')
has=$(echo "$r" | jq -r 'has("choices") and (.choices[0].message | has("reasoning_content"))')
ct=$(echo "$r" | jq -r '.choices[0].message.content // ""')
[ "$has" = "false" ] && ok "no reasoning_content field" || bad "reasoning_content" "present when thinking off"
case "$ct" in *"think>"*) bad "content tag-free" "content=[$ct]";; *) ok "content tag-free (${#ct} chars)";; esac

echo "== 4. /no_think directive =="
r=$(post '{"model":"qwen","messages":[{"role":"user","content":"/no_think Say hi in one word."}],"max_tokens":64}')
has=$(echo "$r" | jq -r '.choices[0].message | has("reasoning_content")')
ct=$(echo "$r" | jq -r '.choices[0].message.content // ""')
[ "$has" = "false" ] && ok "no reasoning_content field" || bad "reasoning_content" "present with /no_think"
case "$ct" in *"think>"*) bad "content tag-free" "content=[$ct]";; *) ok "content tag-free (${#ct} chars)";; esac

# Needs the server started with MINF_ENABLE_JSON_MODE=1 — response_format is
# ignored otherwise, and then the model is free to think (reasoning_content
# showing up would be correct, not a bug).
echo "== 5. json_object mode (force_think=-1) =="
r=$(post '{"model":"qwen","messages":[{"role":"user","content":"Give me {\"a\":1} as JSON."}],"max_tokens":64,"response_format":{"type":"json_object"}}')
ct=$(echo "$r" | jq -r '.choices[0].message.content // ""')
has=$(echo "$r" | jq -r '.choices[0].message | has("reasoning_content")')
[ "$has" = "false" ] && ok "no reasoning_content field" || bad "reasoning_content" "present in json mode"
case "$ct" in *"think>"*) bad "content tag-free" "content=[$ct]";; *) ok "content tag-free (${#ct} chars)";; esac

echo "== 6. cut off mid-reasoning (max_tokens=12, thinking on) =="
r=$(post '{"model":"qwen","messages":[{"role":"user","content":"Prove that there are infinitely many primes, in detail."}],"max_tokens":12,"chat_template_kwargs":{"enable_thinking":true}}')
echo "$r" > "$TMP/t6.json"
rc=$(echo "$r" | jq -r '.choices[0].message.reasoning_content // ""')
fr=$(echo "$r" | jq -r '.choices[0].finish_reason')
[ -n "$rc" ] && ok "partial reasoning kept (${#rc} chars)" || bad "reasoning_content" "response vanished"
[ "$fr" = "length" ] && ok "finish_reason=length" || bad "finish_reason" "got $fr, want length"

echo "== 7. streaming cut off mid-reasoning (max_tokens=12) =="
s=$(curl -s -N --max-time 300 "$URL" -H 'Content-Type: application/json' \
    -d '{"model":"qwen","messages":[{"role":"user","content":"Prove that there are infinitely many primes, in detail."}],"max_tokens":12,"stream":true,"chat_template_kwargs":{"enable_thinking":true}}')
srd=$(echo "$s" | sed -n 's/^data: //p' | grep -v '^\[DONE\]' | jq -rj '.choices[0].delta.reasoning_content // ""' 2>/dev/null)
fr=$(echo "$s" | sed -n 's/^data: //p' | grep -v '^\[DONE\]' | jq -r '.choices[0].finish_reason // empty' 2>/dev/null | tail -1)
[ -n "$srd" ] && ok "partial reasoning streamed (${#srd} chars)" || bad "delta.reasoning_content" "nothing streamed"
[ "$fr" = "length" ] && ok "finish_reason=length" || bad "finish_reason" "got $fr, want length"

echo
[ $fail -eq 0 ] && echo "ALL PASSED" || echo "$fail FAILED"
exit $fail
