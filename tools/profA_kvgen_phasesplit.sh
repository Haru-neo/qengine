#!/usr/bin/env bash
# Experiment A: standalone kvgen phase split @ 100K + 256K (production config:
# single-GPU GPU3, Q8 KV, MINF uniform-MS sparse on). Answers: what dominates
# the kvgen-gen cost (attn O(n^2)? gdn? heads? write?) and whether 53.8s/160s
# still hold. p_cap (tap-gather) is ~0 on single-GPU; body = attn+gdn+mlp is the
# part multi-GPU Inc 2 would pipeline.
set -u
cd /home/paru/qwen-engine
BIN=build/qwen-engine
KVGEN_MODEL=/home/paru/models/gguf/Qwen3.5-0.8B-kvgen-Q8_0.gguf
HEADS=/home/paru/models/kvgen_heads_v1.bin
LOG=/tmp/kvgen_prof.log
rm -f "$LOG"

CUDA_VISIBLE_DEVICES=3 KVGEN_HEADS="$HEADS" KVGEN_MAX_SEQ=262144 Q8KV=1 \
  MINF_SPARSE_ATTN=1 MINF_BUDGET=0.10 MINF_UNIFORM_MS=1 KVGEN_PROFILE=1 \
  "$BIN" "$KVGEN_MODEL" --serve 8011 --mode kvgen > "$LOG" 2>&1 &
KVPID=$!

ok=0
for i in $(seq 1 90); do
  if grep -q "listening on :8011" "$LOG" 2>/dev/null; then ok=1; break; fi
  if ! kill -0 $KVPID 2>/dev/null; then echo "!! KVGEN DIED DURING LOAD"; tail -25 "$LOG"; exit 1; fi
  sleep 2
done
if [ $ok -ne 1 ]; then echo "!! KVGEN LOAD TIMEOUT (180s)"; tail -25 "$LOG"; kill -9 $KVPID 2>/dev/null; exit 1; fi
echo "=== kvgen sidecar up on :8011 (GPU3) ==="

python3 - <<'PY'
import struct, json, socket, time, os
def make_ids(n, path):
    ids = [(1000 + (i*7919) % 40000) for i in range(n)]
    with open(path,'wb') as f:
        f.write(struct.pack('<%di'%n, *ids))
def post_kvgen(n, keep, tag):
    ids_p=f"/dev/shm/profA_ids_{tag}.bin"; out_p=f"/dev/shm/profA_kv_{tag}.bin"
    make_ids(n, ids_p)
    body=json.dumps({"ids_file":ids_p,"keep":keep,"out":out_p})
    req=("POST /kvgen HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
         f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n{body}")
    s=socket.create_connection(("127.0.0.1",8011),timeout=900)
    t0=time.time(); s.sendall(req.encode())
    resp=b""
    while True:
        d=s.recv(8192)
        if not d: break
        resp+=d
    el=time.time()-t0; s.close()
    last=resp.decode(errors='replace').splitlines()[-1] if resp else "<empty>"
    print(f"[EXP-A {tag}] n={n} keep={keep} WALL={el:.1f}s resp={last}", flush=True)
    for p in (ids_p,out_p):
        try: os.remove(p)
        except: pass
for n,tag in [(100000,"100k"),(262000,"256k")]:
    post_kvgen(n, 3072, tag)
PY

echo "=== [kvgen] + [kvgen-prof] lines ==="
grep -E "\[kvgen" "$LOG" || echo "(no kvgen lines found)"
kill -9 $KVPID 2>/dev/null
echo "=== EXP-A DONE ==="
