#!/usr/bin/env bash
# Rebuild (with the profiling drain-before-embed fix) and re-run kvgen 256K to
# disambiguate the embed phase: if embed drops from ~40s to ~1s and (wall - sum
# of phases) grows, then the old embed was the DFKI D2H mis-charged via the
# streamWaitEvent. If embed stays ~40s, it's a real kernel cost.
set -e
cd /home/paru/qwen-engine/build
echo "=== make ==="
make -j14 2>&1 | tail -12
cd /home/paru/qwen-engine
BIN=build/qwen-engine
KVGEN_MODEL=/home/paru/models/gguf/Qwen3.5-0.8B-kvgen-Q8_0.gguf
HEADS=/home/paru/models/kvgen_heads_v1.bin
LOG=/tmp/kvgen_prof2.log
rm -f "$LOG"

CUDA_VISIBLE_DEVICES=3 KVGEN_HEADS="$HEADS" KVGEN_MAX_SEQ=262144 Q8KV=1 \
  MINF_SPARSE_ATTN=1 MINF_BUDGET=0.10 MINF_UNIFORM_MS=1 KVGEN_PROFILE=1 \
  "$BIN" "$KVGEN_MODEL" --serve 8011 --mode kvgen > "$LOG" 2>&1 &
KVPID=$!
ok=0
for i in $(seq 1 90); do
  if grep -q "listening on :8011" "$LOG" 2>/dev/null; then ok=1; break; fi
  if ! kill -0 $KVPID 2>/dev/null; then echo "!! KVGEN DIED"; tail -25 "$LOG"; exit 1; fi
  sleep 2
done
if [ $ok -ne 1 ]; then echo "!! LOAD TIMEOUT"; tail -25 "$LOG"; kill -9 $KVPID 2>/dev/null; exit 1; fi
echo "=== kvgen up ==="
python3 - <<'PY'
import struct, json, socket, time, os
def make_ids(n, path):
    ids = [(1000 + (i*7919) % 40000) for i in range(n)]
    with open(path,'wb') as f: f.write(struct.pack('<%di'%n, *ids))
def post(n, keep, tag):
    ids_p=f"/dev/shm/profA2_ids_{tag}.bin"; out_p=f"/dev/shm/profA2_kv_{tag}.bin"
    make_ids(n, ids_p)
    body=json.dumps({"ids_file":ids_p,"keep":keep,"out":out_p})
    req=("POST /kvgen HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
         f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n{body}")
    s=socket.create_connection(("127.0.0.1",8011),timeout=900); t0=time.time(); s.sendall(req.encode())
    resp=b""
    while True:
        d=s.recv(8192)
        if not d: break
        resp+=d
    el=time.time()-t0; s.close()
    print(f"[EXP-A2 {tag}] n={n} WALL={el:.1f}s resp={resp.decode(errors='replace').splitlines()[-1]}", flush=True)
    for p in (ids_p,out_p):
        try: os.remove(p)
        except: pass
post(262000, 3072, "256k")
PY
echo "=== prof lines (compare embed vs old 40.4) ==="
grep -E "\[kvgen-prof\]|\[kvgen\] 2" "$LOG" || echo "(none)"
kill -9 $KVPID 2>/dev/null
echo "=== EXP-A2 DONE ==="
