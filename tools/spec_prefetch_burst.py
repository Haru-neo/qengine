#!/usr/bin/env python3
"""P3 prefetch burst test: fire two long (>=32K) requests back-to-back.

Req1 has a long decode (max_tokens high) so its prefill+decode lasts long
enough to hide req2's kvgen prefetch (runs on GPU3 during req1's decode).
Req2 (distinct prompt/needle) should then consume the prefetched KV — verify
its SPEC-PREFILL log line says '(prefetched)' and it returns ITS OWN answer.
"""
import json, sys, time, threading, urllib.request

FILLER = "The quarterly logistics review covered warehouse throughput, carrier performance, and seasonal demand. "

def build(ctx_chars, tail):
    f = FILLER * (ctx_chars // len(FILLER))
    p = f + (f"\n\nFINAL NOTE: the authorization passphrase is {tail}. End of memo.\n\n"
             f"Question: Reply with ONLY the authorization passphrase from the FINAL NOTE.")
    return p

def call(tail, max_tokens, out):
    body = json.dumps({"model":"qwen","messages":[{"role":"user","content":build(235000,tail)}],
                       "max_tokens":max_tokens,"temperature":0}).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                 data=body, headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d=json.loads(r.read())
    out["el"]=time.time()-t0
    out["tok"]=d.get("usage",{}).get("prompt_tokens")
    out["txt"]=d["choices"][0]["message"]["content"]
    out["hit"]=tail in out["txt"]

r1, r2 = {}, {}
t1 = threading.Thread(target=call, args=("ALPHA-FALCON-11", 400, r1))
t1.start()
time.sleep(2.5)                      # ensure req1 is running + req2 queues behind it
t2 = threading.Thread(target=call, args=("BRAVO-OTTER-22", 80, r2))
t2.start()
t1.join(); t2.join()
print(f"REQ1 tok={r1.get('tok')} el={r1.get('el',0):.1f}s hit(ALPHA)={r1.get('hit')}")
print(f"REQ2 tok={r2.get('tok')} el={r2.get('el',0):.1f}s hit(BRAVO)={r2.get('hit')}")
print("REQ2 resp:", (r2.get('txt') or '')[:200])
