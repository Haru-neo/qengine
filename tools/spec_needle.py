#!/usr/bin/env python3
"""Spec-prefill needle test.

Builds a long filler prompt that triggers the auto spec-prefill path
([32768, 65536] tokens), with:
  - a TAIL needle placed in the very last lines (inside the KEEP region that
    is REAL-prefilled) -- this is the supported recall case, must pass.
  - a BODY needle ~halfway (inside the kvgen-predicted INJECT region) -- this
    is the known-approximate case; we track it to spot regressions/coherence.

Use identical --tail/--body/--ctx across an A/B (dense vs sparse kvgen,
KEEP=4096 vs 3072) so the comparison is clean. temperature=0.
"""
import argparse, json, time, urllib.request

FILLER = "The quarterly logistics review covered warehouse throughput, carrier performance, and seasonal demand. "

def build(ctx_chars, tail, body, near):
    filler = FILLER * (ctx_chars // len(FILLER))
    mid = len(filler) // 2
    body_line = f"\n[IMPORTANT RECORD: the project codename is {body}.]\n"
    # NEAR needle ~24000 chars (~3500 tok) before the end -> lands in the
    # [3072,4096]-from-end window: real-prefilled at KEEP=4096, injected at 3072.
    near_pos = max(mid + 1, len(filler) - 24000)
    near_line = f"\n[FIELD UPDATE: the shipment tracking code is {near}.]\n"
    prompt = (filler[:mid] + body_line + filler[mid:near_pos]
              + near_line + filler[near_pos:])
    prompt += (f"\n\nFINAL NOTE: the authorization passphrase is {tail}. "
               f"End of memo.\n\n"
               f"Question: On separate lines report (1) the authorization "
               f"passphrase from the FINAL NOTE, (2) the shipment tracking code "
               f"from the FIELD UPDATE, (3) the project codename from the "
               f"IMPORTANT RECORD. Give each value or write UNKNOWN if not found.")
    return prompt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ctx", type=int, default=178000, help="filler chars")
    ap.add_argument("--tail", default="CRIMSON-FALCON-77")
    ap.add_argument("--body", default="EMERALD-OTTER-42")
    ap.add_argument("--near", default="ZEBRA-COMET-91")
    ap.add_argument("--max-tokens", type=int, default=60)
    a = ap.parse_args()
    prompt = build(a.ctx, a.tail, a.body, a.near)
    payload = json.dumps({"model":"qwen","messages":[{"role":"user","content":prompt}],
                          "max_tokens":a.max_tokens,"temperature":0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",
                                 data=payload, headers={"Content-Type":"application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.loads(r.read())
    el = time.time() - t0
    u = data.get("usage", {})
    c = data["choices"][0]["message"]["content"]
    print(f"prompt_tok={u.get('prompt_tokens')} comp={u.get('completion_tokens')} elapsed={el:.1f}s")
    print("--- response ---"); print(c[:500])
    print("--- recall ---")
    print(f"  TAIL  {a.tail}: {'OK' if a.tail in c else 'MISS'}")
    print(f"  NEAR  {a.near}: {'OK' if a.near in c else 'MISS'}")
    print(f"  BODY  {a.body}: {'OK' if a.body in c else 'MISS'}")

if __name__ == "__main__":
    main()
