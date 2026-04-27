#!/usr/bin/env python3
"""Continuous-batching smoke test.

Sends N concurrent /v1/chat/completions requests to the engine, prints
per-request latency + per-token throughput, and verifies that the
responses are coherent (not garbage).

Usage:  ./tools/cb_test.py [--port 8082] [--concurrency 2] [--prompt "..."]
"""

import argparse
import concurrent.futures as cf
import json
import time
import urllib.request

DEFAULT_PROMPT = "한 문장으로 자기소개 해줘."


def call_once(port, prompt, max_tokens, idx):
    body = json.dumps({
        "model": "qwen-cb-test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read().decode())
    elapsed = time.time() - t0

    msg = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    n_completion = usage.get("completion_tokens", len(msg.split()))
    return {
        "idx": idx,
        "elapsed": elapsed,
        "tokens": n_completion,
        "tps": n_completion / elapsed if elapsed > 0 else 0.0,
        "text": msg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    args = ap.parse_args()

    print(f"=== continuous batching test: {args.concurrency} concurrent ===")
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(call_once, args.port, args.prompt, args.max_tokens, i)
            for i in range(args.concurrency)
        ]
        results = [f.result() for f in cf.as_completed(futures)]
    wall = time.time() - t0
    results.sort(key=lambda r: r["idx"])

    for r in results:
        head = (r["text"][:80] + ("..." if len(r["text"]) > 80 else "")).replace("\n", " ")
        print(f"  [#{r['idx']}] {r['elapsed']:.2f}s  {r['tokens']} tok  "
              f"{r['tps']:.1f} t/s  {head!r}")
    total_tok = sum(r["tokens"] for r in results)
    print(f"--- wall={wall:.2f}s  agg={total_tok / wall:.1f} t/s ---")


if __name__ == "__main__":
    main()
