#!/usr/bin/env python3
"""Continuous-batching SSE streaming smoke test.

Sends N concurrent /v1/chat/completions stream=true requests, prints
incremental chunk timings per slot, and verifies that streaming chunks
arrive interleaved (not strictly per-slot serialized).

Usage:  ./tools/cb_stream_test.py [--port 8082] [--concurrency 2]
"""

import argparse
import concurrent.futures as cf
import json
import time
import urllib.request

DIFFERENT_PROMPTS = [
    "한 문장으로 자기소개 해줘.",
    "파이썬에서 리스트와 튜플의 차이를 한 줄로.",
]


def stream_once(port, prompt, max_tokens, idx):
    body = json.dumps({
        "model": "qwen-cb-stream-test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    chunks = []
    text = ""
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            delta = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta:
                chunks.append((time.time() - t0, delta))
                text += delta
    elapsed = time.time() - t0
    return {
        "idx": idx,
        "elapsed": elapsed,
        "chunks": chunks,
        "text": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=120)
    args = ap.parse_args()

    prompts = [DIFFERENT_PROMPTS[i % len(DIFFERENT_PROMPTS)]
               for i in range(args.concurrency)]
    print(f"=== streaming continuous batching test: {args.concurrency} concurrent ===")

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(stream_once, args.port, prompts[i],
                             args.max_tokens, i) for i in range(args.concurrency)]
        results = [f.result() for f in cf.as_completed(futures)]
    wall = time.time() - t0
    results.sort(key=lambda r: r["idx"])

    # Per-slot summary
    for r in results:
        head = r["text"][:80].replace("\n", " ")
        first = r["chunks"][0][0] if r["chunks"] else 0
        print(f"  [#{r['idx']}] elapsed={r['elapsed']:.2f}s  ttft={first*1000:.0f}ms  "
              f"chunks={len(r['chunks'])}  head={head!r}")

    # Interleave check: walk all chunks ordered by arrival time, count
    # alternation between slots. If batched gen interleaves correctly we
    # expect roughly even chunk counts and alternating arrivals.
    all_chunks = []
    for r in results:
        for ts, _ in r["chunks"]:
            all_chunks.append((ts, r["idx"]))
    all_chunks.sort()
    if len(all_chunks) > 1:
        switches = sum(1 for i in range(1, len(all_chunks))
                       if all_chunks[i][1] != all_chunks[i - 1][1])
        print(f"  interleave: {switches} slot-switches over {len(all_chunks)} chunks "
              f"({100.0 * switches / (len(all_chunks) - 1):.0f}% switch rate)")
    print(f"--- wall={wall:.2f}s ---")


if __name__ == "__main__":
    main()
