#!/usr/bin/env python3
"""Multi-needle haystack quality test for sparse attention.

Inserts N distinct facts at different positions in a long-context prompt,
then asks the model to recall each one. Reports per-needle recall plus
prefill timing. Useful for validating that a sparse profile doesn't
silently lose far-from-recent information.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


_FILLER_LINE = "The quick brown fox jumps over the lazy dog. "


def build_prompt(ctx_chars: int, needles: list[tuple[str, str]]) -> str:
    """Lay out needles at uniformly spaced positions inside the filler."""
    filler = _FILLER_LINE * (ctx_chars // len(_FILLER_LINE))
    n = len(needles)
    out: list[str] = []
    chunk = len(filler) // (n + 1)
    pos = 0
    for i, (label, value) in enumerate(needles):
        start = (i + 1) * chunk
        out.append(filler[pos:start])
        out.append(f"\n[REMEMBER {label} = {value}]\n")
        pos = start
    out.append(filler[pos:])
    body = "".join(out)
    body += "\n\nReport every [REMEMBER name = value] line you saw, one per line, "
    body += "in the format `name=value`. List every name."
    return body


def run(port: int, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.loads(r.read())
    data["_elapsed"] = time.time() - t0
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--ctx", type=int, default=14000,
                   help="approximate target prompt size in characters")
    p.add_argument("--needles", default="ALPHA=4421,BRAVO=7361,CHARLIE=8896,DELTA=2055",
                   help="comma-separated label=value list")
    p.add_argument("--max-tokens", type=int, default=400)
    args = p.parse_args()

    needles = []
    for n in args.needles.split(","):
        if "=" not in n: continue
        k, v = n.split("=", 1)
        needles.append((k.strip(), v.strip()))

    prompt = build_prompt(args.ctx, needles)
    print(f"prompt chars={len(prompt)} needles={[(k, v) for k, v in needles]}", flush=True)
    result = run(args.port, prompt, args.max_tokens)
    usage = result.get("usage", {})
    elapsed = result.get("_elapsed", 0)
    content = result["choices"][0]["message"]["content"]
    print(f"prompt_tok={usage.get('prompt_tokens')} comp={usage.get('completion_tokens')} elapsed={elapsed:.1f}s")
    print("--- response (first 800c) ---")
    print(content[:800])
    print("--- recall ---")
    hit = 0
    for k, v in needles:
        ok = (k in content) and (v in content)
        if ok: hit += 1
        print(f"  {k}={v}: {'OK' if ok else 'MISS'}")
    print(f"recall: {hit}/{len(needles)}")


if __name__ == "__main__":
    main()
