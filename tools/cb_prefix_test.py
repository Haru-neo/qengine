#!/usr/bin/env python3
"""Per-slot prefix-cache verification.

Sends two concurrent requests with the SAME long fixed prefix but
different tails. The first request on each slot should miss the cache
(cold) and snapshot. The second request that hits the same slot with the
same first N tokens should report a [CACHE slot=K] hit in the server log
and have a much shorter prefill latency.

Usage: ./tools/cb_prefix_test.py --port 8082 --logfile /tmp/qwen-cb-test.log
"""

import argparse
import concurrent.futures as cf
import json
import time
import urllib.request


PREFIX = (
    "당신은 한국어 감정 분석 봇입니다. 입력 문장의 감정을 [긍정/부정/중립] 중 하나로 분류하고 "
    "왜 그렇게 판단했는지 한 문장으로 설명하세요. 출력 형식은 항상:\n"
    "감정: <라벨>\n이유: <한 문장>\n"
    "예시:\n입력: 오늘 날씨가 정말 좋네요!\n감정: 긍정\n이유: '정말 좋네요'라는 강한 긍정 표현이 있음.\n"
    "입력: 너무 피곤하다.\n감정: 부정\n이유: '너무 피곤'은 강한 부정적 신체 상태를 나타냄.\n"
    "이제 다음 입력을 분류하세요.\n"
)


def call(port, prompt, max_tokens, cached_prompt_tokens, idx):
    body = json.dumps({
        "model": "qwen-cb-prefix-test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "cached_prompt_tokens": cached_prompt_tokens,
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
    return idx, elapsed, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--logfile", type=str, default="/tmp/qwen-cb-test.log")
    args = ap.parse_args()

    pa = PREFIX + "입력: 오늘 회사에서 큰 프로젝트를 무사히 끝냈다.\n"
    pb = PREFIX + "입력: 잠을 한숨도 못 잤어요. 정말 짜증나네요.\n"

    print("=== prefix cache test (cold then warm) ===")

    # Cold: send pa and pb concurrently. Each lands on a different slot.
    print("--- COLD ---")
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(call, args.port, pa, 32, 256, 0)
        f2 = ex.submit(call, args.port, pb, 32, 256, 1)
        cold = sorted([f1.result(), f2.result()])
    for idx, sec, msg in cold:
        head = msg.replace("\n", " ")[:60]
        print(f"  [#{idx}] {sec:.2f}s  {head!r}")

    time.sleep(0.3)
    # Warm: same prompts again. Expect lower latency + [CACHE slot=K] hit lines.
    print("--- WARM (same prompts, expect cache hits) ---")
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(call, args.port, pa, 32, 256, 0)
        f2 = ex.submit(call, args.port, pb, 32, 256, 1)
        warm = sorted([f1.result(), f2.result()])
    for idx, sec, msg in warm:
        head = msg.replace("\n", " ")[:60]
        print(f"  [#{idx}] {sec:.2f}s  {head!r}")

    print("--- log [CACHE] lines ---")
    try:
        with open(args.logfile) as f:
            for line in f:
                if "CACHE" in line:
                    print("  " + line.rstrip())
    except FileNotFoundError:
        print(f"  (no logfile {args.logfile})")


if __name__ == "__main__":
    main()
