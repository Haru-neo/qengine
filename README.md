# qengine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Vibe Coded](https://img.shields.io/badge/vibe%20coded-with%20Claude-8A2BE2)](https://en.wikipedia.org/wiki/Vibe_coding)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Volta sm_70](https://img.shields.io/badge/sm__70-Volta-green)](https://en.wikipedia.org/wiki/Volta_(microarchitecture))
[![Prefill: optimizing](https://img.shields.io/badge/prefill-optimizing%20%F0%9F%9A%A7-orange)](#honest-benchmarks-vs-llamacpp)

> 🚧 **Prefill is currently 2–4× slower than llama.cpp on the same hardware. Active optimization in progress — target: ≥1.5× llama.cpp prefill.** Generation already wins (+30–50%). See [Honest Benchmarks](#honest-benchmarks-vs-llamacpp).

A custom CUDA inference engine for **Qwen3 hybrid (GDN + Attention) models**, written from scratch and tuned for the cards nobody wants — NVIDIA mining cards (CMP 100-210, ex-mining V100), 16 GB HBM2, PCIe Gen1 x1, no P2P.

> A GPU-poor person's `vLLM`, **vibe-coded by a Korean high school student**. Not a fork — every kernel was written for sm_70 with mining-card constraints in mind.

📖 **한국어 README → [README.ko.md](README.ko.md)**

---

## What it does

- Serves **Qwen3.5 / Qwen3.6** dense hybrid models in **27B** and **9B** sizes (GGUF Q8_0). MoE variants (Qwen3-Moe etc.) **not supported**.
- Vision input via **Qwen3-VL mmproj** (ViT + M-RoPE + spatial reshape).
- **OpenAI-compatible HTTP API** (`/v1/chat/completions`, `/v1/models`, streaming, tool calls).
- **Continuous batching** across N concurrent slots, with per-slot prefix caching.
- **MTP draft speculative decoding (K=1)** — works. DFlash + DDTree code is in the repo but **not currently functional** (drafter mismatch — see Limitations).
- **3-bit KV cache (MTP_TQ)** — Walsh-Hadamard rotation + Lloyd-Max scalar quant. Same family of idea as llama.cpp [#21038](https://github.com/ggml-org/llama.cpp/pull/21038), but 3-bit Lloyd-Max instead of 4-bit RTN.
- Multi-GPU layer-parallel split with pinned-host activation bridge (no P2P required).

## Why this exists

Mining cards (CMP 100-210, ex-mining V100) are dirt-cheap on the secondhand market and have **HBM2 + 16 GB** of VRAM that nobody is buying back, but NVIDIA cripples them in software:

- **Tensor Cores throttled** — HMMA latency stretched 64× (8 → 512 cycles), hard cap ~5 TFLOP via cuBLAS WMMA.
- **PCIe Gen1 x1 only**, no P2P, no NVLink.
- **CUPTI blocked** — no vendor profiler, no `torch.profiler`.
- **Driver enforces all of this** via signed firmware, can't be bypassed.

So `vLLM`, `llama.cpp`'s default cuBLAS path, FlashAttention, bitsandbytes — anything that goes through cuBLAS Tensor Cores runs at 1/64 speed or fails outright.

`qengine` works around it by:

- Routing GEMM through **DP4A (int8)** at ~17 TFLOP and **HFMA2 (fp16 SIMD)** at ~24 TFLOP — these paths are *not* throttled on CMP.
- A **hand-written Q8_0 GEMM tile path** for prefill.
- A **hybrid attention layout** that avoids the strict cuBLAS path for `max_seq > 32 K`.
- **Pinned-host activation bridge** between GPUs (since P2P is unavailable).

It's not faster than llama.cpp at everything. See the honest benchmarks below.

## Honest benchmarks (vs `llama.cpp`)

All measurements on the same 4× CMP 100-210 host, same `Q8_0` GGUF, batch 1, single inflight request, `-ngl 999`, FA on, layer split. Both engines built for sm_70 with the int8 (MMQ / DP4A) path. Numbers are **server-side** (excludes streaming SSE handshake). Bigger is better.

### 9B Q8_0 — single GPU

| Prompt | qengine PP t/s | llama.cpp PP t/s | qengine TG t/s | llama.cpp TG t/s |
|---:|---:|---:|---:|---:|
| 128 | 165 | **221** | 70.4 | 46.6 |
| 512 | 161 | **265** | — | — |
| 2048 | 161 | **356** | — | — |
| 8192 | 134 | **352** | — | — |
| 18 K | 75 | — | 27 | — |

### 9B Q8_0 — dual GPU (layer split)

| Prompt | qengine PP t/s | llama.cpp PP t/s | qengine TG t/s | llama.cpp TG t/s |
|---:|---:|---:|---:|---:|
| 128 | 165 | **209** | 62 | 44 |
| 2048 | 157 | **501** | 58 | — |
| 8192 | 132 | **579** | 47 | — |

### 27B Q8_0 — 3 GPU (layer split)

| Prompt | qengine PP t/s | llama.cpp PP t/s | qengine TG t/s | llama.cpp TG t/s |
|---:|---:|---:|---:|---:|
| 128 | 38 | **83** | 26.6 | 17.8 |
| 2048 | 37 | **137** | 23.5 | — |
| 8192 | 33 | **146** | 17 | — |
| 18 K | 22 | — | 7.8 | — |

### What this says

- **Prefill: llama.cpp wins, often by 2–4×.** Their MMQ path is the product of years of contribution and we are not catching up on prefill any time soon. If your workload is prefill-heavy (many long prompts, short responses, batch > 1), use llama.cpp.
- **Generation (decode): qengine wins by ~30–50% on 9B and ~50% on 27B.** This is what users feel as "the model is responding fast" in chat / agent / coding workflows where the prompt is processed once and the response is streamed token-by-token.
- **18 K prefill on 27B is brutal (~14 min on qengine).** Heavy room to improve — likely better Q8_0 GEMM tile, batched matmul pipelining.

## Things qengine does that llama.cpp doesn't (or differs)

Honest take: most of the surface-level features overlap. The list below is what actually differs in practice. Not measured head-to-head where not stated — corrections / PRs welcome.

- **Generation throughput at sm_70 + CMP** — measured +30–50% over llama.cpp on this exact hardware. See benchmarks above.
- **OpenAI Chat Completions API built into the engine binary** — streaming, `image_url`, tool/function calls, no separate server process. llama.cpp has `llama-server` which covers most of this.
- **MTP_TQ uses 3-bit Lloyd-Max + WHT** — llama.cpp's [#21038](https://github.com/ggml-org/llama.cpp/pull/21038) already lands rotation + standard scalar quant types (q4_0 etc.). Ours is 3-bit (vs 4-bit), which gives a slightly higher compression ratio on KV. Whether this beats q4_0 RTN on perplexity is **not yet verified head-to-head**.
- **Continuous batching with per-slot prefix snapshots** — not unique conceptually. The integration with our scheduler is tight; whether it actually beats llama.cpp's batched server is **not yet measured**.
- **Qwen3-VL multimodal** — we have it. So does llama.cpp via `tools/mtmd/models/qwen3vl.cpp`. Not an advantage.
- **DFlash + DDTree speculative decode (experimental, currently broken)** — z-lab pretrained drafter mismatches Qwopus3.6 distill distribution; produces degenerate output. Listed for transparency, not as a feature. Requires drafter fine-tune to be usable.

## Hardware

**Designed for / regularly tested on:**
- 4× NVIDIA **CMP 100-210** (Volta GV100, 16 GB HBM2, sm_70, PCIe Gen1 x1, no P2P)
- Total 64 GB VRAM, ~8 GB system RAM (yes, eight)

**Should also work on (sm_70 / sm_72 / sm_75):**
- V100 16/32 GB (much less throttled than CMP — should be faster)
- Titan V, Quadro GV100
- T4, RTX 20-series (sm_75) — untested, kernels target sm_70 paths

**Will not work on:**
- sm_60 or earlier (no DP4A)
- AMD / Apple Silicon

If you have a modern GPU (RTX 30/40/50, A100, H100), you should use **vLLM** or **SGLang** instead. They are far more optimized for those targets and have actual test coverage.

## Build

Requires CUDA 12.x, GCC 11+, CMake 3.18+.

```bash
git clone https://github.com/Haru-neo/qengine.git
cd qengine
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Run

### 27B server with vision (256 K context, 3-bit KV)

```bash
MTP_TQ=1 ./build/qwen-engine \
  /path/to/Qwen3.6-27B-Q8_0.gguf \
  --serve 8000 --max-seq 262144 --slots 1 \
  --vision-mmproj /path/to/Qwen3.6-27B-mmproj.gguf
```

### 9B server with 4-way continuous batching

```bash
QWEN_SLOTS=4 ./build/qwen-engine \
  /path/to/Qwen3.5-9B-Q8_0.gguf \
  --serve 8001 --max-seq 32768
```

### Pin to a subset of GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1 ./build/qwen-engine ... --serve 8001
```

### Optional MTP draft head

If `mtp_head_<hidden>.bin` exists (set with `--mtp-head <path>` or default `./mtp_work/mtp_head_<hidden>.bin`), the engine will load it for K=1 speculative decoding. Without it the engine runs plain greedy / continuous-batched.

### Call it

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role":"user","content":"hello"}],
    "max_tokens": 256
  }'
```

Vision (27B with mmproj):

```bash
B64=$(base64 -w 0 image.png)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\":\"qwen\",
    \"messages\":[{\"role\":\"user\",\"content\":[
      {\"type\":\"text\",\"text\":\"What is in this image?\"},
      {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,${B64}\"}}
    ]}]
  }"
```

## Environment variables

| Var | Default | Effect |
|---|---|---|
| `MTP_TQ` | `0` | 3-bit KV cache (WHT + Lloyd-Max). Required for 27B at 256 K. |
| `FLASH_ATTN` | auto | `1` forces FA, `0` disables. Auto-on if `max_seq > 32 K`. |
| `QWEN_SLOTS` | `1` | Concurrent slots (continuous batching). Set via `--slots` too. |
| `QWEN_MAX_QUEUE` | `64` | Max queued requests; `0` = unbounded. |
| `BIT_EXACT_GEMM_ON` | `0` | Use bit-exact column-wise GEMV path (for regression tests; slower). |
| `MTP_ACCEPT_TOP2` | `0` | MTP K=2 top-2 verify (small accept rate gain). |
| `CUDA_VISIBLE_DEVICES` | — | Standard CUDA mask; engine splits layers across visible GPUs. |

## Architecture (in 90 seconds)

```
src/
  main.cu             entry, weight load, generation loops, OpenAI server glue
  server.h            HTTP/1.1 + SSE, OpenAI compat parsing
  scheduler.h         continuous batching, queue, cancel propagation
  model.cuh           QwenModel: forward, multi-GPU dispatch, KV state
  tokenizer.h         BPE tokenizer, chat template, <think> strip
  ops.cuh             RMSNorm, SiLU, residual, embedding dequant
  attention.cuh       RoPE, head norm, scoring, softmax, value, KV cache
  gdn_kernels.cuh     Conv1d, GDN recurrent step, output projection
  mtp_head.cuh        MTP draft head (K=1, K=2 opt-in)
  dflash_*.cuh        DFlash + DDTree speculative path (experimental)
  vision.cuh          Qwen3-VL ViT + M-RoPE + spatial reshape + splice
  quant_gemv.cuh      Q5_K / Q6_K / Q8_0 GEMV kernels (DP4A path)
  q8_0_gemm.cuh       Q8_0 GEMM tile path (default for prefill)
  turboquant.cuh      WHT + Lloyd-Max 3-bit KV (MTP_TQ)
  gguf.h              GGUF v3 parser, mmap loader
  gpu_loader.h        multi-GPU parallel weight load (thread pool + streams)
  sampling.h          top-p / top-k / min-p / rep-pen / freq-pen / pres-pen
```

Layer split (4-GPU 27B example): GPU 0 holds layers 0–15 + token embeddings; GPU 3 holds layers 48–63 + output norm + LM head; activations bounce through pinned host memory between GPUs.

## Limitations & known issues

- **MoE not supported.** Only dense Qwen3 hybrid (GDN + Attention) models — Qwen3-Moe and similar mixture-of-experts variants do not load.
- **DFlash + DDTree spec decode is currently broken.** Pretrained drafter (`lucebox-hub/dflash`) is trained on stock Qwen3.5; output distribution doesn't match the Qwopus distill we use, so accept rate ≈ 0% and the chains degenerate. Code is in the repo for the eventual fine-tuned drafter, but as shipped this path is unusable.
- **No batched MTP / spec.** Speculative paths run only when `slots == 1`. With `slots > 1`, the batched gen loop is plain greedy.
- **GGUF Q8_0 is the supported path.** Q5_K_M / Q6_K load but quality is degraded — use Q8_0.
- **sm_70 specific tuning.** Should run on sm_75; sm_80+ has better engines anyway.
- **Single-host.** No tensor parallelism across machines, no multi-node.
- **Linux only.**
- **Prefill is 2–4× slower than llama.cpp.** Biggest known weakness. Active work targeting ≥1.5× llama.cpp prefill (= flipping the gap to a lead). Likely paths: better Q8_0 GEMM tile, batched matmul pipelining, multi-stage SMEM, fused QKV. PRs welcome.
- **Continuous batching with system-prompt-less requests can stop after 1 token** on Qwopus distill models — known issue with empty-system-prompt EOS bias under batched gen. Set `--default-system-prompt`.

## Status

Active personal project. APIs and env vars may change. Issues / PRs welcome but expect slow turnaround — solo project.

## Acknowledgements

- **Qwen team** for the Qwen3 / Qwen3-VL model family and architecture.
- **`llama.cpp`** for the GGUF format, reference quant kernels, and the cargo of measurement / quantization research the broader community has produced. Particularly [#21038](https://github.com/ggml-org/llama.cpp/pull/21038) (rotation for KV quant) which arrived ahead of our MTP_TQ work.
- **`stb_image.h`** (public domain) for image decode in the vision path.
- **TurboQuant**, **DFlash + DDTree** speculative decoding (`lucebox-hub/dflash`) as experimental references.
- **Anthropic Claude** — see authorship below.

## Authorship

This codebase was built via [vibe coding](https://en.wikipedia.org/wiki/Vibe_coding) — code authored primarily by Claude (Anthropic) across many sessions, with direction, debugging, architecture decisions, kernel verification, and hardware reverse-engineering by **HARU-Neo** ([@Haru-neo](https://github.com/Haru-neo)) — a Korean high school student. Bugs are mine; clever kernels are Claude's.

## License

Apache 2.0 — see [LICENSE](LICENSE).
