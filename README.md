# qengine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Volta sm_70](https://img.shields.io/badge/sm__70-Volta-green)](https://en.wikipedia.org/wiki/Volta_(microarchitecture))
[![Prefill: 1.2–3× llama.cpp](https://img.shields.io/badge/prefill-1.2%E2%80%933%C3%97%20llama.cpp-brightgreen)](#honest-benchmarks-vs-llamacpp)

> **9B single-GPU prefill is 1.5–3× llama.cpp at every length up to ~4.6 K tokens** (2026-05-02 default flip, commit `ca30368`). With **split-K FlashAttention** on by default (commit `f2e52b8`), 9B 18 K prefill jumps to **1.22× llama.cpp** and 27B 3-GPU 18 K reaches **parity (0.99×)**. Generation continues to win by +30–50%. Multi-GPU prefill at long context still trails on 9B 2-GPU (open work). See [Honest Benchmarks](#honest-benchmarks-vs-llamacpp).

A custom CUDA inference engine for **Qwen3.5 / Qwen3.6 hybrid (GDN + Attention) models**, written from scratch and tuned for NVIDIA mining cards (CMP 100-210, ex-mining V100) — 16 GB HBM2, sm_70, PCIe Gen1 x1, no P2P. Not a fork — every kernel is written for these constraints.

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
- **All of this is enforced in hardware** — e-fuse + PMU bootrom double-lock on the die. There is no software unlock; we tried.

So `vLLM`, `llama.cpp`'s default cuBLAS path, FlashAttention, bitsandbytes — anything that goes through cuBLAS Tensor Cores runs at 1/64 speed or fails outright.

`qengine` works around it by:

- Routing GEMM through **DP4A (int8)** at ~17 TFLOP and **HFMA2 (fp16 SIMD)** at ~24 TFLOP — these paths are *not* throttled on CMP.
- A **hand-written Q8_0 GEMM tile path** for prefill.
- A **hybrid attention layout** that avoids the strict cuBLAS path for `max_seq > 32 K`.
- **Pinned-host activation bridge** between GPUs (since P2P is unavailable).

It's not faster than llama.cpp at everything. See the honest benchmarks below.

## Honest benchmarks (vs `llama.cpp`)

All measurements on a CMP 100-210 host, same `Q8_0` GGUFs (Qwopus3.5-9B-v3.5, Qwopus3.6-27B-v1-preview), batch 1, single inflight request, FA on, layer split, split-K FA on by default. Both engines built for sm_70 with the int8 (MMQ / DP4A) path. qengine numbers are **server-side prefill wall** (excludes SSE handshake) for `bench_curl.sh`-style real chat-completion prompts; `llama.cpp` numbers are `llama-bench` at matching prompt sizes. Bigger is better; **bold** = winner. Single-GPU 9B measured 2026-05-02; 27B 3-GPU and split-K 9B 18 K measured 2026-05-03. `llama.cpp` build `8462`.

### 9B Q8_0 — single GPU

| Prompt | qengine PP t/s | llama.cpp PP t/s | qengine TG t/s | llama.cpp TG t/s |
|---:|---:|---:|---:|---:|
| 297 | **594** | 199 | 70.4 | — |
| 1.16 K | **683** | 316 | — | — |
| 4.62 K | **584** | 361 | — | — |
| 18.4 K | **393** | 324 | 27.6 | — |
| tg64 | — | — | — | 46.6 |

`qengine`: 1.56–2.99× on prompts up to ~4.6 K, **1.22×** at 18 K (split-K FA, 1.46× over the pre-split-K build). Generation +51% on the comparable short-context point (70.4 vs 46.6 t/s).

### 9B Q8_0 — dual GPU (layer split)

| Prompt | qengine PP t/s | llama.cpp PP t/s | qengine TG t/s | llama.cpp TG t/s |
|---:|---:|---:|---:|---:|
| 297 | **495** | 188 | 68.6 | — |
| 1.16 K | **611** | 412 | — | — |
| 4.62 K | 519 | **574** | — | — |
| 18.4 K | 259 | **545** | 27.4 | — |
| tg64 | — | — | — | 44.2 |

Multi-GPU is still `llama.cpp`'s strong suit at long context — its layer pipeline overlaps activation transfer with compute and roughly doubles long-prompt throughput from single GPU. Our pinned-host bridge between GPUs is sequential, so 9B multi-GPU doesn't speed prefill at 18 K. (At 9B sizes a single GPU is already faster, so multi-GPU is mostly a 27B story.) Open work item — these numbers predate split-K, so the gap may shrink.

### 27B Q8_0 — 3 GPU (layer split, 2026-05-03)

| Prompt | qengine PP t/s | llama.cpp PP t/s | qengine TG t/s | llama.cpp TG t/s |
|---:|---:|---:|---:|---:|
| 297 | **185** | 74.2 | 26.3 | — |
| 1.16 K | **200** | 127.8 | 23.1 | — |
| 4.62 K | **188** | 146.0 | 16.1 | — |
| 18.4 K | 139 | 140.0 | 7.7 | — |
| tg128 | — | — | — | 17.7 |

`qengine`: **2.49× / 1.57× / 1.29×** at 297 / 1.16 K / 4.62 K, **0.99× (parity) at 18 K** with split-K FA. Generation +48% at 297 ctx (26.3 vs 17.7 t/s).

### What this says

- **9B single-GPU prefill: qengine wins at every length now**, 1.22–2.99×. The chat-app sweet spot.
- **27B 3-GPU prefill: qengine wins ≤ 4.6 K (1.29–2.49×), parity at 18 K.** Same code paths as 9B; the default flip + split-K closed the long-context gap.
- **9B dual-GPU 18 K still trails llama.cpp.** Their layer pipeline carries here; ours is sequential. Open work, but at 9B the single-GPU run is already faster than either dual-GPU result.
- **Generation throughput: qengine wins by ~30–50%** on both 9B and 27B. This is what users feel as the chat being responsive.

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
| `FLASH_ATTN` | `1` | FA fused score+softmax+value. `0` falls back to the strict block-per-score path (bit-exact with per-token, ~2× slower prefill). |
| `BIT_EXACT_GEMM_ON` | `0` | Use the strict column-wise GEMV reduction path instead of the GEMM tile (regression / bit-exact testing, ~2.4× slower prefill). |
| `FA_BM` | `32` | FA tile width. `64` halves K/V tile-load iterations (96 KB SMEM opt-in). Marginal on the prompts we measured. |
| `FA_NT` | `1` | Per-block t_idx count. `2` shares K/V tile across 2 query rows; currently 14% slower at long context (kept as infra). |
| `FA_SK` | `4` | FA split-K factor at sub_seq_total ≥ 4 K. Spreads each (kv_head, t_idx) across N blocks (default 4) merged via log-sum-exp; lifts long-prompt prefill ~1.46× on 9B and ~1.34× on 27B. `0` to opt out. fp32 partials keep argmax bit-stable with the base FA path. |
| `QWEN_SLOTS` | `1` | Concurrent slots (continuous batching). Set via `--slots` too. |
| `QWEN_MAX_QUEUE` | `64` | Max queued requests; `0` = unbounded. |
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
- **9B dual-GPU 18 K prefill still trails `llama.cpp`** (~0.48× pre-split-K). Their layer pipeline overlaps activation transfer with compute; ours is sequential. Single-GPU 9B is already faster than either dual-GPU run, so this matters mostly as a multi-GPU correctness benchmark, not a perf path users hit. 27B 3-GPU 18 K is at parity now; closing the 9B 2-GPU gap means pipelining the pinned-host activation bridge. PRs welcome.
- **Continuous batching with system-prompt-less requests can stop after 1 token** on Qwopus distill models — known issue with empty-system-prompt EOS bias under batched gen. Set `--default-system-prompt`.

## Status

Active personal project. APIs and env vars may change. Issues / PRs welcome but expect slow turnaround — solo project.

## Acknowledgements

- **Qwen team** for the Qwen3 / Qwen3-VL model family and architecture.
- **`llama.cpp`** for the GGUF format, reference quant kernels, and the cargo of measurement / quantization research the broader community has produced. Particularly [#21038](https://github.com/ggml-org/llama.cpp/pull/21038) (rotation for KV quant) which arrived ahead of our MTP_TQ work.
- **`stb_image.h`** (public domain) for image decode in the vision path.
- **TurboQuant**, **DFlash + DDTree** speculative decoding (`lucebox-hub/dflash`) as experimental references.
- **Anthropic Claude** — kernel implementation and CUDA work across many sessions.

## License

Apache 2.0 — see [LICENSE](LICENSE).
