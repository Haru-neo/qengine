# qengine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Volta sm_70](https://img.shields.io/badge/sm__70-Volta-green)](https://en.wikipedia.org/wiki/Volta_(microarchitecture))
[![Prefill: 1.2–3× llama.cpp](https://img.shields.io/badge/prefill-1.2%E2%80%933%C3%97%20llama.cpp-brightgreen)](#honest-benchmarks-vs-llamacpp)

> **9B single-GPU prefill is 1.5–3× llama.cpp at every length up to ~4.6 K tokens** (2026-05-02 default flip, commit `ca30368`). With **split-K FlashAttention** on by default (commit `f2e52b8`), 9B 18 K prefill jumps to **1.22× llama.cpp** and 27B 3-GPU 18 K reaches **parity (0.99×)**. Generation continues to win by +30–50%. **Multi-GPU prefill is now pipelined** (per-GPU compute / D2H / H2D streams + double-buffered hidden chunks): 9B dual-GPU 18 K hits **1.32× llama.cpp** and 27B 3-GPU 18 K hits **1.45×** — the long-context multi-GPU gap is closed. See [Honest Benchmarks](#honest-benchmarks-vs-llamacpp).

A custom CUDA inference engine for **Qwen3.5 / Qwen3.6 hybrid (GDN + Attention) models**, written from scratch and tuned for NVIDIA mining cards (CMP 100-210, ex-mining V100) — 16 GB HBM2, sm_70, PCIe Gen1 x1, no P2P. Not a fork — every kernel is written for these constraints.

📖 **한국어 README → [README.ko.md](README.ko.md)**

> **A note on the writing:** I'm not a native English speaker — this README was written in Korean first and translated/polished with an LLM. All of the engineering, kernels, benchmarks, and decisions are my own.

---

## What it does

- Serves **Qwen3.5 / Qwen3.6** dense hybrid models in **27B** and **9B** sizes (GGUF Q8_0). MoE variants (Qwen3-Moe etc.) **not supported**.
- **v2-MTP inline GGUF support** (2026-05-27): nextn head packed as `blk.N.nextn.*` in the GGUF is auto-detected and used as the spec drafter — no external `mtp_head_*.bin` needed. Accept rate jumped 73 → 83 % on Qwopus3.6-27B vs the legacy external head.
- Vision input via **Qwen3-VL mmproj** (ViT + M-RoPE + spatial reshape).
- **OpenAI-compatible HTTP API** (`/v1/chat/completions`, `/v1/models`, streaming, tool calls).
- **Embeddings + reranking** from the same binary (`--mode embed` / `--mode rerank`): Qwen3-Embedding-4B (last-token pool, L2-normalized) and Qwen3-Reranker-4B (cross-encoder via the `cls.output` classifier head, instruction-aware). The chat server reverse-proxies `/v1/embeddings` and `/v1/rerank` to these sidecars.
- **Qwen3 thinking control**: `/no_think` (and `/think`) directives in user **or** system messages, plus vLLM-style `extra_body.chat_template_kwargs.enable_thinking` — all resolve to the same `force_think` switch in the chat template.
- **Continuous batching** across N concurrent slots, with per-slot prefix caching.
- **MTP draft speculative decoding (K=1)** — works. **DFlash block-diffusion speculative decoding** works too now (lossless drafting + chain tree-verify). With a distribution-matched drafter it runs in the **mid-30s to upper-40s t/s** on 27B Q8_0 / 3-GPU. A **pipelined-fold** path (default-on; `DFLASH_FOLD=0` to disable) folds the bonus token into the verify batch for **+15–22%** — quality-equivalent (verified: identical compiled-C++ coding pass-rate) but **not bit-identical** (see Limitations).
- **3-bit KV cache (MTP_TQ)** — Walsh-Hadamard rotation + Lloyd-Max scalar quant. Same family of idea as llama.cpp [#21038](https://github.com/ggml-org/llama.cpp/pull/21038), but 3-bit Lloyd-Max instead of 4-bit RTN.
- **Fused gate+up GEMM** (`MLP_GATEUP_FUSED_KERNEL=1`, default off): two Q8_0 weights share one Q8_1 input tile in SMEM, +5–10 % prefill on v2 models.
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

### 9B Q8_0 — dual GPU (layer split, prefill pipelining on)

| Prompt | qengine PP t/s | llama.cpp PP t/s | qengine TG t/s | llama.cpp TG t/s |
|---:|---:|---:|---:|---:|
| 297 | **594** | 188 | 68.4 | — |
| 1.16 K | **968** | 412 | 66.7 | — |
| 4.62 K | **982** | 574 | 50.8 | — |
| 18.4 K | **720** | 545 | 26.4 | — |
| tg64 | — | — | — | 44.2 |

Cross-GPU prefill pipelining lands 2026-05-03 (default on; `PREFILL_NO_PIPELINE=1` to opt out): per-GPU compute / D2H / H2D streams + double-buffered host transfer + double-buffered per-GPU hidden chunks overlap chunk i's cross-GPU activation transfer with chunk i+1's downstream compute. 9B dual-GPU 18 K jumps from 259 → **720 t/s (2.78×)** and now wins llama.cpp at every length (1.32× even at 18 K). Sampled tokens are bit-equivalent to the sequential path — verified against the per-token greedy argmax up to 18 K.

### 27B Q8_0 — 3 GPU (layer split, prefill pipelining on, 2026-05-03)

| Prompt | qengine PP t/s | llama.cpp PP t/s | qengine TG t/s | llama.cpp TG t/s |
|---:|---:|---:|---:|---:|
| 297 | **212** | 74.2 | 27.1 | — |
| 1.16 K | **264** | 127.8 | 25.6 | — |
| 4.62 K | **268** | 146.0 | 20.4 | — |
| 18.4 K | **203** | 140.0 | 11.4 | — |
| tg128 | — | — | — | 17.7 |

`qengine`: **2.86× / 2.07× / 1.84× / 1.45×** at 297 / 1.16 K / 4.62 K / 18 K — pipelining lifts the long-context number from parity (139, split-K only) to a clean 1.45× win. Generation +53% at 297 ctx (27.1 vs 17.7 t/s).

### What this says

- **9B single-GPU prefill: qengine wins at every length now**, 1.22–2.99×. The chat-app sweet spot.
- **9B dual-GPU prefill: qengine now wins at every length** (1.32× at 18 K) thanks to cross-GPU pipelining.
- **27B 3-GPU prefill: qengine wins everywhere**, 1.45–2.86×. The parity gap at 18 K is gone.
- **Generation throughput: qengine wins by ~30–50%** on both 9B and 27B. This is what users feel as the chat being responsive.

## Things qengine does that llama.cpp doesn't (or differs)

Honest take: most of the surface-level features overlap. The list below is what actually differs in practice. Not measured head-to-head where not stated — corrections / PRs welcome.

- **Generation throughput at sm_70 + CMP** — measured +30–50% over llama.cpp on this exact hardware. See benchmarks above.
- **OpenAI Chat Completions API built into the engine binary** — streaming, `image_url`, tool/function calls, no separate server process. llama.cpp has `llama-server` which covers most of this.
- **MTP_TQ uses 3-bit Lloyd-Max + WHT** — llama.cpp's [#21038](https://github.com/ggml-org/llama.cpp/pull/21038) already lands rotation + standard scalar quant types (q4_0 etc.). Ours is 3-bit (vs 4-bit), which gives a slightly higher compression ratio on KV. Whether this beats q4_0 RTN on perplexity is **not yet verified head-to-head**.
- **Continuous batching with per-slot prefix snapshots** — not unique conceptually. The integration with our scheduler is tight; whether it actually beats llama.cpp's batched server is **not yet measured**.
- **Qwen3-VL multimodal** — we have it. So does llama.cpp via `tools/mtmd/models/qwen3vl.cpp`. Not an advantage.
- **DFlash block-diffusion speculative decoding** — functional and lossless on the engine side (block-diffusion drafter + chain tree-verify, with custom dp4a small-batch GEMV and an SMEM-resident GDN chain-scan for the verify). The catch is the **drafter**: a stock drafter trained on vanilla Qwen3.5 mismatches a distilled model's output distribution (low accept rate), so you need one trained/fine-tuned on **your** model's outputs. With a matched drafter, accept length ≈ 4.5–5 and the **pipelined-fold** path (default) adds another +15–22%.

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

Recommended config — **3 GPUs** (12 % faster prefill than 4 since transfer is one hop shorter; the v2 Q8_0 28 GB model fits in 3×16 GB with the rebalanced split) + fused gate+up + sparse-attention profile:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
MTP_TQ=1 MLP_GATEUP_FUSED=1 MLP_GATEUP_FUSED_KERNEL=1 \
MINF_SPARSE_ATTN=1 MINF_BUDGET=0.10 \
MINF_PROFILE_PATH=./profiles/27B_block_sparse.bin \
./build/qwen-engine \
  /path/to/Qwopus3.6-27B-v2-MTP-Q8_0.gguf \
  --serve 8000 --max-seq 262144 --slots 1 \
  --vision-mmproj /path/to/Qwopus3.6-27B-v2-mmproj.gguf
```

For v1 models (external MTP head): drop `MLP_GATEUP_FUSED_KERNEL=1` (the fused kernel only matches the v2 numerics in our testing) and place `mtp_head_<hidden>.bin` under `/home/paru/mtp_work/`.

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

### Embedding + reranker sidecars

The same binary serves Qwen3 dense 4B embedding and reranker models via
`--mode embed` and `--mode rerank`. Both 4B Q8_0 GGUFs fit on one GPU
(~4 GB each), so a spare CMP card can host both sidecars. The chat server
reverse-proxies `/v1/embeddings` and `/v1/rerank` to the sidecars so
clients only need to know about the chat port:

```bash
# embed sidecar (port 8001, GPU 3)
CUDA_VISIBLE_DEVICES=3 ./build/qwen-engine \
  /path/to/Qwen3-Embedding-4B-Q8_0.gguf --serve 8001 --mode embed

# rerank sidecar (port 8002, GPU 3 — shared with embed).
# Stagger: wait for the embed sidecar to finish loading before launching
# this one. Bringing up two CUDA contexts on the SAME GPU simultaneously
# intermittently corrupts the second one on CMP 100-210.
CUDA_VISIBLE_DEVICES=3 ./build/qwen-engine \
  /path/to/Qwen3-Reranker-4B-Q8_0.gguf --serve 8002 --mode rerank

# chat with proxy (clients hit only :8000)
./build/qwen-engine /path/to/27B-Q8_0.gguf --serve 8000 \
  --proxy-embed  127.0.0.1:8001 \
  --proxy-rerank 127.0.0.1:8002 \
  ...
```

A turnkey launcher that brings up all three (with the embed→rerank stagger
baked in) is at `scripts/launch_all.sh`.

OpenAI-compatible embedding request:
```bash
curl http://localhost:8000/v1/embeddings -H "Content-Type: application/json" \
  -d '{"model":"qwen-embed","input":["the cat sat","a feline rested"]}'
# → {"object":"list","data":[{"embedding":[…2560 floats…], …}, …]}
```

Rerank request (returns documents sorted by relevance):
```bash
curl http://localhost:8000/v1/rerank -H "Content-Type: application/json" \
  -d '{"query":"capital of Korea",
       "documents":["Seoul is the capital of South Korea.",
                    "Paris is in France.",
                    "Beijing is in China."]}'
# → {"results":[{"index":0,"relevance_score":0.997,"document":"Seoul …"}, …]}
```

The reranker is a true **cross-encoder**: it feeds each `query`+`document`
pair through the model under the official Qwen3-Reranker template
(`<Instruct>/<Query>/<Document>`) and reads the **`cls.output` classifier
head** — `relevance_score = softmax([yes_logit, no_logit])[yes]`. This is
the same path llama.cpp's `--reranking` uses, and it's far more
discriminative than reading raw yes/no token logits off the LM head.

Optional **instruction** field (Qwen3-Reranker is instruction-aware — a
clear task description sharpens judgments, especially for non-English docs):

```bash
curl http://localhost:8000/v1/rerank -H "Content-Type: application/json" \
  -d '{"instruction":"Given a question, find the document that answers it",
       "query":"How do I build a voice assistant?",
       "documents":["Combine an LLM with speech-to-text and TTS.",
                    "JARVIS is a character from Iron Man."]}'
```

> **Building the 4B reranker GGUF:** most community Qwen3-Reranker-4B GGUFs
> drop the `cls.output.weight` classifier head during conversion, which
> breaks this path (the model just emits "Okay"). Convert from the official
> `Qwen/Qwen3-Reranker-4B` safetensors with a recent `convert_hf_to_gguf.py`
> (it auto-detects the reranker and synthesizes the cls head), then quantize
> with `--tensor-type "cls.output=f16"` so the tiny head keeps full
> precision while the rest goes Q8_0.

> **Non-ASCII note:** clients that serialize JSON with `ensure_ascii=True`
> (Python `requests` does by default) send Korean/CJK as `\uXXXX` escapes.
> The server decodes these to UTF-8 in all string-extraction paths — earlier
> builds only decoded the `query`, so non-ASCII *documents* scored as
> gibberish. Fixed; no client-side workaround needed.

### MTP draft head (speculative decoding)

The engine looks for the MTP nextn head in two places, in priority order:

1. **Inline GGUF (v2 models)** — if the last block (e.g. `blk.64.nextn.eh_proj.weight`) carries nextn tensors, they're used directly from the already-GPU-resident weights. Matches the model's training distribution exactly; no external file needed. Measured accept rate **~83 %** on Qwopus3.6-27B-v2-MTP.
2. **External binary (v1 fallback)** — `mtp_head_<hidden>.bin` at `--mtp-head <path>` (default `/home/paru/mtp_work/mtp_head_<hidden>.bin`). Use this only when the GGUF has no inline nextn (e.g. legacy v1 GGUFs).

If neither is found, the engine runs plain greedy / continuous-batched.

### Thinking control (Qwen3 `/no_think`)

All three trigger forms work; later wins on conflict:

- Literal `/no_think` or `/think` anywhere in user OR system messages (the directive text is stripped from the prompt before encoding so the model doesn't see it).
- vLLM-compatible `extra_body.chat_template_kwargs.enable_thinking: false`.
- Top-level `chat_template_kwargs.enable_thinking: false` (some clients).

Internally these all set the same `force_think` argument on the chat template (`-1` = insert empty `<think></think>` block, `0` = let the model decide, `1` = prefill `<think>\n`).

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
| `MTP_TQ` | `0` | 3-bit KV cache (WHT + Lloyd-Max). Required for 27B at 256 K. Tradeoff: gen TG drops at long context (18 K: 11.4 → 7.6 t/s on 27B 3-GPU) because per-token attention pays the dequant cost. Set when you need >32 K context; leave off for fastest gen at smaller windows. |
| `MLP_GATEUP_FUSED` | `0` | Route ffn_gate + ffn_up through a single dispatcher (no behavior change; sets up the fused-kernel path). Pair with `MLP_GATEUP_FUSED_KERNEL=1`. |
| `MLP_GATEUP_FUSED_KERNEL` | `0` | Run the actual fused Q8_0 GEMM that shares one Q8_1 input tile in SMEM across both weights. +5–10 % prefill on v2 models. Verified bit-stable on Qwopus3.6-27B-v2; older v1 models showed argmax drift in earlier testing — leave OFF unless your model is v2-MTP. |
| `MINF_SPARSE_ATTN` | `0` | Block-sparse FA path (MInference port). Requires a profile binary at `MINF_PROFILE_PATH` describing per-layer sparsity patterns. ~10 % prefill win at 23 K on 27B with no measurable quality loss. |
| `MINF_BUDGET` | `0.10` | Block budget for `MINF_SPARSE_ATTN` (fraction of K/V blocks kept per Q row). |
| `MINF_PROFILE_PATH` | unset | Path to the offline sparsity profile (e.g. `profiles/27B_block_sparse.bin`). Without this, `MINF_SPARSE_ATTN=1` is a no-op. |
| `FLASH_ATTN` | `1` | FA fused score+softmax+value. `0` falls back to the strict block-per-score path (bit-exact with per-token, ~2× slower prefill). |
| `BIT_EXACT_GEMM_ON` | `0` | Use the strict column-wise GEMV reduction path instead of the GEMM tile (regression / bit-exact testing, ~2.4× slower prefill). |
| `FA_BM` | `32` | FA tile width. `64` halves K/V tile-load iterations (96 KB SMEM opt-in). Marginal on the prompts we measured. |
| `FA_NT` | `1` | Per-block t_idx count. `2` shares K/V tile across 2 query rows; currently 14% slower at long context (kept as infra). |
| `FA_SK` | `4` | FA split-K factor at sub_seq_total ≥ 4 K. Spreads each (kv_head, t_idx) across N blocks (default 4) merged via log-sum-exp; lifts long-prompt prefill ~1.46× on 9B and ~1.34× on 27B. `0` to opt out. fp32 partials keep argmax bit-stable with the base FA path. |
| `PREFILL_NO_PIPELINE` | unset | Set to disable cross-GPU prefill pipelining (per-GPU compute / D2H / H2D streams + double-buffered hidden + host transfer). Pipelining is auto-enabled with ≥2 GPU segments and gives ~2–3× prefill at 1K-18K. |
| `PREFILL_NO_HOST_FENCE` | unset | Set to drop the `cudaEventSynchronize` between cross-GPU D2H and H2D — racy on CMP 100-210 (sampled tokens diverge from the sequential path). Only set this for benchmarking the upper-bound throughput on hardware where cross-device stream waits properly fence pinned memory. |
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
- **DFlash needs a distribution-matched drafter.** The engine-side DFlash path is functional and lossless, but a stock drafter trained on vanilla Qwen3.5 mismatches a distilled model's distribution, so accept rate is poor out of the box — train/fine-tune a drafter on your model's own outputs to get a usable accept length (~4.5–5).
- **Pipelined-fold is quality-equivalent, not bit-identical.** The default fold path (`DFLASH_FOLD=0` to disable) forwards the committed bonus token through the *batched tree-verify* kernel instead of the *single-token* kernel, so near-tie argmax tokens can occasionally flip — different-but-valid wording on some responses. Verified to **not** change correctness (identical 8/8 compiled-C++ coding pass-rate and reasoning accuracy vs the non-fold path), but if you need byte-exact reproducibility set `DFLASH_FOLD=0` for the slower bit-exact path.
- **No batched MTP / spec.** Speculative paths run only when `slots == 1`. With `slots > 1`, the batched gen loop is plain greedy.
- **GGUF Q8_0 is the supported path.** Q5_K_M / Q6_K load but quality is degraded — use Q8_0.
- **sm_70 specific tuning.** Should run on sm_75; sm_80+ has better engines anyway.
- **Single-host.** No tensor parallelism across machines, no multi-node.
- **Linux only.**
- **Cross-GPU prefill pipelining requires a host-side fence** (`cudaEventSynchronize` between D2H and H2D). On CMP 100-210 (PCIe 1.0 x1, no P2P) the cross-device `cudaStreamWaitEvent` doesn't reliably fence pinned host memory between the source GPU's D2H and the destination GPU's H2D — H2D reads stale bytes and the first sampled token diverges from the sequential path on some prompt lengths. The host fence is default-on (`PREFILL_NO_HOST_FENCE=1` to revert to the racy event-only path) and the perf cost is ≤3% at 18 K because chunks-internal overlap is already serialized by stream FIFOs. May or may not affect newer hardware with proper P2P.
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
