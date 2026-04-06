# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Custom CUDA inference engine for Qwen3.5-27B (GGUF Q5_K_M). Runs on 4x CMP 100-210 (Volta sm70, 16GB HBM2, PCIe 1.0 x1, no P2P).

## Current Status

- `<think>` logit=29.25 (llama.cpp 27.12, similar) — correct output
- Generates meaningful reasoning output
- 7.5 t/s (llama.cpp: 22 t/s with `GGML_CUDA_FORCE_MMQ=1`)
- Coherent for ~80 tokens, then repetition starts (fp16 accumulation error)

## Build and Run

```bash
cd /home/paru/qwen-engine/build && cmake .. && make -j14
./qwen-engine /home/paru/models/gguf/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf 248045 846 198 12675 248046 198 248045 74455 198
```

llama.cpp reference (22 t/s):
```bash
GGML_CUDA_FORCE_MMQ=1 /home/paru/ik_llama.cpp/build/bin/llama-cli -m /home/paru/models/gguf/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q5_K_M.gguf -ngl 999 -fa on --threads 14 --temp 0 -p "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n" -n 50
```

`run.py` wraps the engine with ExLlamaV2 tokenizer for encoding/decoding.

## Model: Qwen3.5-27B (Gated DeltaNet + Attention Hybrid)

- 64 layers in pattern: [GDN, GDN, GDN, Attn] × 16
- hidden=5120, num_q_heads=24, num_kv_heads=4, head_dim=256
- GDN: num_k=16, num_v=48, k_dim=128, v_dim=128, conv_kernel=4
- intermediate_size=17408, vocab=248320
- RoPE: partial rotary dim=64 (25% of head_dim)
- GGUF: Q5_K_M mixed (Q5_K + Q6_K), 19GB

## Architecture

**Entry point:** `src/main.cu` — inference loop. Loads GGUF, distributes weights across GPUs, runs token-by-token generation.

**Layer distribution across GPUs:**
- Layers 0-15 → GPU0, 16-31 → GPU1, 32-47 → GPU2, 48-63 → GPU3
- `token_embd.weight` → GPU0; `output.weight`, `output_norm.weight` → GPU3

**Multi-GPU transfers:** No P2P available, so hidden state (5120 bytes fp16) transfers go through pinned host memory between GPUs.

**Key files:**
- `src/gguf.h` — GGUF v3 parser with mmap
- `src/gpu_loader.h` — Multi-GPU parallel weight loading (thread pool + CUDA streams)
- `src/model.cuh` — QwenModel class: MLP, GDN, Attention forward passes, buffer/state management
- `src/quant_gemv.cuh` — Quantized GEMV kernels (Q5_K, Q6_K with dp4a)
- `src/ops.cuh` — RMSNorm, SiLU, residual add, embedding dequant
- `src/gdn_kernels.cuh` — Conv1d update, GDN recurrent step, output projection
- `src/attention.cuh` — RoPE, head norm, QKV scoring, softmax, value aggregation, sigmoid gating, KV cache
- `src/turboquant.cuh` — KV cache compression via WHT + Lloyd-Max 3-bit (implemented but not active)
- `src/dequant.cuh` — Q5_K/Q6_K → fp16 dequantization

All `.cuh`/`.h` files are header-only (no separate compilation units). `src/main.cpp` is a placeholder, not used.

## P0: GEMV Speed Optimization (Core Bottleneck)

Current bandwidth:
- Q5_K: 283 GB/s (previously achieved 519)
- Q6_K: 100 GB/s (previously achieved 347)
- Hardware memcpy limit: 708 GB/s, llama.cpp GEMV: 450 GB/s

**Q6_K kernel issue:** switch statement + per-iteration elem/sg/pos/quarter computation is slow. Since `sub` is loop-constant, hoist the switch and use arithmetic indexing:
- sg = sub/4, quarter = sub%4
- ql pointer = ql + sg*64 + (quarter&1)*32
- ql_shift = (quarter/2)*4 (0 or 4)
- qh_shift = quarter*2 (0,2,4,6)
- scale: sc[sub*2], sc[sub*2+1]

**Q5_K:** byte access → uint32_t vectorized loads (achieved 519 GB/s before).

**dp4a note:** Always use `__dp4a(int, int, int)` overload — cast to `(int)` explicitly.

## P1: Output Stability (Long Generation)

- ~80 tokens then repetition starts
- Mitigations applied: state clamping ±1e6, softmax power-of-2 threads, gate_buf/attn_scores buffer separation

## GDN Recurrence Formula

```
g = ssm_a * softplus(alpha + dt_bias)   // ssm_a is negative
decay = exp(g)                           // 0 < decay < 1
beta = sigmoid(b_proj)
Q, K = L2_norm(conv_out), scale = 1/sqrt(k_dim)
S_new = decay * S + k * beta * (v - decay * S^T @ k)
o = S_new^T @ q * scale
GQA: k_head = v_head % num_k (modulo mapping)
```

## Resolved Bugs

1. GPU P2P unavailable → host pinned memory bridge
2. RoPE multi-GPU + partial rotary dim=64
3. Conv1d weight layout: weight[d*kw+i]
4. dt_bias must be F32
5. Q6_K dequant scale: is = n/16 + l/16
6. GQA modulo mapping: head % num_k
7. gate_buf separated from attn_scores (buffer reuse conflict)
8. State clamping ±1e6 for numerical stability
