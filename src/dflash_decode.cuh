// DFlash speculative decode orchestrator.
//
// Stage 6 of the DFlash integration: combines
//   - DraftModel (dflash_draft.cuh)        — block-diffusion drafter, q_len=16
//   - DDTree builder (dflash_ddtree.cuh)   — best-first heap, budget=22
//   - Target hidden capture (model.cuh DFlashCapture) — 5 layer hiddens/token
//   - Target tree verify (forward_*_tree)  — ancestor-mask attention
//
// Public API (used by main.cu):
//   - dflash_init(state, draft_path, model, max_ctx) — load draft + init capture
//   - dflash_free(state)                              — release everything
//   - dflash_decode_step(...)                         — one spec iteration
//
// State holds per-step scratch buffers so allocations happen once.

#pragma once

#include "dflash_draft.cuh"
#include "dflash_ddtree.cuh"
#include "model.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash {

struct DecodeState {
    DraftModel draft;
    bool       loaded = false;

    // Per-step scratch (reused across calls):
    int32_t* d_pos_q = nullptr;       // [block_size]
    int32_t* d_pos_k = nullptr;       // [max_ctx + block_size]
    half*    d_noise_embed = nullptr; // [block_size, hidden] on draft.device
    half*    d_logits      = nullptr; // [block_size, vocab]  on last_gpu (caller fills)
    float*   h_logits_f32  = nullptr; // [block_size, vocab]  pinned host
    float*   h_top_logp    = nullptr; // [block_size, K]
    int32_t* h_top_ids     = nullptr; // [block_size, K]

    int max_ctx       = 0;
    int draft_gpu     = 0;
    int top_k         = 10;
    int budget        = 22;
    bool chain_seed   = true;
    int  block_size   = DraftConfig::block_size;   // 16
    int  hidden_size  = DraftConfig::hidden_size;
    int  vocab_size   = 0;                          // filled in init
};

// Allocate scratch + load draft + enable capture in `model`.
// `draft_path` points at the draft safetensors file. Returns false on failure.
inline bool dflash_init(
    DecodeState& s,
    const std::string& draft_path,
    QwenModel& model,
    int max_ctx
) {
    s.max_ctx     = max_ctx;
    s.draft_gpu   = 0;
    s.vocab_size  = model.cfg.vocab_size;
    s.hidden_size = model.cfg.hidden_size;

    // Drafter context window. The block-diffusion drafter was trained on short
    // (~2K) sequences, and at large max_ctx the per-token C buffer (and the draft's
    // own K/V buffers) would be enormous (256K -> ~13 GB fp16 on GPU 0). So cap the
    // drafter's context to a recent-W window: bounds VRAM AND keeps the drafter in
    // its trained range. Default 4096 when max_ctx is large; override with
    // DFLASH_WINDOW; window == max_ctx for small contexts (exact, no ring).
    int W = max_ctx;
    if (const char* e = getenv("DFLASH_WINDOW")) { int w = atoi(e); if (w > 0 && w < max_ctx) W = w; }
    else if (max_ctx > 16384) W = 4096;

    if (!load_draft(s.draft, draft_path, s.draft_gpu, W)) {
        printf("[dflash] load_draft failed: %s\n", draft_path.c_str());
        return false;
    }

    // Capture hook: 5 target layer ids, GPU0 ring buffer sized for the window W.
    int layer_ids[5];
    for (int i = 0; i < 5; i++) layer_ids[i] = kTargetLayerIds[i];
    model.init_dflash_capture(max_ctx, layer_ids, 5, W);

    int B = s.block_size;
    int V = s.vocab_size;
    int H = s.hidden_size;

    cudaSetDevice(s.draft_gpu);
    cudaMalloc(&s.d_pos_q, B * sizeof(int32_t));
    cudaMalloc(&s.d_pos_k, (max_ctx + B) * sizeof(int32_t));
    cudaMalloc(&s.d_noise_embed, (size_t)B * H * sizeof(half));

    cudaMallocHost(&s.h_logits_f32, (size_t)B * V * sizeof(float));
    cudaMallocHost(&s.h_top_logp,   (size_t)B * s.top_k * sizeof(float));
    cudaMallocHost(&s.h_top_ids,    (size_t)B * s.top_k * sizeof(int32_t));

    s.loaded = true;
    printf("[dflash] decode state ready: max_ctx=%d block=%d K=%d budget=%d\n",
           max_ctx, B, s.top_k, s.budget);
    return true;
}

inline void dflash_free(DecodeState& s) {
    if (!s.loaded) return;
    cudaSetDevice(s.draft_gpu);
    if (s.d_pos_q)        cudaFree(s.d_pos_q);
    if (s.d_pos_k)        cudaFree(s.d_pos_k);
    if (s.d_noise_embed)  cudaFree(s.d_noise_embed);
    if (s.h_logits_f32)   cudaFreeHost(s.h_logits_f32);
    if (s.h_top_logp)     cudaFreeHost(s.h_top_logp);
    if (s.h_top_ids)      cudaFreeHost(s.h_top_ids);
    free_draft(s.draft);
    s.loaded = false;
}

// Fills draft positions: pos_q = [ctx_len..ctx_len+B-1], pos_k = [0..ctx_len+B-1].
// CPU then memcpy. Could be a kernel, but called once per spec iter so cheap.
inline void prepare_positions(DecodeState& s, int ctx_len) {
    int B = s.block_size;
    std::vector<int32_t> pq(B), pk(ctx_len + B);
    for (int i = 0; i < B;             i++) pq[i] = ctx_len + i;
    for (int i = 0; i < ctx_len + B;   i++) pk[i] = i;
    cudaSetDevice(s.draft_gpu);
    cudaMemcpy(s.d_pos_q, pq.data(), B * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_pos_k, pk.data(), (ctx_len + B) * sizeof(int32_t), cudaMemcpyHostToDevice);
}

// Build noise embedding [last_tok, MASK_ID*15] on draft.device.
// Caller passes the embedding table source (fp16 dequant of token_embd) plus
// the row dequant kernel — that path lives in main.cu so we expose a helper
// that takes a pre-built [B, hidden] fp16 buffer instead. main.cu does:
//   dequant_embd_q*_row → noise_embed_dst[0]   for last_tok
//   dequant_embd_q*_row → noise_embed_dst[i]   for MASK_ID, i=1..15

}  // namespace dflash
