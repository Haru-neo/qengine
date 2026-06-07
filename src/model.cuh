#pragma once
#include "gpu_loader.h"
#include "quant_gemv.cuh"
#include "ops.cuh"
#include "gdn_kernels.cuh"
#include "attention.cuh"
#include "sparse_attn/sparse_config.h"
#include "sparse_attn/sparse_index.cuh"
#include "sparse_attn/a_shape_index.cuh"
#include "sparse_attn/vertical_slash_index.cuh"
#include "sparse_attn/block_sparse.cuh"
#include <string>
#include <cstdio>
#include <chrono>
#include <mutex>
#include <algorithm>

// PROFILE_ATTN=1 (set before forward_attn_chunk runs) enables per-phase sync
// timers inside forward_attn_chunk. Accumulates score/softmax/value/other
// milliseconds into these globals; main.cu prints them at prefill end.
static double g_attn_score_ms   = 0.0;
static double g_attn_softmax_ms = 0.0;
static double g_attn_value_ms   = 0.0;
static double g_attn_fused_ms   = 0.0;
static double g_attn_other_ms   = 0.0;
// Per-token forward_attn phase breakdown (gen path). Same PROFILE_ATTN env.
static double g_pt_qkvr_ms      = 0.0;  // RMSNorm + Q/K/V proj + head norm + RoPE
static double g_pt_kvwrite_ms   = 0.0;  // KV cache write / TQ quantize
static double g_pt_attn_ms      = 0.0;  // attention compute (fused FA, or score+softmax+value)
static double g_pt_oproj_ms     = 0.0;  // gate sigmoid + O projection + residual
static long   g_pt_calls        = 0;    // # forward_attn calls accumulated
static bool   g_profile_attn    = false;
// FLASH_ATTN=1 routes the sub-chunk score+softmax+value trio through a
// single fused kernel (see flash_attn_chunk_fused). Only valid for the
// Qwen3.5-27B attention shape (HD=256, num_q=24, num_kv=4).
static bool   g_use_flash_attn  = false;

// PROFILE_GDN=1 enables per-phase sync timers inside forward_gdn (single-token
// gen path). Accumulates ms into the globals below; main.cu prints them next
// to PROFILE_GEN summary. Adds sync overhead so keep OFF in production.
static double g_gdn_norm_ms     = 0.0;  // input RMSNorm
static double g_gdn_proj_ms     = 0.0;  // 4 GEMVs: qkv / gate / alpha / beta
static double g_gdn_conv_ms     = 0.0;  // conv1d_update_silu
static double g_gdn_recur_ms    = 0.0;  // launch_gdn_recurrent_step
static double g_gdn_rmsg_ms     = 0.0;  // rms_norm_gated_kernel
static double g_gdn_oproj_ms    = 0.0;  // output projection GEMV
static double g_gdn_resi_ms     = 0.0;  // residual add
static long   g_gdn_calls       = 0;
static bool   g_profile_gdn     = false;

// PROFILE_MLP=1 enables per-phase sync timers inside forward_mlp_chunk.
// Phases: norm / q1 (Q8 quantize of normed hidden) / gate (gate_proj GEMV) /
// up (up_proj GEMV) / silu (silu_mul) / q2 (Q8 quantize of intermediate) /
// down (down_proj GEMV) / resi (residual add). main.cu prints them next to
// PROFILE_PREFILL summary. Adds sync overhead so keep OFF in production.
static double g_mlp_norm_ms     = 0.0;
static double g_mlp_q1_ms       = 0.0;
static double g_mlp_gate_ms     = 0.0;
static double g_mlp_up_ms       = 0.0;
static double g_mlp_silu_ms     = 0.0;
static double g_mlp_q2_ms       = 0.0;
static double g_mlp_down_ms     = 0.0;
static double g_mlp_resi_ms     = 0.0;
static long   g_mlp_calls       = 0;
static bool   g_profile_mlp     = false;

struct QwenConfig {
    int hidden_size;
    int intermediate_size;
    int num_layers;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    int vocab_size;
    int linear_k_heads;
    int linear_v_heads;
    int linear_k_dim;
    int linear_v_dim;
    float rms_norm_eps;
    int rope_dim;
    float rope_freq_base = 10000000.0f;  // theta; 27B hybrid=1e7, Qwen3 dense=1e6
    // MoE (Qwen3.x-A3B). is_moe is a model-level flag; per-layer routing uses
    // layer_is_moe[] (a dense model leaves all of these zero/false).
    bool is_moe = false;
    int  num_experts = 0;          // total experts (router output dim)
    int  num_experts_per_tok = 0;  // top-k active experts
    int  moe_intermediate_size = 0;// per-expert SwiGLU intermediate dim
    int  shared_expert_intermediate_size = 0; // 0 = no shared expert
};

struct LayerBuffers {
    half* norm_out;      // [hidden_size]
    half* attn_out;      // [hidden_size] or larger
    half* mlp_gate;      // [intermediate_size]
    half* mlp_up;        // [intermediate_size]
    half* mlp_down;      // [hidden_size]
    half* residual;      // [hidden_size]

    // Chunked prefill MLP buffers (token-major, [CHUNK_SIZE * dim]).
    // Allocated lazily on first chunked prefill.
    half* mlp_chunk_norm = nullptr;  // [CHUNK_SIZE * H]  RMSNorm output
    half* mlp_chunk_gate = nullptr;  // [CHUNK_SIZE * I]  ffn_gate proj
    half* mlp_chunk_up   = nullptr;  // [CHUNK_SIZE * I]  ffn_up proj
    half* mlp_chunk_down = nullptr;  // [CHUNK_SIZE * H]  ffn_down proj
};

// ============ MoE helper kernels (qmoe_ prefix to avoid clashing with the
// identically-purposed kernels in gemma_model.cuh, which shares this TU) ====

// Router GEMV: F32 weight [N x K] row-major, fp16 input [K], float output [N].
__global__ void qmoe_router_gemv_f32(const float* __restrict__ weight,
                                     const half* __restrict__ input,
                                     float* __restrict__ output, int K, int N) {
    int row = blockIdx.x;
    if (row >= N) return;
    const float* w = weight + (size_t)row * K;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        sum += w[i] * __half2float(input[i]);
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    if (threadIdx.x == 0) output[row] = sum;
}

__global__ void qmoe_zero_f32(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0f;
}

// out_f32[i] += w * (float)in_half[i]
__global__ void qmoe_acc_f32(float* __restrict__ out, const half* __restrict__ in,
                             float w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += w * __half2float(in[i]);
}

__global__ void qmoe_f32_to_f16(const float* __restrict__ in, half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// Grouped weighted accumulate: acc[h] += sum_e w[e] * down[e*H + h], with a
// fp16 clamp on each expert's down output folded in (avoids a separate clamp
// launch per expert). down is [G, H]; weights/acc on device.
__global__ void qmoe_weighted_acc_experts(float* __restrict__ acc,
                                          const half* __restrict__ down,
                                          const float* __restrict__ weights,
                                          int G, int H) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    float s = 0.0f;
    for (int e = 0; e < G; e++) {
        float v = __half2float(down[(size_t)e * H + h]);
        if (v != v || v > 65504.f) v = 65504.f; else if (v < -65504.f) v = -65504.f;
        s += weights[e] * v;
    }
    acc[h] += s;
}

__global__ void qmoe_f16_to_f32(const half* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

// GPU-side top-k + normalized softmax over E router logits, writing the selected
// expert ids and softmax weights straight to device buffers — eliminates the
// per-layer host round-trip (D2H copy + CPU partial_sort + sync) that otherwise
// runs once per MoE layer per token (~40 syncs/token, a major decode cost on
// PCIe Gen1 CMP cards). Single block: `topk` iterative block-argmax passes over
// E logits (E≤1024), masking each winner, then softmax over the topk values.
// Launch <<<1, 256, E*sizeof(float)>>>. blockDim must be a power of two.
__global__ void qmoe_topk_softmax(const float* __restrict__ logits, int E, int topk,
                                  int* __restrict__ out_ids, float* __restrict__ out_w) {
    extern __shared__ float s_logits[];     // [E] working copy
    __shared__ float rv[256];
    __shared__ int   ri[256];
    __shared__ int   sel_id[16];
    __shared__ float sel_val[16];
    int tid = threadIdx.x;
    for (int i = tid; i < E; i += blockDim.x) s_logits[i] = logits[i];
    __syncthreads();

    for (int k = 0; k < topk; k++) {
        float bv = -1e30f; int bi = -1;
        for (int i = tid; i < E; i += blockDim.x) {
            float v = s_logits[i];
            if (v > bv) { bv = v; bi = i; }
        }
        rv[tid] = bv; ri[tid] = bi;
        __syncthreads();
        for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
            if (tid < off && rv[tid + off] > rv[tid]) { rv[tid] = rv[tid + off]; ri[tid] = ri[tid + off]; }
            __syncthreads();
        }
        if (tid == 0) { sel_id[k] = ri[0]; sel_val[k] = rv[0]; s_logits[ri[0]] = -1e30f; }
        __syncthreads();
    }
    if (tid == 0) {
        float maxl = sel_val[0], sum = 0.0f;   // sel_val[0] is the global max
        for (int k = 0; k < topk; k++) { sel_val[k] = expf(sel_val[k] - maxl); sum += sel_val[k]; }
        for (int k = 0; k < topk; k++) { out_w[k] = sel_val[k] / sum; out_ids[k] = sel_id[k]; }
    }
}

// In-place sigmoid of a single device scalar (shared-expert gate).
__global__ void qmoe_sigmoid_scalar(float* __restrict__ x) {
    if (threadIdx.x == 0 && blockIdx.x == 0) x[0] = 1.0f / (1.0f + expf(-x[0]));
}

// In-place sigmoid over a vector (per-token shared-expert gates).
__global__ void qmoe_sigmoid_vec(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0f / (1.0f + expf(-x[i]));
}

__global__ void qmoe_fill_f32(float* __restrict__ x, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

// ── Batched (chunk-prefill) MoE helpers ───────────────────────────────────
// Batched F32 router: logits[token, expert] for all N tokens in one launch.
// Grid (E, N), 32 threads/block reducing over K. Input is per-token half norm.
__global__ void qmoe_router_gemv_chunk_f32(const float* __restrict__ weight,
                                           const half* __restrict__ input,
                                           float* __restrict__ output,
                                           int K, int E, int N) {
    int e = blockIdx.x, token = blockIdx.y;
    if (e >= E || token >= N) return;
    const float* w = weight + (size_t)e * K;
    const half*  x = input  + (size_t)token * K;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) sum += w[i] * __half2float(x[i]);
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if (threadIdx.x == 0) output[(size_t)token * E + e] = sum;
}

// Per-token top-k + softmax for a chunk: grid.x = token. Same algorithm as
// qmoe_topk_softmax, one block per token.
__global__ void qmoe_topk_softmax_chunk(const float* __restrict__ logits, int E, int topk,
                                        int* __restrict__ out_ids, float* __restrict__ out_w,
                                        int N) {
    int token = blockIdx.x;
    if (token >= N) return;
    const float* lg = logits + (size_t)token * E;
    extern __shared__ float s_logits[];
    __shared__ float rv[256];
    __shared__ int   ri[256];
    __shared__ int   sel_id[16];
    __shared__ float sel_val[16];
    int tid = threadIdx.x;
    for (int i = tid; i < E; i += blockDim.x) s_logits[i] = lg[i];
    __syncthreads();
    for (int k = 0; k < topk; k++) {
        float bv = -1e30f; int bi = -1;
        for (int i = tid; i < E; i += blockDim.x) { float v = s_logits[i]; if (v > bv) { bv = v; bi = i; } }
        rv[tid] = bv; ri[tid] = bi; __syncthreads();
        for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
            if (tid < off && rv[tid + off] > rv[tid]) { rv[tid] = rv[tid + off]; ri[tid] = ri[tid + off]; }
            __syncthreads();
        }
        if (tid == 0) { sel_id[k] = ri[0]; sel_val[k] = rv[0]; s_logits[ri[0]] = -1e30f; }
        __syncthreads();
    }
    if (tid == 0) {
        float maxl = sel_val[0], sum = 0.0f;
        for (int k = 0; k < topk; k++) { sel_val[k] = expf(sel_val[k] - maxl); sum += sel_val[k]; }
        int* oid = out_ids + (size_t)token * topk;
        float* ow = out_w + (size_t)token * topk;
        for (int k = 0; k < topk; k++) { ow[k] = sel_val[k] / sum; oid[k] = sel_id[k]; }
    }
}

// Scatter-add the flat per-assignment expert outputs back into the chunk hidden:
// hidden[token, h] += sum_k weight[token*topk+k] * clamp(down[(token*topk+k), h]).
// Grid ((H+255)/256, N); each (token,h) sums its own topk assignments (no atomics).
__global__ void qmoe_scatter_acc_chunk(float* __restrict__ hidden, const half* __restrict__ down,
                                       const float* __restrict__ weights,
                                       int topk, int H, int N) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int token = blockIdx.y;
    if (h >= H || token >= N) return;
    float s = 0.0f;
    for (int k = 0; k < topk; k++) {
        int a = token * topk + k;
        float v = __half2float(down[(size_t)a * H + h]);
        if (v != v || v > 65504.f) v = 65504.f; else if (v < -65504.f) v = -65504.f;
        s += weights[a] * v;
    }
    hidden[(size_t)token * H + h] += s;
}

// acc[i] += (*w) * clamp(in[i]); weight read from device (no host round-trip).
__global__ void qmoe_acc_f32_dev(float* __restrict__ acc, const half* __restrict__ in,
                                 const float* __restrict__ w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(in[i]);
    if (v != v || v > 65504.f) v = 65504.f; else if (v < -65504.f) v = -65504.f;
    acc[i] += w[0] * v;
}

// Clamp fp16 to finite range (fp16 max 65504); maps nan→max. Expert GEMV
// outputs can overflow fp16 (256 experts, large distilled weights) → inf →
// silu/accumulate → nan. Mirrors gemma_model.cuh's clamp_fp16_kernel.
__global__ void qmoe_clamp_fp16(half* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(x[i]);
        if (v != v || v > 65504.f) x[i] = __float2half(65504.f);
        else if (v < -65504.f)     x[i] = __float2half(-65504.f);
    }
}

struct QwenModel {
    QwenConfig cfg;
    GPUModel* gpu;
    GGUFFile* gguf_p = nullptr;   // set in init_config, for late metadata reads
    std::string gguf_arch;        // general.architecture
    LayerBuffers bufs[4];
    bool layer_is_attn[256] = {};
    bool layer_is_moe[256]  = {};
    // Per-GPU MoE scratch (single-token; chunk path loops tokens through it).
    // Allocated in alloc_buffers only when cfg.is_moe.
    half*  moe_gate[4]   = {};   // [moe_intermediate_size]
    half*  moe_up[4]     = {};   // [moe_intermediate_size]
    half*  moe_down[4]   = {};   // [hidden_size]
    float* moe_acc[4]    = {};   // [hidden_size] fp32 accumulator
    float* moe_logits[4] = {};   // [num_experts] router logits (device)
    QuantInput gpu_qi_moe[4];    // input quantizer for the moe_intermediate dim
    // Grouped-expert scratch (all top-k experts in one launch — see
    // gemv_q8_0_q8_experts). Sized topk × per-expert dims.
    half*  moe_gate_g[4] = {};   // [topk * moe_intermediate_size]
    half*  moe_up_g[4]   = {};   // [topk * moe_intermediate_size]
    half*  moe_down_g[4] = {};   // [topk * hidden_size]
    int*   moe_ids_dev[4]= {};   // [topk] selected expert ids (device)
    float* moe_w_dev[4]  = {};   // [topk] softmax weights (device)
    int*   moe_ids_host[4] = {}; // pinned [topk]
    float* moe_w_host[4] = {};   // pinned [topk]
    QuantInput gpu_qi_moe_g[4];  // quantizer for [topk * moe_intermediate]
    // Batched chunk-prefill MoE scratch (sized CHUNK_SIZE × topk × dims).
    half*  moe_norm_c[4]  = {};  // [CHUNK_SIZE * H] batched norm output
    float* moe_logits_c[4]= {};  // [CHUNK_SIZE * num_experts]
    int*   moe_ids_c[4]   = {};  // [CHUNK_SIZE * topk]
    float* moe_w_c[4]     = {};  // [CHUNK_SIZE * topk]
    half*  moe_gate_cg[4] = {};  // [CHUNK_SIZE * topk * moe_intermediate]
    half*  moe_up_cg[4]   = {};  // [CHUNK_SIZE * topk * moe_intermediate]
    half*  moe_down_cg[4] = {};  // [CHUNK_SIZE * topk * hidden]
    QuantInput gpu_qi_c[4];      // chunk norm quantizer
    QuantInput gpu_qi_cg[4];     // chunk intermediate quantizer
    bool v2_separate_qkv = false;
    int  attn_num_q_heads = 0;
    int  attn_num_kv_heads = 0;
    int  attn_head_dim = 0;
    int  mtp_layer_idx = -1;
    QuantInput gpu_qi[4];  // per-GPU reusable Q8 buffer (hidden_size)
    QuantInput gpu_qi_inter[4];  // per-GPU for intermediate_size  // one per GPU
    // Second-token buffer set used by the speculative-decoding path
    // (forward_*_n2). Allocated lazily in alloc_buffers_n2() — only the
    // MLP/attn fields are duplicated here because GDN's per-call temp
    // buffers (gdn_bufs[g].conv_out etc.) are reused across the two
    // sequential GDN forwards within a spec iter.
    LayerBuffers bufs2[4];
    QuantInput gpu_qi2[4];
    QuantInput gpu_qi_inter2[4];
    bool n2_buffers_ready = false;
    // Third-token buffer set for MTP K=2 (three-stream batched forward):
    // main_token, draft1, draft2. Allocated lazily in alloc_buffers_n3().
    LayerBuffers bufs3[4];
    QuantInput gpu_qi3[4];
    QuantInput gpu_qi_inter3[4];
    bool n3_buffers_ready = false;

    // Second-token GDN intermediate buffers (conv_out / core_out / normed_out
    // / proj_out) used by the batched forward_gdn_n2.
    struct GDNBuffers2 {
        float* conv_out;
        half*  core_out;
        half*  normed_out;
        half*  proj_out;
    };
    GDNBuffers2 gdn_bufs2[4];
    GDNBuffers2 gdn_bufs3[4];

    // ── DFlash target hidden capture ────────────────────────────────────────
    // Used by speculative decoding via the DFlash draft. The capture buffer
    // accumulates 5 selected layer outputs per committed token in the layout
    // expected by the draft's `target_hidden_cat[ctx_len, 5*hidden]`:
    //   gpu0_buf[(token_pos * n_slots + slot) * hidden + h]
    // Cross-GPU transfer goes through host pinned memory (no P2P on CMP).
    // Disabled by default; init_dflash_capture() turns it on.
    struct DFlashCapture {
        half* gpu0_buf       = nullptr;   // [window * n_slots * hidden] ring on GPU 0
        half* unwrap         = nullptr;   // [window * n_slots * hidden] contiguous scratch (only used when the window wraps)
        half* staging[4]     = {};        // per-GPU [hidden] fp16 staging
        half* host_pinned    = nullptr;   // [hidden] pinned host bridge
        int   max_ctx        = 0;
        int   window         = 0;         // C ring size = drafter context window (== max_ctx when not windowed, e.g. extract)
        int   n_slots        = 0;
        int   layer_to_slot[256] = {};    // -1 if layer is not captured
        bool  enabled        = false;

        // Tree-mode scratch: holds per-slot fp16 hidden for each capture-layer
        // during batched tree verify. Sized [n_slots × tree_budget × hidden].
        // Layout: scratch[slot * tree_budget * H + tok * H + h].
        // Filled by capture_tree_layer() during the tree forward; committed
        // into gpu0_buf via commit_tree_capture() once accept path is known.
        half* tree_scratch   = nullptr;
        half* tree_host_pin  = nullptr;   // [tree_budget * hidden] pinned for cross-GPU
        int   tree_capacity  = 0;         // budget the scratch was sized for
    };
    DFlashCapture dflash_cap;

    // window<=0 -> no windowing (ring size == max_ctx; used by the offline
    // extract path). window>0 and < max_ctx -> the C buffer is a ring of `window`
    // tokens: the drafter only ever conditions on the most-recent `window` context
    // features. This bounds the GPU-0 buffer (max_ctx=256K would be 13 GB fp16) AND
    // keeps the drafter's RoPE positions / context length inside its trained range
    // (the drafter was trained on ~2 K sequences). Lossless vs the 27B greedy — the
    // verify is unchanged; only how much *draft context* the small model sees changes.
    void init_dflash_capture(int max_ctx_tokens, const int* layer_ids, int n_layers,
                             int window = -1) {
        int H = cfg.hidden_size;
        int W = (window > 0 && window < max_ctx_tokens) ? window : max_ctx_tokens;
        dflash_cap.max_ctx = max_ctx_tokens;
        dflash_cap.window  = W;
        dflash_cap.n_slots = n_layers;
        for (int i = 0; i < 256; i++) dflash_cap.layer_to_slot[i] = -1;
        for (int slot = 0; slot < n_layers; slot++) {
            int L = layer_ids[slot];
            if (L >= 0 && L < cfg.num_layers && L < 256)
                dflash_cap.layer_to_slot[L] = slot;
        }
        cudaSetDevice(0);
        size_t buf_bytes = (size_t)W * n_layers * H * sizeof(half);
        cudaMalloc(&dflash_cap.gpu0_buf, buf_bytes);
        cudaMemset(dflash_cap.gpu0_buf, 0, buf_bytes);
        if (W < max_ctx_tokens) {   // windowed: need an unwrap scratch for wrapped reads
            cudaMalloc(&dflash_cap.unwrap, buf_bytes);
        }
        cudaMallocHost(&dflash_cap.host_pinned, H * sizeof(half));
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&dflash_cap.staging[g], H * sizeof(half));
        }
        cudaSetDevice(0);
        dflash_cap.enabled = true;
        printf("[dflash-capture] enabled, max_ctx=%d window=%d slots=%d buf=%.1f MB%s\n",
               max_ctx_tokens, W, n_layers, buf_bytes / 1024.0 / 1024.0,
               W < max_ctx_tokens ? " (windowed ring)" : "");
    }

    // Return a contiguous [ctx_used * n_slots * H] fp16 view of the C ring covering
    // chronological positions [last_pos-ctx_used+1 .. last_pos]. No copy when the
    // window is contiguous in the ring (the common case); 2 D2D copies on wrap.
    const half* dflash_window_view(int last_pos, int ctx_used, cudaStream_t stream = 0) {
        int W = dflash_cap.window, n = dflash_cap.n_slots, H = cfg.hidden_size;
        if (W >= dflash_cap.max_ctx || last_pos < W) {
            // not windowed, or the whole history still fits linearly before the first
            // wrap (positions [0..last_pos] are at gpu0_buf[0..last_pos], chronological).
            return dflash_cap.gpu0_buf;
        }
        int oldest = last_pos - ctx_used + 1;
        int ring_start = ((oldest % W) + W) % W;
        if (ring_start + ctx_used <= W) {
            return dflash_cap.gpu0_buf + (size_t)ring_start * n * H;   // contiguous slice — no copy
        }
        cudaSetDevice(0);
        int first = W - ring_start;
        cudaMemcpyAsync(dflash_cap.unwrap, dflash_cap.gpu0_buf + (size_t)ring_start * n * H,
                        (size_t)first * n * H * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(dflash_cap.unwrap + (size_t)first * n * H, dflash_cap.gpu0_buf,
                        (size_t)(ctx_used - first) * n * H * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        return dflash_cap.unwrap;
    }

    // Allocate tree-mode scratch for batched verify capture. Call once after
    // alloc_tree_decode (so tree_budget is set). Sized to capture every slot
    // for every captured layer in fp16.
    void init_dflash_tree_scratch(int budget) {
        if (!dflash_cap.enabled) return;
        int H = cfg.hidden_size;
        size_t bytes = (size_t)dflash_cap.n_slots * budget * H * sizeof(half);
        cudaSetDevice(0);
        cudaMalloc(&dflash_cap.tree_scratch, bytes);
        cudaMallocHost(&dflash_cap.tree_host_pin, (size_t)budget * H * sizeof(half));
        dflash_cap.tree_capacity = budget;
        printf("[dflash-capture] tree scratch allocated: budget=%d, %.1f MB\n",
               budget, bytes / 1024.0 / 1024.0);
    }

    // Capture every slot's fp16 hidden for `layer` (no-op unless `layer` is in
    // the capture set). h_tree is fp32 [n_tokens × hidden] on `src_gpu`. Slot
    // values land in dflash_cap.tree_scratch[slot_idx][0..n_tokens×H].
    void dflash_capture_tree_layer(int layer, const float* h_tree, int n_tokens,
                                   int src_gpu, cudaStream_t stream = 0) {
        if (!dflash_cap.enabled) return;
        if (layer < 0 || layer >= 256) return;
        int slot = dflash_cap.layer_to_slot[layer];
        if (slot < 0) return;
        if (n_tokens <= 0 || n_tokens > dflash_cap.tree_capacity) return;
        int H = cfg.hidden_size;

        // Convert fp32 → fp16 on src_gpu, then route to GPU 0 scratch.
        cudaSetDevice(src_gpu);
        size_t bytes = (size_t)n_tokens * H * sizeof(half);
        half* src_half = nullptr;
        cudaMalloc(&src_half, bytes);
        int total = n_tokens * H;
        float_to_half_kernel<<<(total+255)/256, 256, 0, stream>>>(
            h_tree, src_half, total);

        half* scratch_dst = dflash_cap.tree_scratch
                          + (size_t)slot * dflash_cap.tree_capacity * H;
        if (src_gpu == 0) {
            cudaMemcpyAsync(scratch_dst, src_half, bytes,
                            cudaMemcpyDeviceToDevice, stream);
        } else {
            cudaMemcpyAsync(dflash_cap.tree_host_pin, src_half, bytes,
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            cudaSetDevice(0);
            cudaMemcpy(scratch_dst, dflash_cap.tree_host_pin, bytes,
                       cudaMemcpyHostToDevice);
        }
        cudaSetDevice(src_gpu);
        cudaFree(src_half);
    }

    // Commit the accepted tree path's hidden states to the main capture buffer.
    // host_slots[i] is the tree-slot index of the i-th accepted token. After
    // this call, gpu0_buf[(ctx_pos+i) * n_slots + slot] = scratch[slot][host_slots[i]]
    // for every captured slot, and i = 0..accept_len-1.
    void dflash_commit_tree_capture(const int* host_slots, int accept_len,
                                    int ctx_pos) {
        if (!dflash_cap.enabled) return;
        if (accept_len <= 0) return;
        int H = cfg.hidden_size;
        cudaSetDevice(0);
        for (int slot_idx = 0; slot_idx < dflash_cap.n_slots; slot_idx++) {
            half* scratch_layer = dflash_cap.tree_scratch
                                + (size_t)slot_idx * dflash_cap.tree_capacity * H;
            for (int i = 0; i < accept_len; i++) {
                int tree_slot = host_slots[i];
                if (tree_slot < 0 || tree_slot >= dflash_cap.tree_capacity) continue;
                half* src = scratch_layer + (size_t)tree_slot * H;
                half* dst = dflash_cap.gpu0_buf
                          + ((size_t)((ctx_pos + i) % dflash_cap.window) * dflash_cap.n_slots + slot_idx) * H;
                cudaMemcpyAsync(dst, src, H * sizeof(half),
                                cudaMemcpyDeviceToDevice, 0);
            }
        }
    }

    // Single-token capture: fp32 hidden on `src_gpu` → fp16 → GPU0 capture slot.
    // No-op unless enabled and `layer` is one of the captured layers.
    void dflash_capture_layer(int layer, const float* h_fp32, int token_pos,
                              int src_gpu, cudaStream_t stream = 0) {
        if (!dflash_cap.enabled) return;
        if (layer < 0 || layer >= 256) return;
        int slot = dflash_cap.layer_to_slot[layer];
        if (slot < 0) return;
        if (token_pos < 0 || token_pos >= dflash_cap.max_ctx) return;
        int H = cfg.hidden_size;

        // DFLASH_DUMP_CAPTURE=1: print first 8 floats of fp32 hidden for the
        // first 3 token positions per slot. Pair with DUMP_LAYERS=1 on a
        // per-token forward to verify capture matches the layer's actual output.
        static const bool dump_cap = getenv("DFLASH_DUMP_CAPTURE") != nullptr;
        if (dump_cap && token_pos < 3) {
            cudaSetDevice(src_gpu); cudaDeviceSynchronize();
            float sample[8];
            cudaMemcpy(sample, h_fp32, 8 * sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[CAP-FP32 L%02d slot%d t%03d] %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
                    layer, slot, token_pos,
                    sample[0], sample[1], sample[2], sample[3],
                    sample[4], sample[5], sample[6], sample[7]);
            fflush(stderr);
        }

        cudaSetDevice(src_gpu);
        half* stg = dflash_cap.staging[src_gpu];
        float_to_half_kernel<<<(H+255)/256, 256, 0, stream>>>(h_fp32, stg, H);

        half* dst = dflash_cap.gpu0_buf
                  + ((size_t)(token_pos % dflash_cap.window) * dflash_cap.n_slots + slot) * H;
        if (src_gpu == 0) {
            cudaMemcpyAsync(dst, stg, H * sizeof(half),
                            cudaMemcpyDeviceToDevice, stream);
        } else {
            cudaMemcpyAsync(dflash_cap.host_pinned, stg, H * sizeof(half),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            cudaSetDevice(0);
            cudaMemcpy(dst, dflash_cap.host_pinned, H * sizeof(half),
                       cudaMemcpyHostToDevice);
        }
    }

    // Chunk capture: fp32 [n_tokens, hidden] on `src_gpu` → fp16 → GPU0 with
    // strided per-token write. Used by the prefill chunked path.
    // ctx_base: offset (in tokens) added to the capture-buffer token index. The
    // serve path uses 0 (one sequence → one capture region). The offline
    // dflash-extract pipeline binds each in-flight pipeline buffer to its own
    // capture region (ctx_base = buf * max_L) so multiple sequences can prefill
    // concurrently across the GPU stages without colliding in gpu0_buf.
    void dflash_capture_chunk(int layer, const float* h_chunk_fp32, int start_pos,
                              int n_tokens, int src_gpu, cudaStream_t stream = 0,
                              int ctx_base = 0) {
        if (!dflash_cap.enabled) return;
        if (layer < 0 || layer >= 256) return;
        int slot = dflash_cap.layer_to_slot[layer];
        if (slot < 0) return;
        if (n_tokens <= 0) return;
        if (start_pos < 0 || ctx_base < 0
            || ctx_base + start_pos + n_tokens > dflash_cap.max_ctx) return;
        int H = cfg.hidden_size;
        int cb = ctx_base + start_pos;  // absolute first token index in gpu0_buf

        static const bool dump_cap = getenv("DFLASH_DUMP_CAPTURE") != nullptr;
        if (dump_cap && start_pos == 0) {
            cudaSetDevice(src_gpu); cudaDeviceSynchronize();
            for (int t = 0; t < std::min(3, n_tokens); t++) {
                float sample[8];
                cudaMemcpy(sample, h_chunk_fp32 + (size_t)t * H,
                           8 * sizeof(float), cudaMemcpyDeviceToHost);
                fprintf(stderr, "[CAP-FP32 L%02d slot%d t%03d] %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
                        layer, slot, t,
                        sample[0], sample[1], sample[2], sample[3],
                        sample[4], sample[5], sample[6], sample[7]);
            }
            fflush(stderr);
        }

        cudaSetDevice(src_gpu);
        size_t bytes = (size_t)n_tokens * H * sizeof(half);
        half* chunk_half = nullptr;
        cudaMalloc(&chunk_half, bytes);
        int total = n_tokens * H;
        float_to_half_kernel<<<(total+255)/256, 256, 0, stream>>>(
            h_chunk_fp32, chunk_half, total);

        if (src_gpu == 0) {
            for (int i = 0; i < n_tokens; i++) {
                half* dst = dflash_cap.gpu0_buf
                          + ((size_t)((cb+i) % dflash_cap.window) * dflash_cap.n_slots + slot) * H;
                cudaMemcpyAsync(dst, chunk_half + (size_t)i * H, H * sizeof(half),
                                cudaMemcpyDeviceToDevice, stream);
            }
        } else {
            half* host_buf = nullptr;
            cudaMallocHost(&host_buf, bytes);
            cudaMemcpyAsync(host_buf, chunk_half, bytes,
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            cudaSetDevice(0);
            for (int i = 0; i < n_tokens; i++) {
                half* dst = dflash_cap.gpu0_buf
                          + ((size_t)((cb+i) % dflash_cap.window) * dflash_cap.n_slots + slot) * H;
                cudaMemcpy(dst, host_buf + (size_t)i * H, H * sizeof(half),
                           cudaMemcpyHostToDevice);
            }
            cudaFreeHost(host_buf);
        }
        cudaSetDevice(src_gpu);
        cudaFree(chunk_half);
    }

    void init_config(GGUFFile& gguf) {
        auto arch = gguf.get_str("general.architecture");
        gguf_p = &gguf;
        gguf_arch = arch;
        cfg.hidden_size = gguf.get_u32(arch + ".embedding_length");
        cfg.num_layers = gguf.get_u32(arch + ".block_count");
        cfg.num_q_heads = gguf.get_u32(arch + ".attention.head_count");
        cfg.num_kv_heads = gguf.get_u32(arch + ".attention.head_count_kv");
        cfg.head_dim = gguf.get_u32(arch + ".attention.key_length", 256);
        cfg.rms_norm_eps = gguf.get_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        
        // Get intermediate size from first MLP weight
        auto* gate = gpu->get("blk.0.ffn_gate.weight");
        cfg.intermediate_size = gate ? gate->dims[1] : 0;
        
        // Get vocab size from output weight; fall back to token_embd for
        // tied-embedding models (Qwen3 dense ships without output.weight).
        auto* out = gpu->get("output.weight");
        cfg.vocab_size = out ? out->dims[1] : 0;
        if (cfg.vocab_size == 0) {
            auto* embd = gpu->get("token_embd.weight");
            if (embd) cfg.vocab_size = embd->dims[1];
        }
        
        // RoPE rotary dim. Qwopus3.6 hybrid (qwen35) uses partial rotary
        // (head_dim/2 = 64). Qwen3 dense (embed/reranker, arch="qwen3") uses
        // FULL rotary = head_dim, and ships no rope.dimension_count key, so
        // the head_dim/2 default would silently halve the rotary span.
        int rope_default = (arch == "qwen3") ? cfg.head_dim : cfg.head_dim / 2;
        cfg.rope_dim = gguf.get_u32(arch + ".rope.dimension_count", rope_default);

        // RoPE theta. 27B hybrid bakes 1e7; Qwen3 dense uses freq_base from
        // the GGUF (1e6 for the 4B embed/reranker). Reading the wrong theta
        // detunes every RoPE frequency and quietly corrupts attention.
        cfg.rope_freq_base = gguf.get_f32(arch + ".rope.freq_base",
                                          (arch == "qwen3") ? 1000000.0f : 10000000.0f);

        // Detect v2 inline MTP: if the last block has nextn tensors, it's a
        // dedicated MTP layer that should not participate in the main forward.
        mtp_layer_idx = -1;
        {
            int last = cfg.num_layers - 1;
            std::string probe = "blk." + std::to_string(last) + ".nextn.eh_proj.weight";
            if (gpu->get(probe.c_str())) {
                mtp_layer_idx = last;
                cfg.num_layers--;
                printf("Detected inline MTP at blk.%d — main forward uses %d layers\n",
                       mtp_layer_idx, cfg.num_layers);
            }
        }

        printf("Config: hidden=%d, inter=%d, layers=%d, heads=%d/%d, vocab=%d, rope_dim=%d\n",
            cfg.hidden_size, cfg.intermediate_size, cfg.num_layers,
            cfg.num_q_heads, cfg.num_kv_heads, cfg.vocab_size, cfg.rope_dim);
    }

    // Chunk size for parallel scan during prompt processing.
    // 128 doubles the per-chunk batched-GEMM utilisation versus the
    // original 64 (each NB=16 GEMM tile gets 8 column blocks instead of
    // 4) and amortises chunk-loop launch overhead. The chunked attention
    // compute path internally splits CHUNK_SIZE into ATTN_NB-token
    // sub-chunks (see ATTN_NB) so the value kernel's register-resident
    // accumulator stays bounded.
    static constexpr int CHUNK_SIZE = 256;
    // Sub-chunk size used by attn_score / softmax / attn_value chunked
    // kernels. Picked so attn_value's per-thread accumulator
    // (ATTN_NB × gqa_max=8 floats) fits in registers without spill on Volta.
    static constexpr int ATTN_NB = 16;

    // GDN temp buffers per GPU
    struct GDNBuffers {
        float* conv_out;    // [qkv_dim] FP32
        half* core_out;     // [num_v * v_dim]
        half* normed_out;   // [num_v * v_dim]
        half* proj_out;     // [hidden_size]

        // Fused projection output (GDN_PROJ_FUSE): contiguous
        // [qkv_n + gate_n + alpha_n + beta_n] half. Sub-pointers alias the
        // 4 traditional buffers (qkv_out, z_out, a_out, b_out).
        half*  fused_proj_out = nullptr;

        // Chunk buffers (per-token data accumulated for chunked GDN)
        float* chunk_qkv;     // [CHUNK_SIZE * qkv_dim] FP32 conv1d outputs
        half*  chunk_qkv_half = nullptr;  // [CHUNK_SIZE * qkv_dim] fp16 staging for batched QKV proj
        half*  chunk_a_proj;  // [CHUNK_SIZE * num_v]
        half*  chunk_b_proj;  // [CHUNK_SIZE * num_v]
        half*  chunk_z_out;   // [CHUNK_SIZE * num_v * v_dim]
        half*  chunk_core_out;// [CHUNK_SIZE * num_v * v_dim]
        half*  chunk_normed;  // [CHUNK_SIZE * num_v * v_dim]
        half*  chunk_proj_out;// [CHUNK_SIZE * hidden_size]
        half*  chunk_norm_out;// [CHUNK_SIZE * hidden_size]
    };
    GDNBuffers gdn_bufs[4];

    // GDN_PROJ_FUSE: fused [qkv|gate|alpha|beta] weight buffer per GDN layer.
    // Built lazily on first forward_gdn call when GDN_PROJ_FUSE=1. Contiguous
    // block_q8_0_aligned layout, [total_n rows × K/32 blocks per row]. Skip if
    // any of the 4 source weights is missing or non-Q8_0.
    struct GdnProjFused {
        void* weight = nullptr;
        int qkv_n = 0, gate_n = 0, alpha_n = 0, beta_n = 0, total_n = 0;
    };
    std::vector<GdnProjFused> gdn_proj_fused;  // sized to num_layers, indexed by layer

    // ATTN_QKV_FUSE: fused [q|k|v] weight per attention layer. attn_q already
    // bakes the gate (q_out_dim = num_q*hd*2). Lazy-built on first forward_attn.
    struct AttnQkvFused {
        void* weight = nullptr;
        int q_n = 0, k_n = 0, v_n = 0, total_n = 0;
    };
    std::vector<AttnQkvFused> attn_qkv_fused;  // sized to num_layers

    void alloc_buffers() {
        int H = cfg.hidden_size;
        int I = cfg.intermediate_size;
        // MoE models have no dense FFN → intermediate_size == 0. But the
        // per-token forward_gdn REUSES bufs.mlp_gate/up as GDN intermediates
        // (z_out = num_v*v_dim, a_out/b_out = num_v). Allocating them at I=0
        // gives zero-size buffers → GDN writes corrupt adjacent memory (the
        // hidden state), zeroing it. Floor the alloc to cover GDN reuse.
        int I_alloc = std::max(I, 8192);
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&bufs[g].norm_out, H * sizeof(half));
            cudaMalloc(&bufs[g].attn_out, std::max(H * 4, cfg.num_q_heads * cfg.head_dim * 2) * sizeof(half));
            cudaMalloc(&bufs[g].mlp_gate, I_alloc * sizeof(half));
            cudaMalloc(&bufs[g].mlp_up, I_alloc * sizeof(half));
            cudaMalloc(&bufs[g].mlp_down, std::max(H, I_alloc) * sizeof(half));
            cudaMalloc(&bufs[g].residual, H * sizeof(half));

            // Chunked prefill MLP buffers (token-major, [CHUNK_SIZE * dim]).
            cudaMalloc(&bufs[g].mlp_chunk_norm, (size_t)CHUNK_SIZE * H * sizeof(half));
            cudaMalloc(&bufs[g].mlp_chunk_gate, (size_t)CHUNK_SIZE * I_alloc * sizeof(half));
            cudaMalloc(&bufs[g].mlp_chunk_up,   (size_t)CHUNK_SIZE * I_alloc * sizeof(half));
            cudaMalloc(&bufs[g].mlp_chunk_down, (size_t)CHUNK_SIZE * H * sizeof(half));

            // GDN buffers (over-allocate for 27B max)
            int qkv_dim = 2 * 16 * 128 + 48 * 128;  // 10240
            int v_total = 48 * 128;  // 6144
            int num_v_max = 48;
            cudaMalloc(&gdn_bufs[g].conv_out, qkv_dim * sizeof(float));
            cudaMalloc(&gdn_bufs[g].core_out, v_total * sizeof(half));
            cudaMalloc(&gdn_bufs[g].normed_out, v_total * sizeof(half));
            cudaMalloc(&gdn_bufs[g].proj_out, H * sizeof(half));
            // Fused proj out: 27B max = qkv 10240 + gate 6144 + alpha 48 + beta 48 = 16480
            cudaMalloc(&gdn_bufs[g].fused_proj_out, 16480 * sizeof(half));

            // Chunk buffers
            cudaMalloc(&gdn_bufs[g].chunk_qkv,      CHUNK_SIZE * qkv_dim * sizeof(float));
            cudaMalloc(&gdn_bufs[g].chunk_a_proj,   CHUNK_SIZE * num_v_max * sizeof(half));
            cudaMalloc(&gdn_bufs[g].chunk_b_proj,   CHUNK_SIZE * num_v_max * sizeof(half));
            cudaMalloc(&gdn_bufs[g].chunk_z_out,    CHUNK_SIZE * v_total * sizeof(half));
            cudaMalloc(&gdn_bufs[g].chunk_core_out, CHUNK_SIZE * v_total * sizeof(half));
            cudaMalloc(&gdn_bufs[g].chunk_normed,   CHUNK_SIZE * v_total * sizeof(half));
            cudaMalloc(&gdn_bufs[g].chunk_proj_out, CHUNK_SIZE * H * sizeof(half));
            cudaMalloc(&gdn_bufs[g].chunk_norm_out, CHUNK_SIZE * H * sizeof(half));
        }
    }

    void alloc_buffers_n2(int max_seq = 4096) {
        if (n2_buffers_ready) return;
        int H = cfg.hidden_size;
        int I = cfg.intermediate_size;
        int qkv_dim = 2 * 16 * 128 + 48 * 128;  // GDN qkv_dim
        int v_total = 48 * 128;
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            // mlp_gate/up/down double as GDN n2 intermediates (z_out=num_v*v_dim,
            // a_out/b_out=num_v) in forward_gdn_n2. MoE has intermediate_size=0,
            // so floor to 8192 like the A-lane bufs[g] (see I_alloc in
            // alloc_buffers) — otherwise the B lane writes GDN state into a
            // 0-size buffer and corrupts spec verify (garbage long-gen output).
            int I_alloc = std::max(I, 8192);
            cudaMalloc(&bufs2[g].norm_out, H * sizeof(half));
            cudaMalloc(&bufs2[g].attn_out, std::max(H * 4, cfg.num_q_heads * cfg.head_dim * 2) * sizeof(half));
            cudaMalloc(&bufs2[g].mlp_gate, I_alloc * sizeof(half));
            cudaMalloc(&bufs2[g].mlp_up,   I_alloc * sizeof(half));
            cudaMalloc(&bufs2[g].mlp_down, std::max(H, I_alloc) * sizeof(half));
            cudaMalloc(&bufs2[g].residual, H * sizeof(half));
            // GDN second-token intermediates
            cudaMalloc(&gdn_bufs2[g].conv_out,   qkv_dim * sizeof(float));
            cudaMalloc(&gdn_bufs2[g].core_out,   v_total * sizeof(half));
            cudaMalloc(&gdn_bufs2[g].normed_out, v_total * sizeof(half));
            cudaMalloc(&gdn_bufs2[g].proj_out,   H * sizeof(half));
            // Attention second-token intermediates
            int num_q  = cfg.num_q_heads;
            int num_kv = cfg.num_kv_heads;
            int hd     = cfg.head_dim;
            cudaMalloc(&attn_bufs2[g].q_proj,      num_q  * hd * 2 * sizeof(half));
            cudaMalloc(&attn_bufs2[g].k_proj,      num_kv * hd * sizeof(half));
            cudaMalloc(&attn_bufs2[g].v_proj,      num_kv * hd * sizeof(half));
            cudaMalloc(&attn_bufs2[g].attn_scores, (size_t)num_q * max_seq * sizeof(float));
            cudaMalloc(&attn_bufs2[g].attn_out,    num_q  * hd * sizeof(half));
            cudaMalloc(&attn_bufs2[g].gate_buf,    num_q  * hd * sizeof(half));
        }
        n2_buffers_ready = true;
        printf("[SPEC] N=2 buffers allocated for speculative decoding\n");
    }

    // Allocate the third-token buffer set for MTP K=2 speculative decoding.
    // Layout mirrors alloc_buffers_n2() — if n2 is not yet allocated, we
    // allocate both. This is only called once per run; idempotent.
    void alloc_buffers_n3(int max_seq = 4096) {
        alloc_buffers_n2(max_seq);
        if (n3_buffers_ready) return;
        int H = cfg.hidden_size;
        int I = cfg.intermediate_size;
        int qkv_dim = 2 * 16 * 128 + 48 * 128;
        int v_total = 48 * 128;
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            // Same GDN-intermediate floor as bufs2 (MoE intermediate_size=0).
            int I_alloc = std::max(I, 8192);
            cudaMalloc(&bufs3[g].norm_out, H * sizeof(half));
            cudaMalloc(&bufs3[g].attn_out, std::max(H * 4, cfg.num_q_heads * cfg.head_dim * 2) * sizeof(half));
            cudaMalloc(&bufs3[g].mlp_gate, I_alloc * sizeof(half));
            cudaMalloc(&bufs3[g].mlp_up,   I_alloc * sizeof(half));
            cudaMalloc(&bufs3[g].mlp_down, std::max(H, I_alloc) * sizeof(half));
            cudaMalloc(&bufs3[g].residual, H * sizeof(half));
            cudaMalloc(&gdn_bufs3[g].conv_out,   qkv_dim * sizeof(float));
            cudaMalloc(&gdn_bufs3[g].core_out,   v_total * sizeof(half));
            cudaMalloc(&gdn_bufs3[g].normed_out, v_total * sizeof(half));
            cudaMalloc(&gdn_bufs3[g].proj_out,   H * sizeof(half));
            int num_q  = cfg.num_q_heads;
            int num_kv = cfg.num_kv_heads;
            int hd     = cfg.head_dim;
            cudaMalloc(&attn_bufs3[g].q_proj,      num_q  * hd * 2 * sizeof(half));
            cudaMalloc(&attn_bufs3[g].k_proj,      num_kv * hd * sizeof(half));
            cudaMalloc(&attn_bufs3[g].v_proj,      num_kv * hd * sizeof(half));
            cudaMalloc(&attn_bufs3[g].attn_scores, (size_t)num_q * max_seq * sizeof(float));
            cudaMalloc(&attn_bufs3[g].attn_out,    num_q  * hd * sizeof(half));
            cudaMalloc(&attn_bufs3[g].gate_buf,    num_q  * hd * sizeof(half));
        }
        n3_buffers_ready = true;
        printf("[SPEC] N=3 buffers allocated for MTP K=2 speculative decoding\n");
    }

    // Get tensor helper
    GPUTensor* t(const std::string& name) { return gpu->get(name); }
    std::string blk(int layer, const std::string& suffix) {
        return "blk." + std::to_string(layer) + "." + suffix;
    }

    // MLP: gate_proj(SiLU) * up_proj -> down_proj
    void forward_mlp(int layer, float* hidden, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& buf = bufs[g];
        int H = cfg.hidden_size;
        int I = cfg.intermediate_size;

        auto* norm_w = t(blk(layer, "post_attention_norm.weight"));
        if (!norm_w) norm_w = t(blk(layer, "ffn_norm.weight"));
        auto* gate_w = t(blk(layer, "ffn_gate.weight"));
        auto* up_w   = t(blk(layer, "ffn_up.weight"));
        auto* down_w = t(blk(layer, "ffn_down.weight"));

        if (!norm_w || !gate_w || !up_w || !down_w) {
            printf("MLP L%d: missing weights!\n", layer);
            return;
        }

        // RMSNorm: read FP32 hidden, output FP16 norm_out
        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w(buf.norm_out, hidden, (float*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);
        } else {
            rms_norm_f32in(buf.norm_out, hidden, (half*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);
        }

        gpu_qi[g].quantize(buf.norm_out, H, stream);
        quant_gemv(gate_w->data, gate_w->type, buf.norm_out, buf.mlp_gate, H, I, &gpu_qi[g], stream);
        quant_gemv(up_w->data, up_w->type, buf.norm_out, buf.mlp_up, H, I, &gpu_qi[g], stream);
        silu_mul_kernel<<<(I+255)/256, 256, 0, stream>>>(buf.mlp_gate, buf.mlp_up, I);
        gpu_qi_inter[g].quantize(buf.mlp_gate, I, stream);
        quant_gemv(down_w->data, down_w->type, buf.mlp_gate, buf.mlp_down, I, H, &gpu_qi_inter[g], stream);

        // Residual add into FP32 hidden
        add_kernel_f32<<<(H+255)/256, 256, 0, stream>>>(hidden, buf.mlp_down, H);
    }

    // ── MoE (Qwen3.x-A3B) ────────────────────────────────────────────────────
    void ensure_moe_buffers(int g) {
        if (moe_gate[g]) return;
        int H = cfg.hidden_size, E = cfg.num_experts;
        // gate/up scratch sized to the larger of routed-expert and shared-expert
        // intermediate dims (they're reused for both).
        int mI = std::max(cfg.moe_intermediate_size, cfg.shared_expert_intermediate_size);
        cudaSetDevice(g);
        cudaMalloc(&moe_gate[g], (size_t)mI * sizeof(half));
        cudaMalloc(&moe_up[g],   (size_t)mI * sizeof(half));
        cudaMalloc(&moe_down[g], (size_t)H  * sizeof(half));
        cudaMalloc(&moe_acc[g],  (size_t)H  * sizeof(float));
        cudaMalloc(&moe_logits[g], (size_t)E * sizeof(float));
        // Grouped path: topk experts in one launch.
        int topk = std::max(cfg.num_experts_per_tok, 1);
        cudaMalloc(&moe_gate_g[g], (size_t)topk * mI * sizeof(half));
        cudaMalloc(&moe_up_g[g],   (size_t)topk * mI * sizeof(half));
        cudaMalloc(&moe_down_g[g], (size_t)topk * H  * sizeof(half));
        cudaMalloc(&moe_ids_dev[g], (size_t)topk * sizeof(int));
        cudaMalloc(&moe_w_dev[g],   (size_t)topk * sizeof(float));
        cudaMallocHost(&moe_ids_host[g], (size_t)topk * sizeof(int));
        cudaMallocHost(&moe_w_host[g],   (size_t)topk * sizeof(float));
        // Batched chunk-prefill scratch (CHUNK_SIZE tokens × topk experts).
        size_t A = (size_t)CHUNK_SIZE * topk;
        cudaMalloc(&moe_norm_c[g],   (size_t)CHUNK_SIZE * H * sizeof(half));
        cudaMalloc(&moe_logits_c[g], (size_t)CHUNK_SIZE * E * sizeof(float));
        cudaMalloc(&moe_ids_c[g],    A * sizeof(int));
        cudaMalloc(&moe_w_c[g],      A * sizeof(float));
        cudaMalloc(&moe_gate_cg[g],  A * mI * sizeof(half));
        cudaMalloc(&moe_up_cg[g],    A * mI * sizeof(half));
        cudaMalloc(&moe_down_cg[g],  A * H  * sizeof(half));
    }

    // Single-token MoE FFN over `hidden` [H] (fp32, in/out via residual add).
    // Router → top-k softmax (normalized, norm_topk_prob) → per-expert SwiGLU
    // → weighted fp32 accumulate → residual. Sparse-expert GEMVs reuse the
    // norm_out Q8 quantization (one input, many expert weights).
    void moe_token_core(int layer, float* hidden, int g, cudaStream_t stream) {
        auto& buf = bufs[g];
        int H = cfg.hidden_size, E = cfg.num_experts;
        int topk = cfg.num_experts_per_tok, mI = cfg.moe_intermediate_size;

        // Pre-FFN norm: hybrid models name it post_attention_norm; fall back
        // to ffn_norm (dense Qwen3 naming). Mirror forward_mlp.
        auto* norm_w = t(blk(layer, "post_attention_norm.weight"));
        if (!norm_w) norm_w = t(blk(layer, "ffn_norm.weight"));
        if (!norm_w) { printf("[MoE] L%d: missing pre-FFN norm weight!\n", layer); return; }
        if (norm_w->type == GGML_TYPE_F32)
            rms_norm_f32in_f32w(buf.norm_out, hidden, (float*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);
        else
            rms_norm_f32in(buf.norm_out, hidden, (half*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);
        gpu_qi[g].quantize(buf.norm_out, H, stream);

        // Router logits → device float (moe_logits[g]). top-k is done on-device
        // in the grouped path (no host round-trip); the legacy fallback copies
        // these to host below.
        auto* router_w = t(blk(layer, "ffn_gate_inp.weight"));
        if (router_w->type == GGML_TYPE_F32) {
            qmoe_router_gemv_f32<<<E, 32, 0, stream>>>((float*)router_w->data,
                                                       buf.norm_out, moe_logits[g], H, E);
        } else {
            quant_gemv(router_w->data, router_w->type, buf.norm_out, buf.mlp_down,
                       H, E, &gpu_qi[g], stream);
            qmoe_f16_to_f32<<<(E + 255) / 256, 256, 0, stream>>>(buf.mlp_down, moe_logits[g], E);
        }

        // Expert tensors (separate gate/up; fused gate_up handled as fallback).
        auto* gate_exps = t(blk(layer, "ffn_gate_exps.weight"));
        auto* up_exps   = t(blk(layer, "ffn_up_exps.weight"));
        auto* down_exps = t(blk(layer, "ffn_down_exps.weight"));
        if (!gate_exps || !up_exps || !down_exps) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                printf("[MoE] L%d missing separate expert tensors "
                       "(gate=%p up=%p down=%p). This build expects "
                       "ffn_gate_exps/ffn_up_exps/ffn_down_exps; if the GGUF "
                       "fuses gate+up (ffn_gate_up_exps) the forward path needs "
                       "the fused variant.\n",
                       layer, (void*)gate_exps, (void*)up_exps, (void*)down_exps);
                fflush(stdout);
            }
            return;
        }
        // Per-expert byte stride. CRITICAL: gpu_loader repacks Q8_0 to
        // block_q8_0_aligned (36 B/block, not the GGUF 34 B), so ggml_row_bytes
        // (34-based) gives the wrong stride and reads misaligned garbage → NaN.
        // Use the GPU-resident block size for Q8_0; other quants aren't repacked.
        auto expert_bytes = [](ggml_type type, size_t n_elems) -> size_t {
            if (type == GGML_TYPE_Q8_0) return n_elems / 32 * 36;  // aligned on GPU
            return ggml_row_bytes(type, (int)n_elems);
        };
        size_t g_bytes = expert_bytes(gate_exps->type, (size_t)mI * H);
        size_t u_bytes = expert_bytes(up_exps->type,   (size_t)mI * H);
        size_t d_bytes = expert_bytes(down_exps->type, (size_t)H  * mI);

        qmoe_zero_f32<<<(H + 255) / 256, 256, 0, stream>>>(moe_acc[g], H);

        // Grouped path: all top-k experts in 3 GEMV launches (+ silu/quant/acc)
        // instead of ~8 launches/expert. The per-token MoE expert loop is
        // launch-bound on these cards (3B-active compute is trivial), so this
        // is the dominant decode win. Requires Q8_0 experts (the dp4a grouped
        // kernel). MOE_GROUPED_OFF=1 forces the legacy per-expert loop.
        static const bool moe_grouped = getenv("MOE_GROUPED_OFF") == nullptr;
        bool all_q8 = gate_exps->type == GGML_TYPE_Q8_0
                   && up_exps->type   == GGML_TYPE_Q8_0
                   && down_exps->type == GGML_TYPE_Q8_0;
        if (moe_grouped && all_q8) {
            // On-device top-k + softmax → moe_ids_dev/moe_w_dev. No host sync.
            qmoe_topk_softmax<<<1, 256, E * sizeof(float), stream>>>(
                moe_logits[g], E, topk, moe_ids_dev[g], moe_w_dev[g]);
            int thr = 128;
            // gate + up share the quantized norm_out (x_stride 0); output [topk, mI]
            gemv_q8_0_q8_experts<<<dim3(mI, topk), thr, 0, stream>>>(
                gate_exps->data, g_bytes, moe_ids_dev[g], gpu_qi[g].q8_buf, 0,
                moe_gate_g[g], H, mI);
            gemv_q8_0_q8_experts<<<dim3(mI, topk), thr, 0, stream>>>(
                up_exps->data, u_bytes, moe_ids_dev[g], gpu_qi[g].q8_buf, 0,
                moe_up_g[g], H, mI);
            int gmI = topk * mI;
            qmoe_clamp_fp16<<<(gmI + 255) / 256, 256, 0, stream>>>(moe_gate_g[g], gmI);
            qmoe_clamp_fp16<<<(gmI + 255) / 256, 256, 0, stream>>>(moe_up_g[g], gmI);
            silu_mul_kernel<<<(gmI + 255) / 256, 256, 0, stream>>>(moe_gate_g[g], moe_up_g[g], gmI);
            // Per-block q8 quant; each expert's mI segment quantizes independently
            // (mI%32==0), and the down kernel reads x at stride mI/32 per expert.
            gpu_qi_moe_g[g].quantize(moe_gate_g[g], gmI, stream);
            gemv_q8_0_q8_experts<<<dim3(H, topk), thr, 0, stream>>>(
                down_exps->data, d_bytes, moe_ids_dev[g], gpu_qi_moe_g[g].q8_buf, mI / 32,
                moe_down_g[g], mI, H);
            qmoe_weighted_acc_experts<<<(H + 255) / 256, 256, 0, stream>>>(
                moe_acc[g], moe_down_g[g], moe_w_dev[g], topk, H);
        } else {
            // Legacy per-expert loop: host top-k (copies device logits back).
            std::vector<float> logits(E);
            cudaMemcpyAsync(logits.data(), moe_logits[g], E * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            std::vector<int> order(E);
            for (int i = 0; i < E; i++) order[i] = i;
            std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                              [&](int a, int b) { return logits[a] > logits[b]; });
            float maxl = logits[order[0]], sum = 0.f;
            std::vector<float> w(topk);
            for (int i = 0; i < topk; i++) { w[i] = expf(logits[order[i]] - maxl); sum += w[i]; }
            for (int i = 0; i < topk; i++) w[i] /= sum;
            for (int t_ = 0; t_ < topk; t_++) {
                int e = order[t_];
                void* gp = (uint8_t*)gate_exps->data + (size_t)e * g_bytes;
                void* up = (uint8_t*)up_exps->data   + (size_t)e * u_bytes;
                quant_gemv(gp, gate_exps->type, buf.norm_out, moe_gate[g], H, mI, &gpu_qi[g], stream);
                quant_gemv(up, up_exps->type,   buf.norm_out, moe_up[g],   H, mI, &gpu_qi[g], stream);
                qmoe_clamp_fp16<<<(mI + 255) / 256, 256, 0, stream>>>(moe_gate[g], mI);
                qmoe_clamp_fp16<<<(mI + 255) / 256, 256, 0, stream>>>(moe_up[g], mI);
                silu_mul_kernel<<<(mI + 255) / 256, 256, 0, stream>>>(moe_gate[g], moe_up[g], mI);
                gpu_qi_moe[g].quantize(moe_gate[g], mI, stream);
                void* dp = (uint8_t*)down_exps->data + (size_t)e * d_bytes;
                quant_gemv(dp, down_exps->type, moe_gate[g], moe_down[g], mI, H, &gpu_qi_moe[g], stream);
                qmoe_clamp_fp16<<<(H + 255) / 256, 256, 0, stream>>>(moe_down[g], H);
                qmoe_acc_f32<<<(H + 255) / 256, 256, 0, stream>>>(moe_acc[g], moe_down[g], w[t_], H);
            }
        }

        // Shared expert (always active; optionally sigmoid-gated). qwen3_5_moe
        // adds shared_expert(h) to the routed sum. Skipped if tensors absent.
        if (cfg.shared_expert_intermediate_size > 0) {
            int sI = cfg.shared_expert_intermediate_size;
            auto* sg = t(blk(layer, "ffn_gate_shexp.weight"));
            auto* su = t(blk(layer, "ffn_up_shexp.weight"));
            auto* sd = t(blk(layer, "ffn_down_shexp.weight"));
            if (sg && su && sd) {
                // Sigmoid gate computed on-device (no host sync). moe_logits[g][0]
                // holds the gated weight; ungated (non-F32 gate / absent) → 1.0.
                auto* sgi = t(blk(layer, "ffn_gate_inp_shexp.weight"));
                bool gated = sgi && sgi->type == GGML_TYPE_F32;
                if (gated) {
                    qmoe_router_gemv_f32<<<1, 32, 0, stream>>>((float*)sgi->data,
                                                              buf.norm_out, moe_logits[g], H, 1);
                    qmoe_sigmoid_scalar<<<1, 1, 0, stream>>>(moe_logits[g]);
                }
                quant_gemv(sg->data, sg->type, buf.norm_out, moe_gate[g], H, sI, &gpu_qi[g], stream);
                quant_gemv(su->data, su->type, buf.norm_out, moe_up[g],   H, sI, &gpu_qi[g], stream);
                qmoe_clamp_fp16<<<(sI + 255) / 256, 256, 0, stream>>>(moe_gate[g], sI);
                qmoe_clamp_fp16<<<(sI + 255) / 256, 256, 0, stream>>>(moe_up[g], sI);
                silu_mul_kernel<<<(sI + 255) / 256, 256, 0, stream>>>(moe_gate[g], moe_up[g], sI);
                gpu_qi_moe[g].quantize(moe_gate[g], sI, stream);
                quant_gemv(sd->data, sd->type, moe_gate[g], moe_down[g], sI, H, &gpu_qi_moe[g], stream);
                qmoe_clamp_fp16<<<(H + 255) / 256, 256, 0, stream>>>(moe_down[g], H);
                if (gated)
                    qmoe_acc_f32_dev<<<(H + 255) / 256, 256, 0, stream>>>(moe_acc[g], moe_down[g], moe_logits[g], H);
                else
                    qmoe_acc_f32<<<(H + 255) / 256, 256, 0, stream>>>(moe_acc[g], moe_down[g], 1.0f, H);
            }
        }

        qmoe_f32_to_f16<<<(H + 255) / 256, 256, 0, stream>>>(moe_acc[g], moe_down[g], H);
        add_kernel_f32<<<(H + 255) / 256, 256, 0, stream>>>(hidden, moe_down[g], H);
    }

    void forward_moe(int layer, float* hidden, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        ensure_moe_buffers(g);
        moe_token_core(layer, hidden, g, stream);
    }

    // Batched chunk-prefill MoE: all N tokens through the router/top-k/experts
    // in a handful of launches instead of N × (~15 launches/token). The routed
    // experts run as a FLAT assignment list (N*topk entries) through
    // gemv_q8_0_q8_moe_chunk (one launch per gate/up/down); the shared expert
    // batches over N tokens via quant_gemv_chunk. No host sync. Requires F32
    // router + Q8_0 experts; caller (forward_moe_chunk) falls back to the loop
    // otherwise. Returns true if it handled the layer.
    bool forward_moe_chunk_batched(int layer, float* hidden_chunk, int N, int g, cudaStream_t stream) {
        int H = cfg.hidden_size, E = cfg.num_experts;
        int topk = cfg.num_experts_per_tok, mI = cfg.moe_intermediate_size;
        auto* router_w  = t(blk(layer, "ffn_gate_inp.weight"));
        auto* gate_exps = t(blk(layer, "ffn_gate_exps.weight"));
        auto* up_exps   = t(blk(layer, "ffn_up_exps.weight"));
        auto* down_exps = t(blk(layer, "ffn_down_exps.weight"));
        if (!router_w || router_w->type != GGML_TYPE_F32 || !gate_exps || !up_exps || !down_exps
            || gate_exps->type != GGML_TYPE_Q8_0 || up_exps->type != GGML_TYPE_Q8_0
            || down_exps->type != GGML_TYPE_Q8_0)
            return false;
        if (N > CHUNK_SIZE) return false;

        // 1. Batched RMSNorm + quantize (token-major).
        auto* norm_w = t(blk(layer, "post_attention_norm.weight"));
        if (!norm_w) norm_w = t(blk(layer, "ffn_norm.weight"));
        if (!norm_w) return false;
        if (norm_w->type == GGML_TYPE_F32)
            rms_norm_f32in_f32w(moe_norm_c[g], hidden_chunk, (float*)norm_w->data, N, H, cfg.rms_norm_eps, stream);
        else
            rms_norm_f32in(moe_norm_c[g], hidden_chunk, (half*)norm_w->data, N, H, cfg.rms_norm_eps, stream);
        gpu_qi_c[g].quantize_chunk(moe_norm_c[g], H, N, stream);

        // 2. Batched router → logits[N,E]; per-token GPU top-k → ids/weights.
        qmoe_router_gemv_chunk_f32<<<dim3(E, N), 32, 0, stream>>>(
            (float*)router_w->data, moe_norm_c[g], moe_logits_c[g], H, E, N);
        qmoe_topk_softmax_chunk<<<N, 256, E * sizeof(float), stream>>>(
            moe_logits_c[g], E, topk, moe_ids_c[g], moe_w_c[g], N);

        // 3. Routed experts as one flat assignment list (A = N*topk).
        int A = N * topk;
        size_t g_bytes = (size_t)mI * H / 32 * 36;
        size_t d_bytes = (size_t)H  * mI / 32 * 36;
        int thr = 128;
        gemv_q8_0_q8_moe_chunk<<<dim3(mI, A), thr, 0, stream>>>(
            gate_exps->data, g_bytes, moe_ids_c[g], gpu_qi_c[g].q8_buf, H / 32, topk,
            moe_gate_cg[g], H, mI);
        gemv_q8_0_q8_moe_chunk<<<dim3(mI, A), thr, 0, stream>>>(
            up_exps->data, g_bytes, moe_ids_c[g], gpu_qi_c[g].q8_buf, H / 32, topk,
            moe_up_cg[g], H, mI);
        int AmI = A * mI;
        qmoe_clamp_fp16<<<(AmI + 255) / 256, 256, 0, stream>>>(moe_gate_cg[g], AmI);
        qmoe_clamp_fp16<<<(AmI + 255) / 256, 256, 0, stream>>>(moe_up_cg[g], AmI);
        silu_mul_kernel<<<(AmI + 255) / 256, 256, 0, stream>>>(moe_gate_cg[g], moe_up_cg[g], AmI);
        gpu_qi_cg[g].quantize_chunk(moe_gate_cg[g], mI, A, stream);
        gemv_q8_0_q8_moe_chunk<<<dim3(H, A), thr, 0, stream>>>(
            down_exps->data, d_bytes, moe_ids_c[g], gpu_qi_cg[g].q8_buf, mI / 32, 0,
            moe_down_cg[g], mI, H);
        // 4. Scatter the routed sum back into the hidden chunk (residual add).
        qmoe_scatter_acc_chunk<<<dim3((H + 255) / 256, N), 256, 0, stream>>>(
            hidden_chunk, moe_down_cg[g], moe_w_c[g], topk, H, N);

        // 5. Shared expert, batched over N tokens (sigmoid-gated when F32 gate).
        if (cfg.shared_expert_intermediate_size > 0) {
            int sI = cfg.shared_expert_intermediate_size;
            auto* sg = t(blk(layer, "ffn_gate_shexp.weight"));
            auto* su = t(blk(layer, "ffn_up_shexp.weight"));
            auto* sd = t(blk(layer, "ffn_down_shexp.weight"));
            if (sg && su && sd) {
                auto* sgi = t(blk(layer, "ffn_gate_inp_shexp.weight"));
                bool gated = sgi && sgi->type == GGML_TYPE_F32;
                // gate[N]: sigmoid(router(sgi)) when gated, else 1.0. Stored in
                // moe_w_c (size A=N*topk ≥ N, reused after the routed scatter).
                if (gated) {
                    qmoe_router_gemv_chunk_f32<<<dim3(1, N), 32, 0, stream>>>(
                        (float*)sgi->data, moe_norm_c[g], moe_w_c[g], H, 1, N);
                    qmoe_sigmoid_vec<<<(N + 255) / 256, 256, 0, stream>>>(moe_w_c[g], N);
                } else {
                    qmoe_fill_f32<<<(N + 255) / 256, 256, 0, stream>>>(moe_w_c[g], 1.0f, N);
                }
                quant_gemv_chunk(sg->data, sg->type, gpu_qi_c[g].q8_buf, moe_gate_cg[g], H, sI, N, stream);
                quant_gemv_chunk(su->data, su->type, gpu_qi_c[g].q8_buf, moe_up_cg[g],   H, sI, N, stream);
                int NsI = N * sI;
                qmoe_clamp_fp16<<<(NsI + 255) / 256, 256, 0, stream>>>(moe_gate_cg[g], NsI);
                qmoe_clamp_fp16<<<(NsI + 255) / 256, 256, 0, stream>>>(moe_up_cg[g], NsI);
                silu_mul_kernel<<<(NsI + 255) / 256, 256, 0, stream>>>(moe_gate_cg[g], moe_up_cg[g], NsI);
                gpu_qi_cg[g].quantize_chunk(moe_gate_cg[g], sI, N, stream);
                quant_gemv_chunk(sd->data, sd->type, gpu_qi_cg[g].q8_buf, moe_down_cg[g], sI, H, N, stream);
                // topk=1 scatter: each token adds gate[token]*shared_down[token].
                qmoe_scatter_acc_chunk<<<dim3((H + 255) / 256, N), 256, 0, stream>>>(
                    hidden_chunk, moe_down_cg[g], moe_w_c[g], 1, H, N);
            }
        }
        return true;
    }

    // Chunked-prefill MoE: token-major [n_tokens × H]. Batched path (no host
    // sync, ~constant launches) with a per-token fallback for non-Q8_0 experts /
    // non-F32 router / MOE_CHUNK_BATCHED_OFF.
    void forward_moe_chunk(int layer, float* hidden_chunk, int n_tokens, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        ensure_moe_buffers(g);
        int H = cfg.hidden_size;
        static const bool batched = getenv("MOE_CHUNK_BATCHED_OFF") == nullptr;
        if (batched && forward_moe_chunk_batched(layer, hidden_chunk, n_tokens, g, stream))
            return;
        for (int t_ = 0; t_ < n_tokens; t_++)
            moe_token_core(layer, hidden_chunk + (size_t)t_ * H, g, stream);
    }

    // N=2 batched MLP. Processes two hidden states sharing the same weight
    // loads (gate_proj, up_proj, down_proj). Memory traffic stays at the
    // N=1 cost since each weight word is read once and dp4a runs against
    // both inputs. The two RMSNorms, two silu_muls, and two residual adds
    // are sequential but cheap (no GEMV).
    void forward_mlp_n2(int layer, float* hidden_a, float* hidden_b, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& bA = bufs[g];
        auto& bB = bufs2[g];
        int H = cfg.hidden_size;
        int I = cfg.intermediate_size;

        auto* norm_w = t(blk(layer, "post_attention_norm.weight"));
        if (!norm_w) norm_w = t(blk(layer, "ffn_norm.weight"));
        auto* gate_w = t(blk(layer, "ffn_gate.weight"));
        auto* up_w   = t(blk(layer, "ffn_up.weight"));
        auto* down_w = t(blk(layer, "ffn_down.weight"));

        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w_n2(bA.norm_out, bB.norm_out,
                                   hidden_a, hidden_b,
                                   (float*)norm_w->data, H, cfg.rms_norm_eps, stream);
        } else {
            rms_norm_f32in_n2(bA.norm_out, bB.norm_out,
                              hidden_a, hidden_b,
                              (half*)norm_w->data, H, cfg.rms_norm_eps, stream);
        }

        gpu_qi[g].quantize(bA.norm_out, H, stream);
        gpu_qi2[g].quantize(bB.norm_out, H, stream);

        quant_gemv_n2(gate_w->data, gate_w->type,
                      bA.norm_out, bB.norm_out,
                      bA.mlp_gate, bB.mlp_gate,
                      H, I, &gpu_qi[g], &gpu_qi2[g], stream);
        quant_gemv_n2(up_w->data, up_w->type,
                      bA.norm_out, bB.norm_out,
                      bA.mlp_up, bB.mlp_up,
                      H, I, &gpu_qi[g], &gpu_qi2[g], stream);

        { dim3 sg((I+255)/256, 2);
          silu_mul_n2_kernel<<<sg, 256, 0, stream>>>(
              bA.mlp_gate, bA.mlp_up,
              bB.mlp_gate, bB.mlp_up, I); }

        gpu_qi_inter[g].quantize(bA.mlp_gate, I, stream);
        gpu_qi_inter2[g].quantize(bB.mlp_gate, I, stream);

        quant_gemv_n2(down_w->data, down_w->type,
                      bA.mlp_gate, bB.mlp_gate,
                      bA.mlp_down, bB.mlp_down,
                      I, H, &gpu_qi_inter[g], &gpu_qi_inter2[g], stream);

        { dim3 ag((H+255)/256, 2);
          add_kernel_f32_n2<<<ag, 256, 0, stream>>>(
              hidden_a, bA.mlp_down,
              hidden_b, bB.mlp_down, H); }
    }

    // ============ forward_mlp_n3 (MTP K=2 three-stream batch) ============
    // Same as forward_mlp_n2 with a third lane. Weight is read ONCE and fed
    // into three dp4a accumulators via quant_gemv_n3.
    void forward_mlp_n3(int layer, float* hidden_a, float* hidden_b, float* hidden_c,
                        cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& bA = bufs[g];
        auto& bB = bufs2[g];
        auto& bC = bufs3[g];
        int H = cfg.hidden_size;
        int I = cfg.intermediate_size;

        auto* norm_w = t(blk(layer, "post_attention_norm.weight"));
        if (!norm_w) norm_w = t(blk(layer, "ffn_norm.weight"));
        auto* gate_w = t(blk(layer, "ffn_gate.weight"));
        auto* up_w   = t(blk(layer, "ffn_up.weight"));
        auto* down_w = t(blk(layer, "ffn_down.weight"));

        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w_n3(bA.norm_out, bB.norm_out, bC.norm_out,
                                   hidden_a, hidden_b, hidden_c,
                                   (float*)norm_w->data, H, cfg.rms_norm_eps, stream);
        } else {
            rms_norm_f32in_n3(bA.norm_out, bB.norm_out, bC.norm_out,
                              hidden_a, hidden_b, hidden_c,
                              (half*)norm_w->data, H, cfg.rms_norm_eps, stream);
        }

        gpu_qi[g].quantize(bA.norm_out, H, stream);
        gpu_qi2[g].quantize(bB.norm_out, H, stream);
        gpu_qi3[g].quantize(bC.norm_out, H, stream);

        quant_gemv_n3(gate_w->data, gate_w->type,
                      bA.norm_out, bB.norm_out, bC.norm_out,
                      bA.mlp_gate, bB.mlp_gate, bC.mlp_gate,
                      H, I, &gpu_qi[g], &gpu_qi2[g], &gpu_qi3[g], stream);
        quant_gemv_n3(up_w->data, up_w->type,
                      bA.norm_out, bB.norm_out, bC.norm_out,
                      bA.mlp_up, bB.mlp_up, bC.mlp_up,
                      H, I, &gpu_qi[g], &gpu_qi2[g], &gpu_qi3[g], stream);

        { dim3 sg((I+255)/256, 3);
          silu_mul_n3_kernel<<<sg, 256, 0, stream>>>(
              bA.mlp_gate, bA.mlp_up,
              bB.mlp_gate, bB.mlp_up,
              bC.mlp_gate, bC.mlp_up, I); }

        gpu_qi_inter[g].quantize(bA.mlp_gate, I, stream);
        gpu_qi_inter2[g].quantize(bB.mlp_gate, I, stream);
        gpu_qi_inter3[g].quantize(bC.mlp_gate, I, stream);

        quant_gemv_n3(down_w->data, down_w->type,
                      bA.mlp_gate, bB.mlp_gate, bC.mlp_gate,
                      bA.mlp_down, bB.mlp_down, bC.mlp_down,
                      I, H, &gpu_qi_inter[g], &gpu_qi_inter2[g], &gpu_qi_inter3[g], stream);

        { dim3 ag((H+255)/256, 3);
          add_kernel_f32_n3<<<ag, 256, 0, stream>>>(
              hidden_a, bA.mlp_down,
              hidden_b, bB.mlp_down,
              hidden_c, bC.mlp_down, H); }
    }

    // ============ Attention state ============
    RoPETable rope;
    TurboQuantCache tq_cache;
    int cur_seq_len = 0;
    
    // Attention temp buffers (per GPU)
    struct AttnBuffers {
        half* q_proj;      // [num_q * head_dim * 2] (Q + gate)
        half* k_proj;      // [num_kv * head_dim]
        half* v_proj;      // [num_kv * head_dim]
        // ATTN_QKV_FUSE: contiguous [q_n + k_n + v_n] half. Sub-pointers alias
        // q_proj / k_proj / v_proj for the duration of one forward_attn call
        // (RAII guard restores originals on exit).
        half* fused_qkv_out = nullptr;
        float* attn_scores; // [num_q * max_seq]
        half* gate_buf;     // [num_q * head_dim] for output gate
        half* attn_out;     // [num_q * head_dim]
        // Chunked-prefill staging: batched Q/K/V projection outputs.
        // Allocated lazily on first chunked attn forward.
        half* attn_chunk_q = nullptr;   // [CHUNK_SIZE * q_out_dim]  raw QKV proj output
        half* attn_chunk_k = nullptr;   // [CHUNK_SIZE * kv_dim]
        half* attn_chunk_v = nullptr;   // [CHUNK_SIZE * kv_dim]
        // Chunked-attention compute scratch:
        half*  attn_chunk_q_post = nullptr;  // [CHUNK_SIZE * num_q * head_dim] post deinterleave + head_rms + RoPE
        half*  attn_chunk_gate   = nullptr;  // [CHUNK_SIZE * num_q * head_dim] gate slice from QKV
        float* attn_chunk_scores = nullptr;  // [ATTN_NB * num_q * max_seq] softmax scratch (sub-chunk reuse)
        half*  attn_chunk_out    = nullptr;  // [CHUNK_SIZE * num_q * head_dim] V-weighted output
        half*  attn_chunk_oproj  = nullptr;  // [CHUNK_SIZE * H] o_proj output staging
        // Split-K FA scratch (lazy-alloc when FA_SK env opts in, sized for
        // K_SPLITS_MAX=8). Layouts: [num_q, ATTN_NB, K_SPLITS_MAX] for m/l,
        // [num_q, ATTN_NB, K_SPLITS_MAX, HD] for o (fp32 to keep merge bit-stable
        // with the base FA kernel, project_qwopus_lang_bias).
        float* attn_split_m = nullptr;
        float* attn_split_l = nullptr;
        float* attn_split_o = nullptr;
        // MInference-style block-sparse scratch (lazy-alloc when MINF_SPARSE_ATTN
        // is on). sparse_k_pool: per-layer mean-pooled K[num_kv, N_blocks, HD].
        // sparse_k_pool_max: companion max-abs signature for mBSA. Same shape.
        // sparse_block_index: per-call top-k selection [num_kv, ATTN_NB, top_k].
        // Sized for the worst case (max_seq, top_k_max=64) so we don't realloc
        // mid-stream.
        half* sparse_k_pool     = nullptr;
        half* sparse_k_pool_max = nullptr;
        int*  sparse_block_index = nullptr;
        // Vertical-slash scratch: per-kv top-V vertical block ids and top-S
        // slash deltas. Recomputed each chunk from this chunk's queries.
        int*  sparse_vertical_idx = nullptr;
        int*  sparse_slash_idx    = nullptr;
    };
    AttnBuffers attn_bufs[4];
    AttnBuffers attn_bufs2[4];  // second-token attn buffers, for forward_attn_n2
    AttnBuffers attn_bufs3[4];  // third-token attn buffers, for forward_attn_n3 (MTP K=2)
    
    void init_attention(int slot_max_seq_arg, int num_slots_arg = 1) {
        std::vector<int> caps(num_slots_arg > 0 ? num_slots_arg : 1, slot_max_seq_arg);
        init_attention_caps(caps);
    }

    // Asymmetric per-slot capacities. `caps[i]` is the max context (in tokens)
    // for slot `i`. KV/GDN/FA buffers allocate enough for the cumulative sum.
    void init_attention_caps(const std::vector<int>& caps) {
        if (caps.empty()) {
            fprintf(stderr, "init_attention_caps: empty caps vector\n");
            return;
        }
        int num_slots_arg = (int)caps.size();
        int max_cap = 0;
        for (int c : caps) if (c > max_cap) max_cap = c;
        int num_q = cfg.num_q_heads;
        int num_kv = cfg.num_kv_heads;
        int hd = cfg.head_dim;
        float theta = cfg.rope_freq_base;
        int score_max_len = max_cap;

        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&attn_bufs[g].q_proj, num_q * hd * 2 * sizeof(half));
            cudaMalloc(&attn_bufs[g].k_proj, num_kv * hd * sizeof(half));
            cudaMalloc(&attn_bufs[g].v_proj, num_kv * hd * sizeof(half));
            cudaMalloc(&attn_bufs[g].attn_scores, num_q * score_max_len * sizeof(float));
            cudaMalloc(&attn_bufs[g].attn_out, num_q * hd * sizeof(half));
            cudaMalloc(&attn_bufs[g].gate_buf, num_q * hd * sizeof(half));
            cudaMalloc(&attn_bufs[g].fused_qkv_out,
                       (num_q * hd * 2 + 2 * num_kv * hd) * sizeof(half));
        }

        rope.init(max_cap, hd, cfg.rope_dim, theta, gpu->num_gpus);

        int attn_count = 0;
        for (int l = 0; l < cfg.num_layers; l++)
            if (is_attn_layer(l)) attn_count++;

        bool asymmetric = false;
        for (int c : caps) if (c != caps[0]) { asymmetric = true; break; }
        printf("Attention: %d layers, %d Q heads, %d KV heads, head_dim=%d\n",
            attn_count, num_q, num_kv, hd);
        if (asymmetric) {
            printf("Max context: asymmetric (");
            for (size_t i = 0; i < caps.size(); i++)
                printf("%s%d", i ? "," : "", caps[i]);
            int total = 0; for (int c : caps) total += c;
            printf(") = %d physical\n", total);
        } else {
            printf("Max context: %d tokens (per slot, %d slots = %d physical)\n",
                   caps[0], num_slots_arg, caps[0] * num_slots_arg);
        }
        init_kv_cache_caps(caps);
    }
    
    // Attention forward (seq_len=1, token generation)
    // ============ FP16 KV Cache (memory + speed) ============
    // Precision is recovered by the attn_score_kernel cross-warp fix +
    // FP32 attention compute (accumulators, softmax). The cache itself
    // can stay fp16: the K/V tensors come straight from fp16 head_norm
    // outputs and storing them in fp16 has no extra cast loss.
    struct KVCache {
        half* k;  // [max_seq, num_kv * head_dim]
        half* v;  // [max_seq, num_kv * head_dim]
    };
    std::unordered_map<int, KVCache> kv_caches;
    // Per-slot max attention range. Existing code (chunk scores stride, RoPE
    // table size, etc.) reads `kv_max_seq` as "max seq the model can attend
    // to in one request" — preserved as that semantic.
    int kv_max_seq = 0;

    // Continuous batching: virtual KV partitioning.
    //   physical KV layout = [slot 0 KV][slot 1 KV] ... [slot N-1 KV]
    //   each slot's logical pos p maps to physical kv_slot = slot*kv_max_seq + p.
    //   Attention against slot s passes k_cache + slot*kv_max_seq*kv_dim and
    //   seq_len = cur_len_in_slot, so the existing per-token kernels work
    //   unchanged. GDN recurrent state is duplicated per slot since it has no
    //   positional indexing.
    int num_slots = 1;
    int slot_max_seq = 0;            // MAX across slot_caps; sizes FA scratch / RoPE.
    int kv_total_seq = 0;            // SUM of slot_caps (physical KV capacity in tokens).
    std::vector<int>    slot_caps;             // [num_slots] per-slot capacity in tokens
    std::vector<size_t> slot_token_offsets;    // [num_slots] cumulative offset in tokens
    // kv_slot_offset returns the token offset of `slot` in the per-layer KV
    // buffer (cumulative for asymmetric capacities). Returned as size_t since
    // callers multiply by kv_dim/bpt to get byte/element offsets.
    // Out-of-range slots return 0 — the silent fallback `slot * slot_max_seq`
    // was wrong for asymmetric caps and masked logic bugs. Returning 0 with
    // an stderr warning surfaces the misuse.
    size_t kv_slot_offset(int slot) const {
        if (slot < 0 || slot >= (int)slot_token_offsets.size()) {
            fprintf(stderr, "[kv_slot_offset] OOB slot=%d (num_slots=%zu) — init not done?\n",
                    slot, slot_token_offsets.size());
            return 0;
        }
        return slot_token_offsets[slot];
    }
    int slot_capacity(int slot) const {
        if (slot < 0 || slot >= (int)slot_caps.size()) {
            fprintf(stderr, "[slot_capacity] OOB slot=%d (num_slots=%zu)\n",
                    slot, slot_caps.size());
            return 0;
        }
        return slot_caps[slot];
    }

    // ============ TurboQuant 3-bit KV cache (optional, MTP_TQ=1) ============
    // Drop-in replacement for the fp16 KV cache that compresses K/V to ~52
    // bytes per 128-element block (~4.7x smaller than fp16). On the read
    // path the active range [0..seq_len) is dequantized into
    // tq_k_buf / tq_v_buf which are then fed to the existing attn_score and
    // attn_value kernels — no kernel rewrite required.
    struct TQKVCache {
        block_tq3* k;       // [max_seq, blocks_per_token]
        block_tq3* v;
        int blocks_per_token;
    };
    std::unordered_map<int, TQKVCache> tq_kv_caches;
    half* tq_k_buf[4] = {nullptr,nullptr,nullptr,nullptr};  // per-GPU dequant scratch
    half* tq_v_buf[4] = {nullptr,nullptr,nullptr,nullptr};
    bool use_turboquant = false;
    // Per-layer "highest fp16-cached pos" for the per-token TQ path. -1 means
    // tq_k_buf/tq_v_buf has nothing valid yet (next forward_attn will bulk-
    // dequant [0, kv_slot+1)). Subsequent calls only dequant the new range
    // [tq_decoded_until[L]+1, kv_slot+1). Reset to -1 on reset_all_states.
    std::vector<int> tq_decoded_until;
    // Per-attn-layer fp16 KV cache that mirrors the TQ cache: TQ encode and
    // a half-precision write happen in the same step, then per-token
    // attention reads the fp16 layout directly (skips the cooperative TQ
    // decode that profiling measured at 84% of fused FA cycles). Sized to
    // `kv_total_seq * kv_dim` per attn layer = 1 GB at max_seq 262144 (per
    // layer per GPU), so leave it opt-in via FA_FP16_CACHE=1 — only safe
    // when max-seq is small enough to fit in HBM (≤ ~32 K with the 27 B
    // weights resident).
    struct FP16DecCache { half* k = nullptr; half* v = nullptr; };
    std::unordered_map<int, FP16DecCache> fp16_dec_caches;
    bool fp16_dec_cache_on = false;
    // ===== MS-block-sparse verify K-pool (DFLASH_VERIFY_BLOCKSPARSE=1) =====
    // Per-attn-layer persistent mean + signed-max-abs K signatures over 64-token
    // position-blocks. Lets the DFlash TQ verify attend only the content-selected
    // top-k blocks (MS selector, same as prefill) → O(top_k·64) TQ-decodes per
    // verify instead of O(context), losslessly (keeps single-token needles that
    // sink+window drops). Pools live on the layer's GPU; fixed stride n_blocks_max
    // so incremental growth never moves existing signatures. pooled_pos = leading
    // positions already folded into FROZEN (complete) blocks; the current partial
    // block is rebuilt every step. Reset to 0 on any KV reset.
    struct VerifyKPool {
        half* k_pool     = nullptr;   // [num_kv, n_blocks_max, HD] mean
        half* k_pool_max = nullptr;   // [num_kv, n_blocks_max, HD] signed max-abs
        int   n_blocks_max = 0;
        int   pooled_pos   = 0;
    };
    std::unordered_map<int, VerifyKPool> verify_kpool;
    // ============ Q8_0 KV cache (Q8KV=1) ============
    // Alternative to the TurboQuant path for hardware that can't run the
    // cooperative TQ3 decode (Walsh-Hadamard + cross-warp shuffle) at
    // tensor-core speed (CMP / sm_70). Each block_q8_0_aligned holds 32
    // int8 + uint16 pad + half scale = 36 B / 32 elem (≈ 1.125 B / elem,
    // 2.6× the TQ footprint, half of fp16). Decode is a single int8 ×
    // fp16-scale multiply per element — no shuffle, no centroid lookup —
    // which keeps long-context gen close to HBM-bound.
    //
    // Layout: per attn layer per GPU, block_q8_0_aligned[kv_total_seq * bpt],
    // bpt = kv_dim / 32. At kv_dim = 1024 we have 32 blocks/token.
    struct Q8KVCache {
        block_q8_0_aligned* k = nullptr;
        block_q8_0_aligned* v = nullptr;
        int blocks_per_token = 0;
    };
    std::unordered_map<int, Q8KVCache> q8_kv_caches;
    bool use_q8_kv = false;
    // MInference-style sparse attention runtime config. Loaded once from env
    // (MINF_SPARSE_ATTN, MINF_PROFILE_PATH, MINF_BUDGET, MINF_MIN_SEQ) at
    // init_kv_cache. When `enabled` is false the chunked attn path stays on
    // the existing dense flash_attn_chunk_fused_split kernel.
    SparseRuntime sparse_rt;

    void init_kv_cache(int slot_max_seq_arg, int num_slots_arg = 1) {
        std::vector<int> caps(num_slots_arg > 0 ? num_slots_arg : 1, slot_max_seq_arg);
        init_kv_cache_caps(caps);
    }

    void init_kv_cache_caps(const std::vector<int>& caps) {
        if (caps.empty()) {
            fprintf(stderr, "init_kv_cache_caps: empty caps vector\n");
            return;
        }
        num_slots    = (int)caps.size();
        slot_caps    = caps;
        slot_token_offsets.assign(num_slots, 0);
        size_t cum = 0;
        for (int i = 0; i < num_slots; i++) {
            slot_token_offsets[i] = cum;
            cum += (size_t)caps[i];
        }
        int max_cap = 0;
        for (int c : caps) if (c > max_cap) max_cap = c;
        slot_max_seq = max_cap;                // max-of-caps; sizes scratch / RoPE
        kv_max_seq   = max_cap;
        // kv_total_seq is int for back-compat with the many downstream sites
        // (FA scratch sizing, etc.) that compute `(int)pos < kv_total_seq`.
        // INT_MAX = 2.1B; with TQ3 KV @ ~52 B/token that's 110 GB → out of
        // reach for any single host. Reject (don't silently clamp) the
        // pathological config: clamping `kv_total_seq` while leaving the true
        // `slot_token_offsets` unclamped would let later slots index past the
        // allocated buffer.
        if (cum > (size_t)INT_MAX) {
            fprintf(stderr, "[KV] sum of slot caps %zu exceeds INT_MAX. "
                            "Lower --slot-caps or shrink num_slots.\n", cum);
            abort();
        }
        kv_total_seq = (int)cum;               // SUM of caps (physical capacity)
        int num_kv = cfg.num_kv_heads;
        int hd = cfg.head_dim;
        int kv_dim = num_kv * hd;
        size_t kv_size = (size_t)kv_total_seq * kv_dim * sizeof(half);
        use_turboquant = (getenv("MTP_TQ") != nullptr);
        use_q8_kv      = (getenv("Q8KV")    != nullptr);
        // Sparse-attention runtime config. Reads MINF_* env, optionally loads
        // the offline profile. Must run BEFORE any forward_attn_chunk call.
        // No buffers allocated here — those are lazy in forward_attn_chunk
        // because they live on the AttnBuffers per-GPU scratch.
        sparse_rt = parse_sparse_runtime();
        if (use_turboquant && use_q8_kv) {
            fprintf(stderr, "[KV] both MTP_TQ and Q8KV set — ignoring Q8KV (TQ wins)\n");
            use_q8_kv = false;
        }

        if (use_q8_kv) {
            // Q8_0 KV: 36 B / 32 elem aligned blocks. bpt = kv_dim / 32.
            int bpt = kv_dim / 32;
            size_t per_layer_bytes = (size_t)kv_total_seq * bpt * sizeof(block_q8_0_aligned);
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                if (!is_attn_layer(layer)) continue;
                int g = gpu->layer_gpu[layer];
                cudaSetDevice(g);
                Q8KVCache kv;
                kv.blocks_per_token = bpt;
                cudaMalloc(&kv.k, per_layer_bytes);
                cudaMalloc(&kv.v, per_layer_bytes);
                cudaMemset(kv.k, 0, per_layer_bytes);
                cudaMemset(kv.v, 0, per_layer_bytes);
                q8_kv_caches[layer] = kv;
            }
            // Per-GPU fp16 dequant scratch (same role as the TQ chunked path's
            // tq_k_buf / tq_v_buf — Q8 chunked attn and the per-token n2/n3
            // paths bulk-dequant into this buffer before running fp16 kernels).
            for (int g = 0; g < gpu->num_gpus; g++) {
                cudaSetDevice(g);
                cudaMalloc(&tq_k_buf[g], (size_t)kv_total_seq * kv_dim * sizeof(half));
                cudaMalloc(&tq_v_buf[g], (size_t)kv_total_seq * kv_dim * sizeof(half));
            }
            float total_mb = (float)q8_kv_caches.size() * per_layer_bytes * 2 / 1e6;
            float fp16_mb  = (float)q8_kv_caches.size() * kv_size * 2 / 1e6;
            printf("KV cache (Q8_0): %d layers, slot_max_seq=%d × %d slots = %d total, %.1f MB (vs %.1f MB fp16, %.2fx compression)\n",
                (int)q8_kv_caches.size(), slot_max_seq, num_slots, kv_total_seq, total_mb, fp16_mb, fp16_mb / total_mb);
            return;
        }

        if (use_turboquant) {
            // 3-bit TurboQuant cache. blocks_per_token = kv_dim / 128.
            // For Qwen35: 4*256 / 128 = 8 blocks per token, 52 B each
            // = 416 B per token (vs 2048 B fp16) → 4.92x compression.
            int bpt = kv_dim / TQ_BLOCK_SIZE;
            for (int g = 0; g < gpu->num_gpus; g++) tq3_init_signs(g);
            size_t per_layer_bytes = (size_t)kv_total_seq * bpt * sizeof(block_tq3);
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                if (!is_attn_layer(layer)) continue;
                int g = gpu->layer_gpu[layer];
                cudaSetDevice(g);
                TQKVCache kv;
                kv.blocks_per_token = bpt;
                cudaMalloc(&kv.k, per_layer_bytes);
                cudaMalloc(&kv.v, per_layer_bytes);
                cudaMemset(kv.k, 0, per_layer_bytes);
                cudaMemset(kv.v, 0, per_layer_bytes);
                tq_kv_caches[layer] = kv;
            }
            // Per-GPU dequant scratch (size = kv_total_seq * kv_dim half).
            // Used by chunked attn TQ path which reads/writes [0..start_pos+n).
            for (int g = 0; g < gpu->num_gpus; g++) {
                cudaSetDevice(g);
                cudaMalloc(&tq_k_buf[g], (size_t)kv_total_seq * kv_dim * sizeof(half));
                cudaMalloc(&tq_v_buf[g], (size_t)kv_total_seq * kv_dim * sizeof(half));
            }
            tq_decoded_until.assign(cfg.num_layers, -1);
            float total_mb = (float)tq_kv_caches.size() * per_layer_bytes * 2 / 1e6;
            float fp16_mb  = (float)tq_kv_caches.size() * kv_size * 2 / 1e6;
            printf("KV cache (TurboQuant 3-bit): %d layers, slot_max_seq=%d × %d slots = %d total, %.1f MB (vs %.1f MB fp16, %.2fx compression)\n",
                (int)tq_kv_caches.size(), slot_max_seq, num_slots, kv_total_seq, total_mb, fp16_mb, fp16_mb / total_mb);
            // Optional fp16 mirror so attention can skip the per-token cooperative
            // TQ decode. Only allocated when env asks for it; the caller is
            // responsible for keeping max-seq small enough to fit.
            fp16_dec_cache_on = (getenv("FA_FP16_CACHE") != nullptr);
            if (fp16_dec_cache_on) {
                size_t bytes_per_layer = (size_t)kv_total_seq * kv_dim * sizeof(half);
                size_t total_alloc = 0;
                for (int layer = 0; layer < cfg.num_layers; layer++) {
                    if (!is_attn_layer(layer)) continue;
                    int g = gpu->layer_gpu[layer];
                    cudaSetDevice(g);
                    FP16DecCache fc;
                    cudaMalloc(&fc.k, bytes_per_layer);
                    cudaMalloc(&fc.v, bytes_per_layer);
                    cudaMemset(fc.k, 0, bytes_per_layer);
                    cudaMemset(fc.v, 0, bytes_per_layer);
                    fp16_dec_caches[layer] = fc;
                    total_alloc += bytes_per_layer * 2;
                }
                printf("FA fp16 decode cache: %d attn layers, %.1f MB total (skips per-token TQ decode)\n",
                    (int)fp16_dec_caches.size(), total_alloc / 1e6);
            }
            return;
        }

        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (!is_attn_layer(layer)) continue;
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            KVCache kv;
            cudaMalloc(&kv.k, kv_size);
            cudaMalloc(&kv.v, kv_size);
            cudaMemset(kv.k, 0, kv_size);
            cudaMemset(kv.v, 0, kv_size);
            kv_caches[layer] = kv;
        }
        printf("KV cache (FP16): %d layers, slot_max_seq=%d × %d slots = %d total, %.1f MB\n",
            (int)kv_caches.size(), slot_max_seq, num_slots, kv_total_seq,
            (float)kv_caches.size() * kv_size / 1e6);
    }

    void reset_kv_cache() {
        for (auto& kv : verify_kpool) kv.second.pooled_pos = 0;  // invalidate MS verify pool
        int num_kv = cfg.num_kv_heads;
        int hd = cfg.head_dim;
        int kv_dim = num_kv * hd;
        if (use_q8_kv) {
            int bpt = kv_dim / 32;
            size_t sz = (size_t)kv_total_seq * bpt * sizeof(block_q8_0_aligned);
            for (auto& [layer, kv] : q8_kv_caches) {
                int g = gpu->layer_gpu[layer];
                cudaSetDevice(g);
                cudaMemset(kv.k, 0, sz);
                cudaMemset(kv.v, 0, sz);
            }
            return;
        }
        if (use_turboquant) {
            int bpt = kv_dim / TQ_BLOCK_SIZE;
            size_t sz = (size_t)kv_total_seq * bpt * sizeof(block_tq3);
            for (auto& [layer, kv] : tq_kv_caches) {
                int g = gpu->layer_gpu[layer];
                cudaSetDevice(g);
                cudaMemset(kv.k, 0, sz);
                cudaMemset(kv.v, 0, sz);
            }
            if (fp16_dec_cache_on) {
                size_t fc_sz = (size_t)kv_total_seq * kv_dim * sizeof(half);
                for (auto& [layer, fc] : fp16_dec_caches) {
                    int g = gpu->layer_gpu[layer];
                    cudaSetDevice(g);
                    cudaMemset(fc.k, 0, fc_sz);
                    cudaMemset(fc.v, 0, fc_sz);
                }
            }
            return;
        }
        size_t kv_size = (size_t)kv_total_seq * kv_dim * sizeof(half);
        for (auto& [layer, kv] : kv_caches) {
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            cudaMemset(kv.k, 0, kv_size);
            cudaMemset(kv.v, 0, kv_size);
        }
    }

    // Reset only slot s's KV range. Cheaper than full reset when num_slots > 1.
    void reset_kv_slot(int slot) {
        if (slot < 0 || slot >= num_slots) return;
        for (auto& kv : verify_kpool) kv.second.pooled_pos = 0;  // invalidate MS verify pool
        int num_kv = cfg.num_kv_heads, hd = cfg.head_dim;
        int kv_dim = num_kv * hd;
        if (use_q8_kv) {
            int bpt = kv_dim / 32;
            size_t per_slot = (size_t)slot_capacity(slot) * bpt * sizeof(block_q8_0_aligned);
            size_t off      = kv_slot_offset(slot) * bpt;
            for (auto& [layer, kv] : q8_kv_caches) {
                int g = gpu->layer_gpu[layer];
                cudaSetDevice(g);
                cudaMemset(kv.k + off, 0, per_slot);
                cudaMemset(kv.v + off, 0, per_slot);
            }
            return;
        }
        if (use_turboquant) {
            int bpt = kv_dim / TQ_BLOCK_SIZE;
            size_t per_slot = (size_t)slot_capacity(slot) * bpt * sizeof(block_tq3);
            size_t off      = kv_slot_offset(slot) * bpt;
            for (auto& [layer, kv] : tq_kv_caches) {
                int g = gpu->layer_gpu[layer];
                cudaSetDevice(g);
                cudaMemset(kv.k + off, 0, per_slot);
                cudaMemset(kv.v + off, 0, per_slot);
            }
            if (fp16_dec_cache_on) {
                size_t fc_per_slot = (size_t)slot_capacity(slot) * kv_dim * sizeof(half);
                size_t fc_off      = kv_slot_offset(slot) * kv_dim;
                for (auto& [layer, fc] : fp16_dec_caches) {
                    int g = gpu->layer_gpu[layer];
                    cudaSetDevice(g);
                    cudaMemset(fc.k + fc_off, 0, fc_per_slot);
                    cudaMemset(fc.v + fc_off, 0, fc_per_slot);
                }
            }
            return;
        }
        size_t per_slot = (size_t)slot_capacity(slot) * kv_dim * sizeof(half);
        size_t off_elem = kv_slot_offset(slot) * kv_dim;
        for (auto& [layer, kv] : kv_caches) {
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            cudaMemset(kv.k + off_elem, 0, per_slot);
            cudaMemset(kv.v + off_elem, 0, per_slot);
        }
    }

    // forward_attn: full per-token attention forward.
    //
    // When `external_proj` is true the caller has ALREADY filled
    // ab.q_proj / ab.k_proj / ab.v_proj for this token (e.g. via a batched
    // chunked Q/K/V projection in forward_attn_chunk). We then skip the
    // RMSNorm + Q/K/V GEMVs and pick up at the deinterleave step. The
    // residual still uses `hidden` so the caller passes the matching token's
    // fp32 hidden slice.
    void forward_attn(int layer, float* hidden, int pos, cudaStream_t stream,
                      bool external_proj = false,
                      int slot_pos = -1,
                      int mask_start = -1,
                      int mask_len = 0,
                      uint32_t mask_bits = 0xffffffffu,
                      int slot = 0) {
        int g = gpu->layer_gpu[layer];
        int H = cfg.hidden_size;
        int num_q = cfg.num_q_heads;
        int num_kv = cfg.num_kv_heads;
        int hd = cfg.head_dim;
        int gqa_ratio = num_q / num_kv;
        float eps = cfg.rms_norm_eps;
        float scale = 1.0f / sqrtf((float)hd);
        auto& ab = attn_bufs[g];
        // Continuous batching: KV virtual slot offset. For slot==0 this is 0
        // and behavior matches single-slot exactly (bit-exact).
        int kv_dim_total = num_kv * hd;
        size_t slot_kv_off = kv_slot_offset(slot) * kv_dim_total;

        auto* norm_w = t(blk(layer, "attn_norm.weight"));
        auto* q_w = t(blk(layer, "attn_q.weight"));
        auto* k_w = t(blk(layer, "attn_k.weight"));
        auto* v_w = t(blk(layer, "attn_v.weight"));
        auto* q_norm_w = t(blk(layer, "attn_q_norm.weight"));
        auto* k_norm_w = t(blk(layer, "attn_k_norm.weight"));
        auto* o_w = t(blk(layer, "attn_output.weight"));

        half* norm_out = bufs[g].norm_out;
        int total_qg = num_q * hd;
        int kv_dim = num_kv * hd;
        // Tree mode separates RoPE position (`pos`) from KV cache slot.
        // slot_pos < 0 means use pos (legacy single-node path).
        int kv_slot = (slot_pos < 0) ? pos : slot_pos;
        int seq_len = kv_slot + 1;  // attention range reaches this slot (inclusive)
        int q_out_dim = q_w->dims[1];

        // ATTN_DUMP_LAYER=L + ATTN_DUMP_POS=P → first 8 elements of each
        // attn sub-step output for layer L at position P. Pair with the
        // same envs in forward_attn_chunk to bisect chunked-vs-pertoken
        // drift to a specific sub-step.
        static const int attn_dump_layer = []{ const char* e=getenv("ATTN_DUMP_LAYER"); return e?atoi(e):-1; }();
        static const int attn_dump_pos   = []{ const char* e=getenv("ATTN_DUMP_POS");   return e?atoi(e):-1; }();
        // POS < 0 → dump every call for the matching layer.
        bool attn_do_dump = (attn_dump_layer == layer) && (attn_dump_pos < 0 || attn_dump_pos == pos);
        auto attn_dump_h = [&](const char* tag, const half* buf, int n=256) {
            if (!attn_do_dump) return;
            cudaSetDevice(g); cudaDeviceSynchronize();
            std::vector<half> h(n); cudaMemcpy(h.data(), buf, n*sizeof(half), cudaMemcpyDeviceToHost);
            double sum_abs = 0.0; for (int i=0;i<n;i++) sum_abs += fabs((double)__half2float(h[i]));
            fprintf(stderr, "[ATTN-PT L%d %-14s sa=%.6f]", layer, tag, sum_abs);
            for (int i=0;i<8;i++) fprintf(stderr, " %.5f", __half2float(h[i]));
            fprintf(stderr, "\n"); fflush(stderr);
        };
        auto attn_dump_f = [&](const char* tag, const float* buf, int n=256) {
            if (!attn_do_dump) return;
            cudaSetDevice(g); cudaDeviceSynchronize();
            std::vector<float> h(n); cudaMemcpy(h.data(), buf, n*sizeof(float), cudaMemcpyDeviceToHost);
            double sum_abs = 0.0; for (int i=0;i<n;i++) sum_abs += fabs((double)h[i]);
            fprintf(stderr, "[ATTN-PT L%d %-14s sa=%.6f]", layer, tag, sum_abs);
            for (int i=0;i<8;i++) fprintf(stderr, " %.5f", h[i]);
            fprintf(stderr, "\n"); fflush(stderr);
        };

        // Phase timers — gated by g_profile_attn. Each lambda syncs the
        // current device's stream then accumulates wall ms into the phase
        // global. Adds sync overhead so keep PROFILE_ATTN OFF in production.
        auto pt_now = [](){ return std::chrono::high_resolution_clock::now(); };
        auto pt_sync_ms = [&](std::chrono::high_resolution_clock::time_point tb) {
            cudaSetDevice(g);
            cudaStreamSynchronize(stream);
            auto te = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(te - tb).count();
        };
        auto pt_t0 = pt_now();

        // RAII guard: if we alias ab.q_proj/k_proj/v_proj into the fused
        // output buffer below, restore them on function exit so other code
        // paths (forward_attn_chunk etc.) keep their original buffers.
        struct AttnPtrSaver {
            AttnBuffers& ab;
            half *sq, *sk, *sv;
            bool active = false;
            AttnPtrSaver(AttnBuffers& a) : ab(a), sq(a.q_proj), sk(a.k_proj), sv(a.v_proj) {}
            ~AttnPtrSaver() { if (active) { ab.q_proj = sq; ab.k_proj = sk; ab.v_proj = sv; } }
        };
        AttnPtrSaver ptr_saver(ab);

        if (!external_proj) {
            // 1. RMSNorm (FP32 hidden in)
            if (norm_w->type == GGML_TYPE_F32)
                rms_norm_f32in_f32w(norm_out, hidden, (float*)norm_w->data, 1, H, eps, stream);
            else
                rms_norm_f32in(norm_out, hidden, (half*)norm_w->data, 1, H, eps, stream);
            attn_dump_h("norm_out", norm_out);

            // 2. Q/K/V projections
            gpu_qi[g].quantize(norm_out, H, stream);
            static const bool use_qkv_fuse = getenv("ATTN_QKV_FUSE") != nullptr;
            bool can_fuse_qkv = use_qkv_fuse
                                && q_w->type == GGML_TYPE_Q8_0
                                && k_w->type == GGML_TYPE_Q8_0
                                && v_w->type == GGML_TYPE_Q8_0;
            if (can_fuse_qkv) {
                if ((int)attn_qkv_fused.size() < cfg.num_layers) attn_qkv_fused.resize(cfg.num_layers);
                auto& af = attn_qkv_fused[layer];
                if (!af.weight) {
                    int blocks_per_row = H / 32;
                    size_t bpr_bytes = (size_t)blocks_per_row * sizeof(block_q8_0_aligned);
                    af.q_n = q_w->dims[1];
                    af.k_n = k_w->dims[1];
                    af.v_n = v_w->dims[1];
                    af.total_n = af.q_n + af.k_n + af.v_n;
                    size_t total_bytes = (size_t)af.total_n * bpr_bytes;
                    cudaSetDevice(g);
                    cudaMalloc(&af.weight, total_bytes);
                    size_t off = 0;
                    cudaMemcpyAsync((char*)af.weight + off, q_w->data,
                                    (size_t)af.q_n * bpr_bytes, cudaMemcpyDeviceToDevice, stream);
                    off += (size_t)af.q_n * bpr_bytes;
                    cudaMemcpyAsync((char*)af.weight + off, k_w->data,
                                    (size_t)af.k_n * bpr_bytes, cudaMemcpyDeviceToDevice, stream);
                    off += (size_t)af.k_n * bpr_bytes;
                    cudaMemcpyAsync((char*)af.weight + off, v_w->data,
                                    (size_t)af.v_n * bpr_bytes, cudaMemcpyDeviceToDevice, stream);
                    cudaStreamSynchronize(stream);
                }
                half* fused_out = ab.fused_qkv_out;
                quant_gemv(af.weight, GGML_TYPE_Q8_0, norm_out, fused_out, H, af.total_n, &gpu_qi[g], stream);
                ab.q_proj = fused_out;
                ab.k_proj = fused_out + af.q_n;
                ab.v_proj = fused_out + af.q_n + af.k_n;
                ptr_saver.active = true;
            } else {
                quant_gemv(q_w->data, q_w->type, norm_out, ab.q_proj, H, q_out_dim, &gpu_qi[g], stream);
                quant_gemv(k_w->data, k_w->type, norm_out, ab.k_proj, H, kv_dim, &gpu_qi[g], stream);
                quant_gemv(v_w->data, v_w->type, norm_out, ab.v_proj, H, kv_dim, &gpu_qi[g], stream);
            }
            attn_dump_h("q_proj", ab.q_proj);
            attn_dump_h("k_proj", ab.k_proj);
            attn_dump_h("v_proj", ab.v_proj);
        }
        // else: caller has already populated ab.q_proj / ab.k_proj / ab.v_proj.

        // 3. Deinterleave Q and gate (Qwopus3.6 hybrid) OR pass Q through
        //    unchanged (Qwen3 dense — no Q-gate). Detect via Q projection
        //    output width: 2 * num_q * hd → gated; num_q * hd → ungated.
        half* q_buf;
        half* gate_buf = nullptr;
        const bool has_q_gate = (q_out_dim == 2 * total_qg);
        if (has_q_gate) {
            q_buf = ab.attn_out;
            gate_buf = ab.gate_buf;
            deinterleave_qg_kernel<<<(total_qg+255)/256, 256, 0, stream>>>(
                ab.q_proj, q_buf, gate_buf, num_q, hd);
        } else {
            q_buf = ab.q_proj;  // dense: use Q projection output directly
        }
        attn_dump_h("q_deint", q_buf);

        // 4. Head RMSNorm
        int tn = min(hd, 128);
        // Debug: print weight types once
        static bool printed_norm_types = false;
        if (!printed_norm_types && layer == 3) {
            printf("[DBG L3] q_norm type=%d (F32=0,F16=1), k_norm type=%d, attn_norm type=%d\n",
                   q_norm_w->type, k_norm_w->type, norm_w->type);
            printed_norm_types = true;
        }
        static const bool skip_qk_norm = getenv("SKIP_QK_NORM") != nullptr;
        if (!skip_qk_norm) {
            head_rms_norm_kernel<<<num_q, tn, tn * sizeof(float), stream>>>(
                q_buf, (float*)q_norm_w->data, num_q, hd, eps);
            head_rms_norm_kernel<<<num_kv, tn, tn * sizeof(float), stream>>>(
                ab.k_proj, (float*)k_norm_w->data, num_kv, hd, eps);
        }
        attn_dump_h("q_after_qnorm", q_buf);
        attn_dump_h("k_after_qnorm", ab.k_proj);

        // 5. RoPE — Qwen3-VL multimodal M-RoPE if enabled (vision prompt),
        // standard 1D RoPE otherwise. The single-token forward_attn is used
        // here both for legacy prefill (run_qwen path) and for gen step,
        // so we gate on `pos < g_mrope_len` so generated tokens fall back
        // safely to the standard kernel.
        int rope_dim = rope.rope_dim;
        int half_rope = rope_dim / 2;
        extern int* g_mrope_pos_t[4];
        extern int* g_mrope_pos_h[4];
        extern int* g_mrope_pos_w[4];
        extern int g_mrope_sec_t, g_mrope_sec_h, g_mrope_sec_w, g_mrope_len;
        if (g_mrope_pos_t[g] && pos < g_mrope_len) {
            const int* pt = g_mrope_pos_t[g] + pos;
            const int* ph = g_mrope_pos_h[g] + pos;
            const int* pw = g_mrope_pos_w[g] + pos;
            dim3 rope_q_grid((num_q  * half_rope + 255) / 256, 1);
            dim3 rope_k_grid((num_kv * half_rope + 255) / 256, 1);
            apply_rope_kernel_mrope_chunk<<<rope_q_grid, 256, 0, stream>>>(
                q_buf, rope.sin_table(g), rope.cos_table(g),
                pt, ph, pw, g_mrope_sec_t, g_mrope_sec_h, g_mrope_sec_w,
                num_q,  hd, rope_dim, 1);
            apply_rope_kernel_mrope_chunk<<<rope_k_grid, 256, 0, stream>>>(
                ab.k_proj, rope.sin_table(g), rope.cos_table(g),
                pt, ph, pw, g_mrope_sec_t, g_mrope_sec_h, g_mrope_sec_w,
                num_kv, hd, rope_dim, 1);
        } else {
            float* sin_pos = rope.sin_table(g) + pos * half_rope;
            float* cos_pos = rope.cos_table(g) + pos * half_rope;
            apply_rope_kernel<<<(num_q * half_rope + 255)/256, 256, 0, stream>>>(
                q_buf, sin_pos, cos_pos, num_q, hd, rope_dim);
            apply_rope_kernel<<<(num_kv * half_rope + 255)/256, 256, 0, stream>>>(
                ab.k_proj, sin_pos, cos_pos, num_kv, hd, rope_dim);
        }
        attn_dump_h("q_after_rope", q_buf);
        attn_dump_h("k_after_rope", ab.k_proj);

        if (g_profile_attn) { g_pt_qkvr_ms += pt_sync_ms(pt_t0); pt_t0 = pt_now(); }

        // 6. Store K, V at slot_offset + kv_slot (may differ from RoPE pos in tree mode)
        if (use_q8_kv) {
            auto& q8 = q8_kv_caches[layer];
            int bpt = q8.blocks_per_token;
            size_t slot_off_blocks = kv_slot_offset(slot) * bpt;
            size_t off = slot_off_blocks + (size_t)kv_slot * bpt;
            quantize_kv_q8_0_kern<<<bpt, 32, 0, stream>>>(ab.k_proj, q8.k + off, bpt);
            quantize_kv_q8_0_kern<<<bpt, 32, 0, stream>>>(ab.v_proj, q8.v + off, bpt);
        } else if (use_turboquant) {
            auto& tq = tq_kv_caches[layer];
            int bpt = tq.blocks_per_token;
            size_t slot_off_blocks = kv_slot_offset(slot) * bpt;
            size_t off = slot_off_blocks + (size_t)kv_slot * bpt;
            tq3_quantize_kernel<<<(bpt+31)/32, 32, 0, stream>>>(
                ab.k_proj, &tq.k[off], bpt);
            tq3_quantize_kernel<<<(bpt+31)/32, 32, 0, stream>>>(
                ab.v_proj, &tq.v[off], bpt);
            if (fp16_dec_cache_on) {
                auto& fc = fp16_dec_caches[layer];
                half* fp16_k_pos = fc.k + slot_kv_off + (size_t)kv_slot * kv_dim;
                half* fp16_v_pos = fc.v + slot_kv_off + (size_t)kv_slot * kv_dim;
                cudaMemcpyAsync(fp16_k_pos, ab.k_proj, kv_dim * sizeof(half),
                                cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(fp16_v_pos, ab.v_proj, kv_dim * sizeof(half),
                                cudaMemcpyDeviceToDevice, stream);
            }
        } else {
            auto& kv = kv_caches[layer];
            half* k_cache_pos = kv.k + slot_kv_off + (size_t)kv_slot * kv_dim;
            half* v_cache_pos = kv.v + slot_kv_off + (size_t)kv_slot * kv_dim;
            cudaMemcpyAsync(k_cache_pos, ab.k_proj, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(v_cache_pos, ab.v_proj, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        }

        if (g_profile_attn) { g_pt_kvwrite_ms += pt_sync_ms(pt_t0); pt_t0 = pt_now(); }

        // 7. Attention: Q[num_q, hd] @ K[seq_len, num_kv, hd]^T → softmax → @ V
        //
        // Fast path (FLASH_ATTN=1, fp16 KV, no tree mask, qwen3-hybrid shape):
        //   single fused per-token FA kernel — chunked FA invoked with sub_n=1.
        //   Avoids the score buffer round-trip through HBM and reads K/V once
        //   per tile instead of twice (score + value). Critical for long-ctx
        //   gen where attention dominates wall time (10K+: ~58 % of step).
        //
        // Fallback (TQ path, tree mode, FA off, exotic shape):
        //   the legacy score / softmax / value split. TQ_FUSED=1 fuses score
        //   and value at the kernel level but still runs them as two passes.
        // Per-token fused FA. Long-context only (≥4 K) because at short ctx
        // the (num_kv,1) base grid launches only 4 blocks (vs sequential's
        // (num_q, seq_len) thousands) and kernel-launch overhead dominates.
        // Supports both fp16 KV and TQ KV: TQ path bulk-dequants the active
        // range once into tq_k_buf/tq_v_buf, then runs the same fused kernel
        // — bulk dequant cost is paid anyway in the legacy TQ path, and we
        // gain by collapsing score/softmax/value into one pass + split-K SM
        // coverage.
        bool can_per_tok_fa = g_use_flash_attn
                           && hd == 256 && num_kv == 4
                           && (num_q == 24 || num_q == 16)
                           && mask_start < 0
                           && (kv_slot + 1) >= 4096;
        if (can_per_tok_fa) {
            constexpr int HD = 256, BM = 32, BLOCK = 256;
            // Split-K scratch is normally lazy-alloc'd in forward_attn_chunk.
            // If chunked prefill never ran (prompt_len == 1), allocate here.
            if (!ab.attn_split_o) {
                constexpr int K_SPLITS_MAX = 16;
                cudaMalloc(&ab.attn_split_m, (size_t)num_q * ATTN_NB * K_SPLITS_MAX * sizeof(float));
                cudaMalloc(&ab.attn_split_l, (size_t)num_q * ATTN_NB * K_SPLITS_MAX * sizeof(float));
                cudaMalloc(&ab.attn_split_o, (size_t)num_q * ATTN_NB * K_SPLITS_MAX * hd * sizeof(float));
            }
            // At per-token gen the base FA grid is (num_kv=4, 1) = 4 blocks,
            // which leaves the 80-SM CMP cards almost entirely idle. Split-K
            // factors the K/V tile loop into K independent blocks per kv_head,
            // merged with log-sum-exp. Tested values:
            //   K=4 : 16 blocks  (chunked-path default)  — TBM
            //   K=8 : 32 blocks  (current sweet spot, +17 % at 18 K vs none)
            //   K=16: 64 blocks  (slower: more merge + MTP accept-rate drop
            //                     from extra fp32 reduction noise)
            // PREFILL_K_SPLITS_PT=N override (4 / 8 / 16).
            //
            // TQ path uses the cooperative-decode variant
            // `flash_attn_chunk_fused_split_tq3`: each warp decodes one
            // 128-elem TQ block in shared memory (4 elems/thread, in-warp
            // shuffle WHT) so we never materialise the full fp16 history.
            // fp16 path uses the regular kernel.
            constexpr int K_SPLITS_PT = 8;
            int seq_len_local = kv_slot + 1;
            int active_end_max = seq_len_local;  // sub_n=1, abs_pos=kv_slot
            int smem_bytes_24 = 6 * HD * sizeof(half)
                              + 2 * BM * HD * sizeof(half)
                              + 6 * BM * sizeof(float);
            int smem_bytes_16 = 4 * HD * sizeof(half)
                              + 2 * BM * HD * sizeof(half)
                              + 4 * BM * sizeof(float);
            // TQ path uses the cooperative-decode kernel
            // (`flash_attn_chunk_fused_split_tq3`) — each warp decodes one
            // 128-elem TQ block in shared memory (4 elems/thread, in-warp
            // shuffle WHT) so we never materialise the full fp16 history.
            // Output is bit-identical to the legacy bulk-dequant-then-fp16-
            // attention path (sequential), but reads 5x less HBM per token.
            // Hybrid (gen-start bulk dequant + per-step incremental into a
            // fp16 scratch + base fp16 fused FA) was tried and measured
            // slower at long ctx: the small bulk-dequant win was eaten by
            // an MTP accept-rate drop (the round-trip endpoint differs from
            // the chunked-prefill fp16-directly path the model attended over
            // during prefill, so the draft head's predictions stop matching
            // as well). Cooperative wins overall.
            if (use_q8_kv) {
                auto& q8 = q8_kv_caches[layer];
                size_t slot_off_blocks_int = kv_slot_offset(slot) * (size_t)q8.blocks_per_token;
                block_q8_0_aligned* k_q8_slot = q8.k + slot_off_blocks_int;
                block_q8_0_aligned* v_q8_slot = q8.v + slot_off_blocks_int;
                if (num_q == 24) {
                    constexpr int GQA = 6;
                    dim3 fg(num_kv, 1, K_SPLITS_PT);
                    flash_attn_chunk_fused_split_q8<HD, GQA, BM, BLOCK, K_SPLITS_PT>
                        <<<fg, BLOCK, smem_bytes_24, stream>>>(
                            q_buf, k_q8_slot, v_q8_slot,
                            ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                            num_q, num_kv, kv_slot, /*sub_n=*/1, ATTN_NB,
                            active_end_max, scale);
                    dim3 mg(num_kv, 1);
                    flash_attn_split_merge<HD, GQA, BLOCK, K_SPLITS_PT>
                        <<<mg, BLOCK, 0, stream>>>(
                            ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                            q_buf, num_q, /*sub_n=*/1, ATTN_NB);
                } else {
                    constexpr int GQA = 4;
                    dim3 fg(num_kv, 1, K_SPLITS_PT);
                    flash_attn_chunk_fused_split_q8<HD, GQA, BM, BLOCK, K_SPLITS_PT>
                        <<<fg, BLOCK, smem_bytes_16, stream>>>(
                            q_buf, k_q8_slot, v_q8_slot,
                            ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                            num_q, num_kv, kv_slot, /*sub_n=*/1, ATTN_NB,
                            active_end_max, scale);
                    dim3 mg(num_kv, 1);
                    flash_attn_split_merge<HD, GQA, BLOCK, K_SPLITS_PT>
                        <<<mg, BLOCK, 0, stream>>>(
                            ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                            q_buf, num_q, /*sub_n=*/1, ATTN_NB);
                }
            } else if (use_turboquant && !fp16_dec_cache_on) {
                auto& tq = tq_kv_caches[layer];
                size_t slot_off_blocks_int = kv_slot_offset(slot) * (size_t)tq.blocks_per_token;
                block_tq3* tq_k_slot = tq.k + slot_off_blocks_int;
                block_tq3* tq_v_slot = tq.v + slot_off_blocks_int;
                if (num_q == 24) {
                    constexpr int GQA = 6;
                    dim3 fg(num_kv, 1, K_SPLITS_PT);
                    flash_attn_chunk_fused_split_tq3<HD, GQA, BM, BLOCK, K_SPLITS_PT>
                        <<<fg, BLOCK, smem_bytes_24, stream>>>(
                            q_buf, tq_k_slot, tq_v_slot,
                            ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                            num_q, num_kv, kv_slot, /*sub_n=*/1, ATTN_NB,
                            active_end_max, scale);
                    dim3 mg(num_kv, 1);
                    flash_attn_split_merge<HD, GQA, BLOCK, K_SPLITS_PT>
                        <<<mg, BLOCK, 0, stream>>>(
                            ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                            q_buf, num_q, /*sub_n=*/1, ATTN_NB);
                } else {
                    constexpr int GQA = 4;
                    dim3 fg(num_kv, 1, K_SPLITS_PT);
                    flash_attn_chunk_fused_split_tq3<HD, GQA, BM, BLOCK, K_SPLITS_PT>
                        <<<fg, BLOCK, smem_bytes_16, stream>>>(
                            q_buf, tq_k_slot, tq_v_slot,
                            ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                            num_q, num_kv, kv_slot, /*sub_n=*/1, ATTN_NB,
                            active_end_max, scale);
                    dim3 mg(num_kv, 1);
                    flash_attn_split_merge<HD, GQA, BLOCK, K_SPLITS_PT>
                        <<<mg, BLOCK, 0, stream>>>(
                            ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                            q_buf, num_q, /*sub_n=*/1, ATTN_NB);
                }
            } else {
                // fp16 cache path: use either the legacy kv_caches (use_turboquant=false)
                // or the fp16 decode mirror (use_turboquant && fp16_dec_cache_on),
                // both populated up to kv_slot in the KV-write step above.
                half* k_src;
                half* v_src;
                if (use_turboquant) {
                    auto& fc = fp16_dec_caches[layer];
                    k_src = fc.k + slot_kv_off;
                    v_src = fc.v + slot_kv_off;
                } else {
                    auto& kv = kv_caches[layer];
                    k_src = kv.k + slot_kv_off;
                    v_src = kv.v + slot_kv_off;
                }
                if (num_q == 24) {
                    constexpr int GQA = 6;
                    dim3 fg(num_kv, 1, K_SPLITS_PT);
                    flash_attn_chunk_fused_split<HD, GQA, BM, BLOCK, K_SPLITS_PT>
                        <<<fg, BLOCK, smem_bytes_24, stream>>>(
                            q_buf, k_src, v_src,
                            ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                            num_q, num_kv, kv_slot, /*sub_n=*/1, ATTN_NB,
                            active_end_max, scale);
                    dim3 mg(num_kv, 1);
                    flash_attn_split_merge<HD, GQA, BLOCK, K_SPLITS_PT>
                        <<<mg, BLOCK, 0, stream>>>(
                            ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                            q_buf, num_q, /*sub_n=*/1, ATTN_NB);
                } else {
                    constexpr int GQA = 4;
                    dim3 fg(num_kv, 1, K_SPLITS_PT);
                    flash_attn_chunk_fused_split<HD, GQA, BM, BLOCK, K_SPLITS_PT>
                        <<<fg, BLOCK, smem_bytes_16, stream>>>(
                            q_buf, k_src, v_src,
                            ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                            num_q, num_kv, kv_slot, /*sub_n=*/1, ATTN_NB,
                            active_end_max, scale);
                    dim3 mg(num_kv, 1);
                    flash_attn_split_merge<HD, GQA, BLOCK, K_SPLITS_PT>
                        <<<mg, BLOCK, 0, stream>>>(
                            ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                            q_buf, num_q, /*sub_n=*/1, ATTN_NB);
                }
            }
            attn_dump_h("fa_out", q_buf);
        } else {
            dim3 score_grid = score_pos_grid(num_q, seq_len);
            static const bool tq_fused = getenv("TQ_FUSED") != nullptr;
            // Q8 path: bulk-dequant K/V into the per-GPU fp16 scratch and
            // run the standard attn_score_kernel_h / attn_value_kernel_h.
            // (kv_slot < 4096, so the long-context fused-split path the gen
            // loop uses is gated off — we still need Q8 to feed something
            // here.) tq_k_buf / tq_v_buf are shared with the chunked-prefill
            // dequant scratch, so this overwrites whatever was there but
            // the chunk loop has already finished by the time we land here.
            half* k_src_q8 = nullptr;
            half* v_src_q8 = nullptr;
            if (use_q8_kv) {
                auto& q8 = q8_kv_caches[layer];
                int bpt = q8.blocks_per_token;
                int n_blocks_total = seq_len * bpt;
                size_t slot_off_blocks = kv_slot_offset(slot) * bpt;
                dim3 dq_grid((n_blocks_total + 7) / 8);
                dim3 dq_block(32, 8);
                dequantize_kv_q8_0_kern<<<dq_grid, dq_block, 0, stream>>>(
                    q8.k + slot_off_blocks, tq_k_buf[g] + slot_kv_off, n_blocks_total);
                dequantize_kv_q8_0_kern<<<dq_grid, dq_block, 0, stream>>>(
                    q8.v + slot_off_blocks, tq_v_buf[g] + slot_kv_off, n_blocks_total);
                k_src_q8 = tq_k_buf[g] + slot_kv_off;
                v_src_q8 = tq_v_buf[g] + slot_kv_off;
            }
            if (use_q8_kv) {
                attn_score_kernel_h<<<score_grid, min(hd, 256), 0, stream>>>(
                    q_buf, k_src_q8, ab.attn_scores,
                    num_q, num_kv, hd, seq_len, scale);
            } else if (use_turboquant) {
                auto& tq = tq_kv_caches[layer];
                size_t slot_off_blocks_int = kv_slot_offset(slot) * (size_t)tq.blocks_per_token;
                block_tq3* tq_k_slot = tq.k + slot_off_blocks_int;
                if (tq_fused) {
                    dim3 tq_score_grid = score_pos_grid(num_kv, seq_len);
                    int smem_bytes = hd * sizeof(float);
                    attn_score_kernel_tq3<<<tq_score_grid, hd, smem_bytes, stream>>>(
                        q_buf, tq_k_slot, ab.attn_scores,
                        num_q, num_kv, hd, seq_len, scale);
                } else {
                    int n_blocks_total = seq_len * tq.blocks_per_token;
                    tq3_dequantize_kernel<<<(n_blocks_total + 31)/32, 32, 0, stream>>>(
                        tq_k_slot, tq_k_buf[g], n_blocks_total);
                    attn_score_kernel_h<<<score_grid, min(hd, 256), 0, stream>>>(
                        q_buf, tq_k_buf[g], ab.attn_scores,
                        num_q, num_kv, hd, seq_len, scale);
                }
            } else {
                auto& kv = kv_caches[layer];
                half* k_slot = kv.k + slot_kv_off;
                if (mask_start >= 0) {
                    attn_score_kernel_h_tree_masked<<<score_grid, min(hd, 256), 0, stream>>>(
                        q_buf, k_slot, ab.attn_scores,
                        num_q, num_kv, hd, seq_len, scale,
                        mask_start, mask_len, mask_bits);
                } else {
                    attn_score_kernel_h<<<score_grid, min(hd, 256), 0, stream>>>(
                        q_buf, k_slot, ab.attn_scores,
                        num_q, num_kv, hd, seq_len, scale);
                }
            }
            attn_dump_f("score", ab.attn_scores, seq_len);

            { int st = 1; while(st < seq_len && st < 256) st <<= 1;
            softmax_kernel<<<num_q, st, st * sizeof(float), stream>>>(
                ab.attn_scores, num_q, seq_len); }
            attn_dump_f("softmax", ab.attn_scores, seq_len);

            if (use_q8_kv) {
                attn_value_kernel_h<<<num_q, min(hd, 256), 0, stream>>>(
                    ab.attn_scores, v_src_q8, q_buf,
                    num_q, num_kv, hd, seq_len);
            } else if (use_turboquant) {
                auto& tq = tq_kv_caches[layer];
                size_t slot_off_blocks_int = kv_slot_offset(slot) * (size_t)tq.blocks_per_token;
                block_tq3* tq_v_slot = tq.v + slot_off_blocks_int;
                if (tq_fused) {
                    int blocks_per_kv_head = hd / TQ_BLOCK_SIZE;
                    dim3 tq_value_grid(num_kv, blocks_per_kv_head);
                    int smem_bytes = TQ_BLOCK_SIZE * sizeof(float);
                    attn_value_kernel_tq3<<<tq_value_grid, TQ_BLOCK_SIZE, smem_bytes, stream>>>(
                        ab.attn_scores, tq_v_slot, q_buf,
                        num_q, num_kv, hd, seq_len);
                } else {
                    int n_blocks_total = seq_len * tq.blocks_per_token;
                    tq3_dequantize_kernel<<<(n_blocks_total + 31)/32, 32, 0, stream>>>(
                        tq_v_slot, tq_v_buf[g], n_blocks_total);
                    attn_value_kernel_h<<<num_q, min(hd, 256), 0, stream>>>(
                        ab.attn_scores, tq_v_buf[g], q_buf,
                        num_q, num_kv, hd, seq_len);
                }
            } else {
                auto& kv = kv_caches[layer];
                half* v_slot = kv.v + slot_kv_off;
                attn_value_kernel_h<<<num_q, min(hd, 256), 0, stream>>>(
                    ab.attn_scores, v_slot, q_buf,
                    num_q, num_kv, hd, seq_len);
            }
            attn_dump_h("value_out", q_buf);
        }

        if (g_profile_attn) { g_pt_attn_ms += pt_sync_ms(pt_t0); pt_t0 = pt_now(); }

        // 8. Output gate (only for Qwopus hybrid): out *= sigmoid(gate)
        if (gate_buf) {
            apply_gate_sigmoid<<<(total_qg+255)/256, 256, 0, stream>>>(
                q_buf, gate_buf, total_qg);
        }
        attn_dump_h("after_gate", q_buf);

        // 9. Output projection
        half* proj_out = bufs[g].mlp_down;
        gpu_qi_inter[g].quantize(q_buf, num_q * hd, stream);
        quant_gemv(o_w->data, o_w->type, q_buf, proj_out, num_q * hd, H, &gpu_qi_inter[g], stream);
        attn_dump_h("o_proj", proj_out);

        // 10. Residual into FP32 hidden
        add_kernel_f32<<<(H+255)/256, 256, 0, stream>>>(hidden, proj_out, H);

        if (g_profile_attn) { g_pt_oproj_ms += pt_sync_ms(pt_t0); g_pt_calls++; }
    }

    // ============ GDN Forward ============
    // conv1d state: [qkv_dim, kernel_width] per layer
    // recurrent state: [num_v_heads, k_head_dim, v_head_dim] per layer
    
    struct GDNState {
        // Per-slot flat arrays. Slot s lives at:
        //   conv_state + s * (qkv_dim * 4)
        //   rec_state  + s * (num_v_heads * k_dim * v_dim)
        // Allocation count = num_slots. With num_slots=1 the layout reduces to
        // the original single-state layout, so single-request behavior is
        // bit-exact preserved.
        float* conv_state;    // [num_slots, qkv_dim, 4]
        float* rec_state;     // [num_slots, num_v_heads, k_dim, v_dim]
    };
    std::vector<GDNState> gdn_states;
    int gdn_qkv_dim_cached = 0;     // qkv_dim (set in init_gdn_states)
    int gdn_rec_per_slot = 0;       // num_v * k_dim * v_dim

    inline float* gdn_conv_slot(int layer, int slot) {
        return gdn_states[layer].conv_state + (size_t)slot * gdn_qkv_dim_cached * 4;
    }
    inline float* gdn_rec_slot(int layer, int slot) {
        return gdn_states[layer].rec_state + (size_t)slot * gdn_rec_per_slot;
    }

    int gdn_qkv_dim() { return 0; } // computed from tensor

    // num_slots_arg defaults to 1 (legacy single-slot). When >1 the GDN
    // recurrent state is allocated per-slot. This must be called BEFORE
    // any forward; it also seeds the model's num_slots member, since
    // init_kv_cache is normally called after this and would re-set it.
    void init_gdn_states(int num_slots_arg = 1) {
        if (num_slots_arg > 0) num_slots = num_slots_arg;

        // Detect layer types from tensor presence (v1 hardcoded vs v2 data-driven)
        memset(layer_is_attn, 0, sizeof(layer_is_attn));
        for (int layer = 0; layer < cfg.num_layers && layer < 256; layer++) {
            std::string prefix = "blk." + std::to_string(layer) + ".";
            bool has_ssm = (t((prefix + "ssm_alpha.weight").c_str()) != nullptr);
            layer_is_attn[layer] = !has_ssm;
        }
        int attn_count_init = 0;
        for (int l = 0; l < cfg.num_layers; l++) if (layer_is_attn[l]) attn_count_init++;
        printf("Layer types: %d attn, %d GDN (detected from tensors)\n",
               attn_count_init, cfg.num_layers - attn_count_init);

        // ── MoE detection (Qwen3.x-A3B) ──────────────────────────────────────
        // A layer is MoE iff it carries a router (ffn_gate_inp). Per-layer so a
        // hybrid with leading dense blocks is handled automatically. Dims come
        // from the stacked expert tensors; metadata keys are a fallback.
        memset(layer_is_moe, 0, sizeof(layer_is_moe));
        int moe_count = 0, first_moe = -1;
        for (int layer = 0; layer < cfg.num_layers && layer < 256; layer++) {
            if (t(blk(layer, "ffn_gate_inp.weight"))) {
                layer_is_moe[layer] = true;
                if (first_moe < 0) first_moe = layer;
                moe_count++;
            }
        }
        if (first_moe >= 0) {
            cfg.is_moe = true;
            auto* router = t(blk(first_moe, "ffn_gate_inp.weight"));
            cfg.num_experts = router->dims[1];  // [hidden, num_experts]
            std::string arch = gguf_arch;
            cfg.num_experts_per_tok = gguf_p
                ? (int)gguf_p->get_u32(arch + ".expert_used_count",
                      gguf_p->get_u32(arch + ".expert_count_used", 8))
                : 8;
            // per-expert intermediate from the stacked gate tensor (separate
            // gate/up). Fall back to fused gate_up (dims/2) then metadata.
            auto* gate_exps = t(blk(first_moe, "ffn_gate_exps.weight"));
            if (gate_exps && gate_exps->n_dims >= 3) {
                // 3D stacked experts [n_embd, n_ff_exp, n_expert] (llama.cpp
                // qwen3moe). dims[1] IS the per-expert intermediate; dims[2] is
                // the authoritative expert count.
                cfg.moe_intermediate_size = gate_exps->dims[1];
                cfg.num_experts = (int)gate_exps->dims[2];
            } else if (gate_exps) {
                // 2D flattened [n_expert*n_ff_exp, n_embd] (Gemma-style).
                cfg.moe_intermediate_size = gate_exps->dims[1] / cfg.num_experts;
            } else if (auto* gu = t(blk(first_moe, "ffn_gate_up_exps.weight"))) {
                cfg.moe_intermediate_size = (gu->dims[1] / cfg.num_experts) / 2;
            } else if (gguf_p) {
                cfg.moe_intermediate_size =
                    gguf_p->get_u32(arch + ".expert_feed_forward_length", 0);
            }
            // Shared expert (qwen3_5_moe brings it back; Qwen3-MoE had none).
            if (auto* sh = t(blk(first_moe, "ffn_gate_shexp.weight")))
                cfg.shared_expert_intermediate_size = sh->dims[1];
            printf("MoE: %d/%d layers, %d experts, top-%d, expert_inter=%d, shared_inter=%d\n",
                   moe_count, cfg.num_layers, cfg.num_experts,
                   cfg.num_experts_per_tok, cfg.moe_intermediate_size,
                   cfg.shared_expert_intermediate_size);
        }

        // Detect v2 separate Q/K/V attention (vs v1 combined attn_qkv)
        {
            int first_attn = -1;
            for (int l = 0; l < cfg.num_layers; l++)
                if (layer_is_attn[l]) { first_attn = l; break; }
            if (first_attn >= 0) {
                std::string pfx = "blk." + std::to_string(first_attn) + ".";
                auto* q_w = t((pfx + "attn_q.weight").c_str());
                auto* combined = t((pfx + "attn_qkv.weight").c_str());
                v2_separate_qkv = (q_w != nullptr && combined == nullptr);
                if (v2_separate_qkv) {
                    auto* k_w = t((pfx + "attn_k.weight").c_str());
                    attn_head_dim = cfg.head_dim;
                    attn_num_q_heads = q_w->dims[1] / attn_head_dim;
                    attn_num_kv_heads = k_w ? (k_w->dims[1] / attn_head_dim) : cfg.num_kv_heads;
                    printf("Attention (v2 separate QKV): %d Q heads, %d KV heads, head_dim=%d\n",
                           attn_num_q_heads, attn_num_kv_heads, attn_head_dim);
                }
            }
        }

        // Get dimensions from first GDN layer
        auto* qkv = t("blk.0.attn_qkv.weight");
        auto* ssm_out = t("blk.0.ssm_out.weight");
        if (!qkv || !ssm_out) { printf("Missing GDN tensors!\n"); return; }

        int qkv_dim = qkv->dims[1];  // output dim of QKV
        int v_total = ssm_out->dims[0]; // input dim of out_proj = num_v * v_dim

        // Get GDN config from tensor shapes
        auto* a_t = t("blk.0.ssm_alpha.weight");
        int num_v = a_t ? a_t->dims[1] : 48;
        int v_dim = v_total / num_v;
        auto* ssm_a_t = t("blk.0.ssm_a");
        int k_dim = 128;
        int num_k = (qkv_dim - num_v * v_dim) / (2 * k_dim);

        printf("GDN config: qkv_dim=%d, num_k=%d, num_v=%d, k_dim=%d, v_dim=%d\n",
            qkv_dim, num_k, num_v, k_dim, v_dim);

        cfg.linear_k_heads = num_k;
        cfg.linear_v_heads = num_v;
        cfg.linear_k_dim = k_dim;
        cfg.linear_v_dim = v_dim;

        gdn_qkv_dim_cached = qkv_dim;
        gdn_rec_per_slot   = num_v * k_dim * v_dim;
        size_t conv_bytes = (size_t)num_slots * qkv_dim * 4 * sizeof(float);
        size_t rec_bytes  = (size_t)num_slots * gdn_rec_per_slot * sizeof(float);

        gdn_states.resize(cfg.num_layers);
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (layer_is_attn[layer]) {
                gdn_states[layer].conv_state = nullptr;
                gdn_states[layer].rec_state = nullptr;
                continue;
            }
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            cudaMalloc(&gdn_states[layer].conv_state, conv_bytes);
            cudaMemset(gdn_states[layer].conv_state, 0, conv_bytes);
            cudaMalloc(&gdn_states[layer].rec_state, rec_bytes);
            cudaMemset(gdn_states[layer].rec_state, 0, rec_bytes);
        }
        printf("GDN states allocated for %d layers × %d slots (conv %.1f MB, rec %.1f MB total)\n",
               cfg.num_layers, num_slots,
               cfg.num_layers * (double)conv_bytes / 1e6,
               cfg.num_layers * (double)rec_bytes  / 1e6);
    }
    
    void reset_all_states() {
        reset_kv_cache();
        reset_gdn_states_inner();
        if (!tq_decoded_until.empty())
            std::fill(tq_decoded_until.begin(), tq_decoded_until.end(), -1);
    }

    // STAGE 0 (embed/rerank latency): reset everything EXCEPT the ~1.15 GB KV
    // cache memset. Safe ONLY for the fresh-KV-per-request fp16 path (embed/
    // rerank sidecars, no Q8/TQ KV): a new request writes fresh K/V over
    // [0, plen) and attends strictly causally within [0, pos<=plen), while the
    // FA kernel masks query rows >= active_end to -inf and zero-fills out-of-
    // range K/V tiles — so stale KV at pos >= plen (from a longer previous doc)
    // is NEVER read. Skipping the per-doc full-cache memset removes a length-
    // independent ~1.15 GB (36 layers × kv_total_seq × kv_dim × 2B × 2) tax paid
    // on EVERY forward. NOT safe for Q8/TQ KV (dequant may touch wider ranges),
    // so callers must only use this on the fp16 dense sidecar path.
    void reset_states_no_kv_memset() {
        reset_gdn_states_inner();
        if (!tq_decoded_until.empty())
            std::fill(tq_decoded_until.begin(), tq_decoded_until.end(), -1);
    }

    // ============ Prefix cache (per-slot) ============
    // Each slot owns a snapshot of (KV [0, N) + GDN recurrent state) taken at
    // a chunk-aligned position. On the next request that hits the same first
    // N tokens, the engine restores the snapshot into that slot's KV/GDN
    // state and skips re-prefilling those positions.
    //
    // Going per-slot lets concurrent clients with different system prompts
    // each cache their own prefix — which is the actual win for the bot
    // workload (each request shares a long fixed prefix with only the tail
    // varying). Prior single-slot design invalidated the cache whenever a
    // request with a different prefix landed.
    struct PrefixSnapshot {
        std::vector<int> tokens;
        int  n_pos = 0;
        bool valid = false;
        // Pinned host RAM (no GPU affinity for storage). Lives off-VRAM so
        // multiple long-prefix snapshots can coexist without eating HBM. On
        // hit we DMA-copy host->device into the layer's GPU KV/GDN buffers.
        // [layer] -> (k_copy[N*kv_dim], v_copy[N*kv_dim]). fp16 KV path.
        std::unordered_map<int, std::pair<half*, half*>> kv_copy;
        // [layer] -> raw block_tq3 bytes (k,v) for the MTP_TQ path, and raw
        // block_q8_0_aligned bytes for the Q8KV path. Only one of the three
        // KV maps is populated, matching the live KV format.
        std::unordered_map<int, std::pair<char*, char*>> tqkv_copy;
        std::unordered_map<int, std::pair<char*, char*>> q8kv_copy;
        // [layer] -> (conv_state_copy, rec_state_copy)
        std::unordered_map<int, std::pair<float*, float*>> gdn_copy;
        int kv_capacity = 0;  // currently allocated KV slice length
    };
    std::vector<PrefixSnapshot> prefix_caches;   // size = num_slots
    std::mutex                  prefix_caches_mu; // protects access from worker threads
    void ensure_prefix_caches() {
        if ((int)prefix_caches.size() != num_slots) prefix_caches.resize(num_slots);
    }

    void free_prefix_cache_buffers(int slot) {
        ensure_prefix_caches();
        if (slot < 0 || slot >= num_slots) return;
        PrefixSnapshot& pc = prefix_caches[slot];
        for (auto& [layer, kv] : pc.kv_copy) {
            cudaFreeHost(kv.first); cudaFreeHost(kv.second);
        }
        for (auto& [layer, kv] : pc.tqkv_copy) {
            cudaFreeHost(kv.first); cudaFreeHost(kv.second);
        }
        for (auto& [layer, kv] : pc.q8kv_copy) {
            cudaFreeHost(kv.first); cudaFreeHost(kv.second);
        }
        for (auto& [layer, gd] : pc.gdn_copy) {
            cudaFreeHost(gd.first); cudaFreeHost(gd.second);
        }
        pc.kv_copy.clear();
        pc.tqkv_copy.clear();
        pc.q8kv_copy.clear();
        pc.gdn_copy.clear();
        pc.kv_capacity = 0;
        pc.valid = false;
        pc.n_pos = 0;
        pc.tokens.clear();
    }

    // Returns N restored if hit (caller skips first N tokens of prefill),
    // or 0 on miss. The slot's state is fully reset on hit (not via the
    // global reset), so this method must run BEFORE the caller's reset.
    int try_restore_prefix_cache(const std::vector<int>& prompt_tokens,
                                 int requested_cached, int slot = 0) {
        ensure_prefix_caches();
        if (slot < 0 || slot >= num_slots) return 0;
        std::lock_guard<std::mutex> lk(prefix_caches_mu);
        PrefixSnapshot& pc = prefix_caches[slot];
        if (requested_cached <= 0 || !pc.valid) return 0;
        int N = (requested_cached / CHUNK_SIZE) * CHUNK_SIZE;
        if (N <= 0 || N != pc.n_pos) return 0;
        if (N > (int)prompt_tokens.size()) return 0;
        for (int i = 0; i < N; i++) {
            if (prompt_tokens[i] != pc.tokens[i]) return 0;
        }
        restore_snapshot_into_slot(pc, slot, N);
        return N;
    }

    // Automatic prefix reuse (no client-specified length). Reuses whatever
    // was snapshotted on this slot last turn, as long as those pc.n_pos tokens
    // are still a bit-exact prefix of the current prompt and leave at least
    // one token to decode. This is what makes prefix caching transparent: the
    // append-only chat history means turn N's snapshot is a valid prefix of
    // turn N+1, so the persona + prior turns are restored instead of
    // reprefilled. Caveat: any volatile content (current time, per-query RAG
    // hits) placed BEFORE this boundary breaks the bit-exact match — keep it
    // at the tail of the prompt. Returns tokens restored (pc.n_pos) or 0.
    int try_restore_prefix_cache_auto(const std::vector<int>& prompt_tokens, int slot = 0) {
        ensure_prefix_caches();
        if (slot < 0 || slot >= num_slots) return 0;
        std::lock_guard<std::mutex> lk(prefix_caches_mu);
        // Search every slot's snapshot, not just this one: a single session
        // can bounce between physical slots across turns (the scheduler picks
        // a free slot), so the matching snapshot may live elsewhere. Restoring
        // another slot's snapshot into this slot is correct — KV/GDN state at
        // position i depends only on tokens[0..i], not on which slot held it.
        int best = -1, best_N = 0;
        for (int s = 0; s < num_slots; s++) {
            PrefixSnapshot& pc = prefix_caches[s];
            if (!pc.valid) continue;
            int N = pc.n_pos;  // already chunk-aligned at save time
            if (N <= best_N || N > (int)prompt_tokens.size() - 1) continue;
            bool match = true;
            for (int i = 0; i < N; i++) {
                if (prompt_tokens[i] != pc.tokens[i]) { match = false; break; }
            }
            if (match) { best = s; best_N = N; }
        }
        if (best < 0) return 0;
        restore_snapshot_into_slot(prefix_caches[best], slot, best_N);
        return best_N;
    }

    // Copy a validated snapshot (pc, first N tokens) into the slot's KV[0,N)
    // and GDN state. Caller holds prefix_caches_mu and has verified the match.
    void restore_snapshot_into_slot(PrefixSnapshot& pc, int slot, int N) {
        reset_slot_states(slot);
        int num_kv = cfg.num_kv_heads, hd = cfg.head_dim, kv_dim = num_kv * hd;
        size_t slot_kv_off = kv_slot_offset(slot) * kv_dim;
        if (use_turboquant) {
            for (auto& [layer, kv] : tq_kv_caches) {
                auto it = pc.tqkv_copy.find(layer);
                if (it == pc.tqkv_copy.end()) continue;
                size_t off   = kv_slot_offset(slot) * kv.blocks_per_token;
                size_t bytes = (size_t)N * kv.blocks_per_token * sizeof(block_tq3);
                int g = gpu->layer_gpu[layer];
                cudaSetDevice(g);
                cudaMemcpy(kv.k + off, it->second.first,  bytes, cudaMemcpyHostToDevice);
                cudaMemcpy(kv.v + off, it->second.second, bytes, cudaMemcpyHostToDevice);
            }
            // Force the per-token TQ read path to re-dequant [0,N) on the next
            // forward_attn; the dequant scratch otherwise holds stale positions.
            if (!tq_decoded_until.empty())
                std::fill(tq_decoded_until.begin(), tq_decoded_until.end(), -1);
        } else if (use_q8_kv) {
            for (auto& [layer, kv] : q8_kv_caches) {
                auto it = pc.q8kv_copy.find(layer);
                if (it == pc.q8kv_copy.end()) continue;
                size_t off   = kv_slot_offset(slot) * kv.blocks_per_token;
                size_t bytes = (size_t)N * kv.blocks_per_token * sizeof(block_q8_0_aligned);
                int g = gpu->layer_gpu[layer];
                cudaSetDevice(g);
                cudaMemcpy(kv.k + off, it->second.first,  bytes, cudaMemcpyHostToDevice);
                cudaMemcpy(kv.v + off, it->second.second, bytes, cudaMemcpyHostToDevice);
            }
        } else {
            for (auto& [layer, kv] : kv_caches) {
                auto it = pc.kv_copy.find(layer);
                if (it == pc.kv_copy.end()) continue;
                int g = gpu->layer_gpu[layer];
                cudaSetDevice(g);
                cudaMemcpy(kv.k + slot_kv_off, it->second.first,
                           (size_t)N * kv_dim * sizeof(half), cudaMemcpyHostToDevice);
                cudaMemcpy(kv.v + slot_kv_off, it->second.second,
                           (size_t)N * kv_dim * sizeof(half), cudaMemcpyHostToDevice);
            }
        }
        size_t conv_sz = (size_t)gdn_qkv_dim_cached * 4 * sizeof(float);
        size_t rec_sz  = (size_t)gdn_rec_per_slot * sizeof(float);
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (!gdn_states[layer].conv_state) continue;
            auto it = pc.gdn_copy.find(layer);
            if (it == pc.gdn_copy.end()) continue;
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            cudaMemcpy(gdn_conv_slot(layer, slot), it->second.first,
                       conv_sz, cudaMemcpyHostToDevice);
            cudaMemcpy(gdn_rec_slot(layer, slot), it->second.second,
                       rec_sz,  cudaMemcpyHostToDevice);
        }
    }

    // Snapshot the slot's current KV[0..n_pos) + GDN state into its prefix
    // slot. Lazily (re)allocates the snapshot buffers; reallocates when the
    // requested length differs from the prior allocation.
    void save_prefix_snapshot(const std::vector<int>& prompt_tokens, int n_pos, int slot = 0) {
        ensure_prefix_caches();
        if (slot < 0 || slot >= num_slots) return;
        if (n_pos <= 0 || n_pos > (int)prompt_tokens.size()) return;
        std::lock_guard<std::mutex> lk(prefix_caches_mu);
        PrefixSnapshot& pc = prefix_caches[slot];
        int num_kv = cfg.num_kv_heads, hd = cfg.head_dim, kv_dim = num_kv * hd;
        size_t slot_kv_off = kv_slot_offset(slot) * kv_dim;

        bool realloc = (pc.kv_capacity != n_pos);
        if (realloc) {
            // free without re-locking — we already hold prefix_caches_mu and
            // free_prefix_cache_buffers does no locking.
            for (auto& [layer, kv] : pc.kv_copy)   { cudaFreeHost(kv.first); cudaFreeHost(kv.second); }
            for (auto& [layer, kv] : pc.tqkv_copy) { cudaFreeHost(kv.first); cudaFreeHost(kv.second); }
            for (auto& [layer, kv] : pc.q8kv_copy) { cudaFreeHost(kv.first); cudaFreeHost(kv.second); }
            for (auto& [layer, gd] : pc.gdn_copy)  { cudaFreeHost(gd.first); cudaFreeHost(gd.second); }
            pc.kv_copy.clear();
            pc.tqkv_copy.clear();
            pc.q8kv_copy.clear();
            pc.gdn_copy.clear();
            pc.kv_capacity = 0;
            pc.valid = false;
        }
        // KV save — format-aware (only one of the three caches is live).
        if (use_turboquant) {
            for (auto& [layer, kv] : tq_kv_caches) {
                size_t bytes = (size_t)n_pos * kv.blocks_per_token * sizeof(block_tq3);
                size_t off   = kv_slot_offset(slot) * kv.blocks_per_token;
                int g = gpu->layer_gpu[layer];
                cudaSetDevice(g);
                if (realloc) {
                    char *kc = nullptr, *vc = nullptr;
                    cudaMallocHost(&kc, bytes); cudaMallocHost(&vc, bytes);
                    pc.tqkv_copy[layer] = {kc, vc};
                }
                auto& dst = pc.tqkv_copy[layer];
                cudaMemcpy(dst.first,  kv.k + off, bytes, cudaMemcpyDeviceToHost);
                cudaMemcpy(dst.second, kv.v + off, bytes, cudaMemcpyDeviceToHost);
            }
        } else if (use_q8_kv) {
            for (auto& [layer, kv] : q8_kv_caches) {
                size_t bytes = (size_t)n_pos * kv.blocks_per_token * sizeof(block_q8_0_aligned);
                size_t off   = kv_slot_offset(slot) * kv.blocks_per_token;
                int g = gpu->layer_gpu[layer];
                cudaSetDevice(g);
                if (realloc) {
                    char *kc = nullptr, *vc = nullptr;
                    cudaMallocHost(&kc, bytes); cudaMallocHost(&vc, bytes);
                    pc.q8kv_copy[layer] = {kc, vc};
                }
                auto& dst = pc.q8kv_copy[layer];
                cudaMemcpy(dst.first,  kv.k + off, bytes, cudaMemcpyDeviceToHost);
                cudaMemcpy(dst.second, kv.v + off, bytes, cudaMemcpyDeviceToHost);
            }
        } else {
            for (auto& [layer, kv] : kv_caches) {
                int g = gpu->layer_gpu[layer];
                cudaSetDevice(g);
                if (realloc) {
                    half *kc = nullptr, *vc = nullptr;
                    cudaMallocHost(&kc, (size_t)n_pos * kv_dim * sizeof(half));
                    cudaMallocHost(&vc, (size_t)n_pos * kv_dim * sizeof(half));
                    pc.kv_copy[layer] = {kc, vc};
                }
                auto& dst = pc.kv_copy[layer];
                cudaMemcpy(dst.first,  kv.k + slot_kv_off,
                           (size_t)n_pos * kv_dim * sizeof(half), cudaMemcpyDeviceToHost);
                cudaMemcpy(dst.second, kv.v + slot_kv_off,
                           (size_t)n_pos * kv_dim * sizeof(half), cudaMemcpyDeviceToHost);
            }
        }
        if (realloc) {
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                if (!gdn_states[layer].conv_state) continue;
                float *cc = nullptr, *rc = nullptr;
                cudaMallocHost(&cc, gdn_qkv_dim_cached * 4 * sizeof(float));
                cudaMallocHost(&rc, gdn_rec_per_slot * sizeof(float));
                pc.gdn_copy[layer] = {cc, rc};
            }
            pc.kv_capacity = n_pos;
        }
        size_t conv_sz = (size_t)gdn_qkv_dim_cached * 4 * sizeof(float);
        size_t rec_sz  = (size_t)gdn_rec_per_slot * sizeof(float);
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (!gdn_states[layer].conv_state) continue;
            auto it = pc.gdn_copy.find(layer);
            if (it == pc.gdn_copy.end()) continue;
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            cudaMemcpy(it->second.first,  gdn_conv_slot(layer, slot),
                       conv_sz, cudaMemcpyDeviceToHost);
            cudaMemcpy(it->second.second, gdn_rec_slot(layer, slot),
                       rec_sz,  cudaMemcpyDeviceToHost);
        }
        pc.tokens.assign(prompt_tokens.begin(), prompt_tokens.begin() + n_pos);
        pc.n_pos = n_pos;
        pc.valid = true;
    }

    void reset_gdn_states_inner() {
        if (gdn_states.empty()) return;  // dense model (no SSM): nothing to reset
        size_t conv_bytes = (size_t)num_slots * gdn_qkv_dim_cached * 4 * sizeof(float);
        size_t rec_bytes  = (size_t)num_slots * gdn_rec_per_slot * sizeof(float);
        for (int layer = 0; layer < cfg.num_layers && layer < (int)gdn_states.size(); layer++) {
            if (!gdn_states[layer].conv_state) continue;
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            cudaMemset(gdn_states[layer].conv_state, 0, conv_bytes);
            cudaMemset(gdn_states[layer].rec_state, 0, rec_bytes);
        }
    }

    // Reset only slot s's GDN state.
    void reset_gdn_slot(int slot) {
        if (slot < 0 || slot >= num_slots) return;
        size_t conv_off = (size_t)slot * gdn_qkv_dim_cached * 4;
        size_t rec_off  = (size_t)slot * gdn_rec_per_slot;
        size_t conv_sz  = gdn_qkv_dim_cached * 4 * sizeof(float);
        size_t rec_sz   = gdn_rec_per_slot * sizeof(float);
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (!gdn_states[layer].conv_state) continue;
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            cudaMemset(gdn_states[layer].conv_state + conv_off, 0, conv_sz);
            cudaMemset(gdn_states[layer].rec_state  + rec_off,  0, rec_sz);
        }
    }

    // Per-slot reset = KV range + GDN state for that slot only.
    void reset_slot_states(int slot) {
        reset_kv_slot(slot);
        reset_gdn_slot(slot);
    }

    // Stream-ordered per-slot reset restricted to the layers [l_lo, l_hi) that
    // live on GPU `g`. Issued on `stream` (the GPU's compute stream) so the
    // memsets are correctly ordered BEFORE that segment's forward kernels —
    // used by the offline dflash-extract cross-sequence pipeline, where each
    // GPU stage resets the next sequence's slot just before computing it. Only
    // the fp16-KV path is supported (extract never sets MTP_TQ/Q8KV).
    void reset_slot_segment_stream(int slot, int l_lo, int l_hi, int g,
                                   cudaStream_t stream) {
        if (slot < 0 || slot >= num_slots) return;
        cudaSetDevice(g);
        // GDN conv + recurrent state for this slot, this GPU's GDN layers.
        size_t conv_off = (size_t)slot * gdn_qkv_dim_cached * 4;
        size_t rec_off  = (size_t)slot * gdn_rec_per_slot;
        size_t conv_sz  = (size_t)gdn_qkv_dim_cached * 4 * sizeof(float);
        size_t rec_sz   = (size_t)gdn_rec_per_slot * sizeof(float);
        for (int layer = l_lo; layer < l_hi; layer++) {
            if (layer < 0 || layer >= cfg.num_layers) continue;
            if (!gdn_states[layer].conv_state) continue;
            cudaMemsetAsync(gdn_states[layer].conv_state + conv_off, 0, conv_sz, stream);
            cudaMemsetAsync(gdn_states[layer].rec_state  + rec_off,  0, rec_sz,  stream);
        }
        // fp16 KV range for this slot, this GPU's attn layers.
        int kv_dim = cfg.num_kv_heads * cfg.head_dim;
        size_t per_slot = (size_t)slot_capacity(slot) * kv_dim * sizeof(half);
        size_t off_elem = kv_slot_offset(slot) * kv_dim;
        for (auto& [layer, kv] : kv_caches) {
            if (layer < l_lo || layer >= l_hi) continue;
            cudaMemsetAsync(kv.k + off_elem, 0, per_slot, stream);
            cudaMemsetAsync(kv.v + off_elem, 0, per_slot, stream);
        }
    }

    bool is_attn_layer(int layer) { return (layer >= 0 && layer < 256) ? layer_is_attn[layer] : false; }


    // GDN forward (seq_len=1, with cache).  `slot` selects the per-request
    // recurrent state (default 0 = legacy single-slot behavior).
    void forward_gdn(int layer, float* hidden, cudaStream_t stream, int slot = 0) {
        int g = gpu->layer_gpu[layer];
        auto& buf = bufs[g];
        int H = cfg.hidden_size;
        int num_k = cfg.linear_k_heads;
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        int qkv_dim = 2 * num_k * k_dim + num_v * v_dim;
        float* conv_state_slot = gdn_conv_slot(layer, slot);
        float* rec_state_slot  = gdn_rec_slot(layer, slot);

        auto* norm_w  = t(blk(layer, "attn_norm.weight"));
        auto* qkv_w   = t(blk(layer, "attn_qkv.weight"));
        auto* gate_w  = t(blk(layer, "attn_gate.weight"));  // Z projection
        auto* alpha_w = t(blk(layer, "ssm_alpha.weight"));
        auto* beta_w  = t(blk(layer, "ssm_beta.weight"));
        auto* conv_w  = t(blk(layer, "ssm_conv1d.weight"));
        auto* a_log_t = t(blk(layer, "ssm_a"));
        auto* dt_bias_t = t(blk(layer, "ssm_dt.bias"));
        auto* ssm_norm_w = t(blk(layer, "ssm_norm.weight"));
        auto* out_w   = t(blk(layer, "ssm_out.weight"));

        // Temp buffers (reuse attn_out which is large enough)
        half* norm_out = buf.norm_out;
        half* qkv_out = buf.attn_out;  // [qkv_dim]
        half* z_out = buf.mlp_gate;    // reuse, [num_v * v_dim]
        half* a_out = buf.mlp_up;      // reuse, [num_v]
        half* b_out = buf.mlp_down;    // reuse, [num_v]

        // PROFILE_GDN sub-phase timers. Sync between phases is heavy, so kept
        // OFF unless explicitly enabled.
        auto pt_now_g = [](){ return std::chrono::high_resolution_clock::now(); };
        auto pt_sync_g = [&](std::chrono::high_resolution_clock::time_point tb) {
            cudaSetDevice(g); cudaStreamSynchronize(stream);
            auto te = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(te - tb).count();
        };
        auto pg_t0 = pt_now_g();

        // 1. RMSNorm (FP32 hidden in)
        if (norm_w->type == GGML_TYPE_F32)
            rms_norm_f32in_f32w(norm_out, hidden, (float*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);
        else
            rms_norm_f32in(norm_out, hidden, (half*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);
        if (g_profile_gdn) { g_gdn_norm_ms += pt_sync_g(pg_t0); pg_t0 = pt_now_g(); }

        // 2. Projections: QKV, Z(gate), alpha, beta — quantize once, reuse
        gpu_qi[g].quantize(norm_out, H, stream);
        static const bool use_proj_fuse = getenv("GDN_PROJ_FUSE") != nullptr;
        bool can_fuse = use_proj_fuse
                        && qkv_w->type   == GGML_TYPE_Q8_0
                        && gate_w->type  == GGML_TYPE_Q8_0
                        && alpha_w->type == GGML_TYPE_Q8_0
                        && beta_w->type  == GGML_TYPE_Q8_0;
        if (can_fuse) {
            if ((int)gdn_proj_fused.size() < cfg.num_layers) gdn_proj_fused.resize(cfg.num_layers);
            auto& pf = gdn_proj_fused[layer];
            if (!pf.weight) {
                int blocks_per_row = H / 32;
                size_t bpr_bytes = (size_t)blocks_per_row * sizeof(block_q8_0_aligned);
                pf.qkv_n   = qkv_w->dims[1];
                pf.gate_n  = gate_w->dims[1];
                pf.alpha_n = alpha_w->dims[1];
                pf.beta_n  = beta_w->dims[1];
                pf.total_n = pf.qkv_n + pf.gate_n + pf.alpha_n + pf.beta_n;
                size_t total_bytes = (size_t)pf.total_n * bpr_bytes;
                cudaSetDevice(g);
                cudaMalloc(&pf.weight, total_bytes);
                size_t off = 0;
                cudaMemcpyAsync((char*)pf.weight + off, qkv_w->data,
                                (size_t)pf.qkv_n * bpr_bytes, cudaMemcpyDeviceToDevice, stream);
                off += (size_t)pf.qkv_n * bpr_bytes;
                cudaMemcpyAsync((char*)pf.weight + off, gate_w->data,
                                (size_t)pf.gate_n * bpr_bytes, cudaMemcpyDeviceToDevice, stream);
                off += (size_t)pf.gate_n * bpr_bytes;
                cudaMemcpyAsync((char*)pf.weight + off, alpha_w->data,
                                (size_t)pf.alpha_n * bpr_bytes, cudaMemcpyDeviceToDevice, stream);
                off += (size_t)pf.alpha_n * bpr_bytes;
                cudaMemcpyAsync((char*)pf.weight + off, beta_w->data,
                                (size_t)pf.beta_n * bpr_bytes, cudaMemcpyDeviceToDevice, stream);
                cudaStreamSynchronize(stream);
            }
            half* fused_out = gdn_bufs[g].fused_proj_out;
            quant_gemv(pf.weight, GGML_TYPE_Q8_0, norm_out, fused_out, H, pf.total_n, &gpu_qi[g], stream);
            qkv_out = fused_out;
            z_out   = fused_out + pf.qkv_n;
            a_out   = fused_out + pf.qkv_n + pf.gate_n;
            b_out   = fused_out + pf.qkv_n + pf.gate_n + pf.alpha_n;
        } else {
            quant_gemv(qkv_w->data, qkv_w->type, norm_out, qkv_out, H, qkv_dim, &gpu_qi[g], stream);
            quant_gemv(gate_w->data, gate_w->type, norm_out, z_out, H, num_v * v_dim, &gpu_qi[g], stream);
            quant_gemv(alpha_w->data, alpha_w->type, norm_out, a_out, H, num_v, &gpu_qi[g], stream);
            quant_gemv(beta_w->data, beta_w->type, norm_out, b_out, H, num_v, &gpu_qi[g], stream);
        }
        if (g_profile_gdn) { g_gdn_proj_ms += pt_sync_g(pg_t0); pg_t0 = pt_now_g(); }

        // 3. Conv1d update (FP32) — or fold into layer-coop kernel below.
        float* conv_out = gdn_bufs[g].conv_out;
        int kw = 4;
        static const bool use_layer_coop = getenv("GDN_LAYER_COOP") != nullptr;
        bool can_coop = use_layer_coop && k_dim == 128 && v_dim == 128;
        if (!can_coop) {
            int threads_conv = min(qkv_dim, 256);
            int blocks_conv = (qkv_dim + threads_conv - 1) / threads_conv;
            conv1d_update_silu<<<blocks_conv, threads_conv, 0, stream>>>(
                conv_state_slot,
                qkv_out,
                (float*)conv_w->data,
                conv_out,
                qkv_dim, kw
            );
            if (g_profile_gdn) { g_gdn_conv_ms += pt_sync_g(pg_t0); pg_t0 = pt_now_g(); }
        }
        static const bool dump_gdn = getenv("DUMP_GDN") != nullptr;
        if (dump_gdn && layer == 0) {
            cudaDeviceSynchronize();
            float sample[8];
            cudaMemcpy(sample, conv_out, 8 * sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[GDN L0 conv]  %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
                    sample[0], sample[1], sample[2], sample[3],
                    sample[4], sample[5], sample[6], sample[7]);
            fflush(stderr);
        }

        // 4+5. Recurrent GDN step + RMSNorm Gated (fused if GDN_LAYER_FUSE=1,
        //      or full conv1d+recur+rmsg cooperative launch if GDN_LAYER_COOP=1)
        half* core_out = gdn_bufs[g].core_out;
        half* normed_out = gdn_bufs[g].normed_out;
        static const bool use_layer_fuse = getenv("GDN_LAYER_FUSE") != nullptr;
        if (can_coop) {
            int threads = 256;
            int grid_dim = std::max((qkv_dim + threads - 1) / threads, num_v);
            int smem_coop = (2 * 128 + 1 + threads) * sizeof(float);
            half* qkv_in_local = qkv_out;
            float* conv_w_data = (float*)conv_w->data;
            float* a_log_data = (float*)a_log_t->data;
            float* dt_bias_data = (float*)dt_bias_t->data;
            float* ssm_norm_data = (float*)ssm_norm_w->data;
            float eps_v = 1e-6f;
            void* kernelArgs[] = {
                (void*)&conv_state_slot,
                (void*)&qkv_in_local,
                (void*)&conv_w_data,
                (void*)&conv_out,
                (void*)&qkv_dim, (void*)&kw,
                (void*)&a_log_data,
                (void*)&dt_bias_data,
                (void*)&a_out,
                (void*)&b_out,
                (void*)&rec_state_slot,
                (void*)&z_out,
                (void*)&ssm_norm_data,
                (void*)&normed_out,
                (void*)&num_k, (void*)&num_v, (void*)&eps_v
            };
            cudaSetDevice(g);
            cudaLaunchCooperativeKernel(
                (const void*)gdn_layer_fused_recur_rmsg<128, 128>,
                dim3(grid_dim), dim3(threads),
                kernelArgs, (size_t)smem_coop, stream
            );
            if (g_profile_gdn) {
                double dt = pt_sync_g(pg_t0); pg_t0 = pt_now_g();
                g_gdn_conv_ms  += dt * 0.0;  // (folded; charge to recur for now)
                g_gdn_recur_ms += dt;
            }
        } else if (use_layer_fuse && k_dim == 128 && v_dim == 128) {
            int fuse_smem = (2 * 128 + 1 + 128) * sizeof(float);
            gdn_fused_recur_rmsg<128, 128><<<num_v, 128, fuse_smem, stream>>>(
                conv_out,
                (float*)a_log_t->data,
                (float*)dt_bias_t->data,
                a_out, b_out,
                rec_state_slot,
                z_out,
                (float*)ssm_norm_w->data,
                normed_out,
                num_k, num_v, 1e-6f
            );
            if (g_profile_gdn) { g_gdn_recur_ms += pt_sync_g(pg_t0); pg_t0 = pt_now_g(); }
        } else {
            int gdn_smem = (2 * cfg.linear_k_dim + 1) * sizeof(float);
            launch_gdn_recurrent_step(num_v, min(v_dim, 128), gdn_smem, stream,
                conv_out,
                (float*)a_log_t->data,
                (float*)dt_bias_t->data,
                a_out,
                b_out,
                rec_state_slot,
                core_out,
                num_k, num_v, k_dim, v_dim
            );
            if (g_profile_gdn) { g_gdn_recur_ms += pt_sync_g(pg_t0); pg_t0 = pt_now_g(); }
            if (dump_gdn && layer == 0) {
                cudaDeviceSynchronize();
                half sample[8];
                cudaMemcpy(sample, core_out, 8 * sizeof(half), cudaMemcpyDeviceToHost);
                fprintf(stderr, "[GDN L0 core]  %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
                        __half2float(sample[0]), __half2float(sample[1]), __half2float(sample[2]),
                        __half2float(sample[3]), __half2float(sample[4]), __half2float(sample[5]),
                        __half2float(sample[6]), __half2float(sample[7]));
                fflush(stderr);
            }
            rms_norm_gated_kernel<<<num_v, min(v_dim, 128), 128 * sizeof(float), stream>>>(
                core_out, z_out, (float*)ssm_norm_w->data, normed_out,
                num_v, v_dim, 1e-6f
            );
            if (g_profile_gdn) { g_gdn_rmsg_ms += pt_sync_g(pg_t0); pg_t0 = pt_now_g(); }
        }
        if (dump_gdn && layer == 0) {
            cudaDeviceSynchronize();
            half sample[8];
            cudaMemcpy(sample, normed_out, 8 * sizeof(half), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[GDN L0 normed] %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
                    __half2float(sample[0]), __half2float(sample[1]), __half2float(sample[2]),
                    __half2float(sample[3]), __half2float(sample[4]), __half2float(sample[5]),
                    __half2float(sample[6]), __half2float(sample[7]));
            fflush(stderr);
        }

        // 6. Output projection
        half* proj_out = gdn_bufs[g].proj_out;
        gpu_qi_inter[g].quantize(normed_out, num_v * v_dim, stream);
        quant_gemv(out_w->data, out_w->type, normed_out, proj_out, num_v * v_dim, H, &gpu_qi_inter[g], stream);
        if (g_profile_gdn) { g_gdn_oproj_ms += pt_sync_g(pg_t0); pg_t0 = pt_now_g(); }
        if (dump_gdn && layer == 0) {
            cudaDeviceSynchronize();
            half sample[8];
            cudaMemcpy(sample, proj_out, 8 * sizeof(half), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[GDN L0 proj]  %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
                    __half2float(sample[0]), __half2float(sample[1]), __half2float(sample[2]),
                    __half2float(sample[3]), __half2float(sample[4]), __half2float(sample[5]),
                    __half2float(sample[6]), __half2float(sample[7]));
            fflush(stderr);
        }

        // 7. Residual add into FP32 hidden
        add_kernel_f32<<<(H+255)/256, 256, 0, stream>>>(hidden, proj_out, H);
        if (g_profile_gdn) { g_gdn_resi_ms += pt_sync_g(pg_t0); g_gdn_calls++; }
    }

    // ============ Chunked GDN forward (process N tokens together) ============
    // hidden_chunk: [n_tokens, H] FP32 — read & updated in-place.
    // `slot` selects per-request recurrent state (default 0 = legacy single-slot).
    void forward_gdn_chunk(int layer, float* hidden_chunk, int n_tokens, cudaStream_t stream, int slot = 0) {
        int g = gpu->layer_gpu[layer];
        auto& buf = bufs[g];
        auto& gb = gdn_bufs[g];
        int H = cfg.hidden_size;
        int num_k = cfg.linear_k_heads;
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        int qkv_dim = 2 * num_k * k_dim + num_v * v_dim;
        int v_total = num_v * v_dim;
        float* conv_state_slot = gdn_conv_slot(layer, slot);
        float* rec_state_slot  = gdn_rec_slot(layer, slot);

        auto* norm_w  = t(blk(layer, "attn_norm.weight"));
        auto* qkv_w   = t(blk(layer, "attn_qkv.weight"));
        auto* gate_w  = t(blk(layer, "attn_gate.weight"));
        auto* alpha_w = t(blk(layer, "ssm_alpha.weight"));
        auto* beta_w  = t(blk(layer, "ssm_beta.weight"));
        auto* conv_w  = t(blk(layer, "ssm_conv1d.weight"));
        auto* a_log_t = t(blk(layer, "ssm_a"));
        auto* dt_bias_t = t(blk(layer, "ssm_dt.bias"));
        auto* ssm_norm_w = t(blk(layer, "ssm_norm.weight"));
        auto* out_w   = t(blk(layer, "ssm_out.weight"));

        // ============ Batched RMSNorm + Q/K/V/Z/α/β projection ============
        // All four projections (qkv, gate, alpha, beta) share the same
        // pre-quantized norm output. The N-token batched GEMV reuses each
        // weight word across n_tokens dp4a accumulators, which is the actual
        // prefill speedup vs the per-token loop.
        // DEBUG env — disable batched projections to isolate chunked kernel
        // drift. Forces GDN layer to loop quant_gemv per-token while still
        // going through the chunked control flow (conv1d_chunk, gdn_chunk_step).
        static const bool gdn_no_batch = getenv("GDN_NO_BATCH_PROJ") != nullptr;
        bool can_batch_gdn = !gdn_no_batch
                          && (qkv_w->type   == GGML_TYPE_Q8_0)
                          && (gate_w->type  == GGML_TYPE_Q8_0)
                          && (alpha_w->type == GGML_TYPE_Q8_0)
                          && (beta_w->type  == GGML_TYPE_Q8_0);

        if (can_batch_gdn) {
            // 1. Batched RMSNorm: rows=n_tokens, write into chunk_norm_out
            if (norm_w->type == GGML_TYPE_F32) {
                rms_norm_f32in_f32w(gb.chunk_norm_out, hidden_chunk, (float*)norm_w->data,
                                    n_tokens, H, cfg.rms_norm_eps, stream);
            } else {
                rms_norm_f32in(gb.chunk_norm_out, hidden_chunk, (half*)norm_w->data,
                               n_tokens, H, cfg.rms_norm_eps, stream);
            }

            // 2. Quantize all n_tokens × H normed values once
            gpu_qi[g].quantize_chunk(gb.chunk_norm_out, H, n_tokens, stream);

            // 3. Batched QKV projection. Output is [n_tokens, qkv_dim] half,
            //    we write into buf.attn_out as scratch and immediately
            //    half→float into chunk_qkv. Need scratch sized n_tokens*qkv_dim.
            //    buf.attn_out is sized max(H*4, num_q*hd*2) ≥ 12288, which
            //    is < n_tokens*qkv_dim for chunks > 1. We instead reuse
            //    chunk_z_out as fp16 staging since it's [CHUNK*v_total]
            //    (v_total=6144) — that's only 6144 elem per token but qkv_dim
            //    is 10240. So allocate a dedicated qkv_chunk buffer instead.
            //    For now: write directly to chunk_qkv as fp16 then convert.
            //    Actually the simplest: do batched GEMV writing into a fp16
            //    chunk staging, then half→float.
            //
            //    Simpler still: write the batched GEMV output into the FRONT
            //    of chunk_qkv (which is fp32 [CHUNK*qkv_dim]). We need fp16
            //    staging though, because gemv_q8_0_q8_nN writes half outputs.
            //    Use chunk_proj_out as fp16 staging (it's CHUNK*H half = at
            //    least CHUNK*5120; for 27B qkv_dim=10240 that's not enough).
            //
            //    Allocate a dedicated fp16 staging if not already there.
            if (!gb.chunk_qkv_half) {
                cudaMalloc(&gb.chunk_qkv_half, (size_t)CHUNK_SIZE * qkv_dim * sizeof(half));
            }

            quant_gemv_chunk(qkv_w->data, qkv_w->type,
                             gpu_qi[g].q8_buf, gb.chunk_qkv_half,
                             H, qkv_dim, n_tokens, stream);

            // Convert chunk_qkv_half [n_tokens, qkv_dim] → chunk_qkv [n_tokens, qkv_dim] fp32
            {
                int n_elem = n_tokens * qkv_dim;
                half_to_float_kernel<<<(n_elem + 255) / 256, 256, 0, stream>>>(
                    gb.chunk_qkv_half, gb.chunk_qkv, n_elem);
            }

            // Z (gate) projection — direct fp16 output to chunk_z_out
            quant_gemv_chunk(gate_w->data, gate_w->type,
                             gpu_qi[g].q8_buf, gb.chunk_z_out,
                             H, num_v * v_dim, n_tokens, stream);

            // alpha, beta projections — direct fp16 outputs
            quant_gemv_chunk(alpha_w->data, alpha_w->type,
                             gpu_qi[g].q8_buf, gb.chunk_a_proj,
                             H, num_v, n_tokens, stream);
            quant_gemv_chunk(beta_w->data, beta_w->type,
                             gpu_qi[g].q8_buf, gb.chunk_b_proj,
                             H, num_v, n_tokens, stream);
        } else {
            // Fallback: original per-token path for non-Q8_0 quantizations.
            for (int t = 0; t < n_tokens; t++) {
                float* h_t = hidden_chunk + (size_t)t * H;
                half* nrm = gb.chunk_norm_out + (size_t)t * H;

                if (norm_w->type == GGML_TYPE_F32)
                    rms_norm_f32in_f32w(nrm, h_t, (float*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);
                else
                    rms_norm_f32in(nrm, h_t, (half*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);

                gpu_qi[g].quantize(nrm, H, stream);
                quant_gemv(qkv_w->data, qkv_w->type, nrm, buf.attn_out, H, qkv_dim, &gpu_qi[g], stream);
                half_to_float_kernel<<<(qkv_dim+255)/256, 256, 0, stream>>>(
                    buf.attn_out, gb.chunk_qkv + (size_t)t * qkv_dim, qkv_dim);
                quant_gemv(gate_w->data, gate_w->type, nrm,
                           gb.chunk_z_out + (size_t)t * v_total, H, num_v * v_dim, &gpu_qi[g], stream);
                quant_gemv(alpha_w->data, alpha_w->type, nrm,
                           gb.chunk_a_proj + (size_t)t * num_v, H, num_v, &gpu_qi[g], stream);
                quant_gemv(beta_w->data, beta_w->type, nrm,
                           gb.chunk_b_proj + (size_t)t * num_v, H, num_v, &gpu_qi[g], stream);
            }
        }

        // Conv1d update — single chunked launch instead of n_tokens
        // per-token launches. conv1d_update_silu_chunk walks the n_tokens
        // dimension inside one block per dim element, sharing the conv
        // state register accumulator across the whole sub-sequence.
        //
        // Input format: needs fp16 [n_tokens, qkv_dim]. We have fp32
        // [n_tokens, qkv_dim] in gb.chunk_qkv from the projection above
        // (the batched path) or from the fallback. Convert in one shot
        // into chunk_qkv_half (already alloc'd by the batched projection
        // path; if the fallback was used, allocate it on demand).
        {
            int kw = 4;
            if (!gb.chunk_qkv_half) {
                cudaMalloc(&gb.chunk_qkv_half, (size_t)CHUNK_SIZE * qkv_dim * sizeof(half));
            }
            int n_elem = n_tokens * qkv_dim;
            float_to_half_kernel<<<(n_elem + 255) / 256, 256, 0, stream>>>(
                gb.chunk_qkv, gb.chunk_qkv_half, n_elem);

            int threads = min(qkv_dim, 256);
            int blocks  = (qkv_dim + threads - 1) / threads;
            conv1d_update_silu_chunk<<<blocks, threads, 0, stream>>>(
                conv_state_slot, gb.chunk_qkv_half,
                (float*)conv_w->data, gb.chunk_qkv, qkv_dim, kw, n_tokens);
        }

        // Chunked GDN recurrent step (single kernel call for all N tokens).
        // The state tensor (k_dim × v_dim = 64 KB at k_dim=128, v_dim=128) is
        // kept resident in SMEM for the whole chunk, so the kd re-reads inside
        // the per-token loop hit SMEM instead of global memory. Requires the
        // >48 KB opt-in attribute on Volta.
        {
            int threads    = min(v_dim, 128);
            int nwarp      = (threads + 31) / 32;
            int state_len  = k_dim * v_dim;
            // state + sQ(k_dim+1) + sK(k_dim) + sWQ[32] + sWK[32] + sWA[32].
            // The 32-slot warp scratch matches gdn_recurrent_step so the
            // cross-warp tree reduce is bit-identical.
            int gdn_smem   = (state_len + 2 * k_dim + 1 + 96) * sizeof(float);
            // Opt-in 96 KB dynamic SMEM. `cudaFuncSetAttribute` is per-device,
            // so we must set it once *per GPU* — previously a shared `static
            // bool` guarded the call, which meant only the first GPU that ran
            // this kernel got the opt-in. On a multi-GPU layout (L0-15 GPU0,
            // L16-31 GPU1, ...), GPUs 1-3 silently ran with the 48 KB default
            // limit, so the 66 KB SMEM request either failed at launch or hit
            // undefined behavior — producing layer-wise drift starting at the
            // first GPU-boundary GDN layer (reproduced on 9B at L08).
            static bool smem_attr_set[16] = {false};
            int cur_dev = 0; cudaGetDevice(&cur_dev);
            if (cur_dev >= 0 && cur_dev < 16 && !smem_attr_set[cur_dev]) {
                cudaFuncSetAttribute(
                    (const void*)gdn_chunk_step,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    96 * 1024);
                smem_attr_set[cur_dev] = true;
            }
            gdn_chunk_step<<<num_v, threads, gdn_smem, stream>>>(
                gb.chunk_qkv,
                (float*)a_log_t->data,
                (float*)dt_bias_t->data,
                gb.chunk_a_proj,
                gb.chunk_b_proj,
                rec_state_slot,
                gb.chunk_core_out,
                n_tokens, num_k, num_v, k_dim, v_dim
            );
        }

        // RMSNorm gated (chunked) + batched output projection + batched residual
        {
            // Chunked rms_norm_gated → gb.chunk_normed [n_tokens * v_total] half
            int total_blocks = num_v * n_tokens;
            rms_norm_gated_chunk_kernel<<<total_blocks, min(v_dim, 128), 128*sizeof(float), stream>>>(
                gb.chunk_core_out, gb.chunk_z_out, (float*)ssm_norm_w->data,
                gb.chunk_normed, num_v, v_dim, n_tokens, 1e-6f);

            if (out_w->type == GGML_TYPE_Q8_0) {
                // Batched output projection: quantize all n_tokens × v_total
                // values once, then a single chunked GEMM call for the
                // [v_total → H] projection. This was the largest leftover
                // per-token GEMV in the GDN path (out_w is the same size
                // class as MLP down_proj).
                gpu_qi_inter[g].quantize_chunk(gb.chunk_normed, num_v * v_dim, n_tokens, stream);
                quant_gemv_chunk(out_w->data, out_w->type,
                                 gpu_qi_inter[g].q8_buf, gb.chunk_proj_out,
                                 num_v * v_dim, H, n_tokens, stream);

                // Batched residual into FP32 hidden (one launch).
                int n_elem = n_tokens * H;
                add_kernel_f32<<<(n_elem + 255) / 256, 256, 0, stream>>>(
                    hidden_chunk, gb.chunk_proj_out, n_elem);
            } else {
                // Non-Q8_0 fallback: per-token (rare).
                for (int t = 0; t < n_tokens; t++) {
                    half* normed_t = gb.chunk_normed + (size_t)t * v_total;
                    half* proj_t   = gb.chunk_proj_out + (size_t)t * H;
                    gpu_qi_inter[g].quantize(normed_t, num_v * v_dim, stream);
                    quant_gemv(out_w->data, out_w->type, normed_t, proj_t,
                               num_v * v_dim, H, &gpu_qi_inter[g], stream);
                    add_kernel_f32<<<(H+255)/256, 256, 0, stream>>>(
                        hidden_chunk + (size_t)t * H, proj_t, H);
                }
            }
        }
    }

    // ============ DDTree: GDN forward over n_tokens tree nodes ============
    // Structure mirrors forward_gdn_chunk but swaps the two state-carrying
    // kernels (conv1d, gdn_recurrent) for their tree-mode variants so each
    // node can reload its parent's state instead of inheriting the sequential
    // predecessor. `parent_ids_dev` is a [n_tokens] int32 buffer on the layer's
    // GPU; parent_ids_dev[t] == -1 means "parent is the pre-tree state".
    // Caller must have invoked `alloc_tree_decode(budget)` first.
    void forward_gdn_tree(int layer, float* hidden_tree, int n_tokens,
                          const int* parent_ids_dev, cudaStream_t stream, int slot = 0) {
        // Debug fallback: sequentially call non-tree forward_gdn for each node.
        // Correct for chain parent_ids=[-1, 0, 1, ..., n-1] because each call
        // updates rec_state/conv_state in-place, which is exactly what the
        // chain expects as each node's "parent state".
        //
        // WARNING: this mutates rec_state/conv_state in-place, so accept-time
        // commit is redundant. Used only to isolate whether the tree GDN
        // kernels are the correctness bug.
        static const bool fallback_gdn = getenv("TREE_FALLBACK_GDN") != nullptr;
        if (fallback_gdn) {
            int H = cfg.hidden_size;
            for (int t = 0; t < n_tokens; t++) {
                forward_gdn(layer, hidden_tree + (size_t)t * H, stream, slot);
            }
            return;
        }
        int g = gpu->layer_gpu[layer];
        auto& gb = gdn_bufs[g];
        int H = cfg.hidden_size;
        int num_k = cfg.linear_k_heads;
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        int qkv_dim = 2 * num_k * k_dim + num_v * v_dim;
        int v_total = num_v * v_dim;
        float* conv_state_slot = gdn_conv_slot(layer, slot);
        float* rec_state_slot  = gdn_rec_slot(layer, slot);

        auto* norm_w    = t(blk(layer, "attn_norm.weight"));
        auto* qkv_w     = t(blk(layer, "attn_qkv.weight"));
        auto* gate_w    = t(blk(layer, "attn_gate.weight"));
        auto* alpha_w   = t(blk(layer, "ssm_alpha.weight"));
        auto* beta_w    = t(blk(layer, "ssm_beta.weight"));
        auto* conv_w    = t(blk(layer, "ssm_conv1d.weight"));
        auto* a_log_t   = t(blk(layer, "ssm_a"));
        auto* dt_bias_t = t(blk(layer, "ssm_dt.bias"));
        auto* ssm_norm_w = t(blk(layer, "ssm_norm.weight"));
        auto* out_w     = t(blk(layer, "ssm_out.weight"));

        // 1. Batched RMSNorm over n_tokens.
        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w(gb.chunk_norm_out, hidden_tree, (float*)norm_w->data,
                                n_tokens, H, cfg.rms_norm_eps, stream);
        } else {
            rms_norm_f32in(gb.chunk_norm_out, hidden_tree, (half*)norm_w->data,
                           n_tokens, H, cfg.rms_norm_eps, stream);
        }

        // 2. Single quantize of all n_tokens × H normed values.
        gpu_qi[g].quantize_chunk(gb.chunk_norm_out, H, n_tokens, stream);

        // 3. Batched Q/K/V + Z + alpha + beta projections.
        if (!gb.chunk_qkv_half) {
            cudaMalloc(&gb.chunk_qkv_half, (size_t)CHUNK_SIZE * qkv_dim * sizeof(half));
        }
        quant_gemv_chunk(qkv_w->data, qkv_w->type, gpu_qi[g].q8_buf,
                         gb.chunk_qkv_half, H, qkv_dim, n_tokens, stream);
        quant_gemv_chunk(gate_w->data, gate_w->type, gpu_qi[g].q8_buf,
                         gb.chunk_z_out, H, num_v * v_dim, n_tokens, stream);
        quant_gemv_chunk(alpha_w->data, alpha_w->type, gpu_qi[g].q8_buf,
                         gb.chunk_a_proj, H, num_v, n_tokens, stream);
        quant_gemv_chunk(beta_w->data, beta_w->type, gpu_qi[g].q8_buf,
                         gb.chunk_b_proj, H, num_v, n_tokens, stream);

        // 3b. Persist each node's conv1d input (post-qkv-projection half values)
        //     so the host can commit accepted prefix's conv_state after verify,
        //     without needing to re-run the projection. The generic
        //     `chunk_qkv_half` buffer will be overwritten by later layers.
        if (tree_qkv_persist_ready && tree_qkv_persist[layer]) {
            cudaMemcpyAsync(tree_qkv_persist[layer], gb.chunk_qkv_half,
                            (size_t)n_tokens * qkv_dim * sizeof(half),
                            cudaMemcpyDeviceToDevice, stream);
        }

        // 4. Tree-mode conv1d: walks parent chain to reconstruct each node's
        //    (kw-1)-wide window. Input [n_tokens, qkv_dim] half → output
        //    [n_tokens, qkv_dim] f32. Does NOT update conv_state (we feed the
        //    pre-tree conv_state as the read-only `pre_state`).
        {
            int kw = 4;
            int threads = min(qkv_dim, 256);
            int blocks  = (qkv_dim + threads - 1) / threads;
            conv1d_update_silu_tree<8><<<blocks, threads, 0, stream>>>(
                conv_state_slot,
                gb.chunk_qkv_half,
                (float*)conv_w->data,
                parent_ids_dev,
                gb.chunk_qkv,
                qkv_dim, kw, n_tokens);
        }

        // 5. Tree-mode GDN recurrent: per-node state reload via parent_ids +
        //    persistent intermediate buffer. rec_state remains the pre-tree
        //    state (read-only here); tree_gdn_inter[layer] holds post-token
        //    states for each of the n_tokens nodes.
        //
        // Chain fast path: when the tree is a pure chain (tree_is_chain) the
        // state is strictly sequential, so gdn_recurrent_step_tree_chain keeps
        // it SMEM-resident across the whole block (no per-token global reload).
        // Bit-identical to the per-token path on a chain; ~2.7× faster on the
        // scan. Needs the >48KB SMEM opt-in (per-GPU, like gdn_chunk_step).
        // Opt out via DFLASH_GDN_CHAIN=0.
        static const bool gdn_chain =
            !(getenv("DFLASH_GDN_CHAIN") && atoi(getenv("DFLASH_GDN_CHAIN")) == 0);
        if (gdn_chain && tree_is_chain) {
            int threads   = min(v_dim, 128);
            int state_len = k_dim * v_dim;
            int gdn_smem  = (state_len + 2 * k_dim + 1 + 96) * sizeof(float);
            static bool chain_smem_set[16] = {false};
            int cur_dev = 0; cudaGetDevice(&cur_dev);
            if (cur_dev >= 0 && cur_dev < 16 && !chain_smem_set[cur_dev]) {
                cudaFuncSetAttribute((const void*)gdn_recurrent_step_tree_chain,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
                chain_smem_set[cur_dev] = true;
            }
            gdn_recurrent_step_tree_chain<<<num_v, threads, gdn_smem, stream>>>(
                gb.chunk_qkv,
                (float*)a_log_t->data,
                (float*)dt_bias_t->data,
                gb.chunk_a_proj,
                gb.chunk_b_proj,
                rec_state_slot,
                tree_gdn_inter[layer],
                gb.chunk_core_out,
                num_k, num_v, k_dim, v_dim, n_tokens);
        } else {
            int threads = min(v_dim, 128);
            int gdn_smem = (2 * k_dim + 1) * sizeof(float);
            gdn_recurrent_step_tree<<<num_v, threads, gdn_smem, stream>>>(
                gb.chunk_qkv,
                (float*)a_log_t->data,
                (float*)dt_bias_t->data,
                gb.chunk_a_proj,
                gb.chunk_b_proj,
                rec_state_slot,
                tree_gdn_inter[layer],
                parent_ids_dev,
                gb.chunk_core_out,
                num_k, num_v, k_dim, v_dim, n_tokens);
        }

        // 6. RMSNorm-gated (same per-token chunked kernel; each token is
        //    independent so no tree-mode needed here).
        {
            int total_blocks = num_v * n_tokens;
            rms_norm_gated_chunk_kernel<<<total_blocks, min(v_dim, 128),
                                          128 * sizeof(float), stream>>>(
                gb.chunk_core_out, gb.chunk_z_out, (float*)ssm_norm_w->data,
                gb.chunk_normed, num_v, v_dim, n_tokens, 1e-6f);
        }

        // 7. Batched output projection + batched residual add.
        if (out_w->type == GGML_TYPE_Q8_0) {
            gpu_qi_inter[g].quantize_chunk(gb.chunk_normed, v_total, n_tokens, stream);
            quant_gemv_chunk(out_w->data, out_w->type,
                             gpu_qi_inter[g].q8_buf, gb.chunk_proj_out,
                             v_total, H, n_tokens, stream);
            int n_elem = n_tokens * H;
            add_kernel_f32<<<(n_elem + 255) / 256, 256, 0, stream>>>(
                hidden_tree, gb.chunk_proj_out, n_elem);
        } else {
            for (int tok = 0; tok < n_tokens; tok++) {
                half* normed_t = gb.chunk_normed + (size_t)tok * v_total;
                half* proj_t   = gb.chunk_proj_out + (size_t)tok * H;
                gpu_qi_inter[g].quantize(normed_t, v_total, stream);
                quant_gemv(out_w->data, out_w->type, normed_t, proj_t,
                           v_total, H, &gpu_qi_inter[g], stream);
                add_kernel_f32<<<(H + 255) / 256, 256, 0, stream>>>(
                    hidden_tree + (size_t)tok * H, proj_t, H);
            }
        }
    }

    // ============ DDTree: MLP forward over n_tokens tree nodes ============
    // MLP is stateless per token, so tree-mode is just forward_mlp_chunk —
    // parent_ids is unused. Keeping the API parallel to forward_gdn_tree so
    // main.cu can call forward_X_tree uniformly.
    void forward_mlp_tree(int layer, float* hidden_tree, int n_tokens,
                          const int* parent_ids_dev, cudaStream_t stream) {
        (void)parent_ids_dev;
        forward_mlp_chunk(layer, hidden_tree, n_tokens, stream);
    }

    // ============ DDTree: attention forward over n_tokens tree nodes =========
    // Sequential per-node forward_attn. Each node t lands in KV cache slot
    // (pos_base + t), uses RoPE position (pos_base + depth[t]), and attends
    // only its ancestor chain within the tree window — other tree slots
    // (siblings, or nodes from parallel branches) are masked to -INF. For
    // chain trees the mask is a noop because every slot < t is also an
    // ancestor, but branching trees require this path for correctness.
    // Requires upload_parent_ids() to have been called for the current tree.
    void forward_attn_tree(int layer, float* hidden_tree, int pos_base, int n_tokens,
                           const int* parent_ids_dev, cudaStream_t stream, int slot = 0) {
        (void)parent_ids_dev;
        int H = cfg.hidden_size;

        // Chain fast path: route through forward_attn_step_batched. Each of the
        // n_tokens chain nodes goes to the same KV slot at consecutive positions
        // pos_base+0..pos_base+n-1; the batched per-slot attention seq_len cutoff
        // (pos+1) reproduces the chain ancestor mask exactly. Projections (Q/K/V,
        // RoPE, output) are batched across all nodes = the win. fp16 KV only
        // (the batched scatter path); TQ/Q8 fall through to the per-token loop.
        static const bool batch_chain =
            !(getenv("DFLASH_ATTN_BATCH") && atoi(getenv("DFLASH_ATTN_BATCH")) == 0);
        // TQ KV chains can now batch too: forward_attn_step_batched runs ONE
        // multi-query FA (flash_attn_chunk_fused_split_tq3, sub_n=N) over the TQ
        // cache instead of N per-token forward_attn calls → KV read once, not N×.
        // (DFLASH_VERIFY_BATCHED=0 reverts to the per-token path.) Q8 KV still
        // routes per-token.
        static const bool tq_verify_batched_tree =
            !(getenv("DFLASH_VERIFY_BATCHED") && atoi(getenv("DFLASH_VERIFY_BATCHED")) == 0);
        bool kv_ok_for_batch = (!use_turboquant || tq_verify_batched_tree) && !use_q8_kv;
        if (batch_chain && tree_is_chain && kv_ok_for_batch
            && tree_chain_pos_ready && n_tokens > 1) {
            int g = gpu->layer_gpu[layer];
            int base_off = (int)kv_slot_offset(slot);
            // Reuse host scratch sized to budget (set once; positions shift per
            // iter so we refill, but it's tiny — n_tokens ints).
            tree_chain_slot_ids_h.assign(n_tokens, slot);
            tree_chain_pos_h.resize(n_tokens);
            tree_chain_dst_h.resize(n_tokens);
            for (int i = 0; i < n_tokens; i++) {
                tree_chain_pos_h[i] = pos_base + i;
                tree_chain_dst_h[i] = base_off + pos_base + i;
            }
            cudaSetDevice(g);
            cudaMemcpyAsync(tree_chain_pos_d[g], tree_chain_pos_h.data(),
                            n_tokens * sizeof(int), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(tree_chain_dst_d[g], tree_chain_dst_h.data(),
                            n_tokens * sizeof(int), cudaMemcpyHostToDevice, stream);
            forward_attn_step_batched(layer, hidden_tree, n_tokens,
                                      tree_chain_slot_ids_h.data(),
                                      tree_chain_pos_h.data(),
                                      tree_chain_dst_d[g], tree_chain_pos_d[g], stream);
            return;
        }

        for (int t = 0; t < n_tokens; t++) {
            int rope_pos = pos_base + tree_depth_host[t];
            int kv_slot  = pos_base + t;
            uint32_t mbits = tree_ancestor_bits_host[t];
            forward_attn(layer, hidden_tree + (size_t)t * H, rope_pos, stream,
                         /*external_proj=*/false,
                         /*slot_pos=*/kv_slot,
                         /*mask_start=*/pos_base,
                         /*mask_len=*/n_tokens,
                         /*mask_bits=*/mbits,
                         /*slot=*/slot);
        }
    }

    // ============ Batched gen-step forward for attn layers =================
    // Process one new token from each of N slots in a single forward — the
    // throughput core of continuous batching. Each slot has its own logical
    // position (`slot_pos_host[i]`) and KV virtual partition (`slot_ids_host[i]`).
    // Projections (RMSNorm + Q/K/V + output) are batched across N rows so
    // weight memory traffic is amortized; the attention compute itself loops
    // per-slot since each slot has its own KV range. For long-running gen
    // (the common workload) MLP + projections dominate, so the batched
    // GEMVs dominate the speedup.
    //
    // hidden_batch: [N, H] FP32 in/out, on the layer's GPU.
    // slot_ids_dev / slot_pos_dev: int32 device arrays length N. The host
    // also passes the same data so we can issue per-slot kernel launches
    // without a device→host round-trip.
    void forward_attn_step_batched(int layer,
                                   float* hidden_batch,
                                   int N,
                                   const int* slot_ids_host,
                                   const int* slot_pos_host,
                                   const int* dst_kv_pos_dev,   // [N] = slot*slot_max_seq + pos
                                   const int* slot_pos_dev,     // [N] (for RoPE)
                                   cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& ab = attn_bufs[g];
        auto& buf = bufs[g];
        int H = cfg.hidden_size;
        int num_q  = cfg.num_q_heads;
        int num_kv = cfg.num_kv_heads;
        int hd     = cfg.head_dim;
        int kv_dim = num_kv * hd;
        int total_qg = num_q * hd;
        float eps = cfg.rms_norm_eps;
        float scale = 1.0f / sqrtf((float)hd);

        auto* norm_w   = t(blk(layer, "attn_norm.weight"));
        auto* q_w      = t(blk(layer, "attn_q.weight"));
        auto* k_w      = t(blk(layer, "attn_k.weight"));
        auto* v_w      = t(blk(layer, "attn_v.weight"));
        auto* q_norm_w = t(blk(layer, "attn_q_norm.weight"));
        auto* k_norm_w = t(blk(layer, "attn_k_norm.weight"));
        auto* o_w      = t(blk(layer, "attn_output.weight"));

        // Fallback to per-slot per-token forward if any quant type isn't
        // Q8_0 (the batched quant_gemv_chunk only supports Q8_0).
        bool can_batch = (q_w->type == GGML_TYPE_Q8_0)
                      && (k_w->type == GGML_TYPE_Q8_0)
                      && (v_w->type == GGML_TYPE_Q8_0)
                      && (o_w->type == GGML_TYPE_Q8_0);
        if (!can_batch || N <= 0) {
            for (int i = 0; i < N; i++) {
                forward_attn(layer, hidden_batch + (size_t)i * H,
                             slot_pos_host[i], stream,
                             /*external_proj=*/false, /*slot_pos=*/-1,
                             /*mask_start=*/-1, /*mask_len=*/0,
                             /*mask_bits=*/0xffffffffu,
                             /*slot=*/slot_ids_host[i]);
            }
            return;
        }

        int q_out_dim = q_w->dims[1];

        // Lazy alloc the chunk buffers if forward_attn_chunk hasn't yet.
        if (!ab.attn_chunk_q) {
            cudaMalloc(&ab.attn_chunk_q,        (size_t)CHUNK_SIZE * q_out_dim * sizeof(half));
            cudaMalloc(&ab.attn_chunk_k,        (size_t)CHUNK_SIZE * kv_dim    * sizeof(half));
            cudaMalloc(&ab.attn_chunk_v,        (size_t)CHUNK_SIZE * kv_dim    * sizeof(half));
            cudaMalloc(&ab.attn_chunk_q_post,   (size_t)CHUNK_SIZE * total_qg  * sizeof(half));
            cudaMalloc(&ab.attn_chunk_gate,     (size_t)CHUNK_SIZE * total_qg  * sizeof(half));
            cudaMalloc(&ab.attn_chunk_scores,   (size_t)ATTN_NB * num_q * kv_max_seq * sizeof(float));
            cudaMalloc(&ab.attn_chunk_out,      (size_t)CHUNK_SIZE * total_qg  * sizeof(half));
            cudaMalloc(&ab.attn_chunk_oproj,    (size_t)CHUNK_SIZE * H         * sizeof(half));
            constexpr int K_SPLITS_MAX = 16;
            cudaMalloc(&ab.attn_split_m, (size_t)num_q * ATTN_NB * K_SPLITS_MAX * sizeof(float));
            cudaMalloc(&ab.attn_split_l, (size_t)num_q * ATTN_NB * K_SPLITS_MAX * sizeof(float));
            cudaMalloc(&ab.attn_split_o, (size_t)num_q * ATTN_NB * K_SPLITS_MAX * hd * sizeof(float));
        }

        // 1. Batched RMSNorm
        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w(buf.mlp_chunk_norm, hidden_batch, (float*)norm_w->data,
                                N, H, eps, stream);
        } else {
            rms_norm_f32in(buf.mlp_chunk_norm, hidden_batch, (half*)norm_w->data,
                           N, H, eps, stream);
        }

        // 2. Quantize all N × H normed values
        gpu_qi[g].quantize_chunk(buf.mlp_chunk_norm, H, N, stream);

        // 3. Batched Q/K/V projections — 3 GEMV calls instead of 3*N
        quant_gemv_chunk(q_w->data, q_w->type, gpu_qi[g].q8_buf, ab.attn_chunk_q,
                         H, q_out_dim, N, stream);
        quant_gemv_chunk(k_w->data, k_w->type, gpu_qi[g].q8_buf, ab.attn_chunk_k,
                         H, kv_dim, N, stream);
        quant_gemv_chunk(v_w->data, v_w->type, gpu_qi[g].q8_buf, ab.attn_chunk_v,
                         H, kv_dim, N, stream);

        // 4. Batched deinterleave Q/gate, head-RMS, then per-slot RoPE.
        int rope_dim = rope.rope_dim;
        int half_rope = rope_dim / 2;
        int tn = std::min(hd, 128);
        {
            dim3 deint_grid((total_qg + 255) / 256, N);
            deinterleave_qg_kernel_chunk<<<deint_grid, 256, 0, stream>>>(
                ab.attn_chunk_q, ab.attn_chunk_q_post, ab.attn_chunk_gate,
                num_q, hd, N);

            static const bool skip_qk_norm_c = getenv("SKIP_QK_NORM") != nullptr;
            if (!skip_qk_norm_c) {
                dim3 q_rms_grid(num_q,  N);
                head_rms_norm_kernel_chunk<<<q_rms_grid, tn, tn * sizeof(float), stream>>>(
                    ab.attn_chunk_q_post, (float*)q_norm_w->data, num_q, hd, eps, N);
                dim3 k_rms_grid(num_kv, N);
                head_rms_norm_kernel_chunk<<<k_rms_grid, tn, tn * sizeof(float), stream>>>(
                    ab.attn_chunk_k, (float*)k_norm_w->data, num_kv, hd, eps, N);
            }

            dim3 rope_q_grid((num_q  * half_rope + 255) / 256, N);
            dim3 rope_k_grid((num_kv * half_rope + 255) / 256, N);
            apply_rope_kernel_batched<<<rope_q_grid, 256, 0, stream>>>(
                ab.attn_chunk_q_post, rope.sin_table(g), rope.cos_table(g),
                slot_pos_dev, num_q,  hd, rope_dim, N);
            apply_rope_kernel_batched<<<rope_k_grid, 256, 0, stream>>>(
                ab.attn_chunk_k, rope.sin_table(g), rope.cos_table(g),
                slot_pos_dev, num_kv, hd, rope_dim, N);
        }

        // 5/6. KV write + attention.
        //
        // TQ KV (the 256K config): write the N new (post-RoPE) K/V into the TQ
        // cache at the chain's consecutive positions, then run ONE multi-query
        // FlashAttention over the TQ cache. flash_attn_chunk_fused_split_tq3
        // decodes the 3-bit cache cooperatively in shared memory (no fp16 scratch,
        // no history re-dequant) and self-limits each chain query t_idx to
        // [0..pos_base+t_idx] (active_end_t = abs_pos+1), so the KV cache is read
        // ONCE for all N queries instead of N× (the old per-token forward_attn
        // loop). Same kernel as the per-token FA path (sub_n=N vs sub_n=1) → the
        // 27B verify is greedy-equivalent. DFLASH_VERIFY_BATCHED=0 reverts.
        static const bool tq_verify_batched =
            !(getenv("DFLASH_VERIFY_BATCHED") && atoi(getenv("DFLASH_VERIFY_BATCHED")) == 0);
        if (use_turboquant) {
            if (!tq_verify_batched) {
                // Per-token fallback for TQ (legacy; reads KV N× per iteration).
                for (int i = 0; i < N; i++) {
                    forward_attn(layer, hidden_batch + (size_t)i * H,
                                 slot_pos_host[i], stream,
                                 /*external_proj=*/false, /*slot_pos=*/-1,
                                 /*mask_start=*/-1, /*mask_len=*/0,
                                 /*mask_bits=*/0xffffffffu,
                                 /*slot=*/slot_ids_host[i]);
                }
                return;
            }
            int pos_base = slot_pos_host[0];          // chain → consecutive positions
            int slot_s   = slot_ids_host[0];
            auto& tq = tq_kv_caches[layer];
            int bpt = tq.blocks_per_token;
            int new_blocks = N * bpt;
            size_t slot_blk_off = kv_slot_offset(slot_s) * (size_t)bpt;
            block_tq3* tq_k_slot = tq.k + slot_blk_off;
            block_tq3* tq_v_slot = tq.v + slot_blk_off;
            // Quantize the N new post-RoPE K/V into the TQ cache (positions
            // [pos_base..pos_base+N)); mirrors the chunked-prefill KV write.
            tq3_quantize_kernel<<<(new_blocks + 31) / 32, 32, 0, stream>>>(
                ab.attn_chunk_k, &tq_k_slot[(size_t)pos_base * bpt], new_blocks);
            tq3_quantize_kernel<<<(new_blocks + 31) / 32, 32, 0, stream>>>(
                ab.attn_chunk_v, &tq_v_slot[(size_t)pos_base * bpt], new_blocks);

            constexpr int HD = 256, BM = 32, BLOCK = 256, K_SPLITS = 8;
            int active_end_max = pos_base + N;
            int smem24 = 6 * HD * sizeof(half) + 2 * BM * HD * sizeof(half) + 6 * BM * sizeof(float);
            int smem16 = 4 * HD * sizeof(half) + 2 * BM * HD * sizeof(half) + 4 * BM * sizeof(float);
            dim3 fg(num_kv, N, K_SPLITS);
            dim3 mg(num_kv, N);
            // Optional sink+window sparse verify (DFLASH_VERIFY_SPARSE=1). Makes
            // the verify TQ-decode O(sink+window) instead of O(context) so the
            // long-context gen curve flattens. The drafter is windowed to ~4096,
            // so window>=that verifies its predictions over the same span. Lossy
            // for far-middle-context tasks → default OFF (dense). Tunable:
            // DFLASH_VERIFY_SINK (tok, def 256), DFLASH_VERIFY_WINDOW (tok, def 4096).
            static const int vsink = []{ const char* e=getenv("DFLASH_VERIFY_SPARSE");
                if (!e || atoi(e)==0) return 0; const char* s=getenv("DFLASH_VERIFY_SINK"); return s?atoi(s):256; }();
            static const int vwin  = []{ const char* e=getenv("DFLASH_VERIFY_SPARSE");
                if (!e || atoi(e)==0) return 0; const char* w=getenv("DFLASH_VERIFY_WINDOW"); return w?atoi(w):4096; }();
            // ===== MS-block-sparse verify (DFLASH_VERIFY_BLOCKSPARSE=1) =====
            // Content-selected top-k 64-token blocks (same MS selector the prefill
            // path uses) → O(top_k·64) TQ-decodes per verify instead of O(context),
            // keeping the needle blocks sink+window would drop. Per-layer MS K-pool
            // (verify_kpool) is grown incrementally: rebuild [first-unfrozen-block ..
            // newest], then freeze all complete blocks.
            static const bool  vbs_on   = []{ const char* e=getenv("DFLASH_VERIFY_BLOCKSPARSE"); return e && atoi(e)!=0; }();
            static const float vbs_bud  = []{ const char* e=getenv("DFLASH_VERIFY_BUDGET"); return e?(float)atof(e):0.10f; }();
            static const int   vbs_tk   = []{ const char* e=getenv("DFLASH_VERIFY_TOPK"); return e?atoi(e):0; }();
            static const float vbs_beta = []{ const char* e=getenv("MINF_MS_BETA"); return e?(float)atof(e):0.5f; }();
            if (vbs_on) {
                constexpr int BLOCK_N = 64;
                int total_kv   = pos_base + N;                     // causal end
                int cur_blocks = (total_kv + BLOCK_N - 1) / BLOCK_N;
                auto& vp = verify_kpool[layer];
                if (vp.k_pool == nullptr) {
                    vp.n_blocks_max = (slot_capacity(slot_s) + BLOCK_N - 1) / BLOCK_N;
                    size_t pe = (size_t)num_kv * vp.n_blocks_max * HD;
                    cudaMalloc(&vp.k_pool,     pe * sizeof(half));
                    cudaMalloc(&vp.k_pool_max, pe * sizeof(half));
                    vp.pooled_pos = 0;
                }
                if (!ab.sparse_block_index) {
                    cudaMalloc(&ab.sparse_block_index,
                               (size_t)num_kv * ATTN_NB * 64 * sizeof(int));
                }
                // (re)build pool from first not-yet-frozen block to the newest.
                int blk_lo = vp.pooled_pos / BLOCK_N;
                if (cur_blocks > blk_lo) {
                    dim3 pg(num_kv, cur_blocks - blk_lo);
                    int pool_smem = BLOCK_N * HD * sizeof(half);
                    tq3_kpool_ms_update_kern<HD, 256><<<pg, 256, pool_smem, stream>>>(
                        tq_k_slot, vp.k_pool, vp.k_pool_max, num_kv, total_kv,
                        blk_lo, vp.n_blocks_max, BLOCK_N);
                }
                vp.pooled_pos = (cur_blocks > 0 ? cur_blocks - 1 : 0) * BLOCK_N;
                // top-k blocks (budget-scaled, capped at the 64-block buffer).
                int top_k = vbs_tk > 0 ? vbs_tk : (int)(cur_blocks * vbs_bud);
                if (top_k < 1) top_k = 1;
                if (top_k > 64) top_k = 64;
                if (top_k > cur_blocks) top_k = cur_blocks;
                dim3 bi_grid(num_kv, N);
                size_t bi_smem = HD * sizeof(half) + (size_t)cur_blocks * sizeof(float);
                if (num_q == 24) {
                    constexpr int GQA = 6;
                    build_block_index_ms_kern<HD, GQA, BLOCK><<<bi_grid, BLOCK, bi_smem, stream>>>(
                        ab.attn_chunk_q_post, vp.k_pool, vp.k_pool_max, ab.sparse_block_index,
                        num_q, num_kv, N, ATTN_NB, pos_base, top_k, BLOCK_N, cur_blocks,
                        vp.n_blocks_max, vbs_beta);
                    flash_attn_chunk_block_sparse_split_tq3<HD, GQA, BM, BLOCK, K_SPLITS>
                        <<<fg, BLOCK, smem24, stream>>>(
                            ab.attn_chunk_q_post, tq_k_slot, tq_v_slot, ab.sparse_block_index,
                            ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                            num_q, num_kv, pos_base, N, ATTN_NB, scale, top_k, BLOCK_N);
                    flash_attn_split_merge<HD, GQA, BLOCK, K_SPLITS><<<mg, BLOCK, 0, stream>>>(
                        ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                        ab.attn_chunk_out, num_q, N, ATTN_NB);
                } else {
                    constexpr int GQA = 4;
                    build_block_index_ms_kern<HD, GQA, BLOCK><<<bi_grid, BLOCK, bi_smem, stream>>>(
                        ab.attn_chunk_q_post, vp.k_pool, vp.k_pool_max, ab.sparse_block_index,
                        num_q, num_kv, N, ATTN_NB, pos_base, top_k, BLOCK_N, cur_blocks,
                        vp.n_blocks_max, vbs_beta);
                    flash_attn_chunk_block_sparse_split_tq3<HD, GQA, BM, BLOCK, K_SPLITS>
                        <<<fg, BLOCK, smem16, stream>>>(
                            ab.attn_chunk_q_post, tq_k_slot, tq_v_slot, ab.sparse_block_index,
                            ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                            num_q, num_kv, pos_base, N, ATTN_NB, scale, top_k, BLOCK_N);
                    flash_attn_split_merge<HD, GQA, BLOCK, K_SPLITS><<<mg, BLOCK, 0, stream>>>(
                        ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                        ab.attn_chunk_out, num_q, N, ATTN_NB);
                }
            } else if (num_q == 24) {
                constexpr int GQA = 6;
                flash_attn_chunk_fused_split_tq3<HD, GQA, BM, BLOCK, K_SPLITS>
                    <<<fg, BLOCK, smem24, stream>>>(
                        ab.attn_chunk_q_post, tq_k_slot, tq_v_slot,
                        ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                        num_q, num_kv, pos_base, N, ATTN_NB, active_end_max, scale, vsink, vwin);
                flash_attn_split_merge<HD, GQA, BLOCK, K_SPLITS>
                    <<<mg, BLOCK, 0, stream>>>(
                        ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                        ab.attn_chunk_out, num_q, N, ATTN_NB);
            } else {
                constexpr int GQA = 4;
                flash_attn_chunk_fused_split_tq3<HD, GQA, BM, BLOCK, K_SPLITS>
                    <<<fg, BLOCK, smem16, stream>>>(
                        ab.attn_chunk_q_post, tq_k_slot, tq_v_slot,
                        ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                        num_q, num_kv, pos_base, N, ATTN_NB, active_end_max, scale, vsink, vwin);
                flash_attn_split_merge<HD, GQA, BLOCK, K_SPLITS>
                    <<<mg, BLOCK, 0, stream>>>(
                        ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                        ab.attn_chunk_out, num_q, N, ATTN_NB);
            }
            // → attention output in ab.attn_chunk_out; fall through to gate/oproj.
        } else {
            // fp16 KV path: scatter K/V then per-slot attention into attn_chunk_out.
            auto& kv = kv_caches[layer];
            {
                dim3 sc_grid((kv_dim + 255) / 256, N);
                scatter_kv_kernel<<<sc_grid, 256, 0, stream>>>(
                    ab.attn_chunk_k, ab.attn_chunk_v,
                    kv.k, kv.v, dst_kv_pos_dev, kv_dim, N);
            }
            // Per-slot attention compute. Each slot's Q attends to KV[0..pos].
            for (int i = 0; i < N; i++) {
                int slot_i  = slot_ids_host[i];
                int pos_i   = slot_pos_host[i];
                int seq_len = pos_i + 1;
                half*  q_buf_i = ab.attn_chunk_q_post + (size_t)i * total_qg;
                float* scores_i = ab.attn_chunk_scores;  // single slot at a time, reuse base
                half*  out_i   = ab.attn_chunk_out + (size_t)i * total_qg;
                half*  k_slot  = kv.k + kv_slot_offset(slot_i) * kv_dim;
                half*  v_slot  = kv.v + kv_slot_offset(slot_i) * kv_dim;

                dim3 score_grid = score_pos_grid(num_q, seq_len);
                attn_score_kernel_h<<<score_grid, std::min(hd, 256), 0, stream>>>(
                    q_buf_i, k_slot, scores_i,
                    num_q, num_kv, hd, seq_len, scale);
                int st = 1; while (st < seq_len && st < 256) st <<= 1;
                softmax_kernel<<<num_q, st, st * sizeof(float), stream>>>(
                    scores_i, num_q, seq_len);
                attn_value_kernel_h<<<num_q, std::min(hd, 256), 0, stream>>>(
                    scores_i, v_slot, out_i, num_q, num_kv, hd, seq_len);
            }
        }

        // 7. Batched output gate (sigmoid)
        {
            dim3 gate_grid((total_qg + 255) / 256, N);
            apply_gate_sigmoid_chunk<<<gate_grid, 256, 0, stream>>>(
                ab.attn_chunk_out, ab.attn_chunk_gate, total_qg, N);
        }

        // 8. Batched output projection: quantize attn_chunk_out [N, total_qg]
        gpu_qi_inter[g].quantize_chunk(ab.attn_chunk_out, total_qg, N, stream);
        quant_gemv_chunk(o_w->data, o_w->type,
                         gpu_qi_inter[g].q8_buf,
                         ab.attn_chunk_oproj,
                         total_qg, H, N, stream);

        // 9. Residual add: hidden_batch += attn_chunk_oproj (flat N*H length)
        {
            int total = N * H;
            add_kernel_f32<<<(total + 255) / 256, 256, 0, stream>>>(
                hidden_batch, ab.attn_chunk_oproj, total);
        }
    }

    // ============ Batched gen-step forward for GDN layers ==================
    // Mirror of forward_attn_step_batched for GDN: batched RMSNorm + the four
    // projections (qkv, gate, alpha, beta), then per-slot conv1d update +
    // recurrent step + gated norm (since each slot has its own conv_state and
    // rec_state), then batched output projection + residual.
    void forward_gdn_step_batched(int layer,
                                  float* hidden_batch,
                                  int N,
                                  const int* slot_ids_host,
                                  cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& buf = bufs[g];
        auto& gb = gdn_bufs[g];
        int H = cfg.hidden_size;
        int num_k = cfg.linear_k_heads;
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        int qkv_dim = 2 * num_k * k_dim + num_v * v_dim;
        int v_total = num_v * v_dim;

        auto* norm_w  = t(blk(layer, "attn_norm.weight"));
        auto* qkv_w   = t(blk(layer, "attn_qkv.weight"));
        auto* gate_w  = t(blk(layer, "attn_gate.weight"));
        auto* alpha_w = t(blk(layer, "ssm_alpha.weight"));
        auto* beta_w  = t(blk(layer, "ssm_beta.weight"));
        auto* conv_w  = t(blk(layer, "ssm_conv1d.weight"));
        auto* a_log_t = t(blk(layer, "ssm_a"));
        auto* dt_bias_t = t(blk(layer, "ssm_dt.bias"));
        auto* ssm_norm_w = t(blk(layer, "ssm_norm.weight"));
        auto* out_w   = t(blk(layer, "ssm_out.weight"));

        bool can_batch = (qkv_w->type   == GGML_TYPE_Q8_0)
                      && (gate_w->type  == GGML_TYPE_Q8_0)
                      && (alpha_w->type == GGML_TYPE_Q8_0)
                      && (beta_w->type  == GGML_TYPE_Q8_0)
                      && (out_w->type   == GGML_TYPE_Q8_0);
        if (!can_batch || N <= 0) {
            for (int i = 0; i < N; i++) {
                forward_gdn(layer, hidden_batch + (size_t)i * H, stream, slot_ids_host[i]);
            }
            return;
        }

        // 1. Batched RMSNorm
        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w(gb.chunk_norm_out, hidden_batch, (float*)norm_w->data,
                                N, H, cfg.rms_norm_eps, stream);
        } else {
            rms_norm_f32in(gb.chunk_norm_out, hidden_batch, (half*)norm_w->data,
                           N, H, cfg.rms_norm_eps, stream);
        }
        gpu_qi[g].quantize_chunk(gb.chunk_norm_out, H, N, stream);

        // 2. Batched projections — qkv (largest), gate, alpha, beta
        if (!gb.chunk_qkv_half) {
            cudaMalloc(&gb.chunk_qkv_half, (size_t)CHUNK_SIZE * qkv_dim * sizeof(half));
        }
        quant_gemv_chunk(qkv_w->data, qkv_w->type, gpu_qi[g].q8_buf,
                         gb.chunk_qkv_half, H, qkv_dim, N, stream);
        quant_gemv_chunk(gate_w->data, gate_w->type, gpu_qi[g].q8_buf,
                         gb.chunk_z_out, H, num_v * v_dim, N, stream);
        quant_gemv_chunk(alpha_w->data, alpha_w->type, gpu_qi[g].q8_buf,
                         gb.chunk_a_proj, H, num_v, N, stream);
        quant_gemv_chunk(beta_w->data, beta_w->type, gpu_qi[g].q8_buf,
                         gb.chunk_b_proj, H, num_v, N, stream);

        // 3. Per-slot conv1d update + recurrent step + gated RMSNorm.
        //    The state-bound parts must be per-slot since each slot has its
        //    own conv_state and rec_state — fusing across slots would need
        //    a new SMEM-resident-state-batched kernel (Phase D).
        int kw = 4;
        int threads_conv = std::min(qkv_dim, 256);
        int blocks_conv  = (qkv_dim + threads_conv - 1) / threads_conv;
        int gdn_smem = (2 * cfg.linear_k_dim + 1) * sizeof(float);
        for (int i = 0; i < N; i++) {
            int slot_i = slot_ids_host[i];
            // Convert this token's qkv fp16 → fp32 conv input scratch
            half*  qkv_half_i = gb.chunk_qkv_half + (size_t)i * qkv_dim;
            float* conv_in_i  = gb.chunk_qkv      + (size_t)i * qkv_dim;
            // Reinterpret: conv1d_update_silu expects half qkv_in, writes
            // fp32 conv_out. So we feed qkv_half_i directly.
            conv1d_update_silu<<<blocks_conv, threads_conv, 0, stream>>>(
                gdn_conv_slot(layer, slot_i),
                qkv_half_i,
                (float*)conv_w->data,
                conv_in_i,   // overwrite our fp32 slot with conv1d output
                qkv_dim, kw);
            half* a_proj_i = gb.chunk_a_proj  + (size_t)i * num_v;
            half* b_proj_i = gb.chunk_b_proj  + (size_t)i * num_v;
            half* core_i   = gb.chunk_core_out + (size_t)i * v_total;
            launch_gdn_recurrent_step(num_v, std::min(v_dim, 128), gdn_smem, stream,
                conv_in_i,
                (float*)a_log_t->data,
                (float*)dt_bias_t->data,
                a_proj_i,
                b_proj_i,
                gdn_rec_slot(layer, slot_i),
                core_i,
                num_k, num_v, k_dim, v_dim);
            half* z_i      = gb.chunk_z_out   + (size_t)i * v_total;
            half* normed_i = gb.chunk_normed  + (size_t)i * v_total;
            rms_norm_gated_kernel<<<num_v, std::min(v_dim, 128), 128 * sizeof(float), stream>>>(
                core_i, z_i, (float*)ssm_norm_w->data, normed_i,
                num_v, v_dim, 1e-6f);
        }

        // 4. Batched output projection + residual.
        gpu_qi_inter[g].quantize_chunk(gb.chunk_normed, num_v * v_dim, N, stream);
        quant_gemv_chunk(out_w->data, out_w->type, gpu_qi_inter[g].q8_buf,
                         gb.chunk_proj_out, num_v * v_dim, H, N, stream);
        {
            int total = N * H;
            add_kernel_f32<<<(total + 255) / 256, 256, 0, stream>>>(
                hidden_batch, gb.chunk_proj_out, total);
        }
    }

    // ============ Chunked attention forward (process N tokens) ============
    // For attention layers in prompt phase: process tokens sequentially through
    // forward_attn (KV cache requires sequential append). hidden_chunk is FP32.
    void forward_attn_chunk(int layer, float* hidden_chunk, int start_pos, int n_tokens, cudaStream_t stream, int slot = 0) {
        int g = gpu->layer_gpu[layer];
        auto& ab = attn_bufs[g];
        auto& buf = bufs[g];
        int H = cfg.hidden_size;
        int num_q  = cfg.num_q_heads;
        int num_kv = cfg.num_kv_heads;
        int hd     = cfg.head_dim;
        int kv_dim = num_kv * hd;
        int total_qg = num_q * hd;
        float eps = cfg.rms_norm_eps;
        float scale = 1.0f / sqrtf((float)hd);
        // Continuous batching: slot KV offset (0 for single-slot, no behavior change).
        size_t slot_kv_off = kv_slot_offset(slot) * kv_dim;
        size_t slot_pos_off = kv_slot_offset(slot);

        auto* norm_w   = t(blk(layer, "attn_norm.weight"));
        auto* q_w      = t(blk(layer, "attn_q.weight"));
        auto* k_w      = t(blk(layer, "attn_k.weight"));
        auto* v_w      = t(blk(layer, "attn_v.weight"));
        auto* q_norm_w = t(blk(layer, "attn_q_norm.weight"));
        auto* k_norm_w = t(blk(layer, "attn_k_norm.weight"));
        auto* o_w      = t(blk(layer, "attn_output.weight"));
        if (!norm_w || !q_w || !k_w || !v_w || !o_w) {
            for (int tt = 0; tt < n_tokens; tt++)
                forward_attn(layer, hidden_chunk + (size_t)tt * H, start_pos + tt, stream,
                             /*external_proj=*/false, /*slot_pos=*/-1,
                             /*mask_start=*/-1, /*mask_len=*/0,
                             /*mask_bits=*/0xffffffffu, slot);
            return;
        }
        int q_out_dim = q_w->dims[1];
        // Gated attention (Qwopus3.6 hybrid) bakes a Q-gate into attn_q so
        // q_out_dim = 2*total_qg; Qwen3 dense (embed/reranker) has no gate
        // (q_out_dim = total_qg). Mirror forward_attn's detection so the
        // chunked path is correct for BOTH — without this the dense models
        // get the gated deinterleave + sigmoid and produce garbage.
        const bool has_q_gate = (q_out_dim == 2 * total_qg);

        // TQ path: dequant historical [0..start_pos-1] from the TQ cache into
        // the per-GPU fp16 scratch once per chunk, append the new fp16 K/V at
        // [start_pos:start_pos+n_tokens], and feed stock chunked attention
        // kernels with the scratch in place of kv.k / kv.v. Avoids the
        // O(seq²) bulk-dequant-per-token cost the old per-token fallback paid.
        bool can_batch_attn = (q_w->type == GGML_TYPE_Q8_0)
                           && (k_w->type == GGML_TYPE_Q8_0)
                           && (v_w->type == GGML_TYPE_Q8_0)
                           && (o_w->type == GGML_TYPE_Q8_0);
        if (!can_batch_attn) {
            for (int tt = 0; tt < n_tokens; tt++)
                forward_attn(layer, hidden_chunk + (size_t)tt * H, start_pos + tt, stream,
                             /*external_proj=*/false, /*slot_pos=*/-1,
                             /*mask_start=*/-1, /*mask_len=*/0,
                             /*mask_bits=*/0xffffffffu, slot);
            return;
        }

        // Lazy alloc per-GPU chunked attention buffers.
        if (!ab.attn_chunk_q) {
            cudaMalloc(&ab.attn_chunk_q,        (size_t)CHUNK_SIZE * q_out_dim * sizeof(half));
            cudaMalloc(&ab.attn_chunk_k,        (size_t)CHUNK_SIZE * kv_dim    * sizeof(half));
            cudaMalloc(&ab.attn_chunk_v,        (size_t)CHUNK_SIZE * kv_dim    * sizeof(half));
            cudaMalloc(&ab.attn_chunk_q_post,   (size_t)CHUNK_SIZE * total_qg  * sizeof(half));
            cudaMalloc(&ab.attn_chunk_gate,     (size_t)CHUNK_SIZE * total_qg  * sizeof(half));
            cudaMalloc(&ab.attn_chunk_scores,   (size_t)ATTN_NB * num_q * kv_max_seq * sizeof(float));
            cudaMalloc(&ab.attn_chunk_out,      (size_t)CHUNK_SIZE * total_qg  * sizeof(half));
            cudaMalloc(&ab.attn_chunk_oproj,    (size_t)CHUNK_SIZE * H         * sizeof(half));
            constexpr int K_SPLITS_MAX = 16;
            cudaMalloc(&ab.attn_split_m, (size_t)num_q * ATTN_NB * K_SPLITS_MAX * sizeof(float));
            cudaMalloc(&ab.attn_split_l, (size_t)num_q * ATTN_NB * K_SPLITS_MAX * sizeof(float));
            cudaMalloc(&ab.attn_split_o, (size_t)num_q * ATTN_NB * K_SPLITS_MAX * hd * sizeof(float));
        }

        // ATTN_DUMP_LAYER + ATTN_DUMP_POS dump for an arbitrary token in
        // this chunk. POS is the absolute token position; if it falls inside
        // [start_pos, start_pos + n_tokens) we dump that t_idx's sub-step
        // outputs (otherwise no-op).
        static const int attn_dump_layer = []{ const char* e=getenv("ATTN_DUMP_LAYER"); return e?atoi(e):-1; }();
        static const int attn_dump_pos   = []{ const char* e=getenv("ATTN_DUMP_POS");   return e?atoi(e):-1; }();
        int target_t = attn_dump_pos - start_pos;
        bool attn_do_dump = (attn_dump_layer == layer)
                          && (target_t >= 0 && target_t < n_tokens);
        auto attn_dump_h = [&](const char* tag, const half* buf, int n=256) {
            if (!attn_do_dump) return;
            cudaSetDevice(g); cudaDeviceSynchronize();
            std::vector<half> h(n); cudaMemcpy(h.data(), buf, n*sizeof(half), cudaMemcpyDeviceToHost);
            double sum_abs = 0.0; for (int i=0;i<n;i++) sum_abs += fabs((double)__half2float(h[i]));
            fprintf(stderr, "[ATTN-CK L%d %-14s sa=%.6f]", layer, tag, sum_abs);
            for (int i=0;i<8;i++) fprintf(stderr, " %.5f", __half2float(h[i]));
            fprintf(stderr, "\n"); fflush(stderr);
        };
        auto attn_dump_f = [&](const char* tag, const float* buf, int n=256) {
            if (!attn_do_dump) return;
            cudaSetDevice(g); cudaDeviceSynchronize();
            std::vector<float> h(n); cudaMemcpy(h.data(), buf, n*sizeof(float), cudaMemcpyDeviceToHost);
            double sum_abs = 0.0; for (int i=0;i<n;i++) sum_abs += fabs((double)h[i]);
            fprintf(stderr, "[ATTN-CK L%d %-14s sa=%.6f]", layer, tag, sum_abs);
            for (int i=0;i<8;i++) fprintf(stderr, " %.5f", h[i]);
            fprintf(stderr, "\n"); fflush(stderr);
        };

        // 1. Batched RMSNorm: rows=n_tokens, output → mlp_chunk_norm
        //    (mlp_chunk_norm is reused; MLP runs after attn and re-norms).
        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w(buf.mlp_chunk_norm, hidden_chunk, (float*)norm_w->data,
                                n_tokens, H, eps, stream);
        } else {
            rms_norm_f32in(buf.mlp_chunk_norm, hidden_chunk, (half*)norm_w->data,
                           n_tokens, H, eps, stream);
        }
        attn_dump_h("norm_out", buf.mlp_chunk_norm + (size_t)target_t * H);

        // 2. Quantize all n_tokens × H normed values once
        gpu_qi[g].quantize_chunk(buf.mlp_chunk_norm, H, n_tokens, stream);

        // 3. Batched Q/K/V projections — 3 GEMV calls instead of 3*n_tokens.
        quant_gemv_chunk(q_w->data, q_w->type, gpu_qi[g].q8_buf, ab.attn_chunk_q,
                         H, q_out_dim, n_tokens, stream);
        quant_gemv_chunk(k_w->data, k_w->type, gpu_qi[g].q8_buf, ab.attn_chunk_k,
                         H, kv_dim, n_tokens, stream);
        quant_gemv_chunk(v_w->data, v_w->type, gpu_qi[g].q8_buf, ab.attn_chunk_v,
                         H, kv_dim, n_tokens, stream);
        attn_dump_h("q_proj", ab.attn_chunk_q + (size_t)target_t * q_out_dim);
        attn_dump_h("k_proj", ab.attn_chunk_k + (size_t)target_t * kv_dim);
        attn_dump_h("v_proj", ab.attn_chunk_v + (size_t)target_t * kv_dim);

        // 4. Batched deinterleave Q/gate, head-RMS, RoPE across all n_tokens.
        //    Single launch per op (blockIdx.y strides tokens) instead of
        //    n_tokens × 5 launches → ~128× fewer launches per attn layer at
        //    chunk=128.
        int rope_dim = rope.rope_dim;
        int half_rope = rope_dim / 2;
        int tn = min(hd, 128);
        // For TQ path, `kv.k` / `kv.v` are not allocated. We route attention
        // through the per-GPU fp16 scratch (tq_k_buf/tq_v_buf), populated below
        // with bulk-dequanted history + newly-computed tokens.
        half* k_cache_ptr;
        half* v_cache_ptr;
        // For multi-slot: each pointer is offset to the slot's KV virtual
        // partition. Downstream chunked attn kernels read [0, sub_seq_total)
        // through this base, naturally seeing only the slot's tokens.
        if (use_q8_kv) {
            // Q8_0 chunked path mirrors the TQ chunked path: history is bulk
            // dequanted into the per-GPU fp16 scratch and the chunked attn
            // kernels read fp16 from there.
            k_cache_ptr = tq_k_buf[g] + slot_kv_off;
            v_cache_ptr = tq_v_buf[g] + slot_kv_off;
        } else if (use_turboquant) {
            k_cache_ptr = tq_k_buf[g] + slot_kv_off;
            v_cache_ptr = tq_v_buf[g] + slot_kv_off;
        } else {
            auto& kv = kv_caches[layer];
            k_cache_ptr = kv.k + slot_kv_off;
            v_cache_ptr = kv.v + slot_kv_off;
        }
        {
            if (has_q_gate) {
                dim3 deint_grid((total_qg + 255) / 256, n_tokens);
                deinterleave_qg_kernel_chunk<<<deint_grid, 256, 0, stream>>>(
                    ab.attn_chunk_q, ab.attn_chunk_q_post, ab.attn_chunk_gate,
                    num_q, hd, n_tokens);
            } else {
                // Ungated: attn_chunk_q is already [n_tokens × total_qg]; use
                // it directly as q_post (no gate split, no gate sigmoid later).
                cudaMemcpyAsync(ab.attn_chunk_q_post, ab.attn_chunk_q,
                                (size_t)n_tokens * total_qg * sizeof(half),
                                cudaMemcpyDeviceToDevice, stream);
            }
            attn_dump_h("q_deint", ab.attn_chunk_q_post + (size_t)target_t * total_qg);

            static const bool skip_qk_norm_c = getenv("SKIP_QK_NORM") != nullptr;
            if (!skip_qk_norm_c) {
                dim3 q_rms_grid(num_q, n_tokens);
                head_rms_norm_kernel_chunk<<<q_rms_grid, tn, tn * sizeof(float), stream>>>(
                    ab.attn_chunk_q_post, (float*)q_norm_w->data, num_q, hd, eps, n_tokens);
                dim3 k_rms_grid(num_kv, n_tokens);
                head_rms_norm_kernel_chunk<<<k_rms_grid, tn, tn * sizeof(float), stream>>>(
                    ab.attn_chunk_k, (float*)k_norm_w->data, num_kv, hd, eps, n_tokens);
            }
            attn_dump_h("q_after_qnorm", ab.attn_chunk_q_post + (size_t)target_t * total_qg);
            attn_dump_h("k_after_qnorm", ab.attn_chunk_k + (size_t)target_t * kv_dim);

            dim3 rope_q_grid((num_q  * half_rope + 255) / 256, n_tokens);
            dim3 rope_k_grid((num_kv * half_rope + 255) / 256, n_tokens);
            // M-RoPE path is engaged when main() set up the per-token
            // (pos_t, pos_h, pos_w) arrays for vision prompts. For text-only
            // chunks all three axes carry the same value, so the result is
            // bit-identical to the legacy 1D path.
            extern int* g_mrope_pos_t[4];
            extern int* g_mrope_pos_h[4];
            extern int* g_mrope_pos_w[4];
            extern int g_mrope_sec_t, g_mrope_sec_h, g_mrope_sec_w;
            if (g_mrope_pos_t[g] && g_mrope_pos_h[g] && g_mrope_pos_w[g]) {
                const int* pt = g_mrope_pos_t[g] + start_pos;
                const int* ph = g_mrope_pos_h[g] + start_pos;
                const int* pw = g_mrope_pos_w[g] + start_pos;
                apply_rope_kernel_mrope_chunk<<<rope_q_grid, 256, 0, stream>>>(
                    ab.attn_chunk_q_post, rope.sin_table(g), rope.cos_table(g),
                    pt, ph, pw, g_mrope_sec_t, g_mrope_sec_h, g_mrope_sec_w,
                    num_q,  hd, rope_dim, n_tokens);
                apply_rope_kernel_mrope_chunk<<<rope_k_grid, 256, 0, stream>>>(
                    ab.attn_chunk_k, rope.sin_table(g), rope.cos_table(g),
                    pt, ph, pw, g_mrope_sec_t, g_mrope_sec_h, g_mrope_sec_w,
                    num_kv, hd, rope_dim, n_tokens);
            } else {
                apply_rope_kernel_chunk<<<rope_q_grid, 256, 0, stream>>>(
                    ab.attn_chunk_q_post, rope.sin_table(g), rope.cos_table(g),
                    start_pos, num_q,  hd, rope_dim, n_tokens);
                apply_rope_kernel_chunk<<<rope_k_grid, 256, 0, stream>>>(
                    ab.attn_chunk_k, rope.sin_table(g), rope.cos_table(g),
                    start_pos, num_kv, hd, rope_dim, n_tokens);
            }
            attn_dump_h("q_after_rope", ab.attn_chunk_q_post + (size_t)target_t * total_qg);
            attn_dump_h("k_after_rope", ab.attn_chunk_k + (size_t)target_t * kv_dim);
        }
        // Bulk KV cache append. fp16 path: copy attn_chunk_k/v directly into
        // kv.k/kv.v at offset start_pos. TQ path: (1) quantize new K/V into
        // the TQ cache at offset start_pos, (2) bulk-dequant the historical
        // [0..start_pos-1] range from TQ into the per-GPU fp16 scratch, (3)
        // copy the fp16 new tokens at [start_pos:start_pos+n_tokens] into
        // scratch so the stock chunked attn kernels see a contiguous fp16
        // cache. Steps 2+3 run once per attn layer per chunk instead of the
        // previous per-token O(seq²) fallback.
        {
            size_t new_bytes = (size_t)n_tokens * kv_dim * sizeof(half);
            if (use_q8_kv) {
                auto& q8 = q8_kv_caches[layer];
                int bpt = q8.blocks_per_token;
                int new_blocks = n_tokens * bpt;
                size_t slot_blk_off = kv_slot_offset(slot) * bpt;
                block_q8_0_aligned* q8_k_slot = q8.k + slot_blk_off;
                block_q8_0_aligned* q8_v_slot = q8.v + slot_blk_off;
                quantize_kv_q8_0_kern<<<new_blocks, 32, 0, stream>>>(
                    ab.attn_chunk_k, &q8_k_slot[(size_t)start_pos * bpt], new_blocks);
                quantize_kv_q8_0_kern<<<new_blocks, 32, 0, stream>>>(
                    ab.attn_chunk_v, &q8_v_slot[(size_t)start_pos * bpt], new_blocks);
                // The bulk dequant + new-fp16 copy below is only needed when
                // attention will read fp16 from tq_k_buf/tq_v_buf. With
                // Q8KV_FUSED_CHUNK=1 the chunked attn kernel decodes Q8 in-tile
                // so we can skip both passes — biggest cost at long context
                // (256K: ~O(seq) per chunk × N chunks = O(seq²) total).
                //
                // Skip only when every sub-chunk in this call will use the
                // split-K fused-Q8 path (sub_seq_total >= 4096). Below that
                // the non-split kernels still read fp16 and need the scratch
                // populated, so keep the bulk pass.
                static const bool q8_fused_chunk = (getenv("Q8KV_FUSED_CHUNK") != nullptr);
                // Sparse path reads fp16 K from k_cache_ptr (= tq_k_buf for Q8KV
                // mode), so we must keep the bulk dequant alive when sparse_rt
                // is enabled even if Q8KV_FUSED_CHUNK is also set.
                bool skip_dequant = q8_fused_chunk && (start_pos + ATTN_NB >= 4096)
                                  && !sparse_rt.enabled;
                if (!skip_dequant) {
                    if (start_pos > 0) {
                        int hist_blocks = start_pos * bpt;
                        dim3 dq_grid((hist_blocks + 7) / 8);
                        dim3 dq_block(32, 8);
                        dequantize_kv_q8_0_kern<<<dq_grid, dq_block, 0, stream>>>(
                            q8_k_slot, tq_k_buf[g] + slot_kv_off, hist_blocks);
                        dequantize_kv_q8_0_kern<<<dq_grid, dq_block, 0, stream>>>(
                            q8_v_slot, tq_v_buf[g] + slot_kv_off, hist_blocks);
                    }
                    cudaMemcpyAsync(tq_k_buf[g] + slot_kv_off + (size_t)start_pos * kv_dim,
                                    ab.attn_chunk_k, new_bytes,
                                    cudaMemcpyDeviceToDevice, stream);
                    cudaMemcpyAsync(tq_v_buf[g] + slot_kv_off + (size_t)start_pos * kv_dim,
                                    ab.attn_chunk_v, new_bytes,
                                    cudaMemcpyDeviceToDevice, stream);
                }
            } else if (use_turboquant) {
                auto& tq = tq_kv_caches[layer];
                int bpt = tq.blocks_per_token;
                int new_blocks = n_tokens * bpt;
                size_t slot_blk_off = kv_slot_offset(slot) * bpt;
                block_tq3* tq_k_slot = tq.k + slot_blk_off;
                block_tq3* tq_v_slot = tq.v + slot_blk_off;
                tq3_quantize_kernel<<<(new_blocks + 31)/32, 32, 0, stream>>>(
                    ab.attn_chunk_k, &tq_k_slot[(size_t)start_pos * bpt], new_blocks);
                tq3_quantize_kernel<<<(new_blocks + 31)/32, 32, 0, stream>>>(
                    ab.attn_chunk_v, &tq_v_slot[(size_t)start_pos * bpt], new_blocks);
                if (start_pos > 0) {
                    int hist_blocks = start_pos * bpt;
                    // Dequant slot's history into the slot-local fp16 scratch.
                    tq3_dequantize_kernel<<<(hist_blocks + 31)/32, 32, 0, stream>>>(
                        tq_k_slot, tq_k_buf[g] + slot_kv_off, hist_blocks);
                    tq3_dequantize_kernel<<<(hist_blocks + 31)/32, 32, 0, stream>>>(
                        tq_v_slot, tq_v_buf[g] + slot_kv_off, hist_blocks);
                }
                cudaMemcpyAsync(tq_k_buf[g] + slot_kv_off + (size_t)start_pos * kv_dim,
                                ab.attn_chunk_k, new_bytes,
                                cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(tq_v_buf[g] + slot_kv_off + (size_t)start_pos * kv_dim,
                                ab.attn_chunk_v, new_bytes,
                                cudaMemcpyDeviceToDevice, stream);
                if (fp16_dec_cache_on) {
                    auto& fc = fp16_dec_caches[layer];
                    cudaMemcpyAsync(fc.k + slot_kv_off + (size_t)start_pos * kv_dim,
                                    ab.attn_chunk_k, new_bytes,
                                    cudaMemcpyDeviceToDevice, stream);
                    cudaMemcpyAsync(fc.v + slot_kv_off + (size_t)start_pos * kv_dim,
                                    ab.attn_chunk_v, new_bytes,
                                    cudaMemcpyDeviceToDevice, stream);
                }
            } else {
                half* k_dst = k_cache_ptr + (size_t)start_pos * kv_dim;
                half* v_dst = v_cache_ptr + (size_t)start_pos * kv_dim;
                cudaMemcpyAsync(k_dst, ab.attn_chunk_k, new_bytes, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(v_dst, ab.attn_chunk_v, new_bytes, cudaMemcpyDeviceToDevice, stream);
            }
        }

        // 5–7. Chunked attention compute, processed in ATTN_NB-token
        //      sub-chunks so the value kernel's register-resident
        //      accumulator stays bounded (see ATTN_NB / acc[16][8]).
        //      `kv` already aliased above for the bulk K/V append.
        int sub_processed = 0;
        auto pa_now = [](){ return std::chrono::high_resolution_clock::now(); };
        auto pa_sync_ms = [&](std::chrono::high_resolution_clock::time_point tb) {
            cudaStreamSynchronize(stream);
            auto te = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(te - tb).count();
        };
        // FlashAttention fused path: score+softmax+value in one kernel per
        // (kv_head, t_idx). Supports Qwen3.5-27B (GQA=6) and -9B (GQA=4)
        // attention shapes.
        // 27B/9B hybrid: HD=256, num_kv=4, GQA = num_q/num_kv (6 or 4).
        // Qwen3-4B dense embed/reranker: HD=128, num_q=32, num_kv=8, GQA=4
        // (full rope_dim=128, applied pre-FA below — FA kernels are rope-agnostic).
        bool can_flash_hybrid = (hd == 256 && num_kv == 4
                              && (num_q == 24 || num_q == 16));
        bool can_flash_dense  = (hd == 128 && num_kv == 8 && num_q == 32);
        bool can_flash = g_use_flash_attn && (can_flash_hybrid || can_flash_dense);
        while (sub_processed < n_tokens) {
            int sub_n = std::min(ATTN_NB, n_tokens - sub_processed);
            int sub_start_pos = start_pos + sub_processed;
            int sub_seq_total = sub_start_pos + sub_n;

            // Pointers into the chunk-major Q-post / scores / out buffers,
            // offset by sub_processed tokens.
            half*  q_post_sub = ab.attn_chunk_q_post + (size_t)sub_processed * total_qg;
            // Per-sub-chunk score scratch: sub-chunks are processed sequentially
            // and never read each other's scores, so reuse the same ATTN_NB-row
            // window (offset 0) instead of a CHUNK_SIZE-row slab. At 256K ctx the
            // old CHUNK_SIZE sizing was 4.3 GB/GPU (num_q*kv_max_seq*256*4) which
            // overcommitted 16 GB cards on the smaller-hidden A3B model and
            // silently zeroed the hidden state.
            float* scores_sub = ab.attn_chunk_scores;
            half*  out_sub    = ab.attn_chunk_out    + (size_t)sub_processed * total_qg;

            if (can_flash) {
                auto tf0 = pa_now();
                // FA_BM:  32 (default) | 64 — BM-generic kernel, 96KB SMEM opt-in
                // FA_NT:  1  (default) | 2 — K/V-sharing kernel, halves K/V HBM
                //                            traffic by processing 2 t_idx per
                //                            block. BM is forced to 32 for NT.
                static const int fa_bm = []{
                    const char* e = getenv("FA_BM");
                    int v = e ? atoi(e) : 32;
                    return (v == 64) ? 64 : 32;
                }();
                static const int fa_nt = []{
                    const char* e = getenv("FA_NT");
                    int v = e ? atoi(e) : 1;
                    return (v == 2) ? 2 : 1;
                }();
                // FA_SK: 4 (default) | 0 | 2 | 8 — split-K. Each (kv_head,t_idx)
                // maps to FA_SK blocks; partials merged via log-sum-exp. Helps
                // long contexts (SM under-utilisation: 64-block grid ≪ 204 SMs).
                // Gated to sub_seq_total>=4096 below — short ctx unaffected.
                // 9B 1GPU 18K 1.46x, 27B 3GPU 18K 1.34x (llama parity).
                // part_o stored fp32 so merge order matches base FA's fp32 noise
                // floor — greedy argmax stable for Korean coding prompts.
                // FA_SK=0 to opt out.
                static const int fa_sk = []{
                    const char* e = getenv("FA_SK");
                    if (!e) return 4;
                    int v = atoi(e);
                    return (v == 0 || v == 2 || v == 4 || v == 8) ? v : 4;
                }();
                constexpr int HD = 256, BLOCK = 256;
                static bool fa_smem_set[16] = {false};
                int cur_dev = 0; cudaGetDevice(&cur_dev);
                auto ensure_smem = [&](const void* fn) {
                    if (fa_bm == 64 && cur_dev >= 0 && cur_dev < 16
                        && !fa_smem_set[cur_dev]) {
                        cudaFuncSetAttribute(
                            fn,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            96 * 1024);
                        fa_smem_set[cur_dev] = true;
                    }
                };
                // Split-K shortcut. Skipped for short contexts where the base
                // grid (num_kv*sub_n) already fills SMs and the merge overhead
                // would dominate.
                if (fa_sk > 0 && sub_seq_total >= 4096 && hd == 256) {
                    // BM=16 (was 32): with __launch_bounds__(BLOCK,4) on the
                    // split kernels this drops dyn smem to 19.4 KB so 4 blocks
                    // fit per SM (50% occ vs 25%) — dense prefill score ~1.78×
                    // (microbench). Smaller key tiles = more tile iters but
                    // bit-identical result.
                    constexpr int BM = 16;
                    int active_end_max = sub_start_pos + sub_n;
                    int smem_bytes = (num_q == 24 ? 6 : 4) * HD * sizeof(half)
                                   + 2 * BM * HD * sizeof(half)
                                   + (num_q == 24 ? 6 : 4) * BM * sizeof(float);
                    dim3 fg(num_kv, sub_n, fa_sk);
                    // MInference-style block-sparse path. Reads fp16 K/V from
                    // k_cache_ptr (already populated for use_q8_kv via the
                    // legacy bulk-dequant pass — sparse mode forces that pass
                    // to keep tq_k_buf valid). Per (kv_head, t_idx) we build
                    // a top-k block index against the mean-pooled K cache,
                    // then drive flash_attn_chunk_block_sparse_split with that
                    // index. Existing flash_attn_split_merge handles K-split
                    // log-sum-exp combine.
                    // Per-layer pattern routing. Phase 2A: pick one pattern
                    // for the whole layer from the profile (or fall back to
                    // BLOCK_SPARSE without a profile). DENSE → fall through
                    // to the existing dense FA dispatch. BLOCK_SPARSE → score
                    // K_pool + top-k. A_SHAPE → deterministic sink+window
                    // index. Both index builders feed the same
                    // flash_attn_chunk_block_sparse_split kernel.
                    // Research knob: in uniform mode (no profile), force the
                    // multi-signature (mean + max-abs) selector so a spiky
                    // single-token needle survives mean-pool dilution. Default
                    // off → SPARSE_BLOCK (mean-only), unchanged behavior.
                    static const bool uniform_ms =
                        getenv("MINF_UNIFORM_MS") && atoi(getenv("MINF_UNIFORM_MS")) != 0;
                    SparsePattern layer_pattern = uniform_ms ? SPARSE_BLOCK_MS : SPARSE_BLOCK;
                    uint32_t a_sink   = 64;
                    uint32_t a_window = 1024;
                    if (sparse_rt.enabled && !sparse_rt.profile.empty()
                        && (uint32_t)layer < sparse_rt.profile.num_layers) {
                        // Pattern of head 0 (representative). The profiler
                        // writes the same pattern for every q_head in a layer
                        // when running per-layer mode.
                        const auto& h0 = sparse_rt.profile.at((int)layer, 0);
                        layer_pattern = h0.pattern;
                        if (h0.window > 0) a_window = h0.window;
                        if (h0.sink   > 0) a_sink   = h0.sink;
                    }
                    bool layer_sparse_ok = sparse_rt.enabled
                        && sub_seq_total >= sparse_rt.min_seq_for_sparse
                        && (layer_pattern == SPARSE_BLOCK
                         || layer_pattern == SPARSE_A_SHAPE
                         || layer_pattern == SPARSE_BLOCK_MS
                         || layer_pattern == SPARSE_VERTICAL_SLASH);
                    if (layer_sparse_ok) {
                        constexpr int BLOCK_N    = 64;
                        constexpr int BLOCK_THR  = 256;
                        int n_blocks = (sub_seq_total + BLOCK_N - 1) / BLOCK_N;
                        int top_k = max(1, (int)(n_blocks * sparse_rt.flops_budget));
                        // Research knob: lift the top-k block cap (default 64).
                        // At long ctx the needle block must rank in the top-k of
                        // many distractors; more room helps recall. Buffer below
                        // is sized to this same cap.
                        static const int topk_cap = []{
                            const char* e = getenv("MINF_TOPK_CAP");
                            int v = e ? atoi(e) : 64;
                            return v > 0 ? v : 64;
                        }();
                        if (top_k > topk_cap) top_k = topk_cap;
                        // Lazy alloc K_pool (per-layer, max-sized) + block_index
                        // (per-call, sub_n_max × top_k_max).
                        int n_blocks_max = (slot_max_seq + BLOCK_N - 1) / BLOCK_N;
                        if (!ab.sparse_k_pool) {
                            cudaMalloc(&ab.sparse_k_pool,
                                       (size_t)num_kv * n_blocks_max * HD * sizeof(half));
                        }
                        if (!ab.sparse_block_index) {
                            cudaMalloc(&ab.sparse_block_index,
                                       (size_t)num_kv * ATTN_NB * topk_cap * sizeof(int));
                        }
                        // Launch-error tripwire. After every sparse kernel we
                        // peek at the last error so a fault that surfaces on
                        // the next CUDA call (cudaPeekAtLastError) tells us
                        // the most recent sparse kernel name + context.
                        // MINF_DEBUG_LAUNCH=1 forces a sync-after-launch so a
                        // kernel that asserts mid-execution shows up here
                        // instead of corrupting state for the next call.
                        // Default mode keeps it cheap (no sync, just peek).
                        // Always logs on error regardless of env so a real
                        // fault in production leaves a breadcrumb.
                        static const bool minf_debug_launch =
                            getenv("MINF_DEBUG_LAUNCH") != nullptr;
                        // Drain any sticky error inherited from non-sparse code so
                        // our per-launch checks below only attribute errors that
                        // actually originated inside the sparse path.
                        {
                            cudaError_t pre = cudaGetLastError();
                            if (pre != cudaSuccess) {
                                fprintf(stderr,
                                    "[sparse-pre] L%d gpu=%d slot=%d sub_start=%d sub_n=%d "
                                    "INHERITED_STICKY_ERR -> %s\n",
                                    layer, g, slot, sub_start_pos, sub_n,
                                    cudaGetErrorString(pre));
                                fflush(stderr);
                            }
                        }
                        auto check_launch = [&](const char* tag) {
                            if (minf_debug_launch) cudaStreamSynchronize(stream);
                            // cudaGetLastError clears the error flag so the next
                            // check only reports problems caused by the *next*
                            // kernel — gives us an exact culprit instead of a
                            // sticky cascade across every subsequent peek.
                            cudaError_t err = cudaGetLastError();
                            if (err != cudaSuccess || minf_debug_launch) {
                                fprintf(stderr,
                                    "[sparse-launch] L%d gpu=%d slot=%d sub_start=%d sub_n=%d "
                                    "n_blocks=%d top_k=%d pattern=%d %s -> %s\n",
                                    layer, g, slot, sub_start_pos, sub_n,
                                    n_blocks, top_k, (int)layer_pattern, tag,
                                    err == cudaSuccess ? "OK" : cudaGetErrorString(err));
                                fflush(stderr);
                            }
                        };
                        if (layer_pattern == SPARSE_BLOCK || layer_pattern == SPARSE_BLOCK_MS) {
                            // 1. K_pool[num_kv, n_blocks, HD]
                            dim3 kp_grid(num_kv, n_blocks, (HD + BLOCK_THR - 1) / BLOCK_THR);
                            build_k_pool_kern<HD, BLOCK_N, BLOCK_THR>
                                <<<kp_grid, BLOCK_THR, 0, stream>>>(
                                    k_cache_ptr, ab.sparse_k_pool, num_kv, sub_seq_total);
                            check_launch("build_k_pool");
                            // 1b. K_pool_max — only built when this layer asked for
                            // mBSA. Lazy-alloced on first MS layer encountered. The
                            // grid layout matches mean-pool, just a different output.
                            if (layer_pattern == SPARSE_BLOCK_MS) {
                                if (!ab.sparse_k_pool_max) {
                                    cudaMalloc(&ab.sparse_k_pool_max,
                                               (size_t)num_kv * n_blocks_max * HD * sizeof(half));
                                }
                                build_k_pool_max_kern<HD, BLOCK_N, BLOCK_THR>
                                    <<<kp_grid, BLOCK_THR, 0, stream>>>(
                                        k_cache_ptr, ab.sparse_k_pool_max, num_kv, sub_seq_total);
                                check_launch("build_k_pool_max");
                            }
                            // 2. block_index[num_kv, sub_n_max, top_k]
                            dim3 bi_grid(num_kv, sub_n);
                            size_t bi_smem = HD * sizeof(half) + (size_t)n_blocks * sizeof(float);
                            // β default 0.5: max-sig contributes at half the weight
                            // of mean-sig. Tunable via MINF_MS_BETA env if needed.
                            static const float ms_beta = []{
                                const char* e = getenv("MINF_MS_BETA");
                                return e ? atof(e) : 0.5f;
                            }();
                            if (layer_pattern == SPARSE_BLOCK_MS) {
                                if (num_q == 24) {
                                    build_block_index_ms_kern<HD, 6, BLOCK_THR>
                                        <<<bi_grid, BLOCK_THR, bi_smem, stream>>>(
                                            q_post_sub, ab.sparse_k_pool, ab.sparse_k_pool_max,
                                            ab.sparse_block_index,
                                            num_q, num_kv, sub_n, ATTN_NB, sub_start_pos,
                                            top_k, BLOCK_N, n_blocks, n_blocks, ms_beta);
                                } else {
                                    build_block_index_ms_kern<HD, 4, BLOCK_THR>
                                        <<<bi_grid, BLOCK_THR, bi_smem, stream>>>(
                                            q_post_sub, ab.sparse_k_pool, ab.sparse_k_pool_max,
                                            ab.sparse_block_index,
                                            num_q, num_kv, sub_n, ATTN_NB, sub_start_pos,
                                            top_k, BLOCK_N, n_blocks, n_blocks, ms_beta);
                                }
                                check_launch("build_block_index_ms");
                            } else if (num_q == 24) {
                                build_block_index_kern<HD, 6, BLOCK_THR>
                                    <<<bi_grid, BLOCK_THR, bi_smem, stream>>>(
                                        q_post_sub, ab.sparse_k_pool, ab.sparse_block_index,
                                        num_q, num_kv, sub_n, ATTN_NB, sub_start_pos,
                                        top_k, BLOCK_N, n_blocks);
                                check_launch("build_block_index<6>");
                            } else {
                                build_block_index_kern<HD, 4, BLOCK_THR>
                                    <<<bi_grid, BLOCK_THR, bi_smem, stream>>>(
                                        q_post_sub, ab.sparse_k_pool, ab.sparse_block_index,
                                        num_q, num_kv, sub_n, ATTN_NB, sub_start_pos,
                                        top_k, BLOCK_N, n_blocks);
                                check_launch("build_block_index<4>");
                            }
                        } else if (layer_pattern == SPARSE_VERTICAL_SLASH) {
                            // SPARSE_VERTICAL_SLASH: build K_pool, then aggregate
                            // chunk-wide vertical / slash patterns and assemble
                            // per-token block_index = vertical ∪ slash@t_idx.
                            dim3 kp_grid(num_kv, n_blocks, (HD + BLOCK_THR - 1) / BLOCK_THR);
                            build_k_pool_kern<HD, BLOCK_N, BLOCK_THR>
                                <<<kp_grid, BLOCK_THR, 0, stream>>>(
                                    k_cache_ptr, ab.sparse_k_pool, num_kv, sub_seq_total);
                            check_launch("build_k_pool[vs]");
                            // V_top_k + S_top_k are pulled from the profile head
                            // record; sane defaults if both are zero.
                            int V_topk = sparse_rt.profile.empty()
                                ? 16 : (int)sparse_rt.profile.at((int)layer, 0).vertical_top_k;
                            int S_topk = sparse_rt.profile.empty()
                                ? 8  : (int)sparse_rt.profile.at((int)layer, 0).slash_top_k;
                            if (V_topk <= 0) V_topk = 16;
                            if (S_topk <= 0) S_topk = 8;
                            int V_max = 64, S_max = 64;
                            if (!ab.sparse_vertical_idx)
                                cudaMalloc(&ab.sparse_vertical_idx,
                                           (size_t)num_kv * V_max * sizeof(int));
                            if (!ab.sparse_slash_idx)
                                cudaMalloc(&ab.sparse_slash_idx,
                                           (size_t)num_kv * S_max * sizeof(int));
                            // Stage A: per-kv aggregation. SMEM = sub_n*HD halves
                            // + (sub_n+2)*n_blocks floats.
                            size_t vs_smem = (size_t)sub_n * HD * sizeof(half)
                                           + (size_t)sub_n * n_blocks * sizeof(float)
                                           + (size_t)2 * n_blocks * sizeof(float);
                            dim3 vs_grid(num_kv);
                            if (num_q == 24) {
                                build_vs_aggregate_kern<HD, 6, BLOCK_THR>
                                    <<<vs_grid, BLOCK_THR, vs_smem, stream>>>(
                                        q_post_sub, ab.sparse_k_pool,
                                        ab.sparse_vertical_idx, ab.sparse_slash_idx,
                                        num_q, num_kv, sub_n, ATTN_NB,
                                        sub_start_pos,
                                        V_topk, S_topk, BLOCK_N, n_blocks);
                            } else {
                                build_vs_aggregate_kern<HD, 4, BLOCK_THR>
                                    <<<vs_grid, BLOCK_THR, vs_smem, stream>>>(
                                        q_post_sub, ab.sparse_k_pool,
                                        ab.sparse_vertical_idx, ab.sparse_slash_idx,
                                        num_q, num_kv, sub_n, ATTN_NB,
                                        sub_start_pos,
                                        V_topk, S_topk, BLOCK_N, n_blocks);
                            }
                            check_launch("build_vs_aggregate");
                            // Stage B: per-token assembly.
                            int needed = V_topk + S_topk;
                            if (needed > top_k) top_k = needed;
                            if (top_k > 64) top_k = 64;
                            dim3 vsi_grid(num_kv, sub_n);
                            build_vs_index_kern<<<vsi_grid, 1, 0, stream>>>(
                                ab.sparse_vertical_idx, ab.sparse_slash_idx,
                                ab.sparse_block_index,
                                num_kv, sub_n, ATTN_NB,
                                sub_start_pos, top_k,
                                BLOCK_N, V_topk, S_topk);
                            check_launch("build_vs_index");
                        } else {
                            // SPARSE_A_SHAPE: deterministic sink+window block list,
                            // no Q/K scoring needed. Index built directly from
                            // (sink, window) parameters provided by the profile.
                            int sink_blocks   = ((int)a_sink   + BLOCK_N - 1) / BLOCK_N;
                            int window_blocks = ((int)a_window + BLOCK_N - 1) / BLOCK_N + 1;
                            // Recompute top_k to fit the deterministic budget
                            // (cap at 64 like the kernel-side block_index allocation).
                            int needed = sink_blocks + window_blocks;
                            if (needed > top_k) top_k = needed;
                            if (top_k > 64) top_k = 64;
                            dim3 ai_grid(num_kv, sub_n);
                            build_a_shape_index_kern<<<ai_grid, 1, 0, stream>>>(
                                ab.sparse_block_index,
                                num_kv, sub_n, ATTN_NB,
                                sub_start_pos, top_k,
                                BLOCK_N, sink_blocks, window_blocks);
                            check_launch("build_a_shape_index");
                        }
                        auto launch_sp = [&](auto gqa_const, auto k_const) {
                            constexpr int GQA = decltype(gqa_const)::value;
                            constexpr int K   = decltype(k_const)::value;
                            flash_attn_chunk_block_sparse_split<HD, GQA, BM, BLOCK, K>
                                <<<fg, BLOCK, smem_bytes, stream>>>(
                                    q_post_sub, k_cache_ptr, v_cache_ptr,
                                    ab.sparse_block_index,
                                    ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                                    num_q, num_kv, sub_start_pos, sub_n, ATTN_NB,
                                    active_end_max, scale, top_k, BLOCK_N);
                            check_launch("flash_attn_block_sparse_split");
                            dim3 mg(num_kv, sub_n);
                            flash_attn_split_merge<HD, GQA, BLOCK, K>
                                <<<mg, BLOCK, 0, stream>>>(
                                    ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                                    out_sub, num_q, sub_n, ATTN_NB);
                            check_launch("flash_attn_split_merge[sparse]");
                        };
                        if (num_q == 24) {
                            if      (fa_sk == 2) launch_sp(std::integral_constant<int,6>{}, std::integral_constant<int,2>{});
                            else if (fa_sk == 4) launch_sp(std::integral_constant<int,6>{}, std::integral_constant<int,4>{});
                            else                 launch_sp(std::integral_constant<int,6>{}, std::integral_constant<int,8>{});
                        } else {
                            if      (fa_sk == 2) launch_sp(std::integral_constant<int,4>{}, std::integral_constant<int,2>{});
                            else if (fa_sk == 4) launch_sp(std::integral_constant<int,4>{}, std::integral_constant<int,4>{});
                            else                 launch_sp(std::integral_constant<int,4>{}, std::integral_constant<int,8>{});
                        }
                        if (g_profile_attn) g_attn_fused_ms += pa_sync_ms(tf0);
                        sub_processed += sub_n;
                        continue;
                    }
                    // Q8KV_FUSED_CHUNK=1: read directly from the Q8 cache via
                    // cooperative dequant inside the kernel, skipping the
                    // bulk-dequant + fp16-scratch round-trip the legacy Q8 path
                    // takes. Eliminates the per-chunk O(seq²) dequant pass and
                    // the kv_total_seq×kv_dim fp16 scratch — the latter is the
                    // dominant cost at 256K (≈1 GB/GPU). Opt-in until correctness
                    // and speed are confirmed against the bulk path.
                    static const bool q8_fused_chunk = use_q8_kv && (getenv("Q8KV_FUSED_CHUNK") != nullptr);
                    if (q8_fused_chunk) {
                        auto& q8 = q8_kv_caches[layer];
                        size_t slot_off_blocks = kv_slot_offset(slot) * q8.blocks_per_token;
                        const block_q8_0_aligned* k_q8_slot = q8.k + slot_off_blocks;
                        const block_q8_0_aligned* v_q8_slot = q8.v + slot_off_blocks;
                        auto launch_q8 = [&](auto gqa_const, auto k_const) {
                            constexpr int GQA = decltype(gqa_const)::value;
                            constexpr int K   = decltype(k_const)::value;
                            flash_attn_chunk_fused_split_q8<HD, GQA, BM, BLOCK, K>
                                <<<fg, BLOCK, smem_bytes, stream>>>(
                                    q_post_sub, k_q8_slot, v_q8_slot,
                                    ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                                    num_q, num_kv, sub_start_pos, sub_n, ATTN_NB,
                                    active_end_max, scale);
                            dim3 mg(num_kv, sub_n);
                            flash_attn_split_merge<HD, GQA, BLOCK, K>
                                <<<mg, BLOCK, 0, stream>>>(
                                    ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                                    out_sub, num_q, sub_n, ATTN_NB);
                        };
                        if (num_q == 24) {
                            if      (fa_sk == 2) launch_q8(std::integral_constant<int,6>{}, std::integral_constant<int,2>{});
                            else if (fa_sk == 4) launch_q8(std::integral_constant<int,6>{}, std::integral_constant<int,4>{});
                            else                 launch_q8(std::integral_constant<int,6>{}, std::integral_constant<int,8>{});
                        } else {
                            if      (fa_sk == 2) launch_q8(std::integral_constant<int,4>{}, std::integral_constant<int,2>{});
                            else if (fa_sk == 4) launch_q8(std::integral_constant<int,4>{}, std::integral_constant<int,4>{});
                            else                 launch_q8(std::integral_constant<int,4>{}, std::integral_constant<int,8>{});
                        }
                        if (g_profile_attn) g_attn_fused_ms += pa_sync_ms(tf0);
                        sub_processed += sub_n;
                        continue;
                    }
                    // PREFILL_FA_V2: NT-batched dense split kernel (loads K/V
                    // once per NT t_idx instead of per t_idx). NT=4 BM=8
                    // K_SPLITS=16 (sweep winner). Default ON — validated:
                    // microbench bit-matches the fp32 reference at 14608; engine
                    // long-ctx gen (9522 tok) -> correct, multi-turn cache ->
                    // correct; production prefill 18K -16% / 26K -22% vs the
                    // occupancy-only build. PREFILL_FA_V2=0 opts back into the
                    // per-t_idx split kernel without a rebuild.
                    static const bool fa_v2 = [](){
                        const char* e = getenv("PREFILL_FA_V2");
                        return !e || e[0] != '0';
                    }();
                    if (fa_v2) {
                        constexpr int NT = 4, BMV = 8, KV = 16;
                        int smem_v2 = (num_q == 24 ? 6 : 4) * NT * HD * sizeof(half)
                                    + 2 * BMV * HD * sizeof(half)
                                    + (num_q == 24 ? 6 : 4) * NT * BMV * sizeof(float);
                        auto launch_v2 = [&](auto gqa_const) {
                            constexpr int GQA = decltype(gqa_const)::value;
                            void* fn = (void*)flash_attn_chunk_fused_split_ntb<HD,GQA,BMV,BLOCK,KV,NT>;
                            if (smem_v2 > 48*1024 && cur_dev>=0 && cur_dev<16 && !fa_smem_set[cur_dev]) {
                                cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024);
                            }
                            dim3 fgv(num_kv, (sub_n + NT - 1)/NT, KV);
                            flash_attn_chunk_fused_split_ntb<HD,GQA,BMV,BLOCK,KV,NT>
                                <<<fgv, BLOCK, smem_v2, stream>>>(
                                    q_post_sub, k_cache_ptr, v_cache_ptr,
                                    ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                                    num_q, num_kv, sub_start_pos, sub_n, ATTN_NB,
                                    active_end_max, scale);
                            dim3 mg(num_kv, sub_n);
                            flash_attn_split_merge<HD, GQA, BLOCK, KV>
                                <<<mg, BLOCK, 0, stream>>>(
                                    ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                                    out_sub, num_q, sub_n, ATTN_NB);
                        };
                        if (num_q == 24) launch_v2(std::integral_constant<int,6>{});
                        else             launch_v2(std::integral_constant<int,4>{});
                        if (g_profile_attn) g_attn_fused_ms += pa_sync_ms(tf0);
                        sub_processed += sub_n;
                        continue;
                    }
                    auto launch_split = [&](auto gqa_const, auto k_const) {
                        constexpr int GQA = decltype(gqa_const)::value;
                        constexpr int K   = decltype(k_const)::value;
                        flash_attn_chunk_fused_split<HD, GQA, BM, BLOCK, K>
                            <<<fg, BLOCK, smem_bytes, stream>>>(
                                q_post_sub, k_cache_ptr, v_cache_ptr,
                                ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                                num_q, num_kv, sub_start_pos, sub_n, ATTN_NB,
                                active_end_max, scale);
                        dim3 mg(num_kv, sub_n);
                        flash_attn_split_merge<HD, GQA, BLOCK, K>
                            <<<mg, BLOCK, 0, stream>>>(
                                ab.attn_split_m, ab.attn_split_l, ab.attn_split_o,
                                out_sub, num_q, sub_n, ATTN_NB);
                    };
                    if (num_q == 24) {
                        if      (fa_sk == 2) launch_split(std::integral_constant<int,6>{}, std::integral_constant<int,2>{});
                        else if (fa_sk == 4) launch_split(std::integral_constant<int,6>{}, std::integral_constant<int,4>{});
                        else                 launch_split(std::integral_constant<int,6>{}, std::integral_constant<int,8>{});
                    } else {
                        if      (fa_sk == 2) launch_split(std::integral_constant<int,4>{}, std::integral_constant<int,2>{});
                        else if (fa_sk == 4) launch_split(std::integral_constant<int,4>{}, std::integral_constant<int,4>{});
                        else                 launch_split(std::integral_constant<int,4>{}, std::integral_constant<int,8>{});
                    }
                    if (g_profile_attn) g_attn_fused_ms += pa_sync_ms(tf0);
                    sub_processed += sub_n;
                    continue;
                }
                if (hd == 128) {
                    // Qwen3-4B dense embed/reranker: HD=128, GQA=4, num_kv=8.
                    // Mirrors the GQA=4 base-path dispatch below but with a
                    // 128-wide head. fp16 KV (k_cache_ptr=kv.k), short ctx,
                    // no split-K/sparse/Q8 (gated to hd==256 above).
                    constexpr int HD128 = 128, GQA = 4;
                    if (fa_nt == 2) {
                        // GQA=4 N_T=2 → 8 active warps, BLOCK=256 (exact fit).
                        constexpr int BM = 32, NT = 2;
                        int smem_bytes = NT * GQA * HD128 * sizeof(half)
                                       + 2 * BM * HD128 * sizeof(half)
                                       + NT * GQA * BM * sizeof(float);
                        dim3 fg(num_kv, (sub_n + NT - 1) / NT);
                        flash_attn_chunk_fused_nt<HD128, GQA, BM, BLOCK, NT>
                            <<<fg, BLOCK, smem_bytes, stream>>>(
                                q_post_sub, k_cache_ptr, v_cache_ptr, out_sub,
                                num_q, num_kv, sub_start_pos, sub_n, scale);
                    } else if (fa_bm == 64) {
                        constexpr int BM = 64;
                        dim3 fg(num_kv, sub_n);
                        int smem_bytes = GQA * HD128 * sizeof(half)
                                       + 2 * BM * HD128 * sizeof(half)
                                       + GQA * BM * sizeof(float);
                        ensure_smem((const void*)flash_attn_chunk_fused_bm<HD128, GQA, BM, BLOCK>);
                        flash_attn_chunk_fused_bm<HD128, GQA, BM, BLOCK>
                            <<<fg, BLOCK, smem_bytes, stream>>>(
                                q_post_sub, k_cache_ptr, v_cache_ptr, out_sub,
                                num_q, num_kv, sub_start_pos, sub_n, scale);
                    } else {
                        constexpr int BM = 32;
                        dim3 fg(num_kv, sub_n);
                        int smem_bytes = GQA * HD128 * sizeof(half)
                                       + 2 * BM * HD128 * sizeof(half)
                                       + GQA * BM * sizeof(float);
                        flash_attn_chunk_fused<HD128, GQA, BM, BLOCK>
                            <<<fg, BLOCK, smem_bytes, stream>>>(
                                q_post_sub, k_cache_ptr, v_cache_ptr, out_sub,
                                num_q, num_kv, sub_start_pos, sub_n, scale);
                    }
                    if (g_profile_attn) g_attn_fused_ms += pa_sync_ms(tf0);
                    sub_processed += sub_n;
                    continue;
                }
                if (num_q == 24) {
                    constexpr int GQA = 6;
                    if (fa_nt == 2) {
                        // 27B GQA=6: N_T=2 → 12 active warps, BLOCK=512.
                        constexpr int BM = 32, NT = 2, NT_BLOCK = 512;
                        int smem_bytes = NT * GQA * HD * sizeof(half)
                                       + 2 * BM * HD * sizeof(half)
                                       + NT * GQA * BM * sizeof(float);
                        dim3 fg(num_kv, (sub_n + NT - 1) / NT);
                        flash_attn_chunk_fused_nt<HD, GQA, BM, NT_BLOCK, NT>
                            <<<fg, NT_BLOCK, smem_bytes, stream>>>(
                                q_post_sub, k_cache_ptr, v_cache_ptr, out_sub,
                                num_q, num_kv, sub_start_pos, sub_n, scale);
                    } else if (fa_bm == 64) {
                        constexpr int BM = 64;
                        dim3 fg(num_kv, sub_n);
                        int smem_bytes = GQA * HD * sizeof(half)
                                       + 2 * BM * HD * sizeof(half)
                                       + GQA * BM * sizeof(float);
                        ensure_smem((const void*)flash_attn_chunk_fused_bm<HD, GQA, BM, BLOCK>);
                        flash_attn_chunk_fused_bm<HD, GQA, BM, BLOCK>
                            <<<fg, BLOCK, smem_bytes, stream>>>(
                                q_post_sub, k_cache_ptr, v_cache_ptr, out_sub,
                                num_q, num_kv, sub_start_pos, sub_n, scale);
                    } else {
                        constexpr int BM = 32;
                        dim3 fg(num_kv, sub_n);
                        int smem_bytes = GQA * HD * sizeof(half)
                                       + 2 * BM * HD * sizeof(half)
                                       + GQA * BM * sizeof(float);
                        flash_attn_chunk_fused<HD, GQA, BM, BLOCK>
                            <<<fg, BLOCK, smem_bytes, stream>>>(
                                q_post_sub, k_cache_ptr, v_cache_ptr, out_sub,
                                num_q, num_kv, sub_start_pos, sub_n, scale);
                    }
                } else {  // num_q == 16, GQA = 4
                    constexpr int GQA = 4;
                    if (fa_nt == 2) {
                        // 9B GQA=4: N_T=2 → 8 active warps, BLOCK=256 (exact fit).
                        constexpr int BM = 32, NT = 2;
                        int smem_bytes = NT * GQA * HD * sizeof(half)
                                       + 2 * BM * HD * sizeof(half)
                                       + NT * GQA * BM * sizeof(float);
                        dim3 fg(num_kv, (sub_n + NT - 1) / NT);
                        flash_attn_chunk_fused_nt<HD, GQA, BM, BLOCK, NT>
                            <<<fg, BLOCK, smem_bytes, stream>>>(
                                q_post_sub, k_cache_ptr, v_cache_ptr, out_sub,
                                num_q, num_kv, sub_start_pos, sub_n, scale);
                    } else if (fa_bm == 64) {
                        constexpr int BM = 64;
                        dim3 fg(num_kv, sub_n);
                        int smem_bytes = GQA * HD * sizeof(half)
                                       + 2 * BM * HD * sizeof(half)
                                       + GQA * BM * sizeof(float);
                        ensure_smem((const void*)flash_attn_chunk_fused_bm<HD, GQA, BM, BLOCK>);
                        flash_attn_chunk_fused_bm<HD, GQA, BM, BLOCK>
                            <<<fg, BLOCK, smem_bytes, stream>>>(
                                q_post_sub, k_cache_ptr, v_cache_ptr, out_sub,
                                num_q, num_kv, sub_start_pos, sub_n, scale);
                    } else {
                        constexpr int BM = 32;
                        dim3 fg(num_kv, sub_n);
                        int smem_bytes = GQA * HD * sizeof(half)
                                       + 2 * BM * HD * sizeof(half)
                                       + GQA * BM * sizeof(float);
                        flash_attn_chunk_fused<HD, GQA, BM, BLOCK>
                            <<<fg, BLOCK, smem_bytes, stream>>>(
                                q_post_sub, k_cache_ptr, v_cache_ptr, out_sub,
                                num_q, num_kv, sub_start_pos, sub_n, scale);
                    }
                }
                if (g_profile_attn) g_attn_fused_ms += pa_sync_ms(tf0);
                sub_processed += sub_n;
                continue;
            }

            // 5. Score. Default: strict kernel (one block per score, reduction
            //    tree identical to per-token `attn_score_kernel_h`) — keeps
            //    greedy argmax bit-stable with the per-token path so Korean
            //    coding prompts don't flip. Set CHUNK_ATTN_FAST=1 for the old
            //    warp-per-score kernel (faster but ~1% argmax drift risk).
            static const bool chunk_attn_fast = getenv("CHUNK_ATTN_FAST") != nullptr;
            auto ts0 = pa_now();
            if (chunk_attn_fast) {
                int smem_bytes = (hd + 8) * sizeof(float);
                dim3 sg = score_pos_grid(num_kv, sub_seq_total);
                attn_score_kernel_h_chunk<<<sg, hd, smem_bytes, stream>>>(
                    q_post_sub, k_cache_ptr, scores_sub,
                    num_q, num_kv, hd, sub_seq_total, sub_start_pos, sub_n, scale,
                    /*row_stride=*/kv_max_seq);
            } else {
                dim3 sg(sub_seq_total, num_q, sub_n);
                attn_score_kernel_h_chunk_strict<<<sg, hd, 0, stream>>>(
                    q_post_sub, k_cache_ptr, scores_sub,
                    num_q, num_kv, hd, sub_seq_total, sub_start_pos, sub_n, scale,
                    /*row_stride=*/kv_max_seq);
            }
            if (g_profile_attn) g_attn_score_ms += pa_sync_ms(ts0);
            bool sub_has_target = (sub_processed <= target_t && target_t < sub_processed + sub_n);
            if (attn_do_dump && sub_has_target) {
                int target_t_in_sub = target_t - sub_processed;
                attn_dump_f("score",   scores_sub + (size_t)target_t_in_sub * num_q * kv_max_seq, sub_seq_total);
            }

            // 6. Softmax (one block per (t_idx, q_head) row in this sub-chunk).
            auto tm0 = pa_now();
            {
                int st = 1;
                while (st < sub_seq_total && st < 256) st <<= 1;
                dim3 sm_grid(sub_n, num_q);
                softmax_kernel_chunk<<<sm_grid, st, st * sizeof(float), stream>>>(
                    scores_sub, num_q, kv_max_seq, sub_start_pos,
                    /*wipe_end=*/sub_seq_total);
            }
            if (g_profile_attn) g_attn_softmax_ms += pa_sync_ms(tm0);
            if (attn_do_dump && sub_has_target) {
                int target_t_in_sub = target_t - sub_processed;
                attn_dump_f("softmax", scores_sub + (size_t)target_t_in_sub * num_q * kv_max_seq, sub_seq_total);
            }

            // 7. Value multiply. v2 grid: (num_kv × sub_n, d_blocks).
            auto tv0 = pa_now();
            {
                int threads = min(hd, 128);
                int d_blocks = (hd + threads - 1) / threads;
                dim3 vg(num_kv * sub_n, d_blocks);
                attn_value_kernel_h_chunk<<<vg, threads, 0, stream>>>(
                    scores_sub, v_cache_ptr, out_sub,
                    num_q, num_kv, hd, sub_seq_total, sub_start_pos, sub_n,
                    /*row_stride=*/kv_max_seq);
            }
            if (g_profile_attn) g_attn_value_ms += pa_sync_ms(tv0);
            if (attn_do_dump && sub_has_target) {
                int target_t_in_sub = target_t - sub_processed;
                attn_dump_h("value_out", out_sub + (size_t)target_t_in_sub * total_qg);
            }

            sub_processed += sub_n;
        }
        int seq_len_total = start_pos + n_tokens;  // for downstream code (unused)
        (void)seq_len_total;

        // Profiler attention dump. When MINF_DUMP_ATTN_OUT=<dir> is set we
        // write the raw post-attention output (pre-gate, pre-o_proj) for this
        // chunk to <dir>/L{layer}_S{start_pos}.bin as fp16 bytes shaped
        // [n_tokens, num_q, HD]. The offline profiler diff's dense vs sparse
        // dumps to choose the per-head pattern.
        {
            static const char* dump_dir = getenv("MINF_DUMP_ATTN_OUT");
            if (dump_dir) {
                cudaSetDevice(g);
                cudaStreamSynchronize(stream);
                size_t n_halves = (size_t)n_tokens * total_qg;
                std::vector<half> host(n_halves);
                cudaMemcpy(host.data(), ab.attn_chunk_out, n_halves * sizeof(half),
                           cudaMemcpyDeviceToHost);
                char path[512];
                snprintf(path, sizeof(path), "%s/L%d_S%d.bin", dump_dir, layer, start_pos);
                FILE* f = fopen(path, "wb");
                if (f) { fwrite(host.data(), sizeof(half), n_halves, f); fclose(f); }
            }
        }

        // 8. Batched output gate (sigmoid) — single launch for all n_tokens.
        //    Only when the model has a Q-gate (skip for ungated dense models).
        if (has_q_gate) {
            dim3 gate_grid((total_qg + 255) / 256, n_tokens);
            apply_gate_sigmoid_chunk<<<gate_grid, 256, 0, stream>>>(
                ab.attn_chunk_out, ab.attn_chunk_gate, total_qg, n_tokens);
        }
        attn_dump_h("after_gate", ab.attn_chunk_out + (size_t)target_t * total_qg);

        // 9. Batched output projection: quantize attn_chunk_out → q8_1 chunk,
        //    then chunked GEMV for o_proj.
        gpu_qi_inter[g].quantize_chunk(ab.attn_chunk_out, total_qg, n_tokens, stream);
        quant_gemv_chunk(o_w->data, o_w->type,
                         gpu_qi_inter[g].q8_buf, ab.attn_chunk_oproj,
                         total_qg, H, n_tokens, stream);
        attn_dump_h("o_proj", ab.attn_chunk_oproj + (size_t)target_t * H);

        // 10. Residual add into the fp32 hidden chunk (one launch).
        {
            int n_elem = n_tokens * H;
            add_kernel_f32<<<(n_elem + 255) / 256, 256, 0, stream>>>(
                hidden_chunk, ab.attn_chunk_oproj, n_elem);
        }
    }

    // ===== Varlen-packed attention layer (DENSE embed/rerank sidecar only) =====
    // Stage-2 embed batch packing: multiple independent texts are packed into
    // one token stream so the per-token GEMMs (norm/QKV/o_proj/MLP) sweep the
    // Q8_0 weights once per CHUNK instead of once per text — on the dp4a
    // weight-traffic-bound sidecar this is the entire win (~N× for N short
    // texts). hidden_chunk holds n_tokens packed rows at PACKED positions
    // [start_pos, start_pos+n_tokens); K/V are appended at those packed
    // positions. Attention must NOT cross segment boundaries, so instead of
    // one fused call we launch the existing dense flash kernel once per
    // segment piece with the K/V base pointer offset to the segment start and
    // start_pos remapped to the segment-LOCAL query position — the kernel
    // only ever reads keys [base, base + local_pos], which is exactly the
    // segment's causal range in the packed cache. No new attention kernel.
    //
    // pos_arr_dev[t] = segment-local position of packed row t (device int
    // array, length n_tokens) — drives RoPE. pieces = the segment spans
    // intersecting this chunk: queries [q_lo, q_lo+q_n) of the segment that
    // starts at packed position seg_start.
    //
    // Returns false BEFORE any state mutation when this layer can't take the
    // packed path (GDN layer, gated/hybrid Q, non-Q8_0 weights, non-fp16 KV,
    // non-dense attention shape, FA off) — the caller then falls back to the
    // per-text path. The 27B chat path always fails these gates, so this
    // function is unreachable for the hybrid model.
    struct VarlenPiece { int seg_start; int q_lo; int q_n; };
    bool forward_attn_chunk_varlen(int layer, float* hidden_chunk, int start_pos,
                                   int n_tokens, const int* pos_arr_dev,
                                   const std::vector<VarlenPiece>& pieces,
                                   cudaStream_t stream) {
        if (!is_attn_layer(layer)) return false;        // GDN is recurrent: can't pack
        if (use_q8_kv || use_turboquant) return false;  // fp16 KV path only
        if (!g_use_flash_attn) return false;
        int g = gpu->layer_gpu[layer];
        auto& ab = attn_bufs[g];
        auto& buf = bufs[g];
        int H = cfg.hidden_size;
        int num_q  = cfg.num_q_heads;
        int num_kv = cfg.num_kv_heads;
        int hd     = cfg.head_dim;
        int kv_dim = num_kv * hd;
        int total_qg = num_q * hd;
        float eps = cfg.rms_norm_eps;
        float scale = 1.0f / sqrtf((float)hd);
        if (!(hd == 128 && num_kv == 8 && num_q == 32)) return false;  // dense shape

        auto* norm_w   = t(blk(layer, "attn_norm.weight"));
        auto* q_w      = t(blk(layer, "attn_q.weight"));
        auto* k_w      = t(blk(layer, "attn_k.weight"));
        auto* v_w      = t(blk(layer, "attn_v.weight"));
        auto* q_norm_w = t(blk(layer, "attn_q_norm.weight"));
        auto* k_norm_w = t(blk(layer, "attn_k_norm.weight"));
        auto* o_w      = t(blk(layer, "attn_output.weight"));
        if (!norm_w || !q_w || !k_w || !v_w || !q_norm_w || !k_norm_w || !o_w)
            return false;
        int q_out_dim = q_w->dims[1];
        if (q_out_dim != total_qg) return false;        // ungated dense only
        if (q_w->type != GGML_TYPE_Q8_0 || k_w->type != GGML_TYPE_Q8_0 ||
            v_w->type != GGML_TYPE_Q8_0 || o_w->type != GGML_TYPE_Q8_0)
            return false;

        // Lazy alloc — identical shapes to forward_attn_chunk (shared buffers,
        // either function may run first).
        if (!ab.attn_chunk_q) {
            cudaMalloc(&ab.attn_chunk_q,        (size_t)CHUNK_SIZE * q_out_dim * sizeof(half));
            cudaMalloc(&ab.attn_chunk_k,        (size_t)CHUNK_SIZE * kv_dim    * sizeof(half));
            cudaMalloc(&ab.attn_chunk_v,        (size_t)CHUNK_SIZE * kv_dim    * sizeof(half));
            cudaMalloc(&ab.attn_chunk_q_post,   (size_t)CHUNK_SIZE * total_qg  * sizeof(half));
            cudaMalloc(&ab.attn_chunk_gate,     (size_t)CHUNK_SIZE * total_qg  * sizeof(half));
            cudaMalloc(&ab.attn_chunk_scores,   (size_t)ATTN_NB * num_q * kv_max_seq * sizeof(float));
            cudaMalloc(&ab.attn_chunk_out,      (size_t)CHUNK_SIZE * total_qg  * sizeof(half));
            cudaMalloc(&ab.attn_chunk_oproj,    (size_t)CHUNK_SIZE * H         * sizeof(half));
            constexpr int K_SPLITS_MAX = 16;
            cudaMalloc(&ab.attn_split_m, (size_t)num_q * ATTN_NB * K_SPLITS_MAX * sizeof(float));
            cudaMalloc(&ab.attn_split_l, (size_t)num_q * ATTN_NB * K_SPLITS_MAX * sizeof(float));
            cudaMalloc(&ab.attn_split_o, (size_t)num_q * ATTN_NB * K_SPLITS_MAX * hd * sizeof(float));
        }

        // 1. Batched RMSNorm over the whole packed chunk (segment-agnostic).
        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w(buf.mlp_chunk_norm, hidden_chunk, (float*)norm_w->data,
                                n_tokens, H, eps, stream);
        } else {
            rms_norm_f32in(buf.mlp_chunk_norm, hidden_chunk, (half*)norm_w->data,
                           n_tokens, H, eps, stream);
        }

        // 2.+3. Quantize once, batched Q/K/V projections (segment-agnostic).
        gpu_qi[g].quantize_chunk(buf.mlp_chunk_norm, H, n_tokens, stream);
        quant_gemv_chunk(q_w->data, q_w->type, gpu_qi[g].q8_buf, ab.attn_chunk_q,
                         H, q_out_dim, n_tokens, stream);
        quant_gemv_chunk(k_w->data, k_w->type, gpu_qi[g].q8_buf, ab.attn_chunk_k,
                         H, kv_dim, n_tokens, stream);
        quant_gemv_chunk(v_w->data, v_w->type, gpu_qi[g].q8_buf, ab.attn_chunk_v,
                         H, kv_dim, n_tokens, stream);

        // 4. Head-RMS + varlen RoPE. Ungated: attn_chunk_q IS q_post (skip the
        //    deinterleave/copy the gated path needs).
        half* q_post = ab.attn_chunk_q;
        int rope_dim = rope.rope_dim;
        int half_rope = rope_dim / 2;
        int tn = min(hd, 128);
        static const bool skip_qk_norm_v = getenv("SKIP_QK_NORM") != nullptr;
        if (!skip_qk_norm_v) {
            dim3 q_rms_grid(num_q, n_tokens);
            head_rms_norm_kernel_chunk<<<q_rms_grid, tn, tn * sizeof(float), stream>>>(
                q_post, (float*)q_norm_w->data, num_q, hd, eps, n_tokens);
            dim3 k_rms_grid(num_kv, n_tokens);
            head_rms_norm_kernel_chunk<<<k_rms_grid, tn, tn * sizeof(float), stream>>>(
                ab.attn_chunk_k, (float*)k_norm_w->data, num_kv, hd, eps, n_tokens);
        }
        {
            dim3 rope_q_grid((num_q  * half_rope + 255) / 256, n_tokens);
            dim3 rope_k_grid((num_kv * half_rope + 255) / 256, n_tokens);
            apply_rope_kernel_chunk_varlen<<<rope_q_grid, 256, 0, stream>>>(
                q_post, rope.sin_table(g), rope.cos_table(g),
                pos_arr_dev, num_q,  hd, rope_dim, n_tokens);
            apply_rope_kernel_chunk_varlen<<<rope_k_grid, 256, 0, stream>>>(
                ab.attn_chunk_k, rope.sin_table(g), rope.cos_table(g),
                pos_arr_dev, num_kv, hd, rope_dim, n_tokens);
        }

        // 5. fp16 K/V append at the PACKED positions.
        auto& kv = kv_caches[layer];
        {
            size_t new_bytes = (size_t)n_tokens * kv_dim * sizeof(half);
            cudaMemcpyAsync(kv.k + (size_t)start_pos * kv_dim, ab.attn_chunk_k,
                            new_bytes, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(kv.v + (size_t)start_pos * kv_dim, ab.attn_chunk_v,
                            new_bytes, cudaMemcpyDeviceToDevice, stream);
        }

        // 6. Per-piece flash attention. Base pointer offset to the segment
        //    start + segment-local start_pos = causal range [seg_start,
        //    seg_start+local_pos] in the packed cache. ATTN_NB sub-chunking
        //    mirrors the dense dispatch in forward_attn_chunk.
        for (const auto& pc : pieces) {
            int done = 0;
            while (done < pc.q_n) {
                int sub_n = std::min(ATTN_NB, pc.q_n - done);
                int row0 = pc.q_lo - start_pos + done;        // chunk-row offset
                int local_start = pc.q_lo - pc.seg_start + done;
                constexpr int HD128 = 128, GQA = 4, BM = 32, BLOCK = 256;
                int smem_bytes = GQA * HD128 * sizeof(half)
                               + 2 * BM * HD128 * sizeof(half)
                               + GQA * BM * sizeof(float);
                dim3 fg(num_kv, sub_n);
                flash_attn_chunk_fused<HD128, GQA, BM, BLOCK>
                    <<<fg, BLOCK, smem_bytes, stream>>>(
                        q_post + (size_t)row0 * total_qg,
                        kv.k + (size_t)pc.seg_start * kv_dim,
                        kv.v + (size_t)pc.seg_start * kv_dim,
                        ab.attn_chunk_out + (size_t)row0 * total_qg,
                        num_q, num_kv, local_start, sub_n, scale);
                done += sub_n;
            }
        }

        // 7. Batched o_proj + residual over the whole packed chunk.
        gpu_qi_inter[g].quantize_chunk(ab.attn_chunk_out, total_qg, n_tokens, stream);
        quant_gemv_chunk(o_w->data, o_w->type,
                         gpu_qi_inter[g].q8_buf, ab.attn_chunk_oproj,
                         total_qg, H, n_tokens, stream);
        {
            int n_elem = n_tokens * H;
            add_kernel_f32<<<(n_elem + 255) / 256, 256, 0, stream>>>(
                hidden_chunk, ab.attn_chunk_oproj, n_elem);
        }
        return true;
    }

    // ============ Chunked MLP forward (batched, prefill fast path) ============
    // Processes n_tokens hidden states through one MLP layer with shared
    // weight loads. Each weight word in ffn_gate/ffn_up/ffn_down is read
    // ONCE per output row and dp4a'd against up to NB token inputs at the
    // same time (NB=16/8/4 peeling inside quant_gemv_chunk). For Q8_0 27B
    // this collapses 64 sequential GEMVs into ~4 batched calls per
    // projection — same total compute, ~1/16 the memory traffic, which is
    // the actual prefill bottleneck on HBM2-bound Volta.
    void forward_mlp_chunk(int layer, float* hidden_chunk, int n_tokens, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& buf = bufs[g];
        int H = cfg.hidden_size;
        int I = cfg.intermediate_size;

        // Qwopus3.6 hybrid uses `post_attention_norm`; Qwen3 dense uses
        // `ffn_norm`. Try the hybrid name first then fall back.
        auto* norm_w = t(blk(layer, "post_attention_norm.weight"));
        if (!norm_w) norm_w = t(blk(layer, "ffn_norm.weight"));
        auto* gate_w = t(blk(layer, "ffn_gate.weight"));
        auto* up_w   = t(blk(layer, "ffn_up.weight"));
        auto* down_w = t(blk(layer, "ffn_down.weight"));
        if (!norm_w || !gate_w || !up_w || !down_w) {
            printf("MLP L%d: missing weights!\n", layer);
            return;
        }

        // Q8_0 path uses the new chunked dispatch. Q5_K / Q6_K and any
        // mixed quant fall back to the old per-token loop — those models
        // don't use the chunked GEMV anyway, and per-token still works.
        bool can_batch = (gate_w->type == GGML_TYPE_Q8_0)
                      && (up_w->type   == GGML_TYPE_Q8_0)
                      && (down_w->type == GGML_TYPE_Q8_0);
        if (!can_batch) {
            for (int tt = 0; tt < n_tokens; tt++) {
                forward_mlp(layer, hidden_chunk + (size_t)tt * H, stream);
            }
            return;
        }

        auto mlp_prof_now = [](){ return std::chrono::high_resolution_clock::now(); };
        auto mlp_prof_sync_ms = [&](std::chrono::high_resolution_clock::time_point tb){
            cudaStreamSynchronize(stream);
            auto te = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(te - tb).count();
        };
        auto t0 = mlp_prof_now();

        // 1. Batched RMSNorm: rows = n_tokens, all writing into mlp_chunk_norm.
        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w(buf.mlp_chunk_norm, hidden_chunk, (float*)norm_w->data,
                                n_tokens, H, cfg.rms_norm_eps, stream);
        } else {
            rms_norm_f32in(buf.mlp_chunk_norm, hidden_chunk, (half*)norm_w->data,
                           n_tokens, H, cfg.rms_norm_eps, stream);
        }
        if (g_profile_mlp) { g_mlp_norm_ms += mlp_prof_sync_ms(t0); t0 = mlp_prof_now(); }

        // 2. Quantize all n_tokens × H normed values in one shot.
        gpu_qi[g].quantize_chunk(buf.mlp_chunk_norm, H, n_tokens, stream);
        if (g_profile_mlp) { g_mlp_q1_ms += mlp_prof_sync_ms(t0); t0 = mlp_prof_now(); }

        // 3. ffn_gate + ffn_up projections. MLP_GATEUP_FUSED=1 uses a fused
        // kernel that loads the shared Q8 input tile once and computes both
        // gate and up outputs in one launch. Default OFF (env-gated until
        // measured).
        static const bool fuse_gateup = getenv("MLP_GATEUP_FUSED") != nullptr;
        if (fuse_gateup && gate_w->type == GGML_TYPE_Q8_0 && up_w->type == GGML_TYPE_Q8_0) {
            quant_gemv_chunk_fused2(gate_w->data, up_w->data, gate_w->type,
                                    gpu_qi[g].q8_buf,
                                    buf.mlp_chunk_gate, buf.mlp_chunk_up,
                                    H, I, n_tokens, stream);
            if (g_profile_mlp) { g_mlp_gate_ms += mlp_prof_sync_ms(t0); t0 = mlp_prof_now(); }
            // up phase rolled into gate timer when fused.
        } else {
            // 3a. Batched ffn_gate projection.
            quant_gemv_chunk(gate_w->data, gate_w->type,
                             gpu_qi[g].q8_buf, buf.mlp_chunk_gate,
                             H, I, n_tokens, stream);
            if (g_profile_mlp) { g_mlp_gate_ms += mlp_prof_sync_ms(t0); t0 = mlp_prof_now(); }

            // 3b. Batched ffn_up projection.
            quant_gemv_chunk(up_w->data,   up_w->type,
                             gpu_qi[g].q8_buf, buf.mlp_chunk_up,
                             H, I, n_tokens, stream);
            if (g_profile_mlp) { g_mlp_up_ms += mlp_prof_sync_ms(t0); t0 = mlp_prof_now(); }
        }

        // 4. Fused SiLU(gate) * up across the whole chunk.
        {
            int n_elem = n_tokens * I;
            silu_mul_kernel<<<(n_elem + 255) / 256, 256, 0, stream>>>(
                buf.mlp_chunk_gate, buf.mlp_chunk_up, n_elem);
        }
        if (g_profile_mlp) { g_mlp_silu_ms += mlp_prof_sync_ms(t0); t0 = mlp_prof_now(); }

        // 4b. (Optional) Measure activation sparsity for the down_proj input.
        //     MEASURE_SPARSITY=1: prints per-layer energy distribution once
        //     (first chunk of each layer, first prefill only).
        {
            static const bool measure_sparsity = getenv("MEASURE_SPARSITY") != nullptr;
            // Track which layers have been measured (up to 128 layers).
            static bool layer_measured[128] = {};
            if (measure_sparsity && layer < 128 && !layer_measured[layer]) {
                layer_measured[layer] = true;
                int n_blocks = I / 32;  // 544 for I=17408

                // Lazy-allocate a small device buffer for block magnitudes.
                static float* d_block_mag = nullptr;
                static float* h_block_mag = nullptr;
                if (!d_block_mag) {
                    cudaMalloc(&d_block_mag, n_blocks * sizeof(float));
                    h_block_mag = (float*)malloc(n_blocks * sizeof(float));
                }

                // Launch the measurement kernel.
                int threads_mag = 256;
                int grid_mag = (n_blocks + threads_mag - 1) / threads_mag;
                block_magnitude_kernel<<<grid_mag, threads_mag, 0, stream>>>(
                    buf.mlp_chunk_gate, d_block_mag, I, n_tokens, n_blocks);

                // Copy results to host.
                cudaMemcpyAsync(h_block_mag, d_block_mag,
                                n_blocks * sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);

                // Compute statistics: sort magnitudes descending.
                std::vector<float> mags(h_block_mag, h_block_mag + n_blocks);
                float total_mag = 0.0f;
                for (float m : mags) total_mag += m;
                std::sort(mags.begin(), mags.end(), std::greater<float>());

                // Cumulative energy at various keep ratios.
                float cum = 0.0f;
                float energy_at[5];  // energy% at keep 20/40/50/60/80
                const float check_ratios[] = {0.20f, 0.40f, 0.50f, 0.60f, 0.80f};
                int ci = 0;
                for (int i = 0; i < n_blocks && ci < 5; i++) {
                    cum += mags[i];
                    float frac = (float)(i + 1) / n_blocks;
                    while (ci < 5 && frac >= check_ratios[ci] - 1e-6f) {
                        energy_at[ci] = (total_mag > 0) ? cum / total_mag * 100.0f : 0.0f;
                        ci++;
                    }
                }
                while (ci < 5) { energy_at[ci++] = 100.0f; }

                // Count near-zero blocks.
                float per_tok = (n_tokens > 0) ? (float)n_tokens : 1.0f;
                int cnt_01 = 0;
                for (int i = 0; i < n_blocks; i++) {
                    float avg = h_block_mag[i] / (per_tok * 32.0f);
                    if (avg < 0.1f) cnt_01++;
                }

                printf("[SPARSITY] L%02d ntok=%d nblk=%d | "
                       "keep20%%=%.1f%% keep40%%=%.1f%% keep50%%=%.1f%% "
                       "keep60%%=%.1f%% keep80%%=%.1f%% | "
                       "near-zero(<0.1)=%d/%d(%.0f%%)\n",
                       layer, n_tokens, n_blocks,
                       energy_at[0], energy_at[1], energy_at[2],
                       energy_at[3], energy_at[4],
                       cnt_01, n_blocks, 100.0f * cnt_01 / n_blocks);
            }
        }

        // 5. Quantize the (silu_mul'd) intermediate chunk for the down proj.
        gpu_qi_inter[g].quantize_chunk(buf.mlp_chunk_gate, I, n_tokens, stream);
        if (g_profile_mlp) { g_mlp_q2_ms += mlp_prof_sync_ms(t0); t0 = mlp_prof_now(); }

        // 6. Batched ffn_down projection.
        quant_gemv_chunk(down_w->data, down_w->type,
                         gpu_qi_inter[g].q8_buf, buf.mlp_chunk_down,
                         I, H, n_tokens, stream);
        if (g_profile_mlp) { g_mlp_down_ms += mlp_prof_sync_ms(t0); t0 = mlp_prof_now(); }

        // 7. Residual add.
        {
            int n_elem = n_tokens * H;
            add_kernel_f32<<<(n_elem + 255) / 256, 256, 0, stream>>>(
                hidden_chunk, buf.mlp_chunk_down, n_elem);
        }
        if (g_profile_mlp) { g_mlp_resi_ms += mlp_prof_sync_ms(t0); g_mlp_calls++; }
    }

#if 0  // unused — kept for reference, signatures need updating to fp32 hidden
    // ============ Full Pipeline Forward (1 token) ============
    void forward_one_token(half* hidden, int pos, int gpu_start = 0) {
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            int g = gpu->layer_gpu[layer];
            
            // Transfer hidden to correct GPU if needed
            int prev_g = (layer == 0) ? gpu_start : gpu->layer_gpu[layer - 1];
            if (g != prev_g) {
                half* new_hidden;
                cudaSetDevice(g);
                cudaMalloc(&new_hidden, cfg.hidden_size * sizeof(half));
                cudaMemcpy(new_hidden, hidden, cfg.hidden_size * sizeof(half), cudaMemcpyDeviceToDevice);
                if (layer > 0) {
                    cudaSetDevice(prev_g);
                    // Don't free — reuse buffer. Actually we need persistent buffers.
                }
                hidden = new_hidden;
                // TODO: use persistent transfer buffers
            }
            
            cudaSetDevice(g);
            cudaStream_t stream = 0;  // default stream for simplicity
            
            if (is_attn_layer(layer)) {
                // Attention + MLP
                forward_attn(layer, hidden, pos, stream);
            } else {
                // GDN + MLP
                forward_gdn(layer, hidden, stream);
            }
            
            // MLP (shared by both layer types)
            forward_mlp(layer, hidden, stream);
        }
    }

    // Full pipeline benchmark
    void benchmark_pipeline() {
        int H = cfg.hidden_size;
        
        // Allocate persistent hidden state on GPU 0
        cudaSetDevice(0);
        half* hidden;
        cudaMalloc(&hidden, H * sizeof(half));
        
        // Fill with test data
        std::vector<half> h_data(H);
        for (int i = 0; i < H; i++) h_data[i] = __float2half(0.01f * (i % 100 - 50));
        cudaMemcpy(hidden, h_data.data(), H * sizeof(half), cudaMemcpyHostToDevice);
        
        // Allocate transfer buffers on each GPU
        half* gpu_hidden[4];
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&gpu_hidden[g], H * sizeof(half));
        }
        
        // Reset GDN states
        reset_all_states();
        
        printf("\n=== Full Pipeline Benchmark ===\n");
        
        // Warmup: 2 tokens
        for (int tok = 0; tok < 2; tok++) {
            cudaMemcpy(gpu_hidden[0], hidden, H * sizeof(half), cudaMemcpyDeviceToDevice);
            half* h = gpu_hidden[0];
            
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int g = gpu->layer_gpu[layer];
                int prev_g = (layer == 0) ? 0 : gpu->layer_gpu[layer - 1];
                
                if (g != prev_g) {
                    cudaSetDevice(prev_g);
                    cudaDeviceSynchronize();
                    cudaMemcpy(gpu_hidden[g], h, H * sizeof(half), cudaMemcpyDeviceToDevice);
                    h = gpu_hidden[g];
                }
                
                cudaSetDevice(g);
                if (is_attn_layer(layer))
                    forward_attn(layer, h, tok, 0);
                else
                    forward_gdn(layer, h, 0);
                forward_mlp(layer, h, 0);
            }
            for (int g = 0; g < gpu->num_gpus; g++) {
                cudaSetDevice(g);
                cudaDeviceSynchronize();
            }
        }
        
        // Benchmark: N tokens
        int N = 20;
        
        // Sync all GPUs
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaDeviceSynchronize();
        }
        
        auto t0 = std::chrono::high_resolution_clock::now();
        
        for (int tok = 0; tok < N; tok++) {
            cudaMemcpy(gpu_hidden[0], hidden, H * sizeof(half), cudaMemcpyDeviceToDevice);
            half* h = gpu_hidden[0];
            
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int g = gpu->layer_gpu[layer];
                int prev_g = (layer == 0) ? 0 : gpu->layer_gpu[layer - 1];
                
                if (g != prev_g) {
                    // Sync previous GPU before transfer
                    cudaSetDevice(prev_g);
                    cudaDeviceSynchronize();
                    cudaMemcpy(gpu_hidden[g], h, H * sizeof(half), cudaMemcpyDeviceToDevice);
                    h = gpu_hidden[g];
                }
                
                cudaSetDevice(g);
                if (is_attn_layer(layer))
                    forward_attn(layer, h, tok + 2, 0);
                else
                    forward_gdn(layer, h, 0);
                forward_mlp(layer, h, 0);
            }
            
            // Sync all GPUs
            for (int g = 0; g < gpu->num_gpus; g++) {
                cudaSetDevice(g);
                cudaDeviceSynchronize();
            }
        }
        
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_per_tok = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
        double tps = 1000.0 / ms_per_tok;
        
        // Print result on last GPU
        int last_g = gpu->layer_gpu[cfg.num_layers - 1];
        half host_buf[8];
        cudaMemcpy(host_buf, gpu_hidden[last_g], 8 * sizeof(half), cudaMemcpyDeviceToHost);
        
        printf("Pipeline: %.1f ms/token = %.1f t/s\n", ms_per_tok, tps);
        printf("Output[0:4]: %.4f %.4f %.4f %.4f\n",
            __half2float(host_buf[0]), __half2float(host_buf[1]),
            __half2float(host_buf[2]), __half2float(host_buf[3]));
        
        // Breakdown estimate
        printf("\nBreakdown (from individual benchmarks):\n");
        printf("  GDN: ~%.0f us x 48 = %.1f ms\n", 330.0, 330.0 * 48 / 1000);
        printf("  Attn: ~%.0f us x 16 = %.1f ms\n", 335.0, 335.0 * 16 / 1000);
        printf("  MLP: ~%.0f us x 64 = %.1f ms\n", 524.0, 524.0 * 64 / 1000);
        printf("  Sum: ~%.1f ms (vs actual pipeline)\n", 
            (330.0*48 + 335.0*16 + 524.0*64) / 1000);
        
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaFree(gpu_hidden[g]);
        }
    }
#endif

    // test_gdn / test_mlp removed — needed updating to fp32 hidden

    // ============ N=2 sequential wrappers for spec decoding ============
    // forward_attn / forward_gdn are inherently sequential because of KV cache
    // append + GDN recurrent state. We just call them twice with the two
    // positions / hidden states. The "speedup" comes entirely from MLP
    // batching (forward_mlp_n2) which shares weight loads across both tokens.

    // Batched attention forward for two tokens. The 4 projections (q, k, v,
    // o) are run via quant_gemv_n2 sharing weight loads. The per-token bits
    // (deinterleave, head norm, RoPE, KV append, Q@K, softmax, @V, gate
    // sigmoid) are inherently sequential per-position.
    void forward_attn_n2(int layer, float* hidden_a, float* hidden_b,
                         int pos_a, int pos_b, cudaStream_t stream, int slot = 0) {
        int g = gpu->layer_gpu[layer];
        int H = cfg.hidden_size;
        int num_q = cfg.num_q_heads;
        int num_kv = cfg.num_kv_heads;
        int hd = cfg.head_dim;
        float eps = cfg.rms_norm_eps;
        float scale = 1.0f / sqrtf((float)hd);
        auto& abA = attn_bufs[g];   auto& abB = attn_bufs2[g];
        auto& bA  = bufs[g];        auto& bB  = bufs2[g];
        // Continuous batching: slot KV virtual offset (0 → no behavior change).
        size_t slot_kv_off = kv_slot_offset(slot) * (num_kv * hd);

        auto* norm_w   = t(blk(layer, "attn_norm.weight"));
        auto* q_w      = t(blk(layer, "attn_q.weight"));
        auto* k_w      = t(blk(layer, "attn_k.weight"));
        auto* v_w      = t(blk(layer, "attn_v.weight"));
        auto* q_norm_w = t(blk(layer, "attn_q_norm.weight"));
        auto* k_norm_w = t(blk(layer, "attn_k_norm.weight"));
        auto* o_w      = t(blk(layer, "attn_output.weight"));

        int total_qg = num_q * hd;
        int kv_dim   = num_kv * hd;
        int q_out_dim = q_w->dims[1];

        // 1. RMSNorm both tokens (fused n2)
        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w_n2(bA.norm_out, bB.norm_out,
                                   hidden_a, hidden_b,
                                   (float*)norm_w->data, H, eps, stream);
        } else {
            rms_norm_f32in_n2(bA.norm_out, bB.norm_out,
                              hidden_a, hidden_b,
                              (half*)norm_w->data, H, eps, stream);
        }

        // 2. Quantize both norm outputs
        gpu_qi[g].quantize(bA.norm_out, H, stream);
        gpu_qi2[g].quantize(bB.norm_out, H, stream);

        // 3. Batched Q / K / V projections
        quant_gemv_n2(q_w->data, q_w->type,
                      bA.norm_out, bB.norm_out,
                      abA.q_proj, abB.q_proj,
                      H, q_out_dim, &gpu_qi[g], &gpu_qi2[g], stream);
        quant_gemv_n2(k_w->data, k_w->type,
                      bA.norm_out, bB.norm_out,
                      abA.k_proj, abB.k_proj,
                      H, kv_dim, &gpu_qi[g], &gpu_qi2[g], stream);
        quant_gemv_n2(v_w->data, v_w->type,
                      bA.norm_out, bB.norm_out,
                      abA.v_proj, abB.v_proj,
                      H, kv_dim, &gpu_qi[g], &gpu_qi2[g], stream);

        // 4. Per-token attention compute (deinterleave, norm, RoPE, KV
        //    append, Q@K, softmax, @V, gate sigmoid). Lambda to avoid
        //    duplicating ~30 lines.
        int rope_dim = rope.rope_dim;
        int half_rope = rope_dim / 2;
        auto run_one = [&](AttnBuffers& ab, int pos) {
            half* q_buf    = ab.attn_out;
            half* gate_buf = ab.gate_buf;
            deinterleave_qg_kernel<<<(total_qg+255)/256, 256, 0, stream>>>(
                ab.q_proj, q_buf, gate_buf, num_q, hd);
            int tn = min(hd, 128);
            head_rms_norm_kernel<<<num_q, tn, tn * sizeof(float), stream>>>(
                q_buf, (float*)q_norm_w->data, num_q, hd, eps);
            head_rms_norm_kernel<<<num_kv, tn, tn * sizeof(float), stream>>>(
                ab.k_proj, (float*)k_norm_w->data, num_kv, hd, eps);
            float* sin_pos = rope.sin_table(g) + (size_t)pos * half_rope;
            float* cos_pos = rope.cos_table(g) + (size_t)pos * half_rope;
            apply_rope_kernel<<<(num_q  * half_rope + 255)/256, 256, 0, stream>>>(
                q_buf, sin_pos, cos_pos, num_q, hd, rope_dim);
            apply_rope_kernel<<<(num_kv * half_rope + 255)/256, 256, 0, stream>>>(
                ab.k_proj, sin_pos, cos_pos, num_kv, hd, rope_dim);
            if (use_q8_kv) {
                auto& q8 = q8_kv_caches[layer];
                int bpt = q8.blocks_per_token;
                size_t slot_blk_off = kv_slot_offset(slot) * bpt;
                size_t off = slot_blk_off + (size_t)pos * bpt;
                quantize_kv_q8_0_kern<<<bpt, 32, 0, stream>>>(ab.k_proj, q8.k + off, bpt);
                quantize_kv_q8_0_kern<<<bpt, 32, 0, stream>>>(ab.v_proj, q8.v + off, bpt);
            } else if (use_turboquant) {
                auto& tq = tq_kv_caches[layer];
                int bpt = tq.blocks_per_token;
                size_t slot_blk_off = kv_slot_offset(slot) * bpt;
                size_t off = slot_blk_off + (size_t)pos * bpt;
                tq3_quantize_kernel<<<(bpt+31)/32, 32, 0, stream>>>(ab.k_proj, &tq.k[off], bpt);
                tq3_quantize_kernel<<<(bpt+31)/32, 32, 0, stream>>>(ab.v_proj, &tq.v[off], bpt);
                if (fp16_dec_cache_on) {
                    auto& fc = fp16_dec_caches[layer];
                    half* fp16_k_pos = fc.k + slot_kv_off + (size_t)pos * kv_dim;
                    half* fp16_v_pos = fc.v + slot_kv_off + (size_t)pos * kv_dim;
                    cudaMemcpyAsync(fp16_k_pos, ab.k_proj, kv_dim * sizeof(half),
                                    cudaMemcpyDeviceToDevice, stream);
                    cudaMemcpyAsync(fp16_v_pos, ab.v_proj, kv_dim * sizeof(half),
                                    cudaMemcpyDeviceToDevice, stream);
                }
            } else {
                auto& kv = kv_caches[layer];
                half* k_cache_pos = kv.k + slot_kv_off + (size_t)pos * kv_dim;
                half* v_cache_pos = kv.v + slot_kv_off + (size_t)pos * kv_dim;
                cudaMemcpyAsync(k_cache_pos, ab.k_proj, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(v_cache_pos, ab.v_proj, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            }
            int seq_len = pos + 1;
            dim3 score_grid = score_pos_grid(num_q, seq_len);
            static const bool tq_fused = getenv("TQ_FUSED") != nullptr;
            // Pick fp16 source: legacy fp16 KV, fp16 mirror, or Q8 bulk-dequant
            // into the per-GPU scratch (re-used from the TQ chunked path).
            half* k_src_fp16 = nullptr;
            half* v_src_fp16 = nullptr;
            if (use_q8_kv) {
                auto& q8 = q8_kv_caches[layer];
                int bpt = q8.blocks_per_token;
                int n_blocks_total = seq_len * bpt;
                size_t slot_off_blocks = kv_slot_offset(slot) * bpt;
                dim3 dq_grid((n_blocks_total + 7) / 8);
                dim3 dq_block(32, 8);
                dequantize_kv_q8_0_kern<<<dq_grid, dq_block, 0, stream>>>(
                    q8.k + slot_off_blocks, tq_k_buf[g] + slot_kv_off, n_blocks_total);
                dequantize_kv_q8_0_kern<<<dq_grid, dq_block, 0, stream>>>(
                    q8.v + slot_off_blocks, tq_v_buf[g] + slot_kv_off, n_blocks_total);
                k_src_fp16 = tq_k_buf[g] + slot_kv_off;
                v_src_fp16 = tq_v_buf[g] + slot_kv_off;
            } else if (!use_turboquant) {
                auto& kv = kv_caches[layer];
                k_src_fp16 = kv.k + slot_kv_off;
                v_src_fp16 = kv.v + slot_kv_off;
            } else if (fp16_dec_cache_on) {
                auto& fc = fp16_dec_caches[layer];
                k_src_fp16 = fc.k + slot_kv_off;
                v_src_fp16 = fc.v + slot_kv_off;
            }
            if (use_turboquant && !fp16_dec_cache_on && !use_q8_kv) {
                auto& tq = tq_kv_caches[layer];
                size_t slot_off_blocks_int = kv_slot_offset(slot) * (size_t)tq.blocks_per_token;
                block_tq3* tq_k_slot = tq.k + slot_off_blocks_int;
                if (tq_fused) {
                    dim3 tq_score_grid = score_pos_grid(num_kv, seq_len);
                    int smem_bytes = hd * sizeof(float);
                    attn_score_kernel_tq3<<<tq_score_grid, hd, smem_bytes, stream>>>(
                        q_buf, tq_k_slot, ab.attn_scores,
                        num_q, num_kv, hd, seq_len, scale);
                } else {
                    int n_blocks_total = seq_len * tq.blocks_per_token;
                    tq3_dequantize_kernel<<<(n_blocks_total + 31)/32, 32, 0, stream>>>(
                        tq_k_slot, tq_k_buf[g], n_blocks_total);
                    attn_score_kernel_h<<<score_grid, min(hd, 256), 0, stream>>>(
                        q_buf, tq_k_buf[g], ab.attn_scores, num_q, num_kv, hd, seq_len, scale);
                }
            } else {
                attn_score_kernel_h<<<score_grid, min(hd, 256), 0, stream>>>(
                    q_buf, k_src_fp16, ab.attn_scores, num_q, num_kv, hd, seq_len, scale);
            }
            { int st = 1; while(st < seq_len && st < 256) st <<= 1;
            softmax_kernel<<<num_q, st, st * sizeof(float), stream>>>(
                ab.attn_scores, num_q, seq_len); }
            if (use_turboquant && !fp16_dec_cache_on && !use_q8_kv) {
                auto& tq = tq_kv_caches[layer];
                size_t slot_off_blocks_int = kv_slot_offset(slot) * (size_t)tq.blocks_per_token;
                block_tq3* tq_v_slot = tq.v + slot_off_blocks_int;
                if (tq_fused) {
                    int blocks_per_kv_head = hd / TQ_BLOCK_SIZE;
                    dim3 tq_value_grid(num_kv, blocks_per_kv_head);
                    int smem_bytes = TQ_BLOCK_SIZE * sizeof(float);
                    attn_value_kernel_tq3<<<tq_value_grid, TQ_BLOCK_SIZE, smem_bytes, stream>>>(
                        ab.attn_scores, tq_v_slot, q_buf,
                        num_q, num_kv, hd, seq_len);
                } else {
                    int n_blocks_total = seq_len * tq.blocks_per_token;
                    tq3_dequantize_kernel<<<(n_blocks_total + 31)/32, 32, 0, stream>>>(
                        tq_v_slot, tq_v_buf[g], n_blocks_total);
                    attn_value_kernel_h<<<num_q, min(hd, 256), 0, stream>>>(
                        ab.attn_scores, tq_v_buf[g], q_buf, num_q, num_kv, hd, seq_len);
                }
            } else {
                attn_value_kernel_h<<<num_q, min(hd, 256), 0, stream>>>(
                    ab.attn_scores, v_src_fp16, q_buf, num_q, num_kv, hd, seq_len);
            }
            apply_gate_sigmoid<<<(total_qg+255)/256, 256, 0, stream>>>(
                q_buf, gate_buf, total_qg);
        };
        run_one(abA, pos_a);
        run_one(abB, pos_b);

        // 5. Batched output projection
        gpu_qi_inter[g].quantize(abA.attn_out,  num_q * hd, stream);
        gpu_qi_inter2[g].quantize(abB.attn_out, num_q * hd, stream);
        quant_gemv_n2(o_w->data, o_w->type,
                      abA.attn_out, abB.attn_out,
                      bA.mlp_down,  bB.mlp_down,
                      num_q * hd, H, &gpu_qi_inter[g], &gpu_qi_inter2[g], stream);

        // 6. Residual add into FP32 hidden (both tokens, fused n2)
        { dim3 ag((H+255)/256, 2);
          add_kernel_f32_n2<<<ag, 256, 0, stream>>>(
              hidden_a, bA.mlp_down,
              hidden_b, bB.mlp_down, H); }
    }

    // Batched GDN forward for two tokens. The 5 projections (qkv, gate,
    // alpha, beta, output) are run via quant_gemv_n2 sharing weight loads,
    // saving roughly half the GDN GEMV memory traffic. The recurrent path
    // (conv1d, gdn_recurrent_step, rms_norm_gated) is inherently per-token
    // and runs sequentially with a state snapshot taken between the two
    // tokens so a rejected second token can be rolled back exactly.
    void forward_gdn_n2(int layer, float* hidden_a, float* hidden_b, cudaStream_t stream, int slot = 0) {
        int g = gpu->layer_gpu[layer];
        auto& bA = bufs[g];      auto& bB = bufs2[g];
        auto& gA = gdn_bufs[g];  auto& gB = gdn_bufs2[g];
        int H = cfg.hidden_size;
        int num_k = cfg.linear_k_heads;
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        int qkv_dim = 2 * num_k * k_dim + num_v * v_dim;
        int v_total = num_v * v_dim;
        float* conv_state_slot = gdn_conv_slot(layer, slot);
        float* rec_state_slot  = gdn_rec_slot(layer, slot);

        auto* norm_w  = t(blk(layer, "attn_norm.weight"));
        auto* qkv_w   = t(blk(layer, "attn_qkv.weight"));
        auto* gate_w  = t(blk(layer, "attn_gate.weight"));
        auto* alpha_w = t(blk(layer, "ssm_alpha.weight"));
        auto* beta_w  = t(blk(layer, "ssm_beta.weight"));
        auto* conv_w  = t(blk(layer, "ssm_conv1d.weight"));
        auto* a_log_t = t(blk(layer, "ssm_a"));
        auto* dt_bias_t = t(blk(layer, "ssm_dt.bias"));
        auto* ssm_norm_w = t(blk(layer, "ssm_norm.weight"));
        auto* out_w   = t(blk(layer, "ssm_out.weight"));

        // 1. RMSNorm both tokens (fused n2)
        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w_n2(bA.norm_out, bB.norm_out,
                                   hidden_a, hidden_b,
                                   (float*)norm_w->data, H, cfg.rms_norm_eps, stream);
        } else {
            rms_norm_f32in_n2(bA.norm_out, bB.norm_out,
                              hidden_a, hidden_b,
                              (half*)norm_w->data, H, cfg.rms_norm_eps, stream);
        }

        // 2. Quantize both norm outputs once
        gpu_qi[g].quantize(bA.norm_out, H, stream);
        gpu_qi2[g].quantize(bB.norm_out, H, stream);

        // 3. Batched projections — qkv (largest), gate, alpha, beta
        // Per-token output buffers reuse the existing aliases used by forward_gdn:
        //   qkv_out → attn_out, z_out → mlp_gate, a_out → mlp_up, b_out → mlp_down
        quant_gemv_n2(qkv_w->data, qkv_w->type,
                      bA.norm_out, bB.norm_out,
                      bA.attn_out, bB.attn_out,
                      H, qkv_dim, &gpu_qi[g], &gpu_qi2[g], stream);
        quant_gemv_n2(gate_w->data, gate_w->type,
                      bA.norm_out, bB.norm_out,
                      bA.mlp_gate, bB.mlp_gate,
                      H, num_v * v_dim, &gpu_qi[g], &gpu_qi2[g], stream);
        quant_gemv_n2(alpha_w->data, alpha_w->type,
                      bA.norm_out, bB.norm_out,
                      bA.mlp_up, bB.mlp_up,
                      H, num_v, &gpu_qi[g], &gpu_qi2[g], stream);
        quant_gemv_n2(beta_w->data, beta_w->type,
                      bA.norm_out, bB.norm_out,
                      bA.mlp_down, bB.mlp_down,
                      H, num_v, &gpu_qi[g], &gpu_qi2[g], stream);

        // 4. Token A: conv1d + recurrent + gated norm  (advances state to post-a)
        int kw = 4;
        int threads_conv = min(qkv_dim, 256);
        int blocks_conv  = (qkv_dim + threads_conv - 1) / threads_conv;
        conv1d_update_silu<<<blocks_conv, threads_conv, 0, stream>>>(
            conv_state_slot, bA.attn_out, (float*)conv_w->data,
            gA.conv_out, qkv_dim, kw);
        int gdn_smem = (2 * cfg.linear_k_dim + 1) * sizeof(float);
        launch_gdn_recurrent_step(num_v, min(v_dim, 128), gdn_smem, stream,
            gA.conv_out, (float*)a_log_t->data, (float*)dt_bias_t->data,
            bA.mlp_up, bA.mlp_down,
            rec_state_slot, gA.core_out,
            num_k, num_v, k_dim, v_dim);
        rms_norm_gated_kernel<<<num_v, min(v_dim, 128), 128 * sizeof(float), stream>>>(
            gA.core_out, bA.mlp_gate, (float*)ssm_norm_w->data, gA.normed_out,
            num_v, v_dim, 1e-6f);

        // 5. Snapshot GDN state for THIS layer (post-a, for reject rollback).
        // Snapshot is taken into the per-slot backing store at `slot`, mirroring
        // the live conv_state_slot/rec_state_slot. With num_slots==1 (current
        // single-stream MTP) slot==0 and this is bit-identical to the original
        // single-buffer snapshot. restore_gdn_states(slot) reverses it.
        if (snapshots_ready && gdn_snapshots[layer].conv_state) {
            size_t conv_sz = qkv_dim * 4 * sizeof(float);
            size_t rec_sz  = (size_t)num_v * k_dim * v_dim * sizeof(float);
            cudaMemcpyAsync(gdn_snap_conv_slot(layer, slot), conv_state_slot,
                            conv_sz, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(gdn_snap_rec_slot(layer, slot), rec_state_slot,
                            rec_sz, cudaMemcpyDeviceToDevice, stream);
        }

        // 6. Token B: conv1d + recurrent + gated norm  (advances state to post-b)
        conv1d_update_silu<<<blocks_conv, threads_conv, 0, stream>>>(
            conv_state_slot, bB.attn_out, (float*)conv_w->data,
            gB.conv_out, qkv_dim, kw);
        launch_gdn_recurrent_step(num_v, min(v_dim, 128), gdn_smem, stream,
            gB.conv_out, (float*)a_log_t->data, (float*)dt_bias_t->data,
            bB.mlp_up, bB.mlp_down,
            rec_state_slot, gB.core_out,
            num_k, num_v, k_dim, v_dim);
        rms_norm_gated_kernel<<<num_v, min(v_dim, 128), 128 * sizeof(float), stream>>>(
            gB.core_out, bB.mlp_gate, (float*)ssm_norm_w->data, gB.normed_out,
            num_v, v_dim, 1e-6f);

        // 7. Batched output projection — both normed_out → proj_out
        gpu_qi_inter[g].quantize(gA.normed_out,  num_v * v_dim, stream);
        gpu_qi_inter2[g].quantize(gB.normed_out, num_v * v_dim, stream);
        quant_gemv_n2(out_w->data, out_w->type,
                      gA.normed_out, gB.normed_out,
                      gA.proj_out,   gB.proj_out,
                      num_v * v_dim, H, &gpu_qi_inter[g], &gpu_qi_inter2[g], stream);

        // 8. Residual add into FP32 hidden (both tokens, fused n2)
        { dim3 ag((H+255)/256, 2);
          add_kernel_f32_n2<<<ag, 256, 0, stream>>>(
              hidden_a, gA.proj_out,
              hidden_b, gB.proj_out, H); }
    }

    // ============ forward_attn_n3 (MTP K=2) ================================
    // Three-token batched attention. QKV/O projections share weight loads via
    // quant_gemv_n3. The per-token attention compute (KV append, score,
    // softmax, value, gate) runs sequentially for each of the three positions.
    void forward_attn_n3(int layer, float* hidden_a, float* hidden_b, float* hidden_c,
                         int pos_a, int pos_b, int pos_c, cudaStream_t stream, int slot = 0) {
        int g = gpu->layer_gpu[layer];
        int H = cfg.hidden_size;
        int num_q = cfg.num_q_heads;
        int num_kv = cfg.num_kv_heads;
        int hd = cfg.head_dim;
        float eps = cfg.rms_norm_eps;
        float scale = 1.0f / sqrtf((float)hd);
        auto& abA = attn_bufs[g];   auto& abB = attn_bufs2[g];   auto& abC = attn_bufs3[g];
        auto& bA  = bufs[g];        auto& bB  = bufs2[g];        auto& bC  = bufs3[g];

        auto* norm_w   = t(blk(layer, "attn_norm.weight"));
        auto* q_w      = t(blk(layer, "attn_q.weight"));
        auto* k_w      = t(blk(layer, "attn_k.weight"));
        auto* v_w      = t(blk(layer, "attn_v.weight"));
        auto* q_norm_w = t(blk(layer, "attn_q_norm.weight"));
        auto* k_norm_w = t(blk(layer, "attn_k_norm.weight"));
        auto* o_w      = t(blk(layer, "attn_output.weight"));

        int total_qg = num_q * hd;
        int kv_dim   = num_kv * hd;
        int q_out_dim = q_w->dims[1];
        // Continuous batching: slot KV virtual offset.
        size_t slot_kv_off = kv_slot_offset(slot) * kv_dim;

        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w_n3(bA.norm_out, bB.norm_out, bC.norm_out,
                                   hidden_a, hidden_b, hidden_c,
                                   (float*)norm_w->data, H, eps, stream);
        } else {
            rms_norm_f32in_n3(bA.norm_out, bB.norm_out, bC.norm_out,
                              hidden_a, hidden_b, hidden_c,
                              (half*)norm_w->data, H, eps, stream);
        }

        gpu_qi[g].quantize(bA.norm_out, H, stream);
        gpu_qi2[g].quantize(bB.norm_out, H, stream);
        gpu_qi3[g].quantize(bC.norm_out, H, stream);

        quant_gemv_n3(q_w->data, q_w->type,
                      bA.norm_out, bB.norm_out, bC.norm_out,
                      abA.q_proj, abB.q_proj, abC.q_proj,
                      H, q_out_dim, &gpu_qi[g], &gpu_qi2[g], &gpu_qi3[g], stream);
        quant_gemv_n3(k_w->data, k_w->type,
                      bA.norm_out, bB.norm_out, bC.norm_out,
                      abA.k_proj, abB.k_proj, abC.k_proj,
                      H, kv_dim, &gpu_qi[g], &gpu_qi2[g], &gpu_qi3[g], stream);
        quant_gemv_n3(v_w->data, v_w->type,
                      bA.norm_out, bB.norm_out, bC.norm_out,
                      abA.v_proj, abB.v_proj, abC.v_proj,
                      H, kv_dim, &gpu_qi[g], &gpu_qi2[g], &gpu_qi3[g], stream);

        int rope_dim = rope.rope_dim;
        int half_rope = rope_dim / 2;
        auto run_one = [&](AttnBuffers& ab, int pos) {
            half* q_buf    = ab.attn_out;
            half* gate_buf = ab.gate_buf;
            deinterleave_qg_kernel<<<(total_qg+255)/256, 256, 0, stream>>>(
                ab.q_proj, q_buf, gate_buf, num_q, hd);
            int tn = min(hd, 128);
            head_rms_norm_kernel<<<num_q, tn, tn * sizeof(float), stream>>>(
                q_buf, (float*)q_norm_w->data, num_q, hd, eps);
            head_rms_norm_kernel<<<num_kv, tn, tn * sizeof(float), stream>>>(
                ab.k_proj, (float*)k_norm_w->data, num_kv, hd, eps);
            float* sin_pos = rope.sin_table(g) + (size_t)pos * half_rope;
            float* cos_pos = rope.cos_table(g) + (size_t)pos * half_rope;
            apply_rope_kernel<<<(num_q  * half_rope + 255)/256, 256, 0, stream>>>(
                q_buf, sin_pos, cos_pos, num_q, hd, rope_dim);
            apply_rope_kernel<<<(num_kv * half_rope + 255)/256, 256, 0, stream>>>(
                ab.k_proj, sin_pos, cos_pos, num_kv, hd, rope_dim);
            if (use_q8_kv) {
                auto& q8 = q8_kv_caches[layer];
                int bpt = q8.blocks_per_token;
                size_t slot_blk_off = kv_slot_offset(slot) * bpt;
                size_t off = slot_blk_off + (size_t)pos * bpt;
                quantize_kv_q8_0_kern<<<bpt, 32, 0, stream>>>(ab.k_proj, q8.k + off, bpt);
                quantize_kv_q8_0_kern<<<bpt, 32, 0, stream>>>(ab.v_proj, q8.v + off, bpt);
            } else if (use_turboquant) {
                auto& tq = tq_kv_caches[layer];
                int bpt = tq.blocks_per_token;
                size_t slot_blk_off = kv_slot_offset(slot) * bpt;
                size_t off = slot_blk_off + (size_t)pos * bpt;
                tq3_quantize_kernel<<<(bpt+31)/32, 32, 0, stream>>>(ab.k_proj, &tq.k[off], bpt);
                tq3_quantize_kernel<<<(bpt+31)/32, 32, 0, stream>>>(ab.v_proj, &tq.v[off], bpt);
                if (fp16_dec_cache_on) {
                    auto& fc = fp16_dec_caches[layer];
                    half* fp16_k_pos = fc.k + slot_kv_off + (size_t)pos * kv_dim;
                    half* fp16_v_pos = fc.v + slot_kv_off + (size_t)pos * kv_dim;
                    cudaMemcpyAsync(fp16_k_pos, ab.k_proj, kv_dim * sizeof(half),
                                    cudaMemcpyDeviceToDevice, stream);
                    cudaMemcpyAsync(fp16_v_pos, ab.v_proj, kv_dim * sizeof(half),
                                    cudaMemcpyDeviceToDevice, stream);
                }
            } else {
                auto& kv = kv_caches[layer];
                half* k_cache_pos = kv.k + slot_kv_off + (size_t)pos * kv_dim;
                half* v_cache_pos = kv.v + slot_kv_off + (size_t)pos * kv_dim;
                cudaMemcpyAsync(k_cache_pos, ab.k_proj, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(v_cache_pos, ab.v_proj, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            }
            int seq_len = pos + 1;
            dim3 score_grid = score_pos_grid(num_q, seq_len);
            static const bool tq_fused = getenv("TQ_FUSED") != nullptr;
            half* k_src_fp16 = nullptr;
            half* v_src_fp16 = nullptr;
            if (use_q8_kv) {
                auto& q8 = q8_kv_caches[layer];
                int bpt = q8.blocks_per_token;
                int n_blocks_total = seq_len * bpt;
                size_t slot_off_blocks = kv_slot_offset(slot) * bpt;
                dim3 dq_grid((n_blocks_total + 7) / 8);
                dim3 dq_block(32, 8);
                dequantize_kv_q8_0_kern<<<dq_grid, dq_block, 0, stream>>>(
                    q8.k + slot_off_blocks, tq_k_buf[g] + slot_kv_off, n_blocks_total);
                dequantize_kv_q8_0_kern<<<dq_grid, dq_block, 0, stream>>>(
                    q8.v + slot_off_blocks, tq_v_buf[g] + slot_kv_off, n_blocks_total);
                k_src_fp16 = tq_k_buf[g] + slot_kv_off;
                v_src_fp16 = tq_v_buf[g] + slot_kv_off;
            } else if (!use_turboquant) {
                auto& kv = kv_caches[layer];
                k_src_fp16 = kv.k + slot_kv_off;
                v_src_fp16 = kv.v + slot_kv_off;
            } else if (fp16_dec_cache_on) {
                auto& fc = fp16_dec_caches[layer];
                k_src_fp16 = fc.k + slot_kv_off;
                v_src_fp16 = fc.v + slot_kv_off;
            }
            if (use_turboquant && !fp16_dec_cache_on && !use_q8_kv) {
                auto& tq = tq_kv_caches[layer];
                size_t slot_off_blocks_int = kv_slot_offset(slot) * (size_t)tq.blocks_per_token;
                block_tq3* tq_k_slot = tq.k + slot_off_blocks_int;
                if (tq_fused) {
                    dim3 tq_score_grid = score_pos_grid(num_kv, seq_len);
                    int smem_bytes = hd * sizeof(float);
                    attn_score_kernel_tq3<<<tq_score_grid, hd, smem_bytes, stream>>>(
                        q_buf, tq_k_slot, ab.attn_scores,
                        num_q, num_kv, hd, seq_len, scale);
                } else {
                    int n_blocks_total = seq_len * tq.blocks_per_token;
                    tq3_dequantize_kernel<<<(n_blocks_total + 31)/32, 32, 0, stream>>>(
                        tq_k_slot, tq_k_buf[g], n_blocks_total);
                    attn_score_kernel_h<<<score_grid, min(hd, 256), 0, stream>>>(
                        q_buf, tq_k_buf[g], ab.attn_scores, num_q, num_kv, hd, seq_len, scale);
                }
            } else {
                attn_score_kernel_h<<<score_grid, min(hd, 256), 0, stream>>>(
                    q_buf, k_src_fp16, ab.attn_scores, num_q, num_kv, hd, seq_len, scale);
            }
            { int st = 1; while(st < seq_len && st < 256) st <<= 1;
              softmax_kernel<<<num_q, st, st * sizeof(float), stream>>>(
                  ab.attn_scores, num_q, seq_len); }
            if (use_turboquant && !fp16_dec_cache_on && !use_q8_kv) {
                auto& tq = tq_kv_caches[layer];
                size_t slot_off_blocks_int = kv_slot_offset(slot) * (size_t)tq.blocks_per_token;
                block_tq3* tq_v_slot = tq.v + slot_off_blocks_int;
                if (tq_fused) {
                    int blocks_per_kv_head = hd / TQ_BLOCK_SIZE;
                    dim3 tq_value_grid(num_kv, blocks_per_kv_head);
                    int smem_bytes = TQ_BLOCK_SIZE * sizeof(float);
                    attn_value_kernel_tq3<<<tq_value_grid, TQ_BLOCK_SIZE, smem_bytes, stream>>>(
                        ab.attn_scores, tq_v_slot, q_buf,
                        num_q, num_kv, hd, seq_len);
                } else {
                    int n_blocks_total = seq_len * tq.blocks_per_token;
                    tq3_dequantize_kernel<<<(n_blocks_total + 31)/32, 32, 0, stream>>>(
                        tq_v_slot, tq_v_buf[g], n_blocks_total);
                    attn_value_kernel_h<<<num_q, min(hd, 256), 0, stream>>>(
                        ab.attn_scores, tq_v_buf[g], q_buf, num_q, num_kv, hd, seq_len);
                }
            } else {
                attn_value_kernel_h<<<num_q, min(hd, 256), 0, stream>>>(
                    ab.attn_scores, v_src_fp16, q_buf, num_q, num_kv, hd, seq_len);
            }
            apply_gate_sigmoid<<<(total_qg+255)/256, 256, 0, stream>>>(
                q_buf, gate_buf, total_qg);
        };
        run_one(abA, pos_a);
        run_one(abB, pos_b);
        run_one(abC, pos_c);

        gpu_qi_inter[g].quantize(abA.attn_out,  num_q * hd, stream);
        gpu_qi_inter2[g].quantize(abB.attn_out, num_q * hd, stream);
        gpu_qi_inter3[g].quantize(abC.attn_out, num_q * hd, stream);
        quant_gemv_n3(o_w->data, o_w->type,
                      abA.attn_out, abB.attn_out, abC.attn_out,
                      bA.mlp_down,  bB.mlp_down,  bC.mlp_down,
                      num_q * hd, H, &gpu_qi_inter[g], &gpu_qi_inter2[g], &gpu_qi_inter3[g], stream);

        { dim3 ag((H+255)/256, 3);
          add_kernel_f32_n3<<<ag, 256, 0, stream>>>(
              hidden_a, bA.mlp_down,
              hidden_b, bB.mlp_down,
              hidden_c, bC.mlp_down, H); }
    }

    // ============ forward_gdn_n3 (MTP K=2) =================================
    // Three-token batched GDN. 5 projections share weight via quant_gemv_n3.
    // The recurrent path (conv1d + gdn_recurrent_step + rms_norm_gated) is
    // per-token and advances state three times. Snapshot slots are taken
    // AFTER token A (slot 1 — rollback for "draft1 rejected") and AFTER
    // token B (slot B — rollback for "draft1 ok, draft2 rejected").
    void forward_gdn_n3(int layer, float* hidden_a, float* hidden_b, float* hidden_c,
                        cudaStream_t stream, int slot = 0) {
        int g = gpu->layer_gpu[layer];
        auto& bA = bufs[g];      auto& bB = bufs2[g];     auto& bC = bufs3[g];
        auto& gA = gdn_bufs[g];  auto& gB = gdn_bufs2[g]; auto& gC = gdn_bufs3[g];
        int H = cfg.hidden_size;
        int num_k = cfg.linear_k_heads;
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        int qkv_dim = 2 * num_k * k_dim + num_v * v_dim;
        int v_total = num_v * v_dim;
        float* conv_state_slot = gdn_conv_slot(layer, slot);
        float* rec_state_slot  = gdn_rec_slot(layer, slot);

        auto* norm_w  = t(blk(layer, "attn_norm.weight"));
        auto* qkv_w   = t(blk(layer, "attn_qkv.weight"));
        auto* gate_w  = t(blk(layer, "attn_gate.weight"));
        auto* alpha_w = t(blk(layer, "ssm_alpha.weight"));
        auto* beta_w  = t(blk(layer, "ssm_beta.weight"));
        auto* conv_w  = t(blk(layer, "ssm_conv1d.weight"));
        auto* a_log_t = t(blk(layer, "ssm_a"));
        auto* dt_bias_t = t(blk(layer, "ssm_dt.bias"));
        auto* ssm_norm_w = t(blk(layer, "ssm_norm.weight"));
        auto* out_w   = t(blk(layer, "ssm_out.weight"));

        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w_n3(bA.norm_out, bB.norm_out, bC.norm_out,
                                   hidden_a, hidden_b, hidden_c,
                                   (float*)norm_w->data, H, cfg.rms_norm_eps, stream);
        } else {
            rms_norm_f32in_n3(bA.norm_out, bB.norm_out, bC.norm_out,
                              hidden_a, hidden_b, hidden_c,
                              (half*)norm_w->data, H, cfg.rms_norm_eps, stream);
        }

        gpu_qi[g].quantize(bA.norm_out, H, stream);
        gpu_qi2[g].quantize(bB.norm_out, H, stream);
        gpu_qi3[g].quantize(bC.norm_out, H, stream);

        quant_gemv_n3(qkv_w->data, qkv_w->type,
                      bA.norm_out, bB.norm_out, bC.norm_out,
                      bA.attn_out, bB.attn_out, bC.attn_out,
                      H, qkv_dim, &gpu_qi[g], &gpu_qi2[g], &gpu_qi3[g], stream);
        quant_gemv_n3(gate_w->data, gate_w->type,
                      bA.norm_out, bB.norm_out, bC.norm_out,
                      bA.mlp_gate, bB.mlp_gate, bC.mlp_gate,
                      H, num_v * v_dim, &gpu_qi[g], &gpu_qi2[g], &gpu_qi3[g], stream);
        quant_gemv_n3(alpha_w->data, alpha_w->type,
                      bA.norm_out, bB.norm_out, bC.norm_out,
                      bA.mlp_up, bB.mlp_up, bC.mlp_up,
                      H, num_v, &gpu_qi[g], &gpu_qi2[g], &gpu_qi3[g], stream);
        quant_gemv_n3(beta_w->data, beta_w->type,
                      bA.norm_out, bB.norm_out, bC.norm_out,
                      bA.mlp_down, bB.mlp_down, bC.mlp_down,
                      H, num_v, &gpu_qi[g], &gpu_qi2[g], &gpu_qi3[g], stream);

        int kw = 4;
        int threads_conv = min(qkv_dim, 256);
        int blocks_conv  = (qkv_dim + threads_conv - 1) / threads_conv;
        int gdn_smem = (2 * cfg.linear_k_dim + 1) * sizeof(float);

        // Token A
        conv1d_update_silu<<<blocks_conv, threads_conv, 0, stream>>>(
            conv_state_slot, bA.attn_out, (float*)conv_w->data,
            gA.conv_out, qkv_dim, kw);
        launch_gdn_recurrent_step(num_v, min(v_dim, 128), gdn_smem, stream,
            gA.conv_out, (float*)a_log_t->data, (float*)dt_bias_t->data,
            bA.mlp_up, bA.mlp_down,
            rec_state_slot, gA.core_out,
            num_k, num_v, k_dim, v_dim);
        rms_norm_gated_kernel<<<num_v, min(v_dim, 128), 128 * sizeof(float), stream>>>(
            gA.core_out, bA.mlp_gate, (float*)ssm_norm_w->data, gA.normed_out,
            num_v, v_dim, 1e-6f);

        // Snapshot slot A — state post-token-A, for reject-both rollback.
        // Captured into the per-slot backing store at `slot` (==0 in the current
        // single-stream MTP path → bit-identical to the prior single buffer).
        if (snapshots_ready && gdn_snapshots[layer].conv_state) {
            size_t conv_sz = qkv_dim * 4 * sizeof(float);
            size_t rec_sz  = (size_t)num_v * k_dim * v_dim * sizeof(float);
            cudaMemcpyAsync(gdn_snap_conv_slot(layer, slot), conv_state_slot,
                            conv_sz, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(gdn_snap_rec_slot(layer, slot), rec_state_slot,
                            rec_sz, cudaMemcpyDeviceToDevice, stream);
        }

        // Token B
        conv1d_update_silu<<<blocks_conv, threads_conv, 0, stream>>>(
            conv_state_slot, bB.attn_out, (float*)conv_w->data,
            gB.conv_out, qkv_dim, kw);
        launch_gdn_recurrent_step(num_v, min(v_dim, 128), gdn_smem, stream,
            gB.conv_out, (float*)a_log_t->data, (float*)dt_bias_t->data,
            bB.mlp_up, bB.mlp_down,
            rec_state_slot, gB.core_out,
            num_k, num_v, k_dim, v_dim);
        rms_norm_gated_kernel<<<num_v, min(v_dim, 128), 128 * sizeof(float), stream>>>(
            gB.core_out, bB.mlp_gate, (float*)ssm_norm_w->data, gB.normed_out,
            num_v, v_dim, 1e-6f);

        // Snapshot slot B — state post-token-B, for accept_a-only rollback.
        // Captured into the per-slot B backing store at `slot` (==0 today).
        if (snapshots_b_ready && gdn_snapshots_b[layer].conv_state) {
            size_t conv_sz = qkv_dim * 4 * sizeof(float);
            size_t rec_sz  = (size_t)num_v * k_dim * v_dim * sizeof(float);
            cudaMemcpyAsync(gdn_snap_b_conv_slot(layer, slot), conv_state_slot,
                            conv_sz, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(gdn_snap_b_rec_slot(layer, slot), rec_state_slot,
                            rec_sz, cudaMemcpyDeviceToDevice, stream);
        }

        // Token C
        conv1d_update_silu<<<blocks_conv, threads_conv, 0, stream>>>(
            conv_state_slot, bC.attn_out, (float*)conv_w->data,
            gC.conv_out, qkv_dim, kw);
        launch_gdn_recurrent_step(num_v, min(v_dim, 128), gdn_smem, stream,
            gC.conv_out, (float*)a_log_t->data, (float*)dt_bias_t->data,
            bC.mlp_up, bC.mlp_down,
            rec_state_slot, gC.core_out,
            num_k, num_v, k_dim, v_dim);
        rms_norm_gated_kernel<<<num_v, min(v_dim, 128), 128 * sizeof(float), stream>>>(
            gC.core_out, bC.mlp_gate, (float*)ssm_norm_w->data, gC.normed_out,
            num_v, v_dim, 1e-6f);

        gpu_qi_inter[g].quantize(gA.normed_out,  num_v * v_dim, stream);
        gpu_qi_inter2[g].quantize(gB.normed_out, num_v * v_dim, stream);
        gpu_qi_inter3[g].quantize(gC.normed_out, num_v * v_dim, stream);
        quant_gemv_n3(out_w->data, out_w->type,
                      gA.normed_out, gB.normed_out, gC.normed_out,
                      gA.proj_out,   gB.proj_out,   gC.proj_out,
                      num_v * v_dim, H, &gpu_qi_inter[g], &gpu_qi_inter2[g], &gpu_qi_inter3[g], stream);

        { dim3 ag((H+255)/256, 3);
          add_kernel_f32_n3<<<ag, 256, 0, stream>>>(
              hidden_a, gA.proj_out,
              hidden_b, gB.proj_out,
              hidden_c, gC.proj_out, H); }
    }

    // ============ GDN state snapshot/restore (for spec rollback) ============
    // When the MTP draft for the second token is rejected, GDN state has
    // already advanced past the (wrong) draft input. We need to roll back
    // to "after the first token only". Strategy: snapshot state BEFORE the
    // second token forward, run forward, then either commit (do nothing) or
    // restore. KV cache rollback is automatic — we just don't increment the
    // position counter and the next forward overwrites the rejected slot.

    struct GDNSnapshot {
        float* conv_state;  // backup
        float* rec_state;
    };
    std::vector<GDNSnapshot> gdn_snapshots;    // slot A — post-first-token
    std::vector<GDNSnapshot> gdn_snapshots_b;  // slot B — post-second-token (MTP K=2)
    bool snapshots_ready = false;
    bool snapshots_b_ready = false;

    // Per-slot accessors into the snapshot backing store. Each layer's
    // conv_state/rec_state holds num_slots contiguous copies, laid out
    // identically to gdn_states (slot stride = gdn_qkv_dim_cached*4 for conv,
    // gdn_rec_per_slot for rec). With num_slots==1 (the default single-stream
    // MTP path) slot 0 is the only copy, so the footprint and the accessed
    // bytes are bit-identical to the original single-buffer layout.
    inline float* gdn_snap_conv_slot(int layer, int slot) {
        return gdn_snapshots[layer].conv_state + (size_t)slot * gdn_qkv_dim_cached * 4;
    }
    inline float* gdn_snap_rec_slot(int layer, int slot) {
        return gdn_snapshots[layer].rec_state + (size_t)slot * gdn_rec_per_slot;
    }
    inline float* gdn_snap_b_conv_slot(int layer, int slot) {
        return gdn_snapshots_b[layer].conv_state + (size_t)slot * gdn_qkv_dim_cached * 4;
    }
    inline float* gdn_snap_b_rec_slot(int layer, int slot) {
        return gdn_snapshots_b[layer].rec_state + (size_t)slot * gdn_rec_per_slot;
    }

    // ============ DDTree: per-layer GDN persistent intermediate buffers ========
    // For each GDN layer, we keep a per-tree-node state-history buffer so that
    // the tree verify pass can reload the parent state at branch points without
    // a replay forward. gdn_inter[layer] has shape [budget, num_v, k_dim, v_dim]
    // in FP32. Attention layers do not need this (they use KV cache ancestor
    // masking instead).
    std::vector<float*> tree_gdn_inter;
    int   tree_budget   = 0;
    std::vector<int*> tree_parent_ids_d;   // per-GPU [budget] int32
    bool  tree_ready    = false;
    // Host-side derived tree metadata, populated from parent_ids on upload.
    // depth_host[t]      = distance from root (root=0)
    // ancestor_bits_host = bit k set iff slot k is an ancestor of t (self included)
    std::vector<int>      tree_depth_host;
    std::vector<uint32_t> tree_ancestor_bits_host;
    // Per-GDN-layer saved post-qkv-projection values for the current tree's
    // nodes. commit_tree_gdn_chain reads this to slide conv_state forward
    // along the accepted chain.
    std::vector<half*> tree_qkv_persist;
    bool  tree_qkv_persist_ready = false;
    // Chain-shaped tree → batched attn fast path. tree_is_chain is set in
    // upload_parent_ids (parent[t]==t-1). tree_chain_pos_d / tree_chain_dst_d
    // are per-GPU [budget] scratch holding RoPE positions and KV destinations
    // for the single-slot consecutive-position verify (see forward_attn_tree).
    bool  tree_is_chain = false;
    std::vector<int*> tree_chain_pos_d;
    std::vector<int*> tree_chain_dst_d;
    bool  tree_chain_pos_ready = false;
    std::vector<int> tree_chain_slot_ids_h;
    std::vector<int> tree_chain_pos_h;
    std::vector<int> tree_chain_dst_h;

    void alloc_tree_decode(int budget) {
        if (tree_ready) return;
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        size_t per_layer_sz = (size_t)budget * num_v * k_dim * v_dim * sizeof(float);
        tree_gdn_inter.assign(cfg.num_layers, nullptr);
        size_t total = 0;
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (is_attn_layer(layer)) continue;
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            cudaMalloc(&tree_gdn_inter[layer], per_layer_sz);
            total += per_layer_sz;
        }
        tree_parent_ids_d.assign(gpu->num_gpus, nullptr);
        tree_chain_pos_d.assign(gpu->num_gpus, nullptr);
        tree_chain_dst_d.assign(gpu->num_gpus, nullptr);
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&tree_parent_ids_d[g], (size_t)budget * sizeof(int));
            cudaMalloc(&tree_chain_pos_d[g],  (size_t)budget * sizeof(int));
            cudaMalloc(&tree_chain_dst_d[g],  (size_t)budget * sizeof(int));
        }
        tree_chain_pos_ready = true;
        // Persistent per-GDN-layer qkv_half storage.
        auto* qkv_probe = t("blk.0.attn_qkv.weight");
        if (qkv_probe) {
            int qkv_dim = qkv_probe->dims[1];
            tree_qkv_persist.assign(cfg.num_layers, nullptr);
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                if (is_attn_layer(layer)) continue;
                int g = gpu->layer_gpu[layer];
                cudaSetDevice(g);
                cudaMalloc(&tree_qkv_persist[layer],
                           (size_t)budget * qkv_dim * sizeof(half));
            }
            tree_qkv_persist_ready = true;
        }
        tree_budget = budget;
        tree_ready  = true;
        printf("[TREE] GDN intermediate buffers allocated: %.1f MB total, budget=%d\n",
               total / 1e6, budget);
    }

    // Host → all GPUs broadcast of the current tree's parent_ids array.
    // Caller supplies host_parent_ids[tree_budget].
    // Also derives per-node depth and ancestor-mask bitsets for the
    // sequential attention path. Requires tree_budget <= 32 (one uint32
    // per node's ancestor mask; enforced by alloc_tree_decode clamp).
    void upload_parent_ids(const int* host_parent_ids, cudaStream_t stream = 0) {
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaMemcpyAsync(tree_parent_ids_d[g], host_parent_ids,
                            (size_t)tree_budget * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
        }
        tree_depth_host.assign(tree_budget, 0);
        tree_ancestor_bits_host.assign(tree_budget, 0u);
        bool is_chain = (host_parent_ids[0] < 0);
        for (int t = 0; t < tree_budget; t++) {
            int p = host_parent_ids[t];
            if (p < 0) {
                tree_depth_host[t] = 0;
                tree_ancestor_bits_host[t] = 1u << t;
            } else {
                tree_depth_host[t] = tree_depth_host[p] + 1;
                tree_ancestor_bits_host[t] = tree_ancestor_bits_host[p] | (1u << t);
            }
            if (t > 0 && host_parent_ids[t] != t - 1) is_chain = false;
        }
        // Pure chain (parents = [-1,0,1,...,n-2]) → token t lives at depth t and
        // attends exactly to slots 0..t. That equals positional causal masking
        // over consecutive KV positions, so the batched single-slot attn path
        // is bit-equivalent and far faster than the per-token loop.
        tree_is_chain = is_chain;
    }

    // Scratch per-GPU buffers for chain commit's node_ids, lazily sized to
    // `tree_budget`. For chain commit node_ids is trivially [0,1,...,L-1], so
    // we upload that layout once on first use.
    std::vector<int*> commit_chain_node_ids_d;
    bool commit_chain_node_ids_ready = false;
    void ensure_commit_chain_node_ids() {
        if (commit_chain_node_ids_ready) return;
        commit_chain_node_ids_d.assign(gpu->num_gpus, nullptr);
        std::vector<int> host_ids(tree_budget);
        for (int i = 0; i < tree_budget; i++) host_ids[i] = i;
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&commit_chain_node_ids_d[g], (size_t)tree_budget * sizeof(int));
            cudaMemcpy(commit_chain_node_ids_d[g], host_ids.data(),
                       (size_t)tree_budget * sizeof(int), cudaMemcpyHostToDevice);
        }
        commit_chain_node_ids_ready = true;
    }

    // Commit a chain-shaped accepted prefix of length L into every GDN
    // layer's rec_state and conv_state. Chain assumption: the accepted node
    // indices are exactly [0, 1, ..., L-1] in tree_gdn_inter / tree_qkv_persist.
    // L must satisfy 1 <= L <= kw-1 (kw=4 for Qwen3.5, so L in {1, 2, 3}).
    // DDTree: commit accepted GDN state for an arbitrary slot path (used for
    // branching where the accepted slots are not a prefix of [0..n-1]).
    // host_slots[0..path_len-1] are tree-node indices in branch-root-first
    // order. conv_state_commit_chain_kernel slides the convolution window
    // forward by appending each node's qkv_persist in turn.
    std::vector<int*> commit_path_node_ids_d;
    bool commit_path_node_ids_ready = false;
    void ensure_commit_path_node_ids() {
        if (commit_path_node_ids_ready) return;
        commit_path_node_ids_d.assign(gpu->num_gpus, nullptr);
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&commit_path_node_ids_d[g], (size_t)tree_budget * sizeof(int));
        }
        commit_path_node_ids_ready = true;
    }
    void commit_tree_gdn_path(const int* host_slots, int path_len) {
        if (path_len <= 0) return;
        if (!tree_ready || !tree_qkv_persist_ready) return;
        static const bool fallback_gdn = getenv("TREE_FALLBACK_GDN") != nullptr;
        if (fallback_gdn) return;
        ensure_commit_path_node_ids();

        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        size_t state_sz = (size_t)num_v * k_dim * v_dim * sizeof(float);
        auto* qkv_probe = t("blk.0.attn_qkv.weight");
        int qkv_dim = qkv_probe->dims[1];
        int kw = 4;

        int final_slot = host_slots[path_len - 1];
        // conv_state only holds the last (kw-1) inputs: trim the committed
        // path to its trailing (kw-1) nodes to avoid the negative-index OOB
        // write in conv_state_commit_chain_kernel (see commit_tree_gdn_chain).
        int conv_len = path_len;
        const int* conv_src = host_slots;
        if (conv_len > kw - 1) {
            conv_len = kw - 1;
            conv_src = host_slots + (path_len - conv_len);  // trailing nodes
        }
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaMemcpyAsync(commit_path_node_ids_d[g], conv_src,
                            (size_t)conv_len * sizeof(int),
                            cudaMemcpyHostToDevice, 0);
        }
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (is_attn_layer(layer)) continue;
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            float* src = tree_gdn_inter[layer]
                + (size_t)final_slot * num_v * k_dim * v_dim;
            cudaMemcpyAsync(gdn_states[layer].rec_state, src, state_sz,
                            cudaMemcpyDeviceToDevice, 0);
            int threads = 256;
            int blocks  = (qkv_dim + threads - 1) / threads;
            conv_state_commit_chain_kernel<<<blocks, threads, 0, 0>>>(
                gdn_states[layer].conv_state,
                tree_qkv_persist[layer],
                commit_path_node_ids_d[g],
                qkv_dim, kw, conv_len);
        }
    }

    void commit_tree_gdn_chain(int accept_len) {
        if (accept_len <= 0) return;
        if (!tree_ready || !tree_qkv_persist_ready) return;
        // Debug: fallback mode updates rec_state/conv_state in-place via
        // per-token forward_gdn, so commit here would overwrite with garbage
        // persist_inter values. Skip commit entirely in fallback mode.
        static const bool fallback_gdn = getenv("TREE_FALLBACK_GDN") != nullptr;
        if (fallback_gdn) return;
        ensure_commit_chain_node_ids();

        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        size_t state_sz = (size_t)num_v * k_dim * v_dim * sizeof(float);
        auto* qkv_probe = t("blk.0.attn_qkv.weight");
        int qkv_dim = qkv_probe->dims[1];
        int kw = 4;

        // The convolution state only holds the last (kw-1) inputs. When the
        // accepted chain is longer than (kw-1), only the TRAILING (kw-1) nodes
        // survive the window slide — committing all `accept_len` nodes makes
        // conv_state_commit_chain_kernel index st[(kw-accept_len)+i] with a
        // negative base (out-of-bounds write before the row → corrupts every
        // GDN layer's conv_state and cascades into garbage output). This is the
        // budget>kw-1 regime DFlash (budget=16) hits but MTP_TREE (budget<=8,
        // low accept) never did. Trim to the last (kw-1) accepted nodes.
        int conv_len = accept_len;
        if (conv_len > kw - 1) {
            conv_len = kw - 1;
            // Trailing node indices: [accept_len-(kw-1) .. accept_len-1].
            ensure_commit_path_node_ids();
            std::vector<int> trail(conv_len);
            for (int i = 0; i < conv_len; i++)
                trail[i] = accept_len - conv_len + i;
            for (int g = 0; g < gpu->num_gpus; g++) {
                cudaSetDevice(g);
                cudaMemcpy(commit_path_node_ids_d[g], trail.data(),
                           (size_t)conv_len * sizeof(int), cudaMemcpyHostToDevice);
            }
        }

        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (is_attn_layer(layer)) continue;
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            // rec_state ← tree_gdn_inter[layer][accept_len - 1] (full state at
            // the last accepted node — unaffected by the conv trim).
            float* src = tree_gdn_inter[layer]
                + (size_t)(accept_len - 1) * num_v * k_dim * v_dim;
            cudaMemcpyAsync(gdn_states[layer].rec_state, src, state_sz,
                            cudaMemcpyDeviceToDevice, 0);
            // conv_state slide + append. node_ids = [0..L-1] for L<=kw-1, else
            // the trailing (kw-1) accepted nodes (see trim above).
            const int* node_ids = (accept_len > kw - 1)
                                 ? commit_path_node_ids_d[g]
                                 : commit_chain_node_ids_d[g];
            int threads = 256;
            int blocks  = (qkv_dim + threads - 1) / threads;
            conv_state_commit_chain_kernel<<<blocks, threads, 0, 0>>>(
                gdn_states[layer].conv_state,
                tree_qkv_persist[layer],
                node_ids,
                qkv_dim, kw, conv_len);
        }
    }

    void alloc_gdn_snapshots() {
        if (snapshots_ready) return;
        auto* qkv = t("blk.0.attn_qkv.weight");
        if (!qkv) { printf("[SPEC] no qkv tensor for snapshot alloc\n"); return; }
        int qkv_dim = qkv->dims[1];
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        gdn_snapshots.resize(cfg.num_layers);
        size_t total = 0;
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (is_attn_layer(layer)) {
                gdn_snapshots[layer].conv_state = nullptr;
                gdn_snapshots[layer].rec_state = nullptr;
                continue;
            }
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            // Per-slot backing store: num_slots contiguous copies. num_slots==1
            // reduces to the original single-buffer footprint exactly.
            int ns = num_slots > 0 ? num_slots : 1;
            size_t conv_sz = (size_t)ns * qkv_dim * 4 * sizeof(float);
            size_t rec_sz  = (size_t)ns * num_v * k_dim * v_dim * sizeof(float);
            cudaMalloc(&gdn_snapshots[layer].conv_state, conv_sz);
            cudaMalloc(&gdn_snapshots[layer].rec_state,  rec_sz);
            total += conv_sz + rec_sz;
        }
        snapshots_ready = true;
        printf("[SPEC] GDN snapshot buffers allocated (%.1f MB total, %d slot(s))\n",
               total / 1e6, num_slots > 0 ? num_slots : 1);
    }

    // slot: which per-slot live/snapshot copy to capture. Defaults to 0 so the
    // existing single-stream MTP path (snapshot/restore on slot 0) is unchanged.
    void snapshot_gdn_states(int slot = 0, cudaStream_t stream = 0) {
        auto* qkv = t("blk.0.attn_qkv.weight");
        int qkv_dim = qkv->dims[1];
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        size_t conv_sz = qkv_dim * 4 * sizeof(float);
        size_t rec_sz  = (size_t)num_v * k_dim * v_dim * sizeof(float);
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (!gdn_snapshots[layer].conv_state) continue;
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            cudaMemcpyAsync(gdn_snap_conv_slot(layer, slot), gdn_conv_slot(layer, slot),
                            conv_sz, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(gdn_snap_rec_slot(layer, slot), gdn_rec_slot(layer, slot),
                            rec_sz, cudaMemcpyDeviceToDevice, stream);
        }
    }

    // slot: which per-slot copy to roll back. Defaults to 0; main.cu's
    // single-stream MTP path calls restore_gdn_states(0) which binds slot=0,
    // stream=0 — bit-identical to the original behavior.
    void restore_gdn_states(int slot = 0, cudaStream_t stream = 0) {
        auto* qkv = t("blk.0.attn_qkv.weight");
        int qkv_dim = qkv->dims[1];
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        size_t conv_sz = qkv_dim * 4 * sizeof(float);
        size_t rec_sz  = (size_t)num_v * k_dim * v_dim * sizeof(float);
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (!gdn_snapshots[layer].conv_state) continue;
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            cudaMemcpyAsync(gdn_conv_slot(layer, slot), gdn_snap_conv_slot(layer, slot),
                            conv_sz, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(gdn_rec_slot(layer, slot), gdn_snap_rec_slot(layer, slot),
                            rec_sz, cudaMemcpyDeviceToDevice, stream);
        }
    }

    // ============ MTP K=2 second snapshot slot =============================
    // slot B captures the state AFTER the second token (draft1) has been
    // processed inside forward_gdn_n3. Used when MTP K=2 accepts draft1 but
    // rejects draft2 — we need to roll back to post-draft1, not post-main.
    void alloc_gdn_snapshots_b() {
        alloc_gdn_snapshots();          // ensure slot A is ready too
        if (snapshots_b_ready) return;
        auto* qkv = t("blk.0.attn_qkv.weight");
        if (!qkv) return;
        int qkv_dim = qkv->dims[1];
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        gdn_snapshots_b.resize(cfg.num_layers);
        size_t total = 0;
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (is_attn_layer(layer)) {
                gdn_snapshots_b[layer].conv_state = nullptr;
                gdn_snapshots_b[layer].rec_state  = nullptr;
                continue;
            }
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            // Per-slot backing store: num_slots contiguous copies (see slot A).
            int ns = num_slots > 0 ? num_slots : 1;
            size_t conv_sz = (size_t)ns * qkv_dim * 4 * sizeof(float);
            size_t rec_sz  = (size_t)ns * num_v * k_dim * v_dim * sizeof(float);
            cudaMalloc(&gdn_snapshots_b[layer].conv_state, conv_sz);
            cudaMalloc(&gdn_snapshots_b[layer].rec_state,  rec_sz);
            total += conv_sz + rec_sz;
        }
        snapshots_b_ready = true;
        printf("[SPEC] GDN snapshot B buffers allocated (%.1f MB total, %d slot(s)) — MTP K=2\n",
               total / 1e6, num_slots > 0 ? num_slots : 1);
    }

    // Restore from slot B (post-draft1). slot defaults to 0 so the existing
    // restore_gdn_states_b(0) call binds slot=0, stream=0 (unchanged behavior).
    void restore_gdn_states_b(int slot = 0, cudaStream_t stream = 0) {
        if (!snapshots_b_ready) return;
        auto* qkv = t("blk.0.attn_qkv.weight");
        int qkv_dim = qkv->dims[1];
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        size_t conv_sz = qkv_dim * 4 * sizeof(float);
        size_t rec_sz  = (size_t)num_v * k_dim * v_dim * sizeof(float);
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (!gdn_snapshots_b[layer].conv_state) continue;
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            cudaMemcpyAsync(gdn_conv_slot(layer, slot), gdn_snap_b_conv_slot(layer, slot),
                            conv_sz, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(gdn_rec_slot(layer, slot), gdn_snap_b_rec_slot(layer, slot),
                            rec_sz, cudaMemcpyDeviceToDevice, stream);
        }
    }
};
