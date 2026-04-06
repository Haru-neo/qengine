#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <string>
#include "gguf.h"
#include "gpu_loader.h"
#include "quant_gemv.cuh"
#include "ops.cuh"
#include "attention.cuh"
#include "turboquant.cuh"

// ============ Gemma 4 Config ============

struct GemmaConfig {
    int hidden_size;        // 5376 (31B) / 2816 (26B)
    int intermediate_size;  // 21504 (31B) / 2112 (26B shared expert)
    int num_layers;         // 60 (31B) / 30 (26B)
    int vocab_size;         // 262144
    float rms_norm_eps;     // 1e-6
    float softcap;          // 30.0
    float embd_scale;       // sqrt(hidden_size)

    // Sliding attention
    int slide_num_q;        // 32 (31B) / 16 (26B)
    int slide_num_kv;       // 16 (31B) / 8 (26B)
    int slide_hd;           // 256
    int slide_rope_dim;     // 256 (full rotation)
    float slide_rope_theta; // 10000
    int sliding_window;     // 1024

    // Full attention
    int full_num_q;         // detect from weight shape
    int full_num_kv;        // 4
    int full_hd;            // 512
    int full_rope_dim;      // 128 (partial 25%)
    float full_rope_theta;  // 1000000

    // MoE
    bool is_moe = false;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;  // per-expert intermediate dim
};

// ============ KV Caches ============

struct SlidingKVCache {
    half* k;   // [window_size, kv_dim]
    half* v;   // [window_size, kv_dim]
    int window_size;
    int kv_dim;  // num_kv * head_dim
    int count;   // tokens written so far

    void alloc(int ws, int dim, int gpu) {
        window_size = ws; kv_dim = dim; count = 0;
        cudaSetDevice(gpu);
        size_t bytes = (size_t)ws * dim * sizeof(half);
        cudaMalloc(&k, bytes); cudaMemset(k, 0, bytes);
        cudaMalloc(&v, bytes); cudaMemset(v, 0, bytes);
    }
    void store(const half* k_in, const half* v_in, cudaStream_t s = 0) {
        int pos = count % window_size;
        size_t bytes = kv_dim * sizeof(half);
        cudaMemcpyAsync(k + (size_t)pos * kv_dim, k_in, bytes, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(v + (size_t)pos * kv_dim, v_in, bytes, cudaMemcpyDeviceToDevice, s);
        count++;
    }
    int valid_len() const { return count < window_size ? count : window_size; }
    int oldest_idx() const { return count <= window_size ? 0 : count % window_size; }
    void reset() { count = 0; }
};

struct FullKVCache {
    half* k;    // [max_seq, kv_dim]
    half* v;    // [max_seq, kv_dim] — separate from K (different norms applied)
    int max_seq;
    int kv_dim;
    int count;

    void alloc(int ms, int dim, int gpu) {
        max_seq = ms; kv_dim = dim; count = 0;
        cudaSetDevice(gpu);
        size_t bytes = (size_t)ms * dim * sizeof(half);
        cudaMalloc(&k, bytes); cudaMemset(k, 0, bytes);
        cudaMalloc(&v, bytes); cudaMemset(v, 0, bytes);
    }
    void store(const half* k_in, const half* v_in, cudaStream_t s = 0) {
        size_t bytes = kv_dim * sizeof(half);
        cudaMemcpyAsync(k + (size_t)count * kv_dim, k_in, bytes, cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(v + (size_t)count * kv_dim, v_in, bytes, cudaMemcpyDeviceToDevice, s);
        count++;
    }
    int valid_len() const { return count; }
    void reset() { count = 0; }
};

// ============ Ring buffer attention kernels ============

__global__ void attn_score_ring_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    float* __restrict__ scores,
    int num_q, int num_kv, int head_dim,
    int valid_len, int oldest_idx, int window_size, float scale
) {
    int qh = blockIdx.x;
    int vi = blockIdx.y;  // index in valid range 0..valid_len-1
    if (qh >= num_q || vi >= valid_len) return;

    int kv_head = qh / (num_q / num_kv);
    int cache_pos = (oldest_idx + vi) % window_size;

    const half* q_ptr = q + qh * head_dim;
    const half* k_ptr = k_cache + (size_t)cache_pos * num_kv * head_dim + kv_head * head_dim;

    float dot = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);

    for (int off = 16; off > 0; off >>= 1)
        dot += __shfl_xor_sync(0xffffffff, dot, off);

    if (threadIdx.x == 0)
        scores[qh * valid_len + vi] = dot * scale;
}

__global__ void attn_value_ring_kernel(
    const float* __restrict__ scores,
    const half* __restrict__ v_cache,
    half* __restrict__ output,
    int num_q, int num_kv, int head_dim,
    int valid_len, int oldest_idx, int window_size
) {
    int qh = blockIdx.x;
    if (qh >= num_q) return;
    int kv_head = qh / (num_q / num_kv);

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int vi = 0; vi < valid_len; vi++) {
            int cache_pos = (oldest_idx + vi) % window_size;
            sum += scores[qh * valid_len + vi] *
                   __half2float(v_cache[(size_t)cache_pos * num_kv * head_dim + kv_head * head_dim + d]);
        }
        output[qh * head_dim + d] = __float2half(sum);
    }
}

// ============ Scale + add kernel: x += scale * y ============

__global__ void scale_add_kernel(half* __restrict__ x, const half* __restrict__ y, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = __float2half(__half2float(x[idx]) + scale * __half2float(y[idx]));
}

// ============ MoE kernels ============

// Zero buffer
__global__ void zero_kernel(half* __restrict__ x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = __float2half(0.0f);
}

// Clamp fp16 to prevent inf/nan (fp16 max = 65504)
__global__ void clamp_fp16_kernel(half* __restrict__ x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __half2float(x[idx]);
        if (v != v || v > 65504.f) x[idx] = __float2half(65504.f);    // nan or +inf
        else if (v < -65504.f) x[idx] = __float2half(-65504.f);        // -inf
    }
}

// Weighted accumulate: out += weight * x
__global__ void weighted_add_kernel(half* __restrict__ out, const half* __restrict__ x, float weight, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2half(__half2float(out[idx]) + weight * __half2float(x[idx]));
}

// Zero fp32 buffer
__global__ void zero_f32_kernel(float* __restrict__ x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = 0.0f;
}

// Weighted accumulate into fp32: out_f32 += weight * (half)x
__global__ void weighted_add_f32_kernel(float* __restrict__ out, const half* __restrict__ x, float weight, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] += weight * __half2float(x[idx]);
}

// Weighted accumulate: float out += weight * float in
__global__ void weighted_add_f32f32_kernel(float* __restrict__ out, const float* __restrict__ x, float weight, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] += weight * x[idx];
}

// Convert fp32 buffer to fp16
__global__ void f32_to_fp16_kernel(const float* __restrict__ in, half* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2half(in[idx]);
}

// GPU top-k selection + SIGMOID weights for Gemma 4 MoE router
// Gemma 4 uses: top-k on raw logits, sigmoid for weights (NOT softmax)
// Input: float logits[num_experts] on device
// Output: int indices[topk], float weights[topk] (sigmoid values) on device
__global__ void topk_sigmoid_kernel(const float* __restrict__ logits, int* __restrict__ indices,
                                     float* __restrict__ weights, int num_experts, int topk) {
    extern __shared__ float sdata[];
    float* vals = sdata;
    int* idxs = (int*)(sdata + num_experts);

    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        vals[i] = logits[i];
        idxs[i] = i;
    }
    __syncthreads();

    // Find top-k by selection on raw logits
    for (int k = 0; k < topk; k++) {
        float best_val = -1e30f;
        int best_idx = k;
        for (int i = k + threadIdx.x; i < num_experts; i += blockDim.x) {
            if (vals[i] > best_val) { best_val = vals[i]; best_idx = i; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, best_val, off);
            int other_idx = __shfl_xor_sync(0xffffffff, best_idx, off);
            if (other_val > best_val) { best_val = other_val; best_idx = other_idx; }
        }
        if (threadIdx.x == 0) {
            float tmp_v = vals[k]; int tmp_i = idxs[k];
            vals[k] = vals[best_idx]; idxs[k] = idxs[best_idx];
            vals[best_idx] = tmp_v; idxs[best_idx] = tmp_i;
        }
        __syncthreads();
    }

    // Sigmoid weights (NOT softmax) — unnormalized
    if (threadIdx.x == 0) {
        for (int i = 0; i < topk; i++) {
            indices[i] = idxs[i];
            weights[i] = 1.0f / (1.0f + expf(-vals[i]));  // sigmoid
        }
    }
}

// Scale half vector by float: out = in * scale
__global__ void scale_half_kernel(const half* __restrict__ in, half* __restrict__ out, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__half2float(in[i]) * scale);
}

// Sigmoid router (for Gemma 4 MoE: sigmoid gating, not softmax)
// Router GEMV: F32 weights, fp16 input, float output
__global__ void router_gemv_f32(const float* __restrict__ weight, const half* __restrict__ input,
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

// ============ RMSNorm without learned weight (eps-only) ============
// Used for V normalization in Gemma 4: normalize but don't scale by weight

__global__ void rms_norm_noweight_kernel(half* __restrict__ x, int size, float eps) {
    extern __shared__ float sdata[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float v = __half2float(x[i]);
        local_sum += v * v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / size + eps);
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        x[i] = __float2half(__half2float(x[i]) * rms);
    }
}

// Per-head RMSNorm without learned weight (eps-only), multiple heads
__global__ void head_rms_norm_noweight_kernel(half* __restrict__ x, int num_heads, int head_dim, float eps) {
    int head = blockIdx.x;
    if (head >= num_heads) return;
    half* xh = x + head * head_dim;
    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = __half2float(xh[i]);
        sum += v * v;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / head_dim + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        xh[i] = __float2half(__half2float(xh[i]) * rms);
}

// ============ Per-GPU buffers ============

struct GemmaBuffers {
    half* norm_out;
    half* post_norm;
    half* q_proj;
    half* k_proj;
    half* v_proj;
    half* attn_out;
    half* mlp_gate;
    half* mlp_up;
    half* mlp_down;
    half* residual;
    float* attn_scores;
    QuantInput qi;
    QuantInput qi_inter;

    // MoE buffers (all expert computation in fp32 to prevent overflow)
    float* moe_output_f32;  // [hidden_size] fp32 accumulated output
    float* expert_gate_f32; // [2*moe_intermediate] fused gate+up in fp32
    float* expert_down_f32; // [hidden_size] per-expert down in fp32
    half* expert_gate;      // [2*moe_intermediate] spare fp16 buffer
    half* expert_up;        // spare
    half* expert_down;      // [hidden_size] fp16 conversion buffer
    float* router_logits;   // [num_experts] on device
    int* topk_indices;      // [num_experts_per_tok] GPU top-k results
    float* topk_weights;    // [num_experts_per_tok] GPU top-k softmax weights
    QuantInput qi_expert;   // for expert GEMV input quantization

    // TurboQuant dequant buffers
    half* tq_k_buf = nullptr;  // [max_seq * kv_dim] dequantized K
    half* tq_v_buf = nullptr;  // [max_seq * kv_dim] dequantized V
};

// ============ GemmaModel ============

// Per-layer TurboQuant KV cache (allocated on the layer's GPU)
struct TQLayerCache {
    block_tq3* k_cache = nullptr;  // [max_seq, blocks_per_token]
    block_tq3* v_cache = nullptr;
    int max_seq = 0;
    int kv_dim = 0;
    int blocks_per_token = 0;
    int count = 0;  // tokens stored

    void alloc(int ms, int dim, int gpu) {
        max_seq = ms; kv_dim = dim; count = 0;
        blocks_per_token = dim / TQ_BLOCK_SIZE;
        cudaSetDevice(gpu);
        size_t sz = (size_t)ms * blocks_per_token * sizeof(block_tq3);
        cudaMalloc(&k_cache, sz);
        cudaMalloc(&v_cache, sz);
    }
    void store(const half* k, const half* v, cudaStream_t s = 0) {
        size_t off = (size_t)count * blocks_per_token;
        tq3_quantize_kernel<<<(blocks_per_token+31)/32, 32, 0, s>>>(k, &k_cache[off], blocks_per_token);
        tq3_quantize_kernel<<<(blocks_per_token+31)/32, 32, 0, s>>>(v, &v_cache[off], blocks_per_token);
        count++;
    }
    void load(int seq_len, half* k_out, half* v_out, cudaStream_t s = 0) {
        int total = seq_len * blocks_per_token;
        tq3_dequantize_kernel<<<(total+31)/32, 32, 0, s>>>(k_cache, k_out, total);
        tq3_dequantize_kernel<<<(total+31)/32, 32, 0, s>>>(v_cache, v_out, total);
    }
    // Ring buffer store for sliding window
    void store_ring(int window_size, const half* k, const half* v, cudaStream_t s = 0) {
        int pos = count % window_size;
        size_t off = (size_t)pos * blocks_per_token;
        tq3_quantize_kernel<<<(blocks_per_token+31)/32, 32, 0, s>>>(k, &k_cache[off], blocks_per_token);
        tq3_quantize_kernel<<<(blocks_per_token+31)/32, 32, 0, s>>>(v, &v_cache[off], blocks_per_token);
        count++;
    }
    int valid_len(int window = 0) const {
        if (window > 0) return count < window ? count : window;
        return count;
    }
    int oldest_idx(int window) const { return count <= window ? 0 : count % window; }
    void reset() { count = 0; }
};

struct GemmaModel {
    GPUModel* gpu;
    GemmaConfig cfg;
    GemmaBuffers bufs[4];
    RoPETable rope_slide, rope_full;
    std::unordered_map<int, SlidingKVCache> slide_caches;
    std::unordered_map<int, FullKVCache> full_caches;
    bool use_turboquant = false;
    std::unordered_map<int, TQLayerCache> tq_caches;

    std::vector<uint8_t> swa_pattern;  // per-layer: 1=sliding, 0=full
    std::vector<float> layer_scales;   // per-layer output scale

    bool is_full_attn(int layer) const {
        if (layer < (int)swa_pattern.size()) return swa_pattern[layer] == 0;
        return ((layer + 1) % 6 == 0);  // fallback
    }

    void init_config(GGUFFile& gguf) {
        std::string arch = gguf.get_str("general.architecture", "gemma4");
        cfg.hidden_size = gguf.get_u32(arch + ".embedding_length", 5376);
        cfg.num_layers = gguf.get_u32(arch + ".block_count", 60);
        cfg.rms_norm_eps = gguf.get_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        cfg.softcap = gguf.get_f32(arch + ".final_logit_softcapping", 30.0f);
        cfg.embd_scale = sqrtf((float)cfg.hidden_size);
        cfg.sliding_window = gguf.get_u32(arch + ".attention.sliding_window", 1024);

        // Per-layer sliding window pattern from GGUF
        auto pat_it = gguf.meta_bool_arr.find(arch + ".attention.sliding_window_pattern");
        if (pat_it != gguf.meta_bool_arr.end()) {
            swa_pattern = pat_it->second;
        }

        // Sliding attention params (SWA = sliding window attention)
        cfg.slide_num_q = gguf.get_u32(arch + ".attention.head_count", 32);
        cfg.slide_hd = gguf.get_u32(arch + ".attention.key_length_swa", 256);
        cfg.slide_rope_dim = gguf.get_u32(arch + ".rope.dimension_count_swa", cfg.slide_hd);
        cfg.slide_rope_theta = gguf.get_f32(arch + ".rope.freq_base_swa", 10000.0f);

        // Full/global attention params
        cfg.full_hd = gguf.get_u32(arch + ".attention.key_length", 512);
        cfg.full_rope_dim = cfg.full_hd / 4;  // partial_rotary_factor = 0.25
        cfg.full_rope_theta = gguf.get_f32(arch + ".rope.freq_base", 1000000.0f);

        // Per-layer KV head count from GGUF array
        auto kv_it = gguf.meta_i32_arr.find(arch + ".attention.head_count_kv");
        if (kv_it != gguf.meta_i32_arr.end() && !kv_it->second.empty()) {
            // Find sliding KV heads (from first sliding layer)
            // Find full KV heads (from first full layer)
            cfg.slide_num_kv = 16; cfg.full_num_kv = 4;  // defaults
            for (int i = 0; i < (int)kv_it->second.size(); i++) {
                if (is_full_attn(i)) { cfg.full_num_kv = kv_it->second[i]; break; }
            }
            for (int i = 0; i < (int)kv_it->second.size(); i++) {
                if (!is_full_attn(i)) { cfg.slide_num_kv = kv_it->second[i]; break; }
            }
        } else {
            cfg.slide_num_kv = gguf.get_u32(arch + ".attention.head_count_kv", 16);
            cfg.full_num_kv = 4;
        }

        // Full Q heads: detect from weight shape of first full layer
        cfg.full_num_q = cfg.slide_num_q;  // default: same
        for (int l = 0; l < cfg.num_layers; l++) {
            if (is_full_attn(l)) {
                char name[64]; snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
                auto* qw = gpu->get(name);
                if (qw && qw->n_dims == 2) cfg.full_num_q = qw->dims[1] / cfg.full_hd;
                break;
            }
        }

        // RoPE dimension for full: use gguf if available, else compute from partial_rotary_factor
        uint32_t full_rope = gguf.get_u32(arch + ".rope.dimension_count", cfg.full_hd);
        // proportional RoPE: dimension_count stores full head_dim, but only 25% is rotated
        // Actually from the GGUF we see rope.dimension_count = 512 for full layers
        // The freq denominator should be rope_dim itself for both layer types
        cfg.full_rope_dim = full_rope / 4;  // partial_rotary_factor=0.25

        auto* ffn = gpu->get("blk.0.ffn_gate.weight");
        cfg.intermediate_size = ffn ? ffn->dims[1] : 21504;

        // Detect MoE: check for router weight
        auto* router = gpu->get("blk.0.ffn_gate_inp.weight");
        if (router) {
            cfg.is_moe = true;
            cfg.num_experts = router->dims[1];  // [num_experts, hidden_size]
            cfg.num_experts_per_tok = gguf.get_u32(arch + ".expert_count_used", 8);
            // Detect moe_intermediate from merged expert tensor shape
            auto* gate_exps = gpu->get("blk.0.ffn_gate_exps.weight");
            if (gate_exps) {
                // shape: [num_experts * moe_inter, hidden_size]
                cfg.moe_intermediate_size = gate_exps->dims[1] / cfg.num_experts;
            } else {
                cfg.moe_intermediate_size = gguf.get_u32(arch + ".expert_feed_forward_length", 704);
            }
        }

        auto* embd = gpu->get("token_embd.weight");
        cfg.vocab_size = embd ? embd->dims[1] : 262144;

        printf("Gemma4 config: hidden=%d, inter=%d, layers=%d, vocab=%d\n",
               cfg.hidden_size, cfg.intermediate_size, cfg.num_layers, cfg.vocab_size);
        if (cfg.is_moe)
            printf("  MoE: %d experts, top-%d, expert_inter=%d\n",
                   cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size);
        printf("  Sliding: heads=%d/%d, hd=%d, rope_dim=%d, theta=%.0f, window=%d\n",
               cfg.slide_num_q, cfg.slide_num_kv, cfg.slide_hd, cfg.slide_rope_dim,
               cfg.slide_rope_theta, cfg.sliding_window);
        printf("  Full: heads=%d/%d, hd=%d, rope_dim=%d, theta=%.0f\n",
               cfg.full_num_q, cfg.full_num_kv, cfg.full_hd, cfg.full_rope_dim,
               cfg.full_rope_theta);
        printf("  Softcap=%.1f, embd_scale=%.2f\n", cfg.softcap, cfg.embd_scale);

        int n_full = 0, n_slide = 0;
        for (int i = 0; i < cfg.num_layers; i++) is_full_attn(i) ? n_full++ : n_slide++;
        printf("  Layer pattern: %d sliding + %d full\n", n_slide, n_full);

        // Preload layer output scales
        layer_scales.resize(cfg.num_layers, 1.0f);
        for (int l = 0; l < cfg.num_layers; l++) {
            char name[64];
            snprintf(name, sizeof(name), "blk.%d.layer_output_scale.weight", l);
            auto* t = gpu->get(name);
            if (t) {
                cudaSetDevice(t->gpu_id);
                cudaMemcpy(&layer_scales[l], t->data, sizeof(float), cudaMemcpyDeviceToHost);
            }
        }
        printf("  Layer scales: [0]=%.4f [1]=%.4f [59]=%.4f\n",
               layer_scales[0], layer_scales[1], layer_scales[cfg.num_layers - 1]);
    }

    void alloc_buffers() {
        int H = cfg.hidden_size;
        int I = cfg.intermediate_size;
        int max_q_dim = std::max(cfg.slide_num_q * cfg.slide_hd, cfg.full_num_q * cfg.full_hd);
        int max_kv_dim = std::max(cfg.slide_num_kv * cfg.slide_hd, cfg.full_num_kv * cfg.full_hd);

        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            auto& b = bufs[g];
            cudaMalloc(&b.norm_out, H * sizeof(half));
            cudaMalloc(&b.post_norm, H * sizeof(half));
            cudaMalloc(&b.q_proj, max_q_dim * sizeof(half));
            cudaMalloc(&b.k_proj, max_kv_dim * sizeof(half));
            cudaMalloc(&b.v_proj, max_kv_dim * sizeof(half));
            cudaMalloc(&b.attn_out, max_q_dim * sizeof(half));
            cudaMalloc(&b.mlp_gate, I * sizeof(half));
            cudaMalloc(&b.mlp_up, I * sizeof(half));
            cudaMalloc(&b.mlp_down, H * sizeof(half));
            cudaMalloc(&b.residual, H * sizeof(half));
            // scores: max across sliding (num_q * window) and full (num_q * max_seq)
            int max_scores = cfg.slide_num_q * 4096;  // generous upper bound
            cudaMalloc(&b.attn_scores, max_scores * sizeof(float));

            // TQ dequant buffers (allocated later in init_kv_caches if needed)
            b.tq_k_buf = nullptr;
            b.tq_v_buf = nullptr;

            // MoE buffers (fp32 expert path)
            if (cfg.is_moe) {
                int fused_I = 2 * cfg.moe_intermediate_size;
                cudaMalloc(&b.moe_output_f32, H * sizeof(float));
                cudaMalloc(&b.expert_gate_f32, fused_I * sizeof(float));
                cudaMalloc(&b.expert_down_f32, H * sizeof(float));
                cudaMalloc(&b.expert_gate, fused_I * sizeof(half));
                cudaMalloc(&b.expert_up, fused_I * sizeof(half));
                cudaMalloc(&b.expert_down, H * sizeof(half));
                cudaMalloc(&b.router_logits, cfg.num_experts * sizeof(float));
                cudaMalloc(&b.topk_indices, cfg.num_experts_per_tok * sizeof(int));
                cudaMalloc(&b.topk_weights, cfg.num_experts_per_tok * sizeof(float));
            }
        }
    }

    void init_kv_caches(int max_full_seq, bool turbo = false) {
        use_turboquant = turbo;
        int slide_kv_dim = cfg.slide_num_kv * cfg.slide_hd;
        int full_kv_dim = cfg.full_num_kv * cfg.full_hd;
        float slide_mb = 0, full_mb = 0;

        for (int l = 0; l < cfg.num_layers; l++) {
            int g = gpu->layer_gpu[l];
            if (use_turboquant) {
                if (is_full_attn(l)) {
                    tq_caches[l].alloc(max_full_seq, full_kv_dim, g);
                    float bpt = full_kv_dim / TQ_BLOCK_SIZE;
                    full_mb += (float)max_full_seq * bpt * sizeof(block_tq3) * 2 / 1e6f;
                } else {
                    tq_caches[l].alloc(cfg.sliding_window, slide_kv_dim, g);
                    float bpt = slide_kv_dim / TQ_BLOCK_SIZE;
                    slide_mb += (float)cfg.sliding_window * bpt * sizeof(block_tq3) * 2 / 1e6f;
                }
            } else {
                if (is_full_attn(l)) {
                    full_caches[l].alloc(max_full_seq, full_kv_dim, g);
                    full_mb += (float)max_full_seq * full_kv_dim * 2 * 2 / 1e6f;
                } else {
                    slide_caches[l].alloc(cfg.sliding_window, slide_kv_dim, g);
                    slide_mb += (float)cfg.sliding_window * slide_kv_dim * 2 * 2 / 1e6f;
                }
            }
        }
        // Alloc TQ dequant buffers per GPU
        if (use_turboquant) {
            int max_kv_dim = std::max(slide_kv_dim, full_kv_dim);
            int max_seq = std::max(max_full_seq, cfg.sliding_window);
            for (int g = 0; g < gpu->num_gpus; g++) {
                cudaSetDevice(g);
                cudaMalloc(&bufs[g].tq_k_buf, (size_t)max_seq * max_kv_dim * sizeof(half));
                cudaMalloc(&bufs[g].tq_v_buf, (size_t)max_seq * max_kv_dim * sizeof(half));
            }
        }

        printf("KV cache%s: sliding %.1f MB, full %.1f MB (max_seq=%d)\n",
               use_turboquant ? " [TurboQuant 3-bit]" : " [fp16]",
               slide_mb, full_mb, max_full_seq);
    }

    void init_rope(int max_seq) {
        int ng = gpu->num_gpus;
        // Sliding: theta=10K, rope_dim=256, freq uses rope_dim as denominator
        rope_slide.init(max_seq, cfg.slide_hd, cfg.slide_rope_dim, cfg.slide_rope_theta, ng, cfg.slide_rope_dim);
        // Full: theta=1M, rope_dim=128, partial rotation of 512-dim heads
        rope_full.init(max_seq, cfg.full_hd, cfg.full_rope_dim, cfg.full_rope_theta, ng, cfg.full_rope_dim);
    }

    void reset_all() {
        for (auto& [l, c] : slide_caches) c.reset();
        for (auto& [l, c] : full_caches) c.reset();
        for (auto& [l, c] : tq_caches) c.reset();
    }

    // Residual add: hidden += contribution (no scale — scale applied once at end of layer)
    void residual_add(half* hidden, half* contribution, int H, cudaStream_t stream) {
        add_kernel<<<(H + 255) / 256, 256, 0, stream>>>(hidden, contribution, H);
    }

    // Apply per-layer output scale to the sublayer contributions
    // In Gemma 4 mu-param: each sublayer (attn, mlp/moe) output is scaled before residual add
    // We scale the post-norm outputs in forward_*_attn and forward_moe using this value
    float get_layer_scale(int layer) const {
        if (layer < (int)layer_scales.size()) return layer_scales[layer];
        return 1.0f;
    }

    // ============ Forward: Sliding Window Attention ============
    void forward_sliding_attn(int layer, half* hidden, int pos, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& b = bufs[g];
        int H = cfg.hidden_size;
        int num_q = cfg.slide_num_q, num_kv = cfg.slide_num_kv, hd = cfg.slide_hd;
        int q_dim = num_q * hd, kv_dim = num_kv * hd;
        char lname[64];

        // 1. Input layernorm
        snprintf(lname, sizeof(lname), "blk.%d.attn_norm.weight", layer);
        auto* norm_w = gpu->get(lname);
        if (norm_w->type == GGML_TYPE_F32)
            rms_norm_f32w(b.norm_out, hidden, (float*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);
        else
            rms_norm(b.norm_out, hidden, (half*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);

        // 2. Q, K, V projections
        b.qi.quantize(b.norm_out, H, stream);
        snprintf(lname, sizeof(lname), "blk.%d.attn_q.weight", layer);
        quant_gemv(gpu->get(lname)->data, gpu->get(lname)->type, b.norm_out, b.q_proj, H, q_dim, &b.qi, stream);
        snprintf(lname, sizeof(lname), "blk.%d.attn_k.weight", layer);
        quant_gemv(gpu->get(lname)->data, gpu->get(lname)->type, b.norm_out, b.k_proj, H, kv_dim, &b.qi, stream);
        snprintf(lname, sizeof(lname), "blk.%d.attn_v.weight", layer);
        quant_gemv(gpu->get(lname)->data, gpu->get(lname)->type, b.norm_out, b.v_proj, H, kv_dim, &b.qi, stream);

        // 3. QK-norm (with learned weight) + V RMSNorm (no weight, eps-only)
        snprintf(lname, sizeof(lname), "blk.%d.attn_q_norm.weight", layer);
        auto* qn = gpu->get(lname);
        snprintf(lname, sizeof(lname), "blk.%d.attn_k_norm.weight", layer);
        auto* kn = gpu->get(lname);
        int tn = hd <= 128 ? hd : 128;
        if (qn) head_rms_norm_kernel<<<num_q, tn, tn * sizeof(float), stream>>>(b.q_proj, (float*)qn->data, num_q, hd, cfg.rms_norm_eps);
        if (kn) head_rms_norm_kernel<<<num_kv, tn, tn * sizeof(float), stream>>>(b.k_proj, (float*)kn->data, num_kv, hd, cfg.rms_norm_eps);
        // V gets RMSNorm without learned weight
        head_rms_norm_noweight_kernel<<<num_kv, tn, tn * sizeof(float), stream>>>(b.v_proj, num_kv, hd, cfg.rms_norm_eps);

        // 4. RoPE (sliding: full rotation, theta=10K)
        int half_rope = cfg.slide_rope_dim / 2;
        float* sin_pos = rope_slide.sin_table(g) + pos * half_rope;
        float* cos_pos = rope_slide.cos_table(g) + pos * half_rope;
        apply_rope_kernel<<<(num_q * half_rope + 255) / 256, 256, 0, stream>>>(b.q_proj, sin_pos, cos_pos, num_q, hd, cfg.slide_rope_dim);
        apply_rope_kernel<<<(num_kv * half_rope + 255) / 256, 256, 0, stream>>>(b.k_proj, sin_pos, cos_pos, num_kv, hd, cfg.slide_rope_dim);

        // 5. Store in KV cache (TurboQuant or fp16)
        int vlen, oldest;
        half *k_cache_ptr, *v_cache_ptr;
        if (use_turboquant) {
            auto& tq = tq_caches[layer];
            tq.store_ring(cfg.sliding_window, b.k_proj, b.v_proj, stream);
            vlen = tq.valid_len(cfg.sliding_window);
            oldest = tq.oldest_idx(cfg.sliding_window);
            // Dequant to temp buffers
            tq.load(vlen, b.tq_k_buf, b.tq_v_buf, stream);
            k_cache_ptr = b.tq_k_buf;
            v_cache_ptr = b.tq_v_buf;
        } else {
            auto& kv = slide_caches[layer];
            kv.store(b.k_proj, b.v_proj, stream);
            vlen = kv.valid_len();
            oldest = kv.oldest_idx();
            k_cache_ptr = kv.k;
            v_cache_ptr = kv.v;
        }

        // 6. Attention scoring (ring buffer)
        float scale = 1.0f / sqrtf((float)hd);
        int ws = use_turboquant ? vlen : cfg.sliding_window;  // TQ dequants linearly, fp16 uses ring
        dim3 grid_score(num_q, vlen);
        attn_score_ring_kernel<<<grid_score, 32, 0, stream>>>(
            b.q_proj, k_cache_ptr, b.attn_scores,
            num_q, num_kv, hd, vlen, use_turboquant ? 0 : oldest, ws, scale);

        // 7. Softmax
        int sblk = 1;
        while (sblk < vlen && sblk < 256) sblk <<= 1;
        softmax_kernel<<<num_q, sblk, sblk * sizeof(float), stream>>>(b.attn_scores, num_q, vlen);

        // 8. Value aggregation (ring buffer)
        attn_value_ring_kernel<<<num_q, std::min(hd, 256), 0, stream>>>(
            b.attn_scores, v_cache_ptr, b.attn_out,
            num_q, num_kv, hd, vlen, use_turboquant ? 0 : oldest, ws);

        // 9. Output projection
        b.qi_inter.quantize(b.attn_out, q_dim, stream);
        snprintf(lname, sizeof(lname), "blk.%d.attn_output.weight", layer);
        quant_gemv(gpu->get(lname)->data, gpu->get(lname)->type, b.attn_out, b.mlp_down, q_dim, H, &b.qi_inter, stream);

        // 10. Post-attention norm + scaled residual
        // layer_output_scale not used for residual (handled differently in Gemma 4)
        snprintf(lname, sizeof(lname), "blk.%d.post_attention_norm.weight", layer);
        auto* panw = gpu->get(lname);
        if (panw) {
            if (panw->type == GGML_TYPE_F32)
                rms_norm_f32w(b.post_norm, b.mlp_down, (float*)panw->data, 1, H, cfg.rms_norm_eps, stream);
            else
                rms_norm(b.post_norm, b.mlp_down, (half*)panw->data, 1, H, cfg.rms_norm_eps, stream);
            residual_add(hidden, b.post_norm, H, stream);
        } else {
            residual_add(hidden, b.mlp_down, H, stream);
        }
    }

    // ============ Forward: Full/Global Attention ============
    void forward_full_attn(int layer, half* hidden, int pos, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& b = bufs[g];
        int H = cfg.hidden_size;
        int num_q = cfg.full_num_q, num_kv = cfg.full_num_kv, hd = cfg.full_hd;
        int q_dim = num_q * hd, kv_dim = num_kv * hd;
        char lname[64];

        // 1. Input layernorm
        snprintf(lname, sizeof(lname), "blk.%d.attn_norm.weight", layer);
        auto* norm_w = gpu->get(lname);
        if (norm_w->type == GGML_TYPE_F32)
            rms_norm_f32w(b.norm_out, hidden, (float*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);
        else
            rms_norm(b.norm_out, hidden, (half*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);

        // 2. Q, K projections (V = K projection, but gets different norm)
        b.qi.quantize(b.norm_out, H, stream);
        snprintf(lname, sizeof(lname), "blk.%d.attn_q.weight", layer);
        quant_gemv(gpu->get(lname)->data, gpu->get(lname)->type, b.norm_out, b.q_proj, H, q_dim, &b.qi, stream);
        snprintf(lname, sizeof(lname), "blk.%d.attn_k.weight", layer);
        quant_gemv(gpu->get(lname)->data, gpu->get(lname)->type, b.norm_out, b.k_proj, H, kv_dim, &b.qi, stream);

        // 3. Copy K projection to V buffer before norms (V starts as same projection as K)
        cudaMemcpyAsync(b.v_proj, b.k_proj, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);

        // 4. Apply different norms: K gets learned weight norm, V gets eps-only norm
        snprintf(lname, sizeof(lname), "blk.%d.attn_q_norm.weight", layer);
        auto* qn = gpu->get(lname);
        snprintf(lname, sizeof(lname), "blk.%d.attn_k_norm.weight", layer);
        auto* kn = gpu->get(lname);
        int tn = hd <= 128 ? hd : 128;
        if (qn) head_rms_norm_kernel<<<num_q, tn, tn * sizeof(float), stream>>>(b.q_proj, (float*)qn->data, num_q, hd, cfg.rms_norm_eps);
        if (kn) head_rms_norm_kernel<<<num_kv, tn, tn * sizeof(float), stream>>>(b.k_proj, (float*)kn->data, num_kv, hd, cfg.rms_norm_eps);
        // V gets RMSNorm without learned weight (eps-only)
        head_rms_norm_noweight_kernel<<<num_kv, tn, tn * sizeof(float), stream>>>(b.v_proj, num_kv, hd, cfg.rms_norm_eps);

        // 5. RoPE (full: partial rotation, theta=1M) — only on Q and K, not V
        int half_rope = cfg.full_rope_dim / 2;
        float* sin_pos = rope_full.sin_table(g) + pos * half_rope;
        float* cos_pos = rope_full.cos_table(g) + pos * half_rope;
        apply_rope_kernel<<<(num_q * half_rope + 255) / 256, 256, 0, stream>>>(b.q_proj, sin_pos, cos_pos, num_q, hd, cfg.full_rope_dim);
        apply_rope_kernel<<<(num_kv * half_rope + 255) / 256, 256, 0, stream>>>(b.k_proj, sin_pos, cos_pos, num_kv, hd, cfg.full_rope_dim);

        // 6. Store K and V
        int seq_len;
        half *k_cache_ptr, *v_cache_ptr;
        if (use_turboquant) {
            auto& tq = tq_caches[layer];
            tq.store(b.k_proj, b.v_proj, stream);
            seq_len = tq.valid_len();
            tq.load(seq_len, b.tq_k_buf, b.tq_v_buf, stream);
            k_cache_ptr = b.tq_k_buf;
            v_cache_ptr = b.tq_v_buf;
        } else {
            auto& kv = full_caches[layer];
            kv.store(b.k_proj, b.v_proj, stream);
            seq_len = kv.valid_len();
            k_cache_ptr = kv.k;
            v_cache_ptr = kv.v;
        }

        // 7. Standard attention scoring
        float scale = 1.0f / sqrtf((float)hd);
        dim3 grid_score(num_q, seq_len);
        attn_score_kernel<<<grid_score, 32, 0, stream>>>(
            b.q_proj, k_cache_ptr, b.attn_scores,
            num_q, num_kv, hd, seq_len, scale);

        // 8. Softmax
        int sblk = 1;
        while (sblk < seq_len && sblk < 256) sblk <<= 1;
        softmax_kernel<<<num_q, sblk, sblk * sizeof(float), stream>>>(b.attn_scores, num_q, seq_len);

        // 9. Value aggregation
        attn_value_kernel<<<num_q, std::min(hd, 256), 0, stream>>>(
            b.attn_scores, v_cache_ptr, b.attn_out,
            num_q, num_kv, hd, seq_len);

        // 10. Output projection
        b.qi_inter.quantize(b.attn_out, q_dim, stream);
        snprintf(lname, sizeof(lname), "blk.%d.attn_output.weight", layer);
        quant_gemv(gpu->get(lname)->data, gpu->get(lname)->type, b.attn_out, b.mlp_down, q_dim, H, &b.qi_inter, stream);

        // 11. Post-attention norm + scaled residual
        // layer_output_scale not used for residual (handled differently in Gemma 4)
        snprintf(lname, sizeof(lname), "blk.%d.post_attention_norm.weight", layer);
        auto* panw = gpu->get(lname);
        if (panw) {
            if (panw->type == GGML_TYPE_F32)
                rms_norm_f32w(b.post_norm, b.mlp_down, (float*)panw->data, 1, H, cfg.rms_norm_eps, stream);
            else
                rms_norm(b.post_norm, b.mlp_down, (half*)panw->data, 1, H, cfg.rms_norm_eps, stream);
            residual_add(hidden, b.post_norm, H, stream);
        } else {
            residual_add(hidden, b.mlp_down, H, stream);
        }
    }

    // ============ Forward: MLP (GeGLU + 4 norms) ============
    void forward_mlp(int layer, half* hidden, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& b = bufs[g];
        int H = cfg.hidden_size;
        int I = cfg.intermediate_size;
        char lname[64];

        // 1. Pre-feedforward norm
        snprintf(lname, sizeof(lname), "blk.%d.ffn_norm.weight", layer);
        auto* ffn_norm = gpu->get(lname);
        if (ffn_norm->type == GGML_TYPE_F32)
            rms_norm_f32w(b.norm_out, hidden, (float*)ffn_norm->data, 1, H, cfg.rms_norm_eps, stream);
        else
            rms_norm(b.norm_out, hidden, (half*)ffn_norm->data, 1, H, cfg.rms_norm_eps, stream);

        // 2. Gate and up projections
        b.qi.quantize(b.norm_out, H, stream);
        snprintf(lname, sizeof(lname), "blk.%d.ffn_gate.weight", layer);
        quant_gemv(gpu->get(lname)->data, gpu->get(lname)->type, b.norm_out, b.mlp_gate, H, I, &b.qi, stream);
        snprintf(lname, sizeof(lname), "blk.%d.ffn_up.weight", layer);
        quant_gemv(gpu->get(lname)->data, gpu->get(lname)->type, b.norm_out, b.mlp_up, H, I, &b.qi, stream);

        // 3. GeGLU: gate = gelu_tanh(gate) * up
        gelu_tanh_mul_kernel<<<(I + 255) / 256, 256, 0, stream>>>(b.mlp_gate, b.mlp_up, I);

        // 4. Down projection
        b.qi_inter.quantize(b.mlp_gate, I, stream);
        snprintf(lname, sizeof(lname), "blk.%d.ffn_down.weight", layer);
        quant_gemv(gpu->get(lname)->data, gpu->get(lname)->type, b.mlp_gate, b.mlp_down, I, H, &b.qi_inter, stream);

        // 5. Post-feedforward norm + scaled residual
        // layer_output_scale not used for residual (handled differently in Gemma 4)
        snprintf(lname, sizeof(lname), "blk.%d.post_ffw_norm.weight", layer);
        auto* post_norm = gpu->get(lname);
        if (post_norm) {
            if (post_norm->type == GGML_TYPE_F32)
                rms_norm_f32w(b.post_norm, b.mlp_down, (float*)post_norm->data, 1, H, cfg.rms_norm_eps, stream);
            else
                rms_norm(b.post_norm, b.mlp_down, (half*)post_norm->data, 1, H, cfg.rms_norm_eps, stream);
            residual_add(hidden, b.post_norm, H, stream);
        } else {
            residual_add(hidden, b.mlp_down, H, stream);
        }
    }

    // ============ Forward: MoE (Gemma 4 / LLAMA4 style) ============
    // Key differences from standard MoE:
    //   1. Router uses SIGMOID (not softmax)
    //   2. Weights applied BEFORE expert FFN (weight_before_ffn)
    //   3. Top-k selection on raw logits
    //   4. Shared expert added after routed experts
    void forward_moe(int layer, half* hidden, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& b = bufs[g];
        int H = cfg.hidden_size;
        int E = cfg.num_experts;
        int topk = cfg.num_experts_per_tok;
        int moe_I = cfg.moe_intermediate_size;
        char lname[64];

        // 1. Pre-feedforward norm
        snprintf(lname, sizeof(lname), "blk.%d.ffn_norm.weight", layer);
        auto* ffn_norm = gpu->get(lname);
        if (ffn_norm->type == GGML_TYPE_F32)
            rms_norm_f32w(b.norm_out, hidden, (float*)ffn_norm->data, 1, H, cfg.rms_norm_eps, stream);
        else
            rms_norm(b.norm_out, hidden, (half*)ffn_norm->data, 1, H, cfg.rms_norm_eps, stream);

        // Debug: check hidden state entering MoE
        if (layer <= 5) {
            cudaStreamSynchronize(stream);
            std::vector<half> d(H); cudaMemcpy(d.data(), hidden, H*sizeof(half), cudaMemcpyDeviceToHost);
            float s=0; for(int i=0;i<H;i++) s+=fabsf(__half2float(d[i]));
            cudaMemcpy(d.data(), b.norm_out, H*sizeof(half), cudaMemcpyDeviceToHost);
            float sn=0; for(int i=0;i<H;i++) sn+=fabsf(__half2float(d[i]));
            printf("[MoE-DBG] L%d hidden_in=%.1f norm_out=%.1f\n", layer, s, sn);
        }

        // 2. Router: sigmoid gating
        snprintf(lname, sizeof(lname), "blk.%d.ffn_gate_inp.weight", layer);
        auto* router_w = gpu->get(lname);
        router_gemv_f32<<<E, 32, 0, stream>>>((float*)router_w->data, b.norm_out, b.router_logits, H, E);

        // 3. CPU top-k + softmax (reliable fallback for debugging)
        std::vector<float> h_logits(E);
        cudaMemcpyAsync(h_logits.data(), b.router_logits, E * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        struct ES { int idx; float score; };
        std::vector<ES> experts_vec(E);
        for (int i = 0; i < E; i++) experts_vec[i] = {i, h_logits[i]};
        std::partial_sort(experts_vec.begin(), experts_vec.begin() + topk, experts_vec.end(),
                          [](const ES& a, const ES& b) { return a.score > b.score; });
        float max_s = experts_vec[0].score, sum_exp = 0;
        int h_indices[8]; float h_weights[8];
        for (int i = 0; i < topk; i++) {
            experts_vec[i].score = expf(experts_vec[i].score - max_s);
            sum_exp += experts_vec[i].score;
        }
        for (int i = 0; i < topk; i++) {
            h_indices[i] = experts_vec[i].idx;
            h_weights[i] = experts_vec[i].score / sum_exp;
        }

        // 4. Zero fp32 accumulator
        zero_f32_kernel<<<(H + 255) / 256, 256, 0, stream>>>(b.moe_output_f32, H);

        // 5. Expert tensors
        snprintf(lname, sizeof(lname), "blk.%d.ffn_gate_up_exps.weight", layer);
        auto* gate_up_exps = gpu->get(lname);
        snprintf(lname, sizeof(lname), "blk.%d.ffn_down_exps.weight", layer);
        auto* down_exps = gpu->get(lname);

        int fused_I = 2 * moe_I;
        size_t gu_expert_bytes = (size_t)fused_I * ggml_row_bytes(gate_up_exps->type, H);
        size_t dn_expert_bytes = (size_t)H * ggml_row_bytes(down_exps->type, moe_I);

        // Quantize input once for all experts
        b.qi.quantize(b.norm_out, H, stream);

        // 6. Process each expert with fp16 + clamp (known working path)
        for (int t = 0; t < topk; t++) {
            int eidx = h_indices[t];
            float w = h_weights[t];  // sigmoid weight

            // Fused gate+up GEMV → fp16 + clamp
            void* gu_ptr = (uint8_t*)gate_up_exps->data + eidx * gu_expert_bytes;
            quant_gemv(gu_ptr, gate_up_exps->type, b.norm_out, b.expert_gate, H, fused_I, &b.qi, stream);
            clamp_fp16_kernel<<<(fused_I + 255) / 256, 256, 0, stream>>>(b.expert_gate, fused_I);

            // GeGLU fp16
            gelu_tanh_mul_kernel<<<(moe_I + 255) / 256, 256, 0, stream>>>(b.expert_gate, b.expert_gate + moe_I, moe_I);

            // Down GEMV → fp16 + clamp
            b.qi_expert.quantize(b.expert_gate, moe_I, stream);
            void* dn_ptr = (uint8_t*)down_exps->data + eidx * dn_expert_bytes;
            quant_gemv(dn_ptr, down_exps->type, b.expert_gate, b.expert_down, moe_I, H, &b.qi_expert, stream);
            clamp_fp16_kernel<<<(H + 255) / 256, 256, 0, stream>>>(b.expert_down, H);

            // Accumulate into fp32 (sigmoid weight × expert output)
            weighted_add_f32_kernel<<<(H + 255) / 256, 256, 0, stream>>>(b.moe_output_f32, b.expert_down, w, H);
        }

        // Convert fp32 accumulator to fp16
        f32_to_fp16_kernel<<<(H + 255) / 256, 256, 0, stream>>>(b.moe_output_f32, b.expert_down, H);

        // 7. Post-MoE norm + scaled residual
        // layer_output_scale not used for residual (handled differently in Gemma 4)
        snprintf(lname, sizeof(lname), "blk.%d.post_ffw_norm_1.weight", layer);
        auto* post_moe_norm = gpu->get(lname);
        if (!post_moe_norm) {
            snprintf(lname, sizeof(lname), "blk.%d.post_ffw_norm.weight", layer);
            post_moe_norm = gpu->get(lname);
        }
        // expert_down now holds the fp16 MoE output (converted from fp32 accumulator)
        half* moe_result = b.expert_down;
        if (post_moe_norm) {
            if (post_moe_norm->type == GGML_TYPE_F32)
                rms_norm_f32w(b.post_norm, moe_result, (float*)post_moe_norm->data, 1, H, cfg.rms_norm_eps, stream);
            else
                rms_norm(b.post_norm, moe_result, (half*)post_moe_norm->data, 1, H, cfg.rms_norm_eps, stream);
            residual_add(hidden, b.post_norm, H, stream);
        } else {
            residual_add(hidden, moe_result, H, stream);
        }

        // Debug: check after MoE post-norm
        if (layer <= 5) {
            cudaStreamSynchronize(stream);
            std::vector<half> d(H); cudaMemcpy(d.data(), hidden, H*sizeof(half), cudaMemcpyDeviceToHost);
            float s=0; for(int i=0;i<H;i++) s+=fabsf(__half2float(d[i]));
            printf("[MoE-DBG] L%d hidden_after_moe=%.1f\n", layer, s);
        }

        // Debug: check after MoE experts (before shared)
        if (layer <= 5) {
            cudaStreamSynchronize(stream);
            std::vector<half> d(H); cudaMemcpy(d.data(), hidden, H*sizeof(half), cudaMemcpyDeviceToHost);
            float s=0; for(int i=0;i<H;i++) s+=fabsf(__half2float(d[i]));
            printf("[MoE-DBG] L%d before_shared=%.1f\n", layer, s);
        }

        // 8. Shared expert (separate norm path)
        snprintf(lname, sizeof(lname), "blk.%d.ffn_gate.weight", layer);
        auto* shared_gate_w = gpu->get(lname);
        if (shared_gate_w) {
            int sI = cfg.intermediate_size;

            // Shared expert input norm (pre_ffw_norm_2 or reuse ffn_norm)
            snprintf(lname, sizeof(lname), "blk.%d.pre_ffw_norm_2.weight", layer);
            auto* shared_in_norm = gpu->get(lname);
            half* shared_input = b.norm_out;  // default: reuse MoE normed input
            if (shared_in_norm) {
                // Separate norm for shared expert
                if (shared_in_norm->type == GGML_TYPE_F32)
                    rms_norm_f32w(b.residual, hidden, (float*)shared_in_norm->data, 1, H, cfg.rms_norm_eps, stream);
                else
                    rms_norm(b.residual, hidden, (half*)shared_in_norm->data, 1, H, cfg.rms_norm_eps, stream);
                shared_input = b.residual;
                b.qi.quantize(shared_input, H, stream);
            }

            snprintf(lname, sizeof(lname), "blk.%d.ffn_up.weight", layer);
            auto* shared_up_w = gpu->get(lname);
            snprintf(lname, sizeof(lname), "blk.%d.ffn_down.weight", layer);
            auto* shared_down_w = gpu->get(lname);

            quant_gemv(shared_gate_w->data, shared_gate_w->type, shared_input, b.mlp_gate, H, sI, &b.qi, stream);
            quant_gemv(shared_up_w->data, shared_up_w->type, shared_input, b.mlp_up, H, sI, &b.qi, stream);
            clamp_fp16_kernel<<<(sI + 255) / 256, 256, 0, stream>>>(b.mlp_gate, sI);
            clamp_fp16_kernel<<<(sI + 255) / 256, 256, 0, stream>>>(b.mlp_up, sI);
            gelu_tanh_mul_kernel<<<(sI + 255) / 256, 256, 0, stream>>>(b.mlp_gate, b.mlp_up, sI);
            b.qi_inter.quantize(b.mlp_gate, sI, stream);
            quant_gemv(shared_down_w->data, shared_down_w->type, b.mlp_gate, b.mlp_down, sI, H, &b.qi_inter, stream);
            clamp_fp16_kernel<<<(H + 255) / 256, 256, 0, stream>>>(b.mlp_down, H);

            // Post-shared-expert norm + scaled residual
            snprintf(lname, sizeof(lname), "blk.%d.post_ffw_norm_2.weight", layer);
            auto* post_shared_norm = gpu->get(lname);
            if (post_shared_norm) {
                if (post_shared_norm->type == GGML_TYPE_F32)
                    rms_norm_f32w(b.post_norm, b.mlp_down, (float*)post_shared_norm->data, 1, H, cfg.rms_norm_eps, stream);
                else
                    rms_norm(b.post_norm, b.mlp_down, (half*)post_shared_norm->data, 1, H, cfg.rms_norm_eps, stream);
                residual_add(hidden, b.post_norm, H, stream);
            } else {
                residual_add(hidden, b.mlp_down, H, stream);
            }
        }
    }
};
