#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include "turboquant.cuh"

// ============ RoPE precomputed tables ============

struct RoPETable {
    float* sin_tables[4];  // per GPU
    float* cos_tables[4];  // per GPU
    int max_seq;
    int head_dim;
    int num_gpus;
    
    int rope_dim;
    void init(int _max_seq, int _head_dim, int _rope_dim, float theta, int _num_gpus) {
        max_seq = _max_seq;
        head_dim = _head_dim;
        num_gpus = _num_gpus;
        rope_dim = _rope_dim;
        int half_dim = rope_dim / 2;
        
        // Compute on CPU
        std::vector<float> h_sin(max_seq * half_dim);
        std::vector<float> h_cos(max_seq * half_dim);
        
        for (int pos = 0; pos < max_seq; pos++) {
            for (int i = 0; i < half_dim; i++) {
                // standard NeoX RoPE: freq depends on rope_dim (n_rot), not head_dim
                float freq = 1.0f / powf(theta, (float)(2 * i) / rope_dim);
                float angle = pos * freq;
                h_sin[pos * half_dim + i] = sinf(angle);
                h_cos[pos * half_dim + i] = cosf(angle);
            }
        }
        
        // Copy to ALL GPUs
        size_t bytes = max_seq * half_dim * sizeof(float);
        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&sin_tables[g], bytes);
            cudaMalloc(&cos_tables[g], bytes);
            cudaMemcpy(sin_tables[g], h_sin.data(), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(cos_tables[g], h_cos.data(), bytes, cudaMemcpyHostToDevice);
        }
        printf("RoPE table: %d positions, dim=%d, theta=%.0f (on %d GPUs)\n", max_seq, head_dim, theta, num_gpus);
    }
    
    float* sin_table(int gpu_id) { return sin_tables[gpu_id]; }
    float* cos_table(int gpu_id) { return cos_tables[gpu_id]; }
    
    void free_table() {
        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            cudaFree(sin_tables[g]); cudaFree(cos_tables[g]);
        }
    }
};

// ============ Apply RoPE to Q or K (NeoX style) ============
// x: [num_heads, head_dim], modifies in-place

// RoPE with partial rotary: only first rope_dim dimensions per head
__global__ void apply_rope_kernel(
    half* __restrict__ x,            // [num_heads * head_dim]
    const float* __restrict__ sin_t, // [rope_dim/2]
    const float* __restrict__ cos_t, // [rope_dim/2]
    int num_heads,
    int head_dim,
    int rope_dim      // how many dims to rotate (e.g. 64)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_rope = rope_dim / 2;
    int total = num_heads * half_rope;
    if (idx >= total) return;

    int head = idx / half_rope;
    int i = idx % half_rope;

    half* base = x + head * head_dim;
    float x0 = __half2float(base[i]);
    float x1 = __half2float(base[i + half_rope]);

    float s = sin_t[i];
    float c = cos_t[i];

    base[i]             = __float2half(x0 * c - x1 * s);
    base[i + half_rope] = __float2half(x1 * c + x0 * s);
}

// FP32 variant
__global__ void apply_rope_kernel_f32(
    float* __restrict__ x,
    const float* __restrict__ sin_t,
    const float* __restrict__ cos_t,
    int num_heads,
    int head_dim,
    int rope_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_rope = rope_dim / 2;
    int total = num_heads * half_rope;
    if (idx >= total) return;

    int head = idx / half_rope;
    int i = idx % half_rope;

    float* base = x + head * head_dim;
    float x0 = base[i];
    float x1 = base[i + half_rope];

    float s = sin_t[i];
    float c = cos_t[i];

    base[i]             = x0 * c - x1 * s;
    base[i + half_rope] = x1 * c + x0 * s;
}

// ============ Head-wise RMSNorm ============
// x: [num_heads, head_dim], weight: [head_dim]

__global__ void head_rms_norm_kernel(
    half* __restrict__ x,
    const float* __restrict__ weight,
    int num_heads,
    int head_dim,
    float eps
) {
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
        xh[i] = __float2half(__half2float(xh[i]) * rms * weight[i]);
}

// FP32 variant
__global__ void head_rms_norm_kernel_f32(
    float* __restrict__ x,
    const float* __restrict__ weight,
    int num_heads,
    int head_dim,
    float eps
) {
    int head = blockIdx.x;
    if (head >= num_heads) return;

    float* xh = x + head * head_dim;

    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xh[i];
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
        xh[i] = xh[i] * rms * weight[i];
}

// ============ Attention Score (single query vs cached keys) ============
// q: [num_q_heads, head_dim]
// k_cache: [seq_len, num_kv_heads, head_dim]  (dequantized fp16)
// scores: [num_q_heads, seq_len]
// With GQA: each q head maps to k head = q_head / gqa_ratio

// Convert fp16 K/V to fp32 and store in cache at given position
__global__ void store_kv_fp32_kernel(
    const half* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __half2float(src[idx]);
}

// FP32→FP32 store (no-op type-wise; just copy)
__global__ void store_kv_f32_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

__global__ void attn_score_kernel(
    const half* __restrict__ q,         // [num_q_heads * head_dim]
    const float* __restrict__ k_cache,  // [seq_len * num_kv_heads * head_dim] FP32
    float* __restrict__ scores,         // [num_q_heads * seq_len]
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    float scale
) {
    int q_head = blockIdx.x;
    int pos = blockIdx.y;
    if (q_head >= num_q_heads || pos >= seq_len) return;

    int kv_head = q_head / (num_q_heads / num_kv_heads);

    const half* qh = q + q_head * head_dim;
    const float* kh = k_cache + pos * num_kv_heads * head_dim + kv_head * head_dim;

    // Dot product (Q is fp16, K is fp32, accumulate in fp32)
    float sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        sum += __half2float(qh[i]) * kh[i];

    // Warp-level reduce
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);

    // Cross-warp reduce via shared memory (BUG fix: was missing)
    __shared__ float warp_sums[8];  // up to 8 warps for blockDim 256
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int n_warps = (blockDim.x + 31) >> 5;
    if (lane_id == 0) warp_sums[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        sum = (lane_id < n_warps) ? warp_sums[lane_id] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane_id == 0)
            scores[q_head * seq_len + pos] = sum * scale;
    }
}

// FP16 K-cache variant (Gemma uses fp16 KV cache)
__global__ void attn_score_kernel_h(
    const half* __restrict__ q,         // [num_q_heads * head_dim]
    const half* __restrict__ k_cache,   // [seq_len * num_kv_heads * head_dim] FP16
    float* __restrict__ scores,         // [num_q_heads * seq_len]
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    float scale
) {
    int q_head = blockIdx.x;
    int pos = blockIdx.y;
    if (q_head >= num_q_heads || pos >= seq_len) return;

    int kv_head = q_head / (num_q_heads / num_kv_heads);

    const half* qh = q + q_head * head_dim;
    const half* kh = k_cache + pos * num_kv_heads * head_dim + kv_head * head_dim;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        sum += __half2float(qh[i]) * __half2float(kh[i]);

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);

    __shared__ float warp_sums[8];
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int n_warps = (blockDim.x + 31) >> 5;
    if (lane_id == 0) warp_sums[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        sum = (lane_id < n_warps) ? warp_sums[lane_id] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane_id == 0)
            scores[q_head * seq_len + pos] = sum * scale;
    }
}

// FP32 Q variant
__global__ void attn_score_kernel_f32(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    float* __restrict__ scores,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    float scale
) {
    int q_head = blockIdx.x;
    int pos = blockIdx.y;
    if (q_head >= num_q_heads || pos >= seq_len) return;

    int kv_head = q_head / (num_q_heads / num_kv_heads);

    const float* qh = q + q_head * head_dim;
    const float* kh = k_cache + pos * num_kv_heads * head_dim + kv_head * head_dim;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        sum += qh[i] * kh[i];

    // Warp-level reduce
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);

    // Cross-warp reduce
    __shared__ float warp_sums[8];
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int n_warps = (blockDim.x + 31) >> 5;
    if (lane_id == 0) warp_sums[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        sum = (lane_id < n_warps) ? warp_sums[lane_id] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane_id == 0)
            scores[q_head * seq_len + pos] = sum * scale;
    }
}

// ============ Softmax (per row) ============

__global__ void softmax_kernel(
    float* __restrict__ scores,  // [num_heads, seq_len] — modified inplace
    int num_heads,
    int seq_len
) {
    int head = blockIdx.x;
    if (head >= num_heads) return;
    
    float* row = scores + head * seq_len;
    
    extern __shared__ float sdata[];
    
    // Find max
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x)
        max_val = fmaxf(max_val, row[i]);
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();
    
    // Exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float e = expf(row[i] - max_val);
        row[i] = e;
        sum += e;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    sum = sdata[0];
    
    // Normalize
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x)
        row[i] *= inv_sum;
}

// ============ Attention Value (scores @ V) ============
// scores: [num_q_heads, seq_len] (after softmax)
// v_cache: [seq_len, num_kv_heads, head_dim]
// output: [num_q_heads, head_dim]

__global__ void attn_value_kernel(
    const float* __restrict__ scores,   // [num_q_heads * seq_len]
    const float* __restrict__ v_cache,  // [seq_len * num_kv_heads * head_dim] FP32
    half* __restrict__ output,          // [num_q_heads * head_dim]
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len
) {
    int q_head = blockIdx.x;
    if (q_head >= num_q_heads) return;

    int kv_head = q_head / (num_q_heads / num_kv_heads);
    const float* sc = scores + q_head * seq_len;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            sum += sc[pos] * v_cache[pos * num_kv_heads * head_dim + kv_head * head_dim + d];
        }
        output[q_head * head_dim + d] = __float2half(sum);
    }
}

// FP16 V-cache variant (Gemma uses fp16 KV cache)
__global__ void attn_value_kernel_h(
    const float* __restrict__ scores,
    const half* __restrict__ v_cache,
    half* __restrict__ output,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len
) {
    int q_head = blockIdx.x;
    if (q_head >= num_q_heads) return;

    int kv_head = q_head / (num_q_heads / num_kv_heads);
    const float* sc = scores + q_head * seq_len;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            sum += sc[pos] * __half2float(v_cache[pos * num_kv_heads * head_dim + kv_head * head_dim + d]);
        }
        output[q_head * head_dim + d] = __float2half(sum);
    }
}

// FP32 output variant
__global__ void attn_value_kernel_f32(
    const float* __restrict__ scores,
    const float* __restrict__ v_cache,
    float* __restrict__ output,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len
) {
    int q_head = blockIdx.x;
    if (q_head >= num_q_heads) return;

    int kv_head = q_head / (num_q_heads / num_kv_heads);
    const float* sc = scores + q_head * seq_len;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            sum += sc[pos] * v_cache[pos * num_kv_heads * head_dim + kv_head * head_dim + d];
        }
        output[q_head * head_dim + d] = sum;
    }
}

// ============ Apply output gate: out *= sigmoid(gate) ============

__global__ void apply_gate_sigmoid(
    half* __restrict__ output,    // [num_heads * head_dim]
    const half* __restrict__ gate, // [num_heads * head_dim]
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float o = __half2float(output[idx]);
    float g = __half2float(gate[idx]);
    float sig = 1.0f / (1.0f + expf(-g));
    output[idx] = __float2half(o * sig);
}

__global__ void apply_gate_sigmoid_f32(
    float* __restrict__ output,
    const float* __restrict__ gate,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float o = output[idx];
    float g = gate[idx];
    float sig = 1.0f / (1.0f + expf(-g));
    output[idx] = o * sig;
}

// Deinterleave Q+Gate: [h0_q(hd)|h0_g(hd)|h1_q(hd)|h1_g(hd)...] → q[N*hd], gate[N*hd]
__global__ void deinterleave_qg_kernel(
    const half* __restrict__ src,   // [num_heads * head_dim * 2]
    half* __restrict__ q_out,       // [num_heads * head_dim]
    half* __restrict__ gate_out,    // [num_heads * head_dim]
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_heads * head_dim) return;
    int head = idx / head_dim;
    int d = idx % head_dim;
    q_out[idx]    = src[head * head_dim * 2 + d];
    gate_out[idx] = src[head * head_dim * 2 + head_dim + d];
}

__global__ void deinterleave_qg_kernel_f32(
    const float* __restrict__ src,
    float* __restrict__ q_out,
    float* __restrict__ gate_out,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_heads * head_dim) return;
    int head = idx / head_dim;
    int d = idx % head_dim;
    q_out[idx]    = src[head * head_dim * 2 + d];
    gate_out[idx] = src[head * head_dim * 2 + head_dim + d];
}
