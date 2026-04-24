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

// DDTree: same as attn_score_kernel_h but writes -INF for KV slots that the
// current query is not allowed to attend to (sibling tree branches). A mask
// is applied to positions in [mask_start, mask_start + mask_len) — bit k of
// mask_bits set means slot (mask_start + k) is an ancestor of the query.
// Positions outside the [mask_start, mask_start+mask_len) window are treated
// as the pre-tree prefix and remain visible (no mask).
__global__ void attn_score_kernel_h_tree_masked(
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    float* __restrict__ scores,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    float scale,
    int mask_start,
    int mask_len,
    uint32_t mask_bits
) {
    int q_head = blockIdx.x;
    int pos = blockIdx.y;
    if (q_head >= num_q_heads || pos >= seq_len) return;

    bool in_tree = (pos >= mask_start) && (pos < mask_start + mask_len);
    bool masked  = in_tree && !((mask_bits >> (pos - mask_start)) & 1u);

    int kv_head = q_head / (num_q_heads / num_kv_heads);
    const half* qh = q + q_head * head_dim;
    const half* kh = k_cache + pos * num_kv_heads * head_dim + kv_head * head_dim;

    float sum = 0.0f;
    if (!masked) {
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
            sum += __half2float(qh[i]) * __half2float(kh[i]);
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, off);
    }

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
        if (lane_id == 0) {
            float out = masked ? -1e30f : (sum * scale);
            scores[q_head * seq_len + pos] = out;
        }
    }
}

// ============ TurboQuant 3-bit fused attention score kernel ============
// Block layout: ONE block per (kv_head, pos) processing all `gqa_ratio`
// query heads that share the kv_head. Cooperative dequant of K[pos,
// kv_head] runs once and is reused for all 6 dot products (Qwen3.5-27B
// GQA = 24/4 = 6) — 6x less dequant work than the naive (q_head, pos)
// layout.
//
// Launch: <<< dim3(num_kv_heads, seq_len), head_dim,
//             head_dim*sizeof(float) >>>
// Constraints: head_dim multiple of TQ_BLOCK_SIZE (128); blockDim.x ==
// head_dim; gqa_ratio <= 8 (we keep accumulators in registers).
__global__ void attn_score_kernel_tq3(
    const half* __restrict__ q,            // [num_q_heads * head_dim]
    const block_tq3* __restrict__ k_cache, // [seq_len * blocks_per_token]
    float* __restrict__ scores,            // [num_q_heads * seq_len]
    int num_q_heads, int num_kv_heads, int head_dim, int seq_len, float scale
) {
    int kv_head = blockIdx.x;
    int pos     = blockIdx.y;
    if (kv_head >= num_kv_heads || pos >= seq_len) return;

    int gqa                = num_q_heads / num_kv_heads;
    int blocks_per_token   = num_kv_heads * head_dim / TQ_BLOCK_SIZE;
    int blocks_per_kv_head = head_dim / TQ_BLOCK_SIZE;

    extern __shared__ float k_dequant[];   // [head_dim]

    int tid  = threadIdx.x;
    int bg   = tid / TQ_BLOCK_SIZE;
    int lane = tid % TQ_BLOCK_SIZE;

    // ── 1. Cooperative dequant of K[pos, kv_head] (head_dim elements,
    //       which is blocks_per_kv_head separate block_tq3 each handled
    //       by a 128-thread group).
    if (bg < blocks_per_kv_head) {
        const block_tq3* blk = &k_cache[(size_t)pos * blocks_per_token
                                        + kv_head * blocks_per_kv_head + bg];
        float* my = &k_dequant[bg * TQ_BLOCK_SIZE];

        int byte_off = (lane * 3) >> 3;
        int bit_off  = (lane * 3) & 7;
        uint32_t packed = (uint32_t)blk->qs[byte_off];
        if (byte_off + 1 < 48)
            packed |= ((uint32_t)blk->qs[byte_off + 1]) << 8;
        int idx = (packed >> bit_off) & 0x7;
        my[lane] = d_tq3_centroids[idx];
        __syncthreads();

        for (int s = 1; s < TQ_BLOCK_SIZE; s <<= 1) {
            float a = my[lane];
            float b = my[lane ^ s];
            __syncthreads();
            if ((lane & s) == 0) my[lane] = a + b;
            else                 my[lane] = b - a;
            __syncthreads();
        }
        // Inverse RHT: 1/√d already applied via scale_norm; sign vector
        // closes out the rotation. Encoder applies signs BEFORE the WHT,
        // so we apply them AFTER iWHT (matrix transpose order).
        float scale_norm = rsqrtf((float)TQ_BLOCK_SIZE) * blk->norm;
        my[lane] *= d_tq3_signs[lane] * scale_norm;
    }
    __syncthreads();

    // ── 2. Compute Q · K_dequant for each of the gqa query heads sharing
    //       this kv_head. We hold the K element in a register and walk
    //       through the gqa Q heads sequentially — cheap relative to the
    //       dequant cost we just paid once.
    float k_val = k_dequant[tid];
    __shared__ float warp_sums[8];

    #pragma unroll
    for (int qi = 0; qi < 8; qi++) {
        if (qi >= gqa) break;
        int q_head = kv_head * gqa + qi;
        const half* qh = q + q_head * head_dim;
        float prod = __half2float(qh[tid]) * k_val;

        for (int off = 16; off > 0; off >>= 1)
            prod += __shfl_xor_sync(0xffffffff, prod, off);
        int warp_id = tid >> 5;
        int lane_id = tid & 31;
        int n_warps = (blockDim.x + 31) >> 5;
        if (lane_id == 0) warp_sums[warp_id] = prod;
        __syncthreads();
        if (warp_id == 0) {
            prod = (lane_id < n_warps) ? warp_sums[lane_id] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                prod += __shfl_xor_sync(0xffffffff, prod, off);
            if (lane_id == 0)
                scores[q_head * seq_len + pos] = prod * scale;
        }
        __syncthreads();
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

// ============ Chunked-prefill attention kernels ============
//
// Three batched kernels that process N query tokens against the same K/V
// cache slice in a single launch. The win comes from K/V reuse: each
// kv_head's K[pos] / V[pos] is loaded ONCE (or held in shared mem) and
// dotted against N*gqa query lanes, instead of the N separate kernel
// launches the per-token path needs.
//
// Memory layouts (token-major within the chunk):
//   q_chunk:    [N, num_q,  head_dim] half
//   scores:     [N, num_q,  seq_len]  float
//   out_chunk:  [N, num_q,  head_dim] half
//   k_cache:    [seq_len, num_kv, head_dim] half  (already populated through start_pos+N-1)
//   v_cache:    [seq_len, num_kv, head_dim] half
//
// Causal mask: query at chunk-relative t_idx has absolute position
//   abs_pos(t_idx) = start_pos + t_idx
// and may only attend to k positions p in [0, abs_pos(t_idx)]. Out-of-range
// positions get a -inf score so the subsequent softmax produces 0 weight.

// Bit-exact chunked score kernel: one block per score (q_head, t_idx, pos)
// with the SAME reduction tree as the per-token `attn_score_kernel_h`. This
// matters for greedy argmax stability — the warp-only reduce in the
// original `attn_score_kernel_h_chunk` flipped ~1% of borderline tokens
// (Korean coding prompts in particular) vs per-token runs. Used when
// VCHUNK_STRICT=1 (default when Korean bit-exact is required).
//
// Grid layout: (pos, q_head, t_idx).
//   pos in grid.x supports up to ~2^31 so 128K context fits.
//   num_q in grid.y (≤65535, num_q ≤ 128 realistically).
//   N=ATTN_NB=16 in grid.z.
// Block 256 threads — matches per-token blockDim for bit-identical summation.
__global__ void attn_score_kernel_h_chunk_strict(
    const half* __restrict__ q_chunk,    // [N * num_q * head_dim]
    const half* __restrict__ k_cache,    // [seq_len * num_kv * head_dim]
    float* __restrict__ scores,          // [N * num_q * row_stride]
    int num_q_heads, int num_kv_heads, int head_dim,
    int sub_seq_total, int start_pos, int N, float scale,
    int row_stride
) {
    int pos    = blockIdx.x;
    int q_head = blockIdx.y;
    int t_idx  = blockIdx.z;
    if (pos >= sub_seq_total || q_head >= num_q_heads || t_idx >= N) return;

    int abs_pos = start_pos + t_idx;
    size_t score_idx = (size_t)t_idx * num_q_heads * row_stride
                     + (size_t)q_head * row_stride + pos;

    if (pos > abs_pos) {
        if (threadIdx.x == 0) scores[score_idx] = -1e30f;
        return;
    }

    int kv_head = q_head / (num_q_heads / num_kv_heads);
    const half* qh = q_chunk
                   + (size_t)t_idx * num_q_heads * head_dim
                   + (size_t)q_head * head_dim;
    const half* kh = k_cache
                   + (size_t)pos * num_kv_heads * head_dim
                   + (size_t)kv_head * head_dim;

    // Reduction tree IDENTICAL to attn_score_kernel_h.
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
        if (lane_id == 0) scores[score_idx] = sum * scale;
    }
}

__global__ void attn_score_kernel_h_chunk(
    const half* __restrict__ q_chunk,    // [N * num_q * head_dim]
    const half* __restrict__ k_cache,    // [seq_len * num_kv * head_dim]
    float* __restrict__ scores,          // [N * num_q * row_stride]
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, int start_pos, int N, float scale,
    int row_stride
) {
    // v2: one warp per score. 8 warps per block → 8 scores compute in parallel
    // instead of the prior "256 threads reduce a single score at a time" layout
    // which paid a __syncthreads per score (N*gqa=96 barriers per block for
    // the 27B). Per warp: 32 threads each own head_dim/32 input dims and
    // reduce via __shfl_xor_sync — no cross-warp reduction needed because
    // each score is owned by exactly one warp. Expected 5-8× speedup on the
    // score phase at 2071 tok.
    int kv_head = blockIdx.x;
    int pos     = blockIdx.y;
    if (kv_head >= num_kv_heads || pos >= seq_len) return;

    int gqa = num_q_heads / num_kv_heads;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    int n_warps = blockDim.x >> 5;

    // Shared mem cache of K[pos, kv_head, :].
    extern __shared__ float smem[];
    float* k_smem = smem;  // [head_dim]

    if (tid < head_dim) {
        k_smem[tid] = __half2float(
            k_cache[(size_t)pos * num_kv_heads * head_dim
                    + kv_head * head_dim + tid]);
    }
    __syncthreads();

    // Each warp handles one (t_idx, qi) score per iteration; block strides
    // through N*gqa scores in n_warps-sized chunks.
    int total_scores = N * gqa;
    for (int s = wid; s < total_scores; s += n_warps) {
        int t_idx = s / gqa;
        int qi    = s - t_idx * gqa;
        int q_head = kv_head * gqa + qi;
        int abs_pos = start_pos + t_idx;
        bool causal_ok = (pos <= abs_pos);

        size_t score_idx = (size_t)t_idx * num_q_heads * row_stride
                         + (size_t)q_head * row_stride + pos;

        if (!causal_ok) {
            if (lid == 0) scores[score_idx] = -1e30f;
            continue;
        }

        const half* qh = q_chunk + (size_t)t_idx * num_q_heads * head_dim
                                 + (size_t)q_head * head_dim;
        float prod = 0.0f;
        // Each thread walks head_dim/32 input dims.
        for (int i = lid; i < head_dim; i += 32) {
            prod += __half2float(qh[i]) * k_smem[i];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            prod += __shfl_xor_sync(0xffffffff, prod, off);
        if (lid == 0) scores[score_idx] = prod * scale;
    }
}

// Per (token, q_head) softmax with implicit causal range. Each block
// normalises one (t_idx, q_head) row across positions [0, abs_pos]; positions
// beyond that are stuffed with -inf in the score kernel and contribute 0
// weight after the exp.
//
// `seq_len` is the per-row STRIDE between consecutive (t_idx, q_head) rows
// in `scores` (so it can be the full kv_max_seq when the buffer is sized
// for max context). `wipe_end` bounds the explicit zero-fill that runs
// after normalisation: positions in [active_len, wipe_end) are written to
// 0 so the downstream value kernel can read them safely. The value kernel
// only walks up to (start_pos + N - 1), so wipe_end = start_pos + N is
// enough — anything beyond that stays stale and is never read.
__global__ void softmax_kernel_chunk(
    float* __restrict__ scores,   // [N, num_q, seq_len]
    int num_q_heads, int seq_len, int start_pos, int wipe_end
) {
    int t_idx  = blockIdx.x;       // chunk-relative token
    int q_head = blockIdx.y;
    int abs_pos = start_pos + t_idx;
    int active_len = abs_pos + 1;  // valid positions [0, abs_pos]

    float* row = scores + (size_t)t_idx * gridDim.y * seq_len
                        + (size_t)q_head * seq_len;

    extern __shared__ float sdata[];

    // 1. Find max over active positions only.
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < active_len; i += blockDim.x)
        max_val = fmaxf(max_val, row[i]);
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // 2. Exp + sum (active range only). Masked positions stay 0.
    float sum = 0.0f;
    for (int i = threadIdx.x; i < active_len; i += blockDim.x) {
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
    __syncthreads();

    // 3. Normalise. Positions in [active_len, wipe_end) get an explicit
    //    zero so the value kernel can read them safely; everything else
    //    stays stale (and is never read because the value kernel walks
    //    only up to wipe_end-1).
    float inv = 1.0f / sum;
    for (int i = threadIdx.x; i < active_len; i += blockDim.x)
        row[i] *= inv;
    for (int i = threadIdx.x + active_len; i < wipe_end; i += blockDim.x)
        row[i] = 0.0f;
}

// scores @ V → output for the whole chunk. Same K/V reuse trick as the
// score kernel, but on V[pos, kv_head, d]. Block layout = (kv_head, d_tile)
// where d_tile is one head_dim element produced for all gqa × N queries
// in that kv_head group. Inner loop walks pos with one V load shared
// across all gqa × N accumulators (kept in registers).
__global__ void attn_value_kernel_h_chunk(
    const float* __restrict__ scores,    // [N * num_q * row_stride]
    const half*  __restrict__ v_cache,   // [seq_len * num_kv * head_dim]
    half* __restrict__ out_chunk,        // [N * num_q * head_dim]
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, int start_pos, int N,
    int row_stride
) {
    // v2: grid parallelised across (kv_head, t_idx). The prior layout was
    // (kv_head, d_block) = num_kv × d_blocks = 8 blocks for 27B, leaving
    // ~87% of SMs idle. Per-thread accumulator shrank from [N][gqa] (96
    // floats) to [gqa] (6 floats) so register pressure drops too.
    //
    // Grid: (num_kv × N, d_blocks). blockIdx.x encodes kv_head × N + t_idx.
    int bx = blockIdx.x;
    int t_idx   = bx % N;
    int kv_head = bx / N;
    int d       = blockIdx.y * blockDim.x + threadIdx.x;
    if (kv_head >= num_kv_heads || d >= head_dim) return;

    int gqa = num_q_heads / num_kv_heads;
    int abs_pos = start_pos + t_idx;
    int active_end = abs_pos + 1;  // causal: [0, abs_pos]
    if (active_end > seq_len) active_end = seq_len;

    // Per-thread accumulators for this (t_idx, kv_head) block.
    float acc[8];
    #pragma unroll
    for (int qi = 0; qi < 8; qi++) acc[qi] = 0.0f;

    const float* score_base = scores + (size_t)t_idx * num_q_heads * row_stride
                                     + (size_t)kv_head * gqa * row_stride;

    for (int pos = 0; pos < active_end; pos++) {
        float v = __half2float(
            v_cache[(size_t)pos * num_kv_heads * head_dim
                    + kv_head * head_dim + d]);
        #pragma unroll
        for (int qi = 0; qi < 8; qi++) {
            if (qi >= gqa) break;
            acc[qi] += score_base[(size_t)qi * row_stride + pos] * v;
        }
    }

    half* out_base = out_chunk + (size_t)t_idx * num_q_heads * head_dim
                               + (size_t)kv_head * gqa * head_dim;
    #pragma unroll
    for (int qi = 0; qi < 8; qi++) {
        if (qi >= gqa) break;
        out_base[(size_t)qi * head_dim + d] = __float2half(acc[qi]);
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

// ============ TurboQuant 3-bit fused attention value kernel ============
// Each cuda block handles ONE (kv_head, d_block) and computes the partial
// V output for all `gqa_ratio` query heads that share this kv_head. For
// each position we read ONE block_tq3 (52 B), dequant cooperatively into
// shared memory, and accumulate into a small per-thread per-q_head
// register array. The dequant cost is therefore amortized over
// gqa_ratio * head_dim work per position, instead of paying it once per
// (q_head, d) pair as the dequant-first approach did.
//
// Launch: <<< dim3(num_kv_heads, blocks_per_kv_head), TQ_BLOCK_SIZE,
//             TQ_BLOCK_SIZE * sizeof(float) >>>
__global__ void attn_value_kernel_tq3(
    const float* __restrict__ scores,        // [num_q_heads * seq_len]
    const block_tq3* __restrict__ v_cache,   // [seq_len * blocks_per_token]
    half* __restrict__ output,               // [num_q_heads * head_dim]
    int num_q_heads, int num_kv_heads, int head_dim, int seq_len
) {
    int kv_head = blockIdx.x;
    int d_block = blockIdx.y;
    int gqa     = num_q_heads / num_kv_heads;
    int blocks_per_token   = num_kv_heads * head_dim / TQ_BLOCK_SIZE;
    int blocks_per_kv_head = head_dim / TQ_BLOCK_SIZE;

    int lane = threadIdx.x;  // 0..127

    // Per-q_head accumulator. Hardcoded max 8 to avoid runtime alloc; the
    // actual number used is `gqa` (=6 for Qwen3.5-27B).
    float acc[8];
    #pragma unroll
    for (int q = 0; q < 8; q++) acc[q] = 0.0f;

    extern __shared__ float v_smem[];  // [TQ_BLOCK_SIZE]

    for (int pos = 0; pos < seq_len; pos++) {
        const block_tq3* blk = &v_cache[(size_t)pos * blocks_per_token
                                        + kv_head * blocks_per_kv_head + d_block];

        // Cooperative dequant of one block into v_smem.
        int byte_off = (lane * 3) >> 3;
        int bit_off  = (lane * 3) & 7;
        uint32_t packed = (uint32_t)blk->qs[byte_off];
        if (byte_off + 1 < 48)
            packed |= ((uint32_t)blk->qs[byte_off + 1]) << 8;
        int idx = (packed >> bit_off) & 0x7;
        v_smem[lane] = d_tq3_centroids[idx];
        __syncthreads();

        for (int s = 1; s < TQ_BLOCK_SIZE; s <<= 1) {
            float a = v_smem[lane];
            float b = v_smem[lane ^ s];
            __syncthreads();
            if ((lane & s) == 0) v_smem[lane] = a + b;
            else                 v_smem[lane] = b - a;
            __syncthreads();
        }
        // Inverse RHT (encoder applies signs before WHT; decoder closes
        // it out after iWHT to match the matrix transpose order).
        float scale_norm = rsqrtf((float)TQ_BLOCK_SIZE) * blk->norm;
        float v_val = v_smem[lane] * d_tq3_signs[lane] * scale_norm;
        __syncthreads();

        // Multiply this V slot into all gqa accumulators.
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            if (q >= gqa) break;
            int q_head = kv_head * gqa + q;
            float sc = scores[q_head * seq_len + pos];
            acc[q] += sc * v_val;
        }
    }

    // Write outputs: this thread owns d = d_block * 128 + lane for every
    // q_head in the GQA group.
    int d = d_block * TQ_BLOCK_SIZE + lane;
    #pragma unroll
    for (int q = 0; q < 8; q++) {
        if (q >= gqa) break;
        int q_head = kv_head * gqa + q;
        output[q_head * head_dim + d] = __float2half(acc[q]);
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

// Batched (N-token) variants of the per-token attention prologue kernels.
// These collapse what used to be N separate kernel launches per layer into
// a single launch, using blockIdx.y (or a token loop inside the block) to
// stride across tokens. For 27B at chunk=128 and 16 attn layers per chunk
// this removes ~128× kernel launches per layer on the deinterleave / head
// RMS / RoPE / gate phases (~175K total launches across a 2071-tok prefill
// collapse to ~1.4K).

__global__ void deinterleave_qg_kernel_chunk(
    const half* __restrict__ src,   // [n_tokens, num_heads * head_dim * 2]
    half* __restrict__ q_out,       // [n_tokens, num_heads * head_dim]
    half* __restrict__ gate_out,    // [n_tokens, num_heads * head_dim]
    int num_heads, int head_dim, int n_tokens
) {
    int tt = blockIdx.y;
    if (tt >= n_tokens) return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int per_tok = num_heads * head_dim;
    if (idx >= per_tok) return;
    int head = idx / head_dim;
    int d = idx % head_dim;
    int src_stride = per_tok * 2;
    q_out[tt * per_tok + idx]    = src[tt * src_stride + head * head_dim * 2 + d];
    gate_out[tt * per_tok + idx] = src[tt * src_stride + head * head_dim * 2 + head_dim + d];
}

__global__ void head_rms_norm_kernel_chunk(
    half* __restrict__ x,
    const float* __restrict__ weight,
    int num_heads, int head_dim, float eps, int n_tokens
) {
    int tt = blockIdx.y;
    int head = blockIdx.x;
    if (tt >= n_tokens || head >= num_heads) return;
    half* xh = x + (size_t)tt * num_heads * head_dim + head * head_dim;

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
    float inv = rsqrtf(sdata[0] / head_dim + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        xh[i] = __float2half(__half2float(xh[i]) * inv * weight[i]);
    }
}

// Batched RoPE: each (tt, head, dim-pair) gets its own sin/cos from the
// full table (offset by start_pos + tt). Grid.y strides tokens.
__global__ void apply_rope_kernel_chunk(
    half* __restrict__ x,
    const float* __restrict__ sin_table,  // [max_seq, rope_dim/2]
    const float* __restrict__ cos_table,
    int start_pos, int num_heads, int head_dim, int rope_dim,
    int n_tokens
) {
    int tt = blockIdx.y;
    if (tt >= n_tokens) return;
    int pos = start_pos + tt;
    int half_rope = rope_dim / 2;
    const float* sin_t = sin_table + pos * half_rope;
    const float* cos_t = cos_table + pos * half_rope;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * half_rope;
    if (idx >= total) return;

    int head = idx / half_rope;
    int i    = idx - head * half_rope;
    half* xh = x + (size_t)tt * num_heads * head_dim + head * head_dim;

    float x0 = __half2float(xh[i]);
    float x1 = __half2float(xh[i + half_rope]);
    float c = cos_t[i], s = sin_t[i];
    xh[i]             = __float2half(x0 * c - x1 * s);
    xh[i + half_rope] = __float2half(x0 * s + x1 * c);
}

__global__ void apply_gate_sigmoid_chunk(
    half* __restrict__ output,  // [n_tokens, size]
    const half* __restrict__ gate,
    int size, int n_tokens
) {
    int tt = blockIdx.y;
    if (tt >= n_tokens) return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    size_t off = (size_t)tt * size + idx;
    float o = __half2float(output[off]);
    float g = __half2float(gate[off]);
    float sig = 1.0f / (1.0f + expf(-g));
    output[off] = __float2half(o * sig);
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

// ============ FlashAttention fused score + softmax + value ============
// One block per (kv_head, t_idx) within the current sub-chunk. All GQA
// query heads sharing the kv_head are processed in a single block so K/V
// tile loads are amortised across them. Online softmax keeps O/m/l in
// registers and never materialises the [num_q, seq_len] score tensor.
//
// Layout expectations:
//   q_chunk:   [sub_n, num_q, HD]  half
//   k_cache:   [kv_max_seq, num_kv, HD] half  (causal populated up to active_end)
//   v_cache:   [kv_max_seq, num_kv, HD] half
//   out_chunk: [sub_n, num_q, HD]  half
//
// Template params specialise for the Qwen3.5-27B shape HD=256, GQA=6,
// BM=32, BLOCK=256. Causal mask uses active_end = start_pos + t_idx + 1.
//
// SMEM budget (HD=256, BM=32, GQA=6):
//   q_s    = GQA*HD*2        = 3072  B
//   k_tile = BM*HD*2         = 16384 B
//   v_tile = BM*HD*2         = 16384 B
//   s_smem = GQA*BM*4        = 768   B
//   total ≈ 36.6 KB  (≤ 48 KB default, no dynamic bump needed)
template<int HD, int GQA, int BM, int BLOCK>
__global__ void flash_attn_chunk_fused(
    const half* __restrict__ q_chunk,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    half*       __restrict__ out_chunk,
    int num_q, int num_kv,
    int start_pos, int sub_n, float scale
) {
    constexpr int N_WARPS = BLOCK / 32;
    constexpr int LANE_D  = HD / 32;  // dims owned per lane (8)
    int kv_head = blockIdx.x;
    int t_idx   = blockIdx.y;
    if (t_idx >= sub_n) return;
    int abs_pos    = start_pos + t_idx;
    int active_end = abs_pos + 1;         // causal: [0, abs_pos]
    int tid  = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    extern __shared__ unsigned char smem_fa[];
    half*  q_s    = (half*)smem_fa;
    half*  k_tile = q_s    + GQA * HD;
    half*  v_tile = k_tile + BM  * HD;
    float* s_smem = (float*)(v_tile + BM * HD);

    // Load Q: gqa query heads for this (t_idx, kv_head).
    #pragma unroll
    for (int i = tid; i < GQA * HD; i += BLOCK) {
        int g = i / HD;
        int c = i - g * HD;
        int q_head = kv_head * GQA + g;
        q_s[g * HD + c] = q_chunk[(size_t)t_idx * num_q * HD
                                  + (size_t)q_head * HD + c];
    }
    __syncthreads();

    // Per-warp (== per-q_head for warp<GQA) register-resident accumulators.
    float acc_o[LANE_D];
    #pragma unroll
    for (int i = 0; i < LANE_D; i++) acc_o[i] = 0.0f;
    float m_w = -INFINITY;
    float l_w = 0.0f;

    for (int tile_start = 0; tile_start < active_end; tile_start += BM) {
        int tile_end = min(tile_start + BM, active_end);
        int tile_len = tile_end - tile_start;

        // Load K, V tile. Invalid rows get zero-filled K so the score is 0,
        // which we override with -inf before softmax; V zero keeps P·V safe.
        #pragma unroll
        for (int i = tid; i < BM * HD; i += BLOCK) {
            int r = i / HD;
            int c = i - r * HD;
            half zero_h = __float2half(0.0f);
            if (r < tile_len) {
                size_t base = (size_t)(tile_start + r) * num_kv * HD
                            + kv_head * HD;
                k_tile[r * HD + c] = k_cache[base + c];
                v_tile[r * HD + c] = v_cache[base + c];
            } else {
                k_tile[r * HD + c] = zero_h;
                v_tile[r * HD + c] = zero_h;
            }
        }
        __syncthreads();

        // Score compute: warp-per-score. NITERS × N_WARPS == GQA × BM (192).
        constexpr int NITERS = (GQA * BM + N_WARPS - 1) / N_WARPS;
        #pragma unroll
        for (int k_it = 0; k_it < NITERS; k_it++) {
            int flat = warp + k_it * N_WARPS;
            if (flat >= GQA * BM) break;
            int g = flat / BM;
            int r = flat - g * BM;

            float partial = 0.0f;
            const half2* qs2 = reinterpret_cast<const half2*>(q_s    + g * HD);
            const half2* kt2 = reinterpret_cast<const half2*>(k_tile + r * HD);
            #pragma unroll
            for (int i = 0; i < LANE_D / 2; i++) {
                half2 qv = qs2[lane * (LANE_D / 2) + i];
                half2 kv = kt2[lane * (LANE_D / 2) + i];
                float2 qf = __half22float2(qv);
                float2 kf = __half22float2(kv);
                partial += qf.x * kf.x + qf.y * kf.y;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_xor_sync(0xffffffff, partial, off);
            if (lane == 0) {
                float val = (r < tile_len) ? partial * scale : -INFINITY;
                s_smem[g * BM + r] = val;
            }
        }
        __syncthreads();

        // Softmax update + O accumulation (warps 0..GQA-1).
        if (warp < GQA) {
            int g = warp;
            float s_val = s_smem[g * BM + lane];  // BM == 32 == warp width

            float m_row = s_val;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                m_row = fmaxf(m_row, __shfl_xor_sync(0xffffffff, m_row, off));
            float m_new = fmaxf(m_w, m_row);
            float correction = expf(m_w - m_new);
            float p_lane    = expf(s_val - m_new);  // r>=tile_len → 0
            float sum_p = p_lane;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                sum_p += __shfl_xor_sync(0xffffffff, sum_p, off);

            #pragma unroll
            for (int i = 0; i < LANE_D; i++) acc_o[i] *= correction;
            l_w = l_w * correction + sum_p;

            // acc_o[i] += sum_r P[r] * V_tile[r, lane*LANE_D + i]
            #pragma unroll
            for (int r = 0; r < BM; r++) {
                float p_r = __shfl_sync(0xffffffff, p_lane, r);
                const half2* vt = reinterpret_cast<const half2*>(
                    v_tile + r * HD + lane * LANE_D);
                #pragma unroll
                for (int i = 0; i < LANE_D / 2; i++) {
                    half2 vv = vt[i];
                    float2 vf = __half22float2(vv);
                    acc_o[i * 2]     += p_r * vf.x;
                    acc_o[i * 2 + 1] += p_r * vf.y;
                }
            }
            m_w = m_new;
        }
        __syncthreads();
    }

    // Write normalised O back.
    if (warp < GQA) {
        int g = warp;
        int q_head = kv_head * GQA + g;
        float inv_l = (l_w > 0.0f) ? (1.0f / l_w) : 0.0f;
        size_t out_base = (size_t)t_idx * num_q * HD + (size_t)q_head * HD;
        #pragma unroll
        for (int i = 0; i < LANE_D; i++) {
            int c = lane * LANE_D + i;
            out_chunk[out_base + c] = __float2half(acc_o[i] * inv_l);
        }
    }
}
