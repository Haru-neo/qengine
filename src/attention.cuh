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
    //    only up to wipe_end-1). The +1e-10 matches per-token softmax_kernel
    //    so chunked vs per-token outputs round identically.
    float inv = 1.0f / (sum + 1e-10f);
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
// Per-slot RoPE for the batched gen-step path. Each row of `x` belongs to
// a different slot at its own logical position; positions[tt] supplies the
// pos used for this slot's RoPE table lookup.
__global__ void apply_rope_kernel_batched(
    half* __restrict__ x,
    const float* __restrict__ sin_table,
    const float* __restrict__ cos_table,
    const int* __restrict__ positions,  // [n_tokens] per-slot pos
    int num_heads, int head_dim, int rope_dim,
    int n_tokens
) {
    int tt = blockIdx.y;
    if (tt >= n_tokens) return;
    int pos = positions[tt];
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

// Scatter-write of K/V for a batched gen step. n_tokens rows of K and V come
// from the batched projection; each row is destined for slot s_i at logical
// position p_i — i.e. physical KV index `dst_kv_pos[tt] = s_i * slot_max_seq + p_i`.
__global__ void scatter_kv_kernel(
    const half* __restrict__ k_in,   // [n_tokens, kv_dim]
    const half* __restrict__ v_in,   // [n_tokens, kv_dim]
    half* __restrict__       k_cache, // physical [num_slots * slot_max_seq, kv_dim]
    half* __restrict__       v_cache,
    const int* __restrict__  dst_kv_pos,  // [n_tokens]
    int kv_dim, int n_tokens
) {
    int tt = blockIdx.y;
    if (tt >= n_tokens) return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kv_dim) return;
    size_t dst = (size_t)dst_kv_pos[tt] * kv_dim + idx;
    k_cache[dst] = k_in[(size_t)tt * kv_dim + idx];
    v_cache[dst] = v_in[(size_t)tt * kv_dim + idx];
}

// M-RoPE for the LLM: per-token (pos_t, pos_h, pos_w) drawn from separate
// arrays, with sections={sec_t, sec_h, sec_w, sec_extra} (sec_extra ignored
// here — Qwen3-VL fills it with zeros). Each rotary pair k in [0..rope_dim/2)
// picks its position from one of the three axis arrays based on the section
// boundary it falls into. For text-only callers all three arrays carry the
// same value, recovering standard 1D RoPE bit-for-bit.
__global__ void apply_rope_kernel_mrope_chunk(
    half* __restrict__ x,
    const float* __restrict__ sin_table,
    const float* __restrict__ cos_table,
    const int* __restrict__ pos_t,
    const int* __restrict__ pos_h,
    const int* __restrict__ pos_w,
    int sec_t, int sec_h, int sec_w,
    int num_heads, int head_dim, int rope_dim,
    int n_tokens
) {
    int tt = blockIdx.y;
    if (tt >= n_tokens) return;
    int half_rope = rope_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * half_rope;
    if (idx >= total) return;

    int head = idx / half_rope;
    int i    = idx - head * half_rope;       // pair index in [0..half_rope)
    // Qwen3.5 / Qwen3-VL: ggml_rope_multi(GGML_ROPE_TYPE_IMROPE) → interleaved
    // M-RoPE — pairs cycle through T,H,W,... in mod-3 buckets, capped at
    // 3*sec_axis. Matches transformers' apply_interleaved_mrope.
    // VL_MROPE_CHUNKED=1 reverts to the simpler chunked layout for ablation.
    int pos;
    if (false) {  // VL_MROPE_CHUNKED removed branch — toggle via host #define
        if (i < sec_t)            pos = pos_t[tt];
        else if (i < sec_t+sec_h) pos = pos_h[tt];
        else                       pos = pos_w[tt];
    } else {
        int rem = i % 3;
        if (rem == 0 && i < 3 * sec_t)      pos = pos_t[tt];
        else if (rem == 1 && i < 3 * sec_h) pos = pos_h[tt];
        else if (rem == 2 && i < 3 * sec_w) pos = pos_w[tt];
        else                                 pos = 0;
    }

    const float* sin_t = sin_table + (size_t)pos * half_rope;
    const float* cos_t = cos_table + (size_t)pos * half_rope;
    half* xh = x + (size_t)tt * num_heads * head_dim + head * head_dim;
    float x0 = __half2float(xh[i]);
    float x1 = __half2float(xh[i + half_rope]);
    float c = cos_t[i], s = sin_t[i];
    xh[i]             = __float2half(x0 * c - x1 * s);
    xh[i + half_rope] = __float2half(x0 * s + x1 * c);
}

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

// ============ Split-K FA: parallelize the K/V tile loop across blocks ============
// At long context the base flash_attn_chunk_fused has only num_kv*sub_n = 64
// blocks (≪ 3·68 = 204 SMs across 3 GPUs) and each block runs a sequential
// loop of ⌈active_end/BM⌉ tile iterations. Split-K factors that loop into
// K_SPLITS independent blocks per (kv_head, t_idx), each handling a contiguous
// tile range. Each block emits an unnormalised (m, l, acc_o) triplet; a tiny
// merge kernel combines them with the standard log-sum-exp identity:
//
//   m_global = max_s m_s
//   l_global = Σ_s exp(m_s − m_global) · l_s
//   o_global = Σ_s exp(m_s − m_global) · acc_o_s   (then divide by l_global)
//
// Splits whose range lies past a token's causal end produce m=-inf,l=0,o=0 and
// drop out cleanly via the exp(-inf - m_global) = 0 identity, so we don't need
// per-(t_idx,split) skip logic at the launcher.
//
// SMEM identical to flash_attn_chunk_fused (same Q/K/V/score scratch).
//
// part_m, part_l: [num_q, sub_n_max, K_SPLITS]   float  (lane==0 writes)
// part_o:         [num_q, sub_n_max, K_SPLITS, HD]   float
//
// part_o is float (not half) so the merge step reads exactly the same fp32
// accumulator the compute kernel produced — no fp16 store/reload truncation.
// That keeps split-K within the fp32-reordering noise floor (≈1e-7 per add)
// of the base flash_attn_chunk_fused kernel: greedy argmax matches across
// long generations, so Korean quality stays intact (project_qwopus_lang_bias).
template<int HD, int GQA, int BM, int BLOCK, int K_SPLITS>
__global__ void flash_attn_chunk_fused_split(
    const half* __restrict__ q_chunk,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    float*      __restrict__ part_m,
    float*      __restrict__ part_l,
    float*      __restrict__ part_o,
    int num_q, int num_kv,
    int start_pos, int sub_n, int sub_n_max,
    int active_end_max, float scale
) {
    constexpr int N_WARPS = BLOCK / 32;
    constexpr int LANE_D  = HD / 32;
    int kv_head    = blockIdx.x;
    int t_idx      = blockIdx.y;
    int split_idx  = blockIdx.z;
    if (t_idx >= sub_n) return;
    int abs_pos       = start_pos + t_idx;
    int active_end_t  = abs_pos + 1;
    int tid  = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    int total_tiles     = (active_end_max + BM - 1) / BM;
    int tiles_per_split = (total_tiles + K_SPLITS - 1) / K_SPLITS;
    int tile_lo         = split_idx * tiles_per_split * BM;
    int tile_hi         = min((split_idx + 1) * tiles_per_split * BM, active_end_t);

    extern __shared__ unsigned char smem_fa_split[];
    half*  q_s    = (half*)smem_fa_split;
    half*  k_tile = q_s    + GQA * HD;
    half*  v_tile = k_tile + BM  * HD;
    float* s_smem = (float*)(v_tile + BM * HD);

    #pragma unroll
    for (int i = tid; i < GQA * HD; i += BLOCK) {
        int g = i / HD;
        int c = i - g * HD;
        int q_head = kv_head * GQA + g;
        q_s[g * HD + c] = q_chunk[(size_t)t_idx * num_q * HD
                                  + (size_t)q_head * HD + c];
    }
    __syncthreads();

    float acc_o[LANE_D];
    #pragma unroll
    for (int i = 0; i < LANE_D; i++) acc_o[i] = 0.0f;
    float m_w = -INFINITY;
    float l_w = 0.0f;

    if (tile_lo < tile_hi) {
        for (int tile_start = tile_lo; tile_start < tile_hi; tile_start += BM) {
            int tile_end = min(tile_start + BM, tile_hi);
            int tile_len = tile_end - tile_start;

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

            if (warp < GQA) {
                int g = warp;
                float s_val = s_smem[g * BM + lane];

                float m_row = s_val;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    m_row = fmaxf(m_row, __shfl_xor_sync(0xffffffff, m_row, off));
                float m_new = fmaxf(m_w, m_row);
                float correction = expf(m_w - m_new);
                float p_lane    = expf(s_val - m_new);
                float sum_p = p_lane;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    sum_p += __shfl_xor_sync(0xffffffff, sum_p, off);

                #pragma unroll
                for (int i = 0; i < LANE_D; i++) acc_o[i] *= correction;
                l_w = l_w * correction + sum_p;

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
    }

    if (warp < GQA) {
        int g = warp;
        int q_head = kv_head * GQA + g;
        size_t ml_idx = ((size_t)q_head * sub_n_max + t_idx) * K_SPLITS + split_idx;
        if (lane == 0) {
            part_m[ml_idx] = m_w;
            part_l[ml_idx] = l_w;
        }
        size_t o_base = (((size_t)q_head * sub_n_max + t_idx) * K_SPLITS + split_idx) * HD;
        #pragma unroll
        for (int i = 0; i < LANE_D; i++) {
            int c = lane * LANE_D + i;
            part_o[o_base + c] = acc_o[i];
        }
    }
}

// Merge K_SPLITS partials into final O. One warp per (kv_head, q_head_within_kv,
// t_idx) — block has GQA warps. Each lane reads its LANE_D slice of each split's
// part_o, applies the exp(m_s - m_global) weight, and writes the renormalised
// fp16 output to out_chunk[t_idx, q_head, :].
template<int HD, int GQA, int BLOCK, int K_SPLITS>
__global__ void flash_attn_split_merge(
    const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    const float* __restrict__ part_o,
    half*        __restrict__ out_chunk,
    int num_q, int sub_n, int sub_n_max
) {
    int kv_head = blockIdx.x;
    int t_idx   = blockIdx.y;
    if (t_idx >= sub_n) return;
    int tid  = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    constexpr int LANE_D = HD / 32;
    if (warp >= GQA) return;
    int g = warp;
    int q_head = kv_head * GQA + g;

    float m_global = -INFINITY;
    #pragma unroll
    for (int s = 0; s < K_SPLITS; s++) {
        size_t ml_idx = ((size_t)q_head * sub_n_max + t_idx) * K_SPLITS + s;
        m_global = fmaxf(m_global, part_m[ml_idx]);
    }

    float l_global = 0.0f;
    float o_acc[LANE_D];
    #pragma unroll
    for (int i = 0; i < LANE_D; i++) o_acc[i] = 0.0f;

    #pragma unroll
    for (int s = 0; s < K_SPLITS; s++) {
        size_t ml_idx = ((size_t)q_head * sub_n_max + t_idx) * K_SPLITS + s;
        float ms = part_m[ml_idx];
        float ls = part_l[ml_idx];
        if (!isfinite(ms) || ls <= 0.0f) continue;
        float w = expf(ms - m_global);
        l_global += w * ls;
        size_t o_base = (((size_t)q_head * sub_n_max + t_idx) * K_SPLITS + s) * HD;
        const float* op = part_o + o_base + lane * LANE_D;
        #pragma unroll
        for (int i = 0; i < LANE_D; i++) {
            o_acc[i] += w * op[i];
        }
    }

    float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
    size_t out_base = (size_t)t_idx * num_q * HD + (size_t)q_head * HD;
    #pragma unroll
    for (int i = 0; i < LANE_D; i++) {
        int c = lane * LANE_D + i;
        out_chunk[out_base + c] = __float2half(o_acc[i] * inv_l);
    }
}

// ============ Split-K FA + cooperative TQ3 decode ============
// Same algorithm as flash_attn_chunk_fused_split, but K/V tiles are read
// directly from the TurboQuant 3-bit cache and decoded cooperatively in
// shared memory: one warp per 128-element block, 4 elements per thread,
// inverse-WHT via __shfl_xor (within-warp stages 3-7) + within-thread
// butterflies (stages 1-2). This collapses the legacy "bulk dequant the
// entire K/V history into an fp16 scratch buffer per layer per token"
// into a per-tile decode that lives entirely in shared memory — at long
// context this saves O(seq·hd·num_kv) of fp16 HBM traffic per token.
//
// HD must be a multiple of TQ_BLOCK_SIZE (= 128).  For Qwen3.5/3.6 hybrid
// HD=256 → 2 TQ blocks per (row, kv_head) for K and same for V.
template<int HD, int GQA, int BM, int BLOCK, int K_SPLITS>
__global__ void flash_attn_chunk_fused_split_tq3(
    const half*       __restrict__ q_chunk,
    const block_tq3*  __restrict__ k_tq,
    const block_tq3*  __restrict__ v_tq,
    float*            __restrict__ part_m,
    float*            __restrict__ part_l,
    float*            __restrict__ part_o,
    int num_q, int num_kv,
    int start_pos, int sub_n, int sub_n_max,
    int active_end_max, float scale
) {
    static_assert(HD % TQ_BLOCK_SIZE == 0, "HD must be multiple of 128");
    constexpr int N_WARPS         = BLOCK / 32;
    constexpr int LANE_D          = HD / 32;
    constexpr int BLOCKS_PER_KV   = HD / TQ_BLOCK_SIZE;     // 2 for HD=256
    constexpr int K_DEC_BLOCKS    = BM * BLOCKS_PER_KV;     // 64 for BM=32
    constexpr int TOTAL_DEC_BLOCKS = 2 * K_DEC_BLOCKS;       // K + V
    int kv_head    = blockIdx.x;
    int t_idx      = blockIdx.y;
    int split_idx  = blockIdx.z;
    if (t_idx >= sub_n) return;
    int abs_pos       = start_pos + t_idx;
    int active_end_t  = abs_pos + 1;
    int tid  = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    int total_tiles     = (active_end_max + BM - 1) / BM;
    int tiles_per_split = (total_tiles + K_SPLITS - 1) / K_SPLITS;
    int tile_lo         = split_idx * tiles_per_split * BM;
    int tile_hi         = min((split_idx + 1) * tiles_per_split * BM, active_end_t);

    extern __shared__ unsigned char smem_fa_split_tq[];
    half*  q_s    = (half*)smem_fa_split_tq;
    half*  k_tile = q_s    + GQA * HD;
    half*  v_tile = k_tile + BM  * HD;
    float* s_smem = (float*)(v_tile + BM * HD);

    // Load Q (same as base).
    #pragma unroll
    for (int i = tid; i < GQA * HD; i += BLOCK) {
        int g = i / HD;
        int c = i - g * HD;
        int q_head = kv_head * GQA + g;
        q_s[g * HD + c] = q_chunk[(size_t)t_idx * num_q * HD
                                  + (size_t)q_head * HD + c];
    }
    __syncthreads();

    float acc_o[LANE_D];
    #pragma unroll
    for (int i = 0; i < LANE_D; i++) acc_o[i] = 0.0f;
    float m_w = -INFINITY;
    float l_w = 0.0f;

    if (tile_lo < tile_hi) {
        const float wht_scale = rsqrtf((float)TQ_BLOCK_SIZE);
        for (int tile_start = tile_lo; tile_start < tile_hi; tile_start += BM) {
            int tile_end = min(tile_start + BM, tile_hi);
            int tile_len = tile_end - tile_start;

            // ============ Cooperative TQ3 decode of K + V tile ============
            // 1 warp per 128-elem TQ block, 4 elements per thread.
            for (int blk_iter = warp; blk_iter < TOTAL_DEC_BLOCKS; blk_iter += N_WARPS) {
                bool is_v = (blk_iter >= K_DEC_BLOCKS);
                int b      = is_v ? (blk_iter - K_DEC_BLOCKS) : blk_iter;
                int row    = b / BLOCKS_PER_KV;
                int col_block = b - row * BLOCKS_PER_KV;
                int abs_row = tile_start + row;
                half* tile_dst = (is_v ? v_tile : k_tile)
                               + row * HD + col_block * TQ_BLOCK_SIZE;
                if (abs_row >= tile_end) {
                    // Out-of-range row → zero (causal mask handled by score path).
                    #pragma unroll
                    for (int i = 0; i < 4; i++)
                        tile_dst[lane * 4 + i] = __float2half(0.0f);
                    continue;
                }
                size_t blk_idx = ((size_t)abs_row * num_kv + kv_head) * BLOCKS_PER_KV
                               + col_block;
                const block_tq3* blk = (is_v ? v_tq : k_tq) + blk_idx;

                // Per-thread: 4 contiguous elements (chunked layout).
                int base_elem = lane * 4;
                float v[4];

                // 1. Unpack 3-bit indices → centroid lookup.
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int elem        = base_elem + i;
                    int bit_off     = elem * 3;
                    int byte_off    = bit_off >> 3;
                    int bit_in_byte = bit_off & 7;
                    uint16_t two = (uint16_t)blk->qs[byte_off]
                                 | ((uint16_t)blk->qs[byte_off + 1] << 8);
                    uint8_t idx = (two >> bit_in_byte) & 0x7;
                    v[i] = d_tq3_centroids[idx];
                }

                // 2. Inverse WHT — stages 1-2 within thread, 3-7 cross-thread.
                { float a=v[0], b=v[1]; v[0]=a+b; v[1]=a-b; }
                { float a=v[2], b=v[3]; v[2]=a+b; v[3]=a-b; }
                { float a=v[0], b=v[2]; v[0]=a+b; v[2]=a-b; }
                { float a=v[1], b=v[3]; v[1]=a+b; v[3]=a-b; }
                #pragma unroll
                for (int step = 4; step <= 64; step <<= 1) {
                    int xor_off = step >> 2;
                    int upper   = (lane & xor_off) ? 1 : 0;
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        float other = __shfl_xor_sync(0xffffffff, v[i], xor_off);
                        v[i] = upper ? (other - v[i]) : (v[i] + other);
                    }
                }

                // 3. Normalize WHT (1/√d).
                #pragma unroll
                for (int i = 0; i < 4; i++) v[i] *= wht_scale;

                // 4. Multiply random ±1 signs (Π^T = D · …).
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    v[i] *= d_tq3_signs[base_elem + i];

                // 5. Re-apply norm.
                float norm = blk->norm;
                #pragma unroll
                for (int i = 0; i < 4; i++) v[i] *= norm;

                // 6. Write to shared-mem tile (fp16).
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    tile_dst[base_elem + i] = __float2half(v[i]);
            }
            __syncthreads();

            // ============ Score compute (identical to base split kernel) ============
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

            // Softmax + O accumulation (identical to base split kernel).
            if (warp < GQA) {
                int g = warp;
                float s_val = s_smem[g * BM + lane];
                float m_row = s_val;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    m_row = fmaxf(m_row, __shfl_xor_sync(0xffffffff, m_row, off));
                float m_new = fmaxf(m_w, m_row);
                float correction = expf(m_w - m_new);
                float p_lane    = expf(s_val - m_new);
                float sum_p = p_lane;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    sum_p += __shfl_xor_sync(0xffffffff, sum_p, off);
                #pragma unroll
                for (int i = 0; i < LANE_D; i++) acc_o[i] *= correction;
                l_w = l_w * correction + sum_p;
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
    }

    // Write partial outputs (identical to base split kernel).
    if (warp < GQA) {
        int g = warp;
        int q_head = kv_head * GQA + g;
        size_t ml_idx = ((size_t)q_head * sub_n_max + t_idx) * K_SPLITS + split_idx;
        if (lane == 0) {
            part_m[ml_idx] = m_w;
            part_l[ml_idx] = l_w;
        }
        size_t o_base = (((size_t)q_head * sub_n_max + t_idx) * K_SPLITS + split_idx) * HD;
        #pragma unroll
        for (int i = 0; i < LANE_D; i++) {
            int c = lane * LANE_D + i;
            part_o[o_base + c] = acc_o[i];
        }
    }
}

// K/V-sharing FA kernel. One block processes N_T consecutive t_idx values for
// the same kv_head, so the K/V tile is loaded ONCE per tile_start and reused
// for all N_T softmax/O accumulators. Halves (N_T=2) or quarters (N_T=4) the
// K/V HBM traffic at long contexts where K/V load dominates wall time.
//
// Block layout: BLOCK threads, N_WARPS = BLOCK/32. Each warp owns one (n, g)
// pair where n ∈ [0, N_T), g ∈ [0, GQA), so we need BLOCK >= N_T*GQA*32.
//   9B (GQA=4) N_T=2: 8 warps → BLOCK=256 (exact fit)
//   27B (GQA=6) N_T=2: 12 warps → BLOCK=512 (4 idle warps for K/V load only)
//
// SMEM (HD=256 BM=32 N_T=2):
//   9B  GQA=4: q_s 4KB + k 16KB + v 16KB + s 1KB = 37 KB (fits 48 KB default)
//   27B GQA=6: q_s 6KB + k 16KB + v 16KB + s 1.5KB ≈ 40 KB (fits)
template<int HD, int GQA, int BM, int BLOCK, int N_T>
__global__ void flash_attn_chunk_fused_nt(
    const half* __restrict__ q_chunk,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    half*       __restrict__ out_chunk,
    int num_q, int num_kv,
    int start_pos, int sub_n, float scale
) {
    static_assert(BM == 32, "flash_attn_chunk_fused_nt assumes BM == 32 (lane==r)");
    constexpr int N_WARPS  = BLOCK / 32;
    constexpr int LANE_D   = HD / 32;
    constexpr int N_ACTIVE = N_T * GQA;
    static_assert(N_ACTIVE <= N_WARPS, "Need BLOCK >= N_T*GQA*32 warps");

    int kv_head   = blockIdx.x;
    int group_idx = blockIdx.y;
    int t_base    = group_idx * N_T;
    if (t_base >= sub_n) return;

    int tid  = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    int my_n  = warp / GQA;          // valid only if warp < N_ACTIVE
    int my_g  = warp - my_n * GQA;
    int my_t  = t_base + my_n;
    bool warp_active = (warp < N_ACTIVE) && (my_t < sub_n);

    extern __shared__ unsigned char smem_fa_nt[];
    half*  q_s    = (half*)smem_fa_nt;
    half*  k_tile = q_s    + N_T * GQA * HD;
    half*  v_tile = k_tile + BM  * HD;
    float* s_smem = (float*)(v_tile + BM * HD);

    // Load Q for all (n, g) in this group. Out-of-range t_idx → zero-pad.
    #pragma unroll
    for (int i = tid; i < N_T * GQA * HD; i += BLOCK) {
        int n = i / (GQA * HD);
        int rem = i - n * GQA * HD;
        int g = rem / HD;
        int c = rem - g * HD;
        int t_idx = t_base + n;
        if (t_idx < sub_n) {
            int q_head = kv_head * GQA + g;
            q_s[(n * GQA + g) * HD + c] =
                q_chunk[(size_t)t_idx * num_q * HD + (size_t)q_head * HD + c];
        } else {
            q_s[(n * GQA + g) * HD + c] = __float2half(0.0f);
        }
    }
    __syncthreads();

    float acc_o[LANE_D];
    #pragma unroll
    for (int i = 0; i < LANE_D; i++) acc_o[i] = 0.0f;
    float m_w = -INFINITY;
    float l_w = 0.0f;

    // Furthest position any t_idx in this group looks at. Smaller-t_idx warps
    // mask scores past their own causal end via -INFINITY, so the loop runs
    // up to the group max (1..N_T-1 extra tile loads on the trailing warps).
    int group_size = min(N_T, sub_n - t_base);
    int max_active_end = t_base + group_size - 1 + start_pos + 1;

    for (int tile_start = 0; tile_start < max_active_end; tile_start += BM) {
        int tile_end_max = min(tile_start + BM, max_active_end);
        int tile_len = tile_end_max - tile_start;

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

        // Score: each active (n,g) warp computes BM scores over its own Q row,
        // sharing the same K tile. Per-warp work is BM dot products serially —
        // 2× the original kernel's per-warp score work, but N_T t_idx values
        // produced per K/V load (the actual win at long contexts).
        if (warp_active) {
            const half2* qs2 = reinterpret_cast<const half2*>(
                q_s + (my_n * GQA + my_g) * HD);
            int my_active_end = my_t + start_pos + 1;
            #pragma unroll
            for (int r = 0; r < BM; r++) {
                float partial = 0.0f;
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
                    int abs_pos_r = tile_start + r;
                    bool causal_ok = (abs_pos_r < my_active_end);
                    bool tile_ok   = (r < tile_len);
                    float val = (causal_ok && tile_ok) ? partial * scale : -INFINITY;
                    s_smem[(my_n * GQA + my_g) * BM + r] = val;
                }
            }
        }
        __syncthreads();

        if (warp_active) {
            float s_val = s_smem[(my_n * GQA + my_g) * BM + lane];

            float m_row = s_val;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                m_row = fmaxf(m_row, __shfl_xor_sync(0xffffffff, m_row, off));
            float m_new = fmaxf(m_w, m_row);
            float correction = expf(m_w - m_new);
            float p_lane = expf(s_val - m_new);
            float sum_p = p_lane;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                sum_p += __shfl_xor_sync(0xffffffff, sum_p, off);

            #pragma unroll
            for (int i = 0; i < LANE_D; i++) acc_o[i] *= correction;
            l_w = l_w * correction + sum_p;

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

    if (warp_active) {
        int q_head = kv_head * GQA + my_g;
        float inv_l = (l_w > 0.0f) ? (1.0f / l_w) : 0.0f;
        size_t out_base = (size_t)my_t * num_q * HD + (size_t)q_head * HD;
        #pragma unroll
        for (int i = 0; i < LANE_D; i++) {
            int c = lane * LANE_D + i;
            out_chunk[out_base + c] = __float2half(acc_o[i] * inv_l);
        }
    }
}

// BM-generic FA kernel. Each lane covers R_PER_LANE = BM/32 r values per softmax
// row, so BM > 32 fits without lane==r assumption. BM=32 path (R_PER_LANE=1)
// reduces to the same structure as flash_attn_chunk_fused above.
//
// SMEM at HD=256, GQA=4, BM=64: q_s 2KB + k_tile 32KB + v_tile 32KB + s_smem 1KB
// = 67KB (needs 96KB dynamic SMEM opt-in on Volta — caller responsibility).
template<int HD, int GQA, int BM, int BLOCK>
__global__ void flash_attn_chunk_fused_bm(
    const half* __restrict__ q_chunk,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    half*       __restrict__ out_chunk,
    int num_q, int num_kv,
    int start_pos, int sub_n, float scale
) {
    static_assert(BM % 32 == 0, "BM must be a multiple of 32");
    constexpr int N_WARPS    = BLOCK / 32;
    constexpr int LANE_D     = HD / 32;
    constexpr int R_PER_LANE = BM / 32;
    int kv_head = blockIdx.x;
    int t_idx   = blockIdx.y;
    if (t_idx >= sub_n) return;
    int abs_pos    = start_pos + t_idx;
    int active_end = abs_pos + 1;
    int tid  = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    extern __shared__ unsigned char smem_fa[];
    half*  q_s    = (half*)smem_fa;
    half*  k_tile = q_s    + GQA * HD;
    half*  v_tile = k_tile + BM  * HD;
    float* s_smem = (float*)(v_tile + BM * HD);

    #pragma unroll
    for (int i = tid; i < GQA * HD; i += BLOCK) {
        int g = i / HD;
        int c = i - g * HD;
        int q_head = kv_head * GQA + g;
        q_s[g * HD + c] = q_chunk[(size_t)t_idx * num_q * HD
                                  + (size_t)q_head * HD + c];
    }
    __syncthreads();

    float acc_o[LANE_D];
    #pragma unroll
    for (int i = 0; i < LANE_D; i++) acc_o[i] = 0.0f;
    float m_w = -INFINITY;
    float l_w = 0.0f;

    for (int tile_start = 0; tile_start < active_end; tile_start += BM) {
        int tile_end = min(tile_start + BM, active_end);
        int tile_len = tile_end - tile_start;

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

        // Score: warp-per-(g,r). NITERS = ceil(GQA*BM / N_WARPS).
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

        // Softmax + O update. Each warp owns one g; lane handles R_PER_LANE r's
        // per softmax row (r = rp*32 + lane for rp in [0, R_PER_LANE)).
        if (warp < GQA) {
            int g = warp;
            float s_vals[R_PER_LANE];
            #pragma unroll
            for (int rp = 0; rp < R_PER_LANE; rp++)
                s_vals[rp] = s_smem[g * BM + rp * 32 + lane];

            float m_row = s_vals[0];
            #pragma unroll
            for (int rp = 1; rp < R_PER_LANE; rp++)
                m_row = fmaxf(m_row, s_vals[rp]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                m_row = fmaxf(m_row, __shfl_xor_sync(0xffffffff, m_row, off));
            float m_new = fmaxf(m_w, m_row);
            float correction = expf(m_w - m_new);

            float p_vals[R_PER_LANE];
            float sum_p = 0.0f;
            #pragma unroll
            for (int rp = 0; rp < R_PER_LANE; rp++) {
                p_vals[rp] = expf(s_vals[rp] - m_new);
                sum_p += p_vals[rp];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                sum_p += __shfl_xor_sync(0xffffffff, sum_p, off);

            #pragma unroll
            for (int i = 0; i < LANE_D; i++) acc_o[i] *= correction;
            l_w = l_w * correction + sum_p;

            // O += sum_r P[r] * V[r, lane*LANE_D : lane*LANE_D + LANE_D]
            #pragma unroll
            for (int rp = 0; rp < R_PER_LANE; rp++) {
                #pragma unroll
                for (int r_in = 0; r_in < 32; r_in++) {
                    float p_r = __shfl_sync(0xffffffff, p_vals[rp], r_in);
                    int r = rp * 32 + r_in;
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
            }
            m_w = m_new;
        }
        __syncthreads();
    }

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
