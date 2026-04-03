#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

// ============ TurboQuant KV Cache (PolarQuant MSE-only) ============
// Based on Google's TurboQuant (ICLR 2026)
// - Random orthogonal rotation (Walsh-Hadamard Transform)
// - Lloyd-Max optimal scalar quantizer
// - No QJL (community found it hurts after softmax)
//
// Block: 128 values → 3-bit = 48 bytes data + 4 bytes norm = 52 bytes
// Compression: 128 * 2 bytes (fp16) = 256 bytes → 52 bytes = 4.9x

#define TQ_BLOCK_SIZE 128
#define TQ3_BITS 3
#define TQ3_LEVELS 8   // 2^3

// Pre-computed Lloyd-Max centroids for 3-bit (8 levels)
// For Beta(64,64) distribution after WHT rotation on dim=128
// These are normalized centroids — multiply by norm to get actual values
__constant__ float d_tq3_centroids[TQ3_LEVELS] = {
    -1.2820f, -0.7560f, -0.3422f, -0.0312f,
     0.0312f,  0.3422f,  0.7560f,  1.2820f
};

// Pre-computed Lloyd-Max boundaries (midpoints between centroids)
__constant__ float d_tq3_bounds[TQ3_LEVELS - 1] = {
    -1.0190f, -0.5491f, -0.1867f,  0.0000f,
     0.1867f,  0.5491f,  1.0190f
};

// ============ Fast Walsh-Hadamard Transform (in-place, dim=128) ============

__device__ void fast_wht_128(float* data) {
    // WHT for power-of-2 dim: log2(128) = 7 stages
    for (int step = 1; step < TQ_BLOCK_SIZE; step <<= 1) {
        for (int i = 0; i < TQ_BLOCK_SIZE; i += step * 2) {
            for (int j = i; j < i + step; j++) {
                float a = data[j];
                float b = data[j + step];
                data[j]        = a + b;
                data[j + step] = a - b;
            }
        }
    }
    // Normalize
    float scale = rsqrtf((float)TQ_BLOCK_SIZE);
    for (int i = 0; i < TQ_BLOCK_SIZE; i++)
        data[i] *= scale;
}

__device__ void fast_iwht_128(float* data) {
    // Inverse WHT = same as WHT (self-inverse up to scale)
    fast_wht_128(data);
}

// ============ Quantize a single block (128 fp16 → 52 bytes) ============

struct __align__(4) block_tq3 {
    float norm;                                // 4 bytes: L2 norm
    uint8_t qs[TQ_BLOCK_SIZE * 3 / 8];       // 48 bytes: packed 3-bit indices
};  // Total 52 bytes
static_assert(sizeof(block_tq3) == 52, "TQ3 block size");

__device__ uint8_t tq3_find_bin(float val) {
    // Binary search in boundaries
    uint8_t idx = 0;
    #pragma unroll
    for (int i = 0; i < TQ3_LEVELS - 1; i++) {
        if (val > d_tq3_bounds[i]) idx = i + 1;
    }
    return idx;
}

// Pack 8 x 3-bit values into 3 bytes
__device__ void pack_3bit_8(uint8_t* dst, const uint8_t* indices) {
    // indices[0..7] each 0-7 (3 bits)
    // Pack into 24 bits = 3 bytes
    uint32_t packed = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++)
        packed |= ((uint32_t)indices[i]) << (i * 3);
    dst[0] = packed & 0xFF;
    dst[1] = (packed >> 8) & 0xFF;
    dst[2] = (packed >> 16) & 0xFF;
}

// Unpack 3 bytes → 8 x 3-bit values
__device__ void unpack_3bit_8(const uint8_t* src, uint8_t* indices) {
    uint32_t packed = src[0] | ((uint32_t)src[1] << 8) | ((uint32_t)src[2] << 16);
    #pragma unroll
    for (int i = 0; i < 8; i++)
        indices[i] = (packed >> (i * 3)) & 0x7;
}

// ============ Quantize kernel: fp16[128] → block_tq3 ============

__global__ void tq3_quantize_kernel(
    const half* __restrict__ input,   // [n_blocks * 128]
    block_tq3* __restrict__ output,   // [n_blocks]
    int n_blocks
) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= n_blocks) return;
    
    const half* in = input + bid * TQ_BLOCK_SIZE;
    block_tq3* out = &output[bid];
    
    // Load to float
    float data[TQ_BLOCK_SIZE];
    for (int i = 0; i < TQ_BLOCK_SIZE; i++)
        data[i] = __half2float(in[i]);
    
    // Compute and save norm
    float norm_sq = 0.0f;
    for (int i = 0; i < TQ_BLOCK_SIZE; i++)
        norm_sq += data[i] * data[i];
    float norm = sqrtf(norm_sq);
    out->norm = norm;
    
    if (norm < 1e-10f) {
        // Zero vector
        for (int i = 0; i < TQ_BLOCK_SIZE * 3 / 8; i++)
            out->qs[i] = 0;
        return;
    }
    
    // Normalize
    float inv_norm = 1.0f / norm;
    for (int i = 0; i < TQ_BLOCK_SIZE; i++)
        data[i] *= inv_norm;
    
    // Apply WHT rotation
    fast_wht_128(data);
    
    // Quantize each coordinate with Lloyd-Max
    uint8_t indices[TQ_BLOCK_SIZE];
    for (int i = 0; i < TQ_BLOCK_SIZE; i++)
        indices[i] = tq3_find_bin(data[i]);
    
    // Pack 3-bit indices (groups of 8 → 3 bytes)
    for (int g = 0; g < TQ_BLOCK_SIZE / 8; g++)
        pack_3bit_8(&out->qs[g * 3], &indices[g * 8]);
}

// ============ Dequantize kernel: block_tq3 → fp16[128] ============

__global__ void tq3_dequantize_kernel(
    const block_tq3* __restrict__ input,
    half* __restrict__ output,
    int n_blocks
) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= n_blocks) return;
    
    const block_tq3* in = &input[bid];
    half* out = output + bid * TQ_BLOCK_SIZE;
    
    float norm = in->norm;
    
    // Unpack indices and lookup centroids
    float data[TQ_BLOCK_SIZE];
    for (int g = 0; g < TQ_BLOCK_SIZE / 8; g++) {
        uint8_t indices[8];
        unpack_3bit_8(&in->qs[g * 3], indices);
        for (int j = 0; j < 8; j++)
            data[g * 8 + j] = d_tq3_centroids[indices[j]];
    }
    
    // Inverse WHT
    fast_iwht_128(data);
    
    // Restore norm
    for (int i = 0; i < TQ_BLOCK_SIZE; i++)
        out[i] = __float2half(data[i] * norm);
}

// ============ TurboQuant KV Cache Manager ============

struct TurboQuantCache {
    block_tq3* k_cache;    // [num_layers, max_seq, num_kv_heads * head_dim / 128] blocks
    block_tq3* v_cache;    // same
    int max_seq;
    int num_layers;
    int num_kv_heads;
    int head_dim;
    int blocks_per_token;   // num_kv_heads * head_dim / 128
    
    // Temp fp16 buffer for dequantized KV
    half* k_fp16_buf;
    half* v_fp16_buf;
    
    void init(int _num_layers, int _max_seq, int _num_kv_heads, int _head_dim, int gpu_id) {
        num_layers = _num_layers;
        max_seq = _max_seq;
        num_kv_heads = _num_kv_heads;
        head_dim = _head_dim;
        blocks_per_token = num_kv_heads * head_dim / TQ_BLOCK_SIZE;
        
        cudaSetDevice(gpu_id);
        size_t cache_size = (size_t)num_layers * max_seq * blocks_per_token * sizeof(block_tq3);
        cudaMalloc(&k_cache, cache_size);
        cudaMalloc(&v_cache, cache_size);
        cudaMalloc(&k_fp16_buf, max_seq * num_kv_heads * head_dim * sizeof(half));
        cudaMalloc(&v_fp16_buf, max_seq * num_kv_heads * head_dim * sizeof(half));
        
        printf("TurboQuant KV cache: %.1f MB per K/V (%.1fx compression vs fp16)\n",
            cache_size / 1e6, 
            (float)(num_layers * max_seq * num_kv_heads * head_dim * 2) / cache_size);
    }
    
    // Store K/V for one token at position pos
    void store(int layer, int pos, const half* k, const half* v, cudaStream_t stream) {
        size_t offset = ((size_t)layer * max_seq + pos) * blocks_per_token;
        tq3_quantize_kernel<<<(blocks_per_token+31)/32, 32, 0, stream>>>(
            k, &k_cache[offset], blocks_per_token);
        tq3_quantize_kernel<<<(blocks_per_token+31)/32, 32, 0, stream>>>(
            v, &v_cache[offset], blocks_per_token);
    }
    
    // Load all K/V up to seq_len, dequantized to fp16
    void load(int layer, int seq_len, half* k_out, half* v_out, cudaStream_t stream) {
        int total_blocks = seq_len * blocks_per_token;
        size_t offset = (size_t)layer * max_seq * blocks_per_token;
        tq3_dequantize_kernel<<<(total_blocks+31)/32, 32, 0, stream>>>(
            &k_cache[offset], k_out, total_blocks);
        tq3_dequantize_kernel<<<(total_blocks+31)/32, 32, 0, stream>>>(
            &v_cache[offset], v_out, total_blocks);
    }
    
    void free_cache() {
        cudaFree(k_cache); cudaFree(v_cache);
        cudaFree(k_fp16_buf); cudaFree(v_fp16_buf);
    }
};
