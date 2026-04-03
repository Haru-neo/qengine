#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Q5_K 블록 구조 (llama.cpp와 동일)
// 256 elements per block
typedef struct {
    __half d;                    // super-block scale (fp16)
    __half dmin;                 // super-block min (fp16)
    uint8_t scales[12];          // sub-block scales and mins (6-bit each, packed)
    uint8_t qh[32];             // high bits of quants
    uint8_t qs[128];            // low 4 bits of quants
} block_q5_K;
static_assert(sizeof(block_q5_K) == 176, "Q5_K block size mismatch");

// Q6_K 블록 구조
typedef struct {
    uint8_t ql[128];            // low 4 bits of quants
    uint8_t qh[64];             // high 2 bits of quants
    int8_t scales[16];          // scales (int8)
    __half d;                    // super-block scale (fp16)
} block_q6_K;
static_assert(sizeof(block_q6_K) == 210, "Q6_K block size mismatch");

// Q5_K dequantize kernel: [n_blocks] -> [n_blocks * 256] fp16
__global__ void dequantize_q5_K(const block_q5_K* __restrict__ src, half* __restrict__ dst, int n_blocks) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= n_blocks) return;
    
    const block_q5_K* blk = &src[block_idx];
    half* out = dst + block_idx * 256;
    
    float d = __half2float(blk->d);
    float dmin = __half2float(blk->dmin);
    
    // Unpack 6-bit scales and mins from 12 bytes
    uint8_t sc[8], m[8];
    for (int i = 0; i < 8; i++) {
        if (i < 4) {
            sc[i] = blk->scales[i] & 0x3f;
            m[i]  = blk->scales[i + 4] & 0x3f;
        } else {
            sc[i] = ((blk->scales[i + 4] & 0xF) | ((blk->scales[i - 4] >> 6) << 4));
            m[i]  = ((blk->scales[i + 4] >> 4)   | ((blk->scales[i]     >> 6) << 4));
        }
    }
    
    // Dequantize 256 elements in 8 sub-blocks of 32
    for (int sub = 0; sub < 8; sub++) {
        float scale = d * sc[sub];
        float min_val = dmin * m[sub];
        
        for (int j = 0; j < 32; j++) {
            int idx = sub * 32 + j;
            uint8_t q4 = blk->qs[idx / 2];
            q4 = (idx & 1) ? (q4 >> 4) : (q4 & 0xf);
            uint8_t qh_bit = (blk->qh[idx / 8] >> (idx % 8)) & 1;
            uint8_t q = q4 | (qh_bit << 4);
            float val = scale * q - min_val;
            out[idx] = __float2half(val);
        }
    }
}

// Q6_K dequantize kernel
__global__ void dequantize_q6_K(const block_q6_K* __restrict__ src, half* __restrict__ dst, int n_blocks) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= n_blocks) return;
    
    const block_q6_K* blk = &src[block_idx];
    half* out = dst + block_idx * 256;
    
    float d = __half2float(blk->d);
    
    for (int sub = 0; sub < 16; sub++) {
        float scale = d * blk->scales[sub];
        for (int j = 0; j < 16; j++) {
            int idx = sub * 16 + j;
            uint8_t ql_byte = blk->ql[idx / 2];
            uint8_t ql4 = (idx & 1) ? (ql_byte >> 4) : (ql_byte & 0xf);
            uint8_t qh2 = (blk->qh[idx / 4] >> (2 * (idx % 4))) & 0x3;
            int8_t q = (int8_t)(ql4 | (qh2 << 4)) - 32;
            out[idx] = __float2half(scale * q);
        }
    }
}

// F32 tensor to fp16
__global__ void convert_f32_to_f16(const float* __restrict__ src, half* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(src[idx]);
}

// Dequant any tensor to fp16 buffer on same GPU
// Returns fp16 pointer (caller must free)
half* dequant_to_fp16(void* data, ggml_type type, uint64_t num_elements, uint64_t byte_size) {
    half* fp16_buf;
    cudaMalloc(&fp16_buf, num_elements * sizeof(half));
    
    int n_blocks;
    switch (type) {
        case GGML_TYPE_F16:
            cudaMemcpy(fp16_buf, data, byte_size, cudaMemcpyDeviceToDevice);
            break;
        case GGML_TYPE_F32:
            convert_f32_to_f16<<<(num_elements+255)/256, 256>>>((float*)data, fp16_buf, num_elements);
            break;
        case GGML_TYPE_Q5_K:
            n_blocks = num_elements / 256;
            dequantize_q5_K<<<(n_blocks+31)/32, 32>>>((block_q5_K*)data, fp16_buf, n_blocks);
            break;
        case GGML_TYPE_Q6_K:
            n_blocks = num_elements / 256;
            dequantize_q6_K<<<(n_blocks+31)/32, 32>>>((block_q6_K*)data, fp16_buf, n_blocks);
            break;
        default:
            fprintf(stderr, "Unsupported quant type: %d\n", type);
            cudaFree(fp16_buf);
            return nullptr;
    }
    return fp16_buf;
}
