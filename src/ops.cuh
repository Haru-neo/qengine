#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// RMSNorm: out = x * rsqrt(mean(x^2) + eps) * weight
__global__ void rms_norm_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int hidden_size,
    float eps
) {
    // One block per row
    int row = blockIdx.x;
    const half* x_row = x + row * hidden_size;
    half* out_row = out + row * hidden_size;
    
    extern __shared__ float sdata[];
    
    // Compute sum of squares
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        sum += v * v;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    
    float rms = rsqrtf(sdata[0] / hidden_size + eps);
    
    // Apply norm + weight
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = __half2float(x_row[i]) * rms * __half2float(weight[i]);
        out_row[i] = __float2half(v);
    }
}

void rms_norm(half* out, const half* x, const half* weight, int rows, int hidden_size, float eps, cudaStream_t stream = 0) {
    int threads = min(hidden_size, 256);
    rms_norm_kernel<<<rows, threads, threads * sizeof(float), stream>>>(x, weight, out, hidden_size, eps);
}

// RMSNorm with F32 weights
__global__ void rms_norm_f32w_kernel(
    const half* __restrict__ x,
    const float* __restrict__ weight,
    half* __restrict__ out,
    int hidden_size,
    float eps
) {
    int row = blockIdx.x;
    const half* x_row = x + row * hidden_size;
    half* out_row = out + row * hidden_size;
    
    extern __shared__ float sdata[];
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        sum += v * v;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    
    float rms = rsqrtf(sdata[0] / hidden_size + eps);
    
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = __half2float(x_row[i]) * rms * weight[i];
        out_row[i] = __float2half(v);
    }
}

void rms_norm_f32w(half* out, const half* x, const float* weight, int rows, int hidden_size, float eps, cudaStream_t stream = 0) {
    int threads = min(hidden_size, 256);
    rms_norm_f32w_kernel<<<rows, threads, threads * sizeof(float), stream>>>(x, weight, out, hidden_size, eps);
}

// ============ FP32-input variants for hidden state ============
// Read fp32 hidden, output fp16 (for projections that take fp16)
__global__ void rms_norm_f32in_kernel(
    const float* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int hidden_size,
    float eps
) {
    int row = blockIdx.x;
    const float* x_row = x + row * hidden_size;
    half* out_row = out + row * hidden_size;
    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = x_row[i];
        sum += v * v;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / hidden_size + eps);
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out_row[i] = __float2half(x_row[i] * rms * __half2float(weight[i]));
    }
}

__global__ void rms_norm_f32in_f32w_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    half* __restrict__ out,
    int hidden_size,
    float eps
) {
    int row = blockIdx.x;
    const float* x_row = x + row * hidden_size;
    half* out_row = out + row * hidden_size;
    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = x_row[i];
        sum += v * v;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / hidden_size + eps);
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out_row[i] = __float2half(x_row[i] * rms * weight[i]);
    }
}

inline void rms_norm_f32in(half* out, const float* x, const half* weight, int rows, int hidden_size, float eps, cudaStream_t stream = 0) {
    int threads = min(hidden_size, 256);
    rms_norm_f32in_kernel<<<rows, threads, threads * sizeof(float), stream>>>(x, weight, out, hidden_size, eps);
}

inline void rms_norm_f32in_f32w(half* out, const float* x, const float* weight, int rows, int hidden_size, float eps, cudaStream_t stream = 0) {
    int threads = min(hidden_size, 256);
    rms_norm_f32in_f32w_kernel<<<rows, threads, threads * sizeof(float), stream>>>(x, weight, out, hidden_size, eps);
}

// FP32 residual add: hidden_f32 += proj_out (half)
__global__ void add_kernel_f32(float* __restrict__ x, const half* __restrict__ residual, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = x[idx] + __half2float(residual[idx]);
    }
}

// Convert half → fp32 (used after embedding lookup)
__global__ void half_to_float_kernel(const half* __restrict__ src, float* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __half2float(src[idx]);
}

// Convert fp32 → half (used before output projection)
__global__ void float_to_half_kernel(const float* __restrict__ src, half* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(src[idx]);
}


// cuBLAS fp16 GEMM wrapper
// C = A @ B^T  (A: [M, K], B: [N, K] -> C: [M, N])
void gemm_fp16(cublasHandle_t handle, half* C, const half* A, const half* B,
               int M, int N, int K, cudaStream_t stream = 0) {
    cublasSetStream(handle, stream);
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    // cublas is column-major, so we compute B^T @ A^T = (A @ B)^T
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, K,
                A, K,
                &beta,
                C, N);
}

// SiLU activation: x * sigmoid(x)
__global__ void silu_kernel(half* __restrict__ x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __half2float(x[idx]);
        x[idx] = __float2half(v / (1.0f + expf(-v)));
    }
}

// Element-wise multiply: out = a * b
__global__ void mul_kernel(half* __restrict__ out, const half* __restrict__ a, const half* __restrict__ b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(__half2float(a[idx]) * __half2float(b[idx]));
    }
}

// Residual add: x += residual
__global__ void add_kernel(half* __restrict__ x, const half* __restrict__ residual, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = __float2half(__half2float(x[idx]) + __half2float(residual[idx]));
    }
}

// Fused SiLU(a) * b: out[i] = silu(a[i]) * b[i]
__global__ void silu_mul_kernel(half* __restrict__ a, const half* __restrict__ b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __half2float(a[idx]);
        float silu = va / (1.0f + expf(-va));
        a[idx] = __float2half(silu * __half2float(b[idx]));
    }
}

// Fused GeLU(tanh approx)(a) * b — used by Gemma's GeGLU FFN
__global__ void gelu_tanh_mul_kernel(half* __restrict__ a, const half* __restrict__ b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __half2float(a[idx]);
        const float k = 0.7978845608f;  // sqrt(2/pi)
        float inner = k * (va + 0.044715f * va * va * va);
        float gelu = 0.5f * va * (1.0f + tanhf(inner));
        a[idx] = __float2half(gelu * __half2float(b[idx]));
    }
}

__global__ void dequant_embd_q5k_row(
    const void* __restrict__ embd, half* __restrict__ out, int token_id, int H
) {
    const int QK = 256;
    int blocks_per_row = H / QK;
    struct bq5k { half d; half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
    const bq5k* row = (const bq5k*)embd + (size_t)token_id * blocks_per_row;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H) return;
    int blk = idx / QK, elem = idx % QK;
    const bq5k* b = &row[blk];
    float d = __half2float(b->d), dmin = __half2float(b->dmin);
    int sub = elem >> 5;
    const uint8_t* s = b->scales;
    uint8_t sc, mn;
    if (sub < 4) { sc = s[sub] & 63; mn = s[sub+4] & 63; }
    else { sc = (s[sub+4]&0xF)|((s[sub-4]>>6)<<4); mn = (s[sub+4]>>4)|((s[sub]>>6)<<4); }
    uint8_t q4 = (b->qs[elem>>1] >> (4*(elem&1))) & 0xf;
    uint8_t qh = (b->qh[elem>>3] >> (elem&7)) & 1;
    out[idx] = __float2half(d * sc * (q4|(qh<<4)) - dmin * mn);
}

__global__ void dequant_embd_q6k_row(
    const void* __restrict__ embd, half* __restrict__ out, int token_id, int H
) {
    const int QK = 256;
    int blocks_per_row = H / QK;
    struct bq6k { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; };
    const bq6k* row = (const bq6k*)embd + (size_t)token_id * blocks_per_row;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H) return;
    int blk = idx / QK, elem = idx % QK;
    const bq6k* b = &row[blk];
    float d = __half2float(b->d);
    float scale = d * b->scales[elem >> 4];
    uint8_t ql4 = (b->ql[elem>>1] >> (4*(elem&1))) & 0xf;
    uint8_t qh2 = (b->qh[elem>>2] >> (2*(elem&3))) & 0x3;
    int8_t q = (int8_t)(ql4 | (qh2<<4)) - 32;
    out[idx] = __float2half(scale * q);
}

// ============ CORRECTED Q5_K dequant (matching llama.cpp layout) ============
// Q5_K layout: 256 elements in 4 groups of 64
// Group g (0..3): elements g*64 .. g*64+63
//   sub 0: elements g*64+0..31  → ql[g*32+l] LOW nibble, qh[l] bit (g*2+0)
//   sub 1: elements g*64+32..63 → ql[g*32+l] HIGH nibble, qh[l] bit (g*2+1)

__global__ void dequant_embd_q5k_row_v2(
    const void* __restrict__ embd, half* __restrict__ out, int token_id, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H) return;

    const int QK = 256;
    int blocks_per_row = H / QK;
    struct bq5k { half d; half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
    const bq5k* row = (const bq5k*)embd + (size_t)token_id * blocks_per_row;

    int blk = idx / QK;
    int elem = idx % QK;
    const bq5k* b = &row[blk];

    float d = __half2float(b->d);
    float dmin = __half2float(b->dmin);

    // Decode layout
    int group = elem / 64;            // 0..3
    int pos_in_group = elem % 64;     // 0..63
    int sub = pos_in_group >= 32 ? 1 : 0;
    int l = pos_in_group % 32;        // 0..31

    // Scale index
    int is = group * 2 + sub;

    // Unpack scales
    const uint8_t* s = b->scales;
    uint8_t sc, mn;
    if (is < 4) { sc = s[is] & 63; mn = s[is + 4] & 63; }
    else {
        sc = (s[is + 4] & 0xF) | ((s[is - 4] >> 6) << 4);
        mn = (s[is + 4] >> 4) | ((s[is] >> 6) << 4);
    }

    // ql
    uint8_t ql_byte = b->qs[group * 32 + l];
    uint8_t q4 = sub == 0 ? (ql_byte & 0xF) : (ql_byte >> 4);

    // qh
    uint8_t qh_bit = (b->qh[l] >> (group * 2 + sub)) & 1;

    uint8_t q5 = q4 | (qh_bit << 4);
    out[idx] = __float2half(d * sc * q5 - dmin * mn);
}

// ============ CORRECTED Q6_K dequant ============
// Q6_K layout: 256 elements in 2 super-groups of 128
// SuperGroup sg (0..1): ql[sg*64..sg*64+63], qh[sg*32..sg*32+31]
// Within each 128: for l=0..31:
//   elem l+0:   ql[l] LOW,    qh[l] bits 0-1,  scale[sg*8 + 0]
//   elem l+32:  ql[l+32] LOW, qh[l] bits 2-3,  scale[sg*8 + 2]
//   elem l+64:  ql[l] HIGH,   qh[l] bits 4-5,  scale[sg*8 + 4]
//   elem l+96:  ql[l+32] HIGH,qh[l] bits 6-7,  scale[sg*8 + 6]

__global__ void dequant_embd_q6k_row_v2(
    const void* __restrict__ embd, half* __restrict__ out, int token_id, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H) return;

    const int QK = 256;
    int blocks_per_row = H / QK;
    struct bq6k { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; };
    const bq6k* row = (const bq6k*)embd + (size_t)token_id * blocks_per_row;

    int blk = idx / QK;
    int elem = idx % QK;
    const bq6k* b = &row[blk];
    float d = __half2float(b->d);

    // llama.cpp exact layout:
    // n = (elem/128)*128, local=elem%128, quarter=local/32, l=local%32
    // is = n/16 + l/16
    // scale = sc[is + quarter*2]
    int n = (elem / 128) * 128;
    int local = elem % 128;
    int quarter = local / 32;
    int l = local % 32;
    int is = n / 16 + l / 16;

    const uint8_t* ql = b->ql + n/2;
    const uint8_t* qh = b->qh + n/4;

    uint8_t ql4, qh2;
    switch (quarter) {
        case 0: ql4 = ql[l] & 0xF;      qh2 = (qh[l] >> 0) & 3; break;
        case 1: ql4 = ql[l + 32] & 0xF;  qh2 = (qh[l] >> 2) & 3; break;
        case 2: ql4 = ql[l] >> 4;         qh2 = (qh[l] >> 4) & 3; break;
        default:ql4 = ql[l + 32] >> 4;    qh2 = (qh[l] >> 6) & 3; break;
    }

    int8_t q = (int8_t)(ql4 | (qh2 << 4)) - 32;
    out[idx] = __float2half(d * b->scales[is + quarter*2] * q);
}

// Q8_0: 32 elements per block, half scale + int8 quants
__global__ void dequant_embd_q8_0_row(
    const void* __restrict__ embd, half* __restrict__ out, int token_id, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H) return;
    struct bq8_0 { half d; int8_t qs[32]; };
    int blocks_per_row = H / 32;
    const bq8_0* row = (const bq8_0*)embd + (size_t)token_id * blocks_per_row;
    int blk = idx / 32;
    int elem = idx % 32;
    out[idx] = __float2half(__half2float(row[blk].d) * row[blk].qs[elem]);
}

// Q8_K: 256 elements per block, float scale + int8 quants (used by Gemma)
__global__ void dequant_embd_q8k_row(
    const void* __restrict__ embd, half* __restrict__ out, int token_id, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H) return;
    struct bq8_k { float d; int8_t qs[256]; int16_t bsums[16]; };
    int blocks_per_row = H / 256;
    const bq8_k* row = (const bq8_k*)embd + (size_t)token_id * blocks_per_row;
    int blk = idx / 256;
    int elem = idx % 256;
    out[idx] = __float2half(row[blk].d * row[blk].qs[elem]);
}

// Scale embedding by sqrt(hidden_size) (used by Gemma)
__global__ void scale_embedding_kernel(half* __restrict__ x, float scale, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < H) x[idx] = __float2half(__half2float(x[idx]) * scale);
}

// Softcap: x = scale * tanh(x / scale) (used by Gemma final logits)
__global__ void softcap_kernel(half* __restrict__ x, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = __half2float(x[idx]);
        x[idx] = __float2half(scale * tanhf(v / scale));
    }
}
