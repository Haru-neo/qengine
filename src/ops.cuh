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

// Half-input fused N=2 RMSNorm with FP32 weight (for the output_norm
// step in the spec branch where the input comes pre-converted to half).
__global__ void rms_norm_f32w_n2_kernel(
    const half* __restrict__ x_a, const half* __restrict__ x_b,
    const float* __restrict__ weight,
    half* __restrict__ out_a, half* __restrict__ out_b,
    int hidden_size, float eps
) {
    const half* x = (blockIdx.y == 0) ? x_a : x_b;
    half* out     = (blockIdx.y == 0) ? out_a : out_b;
    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = __half2float(x[i]);
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
        out[i] = __float2half(__half2float(x[i]) * rms * weight[i]);
    }
}

inline void rms_norm_f32w_n2(half* out_a, half* out_b,
                             const half* x_a, const half* x_b,
                             const float* weight, int hidden_size,
                             float eps, cudaStream_t stream = 0) {
    int threads = min(hidden_size, 256);
    dim3 grid(1, 2);
    rms_norm_f32w_n2_kernel<<<grid, threads, threads * sizeof(float), stream>>>(
        x_a, x_b, weight, out_a, out_b, hidden_size, eps);
}

// Half-input/half-weight fused N=2 RMSNorm
__global__ void rms_norm_n2_kernel(
    const half* __restrict__ x_a, const half* __restrict__ x_b,
    const half* __restrict__ weight,
    half* __restrict__ out_a, half* __restrict__ out_b,
    int hidden_size, float eps
) {
    const half* x = (blockIdx.y == 0) ? x_a : x_b;
    half* out     = (blockIdx.y == 0) ? out_a : out_b;
    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = __half2float(x[i]);
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
        out[i] = __float2half(__half2float(x[i]) * rms * __half2float(weight[i]));
    }
}

inline void rms_norm_n2(half* out_a, half* out_b,
                        const half* x_a, const half* x_b,
                        const half* weight, int hidden_size,
                        float eps, cudaStream_t stream = 0) {
    int threads = min(hidden_size, 256);
    dim3 grid(1, 2);
    rms_norm_n2_kernel<<<grid, threads, threads * sizeof(float), stream>>>(
        x_a, x_b, weight, out_a, out_b, hidden_size, eps);
}

// Fused N=2 RMSNorm — processes two non-contiguous (x_a, x_b) inputs in
// one kernel launch. Saves the per-launch CPU overhead in the spec
// decoding hot path (forward_*_n2 invokes this once per layer instead of
// twice). blockIdx.y selects the token: 0 → a, 1 → b.
__global__ void rms_norm_f32in_f32w_n2_kernel(
    const float* __restrict__ x_a, const float* __restrict__ x_b,
    const float* __restrict__ weight,
    half* __restrict__ out_a, half* __restrict__ out_b,
    int hidden_size, float eps
) {
    const float* x = (blockIdx.y == 0) ? x_a : x_b;
    half* out      = (blockIdx.y == 0) ? out_a : out_b;
    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = x[i]; sum += v * v;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / hidden_size + eps);
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out[i] = __float2half(x[i] * rms * weight[i]);
    }
}

__global__ void rms_norm_f32in_n2_kernel(
    const float* __restrict__ x_a, const float* __restrict__ x_b,
    const half* __restrict__ weight,
    half* __restrict__ out_a, half* __restrict__ out_b,
    int hidden_size, float eps
) {
    const float* x = (blockIdx.y == 0) ? x_a : x_b;
    half* out      = (blockIdx.y == 0) ? out_a : out_b;
    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = x[i]; sum += v * v;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / hidden_size + eps);
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out[i] = __float2half(x[i] * rms * __half2float(weight[i]));
    }
}

inline void rms_norm_f32in_n2(half* out_a, half* out_b,
                              const float* x_a, const float* x_b,
                              const half* weight, int hidden_size,
                              float eps, cudaStream_t stream = 0) {
    int threads = min(hidden_size, 256);
    dim3 grid(1, 2);
    rms_norm_f32in_n2_kernel<<<grid, threads, threads * sizeof(float), stream>>>(
        x_a, x_b, weight, out_a, out_b, hidden_size, eps);
}

inline void rms_norm_f32in_f32w_n2(half* out_a, half* out_b,
                                   const float* x_a, const float* x_b,
                                   const float* weight, int hidden_size,
                                   float eps, cudaStream_t stream = 0) {
    int threads = min(hidden_size, 256);
    dim3 grid(1, 2);
    rms_norm_f32in_f32w_n2_kernel<<<grid, threads, threads * sizeof(float), stream>>>(
        x_a, x_b, weight, out_a, out_b, hidden_size, eps);
}

// ============ N=3 variants for MTP K=2 speculative decoding ============
// Each kernel uses blockIdx.y ∈ {0,1,2} to pick a/b/c token. Identical RMSNorm
// math to the N=2 kernels — just with a third lane.

__global__ void rms_norm_f32w_n3_kernel(
    const half* __restrict__ x_a, const half* __restrict__ x_b, const half* __restrict__ x_c,
    const float* __restrict__ weight,
    half* __restrict__ out_a, half* __restrict__ out_b, half* __restrict__ out_c,
    int hidden_size, float eps
) {
    const half* x = (blockIdx.y == 0) ? x_a : (blockIdx.y == 1) ? x_b : x_c;
    half* out     = (blockIdx.y == 0) ? out_a : (blockIdx.y == 1) ? out_b : out_c;
    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = __half2float(x[i]); sum += v * v;
    }
    sdata[threadIdx.x] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / hidden_size + eps);
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out[i] = __float2half(__half2float(x[i]) * rms * weight[i]);
    }
}

inline void rms_norm_f32w_n3(half* out_a, half* out_b, half* out_c,
                             const half* x_a, const half* x_b, const half* x_c,
                             const float* weight, int hidden_size,
                             float eps, cudaStream_t stream = 0) {
    int threads = min(hidden_size, 256);
    dim3 grid(1, 3);
    rms_norm_f32w_n3_kernel<<<grid, threads, threads * sizeof(float), stream>>>(
        x_a, x_b, x_c, weight, out_a, out_b, out_c, hidden_size, eps);
}

__global__ void rms_norm_n3_kernel(
    const half* __restrict__ x_a, const half* __restrict__ x_b, const half* __restrict__ x_c,
    const half* __restrict__ weight,
    half* __restrict__ out_a, half* __restrict__ out_b, half* __restrict__ out_c,
    int hidden_size, float eps
) {
    const half* x = (blockIdx.y == 0) ? x_a : (blockIdx.y == 1) ? x_b : x_c;
    half* out     = (blockIdx.y == 0) ? out_a : (blockIdx.y == 1) ? out_b : out_c;
    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = __half2float(x[i]); sum += v * v;
    }
    sdata[threadIdx.x] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / hidden_size + eps);
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out[i] = __float2half(__half2float(x[i]) * rms * __half2float(weight[i]));
    }
}

inline void rms_norm_n3(half* out_a, half* out_b, half* out_c,
                        const half* x_a, const half* x_b, const half* x_c,
                        const half* weight, int hidden_size,
                        float eps, cudaStream_t stream = 0) {
    int threads = min(hidden_size, 256);
    dim3 grid(1, 3);
    rms_norm_n3_kernel<<<grid, threads, threads * sizeof(float), stream>>>(
        x_a, x_b, x_c, weight, out_a, out_b, out_c, hidden_size, eps);
}

__global__ void rms_norm_f32in_f32w_n3_kernel(
    const float* __restrict__ x_a, const float* __restrict__ x_b, const float* __restrict__ x_c,
    const float* __restrict__ weight,
    half* __restrict__ out_a, half* __restrict__ out_b, half* __restrict__ out_c,
    int hidden_size, float eps
) {
    const float* x = (blockIdx.y == 0) ? x_a : (blockIdx.y == 1) ? x_b : x_c;
    half* out      = (blockIdx.y == 0) ? out_a : (blockIdx.y == 1) ? out_b : out_c;
    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = x[i]; sum += v * v;
    }
    sdata[threadIdx.x] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / hidden_size + eps);
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out[i] = __float2half(x[i] * rms * weight[i]);
    }
}

__global__ void rms_norm_f32in_n3_kernel(
    const float* __restrict__ x_a, const float* __restrict__ x_b, const float* __restrict__ x_c,
    const half* __restrict__ weight,
    half* __restrict__ out_a, half* __restrict__ out_b, half* __restrict__ out_c,
    int hidden_size, float eps
) {
    const float* x = (blockIdx.y == 0) ? x_a : (blockIdx.y == 1) ? x_b : x_c;
    half* out      = (blockIdx.y == 0) ? out_a : (blockIdx.y == 1) ? out_b : out_c;
    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = x[i]; sum += v * v;
    }
    sdata[threadIdx.x] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / hidden_size + eps);
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out[i] = __float2half(x[i] * rms * __half2float(weight[i]));
    }
}

inline void rms_norm_f32in_n3(half* out_a, half* out_b, half* out_c,
                              const float* x_a, const float* x_b, const float* x_c,
                              const half* weight, int hidden_size,
                              float eps, cudaStream_t stream = 0) {
    int threads = min(hidden_size, 256);
    dim3 grid(1, 3);
    rms_norm_f32in_n3_kernel<<<grid, threads, threads * sizeof(float), stream>>>(
        x_a, x_b, x_c, weight, out_a, out_b, out_c, hidden_size, eps);
}

inline void rms_norm_f32in_f32w_n3(half* out_a, half* out_b, half* out_c,
                                   const float* x_a, const float* x_b, const float* x_c,
                                   const float* weight, int hidden_size,
                                   float eps, cudaStream_t stream = 0) {
    int threads = min(hidden_size, 256);
    dim3 grid(1, 3);
    rms_norm_f32in_f32w_n3_kernel<<<grid, threads, threads * sizeof(float), stream>>>(
        x_a, x_b, x_c, weight, out_a, out_b, out_c, hidden_size, eps);
}

// FP32 residual add: hidden_f32 += proj_out (half)
__global__ void add_kernel_f32(float* __restrict__ x, const half* __restrict__ residual, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = x[idx] + __half2float(residual[idx]);
    }
}

// Fused N=2 residual add for spec decoding. Two non-contiguous (x_a/x_b)
// fp32 hidden buffers and their fp16 residuals processed in one launch.
// blockIdx.y selects the token.
__global__ void add_kernel_f32_n2(
    float* __restrict__ x_a, const half* __restrict__ res_a,
    float* __restrict__ x_b, const half* __restrict__ res_b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (blockIdx.y == 0) {
        x_a[idx] = x_a[idx] + __half2float(res_a[idx]);
    } else {
        x_b[idx] = x_b[idx] + __half2float(res_b[idx]);
    }
}

// N=3 fused residual add for MTP K=2 spec decoding.
__global__ void add_kernel_f32_n3(
    float* __restrict__ x_a, const half* __restrict__ res_a,
    float* __restrict__ x_b, const half* __restrict__ res_b,
    float* __restrict__ x_c, const half* __restrict__ res_c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float* x = (blockIdx.y == 0) ? x_a : (blockIdx.y == 1) ? x_b : x_c;
    const half* r = (blockIdx.y == 0) ? res_a : (blockIdx.y == 1) ? res_b : res_c;
    x[idx] = x[idx] + __half2float(r[idx]);
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

// N=2 fused SiLU * mul for spec decoding
__global__ void silu_mul_n2_kernel(
    half* __restrict__ a0, const half* __restrict__ b0,
    half* __restrict__ a1, const half* __restrict__ b1,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    half* a = (blockIdx.y == 0) ? a0 : a1;
    const half* b = (blockIdx.y == 0) ? b0 : b1;
    float va = __half2float(a[idx]);
    float silu = va / (1.0f + expf(-va));
    a[idx] = __float2half(silu * __half2float(b[idx]));
}

// N=3 fused SiLU * mul (MTP K=2 three-stream batch).
__global__ void silu_mul_n3_kernel(
    half* __restrict__ a0, const half* __restrict__ b0,
    half* __restrict__ a1, const half* __restrict__ b1,
    half* __restrict__ a2, const half* __restrict__ b2,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    half* a = (blockIdx.y == 0) ? a0 : (blockIdx.y == 1) ? a1 : a2;
    const half* b = (blockIdx.y == 0) ? b0 : (blockIdx.y == 1) ? b1 : b2;
    float va = __half2float(a[idx]);
    float silu = va / (1.0f + expf(-va));
    a[idx] = __float2half(silu * __half2float(b[idx]));
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

// Q8_0: 32 elements per block, int8 quants + half scale
// Layout matches block_q8_0_aligned in quant_gemv.cuh (qs first, then pad+d)
// because gpu_loader.h repacks all on-disk Q8_0 tensors at load time.
__global__ void dequant_embd_q8_0_row(
    const void* __restrict__ embd, half* __restrict__ out, int token_id, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H) return;
    struct __align__(4) bq8_0 { int8_t qs[32]; uint16_t pad; half d; };
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

// Single-block argmax over an fp16 array of length N. Writes the winning
// index to out_idx[0]. Used by the temp=0 greedy sampling fast path so we
// don't have to copy the full V*2 byte logits buffer over PCIe each token —
// instead just copy the 4-byte int. With V≈248 K, this saves ~2 ms/token
// on PCIe 1.0 x1 hardware.
//
// Designed to be launched with 1024 threads, 1 block. Each thread scans a
// strided slice of N values. Block reduces via shared memory.
__global__ void argmax_half_kernel(const half* __restrict__ logits, int N, int* __restrict__ out_idx) {
    __shared__ float s_val[1024];
    __shared__ int   s_idx[1024];

    float my_max = -1e38f;
    int   my_arg = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = __half2float(logits[i]);
        if (v > my_max) { my_max = v; my_arg = i; }
    }
    s_val[threadIdx.x] = my_max;
    s_idx[threadIdx.x] = my_arg;
    __syncthreads();

    // Block reduction (assumes blockDim.x is a power of two ≤ 1024)
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float a = s_val[threadIdx.x];
            float b = s_val[threadIdx.x + stride];
            if (b > a) {
                s_val[threadIdx.x] = b;
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out_idx[0] = s_idx[0];
}

// Top-2 argmax: out_top2[0]=argmax, out_top2[1]=2nd-largest index.
// Each thread scans strided slice keeping (m1,a1)(m2,a2). Block merges
// per-thread top-2 pairs into the block top-2 via O(N log N) reduction.
__global__ void argmax_top2_half_kernel(const half* __restrict__ logits,
                                        int N, int* __restrict__ out_top2) {
    __shared__ float s_v1[1024]; __shared__ int s_i1[1024];
    __shared__ float s_v2[1024]; __shared__ int s_i2[1024];

    float m1 = -1e38f; int a1 = 0;
    float m2 = -1e38f; int a2 = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = __half2float(logits[i]);
        if (v > m1)      { m2 = m1; a2 = a1; m1 = v; a1 = i; }
        else if (v > m2) { m2 = v;  a2 = i; }
    }
    s_v1[threadIdx.x] = m1; s_i1[threadIdx.x] = a1;
    s_v2[threadIdx.x] = m2; s_i2[threadIdx.x] = a2;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float aL = s_v1[threadIdx.x],       aR  = s_v1[threadIdx.x + stride];
            float aL2 = s_v2[threadIdx.x],      aR2 = s_v2[threadIdx.x + stride];
            int   iL = s_i1[threadIdx.x],       iR  = s_i1[threadIdx.x + stride];
            int   iL2 = s_i2[threadIdx.x],      iR2 = s_i2[threadIdx.x + stride];
            // Block top-1/top-2 = two largest of the four {aL, aL2, aR, aR2}.
            float top1, top2; int top1_i, top2_i;
            if (aL >= aR) {
                top1 = aL; top1_i = iL;
                // top2 = max(aR, aL2) (aR2 <= aR)
                if (aR >= aL2) { top2 = aR;  top2_i = iR;  }
                else           { top2 = aL2; top2_i = iL2; }
            } else {
                top1 = aR; top1_i = iR;
                if (aL >= aR2) { top2 = aL;  top2_i = iL;  }
                else           { top2 = aR2; top2_i = iR2; }
            }
            s_v1[threadIdx.x] = top1; s_i1[threadIdx.x] = top1_i;
            s_v2[threadIdx.x] = top2; s_i2[threadIdx.x] = top2_i;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_top2[0] = s_i1[0];
        out_top2[1] = s_i2[0];
    }
}

// Top-K argmax: out_ids[0..K-1] sorted descending by logit. out_logits[k]
// receives the raw logit (fp32) for that id, or may be nullptr. Used by
// DDTree tree-draft construction (MTP head top-K expansion).
template<int K>
__global__ void argmax_topk_half_kernel(const half* __restrict__ logits,
                                         int N,
                                         int* __restrict__ out_ids,
                                         float* __restrict__ out_logits) {
    // Per-thread sorted-descending top-K.
    float my_v[K]; int my_i[K];
    #pragma unroll
    for (int k = 0; k < K; k++) { my_v[k] = -1e38f; my_i[k] = 0; }

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = __half2float(logits[i]);
        if (v > my_v[K - 1]) {
            int pos = K - 1;
            while (pos > 0 && v > my_v[pos - 1]) {
                my_v[pos] = my_v[pos - 1];
                my_i[pos] = my_i[pos - 1];
                pos--;
            }
            my_v[pos] = v;
            my_i[pos] = i;
        }
    }

    // Block-wide tree reduction, merging top-K lists pair-wise.
    __shared__ float s_v[1024][K];
    __shared__ int   s_i[1024][K];
    #pragma unroll
    for (int k = 0; k < K; k++) {
        s_v[threadIdx.x][k] = my_v[k];
        s_i[threadIdx.x][k] = my_i[k];
    }
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float a_v[K]; int a_i[K];
            float b_v[K]; int b_i[K];
            #pragma unroll
            for (int k = 0; k < K; k++) {
                a_v[k] = s_v[threadIdx.x][k];           a_i[k] = s_i[threadIdx.x][k];
                b_v[k] = s_v[threadIdx.x + stride][k];  b_i[k] = s_i[threadIdx.x + stride][k];
            }
            float m_v[K]; int m_i[K];
            int ai = 0, bi = 0;
            #pragma unroll
            for (int k = 0; k < K; k++) {
                if (ai < K && (bi >= K || a_v[ai] >= b_v[bi])) {
                    m_v[k] = a_v[ai]; m_i[k] = a_i[ai]; ai++;
                } else {
                    m_v[k] = b_v[bi]; m_i[k] = b_i[bi]; bi++;
                }
            }
            #pragma unroll
            for (int k = 0; k < K; k++) {
                s_v[threadIdx.x][k] = m_v[k];
                s_i[threadIdx.x][k] = m_i[k];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int k = 0; k < K; k++) {
            out_ids[k] = s_i[0][k];
            if (out_logits) out_logits[k] = s_v[0][k];
        }
    }
}

template __global__ void argmax_topk_half_kernel<4>(const half*, int, int*, float*);
template __global__ void argmax_topk_half_kernel<8>(const half*, int, int*, float*);
