// Q8_0 multi-token GEMM helpers for inference.
//
// Reuses the kernels already defined in quant_gemv.cuh:
//   - block_q8_0_aligned, block_q8_1 structs
//   - quantize_input_q8_1 (fp16 → q8_1, per 32-block)
//   - gemm_q8_0_q8_1<BM,BN,BK_BLOCKS> (Wq × Xq8_1 → Y fp16)
// Adds:
//   - quantize_weight_q8_0 kernel (fp16 → block_q8_0_aligned, row-wise)
//   - launchers (raw pointer, no PyTorch dependency) for the dflash draft path

#pragma once
#include "quant_gemv.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace q8gemm {

// ── Quantize fp16 weight → Q8_0 (row-wise, per 32-block) ───────────────────
__global__ void quantize_weight_q8_0_kern(
    const half* __restrict__ W,
    block_q8_0_aligned* __restrict__ Wq,
    int M, int K
) {
    int m = blockIdx.x;
    int b = blockIdx.y * blockDim.x + threadIdx.x;
    int bpr = K / 32;
    if (m >= M || b >= bpr) return;

    const half* xb = W + (size_t)m * K + b * 32;
    float vals[32], amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        vals[i] = __half2float(xb[i]);
        amax = fmaxf(amax, fabsf(vals[i]));
    }
    float d  = amax / 127.0f;
    float id = (d > 0.0f) ? 127.0f / amax : 0.0f;

    block_q8_0_aligned* ob = &Wq[(size_t)m * bpr + b];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        ob->qs[i] = (int8_t)__float2int_rn(vals[i] * id);
    }
    ob->pad = 0;
    ob->d   = __float2half(d);
}

// ── Launchers ──────────────────────────────────────────────────────────────

inline void launch_quantize_weight_q8_0(
    const half* W_fp16, block_q8_0_aligned* Wq, int M, int K, cudaStream_t stream = 0
) {
    int bpr = K / 32;
    dim3 grid(M, (bpr + 255) / 256);
    quantize_weight_q8_0_kern<<<grid, 256, 0, stream>>>(W_fp16, Wq, M, K);
}

// X[N,K] fp16 → block_q8_1 (per row × bpr blocks). X_q size = N*bpr blocks.
// Note: quant_gemv.cuh's quantize_input_q8_1 takes K as total-K (N*K), one block per
// blockIdx.x*blockDim.x+tid. We launch enough threads to cover all blocks.
inline void launch_quantize_input_q8_1(
    const half* X_fp16, block_q8_1* X_q, int N, int K, cudaStream_t stream = 0
) {
    int bpr = K / 32;
    int total = N * bpr;
    int threads = 64;
    int blocks  = (total + threads - 1) / threads;
    quantize_input_q8_1<<<blocks, threads, 0, stream>>>(X_fp16, X_q, N * K);
}

// Y[N,M] = X[N,K] @ Wq[M,K]^T (token-major). X must already be q8_1.
inline void launch_gemm_q8_0_q8_1(
    const block_q8_0_aligned* Wq,
    const block_q8_1* X_q,
    half* Y, int M, int N, int K,
    cudaStream_t stream = 0
) {
    if (N >= 32) {
        constexpr int BM = 16, BN = 32, BK_BLOCKS = 4;
        size_t smem = BM * BK_BLOCKS * sizeof(block_q8_0_aligned)
                    + BN * BK_BLOCKS * sizeof(block_q8_1);
        dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
        ::gemm_q8_0_q8_1<BM, BN, BK_BLOCKS><<<grid, BM * BN, smem, stream>>>(
            Wq, X_q, Y, M, N, K);
    } else {
        constexpr int BM = 16, BN = 16, BK_BLOCKS = 8;
        size_t smem = BM * BK_BLOCKS * sizeof(block_q8_0_aligned)
                    + BN * BK_BLOCKS * sizeof(block_q8_1);
        dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
        ::gemm_q8_0_q8_1<BM, BN, BK_BLOCKS><<<grid, BM * BN, smem, stream>>>(
            Wq, X_q, Y, M, N, K);
    }
}

} // namespace q8gemm
