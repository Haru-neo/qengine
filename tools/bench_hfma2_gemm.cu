// Standalone microbench for HFMA2-based HGEMM.
// Goal: measure achievable TFLOPs of a custom HGEMM that uses __hfma2
// SIMD FMA on CMP. Compared against the 17 TFLOPs DP4A baseline from
// bench_mlp_gemm. If this comes in below 20 TFLOPs the HFMA2 path is
// not worth replacing DP4A (memory cost of fp16 weights would erase
// the compute win).
//
// Two kernels:
//   hgemm_fp16:         fp16 W × fp16 X, HFMA2 inner loop, fp32 accum.
//                        Upper bound — ignores dequant cost.
//   hgemm_q8_to_hfma2:   Q8_0 W × fp16 X, on-fly dequant + HFMA2.
//                        Realistic path for actual integration.
//
// Build:
//   nvcc -O3 -arch=sm_70 -std=c++17 tools/bench_hfma2_gemm.cu \
//        -o tools/bench_hfma2_gemm
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#include "../src/gguf.h"

typedef struct __align__(4) { int8_t qs[32]; uint16_t pad; half d; } block_q8_0_aligned;

static void check(cudaError_t e, const char* tag) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", tag, cudaGetErrorString(e));
        exit(1);
    }
}

// ============================================================
// Kernel 1: pure HGEMM fp16 × fp16 → fp32 accum → fp16 out.
// This is the HFMA2 compute ceiling for this shape. Uses the same
// thread-tile structure as gemm_q8_0_q8_1_v2 so the comparison is apples-to-apples.
// ============================================================
template<int BM, int BN, int TM, int TN, int BK>
__global__ void hgemm_hfma2_v1(
    const half* __restrict__ W,  // [M][K] row-major (W[m*K + k])
    const half* __restrict__ X,  // [N][K] row-major (X[n*K + k])
    half* __restrict__ Y,        // [N][M] column-major-ish, Y[n*M + m]
    const int M, const int N, const int K)
{
    constexpr int ROWS = BM / TM;
    constexpr int COLS = BN / TN;
    constexpr int THREADS = ROWS * COLS;
    const int tid = threadIdx.x;
    const int tx = tid % ROWS;  // row within block tile
    const int ty = tid / ROWS;  // col within block tile

    // fp32 accumulators
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    // SMEM tiles in half2. BK is in half2 units (so 2*BK elements per K-step).
    extern __shared__ __align__(16) half smem_raw[];
    half2* sW = (half2*)smem_raw;                 // [BM][BK] half2
    half2* sX = (half2*)(sW + BM * BK);           // [BN][BK] half2

    const int K_half2 = K / 2;
    for (int k0 = 0; k0 < K_half2; k0 += BK) {
        // Cooperative load: all threads load BM*BK + BN*BK half2.
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            int m_in = idx / BK;
            int b_in = idx - m_in * BK;
            int m_g = blockIdx.x * BM + m_in;
            int k_g = k0 + b_in;
            if (m_g < M && k_g < K_half2) {
                sW[idx] = ((const half2*)W)[(size_t)m_g * K_half2 + k_g];
            } else {
                sW[idx] = __floats2half2_rn(0.0f, 0.0f);
            }
        }
        for (int idx = tid; idx < BN * BK; idx += THREADS) {
            int n_in = idx / BK;
            int b_in = idx - n_in * BK;
            int n_g = blockIdx.y * BN + n_in;
            int k_g = k0 + b_in;
            if (n_g < N && k_g < K_half2) {
                sX[idx] = ((const half2*)X)[(size_t)n_g * K_half2 + k_g];
            } else {
                sX[idx] = __floats2half2_rn(0.0f, 0.0f);
            }
        }
        __syncthreads();

        // Inner tile: each thread computes TM×TN output cells.
        // For each K-half2, load TM weight half2 and TN input half2, then
        // do TM×TN __hfma2 calls (into half2 accumulators). Periodically
        // flush to fp32 to avoid fp16 drift over long K.
        half2 h_acc[TM][TN];
        #pragma unroll
        for (int i = 0; i < TM; i++)
            #pragma unroll
            for (int j = 0; j < TN; j++) h_acc[i][j] = __floats2half2_rn(0.0f, 0.0f);

        #pragma unroll
        for (int b = 0; b < BK; b++) {
            half2 w_frag[TM], x_frag[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                w_frag[i] = sW[(tx * TM + i) * BK + b];
            }
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                x_frag[j] = sX[(ty * TN + j) * BK + b];
            }
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    h_acc[i][j] = __hfma2(w_frag[i], x_frag[j], h_acc[i][j]);
                }
            }
        }
        // Flush half2 accum into fp32 accum each K-tile (BK half2 = 2*BK elements).
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                float2 f = __half22float2(h_acc[i][j]);
                acc[i][j] += f.x + f.y;
            }
        }
        __syncthreads();
    }

    // Write back
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int m_g = blockIdx.x * BM + tx * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int n_g = blockIdx.y * BN + ty * TN + j;
            if (m_g < M && n_g < N) {
                Y[(size_t)n_g * M + m_g] = __float2half(acc[i][j]);
            }
        }
    }
}

// ============================================================
// Kernel 2: Q8_0 W × fp16 X → fp32 accum → fp16 out.
// Weight is Q8_0 (32 int8 + half scale per block), dequantized on-fly
// into half2 via (int8 * scale). This is the realistic integration
// path — matches the engine's weight layout with no extra memory cost.
// ============================================================
template<int BM, int BN, int TM, int TN, int BK_BLOCKS>
__global__ void hgemm_q8_to_hfma2(
    const block_q8_0_aligned* __restrict__ W,  // [M][K/32] q8_0 blocks
    const half* __restrict__ X,                 // [N][K] fp16
    half* __restrict__ Y,                       // [N][M] output
    const int M, const int N, const int K)
{
    constexpr int ROWS = BM / TM;
    constexpr int COLS = BN / TN;
    constexpr int THREADS = ROWS * COLS;
    const int tid = threadIdx.x;
    const int tx = tid % ROWS;
    const int ty = tid / ROWS;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    // SMEM: weight Q8_0 blocks (36B each) + input fp16.
    // BK_BLOCKS = number of Q8_0 blocks (32 elements each) per K-tile.
    extern __shared__ __align__(16) char smem_raw2[];
    block_q8_0_aligned* sW = (block_q8_0_aligned*)smem_raw2;
    half* sX_fp16 = (half*)(sW + BM * BK_BLOCKS);

    const int bpr = K / 32;
    const int k_per_tile = BK_BLOCKS * 32;   // elements per K-tile

    for (int kb = 0; kb < bpr; kb += BK_BLOCKS) {
        int k0 = kb * 32;
        // Load weight blocks.
        for (int idx = tid; idx < BM * BK_BLOCKS; idx += THREADS) {
            int m_in = idx / BK_BLOCKS;
            int b_in = idx - m_in * BK_BLOCKS;
            int m_g = blockIdx.x * BM + m_in;
            int b_g = kb + b_in;
            if (m_g < M && b_g < bpr) {
                sW[idx] = W[(size_t)m_g * bpr + b_g];
            }
        }
        // Load input fp16 (BN rows × k_per_tile elements).
        int x_total_half2 = BN * k_per_tile / 2;
        half2* sX_h2 = (half2*)sX_fp16;
        const half2* X_h2 = (const half2*)X;
        int K_h2 = K / 2;
        for (int idx = tid; idx < x_total_half2; idx += THREADS) {
            int n_in = idx / (k_per_tile / 2);
            int b_in = idx - n_in * (k_per_tile / 2);
            int n_g = blockIdx.y * BN + n_in;
            int k_g = (k0 + b_in * 2) / 2;
            if (n_g < N && k_g < K_h2) {
                sX_h2[idx] = X_h2[(size_t)n_g * K_h2 + k_g];
            } else {
                sX_h2[idx] = __floats2half2_rn(0.0f, 0.0f);
            }
        }
        __syncthreads();

        // Compute.
        #pragma unroll
        for (int b = 0; b < BK_BLOCKS; b++) {
            // Dequant TM weight blocks on-fly. Each block = 32 int8 → 16 half2.
            // Scale factor = half d; we lift to half2 via broadcast.
            // Cache weight half2 in regs.
            half2 w_frag[TM][16];
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                int w_row_in = tx * TM + i;
                const block_q8_0_aligned* wb = &sW[w_row_in * BK_BLOCKS + b];
                half d = wb->d;
                half2 d2 = __half2half2(d);
                #pragma unroll
                for (int hh = 0; hh < 16; hh++) {
                    int8_t a = wb->qs[hh * 2];
                    int8_t bq = wb->qs[hh * 2 + 1];
                    half2 q = __floats2half2_rn((float)a, (float)bq);
                    w_frag[i][hh] = __hmul2(q, d2);
                }
            }
            // Cache TN input half2 blocks.
            half2 x_frag[TN][16];
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                int n_row_in = ty * TN + j;
                const half2* xp = (const half2*)(sX_fp16 + n_row_in * k_per_tile + b * 32);
                #pragma unroll
                for (int hh = 0; hh < 16; hh++) {
                    x_frag[j][hh] = xp[hh];
                }
            }
            // Inner MAC — each thread does TM*TN*16 HFMA2 = TM*TN*32 MACs.
            half2 h_acc[TM][TN];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) h_acc[i][j] = __floats2half2_rn(0.0f, 0.0f);

            #pragma unroll
            for (int hh = 0; hh < 16; hh++) {
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    half2 wj = w_frag[i][hh];
                    #pragma unroll
                    for (int j = 0; j < TN; j++) {
                        h_acc[i][j] = __hfma2(wj, x_frag[j][hh], h_acc[i][j]);
                    }
                }
            }
            // Promote once per block to fp32.
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    float2 f = __half22float2(h_acc[i][j]);
                    acc[i][j] += f.x + f.y;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int m_g = blockIdx.x * BM + tx * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int n_g = blockIdx.y * BN + ty * TN + j;
            if (m_g < M && n_g < N) {
                Y[(size_t)n_g * M + m_g] = __float2half(acc[i][j]);
            }
        }
    }
}

// ============================================================
// Init helpers.
// ============================================================
static void init_fp16(half* h, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        float v = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        h[i] = __float2half(v * 0.3f);
    }
}
static void init_q8_0(block_q8_0_aligned* h, size_t n_blocks, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n_blocks; i++) {
        for (int j = 0; j < 32; j++) h[i].qs[j] = (int8_t)((rand() & 0xff) - 128);
        h[i].pad = 0;
        float s = 0.005f + 0.01f * ((float)rand() / RAND_MAX);
        h[i].d = __float2half(s);
    }
}

// ============================================================
// Driver.
// ============================================================
int main(int argc, char** argv) {
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s  SMs=%d  clock=%.2f GHz\n\n",
           prop.name, prop.multiProcessorCount, prop.clockRate / 1e6);

    struct Shape { const char* lbl; int K, M; };
    std::vector<Shape> shapes = {
        {"gate/up  K=5120  M=17408",  5120, 17408},
        {"down     K=17408 M=5120",  17408,  5120},
    };
    std::vector<int> Ns = {256, 1024, 2048};

    const int K_MAX = 17408, M_MAX = 17408, N_MAX = 2048;
    size_t W_q8_bytes = (size_t)M_MAX * (K_MAX / 32) * sizeof(block_q8_0_aligned);
    size_t W_h_bytes  = (size_t)M_MAX * K_MAX * sizeof(half);
    size_t X_bytes    = (size_t)N_MAX * K_MAX * sizeof(half);
    size_t Y_bytes    = (size_t)N_MAX * M_MAX * sizeof(half);

    block_q8_0_aligned* h_Wq = (block_q8_0_aligned*)malloc(W_q8_bytes);
    half* h_Wh = (half*)malloc(W_h_bytes);
    half* h_X  = (half*)malloc(X_bytes);
    init_q8_0(h_Wq, (size_t)M_MAX * (K_MAX / 32), 42);
    init_fp16(h_Wh, (size_t)M_MAX * K_MAX, 43);
    init_fp16(h_X, (size_t)N_MAX * K_MAX, 1337);

    void* dWq; check(cudaMalloc(&dWq, W_q8_bytes), "dWq");
    half* dWh; check(cudaMalloc(&dWh, W_h_bytes),  "dWh");
    half* dX;  check(cudaMalloc(&dX, X_bytes),     "dX");
    half* dY;  check(cudaMalloc(&dY, Y_bytes),     "dY");
    cudaMemcpy(dWq, h_Wq, W_q8_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dWh, h_Wh, W_h_bytes,  cudaMemcpyHostToDevice);
    cudaMemcpy(dX, h_X, X_bytes, cudaMemcpyHostToDevice);
    free(h_Wq); free(h_Wh); free(h_X);

    auto bench = [&](auto launcher, const char* tag, int K, int M, int N, int iters) -> double {
        // warmup
        for (int w = 0; w < 3; w++) launcher();
        cudaDeviceSynchronize();
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int i = 0; i < iters; i++) launcher();
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        double ms_per = ms / iters;
        double flops = 2.0 * M * N * K;
        double tf = flops / (ms_per * 1e-3) / 1e12;
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("%-30s kernel error: %s\n", tag, cudaGetErrorString(err));
            return -1;
        }
        printf("%-30s K=%-6d M=%-6d N=%-5d  %.3f ms  %.2f TFLOPs\n",
               tag, K, M, N, ms_per, tf);
        return tf;
    };

    // Kernel 1: pure HGEMM fp16.
    printf("=== Kernel 1: pure HGEMM (HFMA2 ceiling, no dequant) ===\n");
    for (auto& sh : shapes) {
        for (int N : Ns) {
            constexpr int BM = 64, BN = 32, TM = 4, TN = 4, BK = 16;  // BK in half2 units
            dim3 grid((sh.M + BM - 1)/BM, (N + BN - 1)/BN);
            int threads = (BM/TM) * (BN/TN);
            int sm_bytes = (BM + BN) * BK * sizeof(half2);
            int iters = (N <= 256) ? 60 : 20;
            bench([&]() {
                hgemm_hfma2_v1<BM, BN, TM, TN, BK><<<grid, threads, sm_bytes>>>(
                    dWh, dX, dY, sh.M, N, sh.K);
            }, "pure-HFMA2 64x32 4x4 BK16", sh.K, sh.M, N, iters);
        }
        printf("\n");
    }

    // Kernel 2: Q8_0 -> HFMA2.
    printf("=== Kernel 2: Q8_0 dequant + HFMA2 (integration path) ===\n");
    for (auto& sh : shapes) {
        for (int N : Ns) {
            constexpr int BM = 64, BN = 32, TM = 4, TN = 4, BK_BLOCKS = 4;
            dim3 grid((sh.M + BM - 1)/BM, (N + BN - 1)/BN);
            int threads = (BM/TM) * (BN/TN);
            int sm_bytes = BM * BK_BLOCKS * sizeof(block_q8_0_aligned)
                         + BN * BK_BLOCKS * 32 * sizeof(half);
            int iters = (N <= 256) ? 60 : 20;
            bench([&]() {
                hgemm_q8_to_hfma2<BM, BN, TM, TN, BK_BLOCKS><<<grid, threads, sm_bytes>>>(
                    (block_q8_0_aligned*)dWq, dX, dY, sh.M, N, sh.K);
            }, "Q8->HFMA2 64x32 4x4 BK=4", sh.K, sh.M, N, iters);
        }
        printf("\n");
    }

    cudaFree(dWq); cudaFree(dWh); cudaFree(dX); cudaFree(dY);
    return 0;
}
