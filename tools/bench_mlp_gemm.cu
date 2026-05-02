// Standalone microbench for MLP GEMM paths.
// Measures DP4A-based gemm_q8_0_q8_1_v2 and gemv_q8_0_q8 throughput on the
// CMP 100-210 so we know the true TFLOPs ceiling before attempting an HFMA2
// replacement path.
//
// Build:
//   nvcc -O3 -arch=sm_70 -std=c++17 -Xcompiler -fopenmp \
//        tools/bench_mlp_gemm.cu -o tools/bench_mlp_gemm
//
// Run:
//   ./tools/bench_mlp_gemm           # default shape sweep
//   CUDA_VISIBLE_DEVICES=0 ./tools/bench_mlp_gemm
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

#include "../src/gguf.h"
#include "../src/quant_gemv.cuh"

static void check(cudaError_t e, const char* tag) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", tag, cudaGetErrorString(e));
        exit(1);
    }
}

// Fill a block_q8_0_aligned buffer with random int8 weights + random fp16 scales.
// Layout matches what the runtime has after repack_q8_0 — qs at offset 0, half d at offset 34.
static void init_q8_0_weight(block_q8_0_aligned* h, size_t n_blocks, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n_blocks; i++) {
        for (int j = 0; j < 32; j++) h[i].qs[j] = (int8_t)((rand() & 0xff) - 128);
        h[i].pad = 0;
        float s = 0.005f + 0.01f * ((float)rand() / RAND_MAX);
        h[i].d = __float2half(s);
    }
}

static void init_fp16(half* h, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        float v = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        h[i] = __float2half(v * 0.5f);
    }
}

struct Shape {
    const char* label;
    int K, M;     // GEMM shape: (M x K) @ (K x N) = (M x N), output row-major M-major
    int N;        // batch (chunk size)
};

static double bench_gemm(void* dW, const block_q8_1* dX, half* dY,
                         int K, int M, int N, int iters) {
    const int v2_mode = 9;  // default winner: 32x128 2x8 BK=4
    // First warm.
    for (int w = 0; w < 3; w++) {
        quant_gemv_chunk(dW, GGML_TYPE_Q8_0, dX, dY, K, M, N);
    }
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) {
        quant_gemv_chunk(dW, GGML_TYPE_Q8_0, dX, dY, K, M, N);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    (void)v2_mode;
    return ms / iters;
}

static double bench_gemv(void* dW, const block_q8_1* dX, half* dY,
                         int K, int M, int iters) {
    QuantInput qi;  // not used here — we pass the q8_1 buffer directly.
    (void)qi;
    const int threads = (K >= 8192) ? 256 : 128;
    for (int w = 0; w < 3; w++) {
        gemv_q8_0_q8<<<M, threads>>>(dW, dX, dY, K, M);
    }
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; i++) {
        gemv_q8_0_q8<<<M, threads>>>(dW, dX, dY, K, M);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / iters;
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("GPU: %s  SMs=%d  clock=%.2f GHz  mem=%.1f GB/s\n",
           prop.name, prop.multiProcessorCount, prop.clockRate / 1e6,
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1e6);
    printf("\n");

    // Shapes from Qwopus3.6-27B MLP.
    // gate_proj / up_proj:  Y[N,I] = X[N,H] @ W[I,H]^T    -> K=H=5120, M=I=17408
    // down_proj:            Y[N,H] = X[N,I] @ W[H,I]^T    -> K=I=17408, M=H=5120
    std::vector<Shape> shapes = {
        {"gate/up  K=5120  M=17408",   5120, 17408, 0},
        {"down     K=17408 M=5120",  17408,  5120, 0},
    };
    std::vector<int> chunks = {1, 16, 64, 256, 1024, 2048};

    // Allocate max-size buffers once, reuse.
    const int K_MAX = 17408;
    const int M_MAX = 17408;
    const int N_MAX = 2048;

    size_t w_bytes = (size_t)M_MAX * (K_MAX / 32) * sizeof(block_q8_0_aligned);
    size_t x_bytes = (size_t)N_MAX * (K_MAX / 32) * sizeof(block_q8_1);
    size_t y_bytes = (size_t)M_MAX * N_MAX * sizeof(half);
    size_t xf_bytes = (size_t)N_MAX * K_MAX * sizeof(half);

    block_q8_0_aligned* h_W = (block_q8_0_aligned*)malloc(w_bytes);
    half* h_Xf = (half*)malloc(xf_bytes);
    init_q8_0_weight(h_W, (size_t)M_MAX * (K_MAX / 32), 42);
    init_fp16(h_Xf, (size_t)N_MAX * K_MAX, 1337);

    void* dW; check(cudaMalloc(&dW, w_bytes), "dW");
    half* dXf; check(cudaMalloc(&dXf, xf_bytes), "dXf");
    block_q8_1* dX; check(cudaMalloc(&dX, x_bytes), "dX");
    half* dY; check(cudaMalloc(&dY, y_bytes), "dY");

    check(cudaMemcpy(dW, h_W, w_bytes, cudaMemcpyHostToDevice), "cp W");
    check(cudaMemcpy(dXf, h_Xf, xf_bytes, cudaMemcpyHostToDevice), "cp Xf");
    free(h_W); free(h_Xf);

    // Pre-quantize input once per K.
    auto quantize_x = [&](int K, int N) {
        int total_K = N * K;
        quantize_input_q8_1<<<(total_K / QK8 + 63) / 64, 64>>>(dXf, dX, total_K);
    };

    printf("%-28s %-6s %-10s %-9s %-8s\n",
           "shape", "N", "time(ms)", "TFLOPs", "GB/s_w");
    printf("--------------------------------------------------------------\n");

    for (auto& sh : shapes) {
        for (int N : chunks) {
            sh.N = N;
            quantize_x(sh.K, N);
            cudaDeviceSynchronize();

            int iters = (N <= 16) ? 200 : (N <= 256) ? 80 : 30;
            double ms;
            if (N == 1) {
                ms = bench_gemv(dW, dX, dY, sh.K, sh.M, iters);
            } else {
                ms = bench_gemm(dW, dX, dY, sh.K, sh.M, N, iters);
            }
            double flops = 2.0 * (double)sh.M * N * sh.K;
            double tflops = flops / (ms * 1e-3) / 1e12;
            // Weight bytes read (approx): M * K / 32 * 36 (one pass).
            double w_bytes_read = (double)sh.M * sh.K / 32.0 * 36.0;
            double gbs = w_bytes_read / (ms * 1e-3) / 1e9;
            printf("%-28s %-6d %-10.3f %-9.2f %-8.1f\n",
                   sh.label, N, ms, tflops, gbs);
        }
        printf("\n");
    }

    cudaFree(dW); cudaFree(dX); cudaFree(dXf); cudaFree(dY);
    return 0;
}
