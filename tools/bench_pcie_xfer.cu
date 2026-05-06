// Cross-GPU host-pinned-bridge xfer microbench.
//
// CMP 100-210 has no P2P, so the runtime moves hidden state between GPUs via
// a pinned host buffer (D2H on src, H2D on dst). We measure the achieved
// throughput at three sizes:
//
//   - 16 KB     : single-token gen-time hop (4096 fp32 = 16KB, no chunking)
//   - 1 MB      : chunked prefill chunk (256 tok × 4096 × 4B = 4 MB per chunk;
//                 we test 1MB as ramp + 4MB)
//   - 4 MB      : full prefill chunk
//
// Patterns:
//   - PINNED H2D (single direction)
//   - PINNED D2H (single direction)
//   - ROUNDTRIP D2H+H2D (the actual xfer pattern between GPU N → host → GPU N+1)
//
// Build:
//   nvcc -O3 -arch=sm_70 -std=c++17 tools/bench_pcie_xfer.cu \
//        -o tools/bench_pcie_xfer

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

static void check(cudaError_t e, const char* tag) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", tag, cudaGetErrorString(e));
        exit(1);
    }
}

int main() {
    int n_dev = 0; cudaGetDeviceCount(&n_dev);
    fprintf(stderr, "Found %d devices\n", n_dev);
    if (n_dev < 2) { fprintf(stderr, "need >=2 GPUs\n"); return 1; }

    std::vector<size_t> sizes = {
        16ULL * 1024,           // single-token (4096 fp32)
        1ULL  * 1024 * 1024,    // 1 MB
        4ULL  * 1024 * 1024,    // 4 MB (256 tok chunk × 4096 × 4B)
        16ULL * 1024 * 1024,    // 16 MB (1024 tok chunk × 4096 × 4B)
        64ULL * 1024 * 1024,    // 64 MB
    };
    size_t maxbytes = sizes.back();

    // Allocate buffers
    cudaSetDevice(0);
    void* dA; check(cudaMalloc(&dA, maxbytes), "dA");
    cudaSetDevice(1);
    void* dB; check(cudaMalloc(&dB, maxbytes), "dB");
    void* hPinned; check(cudaMallocHost(&hPinned, maxbytes), "pinned");

    cudaStream_t s0, s1;
    cudaSetDevice(0); cudaStreamCreate(&s0);
    cudaSetDevice(1); cudaStreamCreate(&s1);

    int repeats = 20;
    cudaEvent_t a, b;

    printf("\nGPU0 ↔ pinned host ↔ GPU1\n");
    printf("size       D2H_GPU0_GBs    H2D_GPU1_GBs    ROUNDTRIP_GBs   ROUNDTRIP_ms\n");
    printf("---------------------------------------------------------------------\n");

    for (size_t bytes : sizes) {
        // ---------- D2H from GPU 0 ----------
        cudaSetDevice(0);
        cudaEventCreate(&a); cudaEventCreate(&b);
        for (int i = 0; i < 3; ++i) cudaMemcpyAsync(hPinned, dA, bytes, cudaMemcpyDeviceToHost, s0);
        cudaStreamSynchronize(s0);
        cudaEventRecord(a, s0);
        for (int i = 0; i < repeats; ++i) cudaMemcpyAsync(hPinned, dA, bytes, cudaMemcpyDeviceToHost, s0);
        cudaEventRecord(b, s0);
        cudaEventSynchronize(b);
        float ms_d2h; cudaEventElapsedTime(&ms_d2h, a, b);
        cudaEventDestroy(a); cudaEventDestroy(b);
        double d2h_gbs = (double)bytes * repeats / (ms_d2h / 1000.0) / 1e9;

        // ---------- H2D to GPU 1 ----------
        cudaSetDevice(1);
        cudaEventCreate(&a); cudaEventCreate(&b);
        for (int i = 0; i < 3; ++i) cudaMemcpyAsync(dB, hPinned, bytes, cudaMemcpyHostToDevice, s1);
        cudaStreamSynchronize(s1);
        cudaEventRecord(a, s1);
        for (int i = 0; i < repeats; ++i) cudaMemcpyAsync(dB, hPinned, bytes, cudaMemcpyHostToDevice, s1);
        cudaEventRecord(b, s1);
        cudaEventSynchronize(b);
        float ms_h2d; cudaEventElapsedTime(&ms_h2d, a, b);
        cudaEventDestroy(a); cudaEventDestroy(b);
        double h2d_gbs = (double)bytes * repeats / (ms_h2d / 1000.0) / 1e9;

        // ---------- ROUNDTRIP (D2H GPU0 then H2D GPU1, serialized as runtime does) ----------
        // Time wall-clock for the pair (sync-fenced as "host fence" memo says).
        cudaSetDevice(0); cudaDeviceSynchronize();
        auto t_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repeats; ++i) {
            cudaSetDevice(0);
            cudaMemcpyAsync(hPinned, dA, bytes, cudaMemcpyDeviceToHost, s0);
            cudaStreamSynchronize(s0);
            cudaSetDevice(1);
            cudaMemcpyAsync(dB, hPinned, bytes, cudaMemcpyHostToDevice, s1);
            cudaStreamSynchronize(s1);
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        double ms_rt = std::chrono::duration<double, std::milli>(t_end - t_start).count() / repeats;
        double rt_gbs = (double)bytes / (ms_rt / 1000.0) / 1e9;

        const char* label;
        char buf[32];
        if (bytes < 1024*1024) snprintf(buf, sizeof(buf), "%4zu KB", bytes / 1024);
        else snprintf(buf, sizeof(buf), "%4zu MB", bytes / (1024*1024));
        label = buf;

        printf("%-10s %10.2f       %10.2f       %10.2f      %8.3f\n",
               label, d2h_gbs, h2d_gbs, rt_gbs, ms_rt);
    }

    cudaFree(dA);
    cudaSetDevice(1);
    cudaFree(dB);
    cudaFreeHost(hPinned);
    return 0;
}
