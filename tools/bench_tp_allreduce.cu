// Bench the cost of a 3-GPU all-reduce of the verify hidden state, which
// tensor-parallel verify would require AFTER EACH of the 64 layers.
//
// TP plan: each layer's GEMVs are sharded across GPU0/1/2 (each computes 1/3
// of the output rows of every matmul), then the partial hidden states must be
// gathered/reduced so every GPU has the full hidden for the next layer.
//
// On CMP there is NO P2P. An all-reduce across 3 GPUs over a single pinned
// host buffer is: each GPU D2H its partial (host fence), host concatenates (or
// the partials are non-overlapping shards = pure gather, no add), then each GPU
// H2D the full vector. We measure the realistic wall-clock of one such
// all-gather for the verify hidden: budget=8 tokens x 5120 floats = 160KB.
//
// We also measure budget=1 (single-token root forward = 20KB) and the
// sum-reduce variant (if GEMVs are split along K instead of M, partials
// overlap and must be summed -> 3 full-size D2H + host add + broadcast H2D).
//
// Build:
//   nvcc -O3 -arch=sm_70 -std=c++17 tools/bench_tp_allreduce.cu -o tools/bench_tp_allreduce

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    if (n_dev < 3) { fprintf(stderr, "need >=3 GPUs\n"); return 1; }
    const int NG = 3;

    int H = 5120;
    std::vector<int> budgets = {1, 8, 16};
    int repeats = 50;

    // Per-GPU device buffers (full hidden) + pinned host staging (one per GPU
    // partial, plus one full).
    void* dFull[NG];
    void* hPart[NG];   // pinned, each GPU's partial shard (1/3 of H rows)
    void* hFull;       // pinned, the assembled full hidden
    size_t maxbytes = (size_t)budgets.back() * H * sizeof(float);
    for (int g = 0; g < NG; ++g) {
        cudaSetDevice(g);
        check(cudaMalloc(&dFull[g], maxbytes), "dFull");
        check(cudaMallocHost(&hPart[g], maxbytes), "hPart");
    }
    check(cudaMallocHost(&hFull, maxbytes), "hFull");

    printf("\n3-GPU all-GATHER (M-sharded GEMV: each GPU owns 1/3 output rows, no add)\n");
    printf("Pattern per layer: 3x[D2H partial + fence], host-noop, 3x[H2D full + fence]\n");
    printf("budget  bytes_total   gather_ms   x64layers_ms\n");
    printf("------------------------------------------------\n");
    for (int B : budgets) {
        size_t full_bytes = (size_t)B * H * sizeof(float);
        size_t part_bytes = full_bytes / NG;  // each GPU's shard

        // warmup
        for (int r = 0; r < 3; ++r) {
            for (int g = 0; g < NG; ++g) {
                cudaSetDevice(g);
                cudaMemcpy((char*)hPart[g], dFull[g], part_bytes, cudaMemcpyDeviceToHost);
            }
            // assemble (host memcpy of shards into hFull) - cheap, but count it
            for (int g = 0; g < NG; ++g)
                memcpy((char*)hFull + g*part_bytes, hPart[g], part_bytes);
            for (int g = 0; g < NG; ++g) {
                cudaSetDevice(g);
                cudaMemcpy(dFull[g], hFull, full_bytes, cudaMemcpyHostToDevice);
            }
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < repeats; ++r) {
            // each GPU pushes its 1/3 shard to host (host fence each, as memo requires)
            for (int g = 0; g < NG; ++g) {
                cudaSetDevice(g);
                cudaMemcpy((char*)hPart[g], dFull[g], part_bytes, cudaMemcpyDeviceToHost);
            }
            for (int g = 0; g < NG; ++g)
                memcpy((char*)hFull + g*part_bytes, hPart[g], part_bytes);
            // broadcast full hidden back to every GPU
            for (int g = 0; g < NG; ++g) {
                cudaSetDevice(g);
                cudaMemcpy(dFull[g], hFull, full_bytes, cudaMemcpyHostToDevice);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;
        printf("%4d    %9zu     %8.4f    %9.2f\n", B, full_bytes, ms, ms * 64);
    }

    printf("\n3-GPU all-REDUCE (K-sharded GEMV: partials overlap, must SUM)\n");
    printf("Pattern per layer: 3x[D2H full + fence], host-add, 3x[H2D full + fence]\n");
    printf("budget  bytes_total   reduce_ms   x64layers_ms\n");
    printf("------------------------------------------------\n");
    for (int B : budgets) {
        size_t full_bytes = (size_t)B * H * sizeof(float);
        int n_floats = B * H;
        // warmup
        for (int r = 0; r < 3; ++r) {
            for (int g = 0; g < NG; ++g) {
                cudaSetDevice(g);
                cudaMemcpy(hPart[g], dFull[g], full_bytes, cudaMemcpyDeviceToHost);
            }
            float* acc = (float*)hFull;
            memcpy(acc, hPart[0], full_bytes);
            for (int g = 1; g < NG; ++g) {
                float* p = (float*)hPart[g];
                for (int i = 0; i < n_floats; ++i) acc[i] += p[i];
            }
            for (int g = 0; g < NG; ++g) {
                cudaSetDevice(g);
                cudaMemcpy(dFull[g], hFull, full_bytes, cudaMemcpyHostToDevice);
            }
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < repeats; ++r) {
            for (int g = 0; g < NG; ++g) {
                cudaSetDevice(g);
                cudaMemcpy(hPart[g], dFull[g], full_bytes, cudaMemcpyDeviceToHost);
            }
            float* acc = (float*)hFull;
            memcpy(acc, hPart[0], full_bytes);
            for (int g = 1; g < NG; ++g) {
                float* p = (float*)hPart[g];
                for (int i = 0; i < n_floats; ++i) acc[i] += p[i];
            }
            for (int g = 0; g < NG; ++g) {
                cudaSetDevice(g);
                cudaMemcpy(dFull[g], hFull, full_bytes, cudaMemcpyHostToDevice);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;
        printf("%4d    %9zu     %8.4f    %9.2f\n", B, full_bytes, ms, ms * 64);
    }

    return 0;
}
