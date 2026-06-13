// Microbench: does the kvgen "embed" phase (Q8_0 gather + half->float) really
// cost ~15s at 100K? Replicate the exact two kernels and per-chunk launch
// pattern (391 chunks of 256 tokens, H=1024), with and without sync per chunk.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>

#define CH 256
#define H 1024

struct __align__(4) bq8_0 { int8_t qs[32]; uint16_t pad; half d; };

__global__ void dequant_embd_q8_0_rows(const void* __restrict__ embd, half* __restrict__ out,
                                       const int* __restrict__ ids, int Hh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Hh) return;
    int token_id = ids[blockIdx.y];
    int blocks_per_row = Hh / 32;
    const bq8_0* row = (const bq8_0*)embd + (size_t)token_id * blocks_per_row;
    int blk = idx / 32, elem = idx % 32;
    out[(size_t)blockIdx.y * Hh + idx] = __float2half(__half2float(row[blk].d) * row[blk].qs[elem]);
}

__global__ void half_to_float_kernel(const half* __restrict__ src, float* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __half2float(src[idx]);
}

// Fused: gather Q8 directly to float, no half intermediate. One thread per elem.
__global__ void dequant_embd_q8_0_rows_f32(const void* __restrict__ embd, float* __restrict__ out,
                                           const int* __restrict__ ids, int Hh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Hh) return;
    int token_id = ids[blockIdx.y];
    int blocks_per_row = Hh / 32;
    const bq8_0* row = (const bq8_0*)embd + (size_t)token_id * blocks_per_row;
    int blk = idx / 32, elem = idx % 32;
    out[(size_t)blockIdx.y * Hh + idx] = __half2float(row[blk].d) * row[blk].qs[elem];
}

int main() {
    // vocab Q8_0 embedding table
    const int VOCAB = 248320;
    size_t tbl_bytes = (size_t)VOCAB * (H/32) * sizeof(bq8_0);
    void* embd; cudaMalloc(&embd, tbl_bytes);
    cudaMemset(embd, 1, tbl_bytes);
    half* norm_h; cudaMalloc(&norm_h, (size_t)CH*H*sizeof(half));
    float* hbuf;  cudaMalloc(&hbuf,  (size_t)CH*H*sizeof(float));
    int* ids_dev; cudaMalloc(&ids_dev, (size_t)CH*sizeof(int));
    // Realistic: random token ids spanning the whole 270MB table so each
    // gather is a cold HBM row read (L2 is ~6MB, can't cache the table).
    std::vector<int> ids(CH);
    unsigned seed = 12345;
    for (int i=0;i<CH;i++){ seed = seed*1664525u+1013904223u; ids[i]=seed % VOCAB; }
    cudaMemcpy(ids_dev, ids.data(), CH*sizeof(int), cudaMemcpyHostToDevice);

    const int N_CHUNKS = 391;  // ~100K / 256
    // Pre-generate distinct id sets per chunk (100K distinct random tokens),
    // upload all so each chunk gathers fresh cold rows (no cross-chunk L2 reuse).
    int* ids_all; cudaMalloc(&ids_all, (size_t)N_CHUNKS*CH*sizeof(int));
    std::vector<int> allids((size_t)N_CHUNKS*CH);
    for (size_t i=0;i<allids.size();i++){ seed=seed*1664525u+1013904223u; allids[i]=seed%VOCAB; }
    cudaMemcpy(ids_all, allids.data(), allids.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // ---- A: exact current path, sync per chunk (mimics KVGEN_PROFILE psync) ----
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int c=0;c<N_CHUNKS;c++) {
        int n = CH;
        dim3 eg((H+255)/256, n);
        dequant_embd_q8_0_rows<<<eg,256>>>(embd, norm_h, ids_all+(size_t)c*CH, H);
        half_to_float_kernel<<<(((size_t)n*H)+255)/256,256>>>(norm_h, hbuf, (int)((size_t)n*H));
        cudaDeviceSynchronize();
    }
    double a = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();

    // ---- B: exact current path, NO per-chunk sync (production overlap) ----
    cudaDeviceSynchronize();
    t0 = std::chrono::high_resolution_clock::now();
    for (int c=0;c<N_CHUNKS;c++) {
        int n = CH;
        dim3 eg((H+255)/256, n);
        dequant_embd_q8_0_rows<<<eg,256>>>(embd, norm_h, ids_all+(size_t)c*CH, H);
        half_to_float_kernel<<<(((size_t)n*H)+255)/256,256>>>(norm_h, hbuf, (int)((size_t)n*H));
    }
    cudaDeviceSynchronize();
    double b = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();

    // ---- C: fused single kernel direct-to-float, sync per chunk ----
    cudaDeviceSynchronize();
    t0 = std::chrono::high_resolution_clock::now();
    for (int c=0;c<N_CHUNKS;c++) {
        int n = CH;
        dim3 eg((H+255)/256, n);
        dequant_embd_q8_0_rows_f32<<<eg,256>>>(embd, hbuf, ids_all+(size_t)c*CH, H);
        cudaDeviceSynchronize();
    }
    double cc = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();

    // ---- D: fused single kernel, NO per-chunk sync ----
    cudaDeviceSynchronize();
    t0 = std::chrono::high_resolution_clock::now();
    for (int c=0;c<N_CHUNKS;c++) {
        int n = CH;
        dim3 eg((H+255)/256, n);
        dequant_embd_q8_0_rows_f32<<<eg,256>>>(embd, hbuf, ids_all+(size_t)c*CH, H);
    }
    cudaDeviceSynchronize();
    double d = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();

    printf("N_CHUNKS=%d  (each %d tok x %d)\n", N_CHUNKS, CH, H);
    printf("A current 2-kernel + sync/chunk : %.3f s  (%.3f ms/chunk)\n", a, a*1000/N_CHUNKS);
    printf("B current 2-kernel  no-sync     : %.3f s  (%.3f ms/chunk)\n", b, b*1000/N_CHUNKS);
    printf("C fused-f32        + sync/chunk : %.3f s  (%.3f ms/chunk)\n", cc, cc*1000/N_CHUNKS);
    printf("D fused-f32         no-sync     : %.3f s  (%.3f ms/chunk)\n", d, d*1000/N_CHUNKS);
    return 0;
}
