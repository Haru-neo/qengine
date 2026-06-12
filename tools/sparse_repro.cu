// Standalone repro for flash_attn_chunk_block_sparse_split illegal access
// at the kvgen 0.8B shape: GQA=4, num_kv=2, BM=16, BLOCK=256, K_SPLITS=4,
// HD=256, top_k=6, block_size_n=64, sub_n=sub_n_max=16, start_pos=4080,
// seq=4096 — exact params from the first faulting launch in
// /tmp/kvgen_sparse_debug.log. Valid sorted block_index, fp16 K/V.
//
// nvcc -arch=sm_70 -O2 -o sparse_repro sparse_repro.cu
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/sparse_attn/block_sparse.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA ERR %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

int main() {
    constexpr int HD = 256, GQA = 4, BM = 16, BLOCK = 256, K_SPLITS = 4;
    constexpr int ATTN_NB = 16;
    int num_q = 8, num_kv = 2;
    int sub_n = 16, sub_n_max = ATTN_NB;
    int start_pos = 4080, seq = 4096;
    int top_k = 6, block_size_n = 64, n_blocks = 64;
    float scale = 0.0625f;

    // Buffers sized exactly like the engine.
    half *q, *k, *v;
    int *bi;
    float *pm, *pl, *po;
    CK(cudaMalloc(&q,  (size_t)sub_n * num_q * HD * sizeof(half)));
    CK(cudaMalloc(&k,  (size_t)seq * num_kv * HD * sizeof(half)));
    CK(cudaMalloc(&v,  (size_t)seq * num_kv * HD * sizeof(half)));
    CK(cudaMalloc(&bi, (size_t)num_kv * ATTN_NB * 64 * sizeof(int)));
    constexpr int K_SPLITS_MAX = 16;
    CK(cudaMalloc(&pm, (size_t)num_q * ATTN_NB * K_SPLITS_MAX * sizeof(float)));
    CK(cudaMalloc(&pl, (size_t)num_q * ATTN_NB * K_SPLITS_MAX * sizeof(float)));
    CK(cudaMalloc(&po, (size_t)num_q * ATTN_NB * K_SPLITS_MAX * HD * sizeof(float)));

    // Fill Q/K/V with small values; block_index rows = {0, 13, 27, 41, 55, 63} sorted.
    std::vector<half> hq((size_t)sub_n * num_q * HD);
    for (auto& x : hq) x = __float2half(0.01f);
    CK(cudaMemcpy(q, hq.data(), hq.size() * sizeof(half), cudaMemcpyHostToDevice));
    std::vector<half> hk((size_t)seq * num_kv * HD);
    for (auto& x : hk) x = __float2half(0.02f);
    CK(cudaMemcpy(k, hk.data(), hk.size() * sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(v, hk.data(), hk.size() * sizeof(half), cudaMemcpyHostToDevice));
    std::vector<int> hbi((size_t)num_kv * ATTN_NB * 64, -1);
    int rowvals[6] = {0, 13, 27, 41, 55, 63};
    for (int h = 0; h < num_kv; h++)
        for (int t = 0; t < sub_n; t++)
            for (int j = 0; j < top_k; j++)
                hbi[((size_t)h * sub_n_max + t) * top_k + j] = rowvals[j];
    CK(cudaMemcpy(bi, hbi.data(), hbi.size() * sizeof(int), cudaMemcpyHostToDevice));

    int smem_bytes = GQA * HD * sizeof(half)
                   + 2 * BM * HD * sizeof(half)
                   + GQA * BM * sizeof(float);
    if (getenv("PAD64")) smem_bytes += 64;  // hypothesis probe: absorb the
                                            // s_smem[g*BM+lane] tail OOB read
    dim3 fg(num_kv, sub_n, K_SPLITS);
    printf("launch grid (%d,%d,%d) block %d smem %d\n", num_kv, sub_n, K_SPLITS, BLOCK, smem_bytes);
    flash_attn_chunk_block_sparse_split<HD, GQA, BM, BLOCK, K_SPLITS>
        <<<fg, BLOCK, smem_bytes>>>(
            q, k, v, bi, pm, pl, po,
            num_q, num_kv, start_pos, sub_n, sub_n_max,
            seq, scale, top_k, block_size_n);
    CK(cudaGetLastError());
    cudaError_t e = cudaDeviceSynchronize();
    printf("sync -> %s\n", cudaGetErrorString(e));
    return e == cudaSuccess ? 0 : 2;
}
