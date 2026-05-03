// Standalone round-trip test for the Q8_0 KV-cache quantize/dequantize
// kernels. Generates random fp16 vectors, runs quantize_kv_q8_0_kern →
// dequantize_kv_q8_0_kern → compares against the input. Max relative
// error per block should be ~ amax/127, i.e. ~ 0.008.
//
// Build:
//   nvcc -O3 -arch=sm_70 -std=c++17 -I src tools/test_q8kv_roundtrip.cu \
//        -o tools/test_q8kv_roundtrip

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

// block_q8_0_aligned is defined in src/quant_gemv.cuh; copy locally so this
// test compiles standalone (without dragging in the GGUF / GEMV helpers).
typedef struct __align__(4) {
    int8_t qs[32];
    uint16_t pad;
    half d;
} block_q8_0_aligned;

// Re-declare the same kernels as in attention.cuh so we can compile this
// without dragging the whole engine in. Bodies are copied verbatim.
__global__ void quantize_kv_q8_0_kern(
    const half* __restrict__ in,
    block_q8_0_aligned* __restrict__ out,
    int n_blocks)
{
    int b = blockIdx.x;
    if (b >= n_blocks) return;
    int lane = threadIdx.x;

    float v = __half2float(in[(size_t)b * 32 + lane]);
    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));

    float scale = amax / 127.0f;
    float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    int q = __float2int_rn(v * inv_scale);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    out[b].qs[lane] = (int8_t)q;
    if (lane == 0) out[b].d = __float2half(scale);
}

__global__ void dequantize_kv_q8_0_kern(
    const block_q8_0_aligned* __restrict__ in,
    half* __restrict__ out,
    int n_blocks)
{
    int b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= n_blocks) return;
    int lane = threadIdx.x;
    const block_q8_0_aligned* blk = in + b;
    float s = __half2float(blk->d);
    int8_t q = blk->qs[lane];
    out[(size_t)b * 32 + lane] = __float2half((float)q * s);
}

static void check(cudaError_t e, const char* tag) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s: %s\n", tag, cudaGetErrorString(e)); exit(1); }
}

int main() {
    const int n_blocks = 32;          // 1024 elements (1 token of K)
    const int n_elems  = n_blocks * 32;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<half> h_in(n_elems), h_out(n_elems);
    for (int i = 0; i < n_elems; ++i) h_in[i] = __float2half(dist(rng));

    half* d_in  = nullptr;
    half* d_out = nullptr;
    block_q8_0_aligned* d_q8 = nullptr;
    check(cudaMalloc(&d_in,  n_elems * sizeof(half)), "malloc in");
    check(cudaMalloc(&d_out, n_elems * sizeof(half)), "malloc out");
    check(cudaMalloc(&d_q8,  n_blocks * sizeof(block_q8_0_aligned)), "malloc q8");
    check(cudaMemcpy(d_in, h_in.data(), n_elems * sizeof(half), cudaMemcpyHostToDevice), "h2d");

    quantize_kv_q8_0_kern<<<n_blocks, 32>>>(d_in, d_q8, n_blocks);
    check(cudaGetLastError(), "quant");
    cudaDeviceSynchronize();

    dim3 dq_grid((n_blocks + 7) / 8);
    dim3 dq_block(32, 8);
    dequantize_kv_q8_0_kern<<<dq_grid, dq_block>>>(d_q8, d_out, n_blocks);
    check(cudaGetLastError(), "dequant");
    cudaDeviceSynchronize();

    check(cudaMemcpy(h_out.data(), d_out, n_elems * sizeof(half), cudaMemcpyDeviceToHost), "d2h");

    // Inspect the first block's encoded values for sanity
    std::vector<block_q8_0_aligned> h_q8(n_blocks);
    cudaMemcpy(h_q8.data(), d_q8, n_blocks * sizeof(block_q8_0_aligned), cudaMemcpyDeviceToHost);
    printf("block[0]: scale=%.4f  qs[0..7]=", __half2float(h_q8[0].d));
    for (int i = 0; i < 8; ++i) printf("%d ", h_q8[0].qs[i]);
    printf("\n");
    printf("block[1]: scale=%.4f  qs[0..7]=", __half2float(h_q8[1].d));
    for (int i = 0; i < 8; ++i) printf("%d ", h_q8[1].qs[i]);
    printf("\n");

    double max_abs = 0, sum_abs = 0;
    int    bad = 0;
    for (int i = 0; i < n_elems; ++i) {
        float x = __half2float(h_in[i]);
        float y = __half2float(h_out[i]);
        float d = fabsf(x - y);
        max_abs = fmax(max_abs, d);
        sum_abs += d;
        if (d > 0.05f) bad++;
        if (i < 16) printf("[%2d] in=% .4f  out=% .4f  diff=% .4f\n", i, x, y, x - y);
    }
    printf("avg_abs=%.4f  max_abs=%.4f  bad(>0.05)=%d/%d\n",
           sum_abs / n_elems, max_abs, bad, n_elems);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_q8);
    return 0;
}
