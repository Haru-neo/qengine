// Standalone attention decode-path microbench.
//
// Measures the three kernels that dominate per-token decode attention on
// CMP 100-210 (Volta sm70) when fused FA is bypassed:
//   attn_score_kernel_h  : Q[fp16] · K_cache[fp16]
//   softmax_kernel       : per-row softmax of [num_q, seq_len]
//   attn_value_kernel_h  : (softmax · V_cache[fp16]) -> out[fp16]
//
// And also the fused decode path that the runtime actually uses for short
// contexts (per-token QKV split + score+softmax+value separately).
//
// Goal: measure the achieved HBM bandwidth on each kernel against the
// 818 GB/s ceiling found by bench_hbm_bandwidth (READ pattern). KV cache
// access is essentially a streaming read so peak GB/s is the right ruler.
//
// Build:
//   nvcc -O3 -arch=sm_70 -std=c++17 tools/bench_attn_decode.cu \
//        -o tools/bench_attn_decode
//
// Run:
//   ./tools/bench_attn_decode           # 27B shape sweep
//   ./tools/bench_attn_decode 9b        # 9B shape sweep

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Re-declare the same kernels, copy-paste compatible with src/attention.cuh
// so this bench compiles standalone (no including the giant header).

// ============ attn_score_kernel_h ============
__global__ void attn_score_kernel_h(
    const half* __restrict__ q,         // [num_q_heads * head_dim]
    const half* __restrict__ k_cache,   // [seq_len * num_kv_heads * head_dim]
    float* __restrict__ scores,         // [num_q_heads * seq_len]
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, float scale)
{
    int q_head = blockIdx.x;
    int pos    = blockIdx.y;
    if (q_head >= num_q_heads || pos >= seq_len) return;
    int kv_head = q_head / (num_q_heads / num_kv_heads);
    const half* qh = q + q_head * head_dim;
    const half* kh = k_cache + pos * num_kv_heads * head_dim + kv_head * head_dim;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        sum += __half2float(qh[i]) * __half2float(kh[i]);
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    __shared__ float warp_sums[8];
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int n_warps = (blockDim.x + 31) >> 5;
    if (lane_id == 0) warp_sums[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        sum = (lane_id < n_warps) ? warp_sums[lane_id] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane_id == 0)
            scores[q_head * seq_len + pos] = sum * scale;
    }
}

// ============ softmax_kernel ============
__global__ void softmax_kernel(
    float* __restrict__ scores, int num_heads, int seq_len)
{
    int head = blockIdx.x;
    if (head >= num_heads) return;
    float* row = scores + head * seq_len;
    extern __shared__ float sdata[];
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x)
        max_val = fmaxf(max_val, row[i]);
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float e = expf(row[i] - max_val);
        row[i] = e;
        sum += e;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    sum = sdata[0];
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x)
        row[i] *= inv_sum;
}

// ============ attn_value_kernel_h ============
__global__ void attn_value_kernel_h(
    const float* __restrict__ scores,
    const half*  __restrict__ v_cache,
    half*        __restrict__ output,
    int num_q_heads, int num_kv_heads, int head_dim, int seq_len)
{
    int q_head = blockIdx.x;
    if (q_head >= num_q_heads) return;
    int kv_head = q_head / (num_q_heads / num_kv_heads);
    const float* sc = scores + q_head * seq_len;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            sum += sc[pos] * __half2float(v_cache[pos * num_kv_heads * head_dim + kv_head * head_dim + d]);
        }
        output[q_head * head_dim + d] = __float2half(sum);
    }
}

static void check(cudaError_t e, const char* tag) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", tag, cudaGetErrorString(e));
        exit(1);
    }
}

struct Conf { const char* label; int num_q; int num_kv; int head_dim; };

int main(int argc, char** argv) {
    cudaSetDevice(0);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    fprintf(stderr, "GPU: %s, %d SMs, %.0f GB/s peak HBM\n",
            p.name, p.multiProcessorCount,
            2.0 * p.memoryClockRate * (p.memoryBusWidth / 8.0) / 1e6);

    bool use_9b = (argc > 1) && (strcmp(argv[1], "9b") == 0);
    Conf cf;
    if (use_9b) cf = {"9B  num_q=16 num_kv=4 hd=128", 16, 4, 128};
    else        cf = {"27B num_q=24 num_kv=4 hd=256", 24, 4, 256};

    std::vector<int> seqs = {512, 2048, 8192, 32768, 131072};
    int max_seq = seqs.back();

    size_t q_bytes  = (size_t)cf.num_q * cf.head_dim * sizeof(half);
    size_t k_bytes  = (size_t)max_seq * cf.num_kv * cf.head_dim * sizeof(half);
    size_t v_bytes  = k_bytes;
    size_t sc_bytes = (size_t)cf.num_q * max_seq * sizeof(float);
    size_t o_bytes  = (size_t)cf.num_q * cf.head_dim * sizeof(half);

    half  *dQ, *dK, *dV, *dO;
    float *dS;
    check(cudaMalloc(&dQ, q_bytes),  "dQ");
    check(cudaMalloc(&dK, k_bytes),  "dK");
    check(cudaMalloc(&dV, v_bytes),  "dV");
    check(cudaMalloc(&dO, o_bytes),  "dO");
    check(cudaMalloc(&dS, sc_bytes), "dS");
    check(cudaMemset(dQ, 0x3c, q_bytes), "memset Q");
    check(cudaMemset(dK, 0x3c, k_bytes), "memset K");
    check(cudaMemset(dV, 0x3c, v_bytes), "memset V");

    printf("\n%s\n", cf.label);
    printf("seq_len  score_ms  score_GBs  pct  | softmax_ms | value_ms  value_GBs  pct  | total_ms  KV_GBs  pct\n");
    printf("--------------------------------------------------------------------------------------------------------\n");

    const float scale = 1.0f / 16.0f;
    int iters = 100;
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);

    for (int seq_len : seqs) {
        // -------- score --------
        dim3 grid_s(cf.num_q, seq_len);
        // warmup
        for (int i = 0; i < 3; i++)
            attn_score_kernel_h<<<grid_s, 256>>>(dQ, dK, dS, cf.num_q, cf.num_kv, cf.head_dim, seq_len, scale);
        cudaDeviceSynchronize();
        cudaEventRecord(a);
        for (int i = 0; i < iters; i++)
            attn_score_kernel_h<<<grid_s, 256>>>(dQ, dK, dS, cf.num_q, cf.num_kv, cf.head_dim, seq_len, scale);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms_score; cudaEventElapsedTime(&ms_score, a, b);
        ms_score /= iters;
        // K bytes read: seq_len * num_kv * head_dim * 2 (one pass, kv heads shared by gqa groups)
        double k_read_bytes = (double)seq_len * cf.num_kv * cf.head_dim * 2;
        double score_gbs = k_read_bytes / (ms_score * 1e-3) / 1e9;

        // -------- softmax --------
        // re-init scores so softmax has finite values; use single iteration of score then time softmax
        attn_score_kernel_h<<<grid_s, 256>>>(dQ, dK, dS, cf.num_q, cf.num_kv, cf.head_dim, seq_len, scale);
        cudaDeviceSynchronize();
        int sm_threads = 256;
        size_t sm_smem = sm_threads * sizeof(float);
        for (int i = 0; i < 3; i++)
            softmax_kernel<<<cf.num_q, sm_threads, sm_smem>>>(dS, cf.num_q, seq_len);
        cudaDeviceSynchronize();
        // softmax overwrites dS; restore each iter via a copy is too expensive. Just time the (now exp-saturated) re-runs.
        cudaEventRecord(a);
        for (int i = 0; i < iters; i++)
            softmax_kernel<<<cf.num_q, sm_threads, sm_smem>>>(dS, cf.num_q, seq_len);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms_sm; cudaEventElapsedTime(&ms_sm, a, b);
        ms_sm /= iters;

        // -------- value --------
        // re-prep dS: re-run score+softmax once
        attn_score_kernel_h<<<grid_s, 256>>>(dQ, dK, dS, cf.num_q, cf.num_kv, cf.head_dim, seq_len, scale);
        softmax_kernel<<<cf.num_q, sm_threads, sm_smem>>>(dS, cf.num_q, seq_len);
        cudaDeviceSynchronize();
        for (int i = 0; i < 3; i++)
            attn_value_kernel_h<<<cf.num_q, 256>>>(dS, dV, dO, cf.num_q, cf.num_kv, cf.head_dim, seq_len);
        cudaDeviceSynchronize();
        cudaEventRecord(a);
        for (int i = 0; i < iters; i++)
            attn_value_kernel_h<<<cf.num_q, 256>>>(dS, dV, dO, cf.num_q, cf.num_kv, cf.head_dim, seq_len);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms_val; cudaEventElapsedTime(&ms_val, a, b);
        ms_val /= iters;
        double v_read_bytes = (double)seq_len * cf.num_kv * cf.head_dim * 2;
        // Each q_head loops over the entire seq_len * v_cache, but cache lines may be reused
        // across gqa group → effective read ~= K size if perfectly cached, but per-head re-reads
        // happen if cache thrashes. Lower-bound = K size; upper-bound = K*gqa.
        double v_read_bytes_max = v_read_bytes * (cf.num_q / cf.num_kv);
        double val_gbs = v_read_bytes_max / (ms_val * 1e-3) / 1e9;
        double val_gbs_min = v_read_bytes / (ms_val * 1e-3) / 1e9;

        double peak_gbs = 818.0;  // from bench_hbm_bandwidth READ
        double total_ms = ms_score + ms_sm + ms_val;
        double kv_total_bytes = 2 * v_read_bytes;
        double kv_gbs = kv_total_bytes / (total_ms * 1e-3) / 1e9;

        printf("%-7d  %7.4f   %6.1f   %4.1f%% | %7.4f    | %7.4f   %6.1f   %4.1f%% | %6.3f   %6.1f  %4.1f%%\n",
               seq_len, ms_score, score_gbs, score_gbs / peak_gbs * 100,
               ms_sm,
               ms_val, val_gbs_min, val_gbs_min / peak_gbs * 100,
               total_ms, kv_gbs, kv_gbs / peak_gbs * 100);
    }

    printf("\n  pct = achieved / 818 GB/s (READ peak from bench_hbm_bandwidth)\n");
    printf("  value GB/s uses lower bound (no per-q re-read). Upper bound = gqa_ratio*lower.\n");
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dS);
    return 0;
}
