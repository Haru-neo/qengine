// Micro-bench: drafter attn_full legacy vs FA-tiled rewrite.
// Correctness vs fp64 CPU reference + timing at the drafter's real shapes
// (q_len=16, total_k = ctx_window+16, 32 q-heads / 8 kv-heads, HD=128).
//   nvcc -O3 -arch=sm_70 -o dflash_attn_bench dflash_attn_bench.cu && ./dflash_attn_bench
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// ── copy of the two kernels (keep in sync with src/dflash_draft.cuh) ──────
__global__ void attn_full_kernel(
    const half* __restrict__ Q, const half* __restrict__ K,
    const half* __restrict__ V, half* __restrict__ Y,
    int q_len, int total_k, int n_q_heads, int n_kv_heads, int head_dim,
    float scale
) {
    int q_tok = blockIdx.x;
    int q_head = blockIdx.y;
    int tid = threadIdx.x;
    int kv_head = q_head * n_kv_heads / n_q_heads;
    extern __shared__ float smem[];
    float* scores = smem;
    float* q_local = smem + total_k;
    const half* qp = Q + ((size_t)q_tok * n_q_heads + q_head) * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) q_local[i] = __half2float(qp[i]);
    __syncthreads();
    for (int k = tid; k < total_k; k += blockDim.x) {
        const half* kp = K + ((size_t)k * n_kv_heads + kv_head) * head_dim;
        float s = 0.0f;
        for (int i = 0; i < head_dim; i++) s += q_local[i] * __half2float(kp[i]);
        scores[k] = s * scale;
    }
    __syncthreads();
    __shared__ float sh_max, sh_sum;
    if (tid == 0) {
        float mx = -INFINITY;
        for (int k = 0; k < total_k; k++) if (scores[k] > mx) mx = scores[k];
        sh_max = mx;
    }
    __syncthreads();
    float sum = 0.0f;
    for (int k = tid; k < total_k; k += blockDim.x) {
        scores[k] = expf(scores[k] - sh_max);
        sum += scores[k];
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    __shared__ float warp_sums[32];
    int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
        if (tid == 0) sh_sum = sum;
    }
    __syncthreads();
    float inv_sum = 1.0f / sh_sum;
    half* yp = Y + ((size_t)q_tok * n_q_heads + q_head) * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < total_k; k++) {
            const half* vp = V + ((size_t)k * n_kv_heads + kv_head) * head_dim;
            acc += scores[k] * inv_sum * __half2float(vp[i]);
        }
        yp[i] = __float2half(acc);
    }
}

template<int HD, int GQA, int BM>
__global__ void attn_full_fa_kernel(
    const half* __restrict__ Q, const half* __restrict__ K,
    const half* __restrict__ V, half* __restrict__ Y,
    int q_len, int total_k, int n_q_heads, int n_kv_heads, float scale
) {
    constexpr int PAD    = 2;
    constexpr int STRIDE = HD + PAD;
    int q_tok   = blockIdx.x;
    int kv_head = blockIdx.y;
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    __shared__ half  q_s[GQA * HD];
    __shared__ half  k_tile[BM * STRIDE];
    __shared__ half  v_tile[BM * STRIDE];
    __shared__ float p_smem[GQA * BM];
    __shared__ float fct_s[GQA];
    __shared__ float linv_s[GQA];

    for (int i = tid; i < GQA * HD; i += blockDim.x) {
        int g = i / HD, c = i - g * HD;
        int q_head = kv_head * GQA + g;
        q_s[i] = Q[((size_t)q_tok * n_q_heads + q_head) * HD + c];
    }
    __syncthreads();

    float m_w = -INFINITY;
    float l_w = 0.0f;
    float acc[GQA];
    #pragma unroll
    for (int g = 0; g < GQA; g++) acc[g] = 0.0f;

    constexpr int H2       = HD / 2;
    constexpr int ROWS_PER = 128 / H2;

    for (int tile = 0; tile < total_k; tile += BM) {
        int tile_len = min(BM, total_k - tile);
        #pragma unroll
        for (int pass = 0; pass < BM / ROWS_PER; pass++) {
            int r  = pass * ROWS_PER + tid / H2;
            int c2 = (tid % H2) * 2;
            if (r < tile_len) {
                size_t base = ((size_t)(tile + r) * n_kv_heads + kv_head) * HD + c2;
                *(half2*)&k_tile[r * STRIDE + c2] = *(const half2*)&K[base];
                *(half2*)&v_tile[r * STRIDE + c2] = *(const half2*)&V[base];
            }
        }
        __syncthreads();

        if (warp < GQA) {
            int g = warp;
            float s = -INFINITY;
            if (lane < tile_len) {
                float acc_s = 0.0f;
                const half* qp = q_s + g * HD;
                const half* kp = k_tile + lane * STRIDE;
                #pragma unroll
                for (int i = 0; i < HD; i += 2) {
                    half2 qv = *(const half2*)&qp[i];
                    half2 kv = *(const half2*)&kp[i];
                    float2 qf = __half22float2(qv);
                    float2 kf = __half22float2(kv);
                    acc_s += qf.x * kf.x + qf.y * kf.y;
                }
                s = acc_s * scale;
            }
            float m_tile = s;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, o));
            float m_new  = fmaxf(m_w, m_tile);
            float factor = (m_w == -INFINITY) ? 0.0f : expf(m_w - m_new);
            float p = (lane < tile_len) ? expf(s - m_new) : 0.0f;
            float p_sum = p;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                p_sum += __shfl_xor_sync(0xffffffff, p_sum, o);
            l_w = l_w * factor + p_sum;
            m_w = m_new;
            p_smem[g * BM + lane] = p;
            if (lane == 0) fct_s[g] = factor;
        }
        __syncthreads();

        if (tid < HD) {
            #pragma unroll
            for (int g = 0; g < GQA; g++) {
                float a = acc[g] * fct_s[g];
                const float* pr = p_smem + g * BM;
                for (int r = 0; r < tile_len; r++)
                    a += pr[r] * __half2float(v_tile[r * STRIDE + tid]);
                acc[g] = a;
            }
        }
        __syncthreads();
    }

    if (warp < GQA && lane == 0) linv_s[warp] = 1.0f / l_w;
    __syncthreads();

    if (tid < HD) {
        #pragma unroll
        for (int g = 0; g < GQA; g++) {
            int q_head = kv_head * GQA + g;
            Y[((size_t)q_tok * n_q_heads + q_head) * HD + tid] =
                __float2half(acc[g] * linv_s[g]);
        }
    }
}

// ── host reference (fp64) ─────────────────────────────────────────────────
static void ref_attn(const std::vector<half>& Q, const std::vector<half>& K,
                     const std::vector<half>& V, std::vector<float>& Y,
                     int q_len, int total_k, int nq, int nkv, int hd, float scale) {
    int gqa = nq / nkv;
    for (int t = 0; t < q_len; t++)
        for (int h = 0; h < nq; h++) {
            int kvh = h / gqa;
            std::vector<double> s(total_k);
            double mx = -1e300;
            for (int k = 0; k < total_k; k++) {
                double d = 0;
                for (int i = 0; i < hd; i++)
                    d += (double)__half2float(Q[((size_t)t*nq+h)*hd+i]) *
                         (double)__half2float(K[((size_t)k*nkv+kvh)*hd+i]);
                s[k] = d * scale;
                if (s[k] > mx) mx = s[k];
            }
            double l = 0;
            for (int k = 0; k < total_k; k++) { s[k] = exp(s[k]-mx); l += s[k]; }
            for (int i = 0; i < hd; i++) {
                double a = 0;
                for (int k = 0; k < total_k; k++)
                    a += s[k] * (double)__half2float(V[((size_t)k*nkv+kvh)*hd+i]);
                Y[((size_t)t*nq+h)*hd+i] = (float)(a / l);
            }
        }
}

int main(int argc, char** argv) {
    int total_k = argc > 1 ? atoi(argv[1]) : 4112;
    const int q_len = 16, nq = 32, nkv = 8, hd = 128;
    float scale = 1.0f / sqrtf((float)hd);
    srand(42);
    auto rnd = []{ return (float)rand() / RAND_MAX * 2.0f - 1.0f; };
    std::vector<half> Q((size_t)q_len*nq*hd), K((size_t)total_k*nkv*hd), V((size_t)total_k*nkv*hd);
    for (auto& x : Q) x = __float2half(rnd());
    for (auto& x : K) x = __float2half(rnd()*0.5f);
    for (auto& x : V) x = __float2half(rnd());

    half *dQ, *dK, *dV, *dY1, *dY2;
    CK(cudaMalloc(&dQ, Q.size()*2)); CK(cudaMalloc(&dK, K.size()*2));
    CK(cudaMalloc(&dV, V.size()*2));
    CK(cudaMalloc(&dY1, Q.size()*2)); CK(cudaMalloc(&dY2, Q.size()*2));
    CK(cudaMemcpy(dQ, Q.data(), Q.size()*2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, K.data(), K.size()*2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, V.data(), V.size()*2, cudaMemcpyHostToDevice));

    // legacy
    dim3 g1(q_len, nq);
    size_t smem = (total_k + hd) * sizeof(float);
    attn_full_kernel<<<g1, 128, smem>>>(dQ, dK, dV, dY1, q_len, total_k, nq, nkv, hd, scale);
    CK(cudaDeviceSynchronize());
    // FA
    dim3 g2(q_len, nkv);
    attn_full_fa_kernel<128,4,32><<<g2, 128>>>(dQ, dK, dV, dY2, q_len, total_k, nq, nkv, scale);
    CK(cudaDeviceSynchronize());

    std::vector<half> Y1(Q.size()), Y2(Q.size());
    CK(cudaMemcpy(Y1.data(), dY1, Y1.size()*2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Y2.data(), dY2, Y2.size()*2, cudaMemcpyDeviceToHost));
    std::vector<float> Yr(Q.size());
    ref_attn(Q, K, V, Yr, q_len, total_k, nq, nkv, hd, scale);

    double e1 = 0, e2 = 0;
    for (size_t i = 0; i < Yr.size(); i++) {
        e1 = fmax(e1, fabs(__half2float(Y1[i]) - Yr[i]));
        e2 = fmax(e2, fabs(__half2float(Y2[i]) - Yr[i]));
    }
    printf("total_k=%d  max_abs_err legacy=%.5f  fa=%.5f\n", total_k, e1, e2);

    // timing
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int REPS = 50;
    cudaEventRecord(a);
    for (int r = 0; r < REPS; r++)
        attn_full_kernel<<<g1, 128, smem>>>(dQ, dK, dV, dY1, q_len, total_k, nq, nkv, hd, scale);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms1; cudaEventElapsedTime(&ms1, a, b);
    cudaEventRecord(a);
    for (int r = 0; r < REPS; r++)
        attn_full_fa_kernel<128,4,32><<<g2, 128>>>(dQ, dK, dV, dY2, q_len, total_k, nq, nkv, scale);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms2; cudaEventElapsedTime(&ms2, a, b);
    printf("legacy %.3f ms/call   fa %.3f ms/call   speedup %.2fx\n",
           ms1/REPS, ms2/REPS, ms1/ms2);
    return 0;
}
