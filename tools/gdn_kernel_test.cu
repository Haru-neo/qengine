// Standalone correctness + timing harness for the GDN_CHUNK_FAST path.
//
// Compares the reference gdn_chunk_step against the fast pre-pass +
// gdn_fast_recurrence on random inputs at realistic sizes. Checks max
// relative error of chunk_out and rec_state, and times both with cudaEvent.
//
// Build (CMP / sm_70):
//   nvcc -O3 -arch=sm_70 -o /tmp/gdn_test tools/gdn_kernel_test.cu
// Run:
//   /tmp/gdn_test
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/gdn_kernels.cuh"

#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)

int main() {
    const int N      = 1024;
    const int num_k  = 16;
    const int num_v  = 48;
    const int k_dim  = 128;
    const int v_dim  = 128;
    const int qkv_dim = 2 * num_k * k_dim + num_v * v_dim;
    const int v_total = num_v * v_dim;
    const int state_len = k_dim * v_dim;

    srand(1234);
    auto frand = [](){ return (float)rand() / RAND_MAX * 2.0f - 1.0f; };

    // Host inputs.
    std::vector<float> h_qkv((size_t)N * qkv_dim);
    for (auto& x : h_qkv) x = frand();
    std::vector<float> h_alog(num_v), h_dtbias(num_v);
    for (int i=0;i<num_v;i++){ h_alog[i] = -fabsf(frand()); h_dtbias[i] = frand()*0.5f; }
    std::vector<half> h_a((size_t)N*num_v), h_b((size_t)N*num_v);
    for (size_t i=0;i<h_a.size();i++){ h_a[i]=__float2half(frand()); h_b[i]=__float2half(frand()); }
    std::vector<float> h_state0((size_t)num_v*state_len);
    for (auto& x : h_state0) x = frand()*0.1f;

    // Device buffers.
    float *d_qkv, *d_alog, *d_dtbias, *d_state_ref, *d_state_fast;
    half  *d_a, *d_b, *d_out_ref, *d_out_fast;
    CK(cudaMalloc(&d_qkv, h_qkv.size()*sizeof(float)));
    CK(cudaMalloc(&d_alog, num_v*sizeof(float)));
    CK(cudaMalloc(&d_dtbias, num_v*sizeof(float)));
    CK(cudaMalloc(&d_a, h_a.size()*sizeof(half)));
    CK(cudaMalloc(&d_b, h_b.size()*sizeof(half)));
    CK(cudaMalloc(&d_state_ref,  h_state0.size()*sizeof(float)));
    CK(cudaMalloc(&d_state_fast, h_state0.size()*sizeof(float)));
    CK(cudaMalloc(&d_out_ref,  (size_t)N*v_total*sizeof(half)));
    CK(cudaMalloc(&d_out_fast, (size_t)N*v_total*sizeof(half)));
    CK(cudaMemcpy(d_qkv, h_qkv.data(), h_qkv.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_alog, h_alog.data(), num_v*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dtbias, h_dtbias.data(), num_v*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_a, h_a.data(), h_a.size()*sizeof(half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_b, h_b.data(), h_b.size()*sizeof(half), cudaMemcpyHostToDevice));

    // Fast scratch.
    float *f_qns, *f_kn, *f_attn, *f_g, *f_beta;
    CK(cudaMalloc(&f_qns, (size_t)N*num_k*k_dim*sizeof(float)));
    CK(cudaMalloc(&f_kn,  (size_t)N*num_k*k_dim*sizeof(float)));
    CK(cudaMalloc(&f_attn,(size_t)N*num_v*sizeof(float)));
    CK(cudaMalloc(&f_g,   (size_t)N*num_v*sizeof(float)));
    CK(cudaMalloc(&f_beta,(size_t)N*num_v*sizeof(float)));

    int threads = 128;
    int state_len_full = k_dim*v_dim;
    int ref_smem  = (state_len_full + 2*k_dim + 1 + 96) * sizeof(float);
    int fast_smem = (state_len_full + 2*k_dim) * sizeof(float);
    CK(cudaFuncSetAttribute((const void*)gdn_chunk_step,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024));
    CK(cudaFuncSetAttribute((const void*)gdn_fast_recurrence,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024));

    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    auto run_ref = [&](){
        CK(cudaMemcpy(d_state_ref, h_state0.data(), h_state0.size()*sizeof(float), cudaMemcpyHostToDevice));
        gdn_chunk_step<<<num_v, threads, ref_smem>>>(
            d_qkv, d_alog, d_dtbias, d_a, d_b, d_state_ref, d_out_ref,
            N, num_k, num_v, k_dim, v_dim);
    };
    auto run_fast = [&](){
        CK(cudaMemcpy(d_state_fast, h_state0.data(), h_state0.size()*sizeof(float), cudaMemcpyHostToDevice));
        dim3 g1(N, num_k);
        gdn_fast_prepass_qknorm<<<g1, threads>>>(d_qkv, f_qns, f_kn, N, num_k, k_dim, qkv_dim);
        dim3 g2(N, num_v);
        gdn_fast_prepass_gate<<<g2, threads>>>(f_qns, f_kn, d_alog, d_dtbias, d_a, d_b,
            f_attn, f_g, f_beta, N, num_k, num_v, k_dim);
        gdn_fast_recurrence<<<num_v, threads, fast_smem>>>(
            d_qkv, f_qns, f_kn, f_attn, f_g, f_beta, d_state_fast, d_out_fast,
            N, num_k, num_v, k_dim, v_dim);
    };

    // Warmup + correctness.
    run_ref(); run_fast(); CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    std::vector<half> o_ref((size_t)N*v_total), o_fast((size_t)N*v_total);
    std::vector<float> s_ref(h_state0.size()), s_fast(h_state0.size());
    CK(cudaMemcpy(o_ref.data(), d_out_ref, o_ref.size()*sizeof(half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o_fast.data(), d_out_fast, o_fast.size()*sizeof(half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s_ref.data(), d_state_ref, s_ref.size()*sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s_fast.data(), d_state_fast, s_fast.size()*sizeof(float), cudaMemcpyDeviceToHost));

    auto relerr = [](float a, float b){
        float d = fabsf(a-b); float m = fmaxf(fabsf(a), fabsf(b));
        return m > 1e-4f ? d/m : d;
    };
    float max_out=0, max_state=0; int n_out_bad=0;
    for (size_t i=0;i<o_ref.size();i++){
        float e = relerr(__half2float(o_ref[i]), __half2float(o_fast[i]));
        if (e>max_out) max_out=e;
        if (e>1e-3f) n_out_bad++;
    }
    for (size_t i=0;i<s_ref.size();i++){
        float e = relerr(s_ref[i], s_fast[i]);
        if (e>max_state) max_state=e;
    }
    printf("max relerr  chunk_out = %.3e   rec_state = %.3e   (out>1e-3: %d / %zu)\n",
           max_out, max_state, n_out_bad, o_ref.size());

    // Timing (median of a few runs).
    const int ITERS = 20;
    CK(cudaEventRecord(e0));
    for (int i=0;i<ITERS;i++) run_ref();
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float t_ref=0; CK(cudaEventElapsedTime(&t_ref, e0, e1)); t_ref/=ITERS;

    CK(cudaEventRecord(e0));
    for (int i=0;i<ITERS;i++) run_fast();
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float t_fast=0; CK(cudaEventElapsedTime(&t_fast, e0, e1)); t_fast/=ITERS;

    printf("ref  = %.3f ms   fast = %.3f ms   speedup = %.2fx\n",
           t_ref, t_fast, t_ref/t_fast);

    bool pass = (max_out < 1e-3f) && (max_state < 1e-3f);
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
