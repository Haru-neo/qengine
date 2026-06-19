// Standalone correctness + timing harness for the GDN_CHUNK_FAST path.
//
// Compares the reference gdn_chunk_step against the fast pre-pass +
// gdn_fast_recurrence on random inputs at realistic sizes. Checks max
// relative error of chunk_out and rec_state, and times both with cudaEvent.
//
// The MULTI-CALL section reproduces the full-pipeline conditions that the
// single-call test does NOT: it calls the fast path in a loop of sequential
// sub-tiles (carrying rec_state across calls, reusing the same lazily
// allocated scratch buffers, with a final sub-tile of n_tokens < chunk_cap),
// and compares against the reference run with the SAME carryover. Run it
// under compute-sanitizer --tool memcheck / racecheck / initcheck.
//
// Build (CMP / sm_70):
//   nvcc -O3 -arch=sm_70 -o /tmp/gdn_test tools/gdn_kernel_test.cu
// Build (4090 / sm_89):
//   nvcc -O3 -arch=sm_89 -o /tmp/gdn_test tools/gdn_kernel_test.cu
// Run:
//   /tmp/gdn_test
//   compute-sanitizer --tool memcheck   /tmp/gdn_test
//   compute-sanitizer --tool racecheck  /tmp/gdn_test
//   compute-sanitizer --tool initcheck  /tmp/gdn_test
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../src/gdn_kernels.cuh"

#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)

// ---- problem dims (match the model: k_dim=v_dim=128, num_k=16, num_v=48) ----
static const int num_k  = 16;
static const int num_v  = 48;
static const int k_dim  = 128;
static const int v_dim  = 128;
static const int qkv_dim = 2 * num_k * k_dim + num_v * v_dim;   // 10240
static const int v_total = num_v * v_dim;                       // 6144
static const int state_len = k_dim * v_dim;                     // 16384

int main() {
    // chunk_cap = the lazily-allocated scratch capacity (full pipeline uses
    // QENGINE_PREFILL_CHUNK, here 1024). Scratch is allocated ONCE at this
    // size and reused across every sub-tile, including the short final one.
    const int chunk_cap = 1024;

    srand(1234);
    auto frand = [](){ return (float)rand() / RAND_MAX * 2.0f - 1.0f; };

    // Per-head scalar weights (constant across the whole sequence).
    std::vector<float> h_alog(num_v), h_dtbias(num_v);
    for (int i=0;i<num_v;i++){ h_alog[i] = -fabsf(frand()); h_dtbias[i] = frand()*0.5f; }

    // Initial recurrent state (carried across sub-tiles).
    std::vector<float> h_state0((size_t)num_v*state_len);
    for (auto& x : h_state0) x = frand()*0.1f;

    // ---- device weight / state buffers ----
    float *d_alog, *d_dtbias, *d_state_ref, *d_state_fast;
    CK(cudaMalloc(&d_alog,   num_v*sizeof(float)));
    CK(cudaMalloc(&d_dtbias, num_v*sizeof(float)));
    CK(cudaMalloc(&d_state_ref,  h_state0.size()*sizeof(float)));
    CK(cudaMalloc(&d_state_fast, h_state0.size()*sizeof(float)));
    CK(cudaMemcpy(d_alog,   h_alog.data(),   num_v*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dtbias, h_dtbias.data(), num_v*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_state_ref,  h_state0.data(), h_state0.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_state_fast, h_state0.data(), h_state0.size()*sizeof(float), cudaMemcpyHostToDevice));

    // ---- per-sub-tile input buffers (sized chunk_cap, refilled each tile) ----
    float *d_qkv;
    half  *d_a, *d_b, *d_out_ref, *d_out_fast;
    CK(cudaMalloc(&d_qkv, (size_t)chunk_cap * qkv_dim * sizeof(float)));
    CK(cudaMalloc(&d_a,   (size_t)chunk_cap * num_v   * sizeof(half)));
    CK(cudaMalloc(&d_b,   (size_t)chunk_cap * num_v   * sizeof(half)));
    CK(cudaMalloc(&d_out_ref,  (size_t)chunk_cap * v_total * sizeof(half)));
    CK(cudaMalloc(&d_out_fast, (size_t)chunk_cap * v_total * sizeof(half)));

    // ---- fast scratch (allocated ONCE at chunk_cap, reused every tile,
    //      exactly like the full pipeline's gb.fast_* lazy alloc) ----
    float *f_qns, *f_kn, *f_attn, *f_g, *f_beta;
    CK(cudaMalloc(&f_qns, (size_t)chunk_cap*num_k*k_dim*sizeof(float)));
    CK(cudaMalloc(&f_kn,  (size_t)chunk_cap*num_k*k_dim*sizeof(float)));
    CK(cudaMalloc(&f_attn,(size_t)chunk_cap*num_v*sizeof(float)));
    CK(cudaMalloc(&f_g,   (size_t)chunk_cap*num_v*sizeof(float)));
    CK(cudaMalloc(&f_beta,(size_t)chunk_cap*num_v*sizeof(float)));

    int threads = min(v_dim, 128);
    int ref_smem  = (state_len + 2*k_dim + 1 + 96) * sizeof(float);
    int fast_smem = (state_len + 2*k_dim) * sizeof(float);
    CK(cudaFuncSetAttribute((const void*)gdn_chunk_step,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024));
    CK(cudaFuncSetAttribute((const void*)gdn_fast_recurrence,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024));

    // Run the reference chunk_step on the current d_qkv/d_a/d_b for n tokens,
    // updating d_state_ref in place and writing d_out_ref.
    auto run_ref = [&](int n){
        gdn_chunk_step<<<num_v, threads, ref_smem>>>(
            d_qkv, d_alog, d_dtbias, d_a, d_b, d_state_ref, d_out_ref,
            n, num_k, num_v, k_dim, v_dim);
    };
    // Run the fast path (3 kernels) on the current inputs for n tokens,
    // updating d_state_fast in place and writing d_out_fast.
    auto run_fast = [&](int n){
        int kthreads = min(k_dim, 128);
        dim3 g1(n, num_k);
        gdn_fast_prepass_qknorm<<<g1, kthreads>>>(d_qkv, f_qns, f_kn, n, num_k, k_dim, qkv_dim);
        dim3 g2(n, num_v);
        gdn_fast_prepass_gate<<<g2, kthreads>>>(f_qns, f_kn, d_alog, d_dtbias, d_a, d_b,
            f_attn, f_g, f_beta, n, num_k, num_v, k_dim);
        gdn_fast_recurrence<<<num_v, threads, fast_smem>>>(
            d_qkv, f_qns, f_kn, f_attn, f_g, f_beta, d_state_fast, d_out_fast,
            n, num_k, num_v, k_dim, v_dim);
    };

    auto relerr = [](float a, float b){
        float d = fabsf(a-b); float m = fmaxf(fabsf(a), fabsf(b));
        return m > 1e-4f ? d/m : d;
    };

    // ================= MULTI-CALL CARRYOVER TEST =================
    // ~16 sub-tiles of 1024, plus a short final tile, carrying state and
    // reusing scratch — the exact conditions the single-call test misses.
    const int sub_sizes[] = {1024,1024,1024,1024,1024,1024,1024,1024,
                             1024,1024,1024,1024,1024,1024,1024,1024, 257};
    const int n_sub = sizeof(sub_sizes)/sizeof(sub_sizes[0]);

    std::vector<float> h_qkv((size_t)chunk_cap * qkv_dim);
    std::vector<half>  h_a((size_t)chunk_cap*num_v), h_b((size_t)chunk_cap*num_v);
    std::vector<half>  o_ref((size_t)chunk_cap*v_total), o_fast((size_t)chunk_cap*v_total);

    float max_out_all = 0.0f, max_state_all = 0.0f;
    int   n_out_bad_all = 0;
    for (int s = 0; s < n_sub; s++) {
        int n = sub_sizes[s];
        // Fresh random inputs for this sub-tile (only the first n rows used;
        // leave the tail of the chunk_cap-sized buffers untouched to mimic
        // the pipeline, where a short last tile reuses a larger buffer).
        for (size_t i=0;i<(size_t)n*qkv_dim;i++) h_qkv[i]=frand();
        for (size_t i=0;i<(size_t)n*num_v;i++){ h_a[i]=__float2half(frand()); h_b[i]=__float2half(frand()); }
        CK(cudaMemcpy(d_qkv, h_qkv.data(), (size_t)n*qkv_dim*sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_a,   h_a.data(),   (size_t)n*num_v*sizeof(half),    cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_b,   h_b.data(),   (size_t)n*num_v*sizeof(half),    cudaMemcpyHostToDevice));

        run_ref(n);
        run_fast(n);
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());

        CK(cudaMemcpy(o_ref.data(),  d_out_ref,  (size_t)n*v_total*sizeof(half), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(o_fast.data(), d_out_fast, (size_t)n*v_total*sizeof(half), cudaMemcpyDeviceToHost));
        for (size_t i=0;i<(size_t)n*v_total;i++){
            float e = relerr(__half2float(o_ref[i]), __half2float(o_fast[i]));
            if (e>max_out_all) max_out_all=e;
            if (e>1e-3f) n_out_bad_all++;
        }
    }
    // Final state comparison after all carryover.
    std::vector<float> s_ref(h_state0.size()), s_fast(h_state0.size());
    CK(cudaMemcpy(s_ref.data(),  d_state_ref,  s_ref.size()*sizeof(float),  cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s_fast.data(), d_state_fast, s_fast.size()*sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i=0;i<s_ref.size();i++){
        float e = relerr(s_ref[i], s_fast[i]);
        if (e>max_state_all) max_state_all=e;
    }
    printf("[multi-call %d tiles] max relerr  chunk_out = %.3e   rec_state(final) = %.3e   (out>1e-3: %d)\n",
           n_sub, max_out_all, max_state_all, n_out_bad_all);

    // ================= TIMING (single 1024 tile) =================
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const int N = 1024, ITERS = 20;
    // refill a 1024-tile of inputs
    for (size_t i=0;i<(size_t)N*qkv_dim;i++) h_qkv[i]=frand();
    for (size_t i=0;i<(size_t)N*num_v;i++){ h_a[i]=__float2half(frand()); h_b[i]=__float2half(frand()); }
    CK(cudaMemcpy(d_qkv, h_qkv.data(), (size_t)N*qkv_dim*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_a,   h_a.data(),   (size_t)N*num_v*sizeof(half),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_b,   h_b.data(),   (size_t)N*num_v*sizeof(half),    cudaMemcpyHostToDevice));

    CK(cudaMemcpy(d_state_ref, h_state0.data(), h_state0.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaEventRecord(e0));
    for (int i=0;i<ITERS;i++) run_ref(N);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float t_ref=0; CK(cudaEventElapsedTime(&t_ref, e0, e1)); t_ref/=ITERS;

    CK(cudaMemcpy(d_state_fast, h_state0.data(), h_state0.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaEventRecord(e0));
    for (int i=0;i<ITERS;i++) run_fast(N);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float t_fast=0; CK(cudaEventElapsedTime(&t_fast, e0, e1)); t_fast/=ITERS;

    printf("ref  = %.3f ms   fast = %.3f ms   speedup = %.2fx\n",
           t_ref, t_fast, t_ref/t_fast);

    bool pass = (max_out_all < 1e-3f) && (max_state_all < 1e-3f);
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
