// Isolated round-trip test for TurboQuant 3-bit KV cache.
// Goal: prove whether tq3_quantize + tq3_dequantize kernels by themselves are
// broken, independent of the model / attention plumbing.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "../src/turboquant.cuh"

static void check(cudaError_t e, const char* tag) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", tag, cudaGetErrorString(e));
        exit(1);
    }
}

static float frand() {
    return 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
}

// CPU reference implementations of the exact same algorithm so we can diff
// against the CUDA kernels step by step.
static void cpu_wht_128(float* data) {
    for (int step = 1; step < 128; step <<= 1) {
        for (int i = 0; i < 128; i += step * 2) {
            for (int j = i; j < i + step; j++) {
                float a = data[j];
                float b = data[j + step];
                data[j]        = a + b;
                data[j + step] = a - b;
            }
        }
    }
    float scale = 1.0f / sqrtf(128.0f);
    for (int i = 0; i < 128; i++) data[i] *= scale;
}

__global__ void dump_signs_kernel(float* out) {
    int t = threadIdx.x;
    if (t < 128) out[t] = d_tq3_signs[t];
}

__global__ void dump_centroids_kernel(float* out) {
    int t = threadIdx.x;
    if (t < 8) out[t] = d_tq3_centroids[t];
}

__global__ void dump_bounds_kernel(float* out) {
    int t = threadIdx.x;
    if (t < 7) out[t] = d_tq3_bounds[t];
}

// Expose the WHT by itself so we can compare CUDA vs CPU.
__global__ void wht_only_kernel(const float* in, float* out) {
    // 1 thread, 1 block.
    float data[128];
    for (int i = 0; i < 128; i++) data[i] = in[i];
    fast_wht_128(data);
    for (int i = 0; i < 128; i++) out[i] = data[i];
}

// Encode an input vector but return the intermediate `data` right before the
// tq3_find_bin step — so we can see the rotated coordinates.
__global__ void tq3_rotate_only_kernel(const half* in, float* rotated, float* norm_out) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid != 0) return;
    float data[128];
    for (int i = 0; i < 128; i++) data[i] = __half2float(in[i]);
    float ns = 0.f;
    for (int i = 0; i < 128; i++) ns += data[i]*data[i];
    float n = sqrtf(ns);
    *norm_out = n;
    if (n < 1e-10f) { for (int i=0;i<128;i++) rotated[i]=0.f; return; }
    float inv = 1.f/n;
    for (int i = 0; i < 128; i++) data[i] *= inv;
    for (int i = 0; i < 128; i++) data[i] *= d_tq3_signs[i];
    fast_wht_128(data);
    for (int i = 0; i < 128; i++) rotated[i] = data[i];
}

int main() {
    check(cudaSetDevice(0), "setdev");
    srand(42);

    tq3_init_signs(0);
    cudaDeviceSynchronize();

    // ---- (1) Verify that d_tq3_signs was initialized and is ±1. ----
    float* d_sign_dump; cudaMalloc(&d_sign_dump, 128 * sizeof(float));
    dump_signs_kernel<<<1, 128>>>(d_sign_dump);
    float h_signs[128];
    cudaMemcpy(h_signs, d_sign_dump, 128*sizeof(float), cudaMemcpyDeviceToHost);
    int pos_count = 0, neg_count = 0, bad = 0;
    for (int i = 0; i < 128; i++) {
        if (h_signs[i] == 1.0f) pos_count++;
        else if (h_signs[i] == -1.0f) neg_count++;
        else bad++;
    }
    printf("[signs] +1=%d  -1=%d  other=%d   first8=", pos_count, neg_count, bad);
    for (int i = 0; i < 8; i++) printf("%+g ", h_signs[i]);
    printf("\n");
    if (bad > 0) { printf("  >>> SIGNS NOT INITIALIZED PROPERLY <<<\n"); }

    float h_cent[8]; cudaMemset(d_sign_dump, 0, 128*sizeof(float));
    dump_centroids_kernel<<<1, 8>>>(d_sign_dump);
    cudaMemcpy(h_cent, d_sign_dump, 8*sizeof(float), cudaMemcpyDeviceToHost);
    printf("[cent] "); for (int i=0;i<8;i++) printf("%+.5f ", h_cent[i]); printf("\n");
    float h_bnd[7]; cudaMemset(d_sign_dump, 0, 128*sizeof(float));
    dump_bounds_kernel<<<1, 7>>>(d_sign_dump);
    cudaMemcpy(h_bnd, d_sign_dump, 7*sizeof(float), cudaMemcpyDeviceToHost);
    printf("[bnd] "); for (int i=0;i<7;i++) printf("%+.5f ", h_bnd[i]); printf("\n");

    // ---- (2) Verify fast_wht_128 is self-inverse when applied to a pure unit
    // vector: H(H(x))/(d) ... well, H is self-inverse with 1/sqrt(d).
    float h_v[128]; for (int i=0;i<128;i++) h_v[i] = frand();
    float* d_v; cudaMalloc(&d_v, 128*sizeof(float));
    float* d_w; cudaMalloc(&d_w, 128*sizeof(float));
    cudaMemcpy(d_v, h_v, 128*sizeof(float), cudaMemcpyHostToDevice);
    wht_only_kernel<<<1,1>>>(d_v, d_w);           // w = H(v)
    wht_only_kernel<<<1,1>>>(d_w, d_v);           // v' = H(H(v))
    float h_vp[128]; cudaMemcpy(h_vp, d_v, 128*sizeof(float), cudaMemcpyDeviceToHost);
    float max_err = 0.f;
    cudaMemcpy(h_v, (const void*)nullptr + 0, 0, cudaMemcpyHostToHost); // no-op, keep h_v as originally set
    // Reset h_v to original random values (the GPU buffer was overwritten,
    // but the CPU array is still the original).
    // (Actually h_v is unchanged because we never copied back to it.)
    for (int i = 0; i < 128; i++) {
        float err = fabsf(h_vp[i] - h_v[i]);
        if (err > max_err) max_err = err;
    }
    printf("[wht-self-inverse] max |H(H(x)) - x| = %.3e\n", max_err);
    // Also compare CUDA WHT vs CPU WHT on a single-step.
    cudaMemcpy(d_v, h_v, 128*sizeof(float), cudaMemcpyHostToDevice);
    wht_only_kernel<<<1,1>>>(d_v, d_w);
    float h_wcuda[128]; cudaMemcpy(h_wcuda, d_w, 128*sizeof(float), cudaMemcpyDeviceToHost);
    float h_wcpu[128]; memcpy(h_wcpu, h_v, 128*sizeof(float));
    cpu_wht_128(h_wcpu);
    float wd = 0.f;
    for (int i = 0; i < 128; i++) { float e = fabsf(h_wcuda[i] - h_wcpu[i]); if (e>wd) wd=e; }
    printf("[wht-cuda-vs-cpu] max diff = %.3e\n", wd);

    // ---- (3) Full round trip through tq3_quantize + tq3_dequantize. ----
    // We pick `n_blocks` small. The quantize kernel expects a tensor of
    // [n_blocks * 128] fp16, and the dequantize kernel writes an identical
    // shape back.
    const int n_blocks = 4;
    const int n = n_blocks * 128;

    // Generate an input that MATCHES the real KV statistics: random ±1 signs
    // then scale so ||x|| = sqrt(128) (i.e. unit per-coord variance). That is
    // what attention's K/V actually looks like prior to quantization.
    float h_in[512];
    for (int i = 0; i < n; i++) h_in[i] = 0.2f * frand();
    // Convert to fp16.
    half h_in_h[512];
    for (int i = 0; i < n; i++) h_in_h[i] = __float2half(h_in[i]);

    half* d_in_h; cudaMalloc(&d_in_h, n * sizeof(half));
    half* d_out_h; cudaMalloc(&d_out_h, n * sizeof(half));
    block_tq3* d_blk; cudaMalloc(&d_blk, n_blocks * sizeof(block_tq3));
    cudaMemcpy(d_in_h, h_in_h, n*sizeof(half), cudaMemcpyHostToDevice);

    // Each kernel launch: bid < n_blocks -> thread id in [0..n_blocks)
    tq3_quantize_kernel<<<1, 32>>>(d_in_h, d_blk, n_blocks);
    check(cudaDeviceSynchronize(), "quantize");
    tq3_dequantize_kernel<<<1, 32>>>(d_blk, d_out_h, n_blocks);
    check(cudaDeviceSynchronize(), "dequantize");

    half h_out_h[512];
    cudaMemcpy(h_out_h, d_out_h, n*sizeof(half), cudaMemcpyDeviceToHost);

    // Pull back the block_tq3 so we can look at norm and some indices.
    block_tq3 h_blk[4];
    cudaMemcpy(h_blk, d_blk, n_blocks * sizeof(block_tq3), cudaMemcpyDeviceToHost);
    for (int b = 0; b < n_blocks; b++) {
        printf("[block %d] norm=%.4f  qs[0..5]= %02x %02x %02x %02x %02x %02x\n",
            b, h_blk[b].norm,
            h_blk[b].qs[0], h_blk[b].qs[1], h_blk[b].qs[2],
            h_blk[b].qs[3], h_blk[b].qs[4], h_blk[b].qs[5]);
    }

    // Per-block error stats.
    for (int b = 0; b < n_blocks; b++) {
        float mse = 0.f, denom = 0.f, maxerr = 0.f;
        for (int i = 0; i < 128; i++) {
            float a = h_in[b*128 + i];
            float c = __half2float(h_out_h[b*128 + i]);
            float e = a - c;
            mse += e*e; denom += a*a;
            if (fabsf(e) > maxerr) maxerr = fabsf(e);
        }
        printf("[block %d] relRMSE=%.3f  max|err|=%.3e   in[0..4]=%.3f %.3f %.3f %.3f   out[0..4]=%.3f %.3f %.3f %.3f\n",
            b, sqrtf(mse/denom), maxerr,
            h_in[b*128+0], h_in[b*128+1], h_in[b*128+2], h_in[b*128+3],
            __half2float(h_out_h[b*128+0]), __half2float(h_out_h[b*128+1]),
            __half2float(h_out_h[b*128+2]), __half2float(h_out_h[b*128+3]));
    }

    // ---- (4) Inspect the rotated (post-WHT) coordinates for block 0. ----
    // If the rotation is correct and the input has ~unit-variance coordinates
    // then rotated coords should be distributed ~N(0, 1/128) with σ ≈ 0.088.
    float* d_rot; cudaMalloc(&d_rot, 128*sizeof(float));
    float* d_normbuf; cudaMalloc(&d_normbuf, sizeof(float));
    tq3_rotate_only_kernel<<<1, 1>>>(d_in_h, d_rot, d_normbuf);
    cudaDeviceSynchronize();
    float h_rot[128]; cudaMemcpy(h_rot, d_rot, 128*sizeof(float), cudaMemcpyDeviceToHost);
    float h_norm; cudaMemcpy(&h_norm, d_normbuf, sizeof(float), cudaMemcpyDeviceToHost);
    float mn=0, mx=0, sum=0, sumsq=0;
    for (int i = 0; i < 128; i++) { float x = h_rot[i]; sum += x; sumsq += x*x; if (x<mn) mn=x; if (x>mx) mx=x; }
    float mean = sum/128, var = sumsq/128 - mean*mean;
    printf("[rotated block0] norm=%.4f  min=%+.4f max=%+.4f mean=%+.4f sd=%.4f  first8=%+.3f %+.3f %+.3f %+.3f %+.3f %+.3f %+.3f %+.3f\n",
        h_norm, mn, mx, mean, sqrtf(var),
        h_rot[0], h_rot[1], h_rot[2], h_rot[3],
        h_rot[4], h_rot[5], h_rot[6], h_rot[7]);

    // Show how those rotated coords map to centroid indices.
    printf("[bin hits]   ");
    int bin_count[8] = {0};
    for (int i = 0; i < 128; i++) {
        int idx = 0;
        for (int k = 0; k < 7; k++) if (h_rot[i] > h_bnd[k]) idx = k+1;
        bin_count[idx]++;
    }
    for (int k = 0; k < 8; k++) printf("%d ", bin_count[k]);
    printf("\n");

    // ---- (5) Realistic K/V-like data ----
    // Real attention K/V has per-coordinate std ~0.3-1.0 and ESPECIALLY
    // outlier channels (a handful of coords with mag >5σ). This is the case
    // TurboQuant's randomized Hadamard is supposed to handle.
    auto bench = [&](const char* tag, float base_std, float outlier_mag, int n_outliers) {
        for (int b = 0; b < n_blocks; b++) {
            for (int i = 0; i < 128; i++) {
                // Box-Muller-ish Gaussian from two uniforms
                float u1 = (float)(rand() + 1) / (float)(RAND_MAX + 1.0);
                float u2 = (float)rand() / (float)RAND_MAX;
                float g  = sqrtf(-2.f*logf(u1)) * cosf(2.f*3.14159265f*u2);
                h_in[b*128 + i] = base_std * g;
            }
            // Inject outliers on FIXED channels — realistic "outlier channel" pattern.
            for (int o = 0; o < n_outliers; o++) {
                int ch = (7 + o * 23) % 128;
                h_in[b*128 + ch] = outlier_mag * ((b + o) & 1 ? 1 : -1);
            }
        }
        for (int i = 0; i < n; i++) h_in_h[i] = __float2half(h_in[i]);
        cudaMemcpy(d_in_h, h_in_h, n*sizeof(half), cudaMemcpyHostToDevice);
        tq3_quantize_kernel<<<1, 32>>>(d_in_h, d_blk, n_blocks);
        cudaDeviceSynchronize();
        tq3_dequantize_kernel<<<1, 32>>>(d_blk, d_out_h, n_blocks);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out_h, d_out_h, n*sizeof(half), cudaMemcpyDeviceToHost);
        float tot_mse = 0, tot_den = 0, worst = 0;
        for (int b = 0; b < n_blocks; b++) {
            float mse=0, den=0;
            for (int i = 0; i < 128; i++) {
                float a = h_in[b*128 + i];
                float c = __half2float(h_out_h[b*128 + i]);
                float e = a - c;
                mse += e*e; den += a*a;
                if (fabsf(e) > worst) worst = fabsf(e);
            }
            tot_mse += mse; tot_den += den;
        }
        printf("[%s] relRMSE=%.3f  max|err|=%.3e  (base_std=%.2f outliers %d@%.1f)\n",
            tag, sqrtf(tot_mse/tot_den), worst, base_std, n_outliers, outlier_mag);
    };
    bench("gauss-small  ", 0.1f, 0.0f, 0);
    bench("gauss-unit   ", 1.0f, 0.0f, 0);
    bench("gauss+outlier", 0.5f, 3.0f, 1);
    bench("big-outlier  ", 0.3f, 10.0f, 2);

    return 0;
}
