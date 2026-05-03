// Standalone microbench for CMP 100-210 effective HBM bandwidth.
//
// We have been assuming HBM2 ~700 GB/s for HW-limit math (gen 8.9 t/s = 22%
// of theoretical 41 t/s). This bench measures what we actually achieve in
// several access patterns so the denominator is grounded.
//
// Patterns:
//   D2D    : cudaMemcpyAsync DeviceToDevice (single stream, async)
//   COPY   : kernel float4 read+write A->B
//   READ   : kernel linear scan, accumulate then write 1 fp32 (weight-read shape)
//   STRIDE : kernel strided read with stride S (KV cache pattern proxy)
//   GATHER : kernel random-index gather (worst-case scatter)
//
// Sizes: 16 MB / 256 MB / 2 GB / 5 GB.
//
// Build:
//   nvcc -O3 -arch=sm_70 -std=c++17 tools/bench_hbm_bandwidth.cu \
//        -o tools/bench_hbm_bandwidth

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cudaError_t e, const char* tag) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", tag, cudaGetErrorString(e));
        exit(1);
    }
}

// ===========================================================
// Kernels
// ===========================================================

// vectorized copy: read 16B + write 16B per thread iter
__global__ void k_copy_f4(const float4* __restrict__ src, float4* __restrict__ dst, size_t n4) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < n4; i += stride) dst[i] = src[i];
}

// linear read: each thread strided-reduces, single fp32 written total
__global__ void k_read_f4(const float4* __restrict__ src, float* __restrict__ sink, size_t n4) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (size_t i = tid; i < n4; i += stride) {
        float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (tid == 0) *sink = acc;
    else if (acc == -1234.567e30f) sink[1] = acc; // dead branch keeps acc live
}

// strided read: each warp reads consecutive S * 16B at offset block*S, simulating
// KV cache linear-per-step access at stride S elems.
__global__ void k_stride_read_f4(const float4* __restrict__ src, float* __restrict__ sink,
                                  size_t n4, size_t stride_elems) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t lane_groups = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    // each thread reads at positions tid, tid+stride_elems, tid+2*stride_elems...
    for (size_t base = 0; base < n4; base += stride_elems * lane_groups) {
        size_t off = base + tid;
        if (off < n4) {
            float4 v = src[off];
            acc += v.x + v.y + v.z + v.w;
        }
    }
    if (tid == 0) *sink = acc;
    else if (acc == -1234.567e30f) sink[1] = acc;
}

// gather: random index per thread, 16B per access
__global__ void k_gather_f4(const float4* __restrict__ src, const int* __restrict__ idx,
                             float* __restrict__ sink, size_t n4, size_t niter) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (size_t i = tid; i < niter; i += stride) {
        int j = idx[i];
        float4 v = src[j];
        acc += v.x + v.y + v.z + v.w;
    }
    if (tid == 0) *sink = acc;
    else if (acc == -1234.567e30f) sink[1] = acc;
}

// ===========================================================
// Bench helpers
// ===========================================================

struct Result { double bw_gbs; float ms; };

static Result time_kernel(void (*launch)(cudaStream_t), int repeats, cudaStream_t s) {
    // warmup
    for (int i = 0; i < 3; ++i) launch(s);
    check(cudaStreamSynchronize(s), "warmup sync");

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, s);
    for (int i = 0; i < repeats; ++i) launch(s);
    cudaEventRecord(b, s);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    Result r{0, ms / repeats};
    return r;
}

static double compute_bw(double bytes_per_iter, int repeats, float total_ms) {
    double total_bytes = bytes_per_iter * repeats;
    double seconds = total_ms / 1000.0;
    return total_bytes / seconds / 1e9;
}

// ===========================================================
// Main
// ===========================================================

int main(int argc, char** argv) {
    int dev = (argc > 1) ? atoi(argv[1]) : 0;
    check(cudaSetDevice(dev), "setdev");
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    fprintf(stderr, "GPU %d: %s, %.1f GB, sm_%d%d, %d SMs\n",
            dev, p.name, p.totalGlobalMem / 1e9,
            p.major, p.minor, p.multiProcessorCount);

    // Test sizes (bytes)
    std::vector<size_t> sizes = {
        16ULL * 1024 * 1024,        // 16 MB
        256ULL * 1024 * 1024,       // 256 MB
        2ULL * 1024 * 1024 * 1024,  // 2 GB
        5ULL * 1024 * 1024 * 1024,  // 5 GB (close to per-GPU weight footprint)
    };

    // Allocate the largest needed
    size_t maxbytes = sizes.back();
    void *A = nullptr, *B = nullptr;
    check(cudaMalloc(&A, maxbytes), "malloc A");
    check(cudaMalloc(&B, maxbytes), "malloc B");
    float* d_sink = nullptr;
    check(cudaMalloc(&d_sink, 16), "sink");

    // Random index buffer for gather
    size_t max_n4 = maxbytes / sizeof(float4);
    int* d_idx = nullptr;
    {
        std::vector<int> h_idx(64 * 1024);
        for (size_t i = 0; i < h_idx.size(); ++i) h_idx[i] = (int)((size_t)rand() % max_n4);
        check(cudaMalloc(&d_idx, h_idx.size() * sizeof(int)), "malloc idx");
        check(cudaMemcpy(d_idx, h_idx.data(), h_idx.size() * sizeof(int), cudaMemcpyHostToDevice), "h2d idx");
    }

    cudaStream_t s;
    cudaStreamCreate(&s);

    int repeats = 10;
    int blocks  = p.multiProcessorCount * 4;
    int threads = 256;

    printf("=========================================================================\n");
    printf("Pattern        Size        ms/iter     bytes(GB)   Bandwidth(GB/s)\n");
    printf("=========================================================================\n");

    for (size_t bytes : sizes) {
        size_t n4 = bytes / sizeof(float4);

        // ----- D2D cudaMemcpyAsync
        {
            for (int i = 0; i < 3; ++i) cudaMemcpyAsync(B, A, bytes, cudaMemcpyDeviceToDevice, s);
            cudaStreamSynchronize(s);
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            cudaEventRecord(a, s);
            for (int i = 0; i < repeats; ++i)
                cudaMemcpyAsync(B, A, bytes, cudaMemcpyDeviceToDevice, s);
            cudaEventRecord(b, s); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b);
            cudaEventDestroy(a); cudaEventDestroy(b);
            // D2D moves 2x bytes (read+write) but we report as effective copy throughput
            double bw_copy = (double)bytes * repeats * 2.0 / (ms / 1000.0) / 1e9;
            printf("D2D            %4zu MB    %7.3f     %7.3f     %8.1f  (read+write)\n",
                   bytes >> 20, ms / repeats, bytes / 1e9, bw_copy);
        }

        // ----- Kernel COPY (read+write 2x bytes)
        {
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            for (int i = 0; i < 3; ++i)
                k_copy_f4<<<blocks, threads, 0, s>>>((float4*)A, (float4*)B, n4);
            cudaStreamSynchronize(s);
            cudaEventRecord(a, s);
            for (int i = 0; i < repeats; ++i)
                k_copy_f4<<<blocks, threads, 0, s>>>((float4*)A, (float4*)B, n4);
            cudaEventRecord(b, s); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b);
            cudaEventDestroy(a); cudaEventDestroy(b);
            double bw = (double)bytes * repeats * 2.0 / (ms / 1000.0) / 1e9;
            printf("COPY kernel    %4zu MB    %7.3f     %7.3f     %8.1f  (read+write)\n",
                   bytes >> 20, ms / repeats, bytes / 1e9, bw);
        }

        // ----- Kernel READ (linear scan, 1x bytes read)
        {
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            for (int i = 0; i < 3; ++i)
                k_read_f4<<<blocks, threads, 0, s>>>((float4*)A, d_sink, n4);
            cudaStreamSynchronize(s);
            cudaEventRecord(a, s);
            for (int i = 0; i < repeats; ++i)
                k_read_f4<<<blocks, threads, 0, s>>>((float4*)A, d_sink, n4);
            cudaEventRecord(b, s); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b);
            cudaEventDestroy(a); cudaEventDestroy(b);
            double bw = (double)bytes * repeats / (ms / 1000.0) / 1e9;
            printf("READ kernel    %4zu MB    %7.3f     %7.3f     %8.1f  (read only)\n",
                   bytes >> 20, ms / repeats, bytes / 1e9, bw);
        }

        // ----- Kernel STRIDE (stride 256B = 16 float4)
        if (bytes <= (1ULL << 31)) {
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            size_t stride_elems = 16; // 256-byte stride
            for (int i = 0; i < 3; ++i)
                k_stride_read_f4<<<blocks, threads, 0, s>>>((float4*)A, d_sink, n4, stride_elems);
            cudaStreamSynchronize(s);
            cudaEventRecord(a, s);
            for (int i = 0; i < repeats; ++i)
                k_stride_read_f4<<<blocks, threads, 0, s>>>((float4*)A, d_sink, n4, stride_elems);
            cudaEventRecord(b, s); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b);
            cudaEventDestroy(a); cudaEventDestroy(b);
            double bw = (double)bytes * repeats / (ms / 1000.0) / 1e9;
            printf("STRIDE-256     %4zu MB    %7.3f     %7.3f     %8.1f  (read only, stride=16f4)\n",
                   bytes >> 20, ms / repeats, bytes / 1e9, bw);
        }

        // ----- Kernel GATHER (random index, fixed niter so workload constant)
        {
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            size_t niter = 64 * 1024; // matches d_idx
            size_t bytes_touched = niter * sizeof(float4);
            for (int i = 0; i < 3; ++i)
                k_gather_f4<<<blocks, threads, 0, s>>>((float4*)A, d_idx, d_sink, n4, niter);
            cudaStreamSynchronize(s);
            cudaEventRecord(a, s);
            for (int i = 0; i < repeats; ++i)
                k_gather_f4<<<blocks, threads, 0, s>>>((float4*)A, d_idx, d_sink, n4, niter);
            cudaEventRecord(b, s); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b);
            cudaEventDestroy(a); cudaEventDestroy(b);
            double bw = (double)bytes_touched * repeats / (ms / 1000.0) / 1e9;
            printf("GATHER 64Ki    on %4zu MB %7.4f     %7.4f     %8.2f  (random, 16B/access)\n",
                   bytes >> 20, ms / repeats, bytes_touched / 1e9, bw);
        }

        printf("-------------------------------------------------------------------------\n");
    }

    cudaFree(A); cudaFree(B); cudaFree(d_sink); cudaFree(d_idx);
    cudaStreamDestroy(s);
    return 0;
}
