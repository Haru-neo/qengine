// Measures CUDA kernel launch overhead on CMP 100-210 (sm_70) and whether
// a sequence of small kernels (mimicking one GPU's layer span of the DFlash
// verify) is launch-starved, plus a CUDA-graph replay A/B.
//
// Build: nvcc -O3 -arch=sm_70 -o /tmp/bench_launch tools/bench_launch_overhead.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>

// A kernel that does a fixed amount of FMA work so we can dial its duration.
__global__ void busy_kernel(float* out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = out[i];
    #pragma unroll 1
    for (int k = 0; k < iters; k++) {
        x = x * 1.0000001f + 0.0000001f;
    }
    out[i] = x;
}

// Effectively-empty kernel (single thread, no work) — pure launch latency.
__global__ void empty_kernel(float* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] += 0.0f;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    cudaSetDevice(argc > 1 ? atoi(argv[1]) : 2);  // default GPU2 (free)
    float* d;
    cudaMalloc(&d, 8 * 5120 * sizeof(float));
    cudaMemset(d, 0, 8 * 5120 * sizeof(float));

    // 1) Raw empty-kernel launch latency (back-to-back on default stream).
    const int N_EMPTY = 5000;
    cudaDeviceSynchronize();
    double t0 = now_ms();
    for (int i = 0; i < N_EMPTY; i++) empty_kernel<<<1, 32>>>(d);
    cudaDeviceSynchronize();
    double t1 = now_ms();
    double per_launch_us = (t1 - t0) * 1000.0 / N_EMPTY;
    printf("Empty kernel back-to-back: %.3f us/launch (queue+exec, %d launches)\n",
           per_launch_us, N_EMPTY);

    // True host-side launch overhead: launch without sync between (queue is
    // deep) vs the dispatch cost. Measure single launch issue cost by timing
    // a big batch then dividing (above already does this). Now measure the
    // GPU-side execution of one empty kernel via events.
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    empty_kernel<<<1,32>>>(d);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float exec_ms = 0; cudaEventElapsedTime(&exec_ms, e0, e1);
    printf("Single empty kernel GPU exec: %.3f us\n", exec_ms * 1000.0);

    // 2) Mimic one GPU's layer span of DFlash verify.
    // 3-GPU split: ~21 layers/GPU. Pattern [GDN,GDN,GDN,Attn]. Per layer:
    //   GDN ~12 launches, MLP ~8, Attn ~40 (incl N=8 * 3 attn-compute).
    // Avg over a span of 21 layers: ~16 GDN, ~5 Attn, 21 MLP.
    //   16*12 + 5*40 + 21*8 = 192 + 200 + 168 = 560 launches / GPU span.
    // We approximate by launching K small kernels each doing some real work.
    // We sweep the per-kernel GPU duration to find the launch-starvation point.
    const int SPAN_LAUNCHES = 560;
    int sweep_iters[] = {0, 2, 5, 10, 25, 50, 100, 200, 400};
    int grid = (8 * 5120 + 255) / 256;  // realistic-ish grid for N=8 row-batched
    printf("\nSpan of %d launches (one GPU layer-span), default stream, no inter-sync:\n",
           SPAN_LAUNCHES);
    printf("%-12s %-14s %-14s %-14s\n", "work_iters", "wall_ms", "per_launch_us", "gpu_busy_est_us");
    for (int si = 0; si < (int)(sizeof(sweep_iters)/sizeof(int)); si++) {
        int it = sweep_iters[si];
        // measure pure GPU exec time of one such kernel via events
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        busy_kernel<<<grid, 256>>>(d, 8*5120, it);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float one_ms = 0; cudaEventElapsedTime(&one_ms, e0, e1);

        cudaDeviceSynchronize();
        double s0 = now_ms();
        for (int i = 0; i < SPAN_LAUNCHES; i++) busy_kernel<<<grid, 256>>>(d, 8*5120, it);
        cudaDeviceSynchronize();
        double s1 = now_ms();
        double wall = s1 - s0;
        double pl = wall * 1000.0 / SPAN_LAUNCHES;
        double gpu_busy = one_ms * 1000.0;  // us
        printf("%-12d %-14.3f %-14.3f %-14.3f  %s\n", it, wall, pl, gpu_busy,
               (pl > gpu_busy * 1.15) ? "<-- LAUNCH-STARVED" : "");
    }

    // 3) CUDA graph A/B on a representative span (work_iters=25, ~near the
    //    transition region). Capture the span, then replay, compare to stream.
    {
        int it = 25;
        // warm
        for (int i = 0; i < SPAN_LAUNCHES; i++) busy_kernel<<<grid,256>>>(d, 8*5120, it);
        cudaDeviceSynchronize();

        cudaStream_t cap;
        cudaStreamCreate(&cap);

        // stream baseline (non-default stream, fair vs graph)
        cudaDeviceSynchronize();
        double a0 = now_ms();
        for (int rep = 0; rep < 20; rep++)
            for (int i = 0; i < SPAN_LAUNCHES; i++)
                busy_kernel<<<grid,256,0,cap>>>(d, 8*5120, it);
        cudaStreamSynchronize(cap);
        double a1 = now_ms();
        double stream_ms = (a1 - a0) / 20.0;

        // capture graph
        cudaGraph_t graph; cudaGraphExec_t exec;
        cudaStreamBeginCapture(cap, cudaStreamCaptureModeThreadLocal);
        for (int i = 0; i < SPAN_LAUNCHES; i++)
            busy_kernel<<<grid,256,0,cap>>>(d, 8*5120, it);
        cudaStreamEndCapture(cap, &graph);
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

        // warm replay
        cudaGraphLaunch(exec, cap); cudaStreamSynchronize(cap);

        double g0 = now_ms();
        for (int rep = 0; rep < 20; rep++) {
            cudaGraphLaunch(exec, cap);
        }
        cudaStreamSynchronize(cap);
        double g1 = now_ms();
        double graph_ms = (g1 - g0) / 20.0;

        printf("\nCUDA graph A/B (work_iters=%d, %d launches/span):\n", it, SPAN_LAUNCHES);
        printf("  stream replay : %.3f ms/span (%.3f us/launch)\n",
               stream_ms, stream_ms*1000.0/SPAN_LAUNCHES);
        printf("  graph  replay : %.3f ms/span (%.3f us/launch)\n",
               graph_ms, graph_ms*1000.0/SPAN_LAUNCHES);
        printf("  speedup       : %.2fx\n", stream_ms / graph_ms);
    }
    return 0;
}
