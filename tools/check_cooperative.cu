// CMP 100-210 cooperative launch capability check.
// CMP locks down many GPU features (CUPTI, HMMA, etc) so verify before
// committing to cooperative_groups::this_grid().sync() architecture.

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

__global__ void coop_test(int* out, int N) {
    auto grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) out[tid] = tid;
    grid.sync();  // GLOBAL barrier across all blocks
    if (tid < N) out[tid] = out[(tid + 1) % N];  // post-barrier read
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s, sm_%d%d, %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);
    int coop;
    cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0);
    printf("cudaDevAttrCooperativeLaunch = %d\n", coop);
    if (!coop) {
        printf("REJECTED: CMP blocks cooperative launch\n");
        return 1;
    }

    const int N = 64;
    int *d_out;
    cudaMalloc(&d_out, N * sizeof(int));

    void* args[] = { &d_out, (void*)&N };
    dim3 grid(2), block(32);
    cudaError_t err = cudaLaunchCooperativeKernel((const void*)coop_test, grid, block, args, 0, 0);
    if (err != cudaSuccess) {
        printf("cudaLaunchCooperativeKernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();
    int h[N]; cudaMemcpy(h, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int expected = (i + 1) % N;
        if (h[i] != expected) { ok = false; break; }
    }
    printf("cooperative launch: %s\n", ok ? "WORKS" : "BROKEN");
    return ok ? 0 : 2;
}
