// Standalone occupancy probe for the dense prefill FA kernel.
// Confirms whether flash_attn_chunk_fused_split is occupancy-limited.
#include <cstdio>
#include <vector>
#include "../src/gguf.h"
#include "../src/attention.cuh"

template<int HD,int GQA,int BM,int BLOCK,int K>
void probe(const char* tag){
    void* fn = (void*)flash_attn_chunk_fused_split<HD,GQA,BM,BLOCK,K>;
    int dyn_smem = GQA*HD*sizeof(half) + 2*BM*HD*sizeof(half) + GQA*BM*sizeof(float);
    cudaFuncAttributes a; cudaFuncGetAttributes(&a, fn);
    int blocks=0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, BLOCK, dyn_smem);
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    int warps_per_sm = blocks * (BLOCK/32);
    int max_warps = p.maxThreadsPerMultiProcessor/32;
    printf("[%s] BM=%d BLOCK=%d K=%d\n", tag, BM, BLOCK, K);
    printf("   regs/thread=%d  static_smem=%zu  dyn_smem=%d B (%.1fKB)\n",
           a.numRegs, a.sharedSizeBytes, dyn_smem, dyn_smem/1024.0);
    printf("   -> %d blocks/SM, %d warps/SM of %d max = %.0f%% occupancy\n",
           blocks, warps_per_sm, max_warps, 100.0*warps_per_sm/max_warps);
    // What limits it? regs vs smem
    int reg_limit = p.regsPerMultiprocessor / (a.numRegs * BLOCK);
    int smem_limit = dyn_smem>0 ? (int)(p.sharedMemPerMultiprocessor / dyn_smem) : 999;
    printf("   limiter: reg_cap=%d blk/SM, smem_cap=%d blk/SM\n", reg_limit, smem_limit);
}

int main(){
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("GPU: %s  SMs=%d  maxThr/SM=%d  smem/SM=%zuKB  regs/SM=%d\n\n",
           p.name, p.multiProcessorCount, p.maxThreadsPerMultiProcessor,
           p.sharedMemPerMultiprocessor/1024, p.regsPerMultiprocessor);
    probe<256,6,32,256,4>("27B dense default");
    probe<256,6,64,256,4>("27B BM=64");
    probe<256,6,16,256,4>("27B BM=16");
    return 0;
}
