// Timing microbench for the dense prefill FA kernel at the 18K last-chunk shape.
// Lets us sweep occupancy (via -maxrregcount at compile) + BM without the 110s
// model reload. Times the split kernel only (97% of attn cost per profiling).
#include <cstdio>
#include <vector>
#include <cstdlib>
#include "../src/gguf.h"
#include "../src/attention.cuh"

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA err %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

template<int HD,int GQA,int BM,int BLOCK,int K>
double time_split(int num_q,int num_kv,int seq,int sub_n,int iters){
    int ATTN_NB = sub_n;
    int active_end_max = seq;                 // last chunk attends ~all keys
    int start_pos = seq - sub_n;
    // buffers
    half *q,*kc,*vc; float *pm,*pl,*po;
    size_t qn=(size_t)sub_n*num_q*HD;
    size_t kvn=(size_t)seq*num_kv*HD;
    size_t pn=(size_t)num_q*ATTN_NB*K;
    CK(cudaMalloc(&q,qn*2)); CK(cudaMalloc(&kc,kvn*2)); CK(cudaMalloc(&vc,kvn*2));
    CK(cudaMalloc(&pm,pn*4)); CK(cudaMalloc(&pl,pn*4)); CK(cudaMalloc(&po,pn*HD*4));
    CK(cudaMemset(q,0,qn*2)); CK(cudaMemset(kc,0,kvn*2)); CK(cudaMemset(vc,0,kvn*2));
    int dyn_smem = GQA*HD*sizeof(half) + 2*BM*HD*sizeof(half) + GQA*BM*sizeof(float);
    dim3 fg(num_kv, sub_n, K);
    float scale=0.0625f;
    void* fn=(void*)flash_attn_chunk_fused_split<HD,GQA,BM,BLOCK,K>;
    if(dyn_smem>48*1024) cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,96*1024);
    // warmup
    for(int i=0;i<3;i++)
        flash_attn_chunk_fused_split<HD,GQA,BM,BLOCK,K><<<fg,BLOCK,dyn_smem>>>(
            q,kc,vc,pm,pl,po,num_q,num_kv,start_pos,sub_n,ATTN_NB,active_end_max,scale);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b);
    cudaEventRecord(a);
    for(int i=0;i<iters;i++)
        flash_attn_chunk_fused_split<HD,GQA,BM,BLOCK,K><<<fg,BLOCK,dyn_smem>>>(
            q,kc,vc,pm,pl,po,num_q,num_kv,start_pos,sub_n,ATTN_NB,active_end_max,scale);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms=0; cudaEventElapsedTime(&ms,a,b);
    cudaFree(q);cudaFree(kc);cudaFree(vc);cudaFree(pm);cudaFree(pl);cudaFree(po);
    return ms/iters;
}

int main(int argc,char**argv){
    int seq = argc>1?atoi(argv[1]):14608;
    int sub_n=16, num_q=24, num_kv=4, iters=200;
    cudaFuncAttributes at;
    printf("seq=%d sub_n=%d num_q=%d  (one sub-chunk launch, grid=%dx%d x K)\n",seq,sub_n,num_q,num_kv,sub_n);
    cudaFuncGetAttributes(&at,(void*)flash_attn_chunk_fused_split<256,6,32,256,4>);
    printf("BM=32: regs=%d  ms/launch=%.4f\n", at.numRegs, time_split<256,6,32,256,4>(num_q,num_kv,seq,sub_n,iters));
    cudaFuncGetAttributes(&at,(void*)flash_attn_chunk_fused_split<256,6,16,256,4>);
    printf("BM=16: regs=%d  ms/launch=%.4f\n", at.numRegs, time_split<256,6,16,256,4>(num_q,num_kv,seq,sub_n,iters));
    cudaFuncGetAttributes(&at,(void*)flash_attn_chunk_fused_split<256,6,64,256,4>);
    printf("BM=64: regs=%d  ms/launch=%.4f\n", at.numRegs, time_split<256,6,64,256,4>(num_q,num_kv,seq,sub_n,iters));
    return 0;
}
