// CMP Volta sm_70: does DP4A (int8) overlap with HFMA2 (fp16 SIMD) INTER-WARP?
// If the INT-pipe (dp4a) and FP-pipe (hfma2) are separate execution units and the
// 4 sub-partition schedulers can keep both busy, then warps doing dp4a and warps
// doing hfma2 should run CONCURRENTLY -> combined throughput approaching 17+20 TF,
// breaking the "GEMM at 17 TF dp4a = wall". Memory: intra-thread interleave = 0.65x
// (dead-end); inter-warp DP4A+HFMA2 = UNMEASURED (only HMMA+HFMA2 was tested).
//
// Method: same per-warp work in solo vs concurrent.
//   t_dp   = time(dp4a-only, Wsolo warps)
//   t_hf   = time(hfma2-only, Wsolo warps)
//   t_con  = time(concurrent: Wsolo dp4a-warps + Wsolo hfma2-warps = 2*Wsolo total)
//   overlap = (t_dp + t_hf) / t_con   ->  ~2.0 = full overlap (wall broken), ~1.0 = serial
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

#define NACC 8
#define ITERS 200000

__device__ __forceinline__ void dp4a_body(int lane, int* sink) {
    int acc[NACC];
    #pragma unroll
    for (int j=0;j<NACC;j++) acc[j] = lane + j;
    int a = 0x01020304 ^ lane, b = 0x04030201 ^ (lane<<1);
    for (int i=0;i<ITERS;i++) {
        #pragma unroll
        for (int j=0;j<NACC;j++) acc[j] = __dp4a(a, b, acc[j]);
    }
    int s=0;
    #pragma unroll
    for (int j=0;j<NACC;j++) s ^= acc[j];
    if (s==0x7fffffff) *sink = s;   // prevent dead-code elim, never true
}

__device__ __forceinline__ void hfma2_body(int lane, int* sink) {
    half2 acc[NACC];
    #pragma unroll
    for (int j=0;j<NACC;j++) acc[j] = __float2half2_rn(0.001f*(lane+j));
    half2 a = __float2half2_rn(1.0001f), b = __float2half2_rn(0.9999f);
    for (int i=0;i<ITERS;i++) {
        #pragma unroll
        for (int j=0;j<NACC;j++) acc[j] = __hfma2(a, b, acc[j]);
    }
    float s=0;
    #pragma unroll
    for (int j=0;j<NACC;j++) s += __low2float(acc[j]) + __high2float(acc[j]);
    if (s<-1e30f) *sink = (int)s;
}

__global__ void k_dp4a(int* sink){ dp4a_body(threadIdx.x&31, sink); }
__global__ void k_hfma2(int* sink){ hfma2_body(threadIdx.x&31, sink); }
// concurrent: even warps dp4a (INT pipe), odd warps hfma2 (FP pipe)
__global__ void k_concur(int* sink){
    int warp = threadIdx.x >> 5;
    if ((warp & 1)==0) dp4a_body(threadIdx.x&31, sink);
    else               hfma2_body(threadIdx.x&31, sink);
}

float run(void(*launch)(int*,dim3,dim3), int* sink, dim3 g, dim3 b){
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    launch(sink,g,b); cudaDeviceSynchronize();          // warmup
    cudaEventRecord(s); launch(sink,g,b); cudaEventRecord(e);
    cudaEventSynchronize(e); float ms; cudaEventElapsedTime(&ms,s,e);
    cudaEventDestroy(s); cudaEventDestroy(e); return ms;
}
void L_dp(int*sk,dim3 g,dim3 b){ k_dp4a<<<g,b>>>(sk); }
void L_hf(int*sk,dim3 g,dim3 b){ k_hfma2<<<g,b>>>(sk); }
void L_co(int*sk,dim3 g,dim3 b){ k_concur<<<g,b>>>(sk); }

int main(){
    int dev=0; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p,dev);
    int SM=p.multiProcessorCount;
    int* sink; cudaMalloc(&sink,sizeof(int));
    // Wsolo: 8 warps/SM (256 thr/block) over all SMs -> saturate one pipe.
    // concurrent doubles to 16 warps/SM (8 dp4a + 8 hfma2 per SM).
    dim3 bSolo(256), gSolo(SM*1);     // 256 thr = 8 warps; 1 block/SM
    dim3 bCon(512),  gCon(SM*1);      // 512 thr = 16 warps; half/half per block
    printf("CMP sm_%d%d, %d SMs\n", p.major,p.minor,SM);

    float t_dp = run(L_dp, sink, gSolo, bSolo);
    float t_hf = run(L_hf, sink, gSolo, bSolo);
    float t_co = run(L_co, sink, gCon,  bCon);

    // ops: solo = SM blocks * 256 thr * ITERS * NACC * (dp4a 8 intops | hfma2 4 flops)
    double warps_solo = (double)SM*8;     // 256/32
    double dp_ops = warps_solo*32.0*ITERS*NACC*8.0;
    double hf_ops = warps_solo*32.0*ITERS*NACC*4.0;
    printf("DP4A  solo : %.2f ms  -> %.2f TOPS(int8 MAC*2)\n", t_dp, dp_ops/(t_dp*1e-3)/1e12);
    printf("HFMA2 solo : %.2f ms  -> %.2f TFLOP\n",            t_hf, hf_ops/(t_hf*1e-3)/1e12);
    // concurrent: SM blocks * 512 thr, half dp4a (8 warps/blk... actually 8 dp + 8 hf per block)
    // dp ops in concur = SM * 8warps*32 * ITERS*NACC*8 ; hf = SM*8warps*32*ITERS*NACC*4
    double co_dp = (double)SM*8*32.0*ITERS*NACC*8.0;
    double co_hf = (double)SM*8*32.0*ITERS*NACC*4.0;
    printf("CONCUR     : %.2f ms  -> dp %.2f TOPS + hf %.2f TFLOP simultaneously\n",
        t_co, co_dp/(t_co*1e-3)/1e12, co_hf/(t_co*1e-3)/1e12);
    printf(">>> OVERLAP factor = (t_dp+t_hf)/t_co = %.3f   (2.0=full overlap/wall broken, 1.0=serial)\n",
        (t_dp+t_hf)/t_co);
    cudaFree(sink); return 0;
}
