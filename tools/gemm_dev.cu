// gemm_dev.cu — Q8_0×Q8_1 dp4a GEMM development/measurement harness.
//
// Mirrors tools/fa_dev.cu: self-contained, includes the engine's kernels,
// generates random q8 data, computes an fp32 reference, and benchmarks the
// production mode-9 v2 GEMM + experimental double-buffered variants against it
// at the REAL 27B prefill shapes. Reports effective TOPS and %DP4A-peak.
//
// Build: nvcc -O3 -arch=sm_70 -o tools/gemm_dev tools/gemm_dev.cu
// Run:   ./tools/gemm_dev            (sweeps all real shapes, all kernels)
//        ./tools/gemm_dev M K N      (one custom shape)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
// Minimal ggml_type stub so quant_gemv.cuh's dispatchers compile (we call the
// kernel templates directly; the enum values only need to exist).
enum ggml_type { GGML_TYPE_F32=0, GGML_TYPE_F16=1, GGML_TYPE_Q8_0=8, GGML_TYPE_Q5_K=13, GGML_TYPE_Q6_K=14 };
#include "../src/quant_gemv.cuh"

#define CK(x) do{ cudaError_t _ck_e=(x); if(_ck_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s @ %s:%d\n",cudaGetErrorString(_ck_e),__FILE__,__LINE__); exit(1);} }while(0)

// CMP 100-210 measured DP4A peak (per-GPU). %peak printed relative to this.
static const double DP4A_PEAK_TOPS = 17.2;

// ---- random q8 data (self-consistent: ref + kernels read the same buffers) ----
static uint32_t rng_state = 1234567u;
static inline uint32_t xr(){ rng_state^=rng_state<<13; rng_state^=rng_state>>17; rng_state^=rng_state<<5; return rng_state; }
static inline int8_t  rq(){ return (int8_t)((xr()&0xff)-128); }
static inline float   rd(){ return 0.005f + (xr()%100)*0.0004f; }   // small positive scales

// Fill M×(K/32) weight blocks (row-major: row m, block b at m*bpr+b).
static void fill_weight(std::vector<block_q8_0_aligned>& W, int M, int K){
    int bpr=K/32; W.resize((size_t)M*bpr);
    for(size_t i=0;i<W.size();i++){ for(int j=0;j<32;j++) W[i].qs[j]=rq(); W[i].pad=0; W[i].d=__float2half(rd()); }
}
// Fill N×(K/32) input blocks (row-major: token n, block b at n*bpr+b).
static void fill_input(std::vector<block_q8_1>& X, int N, int K){
    int bpr=K/32; X.resize((size_t)N*bpr);
    for(size_t i=0;i<X.size();i++){ for(int j=0;j<32;j++) X[i].qs[j]=rq();
        float d=rd(); float s=0; for(int j=0;j<32;j++) s+=d*(float)X[i].qs[j];
        X[i].ds=make_half2(__float2half(d),__float2half(s)); }
}

// ---- fp32 reference: same math as gemm_q8_0_q8_1_v2, output layout Y[n*M+m] ----
__global__ void ref_gemm(const block_q8_0_aligned* __restrict__ W,
                         const block_q8_1* __restrict__ X,
                         half* __restrict__ Y, int M, int N, int K){
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if(m>=M || n>=N) return;
    int bpr=K/32; float acc=0.f;
    const block_q8_0_aligned* wr = W + (size_t)m*bpr;
    const block_q8_1*         xr = X + (size_t)n*bpr;
    for(int b=0;b<bpr;b++){
        const int* wp=(const int*)wr[b].qs; const int* xp=(const int*)xr[b].qs;
        int isum=0;
        #pragma unroll
        for(int j=0;j<8;j++) isum=__dp4a(wp[j],xp[j],isum);
        acc += __half2float(wr[b].d)*__half2float(xr[b].ds.x)*(float)isum;
    }
    Y[(size_t)n*M+m]=__float2half(acc);
}

static float maxreldiff(const std::vector<half>&a,const std::vector<half>&b){
    float md=0;
    for(size_t i=0;i<a.size();i++){
        float x=__half2float(a[i]),y=__half2float(b[i]);
        float den=fmaxf(1e-3f,fabsf(y)); md=fmaxf(md,fabsf(x-y)/den);
    }
    return md;
}

struct Shape{ const char* name; int M,K,N; };

// timing helper: returns median ms over `iters`
template<class F>
static float timeit(F launch,int iters){
    cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    launch(); CK(cudaDeviceSynchronize());           // warm
    std::vector<float> ts;
    for(int i=0;i<iters;i++){ CK(cudaEventRecord(s)); launch(); CK(cudaEventRecord(e));
        CK(cudaEventSynchronize(e)); float ms; CK(cudaEventElapsedTime(&ms,s,e)); ts.push_back(ms);}
    std::sort(ts.begin(),ts.end()); return ts[ts.size()/2];
}

static void report(const char* tag,const Shape&S,float ms){
    double flop=2.0*S.M*S.K*S.N;          // 2 ops per MAC
    double tops=flop/(ms*1e-3)/1e12;
    printf("  %-22s %8.4f ms  %7.2f TOPS  %5.1f%% peak\n",tag,ms,tops,100.0*tops/DP4A_PEAK_TOPS);
}

int main(int argc,char**argv){
    std::vector<Shape> shapes;
    if(argc>=4){ static char nm[]="custom"; shapes.push_back({nm,atoi(argv[1]),atoi(argv[2]),atoi(argv[3])}); }
    else shapes = {
        {"mlp_gate_up M17408",17408,5120,256},
        {"mlp_down    M5120 ", 5120,17408,256},
        {"gdn_qkv     M10240",10240,5120,256},
        {"attn_q      M6144 ", 6144,5120,256},
        {"attn_out    M5120 ", 5120,6144,256},
    };
    int iters = getenv("ITERS")?atoi(getenv("ITERS")):30;

    for(auto&S:shapes){
        printf("== %s  (M=%d K=%d N=%d, %.1f GFLOP) ==\n",S.name,S.M,S.K,S.N,2.0*S.M*S.K*S.N/1e9);
        std::vector<block_q8_0_aligned> hW; std::vector<block_q8_1> hX;
        fill_weight(hW,S.M,S.K); fill_input(hX,S.N,S.K);
        block_q8_0_aligned* dW; block_q8_1* dX; half* dY; half* dRef;
        CK(cudaMalloc(&dW,hW.size()*sizeof(block_q8_0_aligned)));
        CK(cudaMalloc(&dX,hX.size()*sizeof(block_q8_1)));
        CK(cudaMalloc(&dY,(size_t)S.M*S.N*sizeof(half)));
        CK(cudaMalloc(&dRef,(size_t)S.M*S.N*sizeof(half)));
        CK(cudaMemcpy(dW,hW.data(),hW.size()*sizeof(block_q8_0_aligned),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dX,hX.data(),hX.size()*sizeof(block_q8_1),cudaMemcpyHostToDevice));

        // reference
        { dim3 g((S.M+127)/128,S.N); ref_gemm<<<g,128>>>(dW,dX,dRef,S.M,S.N,S.K); CK(cudaDeviceSynchronize()); }
        std::vector<half> hRef((size_t)S.M*S.N); CK(cudaMemcpy(hRef.data(),dRef,hRef.size()*sizeof(half),cudaMemcpyDeviceToHost));

        auto check=[&](const char* tag){
            std::vector<half> hY((size_t)S.M*S.N); CK(cudaMemcpy(hY.data(),dY,hY.size()*sizeof(half),cudaMemcpyDeviceToHost));
            float md=maxreldiff(hY,hRef); if(md>0.02f) printf("  !! %s maxreldiff=%.4f\n",tag,md);
        };

        // --- production mode-9 v2 tile <32,128,2,8,4> ---
        if(S.M%32==0 && S.N%128==0){
            constexpr int BM=32,BN=128,TM=2,TN=8,BK=4;
            dim3 grid(S.M/BM,S.N/BN); int threads=(BM/TM)*(BN/TN);
            int sm=(BM+BN)*BK*sizeof(block_q8_0_aligned);
            auto L=[&]{ gemm_q8_0_q8_1_v2<BM,BN,TM,TN,BK><<<grid,threads,sm>>>(dW,dX,dY,S.M,S.N,S.K); };
            float ms=timeit(L,iters); L(); CK(cudaDeviceSynchronize()); check("v2_mode9");
            report("v2_mode9 <32,128,2,8,4>",S,ms);
        }
        // --- fused2 (gate+up: 2 weights share X) — only meaningful for MLP gate/up shape ---
        if(S.M%32==0 && S.N%128==0){
            constexpr int BM=32,BN=128,TM=2,TN=8,BK=4;
            block_q8_0_aligned* dWu; CK(cudaMalloc(&dWu,hW.size()*sizeof(block_q8_0_aligned)));
            CK(cudaMemcpy(dWu,hW.data(),hW.size()*sizeof(block_q8_0_aligned),cudaMemcpyHostToDevice));
            half* dYu; CK(cudaMalloc(&dYu,(size_t)S.M*S.N*sizeof(half)));
            dim3 grid(S.M/BM,S.N/BN); int threads=(BM/TM)*(BN/TN);
            int sm=(2*BM+BN)*BK*sizeof(block_q8_0_aligned);
            auto L=[&]{ gemm_q8_0_q8_1_fused2<BM,BN,TM,TN,BK><<<grid,threads,sm>>>(dW,dWu,dX,dY,dYu,S.M,S.N,S.K); };
            float ms=timeit(L,iters);
            // fused2 does 2x the MACs (gate+up); report as the cost of BOTH matmuls
            double flop=2.0*2.0*S.M*S.K*S.N; double tops=flop/(ms*1e-3)/1e12;
            printf("  %-22s %8.4f ms  %7.2f TOPS  %5.1f%% peak  (gate+up combined)\n","fused2 gate+up",ms,tops,100.0*tops/DP4A_PEAK_TOPS);
            CK(cudaFree(dWu));CK(cudaFree(dYu));
        }

        CK(cudaFree(dW));CK(cudaFree(dX));CK(cudaFree(dY));CK(cudaFree(dRef));
    }
    return 0;
}
