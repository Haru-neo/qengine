// gdn_dev.cu — GDN chunked-prefill kernel timing harness.
// Times the three per-layer chunk kernels at the real 27B GDN shapes:
//   conv1d_update_silu_chunk (40 blocks, seq scan over 256 tok per channel)
//   gdn_chunk_step           (num_v=48 blocks, 1/SM, seq scan over 256 tok)
//   rms_norm_gated_chunk     (num_v*N = 12288 blocks, well-parallel)
// Reports ms/chunk/layer and the implied full-prefill GDN wall cost.
//
// Build: nvcc -O3 -arch=sm_70 -o tools/gdn_dev tools/gdn_dev.cu
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "../src/gdn_kernels.cuh"

#define CK(x) do{ cudaError_t _ck_e=(x); if(_ck_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s @ %d\n",cudaGetErrorString(_ck_e),__LINE__); exit(1);} }while(0)

static uint32_t rs=99173u; static float frand(){ rs^=rs<<13;rs^=rs>>17;rs^=rs<<5; return ((rs>>8)&0xffff)/65535.0f-0.5f; }

template<class F> static float timeit(F f,int it){
  cudaEvent_t s,e;CK(cudaEventCreate(&s));CK(cudaEventCreate(&e));
  f();CK(cudaDeviceSynchronize());
  std::vector<float> ts;
  for(int i=0;i<it;i++){CK(cudaEventRecord(s));f();CK(cudaEventRecord(e));CK(cudaEventSynchronize(e));float ms;CK(cudaEventElapsedTime(&ms,s,e));ts.push_back(ms);}
  std::sort(ts.begin(),ts.end());return ts[ts.size()/2];
}

int main(int argc,char**argv){
  const int N        = argc>1?atoi(argv[1]):256;   // chunk tokens
  const int num_k=16,num_v=48,k_dim=128,v_dim=128,kw=4;
  const int qkv_dim  = 2*num_k*k_dim + num_v*v_dim;  // 10240
  const int v_total  = num_v*v_dim;                  // 6144
  const int n_gdn_layers=48, n_chunks=(18000+N-1)/N; // 18K prefill
  printf("GDN dev: N=%d qkv_dim=%d v_total=%d  (18K prefill: %d GDN layers x %d chunks)\n",
         N,qkv_dim,v_total,n_gdn_layers,n_chunks);
  int it = getenv("ITERS")?atoi(getenv("ITERS")):50;

  // ---- buffers ----
  float *conv_state,*conv_w,*chunk_qkv_f32,*a_log,*dt_bias,*rec_state;
  half  *chunk_qkv_h,*chunk_a,*chunk_b,*chunk_core_out,*chunk_z,*chunk_normed;
  float *gnorm_w;
  CK(cudaMalloc(&conv_state,(size_t)qkv_dim*kw*sizeof(float)));
  CK(cudaMalloc(&conv_w,(size_t)qkv_dim*kw*sizeof(float)));
  CK(cudaMalloc(&chunk_qkv_f32,(size_t)N*qkv_dim*sizeof(float)));
  CK(cudaMalloc(&chunk_qkv_h,(size_t)N*qkv_dim*sizeof(half)));
  CK(cudaMalloc(&a_log,num_v*sizeof(float)));
  CK(cudaMalloc(&dt_bias,num_v*sizeof(float)));
  CK(cudaMalloc(&chunk_a,(size_t)N*num_v*sizeof(half)));
  CK(cudaMalloc(&chunk_b,(size_t)N*num_v*sizeof(half)));
  CK(cudaMalloc(&rec_state,(size_t)num_v*k_dim*v_dim*sizeof(float)));
  CK(cudaMalloc(&chunk_core_out,(size_t)N*v_total*sizeof(half)));
  CK(cudaMalloc(&chunk_z,(size_t)N*v_total*sizeof(half)));
  CK(cudaMalloc(&chunk_normed,(size_t)N*v_total*sizeof(half)));
  CK(cudaMalloc(&gnorm_w,v_dim*sizeof(float)));
  // init small
  { std::vector<float> tmp((size_t)N*qkv_dim); for(auto&v:tmp)v=frand()*0.5f;
    CK(cudaMemcpy(chunk_qkv_f32,tmp.data(),tmp.size()*sizeof(float),cudaMemcpyHostToDevice)); }
  { std::vector<float> tmp(qkv_dim*kw); for(auto&v:tmp)v=frand(); CK(cudaMemcpy(conv_w,tmp.data(),tmp.size()*sizeof(float),cudaMemcpyHostToDevice)); }
  { std::vector<float> tmp(num_v,-2.0f); CK(cudaMemcpy(a_log,tmp.data(),tmp.size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dt_bias,tmp.data(),tmp.size()*sizeof(float),cudaMemcpyHostToDevice)); }
  CK(cudaMemset(conv_state,0,(size_t)qkv_dim*kw*sizeof(float)));
  CK(cudaMemset(rec_state,0,(size_t)num_v*k_dim*v_dim*sizeof(float)));
  { std::vector<float> tmp(v_dim,1.0f); CK(cudaMemcpy(gnorm_w,tmp.data(),tmp.size()*sizeof(float),cudaMemcpyHostToDevice)); }
  CK(cudaMemset(chunk_a,0,(size_t)N*num_v*sizeof(half)));
  CK(cudaMemset(chunk_b,0,(size_t)N*num_v*sizeof(half)));
  CK(cudaMemset(chunk_z,0,(size_t)N*v_total*sizeof(half)));

  // opt-in 96KB smem for gdn_chunk_step
  CK(cudaFuncSetAttribute((const void*)gdn_chunk_step,cudaFuncAttributeMaxDynamicSharedMemorySize,96*1024));

  // ---- conv1d ----
  { int threads=min(qkv_dim,256), blocks=(qkv_dim+threads-1)/threads;
    auto L=[&]{ conv1d_update_silu_chunk<<<blocks,threads>>>(conv_state,chunk_qkv_h,conv_w,chunk_qkv_f32,qkv_dim,kw,N); };
    float ms=timeit(L,it); printf("  conv1d_chunk      %2d blk x%3d  %7.4f ms/layer\n",blocks,threads,ms);
    printf("    -> 18K total: %6.2f ms (1GPU)  %6.2f ms (/3 pipelined)\n",ms*n_gdn_layers*n_chunks, ms*n_gdn_layers*n_chunks/3.0); }

  // ---- gdn_chunk_step (the suspect) ----
  { int threads=min(v_dim,128); int state_len=k_dim*v_dim;
    int gdn_smem=(state_len+2*k_dim+1+96)*sizeof(float);
    auto L=[&]{ gdn_chunk_step<<<num_v,threads,gdn_smem>>>(chunk_qkv_f32,a_log,dt_bias,chunk_a,chunk_b,rec_state,chunk_core_out,N,num_k,num_v,k_dim,v_dim); };
    float ms=timeit(L,it); printf("  gdn_chunk_step    %2d blk x%3d smem=%dB  %7.4f ms/layer\n",num_v,threads,gdn_smem,ms);
    printf("    -> 18K total: %6.2f ms (1GPU)  %6.2f ms (/3 pipelined)\n",ms*n_gdn_layers*n_chunks, ms*n_gdn_layers*n_chunks/3.0); }

  // ---- rms_norm_gated_chunk ----
  { int total_blocks=num_v*N, threads=min(v_dim,128);
    auto L=[&]{ rms_norm_gated_chunk_kernel<<<total_blocks,threads,128*sizeof(float)>>>(chunk_core_out,chunk_z,gnorm_w,chunk_normed,num_v,v_dim,N,1e-6f); };
    float ms=timeit(L,it); printf("  rmsg_chunk     %5d blk x%3d  %7.4f ms/layer\n",total_blocks,threads,ms);
    printf("    -> 18K total: %6.2f ms (1GPU)  %6.2f ms (/3 pipelined)\n",ms*n_gdn_layers*n_chunks, ms*n_gdn_layers*n_chunks/3.0); }

  return 0;
}
