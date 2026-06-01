// Dev harness for redesigning the prefill FA split kernel.
// - naive reference attention (correct, slow)
// - current kernel (flash_attn_chunk_fused_split + merge) vs reference
// - NEW kernel slot vs reference + timing
// Iterate the new kernel here (1s rebuild) before touching the engine.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "../src/gguf.h"
#include "../src/attention.cuh"

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

static const int HD=256;

// Naive reference: one thread per (t, qh). Causal, GQA via qh/GQA -> kv_head.
__global__ void naive_attn(const half* q,const half* k,const half* v,half* out,
                           int num_q,int num_kv,int start_pos,int sub_n,int gqa,float scale){
    int t = blockIdx.x;        // 0..sub_n-1
    int qh= blockIdx.y;        // 0..num_q-1
    if(t>=sub_n||qh>=num_q) return;
    int kvh = qh/gqa;
    int abs_pos = start_pos + t;
    const half* qp = q + ((size_t)t*num_q + qh)*HD;
    // online softmax
    float m=-1e30f,l=0.f,acc[HD];
    for(int d=0;d<HD;d++) acc[d]=0.f;
    for(int j=0;j<=abs_pos;j++){
        const half* kp = k + ((size_t)j*num_kv + kvh)*HD;
        float s=0.f;
        for(int d=0;d<HD;d++) s += __half2float(qp[d])*__half2float(kp[d]);
        s*=scale;
        float mn = fmaxf(m,s);
        float corr = expf(m-mn);
        float p = expf(s-mn);
        l = l*corr + p;
        const half* vp = v + ((size_t)j*num_kv + kvh)*HD;
        for(int d=0;d<HD;d++) acc[d] = acc[d]*corr + p*__half2float(vp[d]);
        m=mn;
    }
    half* op = out + ((size_t)t*num_q + qh)*HD;
    for(int d=0;d<HD;d++) op[d]=__float2half(acc[d]/l);
}

// ===== Instrumented copy of flash_attn_chunk_fused_split (phase timing) =====
__device__ unsigned long long g_phase[4]; // 0=load 1=score 2=value 3=tilecnt
template<int GQA,int BM,int BLOCK,int K_SPLITS>
__global__ void __launch_bounds__(BLOCK,4) fa_prof(
    const half* __restrict__ q_chunk,const half* __restrict__ k_cache,const half* __restrict__ v_cache,
    float* __restrict__ part_m,float* __restrict__ part_l,float* __restrict__ part_o,
    int num_q,int num_kv,int start_pos,int sub_n,int sub_n_max,int active_end_max,float scale){
    constexpr int N_WARPS=BLOCK/32; constexpr int LANE_D=HD/32;
    int kv_head=blockIdx.x,t_idx=blockIdx.y,split_idx=blockIdx.z;
    if(t_idx>=sub_n) return;
    int abs_pos=start_pos+t_idx, active_end_t=abs_pos+1;
    int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
    int total_tiles=(active_end_max+BM-1)/BM, tiles_per_split=(total_tiles+K_SPLITS-1)/K_SPLITS;
    int tile_lo=split_idx*tiles_per_split*BM, tile_hi=min((split_idx+1)*tiles_per_split*BM,active_end_t);
    extern __shared__ unsigned char sm[];
    half* q_s=(half*)sm; half* k_tile=q_s+GQA*HD; half* v_tile=k_tile+BM*HD; float* s_smem=(float*)(v_tile+BM*HD);
    for(int i=tid;i<GQA*HD;i+=BLOCK){int g=i/HD,c=i-g*HD;int qh=kv_head*GQA+g;q_s[g*HD+c]=q_chunk[(size_t)t_idx*num_q*HD+(size_t)qh*HD+c];}
    __syncthreads();
    float acc_o[LANE_D];
    #pragma unroll
    for(int i=0;i<LANE_D;i++)acc_o[i]=0.f;
    float m_w=-INFINITY,l_w=0.f;
    unsigned long long pl=0,ps=0,pv=0,tc=0,t0,t1,t2,t3;
    if(tile_lo<tile_hi){
        for(int tile_start=tile_lo;tile_start<tile_hi;tile_start+=BM){
            int tile_end=min(tile_start+BM,tile_hi),tile_len=tile_end-tile_start;
            if(tid==0)t0=clock64();
            for(int i=tid;i<BM*HD;i+=BLOCK){int r=i/HD,c=i-r*HD;half z=__float2half(0.f);
                if(r<tile_len){size_t base=(size_t)(tile_start+r)*num_kv*HD+kv_head*HD;k_tile[r*HD+c]=k_cache[base+c];v_tile[r*HD+c]=v_cache[base+c];}
                else{k_tile[r*HD+c]=z;v_tile[r*HD+c]=z;}}
            __syncthreads();
            if(tid==0){t1=clock64();pl+=t1-t0;}
            constexpr int NITERS=(GQA*BM+N_WARPS-1)/N_WARPS;
            #pragma unroll
            for(int k_it=0;k_it<NITERS;k_it++){int flat=warp+k_it*N_WARPS;if(flat>=GQA*BM)break;int g=flat/BM,r=flat-g*BM;
                float partial=0.f;const half2* qs2=(const half2*)(q_s+g*HD);const half2* kt2=(const half2*)(k_tile+r*HD);
                #pragma unroll
                for(int i=0;i<LANE_D/2;i++){half2 qv=qs2[lane*(LANE_D/2)+i];half2 kv=kt2[lane*(LANE_D/2)+i];float2 qf=__half22float2(qv),kf=__half22float2(kv);partial+=qf.x*kf.x+qf.y*kf.y;}
                #pragma unroll
                for(int off=16;off>0;off>>=1)partial+=__shfl_xor_sync(0xffffffff,partial,off);
                if(lane==0){float val=(r<tile_len)?partial*scale:-INFINITY;s_smem[g*BM+r]=val;}}
            __syncthreads();
            if(tid==0){t2=clock64();ps+=t2-t1;}
            if(warp<GQA){int g=warp;float s_val=s_smem[g*BM+lane];
                float m_row=s_val;
                #pragma unroll
                for(int off=16;off>0;off>>=1)m_row=fmaxf(m_row,__shfl_xor_sync(0xffffffff,m_row,off));
                float m_new=fmaxf(m_w,m_row),correction=expf(m_w-m_new),p_lane=expf(s_val-m_new),sum_p=p_lane;
                #pragma unroll
                for(int off=16;off>0;off>>=1)sum_p+=__shfl_xor_sync(0xffffffff,sum_p,off);
                #pragma unroll
                for(int i=0;i<LANE_D;i++)acc_o[i]*=correction;
                l_w=l_w*correction+sum_p;
                #pragma unroll
                for(int r=0;r<BM;r++){float p_r=__shfl_sync(0xffffffff,p_lane,r);const half2* vt=(const half2*)(v_tile+r*HD+lane*LANE_D);
                    #pragma unroll
                    for(int i=0;i<LANE_D/2;i++){half2 vv=vt[i];float2 vf=__half22float2(vv);acc_o[i*2]+=p_r*vf.x;acc_o[i*2+1]+=p_r*vf.y;}}
                m_w=m_new;}
            __syncthreads();
            if(tid==0){t3=clock64();pv+=t3-t2;tc++;}
        }
    }
    if(warp<GQA){int g=warp,qh=kv_head*GQA+g;size_t ml=((size_t)qh*sub_n_max+t_idx)*K_SPLITS+split_idx;
        if(lane==0){part_m[ml]=m_w;part_l[ml]=l_w;}
        size_t ob=(((size_t)qh*sub_n_max+t_idx)*K_SPLITS+split_idx)*HD;
        #pragma unroll
        for(int i=0;i<LANE_D;i++)part_o[ob+lane*LANE_D+i]=acc_o[i];}
    if(tid==0){atomicAdd(&g_phase[0],pl);atomicAdd(&g_phase[1],ps);atomicAdd(&g_phase[2],pv);atomicAdd(&g_phase[3],tc);}
}

// ===== fa_v2: NT t_idx per block, K/V loaded ONCE shared across NT*GQA queries =====
// Grid (num_kv, sub_n/NT, K_SPLITS). Amortizes the dominant load phase NT-fold.
template<int GQA,int BM,int BLOCK,int K_SPLITS,int NT>
__global__ void __launch_bounds__(BLOCK,4) fa_v2(
    const half* __restrict__ q_chunk,const half* __restrict__ k_cache,const half* __restrict__ v_cache,
    float* __restrict__ part_m,float* __restrict__ part_l,float* __restrict__ part_o,
    int num_q,int num_kv,int start_pos,int sub_n,int sub_n_max,int active_end_max,float scale){
    constexpr int N_WARPS=BLOCK/32; constexpr int LANE_D=HD/32; constexpr int QPB=NT*GQA;
    constexpr int QPW=(QPB+N_WARPS-1)/N_WARPS; // queries per warp (strided)
    int kv_head=blockIdx.x, tg=blockIdx.y, split_idx=blockIdx.z;
    int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
    int t_base=tg*NT;
    if(t_base>=sub_n) return;
    // max active end over this block's NT tokens (causal handled per-query in score)
    int abs_pos_max=start_pos+min(t_base+NT-1,sub_n-1)+1;
    int total_tiles=(active_end_max+BM-1)/BM, tiles_per_split=(total_tiles+K_SPLITS-1)/K_SPLITS;
    int tile_lo=split_idx*tiles_per_split*BM, tile_hi=min((split_idx+1)*tiles_per_split*BM,abs_pos_max);
    extern __shared__ unsigned char sm[];
    half* q_s=(half*)sm; half* k_tile=q_s+QPB*HD; half* v_tile=k_tile+BM*HD; float* s_smem=(float*)(v_tile+BM*HD);
    // load Q for all QPB queries
    for(int i=tid;i<QPB*HD;i+=BLOCK){int qi=i/HD,c=i-qi*HD;int t=qi/GQA,g=qi-t*GQA;int tok=t_base+t;
        int qh=kv_head*GQA+g; bool ok=tok<sub_n;
        q_s[qi*HD+c]= ok? q_chunk[(size_t)tok*num_q*HD+(size_t)qh*HD+c] : __float2half(0.f);}
    __syncthreads();
    float acc_o[QPW][LANE_D]; float m_w[QPW],l_w[QPW];
    #pragma unroll
    for(int u=0;u<QPW;u++){ m_w[u]=-INFINITY; l_w[u]=0.f;
        #pragma unroll
        for(int i=0;i<LANE_D;i++)acc_o[u][i]=0.f; }
    for(int tile_start=tile_lo;tile_start<tile_hi;tile_start+=BM){
        int tile_end=min(tile_start+BM,tile_hi),tile_len=tile_end-tile_start;
        // load K/V tile ONCE
        for(int i=tid;i<BM*HD;i+=BLOCK){int r=i/HD,c=i-r*HD;half z=__float2half(0.f);
            if(r<tile_len){size_t base=(size_t)(tile_start+r)*num_kv*HD+kv_head*HD;k_tile[r*HD+c]=k_cache[base+c];v_tile[r*HD+c]=v_cache[base+c];}
            else{k_tile[r*HD+c]=z;v_tile[r*HD+c]=z;}}
        __syncthreads();
        // score: each warp does its QPW queries x BM keys
        #pragma unroll
        for(int u=0;u<QPW;u++){int qi=warp+u*N_WARPS; if(qi>=QPB) break;
            const half2* qs2=(const half2*)(q_s+qi*HD);
            for(int r=0;r<BM;r++){
                const half2* kt2=(const half2*)(k_tile+r*HD); float partial=0.f;
                #pragma unroll
                for(int i=0;i<LANE_D/2;i++){half2 qv=qs2[lane*(LANE_D/2)+i];half2 kv=kt2[lane*(LANE_D/2)+i];float2 qf=__half22float2(qv),kf=__half22float2(kv);partial+=qf.x*kf.x+qf.y*kf.y;}
                #pragma unroll
                for(int off=16;off>0;off>>=1)partial+=__shfl_xor_sync(0xffffffff,partial,off);
                if(lane==0) s_smem[qi*BM+r]=partial*scale;}}
        __syncthreads();
        // softmax + AV: each warp its QPW queries; lane = key index within tile
        #pragma unroll
        for(int u=0;u<QPW;u++){int qi=warp+u*N_WARPS; if(qi>=QPB) break;
            int t=qi/GQA; int tok=t_base+t; int active_end=start_pos+tok+1;
            int gkey=tile_start+lane; // absolute key index this lane handles
            float s_val=(lane<tile_len && gkey<active_end)? s_smem[qi*BM+lane] : -INFINITY;
            float m_row=s_val;
            #pragma unroll
            for(int off=16;off>0;off>>=1)m_row=fmaxf(m_row,__shfl_xor_sync(0xffffffff,m_row,off));
            float m_new=fmaxf(m_w[u],m_row),corr=expf(m_w[u]-m_new),p_lane=expf(s_val-m_new),sum_p=p_lane;
            #pragma unroll
            for(int off=16;off>0;off>>=1)sum_p+=__shfl_xor_sync(0xffffffff,sum_p,off);
            #pragma unroll
            for(int i=0;i<LANE_D;i++)acc_o[u][i]*=corr;
            l_w[u]=l_w[u]*corr+sum_p;
            #pragma unroll
            for(int r=0;r<BM;r++){float p_r=__shfl_sync(0xffffffff,p_lane,r);const half2* vt=(const half2*)(v_tile+r*HD+lane*LANE_D);
                #pragma unroll
                for(int i=0;i<LANE_D/2;i++){half2 vv=vt[i];float2 vf=__half22float2(vv);acc_o[u][i*2]+=p_r*vf.x;acc_o[u][i*2+1]+=p_r*vf.y;}}
            m_w[u]=m_new;}
        __syncthreads();
    }
    #pragma unroll
    for(int u=0;u<QPW;u++){int qi=warp+u*N_WARPS; if(qi>=QPB) break;
        int t=qi/GQA,g=qi-t*GQA; int tok=t_base+t; if(tok>=sub_n) continue;
        int qh=kv_head*GQA+g; size_t ml=((size_t)qh*sub_n_max+tok)*K_SPLITS+split_idx;
        if(lane==0){part_m[ml]=m_w[u];part_l[ml]=l_w[u];}
        size_t ob=(((size_t)qh*sub_n_max+tok)*K_SPLITS+split_idx)*HD;
        #pragma unroll
        for(int i=0;i<LANE_D;i++)part_o[ob+lane*LANE_D+i]=acc_o[u][i];}
}

double maxabsdiff(const std::vector<half>&a,const std::vector<half>&b){
    double mx=0; for(size_t i=0;i<a.size();i++){double d=fabs(__half2float(a[i])-__half2float(b[i])); if(d>mx)mx=d;} return mx;
}

template<int GQA,int BM,int BLOCK,int K>
double run_current(half*q,half*k,half*v,half*out,int num_q,int num_kv,int start_pos,int sub_n,int seq,int iters,bool timeit){
    int ATTN_NB=sub_n; int active_end_max=seq;
    static float *pm=0,*pl=0,*po=0;
    size_t pn=(size_t)num_q*ATTN_NB*K;
    if(!pm){CK(cudaMalloc(&pm,pn*4));CK(cudaMalloc(&pl,pn*4));CK(cudaMalloc(&po,pn*HD*4));}
    int dyn=GQA*HD*2+2*BM*HD*2+GQA*BM*4;
    void*fn=(void*)flash_attn_chunk_fused_split<HD,GQA,BM,BLOCK,K>;
    if(dyn>48*1024) cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,96*1024);
    dim3 fg(num_kv,sub_n,K); float scale=1.f/sqrtf((float)HD);
    auto launch=[&]{
        flash_attn_chunk_fused_split<HD,GQA,BM,BLOCK,K><<<fg,BLOCK,dyn>>>(q,k,v,pm,pl,po,num_q,num_kv,start_pos,sub_n,ATTN_NB,active_end_max,scale);
        dim3 mg(num_kv,sub_n);
        flash_attn_split_merge<HD,GQA,BLOCK,K><<<mg,BLOCK>>>(pm,pl,po,out,num_q,sub_n,ATTN_NB);
    };
    launch(); CK(cudaDeviceSynchronize());
    if(!timeit) return 0;
    cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);
    for(int i=0;i<iters;i++) launch();
    cudaEventRecord(b);CK(cudaEventSynchronize(b)); float ms;cudaEventElapsedTime(&ms,a,b); return ms/iters;
}

template<int GQA,int BM,int BLOCK,int K,int NT>
double run_v2(half*q,half*k,half*v,half*out,int num_q,int num_kv,int start_pos,int sub_n,int seq,int iters,bool timeit){
    int ATTN_NB=sub_n,active_end_max=seq;
    static float *pm=0,*pl=0,*po=0; size_t pn=(size_t)num_q*ATTN_NB*K;
    if(!pm){CK(cudaMalloc(&pm,pn*4));CK(cudaMalloc(&pl,pn*4));CK(cudaMalloc(&po,pn*HD*4));}
    constexpr int QPB=NT*GQA;
    int dyn=QPB*HD*2+2*BM*HD*2+QPB*BM*4;
    void*fn=(void*)fa_v2<GQA,BM,BLOCK,K,NT>;
    if(dyn>48*1024) cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,96*1024);
    dim3 fg(num_kv,(sub_n+NT-1)/NT,K); float scale=1.f/sqrtf((float)HD);
    auto launch=[&]{
        fa_v2<GQA,BM,BLOCK,K,NT><<<fg,BLOCK,dyn>>>(q,k,v,pm,pl,po,num_q,num_kv,start_pos,sub_n,ATTN_NB,active_end_max,scale);
        dim3 mg(num_kv,sub_n); flash_attn_split_merge<HD,GQA,BLOCK,K><<<mg,BLOCK>>>(pm,pl,po,out,num_q,sub_n,ATTN_NB);
    };
    launch(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    if(!timeit) return 0;
    cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);
    for(int i=0;i<iters;i++) launch();
    cudaEventRecord(b);CK(cudaEventSynchronize(b));float ms;cudaEventElapsedTime(&ms,a,b);return ms/iters;
}

int main(int argc,char**argv){
    int seq = argc>1?atoi(argv[1]):512;
    int sub_n=16, num_kv=4, GQA=6, num_q=num_kv*GQA; // 24
    float scale=1.f/sqrtf((float)HD);
    int start_pos = seq - sub_n;
    size_t qn=(size_t)sub_n*num_q*HD, kvn=(size_t)seq*num_kv*HD, on=qn;
    std::vector<half> hq(qn),hk(kvn),hv(kvn);
    srand(1234);
    auto rnd=[]{ return (half)(__float2half(((rand()/(float)RAND_MAX)-0.5f)*0.5f)); };
    for(auto&x:hq)x=rnd(); for(auto&x:hk)x=rnd(); for(auto&x:hv)x=rnd();
    half *q,*k,*v,*o_ref,*o_cur;
    CK(cudaMalloc(&q,qn*2));CK(cudaMalloc(&k,kvn*2));CK(cudaMalloc(&v,kvn*2));
    CK(cudaMalloc(&o_ref,on*2));CK(cudaMalloc(&o_cur,on*2));
    CK(cudaMemcpy(q,hq.data(),qn*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(k,hk.data(),kvn*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(v,hv.data(),kvn*2,cudaMemcpyHostToDevice));
    // reference
    dim3 ng(sub_n,num_q); naive_attn<<<ng,1>>>(q,k,v,o_ref,num_q,num_kv,start_pos,sub_n,GQA,scale);
    CK(cudaDeviceSynchronize());
    std::vector<half> ref(on),cur(on);
    CK(cudaMemcpy(ref.data(),o_ref,on*2,cudaMemcpyDeviceToHost));
    // current kernel correctness (seq small) + timing (seq large)
    run_current<6,16,256,4>(q,k,v,o_cur,num_q,num_kv,start_pos,sub_n,seq,0,false);
    CK(cudaMemcpy(cur.data(),o_cur,on*2,cudaMemcpyDeviceToHost));
    printf("seq=%d  current vs ref: maxabsdiff=%.4f  ref[0..2]=%.3f %.3f %.3f cur=%.3f %.3f %.3f\n",
        seq, maxabsdiff(ref,cur),
        __half2float(ref[0]),__half2float(ref[1]),__half2float(ref[2]),
        __half2float(cur[0]),__half2float(cur[1]),__half2float(cur[2]));
    if(seq>=4096){
        double ms=run_current<6,16,256,4>(q,k,v,o_cur,num_q,num_kv,start_pos,sub_n,seq,200,true);
        printf("current kernel: %.4f ms/launch\n",ms);
        // phase breakdown via instrumented copy
        unsigned long long zero[4]={0,0,0,0};
        CK(cudaMemcpyToSymbol(g_phase,zero,sizeof(zero)));
        int dyn=6*HD*2+2*16*HD*2+6*16*4;
        float *pm,*pl,*po; size_t pe=(size_t)num_q*sub_n*4; // *K_SPLITS(4) elems
        CK(cudaMalloc(&pm,pe*4));CK(cudaMalloc(&pl,pe*4));CK(cudaMalloc(&po,pe*HD*4));
        dim3 fg(num_kv,sub_n,4);
        for(int i=0;i<50;i++) fa_prof<6,16,256,4><<<fg,256,dyn>>>(q,k,v,pm,pl,po,num_q,num_kv,start_pos,sub_n,sub_n,seq,scale);
        CK(cudaDeviceSynchronize());
        unsigned long long ph[4]; CK(cudaMemcpyFromSymbol(ph,g_phase,sizeof(ph)));
        double tot=ph[0]+ph[1]+ph[2];
        printf("phase cycles: load=%.0f%%  score=%.0f%%  value=%.0f%%  (tiles=%llu)\n",
            100.0*ph[0]/tot,100.0*ph[1]/tot,100.0*ph[2]/tot,ph[3]);
    }
    // ===== fa_v2 NT sweep: correctness (@small via separate run) + timing =====
    printf("--- fa_v2 (NT batched load) ---\n");
    half* o_v2; CK(cudaMalloc(&o_v2,on*2));
    auto check=[&](const char*tag,double ms){
        std::vector<half> v2(on); CK(cudaMemcpy(v2.data(),o_v2,on*2,cudaMemcpyDeviceToHost));
        printf("%s: maxdiff_vs_ref=%.4f  %.4f ms %s\n",tag,maxabsdiff(ref,v2),ms,
               maxabsdiff(ref,v2)<0.03?"OK":"*** WRONG ***");
    };
    bool t=(seq>=4096);
    #define V2(NT,BM,K) { run_v2<6,BM,256,K,NT>(q,k,v,o_v2,num_q,num_kv,start_pos,sub_n,seq,0,false); \
        check("NT=" #NT " BM=" #BM " K=" #K, t?run_v2<6,BM,256,K,NT>(q,k,v,o_v2,num_q,num_kv,start_pos,sub_n,seq,200,true):0); }
    V2(4,8,12) V2(4,8,16) V2(4,8,20) V2(4,8,24)
    V2(4,4,16) V2(4,4,24) V2(2,8,12) V2(8,8,16)
    return 0;
}
