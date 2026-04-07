#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK_K 256
#define QK8 32

typedef struct { half d; half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K;
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; } block_q6_K;
typedef struct { half d; int8_t qs[32]; } block_q8_0_32;
typedef struct { half2 ds; int8_t qs[QK8]; } block_q8_1;

__global__ void quantize_input_q8_1(const half* __restrict__ x, block_q8_1* __restrict__ out, int K) {
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= K / QK8) return;
    const half* xb = x + blk * QK8;
    float amax = 0.0f, vals[QK8], sum = 0.0f;
    const half2* xb2 = (const half2*)xb;
    #pragma unroll
    for (int i = 0; i < QK8/2; i++) {
        half2 v = xb2[i]; vals[i*2] = __half2float(v.x); vals[i*2+1] = __half2float(v.y);
        amax = fmaxf(amax, fmaxf(fabsf(vals[i*2]), fabsf(vals[i*2+1])));
    }
    float d = amax / 127.0f, id = d > 0.0f ? 127.0f / amax : 0.0f;
    block_q8_1* ob = &out[blk];
    #pragma unroll
    for (int i = 0; i < QK8; i++) { int q = __float2int_rn(vals[i]*id); ob->qs[i]=(int8_t)q; sum+=d*q; }
    ob->ds = make_half2(__float2half(d), __float2half(sum));
}

// ============ Q5_K GEMV — dp4a + vec (이전 동작 버전 그대로, 스레드 128) ============

__global__ void gemv_q5_K_q8(
    const void* __restrict__ weight,
    const block_q8_1* __restrict__ x_q8,
    half* __restrict__ output,
    const int K, const int N
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int bpr = K / QK_K;
    const int nsub = K / QK8;
    const block_q5_K* w_row = (const block_q5_K*)weight + (size_t)row * bpr;

    float thread_sum = 0.0f;
    for (int sb = threadIdx.x; sb < nsub; sb += blockDim.x) {
        int blk_idx = sb >> 3, sub = sb & 7;
        const block_q5_K* b = &w_row[blk_idx];
        const block_q8_1* q8 = &x_q8[sb];
        float d5 = __half2float(b->d), dmin = __half2float(b->dmin);
        float d8 = __half2float(q8->ds.x), s8 = __half2float(q8->ds.y);
        const uint8_t* s = b->scales;
        uint8_t sc, mn;
        if (sub < 4) { sc = s[sub]&63; mn = s[sub+4]&63; }
        else { sc = (s[sub+4]&0xF)|((s[sub-4]>>6)<<4); mn = (s[sub+4]>>4)|((s[sub]>>6)<<4); }
        int base = sub * 32, isum = 0;
        // Q5_K: group=sub/2, sub_half=sub%2
        const uint8_t* ql_ptr = b->qs + (sub >> 1) * 32;
        int ql_shift = (sub & 1) * 4;
        int qh_bit = sub;  // bit position in qh byte
        
        #pragma unroll
        for (int j = 0; j < 32; j += 4) {
            uint8_t q0 = ((ql_ptr[j]   >> ql_shift) & 0xF) | (((b->qh[j]   >> qh_bit) & 1) << 4);
            uint8_t q1 = ((ql_ptr[j+1] >> ql_shift) & 0xF) | (((b->qh[j+1] >> qh_bit) & 1) << 4);
            uint8_t q2 = ((ql_ptr[j+2] >> ql_shift) & 0xF) | (((b->qh[j+2] >> qh_bit) & 1) << 4);
            uint8_t q3 = ((ql_ptr[j+3] >> ql_shift) & 0xF) | (((b->qh[j+3] >> qh_bit) & 1) << 4);
            isum = __dp4a(q0|(q1<<8)|(q2<<16)|(q3<<24), *(const int*)&q8->qs[j], isum);
        }
        thread_sum += d5*sc*d8*isum - dmin*mn*s8;
    }
    for (int off=16;off>0;off>>=1) thread_sum+=__shfl_xor_sync(0xffffffff,thread_sum,off);
    __shared__ float sm[8]; int w2=threadIdx.x>>5,l=threadIdx.x&31,nw=blockDim.x>>5;
    if(l==0)sm[w2]=thread_sum; __syncthreads();
    if(w2==0&&l<nw){float v=sm[l]; for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffff,v,o); if(l==0)output[row]=__float2half(v);}
}

// ============ Q6_K GEMV — dp4a + vec ============

__global__ void gemv_q6_K_q8(
    const void* __restrict__ weight,
    const block_q8_1* __restrict__ x_q8,
    half* __restrict__ output,
    const int K, const int N
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int bpr = K / QK_K;
    const int nsub = K / QK8;
    const block_q6_K* w_row = (const block_q6_K*)weight + (size_t)row * bpr;

    float thread_sum = 0.0f;
    for (int sb = threadIdx.x; sb < nsub; sb += blockDim.x) {
        int blk_idx = sb >> 3, sub = sb & 7;
        const block_q6_K* b = &w_row[blk_idx];
        const block_q8_1* q8 = &x_q8[sb];
        float d6 = __half2float(b->d), d8 = __half2float(q8->ds.x);
        int base = sub*32;
        // scales computed in inner loop

        // Q6_K: exact layout matching dequant_embd_q6k_row_v2
        int8_t sc_val0 = b->scales[(sub*32) / 16];      // scale for first 16
        int8_t sc_val1 = b->scales[(sub*32 + 16) / 16]; // scale for next 16
        
        int isum0 = 0;
        #pragma unroll
        for (int j = 0; j < 16; j += 4) {
            int8_t qq[4];
            #pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                int elem = sub * 32 + j + jj;
                int sg = elem / 128;
                int pos = elem % 128;
                int quarter = pos / 32;
                int l = pos % 32;
                const uint8_t* ql = b->ql + sg * 64;
                const uint8_t* qh = b->qh + sg * 32;
                uint8_t ql4, qh2;
                switch(quarter) {
                    case 0: ql4 = ql[l] & 0xF;     qh2 = (qh[l]>>0)&3; break;
                    case 1: ql4 = ql[l+32] & 0xF;   qh2 = (qh[l]>>2)&3; break;
                    case 2: ql4 = ql[l] >> 4;        qh2 = (qh[l]>>4)&3; break;
                    default:ql4 = ql[l+32] >> 4;     qh2 = (qh[l]>>6)&3; break;
                }
                qq[jj] = (int8_t)(ql4 | (qh2 << 4)) - 32;
            }
            isum0 = __dp4a((int)(((uint8_t)qq[0])|((uint8_t)qq[1]<<8)|((uint8_t)qq[2]<<16)|((uint8_t)qq[3]<<24)), *(const int*)&q8->qs[j], isum0);
        }
        int isum1 = 0;
        #pragma unroll
        for (int j = 16; j < 32; j += 4) {
            int8_t qq[4];
            #pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                int elem = sub * 32 + j + jj;
                int sg = elem / 128;
                int pos = elem % 128;
                int quarter = pos / 32;
                int l = pos % 32;
                const uint8_t* ql = b->ql + sg * 64;
                const uint8_t* qh = b->qh + sg * 32;
                uint8_t ql4, qh2;
                switch(quarter) {
                    case 0: ql4 = ql[l] & 0xF;     qh2 = (qh[l]>>0)&3; break;
                    case 1: ql4 = ql[l+32] & 0xF;   qh2 = (qh[l]>>2)&3; break;
                    case 2: ql4 = ql[l] >> 4;        qh2 = (qh[l]>>4)&3; break;
                    default:ql4 = ql[l+32] >> 4;     qh2 = (qh[l]>>6)&3; break;
                }
                qq[jj] = (int8_t)(ql4 | (qh2 << 4)) - 32;
            }
            isum1 = __dp4a((int)(((uint8_t)qq[0])|((uint8_t)qq[1]<<8)|((uint8_t)qq[2]<<16)|((uint8_t)qq[3]<<24)), *(const int*)&q8->qs[j], isum1);
        }
        thread_sum += d6*d8*(sc_val0*isum0 + sc_val1*isum1);
    }
    for (int off=16;off>0;off>>=1) thread_sum+=__shfl_xor_sync(0xffffffff,thread_sum,off);
    __shared__ float sm[8]; int w2=threadIdx.x>>5,l=threadIdx.x&31,nw=blockDim.x>>5;
    if(l==0)sm[w2]=thread_sum; __syncthreads();
    if(w2==0&&l<nw){float v=sm[l]; for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffff,v,o); if(l==0)output[row]=__float2half(v);}
}

// ============ Q8_0 / FP16 / F32 ============

__global__ void gemv_q8_0(const void* __restrict__ w, const half* __restrict__ x, half* __restrict__ y, int K, int N) {
    int row=blockIdx.x; if(row>=N)return; int bpr=K/32;
    const block_q8_0_32* wr=(const block_q8_0_32*)w+(size_t)row*bpr; float s=0.0f;
    for(int b=threadIdx.x;b<bpr;b+=blockDim.x){float d=__half2float(wr[b].d),bs=0.0f;
    #pragma unroll
    for(int j=0;j<32;j++)bs+=(float)wr[b].qs[j]*__half2float(x[b*32+j]);s+=d*bs;}
    for(int o=16;o>0;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);
    __shared__ float sm[8];int w2=threadIdx.x>>5,l=threadIdx.x&31,nw=blockDim.x>>5;
    if(l==0)sm[w2]=s;__syncthreads();
    if(w2==0&&l<nw){s=sm[l];for(int o=16;o>0;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);if(l==0)y[row]=__float2half(s);}
}

// Q8_0 weight × Q8_1 input via dp4a (much faster — int8 dot products)
// The input gets quantized once via QuantInput::quantize and reused across
// multiple GEMV calls (q, k, v share the same RMSNorm output, etc).
//
// Block layout: 1 block per output row, 128 threads (= 4 warps).
// For K=5120: 160 q8_0/q8_1 blocks per row → ~1.25 blocks/thread.
//
// IMPORTANT: block_q8_0_32 is { half d; int8_t qs[32]; } so qs starts at
// byte offset 2 from each block — NOT 4-byte aligned. We pack bytes
// manually to avoid misaligned int loads. block_q8_1 is { half2 ds; int8_t qs[32]; }
// so its qs IS 4-byte aligned and we can read directly as int.
__global__ void gemv_q8_0_q8(
    const void* __restrict__ weight,
    const block_q8_1* __restrict__ x_q8,
    half* __restrict__ output,
    const int K, const int N
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int bpr = K / 32;
    const block_q8_0_32* w_row = (const block_q8_0_32*)weight + (size_t)row * bpr;

    float thread_sum = 0.0f;
    for (int b = threadIdx.x; b < bpr; b += blockDim.x) {
        const block_q8_0_32* wb = &w_row[b];
        const block_q8_1* xb = &x_q8[b];
        float d_w = __half2float(wb->d);
        float d_x = __half2float(xb->ds.x);

        // wb->qs is 2-byte aligned (offset 2 from wb base, wb itself is 2-byte
        // aligned). On Volta sm70 ld.global.u32 requires 4-byte alignment so
        // we can't read whole ints, but ld.global.u16 only needs 2-byte align,
        // which is half the memory transactions of byte-by-byte loads.
        const uint16_t* wp16 = (const uint16_t*)wb->qs;
        const int* x_qs = (const int*)xb->qs;  // 4-byte aligned (half2 prefix)

        int isum = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            // Two u16 loads → assemble little-endian 32-bit word for dp4a.
            // dp4a interprets the 4 bytes as signed int8 — sign bits land
            // in bits 7/15/23/31, which is exactly where the int8 sign bits
            // sit when packed little-endian.
            int w_lo = wp16[2*j+0];
            int w_hi = wp16[2*j+1];
            int wq = w_lo | (w_hi << 16);
            isum = __dp4a(wq, x_qs[j], isum);
        }
        thread_sum += d_w * d_x * (float)isum;
    }
    for (int off=16;off>0;off>>=1) thread_sum+=__shfl_xor_sync(0xffffffff,thread_sum,off);
    __shared__ float sm[8]; int w2=threadIdx.x>>5,l=threadIdx.x&31,nw=blockDim.x>>5;
    if(l==0)sm[w2]=thread_sum; __syncthreads();
    if(w2==0&&l<nw){float v=sm[l]; for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffff,v,o); if(l==0)output[row]=__float2half(v);}
}
__global__ void gemv_fp16(const half* __restrict__ w,const half* __restrict__ x,half* __restrict__ y,int K,int N){
    int row=blockIdx.x;if(row>=N)return;float s=0.0f;
    for(int i=threadIdx.x;i<K;i+=blockDim.x)s+=__half2float(w[(size_t)row*K+i])*__half2float(x[i]);
    for(int o=16;o>0;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);
    __shared__ float sm[8];int w2=threadIdx.x>>5,l=threadIdx.x&31,nw=blockDim.x>>5;
    if(l==0)sm[w2]=s;__syncthreads();
    if(w2==0&&l<nw){s=sm[l];for(int o=16;o>0;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);if(l==0)y[row]=__float2half(s);}
}
__global__ void gemv_f32(const float* __restrict__ w,const half* __restrict__ x,half* __restrict__ y,int K,int N){
    int row=blockIdx.x;if(row>=N)return;float s=0.0f;
    for(int i=threadIdx.x;i<K;i+=blockDim.x)s+=w[(size_t)row*K+i]*__half2float(x[i]);
    for(int o=16;o>0;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);
    __shared__ float sm[8];int w2=threadIdx.x>>5,l=threadIdx.x&31,nw=blockDim.x>>5;
    if(l==0)sm[w2]=s;__syncthreads();
    if(w2==0&&l<nw){s=sm[l];for(int o=16;o>0;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);if(l==0)y[row]=__float2half(s);}
}

// ============ Dispatch ============
struct QuantInput {
    block_q8_1* q8_buf=nullptr; int max_K=0;
    void ensure(int K){int nb=K/QK8;if(K>max_K){if(q8_buf)cudaFree(q8_buf);cudaMalloc(&q8_buf,nb*sizeof(block_q8_1));max_K=K;}}
    void quantize(const half* input,int K,cudaStream_t stream=0){ensure(K);quantize_input_q8_1<<<(K/QK8+63)/64,64,0,stream>>>(input,q8_buf,K);}
    void free_buf(){if(q8_buf){cudaFree(q8_buf);q8_buf=nullptr;max_K=0;}}
};

inline void quant_gemv(void* weight,ggml_type type,half* input,half* output,int K,int N,
                       QuantInput* qi=nullptr,cudaStream_t stream=0){
    // 128 threads for better occupancy with K=5120
    int threads = 128;
    // When qi is provided we ASSUME the caller has already called
    // qi->quantize(input, K, stream) — every call site in model.cuh,
    // gemma_model.cuh, and main.cu does this immediately before its first
    // quant_gemv call and reuses the same q8_buf for all subsequent GEMVs
    // that share the same input. The previous version called qi->quantize
    // here too which was redundant work (4× per GDN layer, etc.). Speedup
    // is small in single-token generation (well under noise on 27B) but the
    // change is correct and removes wasted launches.
    if(qi&&(type==GGML_TYPE_Q5_K||type==GGML_TYPE_Q6_K||type==GGML_TYPE_Q8_0)){
        if(type==GGML_TYPE_Q5_K)      gemv_q5_K_q8<<<N,threads,0,stream>>>(weight,qi->q8_buf,output,K,N);
        else if(type==GGML_TYPE_Q6_K) gemv_q6_K_q8<<<N,threads,0,stream>>>(weight,qi->q8_buf,output,K,N);
        else                          gemv_q8_0_q8<<<N,threads,0,stream>>>(weight,qi->q8_buf,output,K,N);
    } else {
        switch(type){
        case GGML_TYPE_Q5_K:case GGML_TYPE_Q6_K:{
            QuantInput tmp;tmp.quantize(input,K,stream);
            if(type==GGML_TYPE_Q5_K)gemv_q5_K_q8<<<N,threads,0,stream>>>(weight,tmp.q8_buf,output,K,N);
            else                    gemv_q6_K_q8<<<N,threads,0,stream>>>(weight,tmp.q8_buf,output,K,N);
            cudaStreamSynchronize(stream);tmp.free_buf();break;}
        case GGML_TYPE_Q8_0:{
            // No QuantInput provided — fall back to the slower fp16-input path
            gemv_q8_0<<<N,256,0,stream>>>(weight,input,output,K,N);break;}
        case GGML_TYPE_F16: gemv_fp16<<<N,256,0,stream>>>((half*)weight,input,output,K,N);break;
        case GGML_TYPE_F32: gemv_f32<<<N,256,0,stream>>>((float*)weight,input,output,K,N);break;
        default:fprintf(stderr,"quant_gemv: unsupported type %d\n",type);
        }
    }
}
