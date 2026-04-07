#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK_K 256
#define QK8 32

typedef struct { half d; half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K;
typedef struct { half d; half dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K;
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; } block_q6_K;
typedef struct { half d; int8_t qs[32]; } block_q8_0_32;
typedef struct { half2 ds; int8_t qs[QK8]; } block_q8_1;
// Q5_1: 32 elements per block, half d (scale) + half m (min) + 4 bytes qh (5th bit) + 16 bytes qs (4-bit packed)
typedef struct { half d; half m; uint8_t qh[4]; uint8_t qs[16]; } block_q5_1;
static_assert(sizeof(block_q5_1) == 24, "Q5_1 block must be 24 bytes");

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

// FP32 input variant — quantize fp32 directly (no fp16 cast loss)
__global__ void quantize_input_q8_1_f32(const float* __restrict__ x, block_q8_1* __restrict__ out, int K) {
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= K / QK8) return;
    const float* xb = x + blk * QK8;
    float amax = 0.0f, vals[QK8], sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < QK8; i++) {
        vals[i] = xb[i];
        amax = fmaxf(amax, fabsf(vals[i]));
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

// ============ Q5_1 GEMV ============
// Q5_1 block: 32 elements, half d + half m + 4 bytes qh + 16 bytes qs
// element[i] = d * (qs_4bit[i] | (qh_bit[i] << 4)) + m
// dot(x, w_block) = d * Σ(x[i] * q5[i]) + m * Σ(x[i])
// We use q8_1 input: x_q8 holds int8 quantized x with scale d8 and sum s8.
// Σ(x[i] * q5[i]) = Σ((d8 * x_q8[i]) * q5[i]) = d8 * Σ(x_q8[i] * q5[i])
// Σ(x[i]) = s8
// → dot = d * d8 * Σ(x_q8 * q5) + m * s8
__global__ void gemv_q5_1_q8(
    const void* __restrict__ weight,
    const block_q8_1* __restrict__ x_q8,
    half* __restrict__ output,
    const int K, const int N
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int nblocks = K / 32;  // q5_1 block = 32 elements = same as q8_1 block
    const block_q5_1* w_row = (const block_q5_1*)weight + (size_t)row * nblocks;

    float thread_sum = 0.0f;
    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
        const block_q5_1* wb = &w_row[b];
        const block_q8_1* xb = &x_q8[b];
        float d_w = __half2float(wb->d);
        float m_w = __half2float(wb->m);
        float d_x = __half2float(xb->ds.x);
        float s_x = __half2float(xb->ds.y);

        // Pack qh into a uint32 for fast bit extraction
        uint32_t qh = (uint32_t)wb->qh[0] | ((uint32_t)wb->qh[1] << 8) |
                      ((uint32_t)wb->qh[2] << 16) | ((uint32_t)wb->qh[3] << 24);

        int isum = 0;
        #pragma unroll
        for (int j = 0; j < 32; j += 8) {
            // Each qs byte holds two 4-bit values: low nibble = elem j, high nibble = elem j+16
            // Layout: qs[0] holds (elem 0, elem 16), qs[1] holds (elem 1, elem 17), ...
            // We need 8 consecutive elements [j, j+1, ..., j+7] but they're scattered.
            // For simplicity, just accumulate one element at a time using fp accumulator —
            // simpler than packing for dp4a given the unusual layout.
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int idx = j + k;
                int qs_idx = idx % 16;
                uint8_t qs_byte = wb->qs[qs_idx];
                int q4 = (idx < 16) ? (qs_byte & 0xF) : (qs_byte >> 4);
                int qh_bit = (qh >> idx) & 1;
                int q5 = q4 | (qh_bit << 4);  // [0..31]
                isum += (int)xb->qs[idx] * q5;
            }
        }
        thread_sum += d_w * d_x * (float)isum + m_w * s_x;
    }
    for (int off=16;off>0;off>>=1) thread_sum+=__shfl_xor_sync(0xffffffff,thread_sum,off);
    __shared__ float sm[8]; int w2=threadIdx.x>>5,l=threadIdx.x&31,nw=blockDim.x>>5;
    if(l==0)sm[w2]=thread_sum; __syncthreads();
    if(w2==0&&l<nw){float v=sm[l]; for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffff,v,o); if(l==0)output[row]=__float2half(v);}
}

__global__ void gemv_q5_1_q8_f32o(
    const void* __restrict__ weight,
    const block_q8_1* __restrict__ x_q8,
    float* __restrict__ output,
    const int K, const int N
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int nblocks = K / 32;
    const block_q5_1* w_row = (const block_q5_1*)weight + (size_t)row * nblocks;

    float thread_sum = 0.0f;
    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
        const block_q5_1* wb = &w_row[b];
        const block_q8_1* xb = &x_q8[b];
        float d_w = __half2float(wb->d);
        float m_w = __half2float(wb->m);
        float d_x = __half2float(xb->ds.x);
        float s_x = __half2float(xb->ds.y);

        uint32_t qh = (uint32_t)wb->qh[0] | ((uint32_t)wb->qh[1] << 8) |
                      ((uint32_t)wb->qh[2] << 16) | ((uint32_t)wb->qh[3] << 24);

        int isum = 0;
        #pragma unroll
        for (int idx = 0; idx < 32; idx++) {
            int qs_idx = idx % 16;
            uint8_t qs_byte = wb->qs[qs_idx];
            int q4 = (idx < 16) ? (qs_byte & 0xF) : (qs_byte >> 4);
            int qh_bit = (qh >> idx) & 1;
            int q5 = q4 | (qh_bit << 4);
            isum += (int)xb->qs[idx] * q5;
        }
        thread_sum += d_w * d_x * (float)isum + m_w * s_x;
    }
    for (int off=16;off>0;off>>=1) thread_sum+=__shfl_xor_sync(0xffffffff,thread_sum,off);
    __shared__ float sm[8]; int w2=threadIdx.x>>5,l=threadIdx.x&31,nw=blockDim.x>>5;
    if(l==0)sm[w2]=thread_sum; __syncthreads();
    if(w2==0&&l<nw){float v=sm[l]; for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffff,v,o); if(l==0)output[row]=v;}
}

// ============ Q4_K GEMV — dp4a + vec ============
// Q4_K block: 256 elements, half d + half dmin + 12 bytes scales + 128 bytes qs
// Each sub-block (32 elements) shares one (sc, mn) pair packed into the
// 12-byte scales array using the same 6-bit packing as Q5_K (since Q4_K and
// Q5_K share scale layout — Q5_K just adds a high bit on top of 4 bits).
// qs layout: 8 sub-blocks of 32 elements each, with each pair (sub i, sub i+4)
// of sub-blocks sharing one byte (low 4 bits = sub i, high 4 bits = sub i+4).
// Same as Q5_K but without the qh bit.

__global__ void gemv_q4_K_q8(
    const void* __restrict__ weight,
    const block_q8_1* __restrict__ x_q8,
    half* __restrict__ output,
    const int K, const int N
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int bpr = K / QK_K;
    const int nsub = K / QK8;
    const block_q4_K* w_row = (const block_q4_K*)weight + (size_t)row * bpr;

    float thread_sum = 0.0f;
    for (int sb = threadIdx.x; sb < nsub; sb += blockDim.x) {
        int blk_idx = sb >> 3, sub = sb & 7;
        const block_q4_K* b = &w_row[blk_idx];
        const block_q8_1* q8 = &x_q8[sb];
        float d4 = __half2float(b->d), dmin = __half2float(b->dmin);
        float d8 = __half2float(q8->ds.x), s8 = __half2float(q8->ds.y);
        const uint8_t* s = b->scales;
        uint8_t sc, mn;
        if (sub < 4) { sc = s[sub]&63; mn = s[sub+4]&63; }
        else { sc = (s[sub+4]&0xF)|((s[sub-4]>>6)<<4); mn = (s[sub+4]>>4)|((s[sub]>>6)<<4); }
        const uint8_t* ql_ptr = b->qs + (sub >> 1) * 32;
        int ql_shift = (sub & 1) * 4;
        int isum = 0;
        #pragma unroll
        for (int j = 0; j < 32; j += 4) {
            uint8_t q0 = (ql_ptr[j]   >> ql_shift) & 0xF;
            uint8_t q1 = (ql_ptr[j+1] >> ql_shift) & 0xF;
            uint8_t q2 = (ql_ptr[j+2] >> ql_shift) & 0xF;
            uint8_t q3 = (ql_ptr[j+3] >> ql_shift) & 0xF;
            isum = __dp4a(q0|(q1<<8)|(q2<<16)|(q3<<24), *(const int*)&q8->qs[j], isum);
        }
        thread_sum += d4*sc*d8*isum - dmin*mn*s8;
    }
    for (int off=16;off>0;off>>=1) thread_sum+=__shfl_xor_sync(0xffffffff,thread_sum,off);
    __shared__ float sm[8]; int w2=threadIdx.x>>5,l=threadIdx.x&31,nw=blockDim.x>>5;
    if(l==0)sm[w2]=thread_sum; __syncthreads();
    if(w2==0&&l<nw){float v=sm[l]; for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffff,v,o); if(l==0)output[row]=__float2half(v);}
}

// FP32 output variant
__global__ void gemv_q4_K_q8_f32o(
    const void* __restrict__ weight,
    const block_q8_1* __restrict__ x_q8,
    float* __restrict__ output,
    const int K, const int N
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int bpr = K / QK_K;
    const int nsub = K / QK8;
    const block_q4_K* w_row = (const block_q4_K*)weight + (size_t)row * bpr;

    float thread_sum = 0.0f;
    for (int sb = threadIdx.x; sb < nsub; sb += blockDim.x) {
        int blk_idx = sb >> 3, sub = sb & 7;
        const block_q4_K* b = &w_row[blk_idx];
        const block_q8_1* q8 = &x_q8[sb];
        float d4 = __half2float(b->d), dmin = __half2float(b->dmin);
        float d8 = __half2float(q8->ds.x), s8 = __half2float(q8->ds.y);
        const uint8_t* s = b->scales;
        uint8_t sc, mn;
        if (sub < 4) { sc = s[sub]&63; mn = s[sub+4]&63; }
        else { sc = (s[sub+4]&0xF)|((s[sub-4]>>6)<<4); mn = (s[sub+4]>>4)|((s[sub]>>6)<<4); }
        const uint8_t* ql_ptr = b->qs + (sub >> 1) * 32;
        int ql_shift = (sub & 1) * 4;
        int isum = 0;
        #pragma unroll
        for (int j = 0; j < 32; j += 4) {
            uint8_t q0 = (ql_ptr[j]   >> ql_shift) & 0xF;
            uint8_t q1 = (ql_ptr[j+1] >> ql_shift) & 0xF;
            uint8_t q2 = (ql_ptr[j+2] >> ql_shift) & 0xF;
            uint8_t q3 = (ql_ptr[j+3] >> ql_shift) & 0xF;
            isum = __dp4a(q0|(q1<<8)|(q2<<16)|(q3<<24), *(const int*)&q8->qs[j], isum);
        }
        thread_sum += d4*sc*d8*isum - dmin*mn*s8;
    }
    for (int off=16;off>0;off>>=1) thread_sum+=__shfl_xor_sync(0xffffffff,thread_sum,off);
    __shared__ float sm[8]; int w2=threadIdx.x>>5,l=threadIdx.x&31,nw=blockDim.x>>5;
    if(l==0)sm[w2]=thread_sum; __syncthreads();
    if(w2==0&&l<nw){float v=sm[l]; for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffff,v,o); if(l==0)output[row]=v;}
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

// ============ FP32-output variants ============
// These produce FP32 output instead of FP16, eliminating the cast loss at output.
// Input precision varies: q8_0/q5_K/q6_K still use int8 quantized input (no change),
// fp16/fp32 weights use the input directly.

__global__ void gemv_q5_K_q8_f32o(
    const void* __restrict__ weight,
    const block_q8_1* __restrict__ x_q8,
    float* __restrict__ output,
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
        int isum = 0;
        const uint8_t* ql_ptr = b->qs + (sub >> 1) * 32;
        int ql_shift = (sub & 1) * 4;
        int qh_bit = sub;
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
    if(w2==0&&l<nw){float v=sm[l]; for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffff,v,o); if(l==0)output[row]=v;}
}

__global__ void gemv_q6_K_q8_f32o(
    const void* __restrict__ weight,
    const block_q8_1* __restrict__ x_q8,
    float* __restrict__ output,
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
        int8_t sc_val0 = b->scales[(sub*32) / 16];
        int8_t sc_val1 = b->scales[(sub*32 + 16) / 16];
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
    if(w2==0&&l<nw){float v=sm[l]; for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffff,v,o); if(l==0)output[row]=v;}
}

// Q8_0 with fp32 input + fp32 output
__global__ void gemv_q8_0_f32(const void* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int K, int N) {
    int row=blockIdx.x; if(row>=N)return; int bpr=K/32;
    const block_q8_0_32* wr=(const block_q8_0_32*)w+(size_t)row*bpr; float s=0.0f;
    for(int b=threadIdx.x;b<bpr;b+=blockDim.x){
        float d=__half2float(wr[b].d),bs=0.0f;
        #pragma unroll
        for(int j=0;j<32;j++)bs+=(float)wr[b].qs[j]*x[b*32+j];
        s+=d*bs;
    }
    for(int o=16;o>0;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);
    __shared__ float sm[8];int w2=threadIdx.x>>5,l=threadIdx.x&31,nw=blockDim.x>>5;
    if(l==0)sm[w2]=s;__syncthreads();
    if(w2==0&&l<nw){s=sm[l];for(int o=16;o>0;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);if(l==0)y[row]=s;}
}
// FP16 weight + fp32 input + fp32 output
__global__ void gemv_fp16_f32(const half* __restrict__ w,const float* __restrict__ x,float* __restrict__ y,int K,int N){
    int row=blockIdx.x;if(row>=N)return;float s=0.0f;
    for(int i=threadIdx.x;i<K;i+=blockDim.x)s+=__half2float(w[(size_t)row*K+i])*x[i];
    for(int o=16;o>0;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);
    __shared__ float sm[8];int w2=threadIdx.x>>5,l=threadIdx.x&31,nw=blockDim.x>>5;
    if(l==0)sm[w2]=s;__syncthreads();
    if(w2==0&&l<nw){s=sm[l];for(int o=16;o>0;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);if(l==0)y[row]=s;}
}
// FP32 weight + fp32 input + fp32 output
__global__ void gemv_f32_f32(const float* __restrict__ w,const float* __restrict__ x,float* __restrict__ y,int K,int N){
    int row=blockIdx.x;if(row>=N)return;float s=0.0f;
    for(int i=threadIdx.x;i<K;i+=blockDim.x)s+=w[(size_t)row*K+i]*x[i];
    for(int o=16;o>0;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);
    __shared__ float sm[8];int w2=threadIdx.x>>5,l=threadIdx.x&31,nw=blockDim.x>>5;
    if(l==0)sm[w2]=s;__syncthreads();
    if(w2==0&&l<nw){s=sm[l];for(int o=16;o>0;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);if(l==0)y[row]=s;}
}

// ============ Dispatch ============
struct QuantInput {
    block_q8_1* q8_buf=nullptr; int max_K=0;
    void ensure(int K){int nb=K/QK8;if(K>max_K){if(q8_buf)cudaFree(q8_buf);cudaMalloc(&q8_buf,nb*sizeof(block_q8_1));max_K=K;}}
    void quantize(const half* input,int K,cudaStream_t stream=0){ensure(K);quantize_input_q8_1<<<(K/QK8+63)/64,64,0,stream>>>(input,q8_buf,K);}
    void quantize_f32(const float* input,int K,cudaStream_t stream=0){ensure(K);quantize_input_q8_1_f32<<<(K/QK8+63)/64,64,0,stream>>>(input,q8_buf,K);}
    void free_buf(){if(q8_buf){cudaFree(q8_buf);q8_buf=nullptr;max_K=0;}}
};

inline void quant_gemv(void* weight,ggml_type type,half* input,half* output,int K,int N,
                       QuantInput* qi=nullptr,cudaStream_t stream=0){
    // Use 128 threads for better occupancy with K=5120
    int threads = 128;
    if(qi&&(type==GGML_TYPE_Q4_K||type==GGML_TYPE_Q5_K||type==GGML_TYPE_Q6_K||type==GGML_TYPE_Q5_1)){
        qi->quantize(input,K,stream);
        if(type==GGML_TYPE_Q4_K)      gemv_q4_K_q8<<<N,threads,0,stream>>>(weight,qi->q8_buf,output,K,N);
        else if(type==GGML_TYPE_Q5_K) gemv_q5_K_q8<<<N,threads,0,stream>>>(weight,qi->q8_buf,output,K,N);
        else if(type==GGML_TYPE_Q6_K) gemv_q6_K_q8<<<N,threads,0,stream>>>(weight,qi->q8_buf,output,K,N);
        else                          gemv_q5_1_q8<<<N,threads,0,stream>>>(weight,qi->q8_buf,output,K,N);
    } else {
        switch(type){
        case GGML_TYPE_Q4_K:case GGML_TYPE_Q5_K:case GGML_TYPE_Q6_K:case GGML_TYPE_Q5_1:{
            QuantInput tmp;tmp.quantize(input,K,stream);
            if(type==GGML_TYPE_Q4_K)      gemv_q4_K_q8<<<N,threads,0,stream>>>(weight,tmp.q8_buf,output,K,N);
            else if(type==GGML_TYPE_Q5_K) gemv_q5_K_q8<<<N,threads,0,stream>>>(weight,tmp.q8_buf,output,K,N);
            else if(type==GGML_TYPE_Q6_K) gemv_q6_K_q8<<<N,threads,0,stream>>>(weight,tmp.q8_buf,output,K,N);
            else                          gemv_q5_1_q8<<<N,threads,0,stream>>>(weight,tmp.q8_buf,output,K,N);
            cudaStreamSynchronize(stream);tmp.free_buf();break;}
        case GGML_TYPE_Q8_0:gemv_q8_0<<<N,256,0,stream>>>(weight,input,output,K,N);break;
        case GGML_TYPE_F16: gemv_fp16<<<N,256,0,stream>>>((half*)weight,input,output,K,N);break;
        case GGML_TYPE_F32: gemv_f32<<<N,256,0,stream>>>((float*)weight,input,output,K,N);break;
        default:fprintf(stderr,"quant_gemv: unsupported type %d\n",type);
        }
    }
}

// FP32 input + FP32 output variant — eliminates fp16 cast losses on both ends
inline void quant_gemv_f32(void* weight,ggml_type type,float* input,float* output,int K,int N,
                           QuantInput* qi=nullptr,cudaStream_t stream=0){
    int threads = 128;
    if(qi&&(type==GGML_TYPE_Q4_K||type==GGML_TYPE_Q5_K||type==GGML_TYPE_Q6_K)){
        qi->quantize_f32(input,K,stream);
        if(type==GGML_TYPE_Q4_K)      gemv_q4_K_q8_f32o<<<N,threads,0,stream>>>(weight,qi->q8_buf,output,K,N);
        else if(type==GGML_TYPE_Q5_K) gemv_q5_K_q8_f32o<<<N,threads,0,stream>>>(weight,qi->q8_buf,output,K,N);
        else                          gemv_q6_K_q8_f32o<<<N,threads,0,stream>>>(weight,qi->q8_buf,output,K,N);
    } else {
        switch(type){
        case GGML_TYPE_Q4_K:case GGML_TYPE_Q5_K:case GGML_TYPE_Q6_K:{
            QuantInput tmp;tmp.quantize_f32(input,K,stream);
            if(type==GGML_TYPE_Q4_K)      gemv_q4_K_q8_f32o<<<N,threads,0,stream>>>(weight,tmp.q8_buf,output,K,N);
            else if(type==GGML_TYPE_Q5_K) gemv_q5_K_q8_f32o<<<N,threads,0,stream>>>(weight,tmp.q8_buf,output,K,N);
            else                          gemv_q6_K_q8_f32o<<<N,threads,0,stream>>>(weight,tmp.q8_buf,output,K,N);
            cudaStreamSynchronize(stream);tmp.free_buf();break;}
        case GGML_TYPE_Q8_0:gemv_q8_0_f32<<<N,256,0,stream>>>(weight,input,output,K,N);break;
        case GGML_TYPE_F16: gemv_fp16_f32<<<N,256,0,stream>>>((half*)weight,input,output,K,N);break;
        case GGML_TYPE_F32: gemv_f32_f32<<<N,256,0,stream>>>((float*)weight,input,output,K,N);break;
        default:fprintf(stderr,"quant_gemv_f32: unsupported type %d\n",type);
        }
    }
}
