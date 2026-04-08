#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK_K 256
#define QK8 32

typedef struct { half d; half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K;
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; } block_q6_K;
// GGUF on-disk Q8_0 layout (34 bytes, qs at offset 2 — never 4-byte aligned).
// Used only by the q8_0 repack kernel that converts to block_q8_0_aligned at
// load time.
typedef struct { half d; int8_t qs[32]; } block_q8_0_32;
// GPU-resident Q8_0 layout: qs lives at offset 0 with 4-byte alignment so the
// dp4a kernel can read 32-bit words directly. 36 bytes per block (vs 34 in
// GGUF) — ~6% extra VRAM in exchange for halving weight memory transactions
// in gemv_q8_0_q8.
typedef struct __align__(4) { int8_t qs[32]; uint16_t pad; half d; } block_q8_0_aligned;
typedef struct { half2 ds; int8_t qs[QK8]; } block_q8_1;

// Repack a flat array of GGUF block_q8_0_32 (34 B each) into the GPU-resident
// block_q8_0_aligned (36 B each). 1 thread per block. The src and dst buffers
// MUST be different (the strides differ).
__global__ void q8_0_repack_kernel(const void* __restrict__ src, void* __restrict__ dst, int n_blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_blocks) return;
    const block_q8_0_32* sb = (const block_q8_0_32*)src + idx;
    block_q8_0_aligned* db = (block_q8_0_aligned*)dst + idx;
    half d = sb->d;
    #pragma unroll
    for (int i = 0; i < 32; i++) db->qs[i] = sb->qs[i];
    db->d = d;
    db->pad = 0;
}

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
    const block_q8_0_aligned* wr=(const block_q8_0_aligned*)w+(size_t)row*bpr; float s=0.0f;
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
    const block_q8_0_aligned* w_row = (const block_q8_0_aligned*)weight + (size_t)row * bpr;

    float thread_sum = 0.0f;
    for (int b = threadIdx.x; b < bpr; b += blockDim.x) {
        const block_q8_0_aligned* wb = &w_row[b];
        const block_q8_1* xb = &x_q8[b];
        float d_w = __half2float(wb->d);
        float d_x = __half2float(xb->ds.x);

        // After the load-time repack to block_q8_0_aligned, qs sits at offset
        // 0 of every block and the per-block stride (36 B) is divisible by 4,
        // so wb->qs is 4-byte aligned and we can use ld.global.u32 directly —
        // 8 int loads per block instead of 16 u16 loads or 32 byte loads.
        const int* wp32 = (const int*)wb->qs;
        const int* x_qs = (const int*)xb->qs;  // 4-byte aligned (half2 prefix)

        int isum = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            isum = __dp4a(wp32[j], x_qs[j], isum);
        }
        thread_sum += d_w * d_x * (float)isum;
    }
    for (int off=16;off>0;off>>=1) thread_sum+=__shfl_xor_sync(0xffffffff,thread_sum,off);
    __shared__ float sm[8]; int w2=threadIdx.x>>5,l=threadIdx.x&31,nw=blockDim.x>>5;
    if(l==0)sm[w2]=thread_sum; __syncthreads();
    if(w2==0&&l<nw){float v=sm[l]; for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffff,v,o); if(l==0)output[row]=__float2half(v);}
}
// N=2 BATCHED Q8_0 weight × Q8_1 input GEMV.
// Same as gemv_q8_0_q8 but processes two independent input vectors that share
// the same weight matrix. The hot loop loads each weight word ONCE and runs
// two dp4a accumulators against it, so memory bandwidth (the bottleneck on
// HBM2-bound Volta) stays at the N=1 cost while we get two outputs.
//
// Used by the speculative-decoding path to verify the MTP draft token at the
// same wall-clock cost as a single non-spec forward — see model.cuh's
// forward_mlp_n2.
__global__ void gemv_q8_0_q8_n2(
    const void* __restrict__ weight,
    const block_q8_1* __restrict__ x_q8_a,
    const block_q8_1* __restrict__ x_q8_b,
    half* __restrict__ output_a,
    half* __restrict__ output_b,
    const int K, const int N
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int bpr = K / 32;
    const block_q8_0_aligned* w_row = (const block_q8_0_aligned*)weight + (size_t)row * bpr;

    float thread_sum_a = 0.0f;
    float thread_sum_b = 0.0f;
    for (int b = threadIdx.x; b < bpr; b += blockDim.x) {
        const block_q8_0_aligned* wb = &w_row[b];
        const block_q8_1* xb_a = &x_q8_a[b];
        const block_q8_1* xb_b = &x_q8_b[b];
        float d_w   = __half2float(wb->d);
        float d_x_a = __half2float(xb_a->ds.x);
        float d_x_b = __half2float(xb_b->ds.x);

        const int* wp32  = (const int*)wb->qs;
        const int* xa32 = (const int*)xb_a->qs;
        const int* xb32 = (const int*)xb_b->qs;

        int isum_a = 0;
        int isum_b = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int wj = wp32[j];   // load weight ONCE, reuse for both lanes
            isum_a = __dp4a(wj, xa32[j], isum_a);
            isum_b = __dp4a(wj, xb32[j], isum_b);
        }
        thread_sum_a += d_w * d_x_a * (float)isum_a;
        thread_sum_b += d_w * d_x_b * (float)isum_b;
    }
    for (int off=16; off>0; off>>=1) {
        thread_sum_a += __shfl_xor_sync(0xffffffff, thread_sum_a, off);
        thread_sum_b += __shfl_xor_sync(0xffffffff, thread_sum_b, off);
    }
    __shared__ float sm_a[8], sm_b[8];
    int w2 = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (l == 0) { sm_a[w2] = thread_sum_a; sm_b[w2] = thread_sum_b; }
    __syncthreads();
    if (w2 == 0 && l < nw) {
        float va = sm_a[l], vb = sm_b[l];
        for (int o=16; o>0; o>>=1) {
            va += __shfl_xor_sync(0xffffffff, va, o);
            vb += __shfl_xor_sync(0xffffffff, vb, o);
        }
        if (l == 0) {
            output_a[row] = __float2half(va);
            output_b[row] = __float2half(vb);
        }
    }
}

// N-token batched Q8_0 weight × Q8_1 input GEMV (chunked prefill).
// Each block computes one output row across NB tokens. The weight word is
// loaded ONCE per j-step and reused across NB dp4a accumulators, so total
// weight bandwidth is K * M (just like a single GEMV) instead of K * M * NB.
// This is the prefill analogue of gemv_q8_0_q8_n2 — same shared-weight-load
// trick, just scaled up to NB tokens at a time.
//
// Layout assumptions:
//   weight: [M, K/32] block_q8_0_aligned
//   x_q8:   [NB, K/32] block_q8_1, contiguous (token-major: t*bpr + b)
//   output: [NB, M] half (token-major: t*M + row)
//
// Caller must pre-quantize NB tokens of input into a contiguous block_q8_1
// buffer of size NB * (K/32). Caller must guarantee actual_n_tokens == NB,
// or pass actual_n_tokens < NB (kernel masks the spillover lanes).
template<int NB>
__global__ void gemv_q8_0_q8_nN(
    const void* __restrict__ weight,
    const block_q8_1* __restrict__ x_q8_chunk,   // [NB][K/32]
    half* __restrict__ output_chunk,             // [NB][M]
    const int K, const int M, const int actual_n_tokens
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int bpr = K / 32;
    const block_q8_0_aligned* w_row = (const block_q8_0_aligned*)weight + (size_t)row * bpr;

    float thread_sums[NB];
    #pragma unroll
    for (int n = 0; n < NB; n++) thread_sums[n] = 0.0f;

    for (int b = threadIdx.x; b < bpr; b += blockDim.x) {
        const block_q8_0_aligned* wb = &w_row[b];
        float d_w = __half2float(wb->d);
        const int* wp32 = (const int*)wb->qs;

        int isums[NB];
        #pragma unroll
        for (int n = 0; n < NB; n++) isums[n] = 0;

        // dp4a inner loop. Load weight word once per j, dp4a against NB inputs.
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int wj = wp32[j];
            #pragma unroll
            for (int n = 0; n < NB; n++) {
                if (n < actual_n_tokens) {
                    const block_q8_1* xb_n = x_q8_chunk + (size_t)n * bpr + b;
                    const int* xqs = (const int*)xb_n->qs;
                    isums[n] = __dp4a(wj, xqs[j], isums[n]);
                }
            }
        }

        #pragma unroll
        for (int n = 0; n < NB; n++) {
            if (n < actual_n_tokens) {
                const block_q8_1* xb_n = x_q8_chunk + (size_t)n * bpr + b;
                float d_x = __half2float(xb_n->ds.x);
                thread_sums[n] += d_w * d_x * (float)isums[n];
            }
        }
    }

    // Warp shuffle reduce within each lane
    for (int off = 16; off > 0; off >>= 1) {
        #pragma unroll
        for (int n = 0; n < NB; n++) {
            thread_sums[n] += __shfl_xor_sync(0xffffffff, thread_sums[n], off);
        }
    }

    // Cross-warp reduce via shared mem (max 8 warps per block at blockDim=256)
    __shared__ float sm[NB][8];
    int w_idx = threadIdx.x >> 5;
    int lane  = threadIdx.x & 31;
    int nwarp = blockDim.x >> 5;

    if (lane == 0) {
        #pragma unroll
        for (int n = 0; n < NB; n++) sm[n][w_idx] = thread_sums[n];
    }
    __syncthreads();

    if (w_idx == 0 && lane < nwarp) {
        float vals[NB];
        #pragma unroll
        for (int n = 0; n < NB; n++) vals[n] = sm[n][lane];
        for (int o = 16; o > 0; o >>= 1) {
            #pragma unroll
            for (int n = 0; n < NB; n++) {
                vals[n] += __shfl_xor_sync(0xffffffff, vals[n], o);
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int n = 0; n < NB; n++) {
                if (n < actual_n_tokens) {
                    output_chunk[(size_t)n * M + row] = __float2half(vals[n]);
                }
            }
        }
    }
}

// Explicit instantiations to keep linker happy and force separate code-gen.
template __global__ void gemv_q8_0_q8_nN<4>(const void*, const block_q8_1*, half*, int, int, int);
template __global__ void gemv_q8_0_q8_nN<8>(const void*, const block_q8_1*, half*, int, int, int);
template __global__ void gemv_q8_0_q8_nN<16>(const void*, const block_q8_1*, half*, int, int, int);

// ========================================================================
// Q8_0 weight × Q8_1 input GEMM (chunked prefill fast path)
//
// True GEMM with shared-memory tiling, so each global Q8_0 weight block
// and each global Q8_1 input block is loaded ONCE per K-tile and reused
// across all (BM, BN) outputs in the block. Compared to the per-row NB=16
// batched GEMV (gemv_q8_0_q8_nN<16>) this turns the per-output memory
// traffic from ~K weight + K input bytes (input bandwidth bound) into
// ~K/BN weight + K/BM input — both terms shrink, multiplicatively.
//
// Tile geometry (template constants):
//   BM           = output rows per block
//   BN           = output cols per block (== batched tokens per tile)
//   BK_BLOCKS    = q8_0 blocks of 32 K-elements processed per K-tile iter
//   THREADS      = BM * BN  (one thread per output, makes the inner loop
//                  trivial — every thread accumulates its own dp4a chain
//                  out of the shared tile)
//
// Layouts:
//   W: [M][bpr] block_q8_0_aligned, bpr = K/32
//   X: [N][bpr] block_q8_1
//   Y: [N][M]   half (token-major: y[n*M + m])
template<int BM, int BN, int BK_BLOCKS>
__global__ void gemm_q8_0_q8_1(
    const void* __restrict__ W,
    const block_q8_1* __restrict__ X,
    half* __restrict__ Y,
    const int M, const int N, const int K
) {
    const int bpr = K / 32;
    const int tid = threadIdx.x;
    const int tx  = tid % BM;          // output row inside tile
    const int ty  = tid / BM;          // output col inside tile
    const int gm  = blockIdx.x * BM + tx;
    const int gn  = blockIdx.y * BN + ty;

    // Shared memory tiles. Layout matters for the load-coalescing path,
    // but the simple "block-strided cooperative copy" below works well
    // enough for the first cut.
    extern __shared__ char gemm_smem_raw[];
    block_q8_0_aligned* sW = (block_q8_0_aligned*)gemm_smem_raw;             // [BM * BK_BLOCKS]
    block_q8_1*         sX = (block_q8_1*)(sW + BM * BK_BLOCKS);             // [BN * BK_BLOCKS]

    const block_q8_0_aligned* Wp = (const block_q8_0_aligned*)W;

    float acc = 0.0f;

    // Iterate over K in BK_BLOCKS-sized tiles. The K-loop bound is the
    // ceiling so the last tile is allowed to run with partial blocks
    // (the inner accumulator loop respects bpr).
    for (int kb = 0; kb < bpr; kb += BK_BLOCKS) {
        // ── Cooperative load: weight tile [BM][BK_BLOCKS] → SMEM
        for (int idx = tid; idx < BM * BK_BLOCKS; idx += BM * BN) {
            int row_in = idx / BK_BLOCKS;
            int b_in   = idx - row_in * BK_BLOCKS;
            int m_glob = blockIdx.x * BM + row_in;
            int b_glob = kb + b_in;
            if (m_glob < M && b_glob < bpr) {
                sW[idx] = Wp[(size_t)m_glob * bpr + b_glob];
            }
        }
        // ── Cooperative load: input tile [BN][BK_BLOCKS] → SMEM
        for (int idx = tid; idx < BN * BK_BLOCKS; idx += BM * BN) {
            int row_in = idx / BK_BLOCKS;
            int b_in   = idx - row_in * BK_BLOCKS;
            int n_glob = blockIdx.y * BN + row_in;
            int b_glob = kb + b_in;
            if (n_glob < N && b_glob < bpr) {
                sX[idx] = X[(size_t)n_glob * bpr + b_glob];
            }
        }
        __syncthreads();

        // ── Compute partial dot product for this thread's (tx, ty) output
        if (gm < M && gn < N) {
            #pragma unroll
            for (int b = 0; b < BK_BLOCKS; b++) {
                int b_glob = kb + b;
                if (b_glob >= bpr) break;
                const block_q8_0_aligned* wb = &sW[tx * BK_BLOCKS + b];
                const block_q8_1*         xb = &sX[ty * BK_BLOCKS + b];

                float d_w = __half2float(wb->d);
                float d_x = __half2float(xb->ds.x);

                const int* wp32 = (const int*)wb->qs;
                const int* xqs  = (const int*)xb->qs;

                int isum = 0;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    isum = __dp4a(wp32[j], xqs[j], isum);
                }
                acc += d_w * d_x * (float)isum;
            }
        }
        __syncthreads();
    }

    if (gm < M && gn < N) {
        Y[(size_t)gn * M + gm] = __float2half(acc);
    }
}

// Explicit instantiations.
template __global__ void gemm_q8_0_q8_1<16, 16, 8>(const void*, const block_q8_1*, half*, int, int, int);
template __global__ void gemm_q8_0_q8_1<16, 32, 4>(const void*, const block_q8_1*, half*, int, int, int);
template __global__ void gemm_q8_0_q8_1<32, 16, 4>(const void*, const block_q8_1*, half*, int, int, int);

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
    // Quantize a chunk of n_tokens × K contiguous half values into n_tokens
    // contiguous q8_1 rows. Each token contributes K/QK8 blocks; the kernel
    // operates on the flat n_tokens*K element span (block boundaries align
    // with token boundaries because K is always a multiple of QK8=32).
    void quantize_chunk(const half* input, int K, int n_tokens, cudaStream_t stream=0) {
        int total_K = K * n_tokens;
        ensure(total_K);
        quantize_input_q8_1<<<(total_K/QK8 + 63)/64, 64, 0, stream>>>(input, q8_buf, total_K);
    }
    void free_buf(){if(q8_buf){cudaFree(q8_buf);q8_buf=nullptr;max_K=0;}}
};

inline void quant_gemv(void* weight,ggml_type type,half* input,half* output,int K,int N,
                       QuantInput* qi=nullptr,cudaStream_t stream=0){
    // 128 threads for better occupancy with K=5120; pick larger for big K
    int threads = (K >= 8192) ? 256 : 128;
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

// Chunked Q8_0 GEMV/GEMM dispatch. Processes n_tokens contiguous inputs
// through the same weight matrix using a tiled GEMM kernel for the bulk
// of the chunk and the per-row N-batched GEMV kernels for any leftover.
// Caller pre-quantizes the input chunk into a contiguous block_q8_1 buffer
// of size n_tokens * (K/32).
//
// Falls back to per-token quant_gemv if `type` is not Q8_0 (Q5_K/Q6_K/etc).
inline void quant_gemv_chunk(void* weight, ggml_type type,
                             const block_q8_1* x_q8_chunk,
                             half* output_chunk,
                             int K, int M, int n_tokens,
                             cudaStream_t stream = 0) {
    if (n_tokens <= 0) return;
    int threads = (K >= 8192) ? 256 : 128;

    if (type == GGML_TYPE_Q8_0) {
        const block_q8_1* xp = x_q8_chunk;
        half* op = output_chunk;
        int remaining = n_tokens;

        // True GEMM tile path. Two tile geometries:
        //   • BM=16, BN=32, BK_BLOCKS=4 — preferred when remaining ≥ 32.
        //     512 outputs per block, 16 KB SMEM, 4 blocks/SM ⇒ full Volta
        //     occupancy. Per-output global traffic ≈ K/32 weight + K/16
        //     input — best ratio of the candidates.
        //   • BM=16, BN=16, BK_BLOCKS=8 — fallback for remaining ∈ [16, 31].
        //     256 outputs per block, K/16 weight + K/16 input.
        // Both require M divisible by 16, which holds for 27B's H, I,
        // QKV/output projection sizes.
        if (remaining >= 32 && (M % 16) == 0) {
            constexpr int BM = 16, BN = 32, BK_BLOCKS = 4;
            int n_chunks = remaining / BN;
            int bn_total = n_chunks * BN;
            dim3 grid(M / BM, n_chunks);
            int sm_bytes = (BM * BK_BLOCKS + BN * BK_BLOCKS) * sizeof(block_q8_0_aligned);
            gemm_q8_0_q8_1<BM, BN, BK_BLOCKS><<<grid, BM * BN, sm_bytes, stream>>>(
                weight, xp, op, M, bn_total, K);
            xp += (size_t)bn_total * (K / 32);
            op += (size_t)bn_total * M;
            remaining -= bn_total;
        }
        if (remaining >= 16 && (M % 16) == 0) {
            constexpr int BM = 16, BN = 16, BK_BLOCKS = 8;
            int n_chunks = remaining / BN;
            int bn_total = n_chunks * BN;
            dim3 grid(M / BM, n_chunks);
            int sm_bytes = (BM * BK_BLOCKS + BN * BK_BLOCKS) * sizeof(block_q8_0_aligned);
            gemm_q8_0_q8_1<BM, BN, BK_BLOCKS><<<grid, BM * BN, sm_bytes, stream>>>(
                weight, xp, op, M, bn_total, K);
            xp += (size_t)bn_total * (K / 32);
            op += (size_t)bn_total * M;
            remaining -= bn_total;
        }

        // Tail: < BN tokens. Fall back to N-batched GEMV peeling.
        while (remaining >= 8) {
            gemv_q8_0_q8_nN<8><<<M, threads, 0, stream>>>(weight, xp, op, K, M, 8);
            xp += 8 * (K / 32);
            op += (size_t)8 * M;
            remaining -= 8;
        }
        while (remaining >= 4) {
            gemv_q8_0_q8_nN<4><<<M, threads, 0, stream>>>(weight, xp, op, K, M, 4);
            xp += 4 * (K / 32);
            op += (size_t)4 * M;
            remaining -= 4;
        }
        if (remaining >= 2) {
            gemv_q8_0_q8_nN<4><<<M, threads, 0, stream>>>(weight, xp, op, K, M, remaining);
            xp += (size_t)remaining * (K / 32);
            op += (size_t)remaining * M;
            remaining = 0;
        }
        if (remaining == 1) {
            gemv_q8_0_q8<<<M, threads, 0, stream>>>(weight, xp, op, K, M);
        }
        return;
    }

    // Non-Q8_0 fallback: per-token GEMV. Slower but correct.
    int bpr = K / 32;
    for (int t = 0; t < n_tokens; t++) {
        const block_q8_1* xp = x_q8_chunk + (size_t)t * bpr;
        half* op = output_chunk + (size_t)t * M;
        // For non-Q8_0 we can't reuse the pre-quantized buffer this way (the
        // dequant kernels expect their own input format). Caller should keep
        // using token-by-token quant_gemv for those types — they're rare.
        if (type == GGML_TYPE_Q5_K)
            gemv_q5_K_q8<<<M, threads, 0, stream>>>(weight, xp, op, K, M);
        else if (type == GGML_TYPE_Q6_K)
            gemv_q6_K_q8<<<M, threads, 0, stream>>>(weight, xp, op, K, M);
        else
            gemv_q8_0_q8<<<M, threads, 0, stream>>>(weight, xp, op, K, M);
    }
}

// N=2 batched dispatch. Both inputs MUST be pre-quantized into qi_a / qi_b.
// Only Q8_0 is supported for now (the only quantization the speculative
// decoding path actually exercises on the 27B model). Other types fall back
// to two sequential N=1 calls — correct but no shared memory traffic.
inline void quant_gemv_n2(void* weight, ggml_type type,
                          half* in_a, half* in_b,
                          half* out_a, half* out_b,
                          int K, int N,
                          QuantInput* qi_a, QuantInput* qi_b,
                          cudaStream_t stream = 0) {
    int threads = (K >= 8192) ? 256 : 128;
    if (type == GGML_TYPE_Q8_0 && qi_a && qi_b) {
        gemv_q8_0_q8_n2<<<N, threads, 0, stream>>>(
            weight, qi_a->q8_buf, qi_b->q8_buf, out_a, out_b, K, N);
        return;
    }
    // Fallback: two sequential N=1 calls. Loses the shared-weight-load win
    // but stays correct for any quantization.
    quant_gemv(weight, type, in_a, out_a, K, N, qi_a, stream);
    quant_gemv(weight, type, in_b, out_b, K, N, qi_b, stream);
}
