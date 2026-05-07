// Block-sparse FlashAttention for the qwen-engine, modelled on Microsoft
// MInference's `_triton_block_sparse_attn_fwd_kernel` but rewritten in pure
// CUDA C++ to fit the engine's existing GQA/split-K layout. The compute body
// (load tile → score → online softmax → V-weighted accumulate) is intentionally
// the same as `flash_attn_chunk_fused_split` so per-iteration arithmetic stays
// bit-comparable; only the K-tile selection differs.
//
// Sparsity contract
//   For each (kv_head, t_idx), the caller supplies a sorted list of selected
//   K-block indices in `block_index[kv_head, t_idx, 0 .. top_k)`. Each index
//   is in units of `BLOCK_INDEX_N` K positions (typically 64). The kernel
//   iterates only those blocks, stepping through them in `BM`-sized inner
//   tiles (so a 64-block index entry becomes 64/BM inner tile passes — 2 for
//   BM=32). A sentinel value of -1 ends the list early. Causal masking is
//   applied per inner tile against `abs_pos = start_pos + t_idx`.
//
// Why per-(kv_head, t_idx) selection
//   The per-q_head pattern from the offline profile is converted at runtime
//   into per-kv_head selection (union of the GQA group's selections) so the
//   GQA queries that share K/V load the same tiles. That trades a small
//   amount of over-attendance for a single K/V pass per kv_head. Per-q_head
//   strict sparsity would need GQA× more K/V loads, undoing the gains.

#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

// Block-sparse split-K FA. Mirrors `flash_attn_chunk_fused_split` from
// attention.cuh; the only divergence is the outer iteration over selected
// blocks instead of contiguous tiles.
//
// Template params:
//   HD              head dim (256 for Qwen3.5/3.6)
//   GQA             num_q / num_kv (6 for 27B, 4 for 9B)
//   BM              FA tile size in K direction (32 keeps SMEM ≤48KB without
//                   carveout; 64 doubles K/V SMEM and needs 96KB)
//   BLOCK           threads per block (256)
//   K_SPLITS        partitions of `top_k`; merge happens in flash_attn_split_merge
//
// Runtime args:
//   block_index     [num_kv * sub_n * top_k] int32, sorted asc per row, -1 padding
//   block_size_n    K positions per index entry (typically 64). Must be a
//                   multiple of BM.
//   top_k           length of each block_index row.
//   start_pos       absolute position of t_idx=0 in the sequence
//   sub_n           number of Q tokens in this chunk (=blockDim grid Y)
//   active_end_max  position past which we never attend (causal cap shared
//                   with the dense kernel; passed for parity, not used here
//                   because index builder already enforces causal)
template<int HD, int GQA, int BM, int BLOCK, int K_SPLITS>
__global__ void flash_attn_chunk_block_sparse_split(
    const half* __restrict__ q_chunk,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const int*  __restrict__ block_index,
    float*      __restrict__ part_m,
    float*      __restrict__ part_l,
    float*      __restrict__ part_o,
    int  num_q, int num_kv,
    int  start_pos, int sub_n, int sub_n_max,
    int  /*active_end_max*/, float scale,
    int  top_k, int block_size_n)
{
    constexpr int N_WARPS = BLOCK / 32;
    constexpr int LANE_D  = HD / 32;
    int kv_head   = blockIdx.x;
    int t_idx     = blockIdx.y;
    int split_idx = blockIdx.z;
    if (t_idx >= sub_n) return;
    int abs_pos = start_pos + t_idx;
    int tid  = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    // Partition `top_k` selected blocks across K_SPLITS for SM-level parallelism.
    // Merge is the existing flash_attn_split_merge kernel — log-sum-exp combine
    // over m/l + weighted O sum.
    int per_split = (top_k + K_SPLITS - 1) / K_SPLITS;
    int tk_lo = split_idx * per_split;
    int tk_hi = min((split_idx + 1) * per_split, top_k);

    extern __shared__ unsigned char smem_fa_blksp[];
    half*  q_s    = (half*)smem_fa_blksp;
    half*  k_tile = q_s    + GQA * HD;
    half*  v_tile = k_tile + BM  * HD;
    float* s_smem = (float*)(v_tile + BM * HD);

    // Load Q (GQA queries × HD) for this (t_idx).
    #pragma unroll
    for (int i = tid; i < GQA * HD; i += BLOCK) {
        int g = i / HD;
        int c = i - g * HD;
        int q_head = kv_head * GQA + g;
        q_s[g * HD + c] = q_chunk[(size_t)t_idx * num_q * HD
                                  + (size_t)q_head * HD + c];
    }
    __syncthreads();

    float acc_o[LANE_D];
    #pragma unroll
    for (int i = 0; i < LANE_D; i++) acc_o[i] = 0.0f;
    float m_w = -INFINITY;
    float l_w = 0.0f;

    const int* bi_row = block_index + ((size_t)kv_head * sub_n + t_idx) * top_k;
    int inner_per_block = block_size_n / BM;  // tiles per index entry

    for (int tk = tk_lo; tk < tk_hi; tk++) {
        int blk_idx = bi_row[tk];
        if (blk_idx < 0) break;  // sentinel — no more selected blocks
        int outer_start = blk_idx * block_size_n;

        for (int inner = 0; inner < inner_per_block; inner++) {
            int tile_start = outer_start + inner * BM;
            if (tile_start > abs_pos) break;  // past causal cap
            int tile_end   = min(tile_start + BM, abs_pos + 1);
            int tile_len   = tile_end - tile_start;

            // Load K + V for this tile.
            #pragma unroll
            for (int i = tid; i < BM * HD; i += BLOCK) {
                int r = i / HD;
                int c = i - r * HD;
                half zero_h = __float2half(0.0f);
                if (r < tile_len) {
                    size_t base = (size_t)(tile_start + r) * num_kv * HD
                                + (size_t)kv_head * HD;
                    k_tile[r * HD + c] = k_cache[base + c];
                    v_tile[r * HD + c] = v_cache[base + c];
                } else {
                    k_tile[r * HD + c] = zero_h;
                    v_tile[r * HD + c] = zero_h;
                }
            }
            __syncthreads();

            // Score: q_s[g] · k_tile[r] for every (g, r). One warp computes
            // one (g,r) score; iterate when GQA*BM > N_WARPS.
            constexpr int NITERS = (GQA * BM + N_WARPS - 1) / N_WARPS;
            #pragma unroll
            for (int k_it = 0; k_it < NITERS; k_it++) {
                int flat = warp + k_it * N_WARPS;
                if (flat >= GQA * BM) break;
                int g = flat / BM;
                int r = flat - g * BM;

                float partial = 0.0f;
                const half2* qs2 = reinterpret_cast<const half2*>(q_s    + g * HD);
                const half2* kt2 = reinterpret_cast<const half2*>(k_tile + r * HD);
                #pragma unroll
                for (int i = 0; i < LANE_D / 2; i++) {
                    half2 qv = qs2[lane * (LANE_D / 2) + i];
                    half2 kv = kt2[lane * (LANE_D / 2) + i];
                    float2 qf = __half22float2(qv);
                    float2 kf = __half22float2(kv);
                    partial += qf.x * kf.x + qf.y * kf.y;
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    partial += __shfl_xor_sync(0xffffffff, partial, off);
                if (lane == 0) {
                    float val = (r < tile_len) ? partial * scale : -INFINITY;
                    s_smem[g * BM + r] = val;
                }
            }
            __syncthreads();

            // Online softmax + V accumulate (matches flash_attn_chunk_fused_split).
            if (warp < GQA) {
                int g = warp;
                float s_val = s_smem[g * BM + lane];

                float m_row = s_val;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    m_row = fmaxf(m_row, __shfl_xor_sync(0xffffffff, m_row, off));
                float m_new = fmaxf(m_w, m_row);
                float correction = expf(m_w - m_new);
                float p_lane    = expf(s_val - m_new);
                float sum_p = p_lane;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    sum_p += __shfl_xor_sync(0xffffffff, sum_p, off);

                #pragma unroll
                for (int i = 0; i < LANE_D; i++) acc_o[i] *= correction;
                l_w = l_w * correction + sum_p;

                #pragma unroll
                for (int r = 0; r < BM; r++) {
                    float p_r = __shfl_sync(0xffffffff, p_lane, r);
                    const half2* vt = reinterpret_cast<const half2*>(
                        v_tile + r * HD + lane * LANE_D);
                    #pragma unroll
                    for (int i = 0; i < LANE_D / 2; i++) {
                        half2 vv = vt[i];
                        float2 vf = __half22float2(vv);
                        acc_o[i * 2]     += p_r * vf.x;
                        acc_o[i * 2 + 1] += p_r * vf.y;
                    }
                }
                m_w = m_new;
            }
            __syncthreads();
        }
    }

    // Write split-K partials. The merge kernel (flash_attn_split_merge) is
    // shape-compatible.
    if (warp < GQA) {
        int g      = warp;
        int q_head = kv_head * GQA + g;
        size_t ml_idx = ((size_t)q_head * sub_n_max + t_idx) * K_SPLITS + split_idx;
        if (lane == 0) {
            part_m[ml_idx] = m_w;
            part_l[ml_idx] = l_w;
        }
        size_t o_base = (((size_t)q_head * sub_n_max + t_idx) * K_SPLITS + split_idx) * HD;
        #pragma unroll
        for (int i = 0; i < LANE_D; i++) {
            int c = lane * LANE_D + i;
            part_o[o_base + c] = acc_o[i];
        }
    }
}
