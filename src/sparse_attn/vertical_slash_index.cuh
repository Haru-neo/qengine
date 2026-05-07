// Vertical-slash sparse pattern, MInference-style but block-aligned and
// runtime-estimated per chunk (no profile-side index storage).
//
// What we approximate
//   The Triton reference picks a head's K positions as
//     "vertical"  :  specific columns where many queries attend
//     "slash"     :  diagonal offsets where Q_pos - K_pos is constant
//   This kernel reproduces the idea at block granularity (BLOCK_N=64 tokens
//   per K block):
//     vertical_idx[kv_head, 0..V) = top-V K blocks ranked by Σ_t score[t, n]
//     slash_idx[kv_head, 0..S)    = top-S deltas where score[t, q_block(t)-δ]
//                                   integrates highest across the chunk's
//                                   queries
//   At runtime the per-token index is just the union of vertical and the
//   token-specific slash blocks (q_block(t_idx) - slash_idx[i]). The same
//   flash_attn_chunk_block_sparse_split kernel consumes that index.
//
// Why we estimate per chunk instead of storing in the profile
//   sub_n = 16 queries gives noisy estimates, but adapting per chunk lets
//   the pattern follow the actual content (the calibration prompt's
//   patterns won't match every long-context query). The L2 cost vs a
//   per-layer fixed pattern is small in practice (paper §3.3) — we trade
//   a one-time O(sub_n · n_blocks · HD) score build for fresh selection.

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// Per (kv_head): build scores[sub_n, n_blocks] = q_avg per t_idx ⋅ K_pool[n],
// then vertical_idx[V] = top-V blocks by Σ_t score[t,n], slash_idx[S] = top-S
// deltas by Σ_t score[t, q_block(t)-δ].
//
// Grid:   (num_kv, 1, 1)
// Block:  BLOCK threads (256). All work for a kv_head sits inside one block
//         so vertical / slash aggregation reduces inside SMEM without going
//         back to HBM.
//
// SMEM layout (worst case 16K context with n_blocks=256):
//   q_avg            [sub_n_max * HD]                =  16*256*2 = 8 KB
//   scores           [sub_n_max * n_blocks_max]      =  16*256*4 = 16 KB
//   block_agg        [n_blocks_max]                  =     256*4 =  1 KB
//   delta_agg        [n_blocks_max]                  =     256*4 =  1 KB
//   ≈ 26 KB; comfortably under the default 48 KB limit.
template<int HD, int GQA, int BLOCK>
__global__ void build_vs_aggregate_kern(
    const half* __restrict__ q_chunk,
    const half* __restrict__ k_pool,
    int*        __restrict__ vertical_idx,   // [num_kv * V_top_k]
    int*        __restrict__ slash_idx,      // [num_kv * S_top_k]
    int  num_q, int num_kv,
    int  sub_n, int sub_n_max,
    int  start_pos,
    int  V_top_k, int S_top_k,
    int  block_size_n, int n_blocks)
{
    int kv_head = blockIdx.x;
    int tid     = threadIdx.x;

    extern __shared__ unsigned char smem_vs[];
    half*  q_avg     = (half*)smem_vs;                                   // sub_n*HD
    float* scores    = (float*)(q_avg + (size_t)sub_n * HD);             // sub_n*n_blocks
    float* block_agg = scores + (size_t)sub_n * n_blocks;                // n_blocks
    float* delta_agg = block_agg + n_blocks;                             // n_blocks

    // 1. Build q_avg[t, c] for every t in [0, sub_n).
    int total_q = sub_n * HD;
    for (int i = tid; i < total_q; i += BLOCK) {
        int t = i / HD;
        int c = i - t * HD;
        float s = 0.0f;
        #pragma unroll
        for (int g = 0; g < GQA; g++) {
            int q_head = kv_head * GQA + g;
            s += __half2float(q_chunk[(size_t)t * num_q * HD
                                      + (size_t)q_head * HD + c]);
        }
        q_avg[t * HD + c] = __float2half(s * (1.0f / GQA));
    }
    __syncthreads();

    // 2. score[t, n] = q_avg[t] · K_pool[kv_head, n]. Each thread covers a
    //    strided subset of the (t, n) flat space. n_blocks ≤ 256 typically.
    int total_score = sub_n * n_blocks;
    for (int i = tid; i < total_score; i += BLOCK) {
        int t = i / n_blocks;
        int n = i - t * n_blocks;
        const half* qa = q_avg + t * HD;
        const half* kp = k_pool + ((size_t)kv_head * n_blocks + n) * HD;
        const half2* qa2 = reinterpret_cast<const half2*>(qa);
        const half2* kp2 = reinterpret_cast<const half2*>(kp);
        float s = 0.0f;
        #pragma unroll 8
        for (int c = 0; c < HD / 2; c++) {
            float2 qf = __half22float2(qa2[c]);
            float2 kf = __half22float2(kp2[c]);
            s += qf.x * kf.x + qf.y * kf.y;
        }
        // Causal block mask: t's q_block is (start_pos + t) / block_size_n.
        // Any n_block past q_block is masked.
        int q_block = (start_pos + t) / block_size_n;
        if (n > q_block) s = -INFINITY;
        scores[(size_t)t * n_blocks + n] = s;
    }
    __syncthreads();

    // 3. block_agg[n] = Σ_t score[t, n]. Each thread sums a column.
    for (int n = tid; n < n_blocks; n += BLOCK) {
        float s = 0.0f;
        for (int t = 0; t < sub_n; t++) {
            float v = scores[(size_t)t * n_blocks + n];
            if (isfinite(v)) s += v;
        }
        block_agg[n] = s;
    }
    // 4. delta_agg[δ] = Σ_t score[t, q_block(t) - δ] (diagonal sums). δ ∈
    //    [0, n_blocks). Each thread covers a δ.
    for (int d = tid; d < n_blocks; d += BLOCK) {
        float s = 0.0f;
        int valid = 0;
        for (int t = 0; t < sub_n; t++) {
            int q_block = (start_pos + t) / block_size_n;
            int n = q_block - d;
            if (n < 0 || n >= n_blocks) continue;
            float v = scores[(size_t)t * n_blocks + n];
            if (isfinite(v)) { s += v; valid++; }
        }
        // Normalise so deltas with fewer valid t-positions don't dominate;
        // a delta seen by every t has the same statistical weight as one seen
        // by half the queries.
        delta_agg[d] = (valid > 0) ? (s / valid) : -INFINITY;
    }
    __syncthreads();

    // 5. Top-V vertical indices via single-thread argmax. V_top_k ≤ 64 so
    //    O(V * n_blocks) is cheap.
    int* row_v = vertical_idx + (size_t)kv_head * V_top_k;
    int* row_s = slash_idx    + (size_t)kv_head * S_top_k;
    if (tid == 0) {
        for (int k = 0; k < V_top_k; k++) {
            float best = -INFINITY; int bidx = -1;
            for (int n = 0; n < n_blocks; n++)
                if (block_agg[n] > best) { best = block_agg[n]; bidx = n; }
            if (bidx < 0) {
                for (int kk = k; kk < V_top_k; kk++) row_v[kk] = -1;
                break;
            }
            row_v[k] = bidx;
            block_agg[bidx] = -INFINITY;
        }
        for (int k = 0; k < S_top_k; k++) {
            float best = -INFINITY; int bidx = -1;
            for (int d = 0; d < n_blocks; d++)
                if (delta_agg[d] > best) { best = delta_agg[d]; bidx = d; }
            if (bidx < 0) {
                for (int kk = k; kk < S_top_k; kk++) row_s[kk] = -1;
                break;
            }
            row_s[k] = bidx;
            delta_agg[bidx] = -INFINITY;
        }
    }
}

// Per (kv_head, t_idx): build block_index = sorted(vertical_idx ∪
// {q_block(t_idx) - δ for δ in slash_idx}). Causal-clamped and -1 padded.
//
// Grid:  (num_kv, sub_n, 1)
// Block: 1 thread (the per-token assembly is trivially small).
__global__ void build_vs_index_kern(
    const int* __restrict__ vertical_idx,   // [num_kv, V_top_k]
    const int* __restrict__ slash_idx,      // [num_kv, S_top_k]
    int*       __restrict__ block_index,    // [num_kv, sub_n_max, top_k]
    int  num_kv, int sub_n, int sub_n_max,
    int  start_pos, int top_k,
    int  block_size_n, int V_top_k, int S_top_k)
{
    int kv_head = blockIdx.x;
    int t_idx   = blockIdx.y;
    if (t_idx >= sub_n) return;
    if (threadIdx.x != 0) return;

    int abs_pos     = start_pos + t_idx;
    int q_block_max = abs_pos / block_size_n;

    int* row = block_index + ((size_t)kv_head * sub_n_max + t_idx) * top_k;
    const int* vrow = vertical_idx + (size_t)kv_head * V_top_k;
    const int* srow = slash_idx    + (size_t)kv_head * S_top_k;

    // Assemble: vertical first (already sorted by selection order, but may
    // contain duplicates with slash blocks). We dedup + sort below.
    int wpos = 0;
    for (int i = 0; i < V_top_k && wpos < top_k; i++) {
        int b = vrow[i];
        if (b < 0 || b > q_block_max) continue;
        // Reject duplicates against entries we've already written.
        bool dup = false;
        for (int j = 0; j < wpos; j++) if (row[j] == b) { dup = true; break; }
        if (!dup) row[wpos++] = b;
    }
    for (int i = 0; i < S_top_k && wpos < top_k; i++) {
        int d = srow[i];
        if (d < 0) continue;
        int b = q_block_max - d;
        if (b < 0 || b > q_block_max) continue;
        bool dup = false;
        for (int j = 0; j < wpos; j++) if (row[j] == b) { dup = true; break; }
        if (!dup) row[wpos++] = b;
    }
    for (int i = wpos; i < top_k; i++) row[i] = -1;

    // Insertion sort ascending. wpos ≤ V+S which is ≤ 64.
    for (int i = 1; i < wpos; i++) {
        int x = row[i];
        int j = i;
        while (j > 0 && row[j - 1] > x) { row[j] = row[j - 1]; j--; }
        row[j] = x;
    }
}
