// Sparse-pattern index builder: mean-pool Q/K into 64-token blocks, score
// each (q, k) block pair, top-k select per (kv_head, t_idx). Output is the
// sorted block index list consumed by `flash_attn_chunk_block_sparse_split`.
//
// Two-stage pipeline so K-pool can amortise across queries within a chunk:
//   Stage 1  build_k_pool_kern : K[total_kv, num_kv, HD] → K_pool[num_kv, N_blocks, HD]
//                                (one warp per output element; mean over BLOCK_N tokens)
//   Stage 2  build_block_index_kern : score Q[t_idx, kv_head, :] · K_pool[kv_head, :, :],
//                                     mask blocks past the causal cap, take top_k,
//                                     write sorted ascending into block_index
//
// We use *averaged* Q across the GQA group (`q_avg` over the GQA q_heads
// sharing this kv_head) to keep the index per-kv_head and avoid loading
// multiple K passes during attention. Per-q_head sparsity is more accurate
// in principle but quadruples K traffic — the offline profiler accounts for
// the q-averaged approximation when scoring patterns.

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// Stage 1: K_pool[num_kv, N_blocks, HD] = mean over BLOCK_N tokens of
// k_cache[total_kv, num_kv, HD]. Each block handles ONE (kv_head, n_block, d_chunk).
// Grid:    (num_kv, N_blocks, ceil(HD/BLOCK))
// Block:   BLOCK threads (256)
template<int HD, int BLOCK_N, int BLOCK>
__global__ void build_k_pool_kern(
    const half* __restrict__ k_cache,
    half*       __restrict__ k_pool,
    int  num_kv, int total_kv)
{
    int kv_head = blockIdx.x;
    int n_block = blockIdx.y;
    int d_off   = blockIdx.z * BLOCK + threadIdx.x;
    if (d_off >= HD) return;

    int t_lo = n_block * BLOCK_N;
    int t_hi = min(t_lo + BLOCK_N, total_kv);
    if (t_lo >= t_hi) {
        k_pool[((size_t)kv_head * gridDim.y + n_block) * HD + d_off] = __float2half(0.0f);
        return;
    }

    float sum = 0.0f;
    for (int t = t_lo; t < t_hi; t++) {
        size_t idx = ((size_t)t * num_kv + kv_head) * HD + d_off;
        sum += __half2float(k_cache[idx]);
    }
    sum /= (float)(t_hi - t_lo);
    k_pool[((size_t)kv_head * gridDim.y + n_block) * HD + d_off] = __float2half(sum);
}

// Stage 2: per (kv_head, t_idx) compute scores against K_pool, top-k select.
// Grid:   (num_kv, sub_n, 1)
// Block:  BLOCK threads (256)
//
// Algorithm:
//   1. Compute q_avg[HD] = mean over GQA q_heads of Q[t_idx, kv_head*GQA+g, :].
//      Stored in shared memory.
//   2. Each thread covers a strided subset of the N_blocks. For each n_block
//      it computes score = q_avg · k_pool[kv_head, n_block, :] (warp dot).
//      Stores into smem scores[N_blocks].
//   3. Causal mask: scores[n] = -inf if (n * BLOCK_N) > abs_pos.
//   4. Single-thread top-k selection (K * N ≈ 32 * 4096 = 130k ops, ~100µs).
//   5. Sort the top-k ascending, write to block_index.
//
// `block_index` shape: [num_kv, sub_n_max, top_k]. Sentinel -1 for unused slots.
template<int HD, int GQA, int BLOCK>
__global__ void build_block_index_kern(
    const half* __restrict__ q_chunk,    // [sub_n, num_q, HD]
    const half* __restrict__ k_pool,     // [num_kv, N_blocks, HD]
    int*        __restrict__ block_index,
    int  num_q, int num_kv,
    int  sub_n, int sub_n_max,
    int  start_pos, int top_k,
    int  block_size_n, int n_blocks)
{
    int kv_head = blockIdx.x;
    int t_idx   = blockIdx.y;
    if (t_idx >= sub_n) return;
    int abs_pos = start_pos + t_idx;
    int tid     = threadIdx.x;

    extern __shared__ unsigned char smem_idx[];
    half*  q_avg  = (half*)smem_idx;                   // HD halves
    float* scores = (float*)(q_avg + HD);              // n_blocks floats

    // 1. q_avg[c] = (1/GQA) Σ_g Q[t_idx, kv_head*GQA+g, c]
    for (int c = tid; c < HD; c += BLOCK) {
        float s = 0.0f;
        #pragma unroll
        for (int g = 0; g < GQA; g++) {
            int q_head = kv_head * GQA + g;
            s += __half2float(q_chunk[(size_t)t_idx * num_q * HD
                                      + (size_t)q_head * HD + c]);
        }
        q_avg[c] = __float2half(s * (1.0f / GQA));
    }
    __syncthreads();

    // 2. Score every n_block against q_avg. One thread per n_block, looping
    //    when n_blocks > BLOCK. Score is fp32 dot of HD halves.
    int causal_block_max = (abs_pos / block_size_n);  // last block we may attend to (inclusive)

    for (int n = tid; n < n_blocks; n += BLOCK) {
        const half* kp = k_pool + ((size_t)kv_head * n_blocks + n) * HD;
        float s = 0.0f;
        const half2* qa2 = reinterpret_cast<const half2*>(q_avg);
        const half2* kp2 = reinterpret_cast<const half2*>(kp);
        #pragma unroll 16
        for (int c = 0; c < HD / 2; c++) {
            half2 qv = qa2[c];
            half2 kv = kp2[c];
            float2 qf = __half22float2(qv);
            float2 kf = __half22float2(kv);
            s += qf.x * kf.x + qf.y * kf.y;
        }
        // 3. Causal mask at block granularity. Past blocks → keep; future → -inf.
        if (n > causal_block_max) s = -INFINITY;
        scores[n] = s;
    }
    __syncthreads();

    // 4. Top-k by repeated argmax (single-thread to keep it simple). top_k is
    //    small (≤ 64 typical) so 130 k ops is cheap relative to attention.
    int* row = block_index + ((size_t)kv_head * sub_n_max + t_idx) * top_k;
    if (tid == 0) {
        for (int k = 0; k < top_k; k++) {
            float best = -INFINITY;
            int   bidx = -1;
            for (int n = 0; n < n_blocks; n++) {
                float v = scores[n];
                if (v > best) { best = v; bidx = n; }
            }
            if (bidx < 0) {
                // Fewer real blocks than top_k — pad with -1 sentinels.
                for (int kk = k; kk < top_k; kk++) row[kk] = -1;
                break;
            }
            row[k] = bidx;
            scores[bidx] = -INFINITY;  // mask out
        }
        // 5. Sort ascending so the FA kernel reads K/V tiles in stride order
        //    (each entry is independent for selection, but adjacent reads cache
        //    better at HBM level). Bubble sort fine for top_k ≤ 64.
        for (int i = 1; i < top_k; i++) {
            int x = row[i];
            if (x < 0) break;
            int j = i;
            while (j > 0 && (row[j - 1] < 0 || row[j - 1] > x)) {
                row[j] = row[j - 1];
                j--;
            }
            row[j] = x;
        }
    }
}
