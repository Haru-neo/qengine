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

// Stage 1a: K_pool[num_kv, N_blocks, HD] = mean over BLOCK_N tokens of
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

// =============================================================================
// Multi-signature (mBSA) variant: per-block we additionally keep a "max-abs"
// signature that records, for each head_dim coordinate, the largest-magnitude
// activation seen in the block (sign-preserving). Anchor-heavy heads — the
// kind that read one or two specific tokens out of a 64-token window —
// produce K[i, c] vectors with a few "spike" coordinates. Mean-pool dilutes
// those spikes with 63 noise tokens; max-abs preserves them. Block score
// becomes
//     score(block) = max(Q · mean_K, β · Q · max_abs_K)
// so a query lined up with the block's most distinctive token is enough to
// pull the block into the top-k even when the mean dot product is small.
// =============================================================================

// Stage 1b: K_pool_max[num_kv, N_blocks, HD] = element-wise sign-preserving
// max-abs across BLOCK_N tokens. One thread per (kv_head, n_block, d_off).
// Same grid layout as build_k_pool_kern so the dispatcher can launch them
// back-to-back into the same buffer pair.
template<int HD, int BLOCK_N, int BLOCK>
__global__ void build_k_pool_max_kern(
    const half* __restrict__ k_cache,
    half*       __restrict__ k_pool_max,
    int  num_kv, int total_kv)
{
    int kv_head = blockIdx.x;
    int n_block = blockIdx.y;
    int d_off   = blockIdx.z * BLOCK + threadIdx.x;
    if (d_off >= HD) return;

    int t_lo = n_block * BLOCK_N;
    int t_hi = min(t_lo + BLOCK_N, total_kv);
    if (t_lo >= t_hi) {
        k_pool_max[((size_t)kv_head * gridDim.y + n_block) * HD + d_off] = __float2half(0.0f);
        return;
    }

    float best_mag = 0.0f;
    float best_val = 0.0f;
    for (int t = t_lo; t < t_hi; t++) {
        size_t idx = ((size_t)t * num_kv + kv_head) * HD + d_off;
        float v = __half2float(k_cache[idx]);
        float m = fabsf(v);
        if (m > best_mag) { best_mag = m; best_val = v; }
    }
    k_pool_max[((size_t)kv_head * gridDim.y + n_block) * HD + d_off] = __float2half(best_val);
}

// Stage 2 (mBSA): per (kv_head, t_idx) compute scores against BOTH mean and
// max signatures, take element-wise max, then top-k. Same shape contract as
// build_block_index_kern; writes the same block_index buffer the FA kernel
// already consumes.
//
// SMEM layout matches build_block_index_kern + an extra HD halves for q_avg
// (we share q_avg across both score passes — no separate q_max needed
// because we score by Q · K_max, not Q_max · K_max).
template<int HD, int GQA, int BLOCK>
__global__ void build_block_index_ms_kern(
    const half* __restrict__ q_chunk,
    const half* __restrict__ k_pool,       // [num_kv, n_blocks, HD]
    const half* __restrict__ k_pool_max,   // [num_kv, n_blocks, HD]
    int*        __restrict__ block_index,
    int  num_q, int num_kv,
    int  sub_n, int sub_n_max,
    int  start_pos, int top_k,
    int  block_size_n, int n_blocks,
    int  n_blocks_stride,   // pool alloc stride (>= n_blocks); == n_blocks for prefill
    float beta_max)
{
    int kv_head = blockIdx.x;
    int t_idx   = blockIdx.y;
    if (t_idx >= sub_n) return;
    int abs_pos = start_pos + t_idx;
    int tid     = threadIdx.x;

    extern __shared__ unsigned char smem_idx_ms[];
    half*  q_avg  = (half*)smem_idx_ms;
    float* scores = (float*)(q_avg + HD);

    // 1. q_avg[c] = mean over GQA group of Q[t_idx, kv_head*GQA+g, c]
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

    int causal_block_max = (abs_pos / block_size_n);

    // 2. Score every n_block: max(Q·mean, β·Q·max_abs).
    for (int n = tid; n < n_blocks; n += BLOCK) {
        const half* kp_mean = k_pool      + ((size_t)kv_head * n_blocks_stride + n) * HD;
        const half* kp_maxa = k_pool_max  + ((size_t)kv_head * n_blocks_stride + n) * HD;
        float s_mean = 0.0f;
        float s_maxa = 0.0f;
        const half2* qa2 = reinterpret_cast<const half2*>(q_avg);
        const half2* mn2 = reinterpret_cast<const half2*>(kp_mean);
        const half2* mx2 = reinterpret_cast<const half2*>(kp_maxa);
        #pragma unroll 8
        for (int c = 0; c < HD / 2; c++) {
            half2 qv = qa2[c];
            half2 mv = mn2[c];
            half2 xv = mx2[c];
            float2 qf = __half22float2(qv);
            float2 mf = __half22float2(mv);
            float2 xf = __half22float2(xv);
            s_mean += qf.x * mf.x + qf.y * mf.y;
            s_maxa += qf.x * xf.x + qf.y * xf.y;
        }
        // β scales the max-sig contribution. The kernel keeps the signed
        // dot product (not |·|) — a query strongly anti-aligned with a
        // block's spike still gets a low score, which is correct.
        float s = fmaxf(s_mean, beta_max * s_maxa);
        if (n > causal_block_max) s = -INFINITY;
        scores[n] = s;
    }
    __syncthreads();

    // 3. Block-parallel top-k. Each of the top_k picks is a 256-thread argmax
    //    with lowest-index tie-break, then thread 0 masks the winner — the
    //    SELECTION is bit-identical to the serial repeated-argmax (lowest block
    //    index wins equal scores, exactly like the strict `>` ascending scan),
    //    but the per-pick latency drops from O(n_blocks) to O(n_blocks/BLOCK +
    //    log BLOCK). The single-thread scan over n_blocks (~1540 at 100K) per
    //    pick was the dominant builder cost: capping top_k 64->8 moved the
    //    builder 24.3->8.9s, i.e. the top-k loop is ~17s of it. Reduction
    //    scratch lives just past scores[n_blocks]; the launch sizes smem for it.
    int* row = block_index + ((size_t)kv_head * sub_n_max + t_idx) * top_k;
    // Reduction scratch in STATIC shared memory (BLOCK is a compile-time
    // template param) — keeps the dynamic-smem contract unchanged so every
    // existing launch site (prefill + the 27B block-sparse decode path) works
    // without touching its bi_smem sizing.
    __shared__ float rv[BLOCK];        // reduce values
    __shared__ int   ri[BLOCK];        // reduce indices
    for (int k = 0; k < top_k; k++) {
        float lb = -INFINITY; int li = -1;
        for (int n = tid; n < n_blocks; n += BLOCK) {
            float v = scores[n];
            if (v > lb) { lb = v; li = n; }   // strided local argmax, lowest n in subset
        }
        rv[tid] = lb; ri[tid] = li;
        __syncthreads();
        for (int s = BLOCK >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                float ov = rv[tid + s]; int oi = ri[tid + s];
                float mv = rv[tid];     int mi = ri[tid];
                // higher score wins; equal score -> lower block index
                if (oi >= 0 && (ov > mv || (ov == mv && (mi < 0 || oi < mi)))) {
                    rv[tid] = ov; ri[tid] = oi;
                }
            }
            __syncthreads();
        }
        int best = ri[0];
        if (best < 0) {                 // fewer real blocks than top_k -> pad
            for (int kk = k + tid; kk < top_k; kk += BLOCK) row[kk] = -1;
            __syncthreads();
            break;
        }
        if (tid == 0) { row[k] = best; scores[best] = -INFINITY; }
        __syncthreads();
    }
    // Sort ascending (single-thread, top_k<=64) for HBM-friendly tile order.
    if (tid == 0) {
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
