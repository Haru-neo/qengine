// A-shape sparse pattern: every query attends to a fixed prefix (the
// "attention sink" tokens at positions [0, sink)) plus a sliding local
// window [max(0, abs_pos - window + 1), abs_pos + 1]. When the prefix and
// window blocks together fit in `top_k`, this collapses to a block_index
// the existing block-sparse FA kernel can consume directly — no new
// attention kernel needed, just a deterministic index builder.
//
// Why we reuse the block-sparse kernel
//   The kernel iterates a sorted list of K-block indices. A-shape selects
//   `ceil(sink / BLOCK_N)` sink blocks at the front of the index plus
//   `ceil(window / BLOCK_N) + 1` window blocks straddling abs_pos. After
//   sorting the union ascending we feed it through unchanged. Causal
//   masking inside the kernel handles partial blocks at the window's tail.
//
// Profiler use-case
//   Heads that block_sparse can't match within tolerance (typically the
//   "attention sink" heads in middle layers) often fall under this pattern.
//   The profiler tries a small grid of (sink, window) candidates per head;
//   the engine writes the chosen sink/window into the SparseProfile fields.

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// Build an A-shape block_index per (kv_head, t_idx). Pattern is
// deterministic so we don't need Q/K scoring — each block is decided by
// its position relative to abs_pos.
//
// Grid:    (num_kv, sub_n, 1)
// Block:   1 thread (the work is constant-time per row)
//
// Args:
//   block_index: int32 [num_kv, sub_n_max, top_k]   ← writes here
//   sink_blocks: ceil(sink / block_size_n)
//   window_blocks: ceil(window / block_size_n) + 1   (overlap at tail)
//   start_pos / sub_n: position context for abs_pos = start_pos + t_idx
//   top_k: capacity per row; trailing entries pad with -1
__global__ void build_a_shape_index_kern(
    int*  __restrict__ block_index,
    int  num_kv, int sub_n, int sub_n_max,
    int  start_pos, int top_k,
    int  block_size_n, int sink_blocks, int window_blocks)
{
    int kv_head = blockIdx.x;
    int t_idx   = blockIdx.y;
    if (t_idx >= sub_n) return;
    if (threadIdx.x != 0) return;

    int abs_pos        = start_pos + t_idx;
    int abs_block_max  = abs_pos / block_size_n;     // last touchable block (inclusive)
    int win_first_block = max(0, abs_block_max - window_blocks + 1);

    int* row = block_index + ((size_t)kv_head * sub_n_max + t_idx) * top_k;

    // Walk the index left-to-right. Sink first (always 0..sink_blocks-1),
    // then window blocks, skipping any sink/window overlap so we don't
    // double-list the same block.
    int wpos = 0;  // write cursor

    int s_lim = min(sink_blocks, abs_block_max + 1);
    for (int b = 0; b < s_lim && wpos < top_k; b++) {
        row[wpos++] = b;
    }
    int w_first = max(win_first_block, s_lim);  // skip overlap with sink
    for (int b = w_first; b <= abs_block_max && wpos < top_k; b++) {
        row[wpos++] = b;
    }
    for (; wpos < top_k; wpos++) row[wpos] = -1;  // sentinel pad
}
