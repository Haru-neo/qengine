// DDTree builder for DFlash speculative decoding.
//
// Pure CPU. Inputs are draft logits (fp32) for L positions × vocab. Outputs:
//   - top-K (token_id, log_prob) per position
//   - flat DFS-ordered tree with parent_ids[], token_ids[], depths[]
//
// Algorithm (ported from lucebox-hub/dflash test/test_dflash.cpp ~L132–415,
// itself a port of liranringel/ddtree/ddtree.py):
//   1. extract_draft_topk: per-position online logsumexp + top-K min-heap.
//   2. build_ddtree: best-first heap over prefix log-probs. With chain_seed,
//      pre-seed top-1 chain (1..min(L,budget)) and push sibling candidates;
//      else pure best-first. Pop best, append to tree, push next sibling
//      + first child of popped.
//   3. follow_verified_tree: walk argmax posterior down child_map per node.
//
// Used by main loop: draft forward → logits → extract_draft_topk →
// build_ddtree → tokens + parent_ids → upload_parent_ids → forward_*_tree
// → posterior → follow_verified_tree → accept path.

#pragma once

#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>

namespace dflash_ddtree {

struct DDTree {
    int n_nodes = 0;
    std::vector<int32_t> token_ids;        // size n_nodes (slot 1..n)
    std::vector<int>     depths;           // size n_nodes, values 1..L
    std::vector<int>     parents;          // size n_nodes + 1, parents[0] = -1
    std::vector<std::unordered_map<int32_t, int>> child_maps;  // size n_nodes + 1
};

// Per-position online top-K + logsumexp. Single pass over vocab.
//
// logits        : [n_positions × vocab] fp32 (host)
// out_log_probs : [n_positions × K]     fp32 (host) — descending, rank 0 = argmax
// out_token_ids : [n_positions × K]     i32  (host) — matching token ids
//
// temperature < 1 sharpens (compensates draft Q8_0 softmax flattening).
inline void extract_draft_topk(
    const float*   logits,
    int            n_positions,
    int            vocab,
    int            K,
    float*         out_log_probs,
    int32_t*       out_token_ids,
    float          temperature = 1.0f
) {
    struct Entry { float logit; int32_t id; };
    auto cmp_greater = [](const Entry& a, const Entry& b) {
        return a.logit > b.logit;
    };
    const float inv_t = 1.0f / std::max(1e-3f, temperature);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_positions; i++) {
        const float* li = logits + (size_t)i * vocab;
        std::vector<Entry> heap;
        heap.reserve(K);

        float running_max     = -INFINITY;
        float running_sum_exp = 0.0f;
        for (int j = 0; j < vocab; j++) {
            const float l = li[j] * inv_t;

            if (l > running_max) {
                if (running_max > -INFINITY) {
                    running_sum_exp = running_sum_exp * std::exp(running_max - l);
                }
                running_sum_exp += 1.0f;
                running_max = l;
            } else {
                running_sum_exp += std::exp(l - running_max);
            }

            if ((int)heap.size() < K) {
                heap.push_back({l, (int32_t)j});
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            } else if (l > heap.front().logit) {
                std::pop_heap(heap.begin(), heap.end(), cmp_greater);
                heap.back() = {l, (int32_t)j};
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            }
        }
        const float log_z = running_max + std::log(running_sum_exp);

        std::sort_heap(heap.begin(), heap.end(), cmp_greater);
        for (int k = 0; k < K; k++) {
            out_log_probs[(size_t)i * K + k] = heap[k].logit - log_z;
            out_token_ids[(size_t)i * K + k] = heap[k].id;
        }
    }
}

// Best-first heap over prefix log-probs.
//
// top_log_probs / top_token_ids : [L × K] from extract_draft_topk
// L      : draft block size (max tree depth)
// K      : per-position top-K
// budget : max non-root nodes to add
// chain_seed : true → pre-seed full top-1 chain. Defensive against flat-softmax
//              draft (Q8_0 here, was Q4_K_M in ref). false → pure paper best-first.
inline DDTree build_ddtree(
    const float*    top_log_probs,
    const int32_t*  top_token_ids,
    int             L,
    int             K,
    int             budget,
    bool            chain_seed = true
) {
    DDTree tree;
    tree.parents.push_back(-1);
    tree.child_maps.emplace_back();
    if (budget <= 0 || L <= 0) return tree;

    struct HeapEntry {
        float neg_logw;     // sort ascending → highest log-prob popped first
        int   parent_index;
        int   depth;        // 1..L
        int   rank;         // rank within top-K at this depth
        float logw;
    };
    struct HeapCmp {
        bool operator()(const HeapEntry& a, const HeapEntry& b) const {
            return a.neg_logw > b.neg_logw;  // priority_queue is max-heap
        }
    };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp> heap;

    tree.token_ids.reserve(budget);
    tree.depths.reserve(budget);
    tree.parents.reserve(budget + 1);

    if (chain_seed) {
        const int chain_depth = std::min(L, budget);
        float cum_logw = 0.0f;
        int   prev_idx = 0;
        for (int d = 1; d <= chain_depth; d++) {
            const int32_t tok_id = top_token_ids[(size_t)(d - 1) * K + 0];
            cum_logw += top_log_probs[(size_t)(d - 1) * K + 0];

            const int cur_idx = tree.n_nodes + 1;
            tree.token_ids.push_back(tok_id);
            tree.depths.push_back(d);
            tree.parents.push_back(prev_idx);
            tree.child_maps.emplace_back();
            tree.child_maps[prev_idx][tok_id] = cur_idx;
            tree.n_nodes++;

            if (K > 1) {
                const float sibling_logw = cum_logw
                    - top_log_probs[(size_t)(d - 1) * K + 0]
                    + top_log_probs[(size_t)(d - 1) * K + 1];
                heap.push({-sibling_logw, prev_idx, d, 1, sibling_logw});
            }
            prev_idx = cur_idx;
        }
    } else {
        const float root_logw = top_log_probs[0];
        heap.push({-root_logw, /*parent*/0, /*depth*/1, /*rank*/0, root_logw});
    }

    while (!heap.empty() && tree.n_nodes < budget) {
        HeapEntry top = heap.top();
        heap.pop();

        const int     d_minus_1 = top.depth - 1;
        const int     rank      = top.rank;
        const int32_t tok_id    = top_token_ids[(size_t)d_minus_1 * K + rank];

        const int cur_idx = tree.n_nodes + 1;
        tree.token_ids.push_back(tok_id);
        tree.depths.push_back(top.depth);
        tree.parents.push_back(top.parent_index);
        tree.child_maps.emplace_back();
        tree.child_maps[top.parent_index][tok_id] = cur_idx;
        tree.n_nodes++;

        // Next sibling at same depth.
        if (rank + 1 < K) {
            const float sibling_logw = top.logw
                - top_log_probs[(size_t)d_minus_1 * K + rank]
                + top_log_probs[(size_t)d_minus_1 * K + rank + 1];
            heap.push({-sibling_logw, top.parent_index, top.depth, rank + 1, sibling_logw});
        }

        // First child at next depth, top-1 rank under cur_idx.
        if (top.depth < L) {
            const float child_logw = top.logw
                + top_log_probs[(size_t)top.depth * K + 0];
            heap.push({-child_logw, cur_idx, top.depth + 1, 0, child_logw});
        }
    }

    return tree;
}

// Walk verified tree following posterior[i] = target's argmax token at slot i.
// Returns accepted slot indices including root (slot 0). Sets out_next_token
// to the deepest accepted node's posterior argmax (the bonus token that didn't
// match any of its children).
inline std::vector<int> follow_verified_tree(
    const DDTree&    tree,
    const int32_t*   posterior,
    int&             out_next_token
) {
    std::vector<int> accepted;
    accepted.reserve(tree.n_nodes + 1);
    accepted.push_back(0);

    int cur = 0;
    int next_tok = posterior[cur];
    while (true) {
        const auto& children = tree.child_maps[cur];
        auto it = children.find(next_tok);
        if (it == children.end()) break;
        cur = it->second;
        accepted.push_back(cur);
        next_tok = posterior[cur];
    }
    out_next_token = next_tok;
    return accepted;
}

}  // namespace dflash_ddtree
