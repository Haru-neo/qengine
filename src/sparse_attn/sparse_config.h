// MInference-style sparse attention: configuration & per-head pattern table.
//
// Goal
//   Reduce long-context attention from O(N²) to ~O(N · top_k). Each (layer,
//   q_head) is assigned one of {DENSE, BLOCK_SPARSE, VERTICAL_SLASH, A_SHAPE}
//   based on offline calibration. Block-sparse covers the common case (paper:
//   ~70 % of heads); the others target heads with structured patterns that
//   block-sparse can't capture efficiently. Heads marked DENSE keep the
//   existing FA path, so a missing or partial profile degrades gracefully.
//
// Lifecycle
//   1. Offline: tools/profile_sparse_patterns.py runs the engine on a small
//      calibration set with verbose attention dumps. For each (layer, q_head)
//      it picks the pattern + parameters minimizing L2 deviation from dense
//      output, subject to a FLOPs budget.
//   2. Save: profile is written as a flat binary (`sparse_profile.bin`) — a
//      header followed by per-head records.
//   3. Load: SparseProfile::load(path) at engine startup. The path is taken
//      from MINF_PROFILE_PATH; if unset or load fails, sparse attention is
//      disabled and every head stays on the dense path.
//   4. Inference: forward_attn_chunk dispatches per (layer, q_head) according
//      to the loaded table. Index buffers (vertical_idx, slash_idx, block
//      indices) are built per request from the in-flight Q/K and reused
//      across all q_heads in the same kv group when patterns allow.
//
// File format (sparse_profile.bin)
//   struct Header {
//       uint32_t magic;        // 'MINF' = 0x464E494D
//       uint32_t version;      // 1
//       uint32_t num_layers;
//       uint32_t num_q_heads;  // per layer (24 for 27B, 16 for 9B)
//       float    flops_budget; // 0.0–1.0, sparse / dense ratio target
//       uint32_t reserved[3];
//   };
//   struct HeadRecord {        // num_layers × num_q_heads of these
//       uint8_t  pattern;      // 0=DENSE 1=BLOCK_SPARSE 2=VERTICAL_SLASH 3=A_SHAPE
//       uint8_t  pad[3];
//       uint32_t block_top_k;     // BLOCK_SPARSE
//       uint32_t vertical_top_k;  // VERTICAL_SLASH
//       uint32_t slash_top_k;     // VERTICAL_SLASH
//       uint32_t window;          // A_SHAPE: local window radius
//       uint32_t sink;            // A_SHAPE: leading sink size
//       uint32_t reserved[3];
//   };

#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

enum SparsePattern : uint8_t {
    SPARSE_DENSE          = 0,
    SPARSE_BLOCK          = 1,
    SPARSE_VERTICAL_SLASH = 2,
    SPARSE_A_SHAPE        = 3,
    SPARSE_BLOCK_MS       = 4,  // multi-signature (mean + max-abs) block-sparse
};

struct SparseHeadConfig {
    SparsePattern pattern = SPARSE_DENSE;
    uint32_t block_top_k    = 0;
    uint32_t vertical_top_k = 0;
    uint32_t slash_top_k    = 0;
    uint32_t window         = 0;
    uint32_t sink           = 0;
};

struct SparseProfile {
    static constexpr uint32_t MAGIC   = 0x464E494Du;  // 'MINF'
    static constexpr uint32_t VERSION = 1u;

    uint32_t num_layers   = 0;
    uint32_t num_q_heads  = 0;
    float    flops_budget = 0.0f;
    // Flat [num_layers × num_q_heads] table, layer-major.
    std::vector<SparseHeadConfig> heads;

    bool empty() const { return heads.empty(); }
    const SparseHeadConfig& at(int layer, int q_head) const {
        return heads[(size_t)layer * num_q_heads + q_head];
    }

    // Read the binary file produced by the offline profiler. Returns false
    // (and leaves the profile empty) on any mismatch — callers treat empty as
    // "all dense".
    bool load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        struct { uint32_t magic, version, num_layers, num_q_heads; float flops_budget; uint32_t reserved[3]; } hdr;
        if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return false; }
        if (hdr.magic != MAGIC || hdr.version != VERSION) { fclose(f); return false; }
        num_layers   = hdr.num_layers;
        num_q_heads  = hdr.num_q_heads;
        flops_budget = hdr.flops_budget;
        size_t n = (size_t)num_layers * num_q_heads;
        heads.assign(n, {});
        for (size_t i = 0; i < n; i++) {
            struct { uint8_t pattern; uint8_t pad[3]; uint32_t btk, vtk, stk, win, sink; uint32_t reserved[3]; } r;
            if (fread(&r, sizeof(r), 1, f) != 1) { fclose(f); heads.clear(); return false; }
            heads[i].pattern        = (SparsePattern)r.pattern;
            heads[i].block_top_k    = r.btk;
            heads[i].vertical_top_k = r.vtk;
            heads[i].slash_top_k    = r.stk;
            heads[i].window         = r.win;
            heads[i].sink           = r.sink;
        }
        fclose(f);
        return true;
    }

    bool save(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) return false;
        struct { uint32_t magic, version, num_layers, num_q_heads; float flops_budget; uint32_t reserved[3]; } hdr{};
        hdr.magic = MAGIC; hdr.version = VERSION;
        hdr.num_layers = num_layers; hdr.num_q_heads = num_q_heads;
        hdr.flops_budget = flops_budget;
        if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return false; }
        for (auto& h : heads) {
            struct { uint8_t pattern; uint8_t pad[3]; uint32_t btk, vtk, stk, win, sink; uint32_t reserved[3]; } r{};
            r.pattern = (uint8_t)h.pattern;
            r.btk = h.block_top_k; r.vtk = h.vertical_top_k; r.stk = h.slash_top_k;
            r.win = h.window; r.sink = h.sink;
            if (fwrite(&r, sizeof(r), 1, f) != 1) { fclose(f); return false; }
        }
        fclose(f);
        return true;
    }
};

// Per-call sparse runtime parameters (resolved at engine init from env).
struct SparseRuntime {
    bool   enabled         = false;   // MINF_SPARSE_ATTN
    float  flops_budget    = 0.10f;   // MINF_BUDGET (0–1, fraction of dense)
    int    min_seq_for_sparse = 4096; // below this: stay dense
    int    block_size_M    = 64;      // index granularity (Q side)
    int    block_size_N    = 64;      // index granularity (K side)
    SparseProfile profile;            // empty → all heads stay dense
};

inline SparseRuntime parse_sparse_runtime() {
    SparseRuntime r;
    r.enabled = getenv("MINF_SPARSE_ATTN") != nullptr;
    if (const char* b = getenv("MINF_BUDGET")) r.flops_budget = atof(b);
    if (const char* m = getenv("MINF_MIN_SEQ")) r.min_seq_for_sparse = atoi(m);
    if (r.enabled) {
        const char* path = getenv("MINF_PROFILE_PATH");
        if (path && *path) {
            if (!r.profile.load(path)) {
                fprintf(stderr,
                    "[sparse] failed to load profile %s — disabling sparse attention\n",
                    path);
                r.enabled = false;
            } else {
                fprintf(stderr,
                    "[sparse] loaded profile %s: %u layers × %u q_heads, budget=%.3f\n",
                    path, r.profile.num_layers, r.profile.num_q_heads, r.profile.flops_budget);
            }
        } else {
            // No profile supplied. Run in "uniform block-sparse" mode for
            // bring-up + benchmarking. Every head uses BLOCK_SPARSE with the
            // budget driving top_k. The offline profiler will fill in
            // per-head choices later; until then this lets us measure speed
            // vs dense and check qualitative correctness on small prompts.
            fprintf(stderr,
                "[sparse] MINF_SPARSE_ATTN=1, no MINF_PROFILE_PATH — uniform block-sparse for all heads (budget=%.3f, min_seq=%d)\n",
                r.flops_budget, r.min_seq_for_sparse);
        }
    }
    return r;
}
