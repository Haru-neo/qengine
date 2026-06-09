// DFlash draft model (z-lab/Qwen3.5-27B-DFlash) — 5-layer non-causal block-
// diffusion drafter for Qwen3.5/3.6-27B speculative decoding.
//
// Architecture (per dflash.py + lucebox-hub/dflash):
//   - 5 full-attention transformer layers
//   - 32 Q heads, 8 KV heads, head_dim=128, intermediate=17408 (≠ target's
//     24/4/256 — draft has its own dims)
//   - fc projection [5*hidden, hidden] fuses 5 captured target hidden states
//   - hidden_norm, out_norm RMS scales
//   - non-causal full attention over [target_feat (ctx_len), noise (16)]
//   - lm_head shared with target (so we emit hidden, target projects to logits)
//
// Phase 1 (this commit): structs + safetensors BF16 → fp16 GPU loader.
// Phase 2+ (forward, capture hook, ddtree, main loop) follow in later patches.

#pragma once

// V100 sm_70 has no BF16 ALU. We treat BF16 as raw uint16_t storage and
// convert to fp32 via bit-shift (BF16 = high-16 bits of fp32 mantissa).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "q8_0_gemm.cuh"  // brings block_q8_0_aligned + GEMM launchers
#include "ops.cuh"        // rms_norm(half*, half*, half*, rows, hidden, eps, stream)
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace dflash {

// ── Compile-time draft dimensions (z-lab/Qwen3.5-27B-DFlash, fixed) ────────
struct DraftConfig {
    static constexpr int hidden_size       = 5120;
    static constexpr int intermediate_size = 17408;
    static constexpr int num_layers        = 5;
    static constexpr int num_q_heads       = 32;
    static constexpr int num_kv_heads      = 8;
    static constexpr int head_dim          = 128;
    static constexpr int q_dim             = num_q_heads  * head_dim;  // 4096
    static constexpr int kv_dim            = num_kv_heads * head_dim;  // 1024
    static constexpr int block_size        = 16;
    static constexpr int n_target_layers   = 5;            // {1,16,31,46,61}
    static constexpr int mask_token_id     = 248070;
    static constexpr float rope_theta      = 10000000.0f;
    static constexpr float rms_eps         = 1e-6f;
};

// Indices into target's layer outputs that the draft consumes.
static constexpr int kTargetLayerIds[5] = {1, 16, 31, 46, 61};

// ── Per-layer weights ───────────────────────────────────────────────────────
// GEMM weights (q/k/v/o/gate/up/down) stored as Q8_0 for DP4A path; scalar
// norms (rmsnorm scales, q_norm/k_norm) stay fp16.
struct DraftLayer {
    half* attn_norm      = nullptr;  // [hidden]                  fp16
    half* post_attn_norm = nullptr;  // [hidden]                  fp16
    block_q8_0_aligned* wq     = nullptr;  // [q_dim, hidden]     Q8_0
    block_q8_0_aligned* wk     = nullptr;  // [kv_dim, hidden]    Q8_0
    block_q8_0_aligned* wv     = nullptr;  // [kv_dim, hidden]    Q8_0
    block_q8_0_aligned* wo     = nullptr;  // [hidden, q_dim]     Q8_0
    half* q_norm         = nullptr;  // [head_dim]                fp16
    half* k_norm         = nullptr;  // [head_dim]                fp16
    block_q8_0_aligned* w_gate = nullptr;  // [intermediate,hidden] Q8_0
    block_q8_0_aligned* w_up   = nullptr;  // [intermediate,hidden] Q8_0
    block_q8_0_aligned* w_down = nullptr;  // [hidden,intermediate] Q8_0
};

// ── Draft model: weights + per-step working buffers ────────────────────────
struct DraftModel {
    int device = 0;     // CUDA device the draft lives on
    int max_ctx = 0;    // max context length (for target_feat allocation)

    // Top-level weights
    block_q8_0_aligned* fc = nullptr;  // [hidden, 5*hidden]  Q8_0 (M=hidden, K=5*hidden)
    half* hidden_norm = nullptr;       // [hidden]            fp16
    half* out_norm    = nullptr;       // [hidden]            fp16

    DraftLayer layers[DraftConfig::num_layers];

    // Working buffers — sized once at init.
    // target_hidden_cat[ctx_len, 5*hidden]  fp16, populated by target capture hook
    // target_feat     [ctx_len, hidden]     after fc projection + RMSNorm(hidden_norm)
    // K/V context cache for [target_feat → wk/wv] is recomputed each draft step
    // since it's cheap (5 layers × ctx_len × kv_dim).
    half* target_hidden_cat = nullptr;   // owned
    half* target_feat       = nullptr;   // [ctx_len, hidden]
    half* noise_embed       = nullptr;   // [block_size, hidden]
    half* h_buf             = nullptr;   // [block_size, hidden]  (per-layer in/out)

    // ── Incremental context K/V cache (draft_forward_cached) ────────────────
    // The drafter conditions on the 27B features C of the committed context. C
    // for a given absolute position is fixed once committed (causal), so the
    // PRE-RoPE / post-k_norm context K and the context V depend only on the
    // absolute position and can be cached. (Post-RoPE K cannot — prepare_positions
    // uses WINDOW-RELATIVE positions, which shift on every window slide; so RoPE
    // is re-applied fresh each iteration.) Indexed by absolute_pos % cache_window,
    // exactly like the C ring in model.cuh, so unwrapping mirrors dflash_window_view.
    // This turns the per-iteration drafter cost from O(ctx_len) (full fc+wk/wv
    // recompute, ~50% of gen time at 3.4K) into O(newly-accepted-tokens).
    half* kctx_cache[DraftConfig::num_layers] = {nullptr};  // [W, kv_dim] pre-RoPE, post-k_norm
    half* vctx_cache[DraftConfig::num_layers] = {nullptr};  // [W, kv_dim]
    int   cache_window   = 0;    // == max_ctx_len passed to load_draft (the drafter window W)
    int   cache_last_pos = -1;   // highest absolute position whose K/V is cached; -1 = empty (reset per request)

    // Per-layer scratch
    half* q_buf  = nullptr;              // [block_size, q_dim]
    half* k_buf  = nullptr;              // [ctx+block, kv_dim]
    half* v_buf  = nullptr;              // [ctx+block, kv_dim]
    half* attn_out_buf = nullptr;        // [block_size, q_dim]
    half* gate_buf = nullptr;            // [block_size, intermediate]
    half* up_buf   = nullptr;            // [block_size, intermediate]

    // Q8_1 input scratch: per-token quantized inputs to GEMMs.
    // Sized for the largest input row count we'll quantize:
    //   - target_hidden_cat: [ctx_len, 5*hidden]   → q8_1 needs ctx_len rows × (5*hidden/32) blocks
    //   - target_feat / noise / h_buf : [block_size or ctx_len, hidden]
    block_q8_1* xq_scratch = nullptr;    // [ctx+block_size, max(5*hidden, hidden, intermediate)/32]
    size_t xq_scratch_blocks = 0;

    bool loaded = false;
};

// ── BF16 → fp16 conversion kernel (sm_70 safe — no BF16 ALU used) ─────────
// BF16 layout: 1 sign + 8 exp + 7 mantissa = high 16 bits of fp32. So
// fp32 = ((uint32_t)bf16) << 16. Then __float2half is a regular fp16 op.
__global__ void bf16_to_fp16_kernel(
    const uint16_t* __restrict__ src,   // BF16 stored as raw uint16
    half* __restrict__ dst,
    size_t n
) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t f32_bits = (uint32_t)src[i] << 16;
    float    f;
    // Use a union/memcpy to reinterpret bits as float.
    memcpy(&f, &f32_bits, sizeof(float));
    dst[i] = __float2half(f);
}

inline void launch_bf16_to_fp16(const uint16_t* src, half* dst, size_t n, cudaStream_t st = 0) {
    int threads = 256;
    int blocks  = (int)((n + threads - 1) / threads);
    bf16_to_fp16_kernel<<<blocks, threads, 0, st>>>(src, dst, n);
}

// ── Tiny safetensors header parser ─────────────────────────────────────────
struct StTensor {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t data_start;
    uint64_t data_end;
};

inline bool parse_st_header(const char* h, size_t hlen,
                            std::unordered_map<std::string, StTensor>& out) {
    auto skip_ws = [&](size_t& i) {
        while (i < hlen && (h[i] == ' ' || h[i] == '\t' || h[i] == '\n' || h[i] == '\r')) i++;
    };
    auto match_str = [&](size_t& i, std::string& s) {
        if (i >= hlen || h[i] != '"') return false;
        i++;
        size_t st = i;
        while (i < hlen && h[i] != '"') i++;
        if (i >= hlen) return false;
        s.assign(h + st, i - st);
        i++;
        return true;
    };

    size_t i = 0;
    skip_ws(i);
    if (i >= hlen || h[i] != '{') return false;
    i++;
    while (true) {
        skip_ws(i);
        if (i >= hlen) return false;
        if (h[i] == '}') { i++; break; }
        if (h[i] == ',') { i++; skip_ws(i); }

        std::string name;
        if (!match_str(i, name)) return false;
        skip_ws(i);
        if (i >= hlen || h[i] != ':') return false;
        i++;
        skip_ws(i);
        if (i >= hlen || h[i] != '{') return false;

        // Find matching close brace
        int depth = 0;
        size_t obj_st = i;
        for (; i < hlen; i++) {
            if (h[i] == '{') depth++;
            else if (h[i] == '}') { depth--; if (depth == 0) { i++; break; } }
        }
        if (depth != 0) return false;

        if (name == "__metadata__") continue;

        std::string obj(h + obj_st, i - obj_st);
        StTensor t;

        // dtype
        size_t p = obj.find("\"dtype\"");
        if (p == std::string::npos) return false;
        p = obj.find('"', p + 7); if (p == std::string::npos) return false;
        size_t q = obj.find('"', p + 1); if (q == std::string::npos) return false;
        t.dtype = obj.substr(p + 1, q - p - 1);

        // shape
        p = obj.find("\"shape\"");
        if (p == std::string::npos) return false;
        p = obj.find('[', p);
        size_t sq = obj.find(']', p);
        if (p == std::string::npos || sq == std::string::npos) return false;
        std::string slist = obj.substr(p + 1, sq - p - 1);
        size_t pos = 0;
        while (pos < slist.size()) {
            while (pos < slist.size() && (slist[pos] == ' ' || slist[pos] == ',')) pos++;
            if (pos >= slist.size()) break;
            size_t end = pos;
            while (end < slist.size() && slist[end] != ',') end++;
            t.shape.push_back(std::strtoll(slist.c_str() + pos, nullptr, 10));
            pos = end;
        }

        // data_offsets
        p = obj.find("\"data_offsets\"");
        if (p == std::string::npos) return false;
        p = obj.find('[', p);
        sq = obj.find(']', p);
        if (p == std::string::npos || sq == std::string::npos) return false;
        std::string olist = obj.substr(p + 1, sq - p - 1);
        std::vector<uint64_t> offs;
        pos = 0;
        while (pos < olist.size()) {
            while (pos < olist.size() && (olist[pos] == ' ' || olist[pos] == ',')) pos++;
            if (pos >= olist.size()) break;
            size_t end = pos;
            while (end < olist.size() && olist[end] != ',') end++;
            offs.push_back(std::strtoull(olist.c_str() + pos, nullptr, 10));
            pos = end;
        }
        if (offs.size() != 2) return false;
        t.data_start = offs[0];
        t.data_end   = offs[1];

        out[name] = std::move(t);
    }
    return true;
}

// Load one BF16 tensor by name into a freshly-allocated fp16 GPU buffer.
// Returns nullptr on error.
inline half* load_bf16_to_fp16_gpu(
    const std::unordered_map<std::string, StTensor>& tensors,
    const uint8_t* data_base,
    const std::string& name,
    size_t expected_count,
    int device
) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        fprintf(stderr, "[dflash] missing tensor: %s\n", name.c_str());
        return nullptr;
    }
    const StTensor& t = it->second;
    if (t.dtype != "BF16") {
        fprintf(stderr, "[dflash] %s: expected BF16, got %s\n", name.c_str(), t.dtype.c_str());
        return nullptr;
    }
    size_t bytes = t.data_end - t.data_start;
    size_t count = bytes / 2;  // BF16 = 2 bytes
    if (count != expected_count) {
        fprintf(stderr, "[dflash] %s: expected %zu elements, got %zu\n",
                name.c_str(), expected_count, count);
        return nullptr;
    }

    cudaSetDevice(device);

    // BF16 raw bytes → GPU as uint16, then convert to fp16 via bit-shift kernel.
    uint16_t* d_bf16 = nullptr;
    cudaError_t e = cudaMalloc(&d_bf16, bytes);
    if (e != cudaSuccess) {
        fprintf(stderr, "[dflash] cudaMalloc bf16: %s\n", cudaGetErrorString(e));
        return nullptr;
    }
    e = cudaMemcpy(d_bf16, data_base + t.data_start, bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        fprintf(stderr, "[dflash] cudaMemcpy bf16: %s\n", cudaGetErrorString(e));
        cudaFree(d_bf16);
        return nullptr;
    }

    half* d_fp16 = nullptr;
    e = cudaMalloc(&d_fp16, count * sizeof(half));
    if (e != cudaSuccess) {
        fprintf(stderr, "[dflash] cudaMalloc fp16: %s\n", cudaGetErrorString(e));
        cudaFree(d_bf16);
        return nullptr;
    }

    launch_bf16_to_fp16(d_bf16, d_fp16, count);
    cudaDeviceSynchronize();
    cudaFree(d_bf16);

    return d_fp16;
}

// Load a BF16 weight matrix [M, K], quantize to Q8_0, free intermediate fp16.
// Returns Q8_0 GPU buffer (caller frees with cudaFree).
inline block_q8_0_aligned* load_bf16_to_q8_gpu(
    const std::unordered_map<std::string, StTensor>& tensors,
    const uint8_t* data_base,
    const std::string& name,
    int M, int K,
    int device
) {
    if (K % 32 != 0) {
        fprintf(stderr, "[dflash] %s: K=%d not multiple of 32\n", name.c_str(), K);
        return nullptr;
    }
    half* fp16 = load_bf16_to_fp16_gpu(tensors, data_base, name, (size_t)M * K, device);
    if (!fp16) return nullptr;

    cudaSetDevice(device);
    int bpr = K / 32;
    size_t q_bytes = (size_t)M * bpr * sizeof(block_q8_0_aligned);
    block_q8_0_aligned* q = nullptr;
    cudaError_t e = cudaMalloc(&q, q_bytes);
    if (e != cudaSuccess) {
        fprintf(stderr, "[dflash] cudaMalloc q8_0 (%zu bytes): %s\n", q_bytes, cudaGetErrorString(e));
        cudaFree(fp16);
        return nullptr;
    }
    q8gemm::launch_quantize_weight_q8_0(fp16, q, M, K);
    cudaDeviceSynchronize();
    cudaFree(fp16);
    return q;
}

// Load full draft model from a safetensors file. Single GPU.
inline bool load_draft(DraftModel& m, const std::string& path, int gpu_id, int max_ctx_len) {
    using C = DraftConfig;
    m.device = gpu_id;
    m.max_ctx = max_ctx_len;

    // Read whole file (3.4 GB) — fits in CPU RAM with mmap. Use mmap for low RAM impact.
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) { fprintf(stderr, "[dflash] cannot open %s\n", path.c_str()); return false; }
    fseek(fp, 0, SEEK_END);
    size_t file_size = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint64_t header_len = 0;
    if (fread(&header_len, 8, 1, fp) != 1) { fclose(fp); return false; }
    std::vector<char> header(header_len);
    if (fread(header.data(), 1, header_len, fp) != header_len) { fclose(fp); return false; }

    std::unordered_map<std::string, StTensor> tensors;
    if (!parse_st_header(header.data(), header_len, tensors)) {
        fprintf(stderr, "[dflash] header parse failed\n");
        fclose(fp); return false;
    }

    // Read raw tensor blob (file_size - 8 - header_len) into one big host buffer.
    size_t blob_size = file_size - 8 - header_len;
    std::vector<uint8_t> blob(blob_size);
    if (fread(blob.data(), 1, blob_size, fp) != blob_size) {
        fprintf(stderr, "[dflash] read blob failed\n");
        fclose(fp); return false;
    }
    fclose(fp);

    cudaSetDevice(gpu_id);

    // Top-level
    m.fc          = load_bf16_to_q8_gpu  (tensors, blob.data(), "fc.weight",          C::hidden_size, C::n_target_layers * C::hidden_size, gpu_id);
    m.hidden_norm = load_bf16_to_fp16_gpu(tensors, blob.data(), "hidden_norm.weight", C::hidden_size, gpu_id);
    m.out_norm    = load_bf16_to_fp16_gpu(tensors, blob.data(), "norm.weight",        C::hidden_size, gpu_id);
    if (!m.fc || !m.hidden_norm || !m.out_norm) return false;

    // Per-layer
    char keybuf[256];
    for (int il = 0; il < C::num_layers; il++) {
        DraftLayer& L = m.layers[il];

        snprintf(keybuf, sizeof(keybuf), "layers.%d.input_layernorm.weight", il);
        L.attn_norm = load_bf16_to_fp16_gpu(tensors, blob.data(), keybuf, C::hidden_size, gpu_id);

        snprintf(keybuf, sizeof(keybuf), "layers.%d.post_attention_layernorm.weight", il);
        L.post_attn_norm = load_bf16_to_fp16_gpu(tensors, blob.data(), keybuf, C::hidden_size, gpu_id);

        snprintf(keybuf, sizeof(keybuf), "layers.%d.self_attn.q_proj.weight", il);
        L.wq = load_bf16_to_q8_gpu(tensors, blob.data(), keybuf, C::q_dim, C::hidden_size, gpu_id);

        snprintf(keybuf, sizeof(keybuf), "layers.%d.self_attn.k_proj.weight", il);
        L.wk = load_bf16_to_q8_gpu(tensors, blob.data(), keybuf, C::kv_dim, C::hidden_size, gpu_id);

        snprintf(keybuf, sizeof(keybuf), "layers.%d.self_attn.v_proj.weight", il);
        L.wv = load_bf16_to_q8_gpu(tensors, blob.data(), keybuf, C::kv_dim, C::hidden_size, gpu_id);

        snprintf(keybuf, sizeof(keybuf), "layers.%d.self_attn.o_proj.weight", il);
        L.wo = load_bf16_to_q8_gpu(tensors, blob.data(), keybuf, C::hidden_size, C::q_dim, gpu_id);

        snprintf(keybuf, sizeof(keybuf), "layers.%d.self_attn.q_norm.weight", il);
        L.q_norm = load_bf16_to_fp16_gpu(tensors, blob.data(), keybuf, C::head_dim, gpu_id);

        snprintf(keybuf, sizeof(keybuf), "layers.%d.self_attn.k_norm.weight", il);
        L.k_norm = load_bf16_to_fp16_gpu(tensors, blob.data(), keybuf, C::head_dim, gpu_id);

        snprintf(keybuf, sizeof(keybuf), "layers.%d.mlp.gate_proj.weight", il);
        L.w_gate = load_bf16_to_q8_gpu(tensors, blob.data(), keybuf, C::intermediate_size, C::hidden_size, gpu_id);

        snprintf(keybuf, sizeof(keybuf), "layers.%d.mlp.up_proj.weight", il);
        L.w_up = load_bf16_to_q8_gpu(tensors, blob.data(), keybuf, C::intermediate_size, C::hidden_size, gpu_id);

        snprintf(keybuf, sizeof(keybuf), "layers.%d.mlp.down_proj.weight", il);
        L.w_down = load_bf16_to_q8_gpu(tensors, blob.data(), keybuf, C::hidden_size, C::intermediate_size, gpu_id);

        if (!L.attn_norm || !L.post_attn_norm || !L.wq || !L.wk || !L.wv || !L.wo
            || !L.q_norm || !L.k_norm || !L.w_gate || !L.w_up || !L.w_down) {
            fprintf(stderr, "[dflash] layer %d incomplete\n", il);
            return false;
        }
    }

    // Allocate working buffers (sized for max_ctx_len)
    // target_hidden_cat: [max_ctx, 5*hidden]
    // target_feat:       [max_ctx, hidden]
    // noise_embed/h_buf: [block_size, hidden]
    // q_buf:             [block_size, q_dim]
    // k_buf, v_buf:      [max_ctx + block_size, kv_dim]
    // attn_out_buf:      [block_size, q_dim]
    // gate_buf, up_buf:  [block_size, intermediate]
    auto alloc = [&](half** ptr, size_t n) -> bool {
        cudaError_t e = cudaMalloc(ptr, n * sizeof(half));
        if (e != cudaSuccess) {
            fprintf(stderr, "[dflash] cudaMalloc fail (%zu halfs): %s\n", n, cudaGetErrorString(e));
            return false;
        }
        return true;
    };

    int K = max_ctx_len;
    int Kb = K + C::block_size;
    if (!alloc(&m.target_hidden_cat, (size_t)K * C::n_target_layers * C::hidden_size)) return false;
    if (!alloc(&m.target_feat,       (size_t)K * C::hidden_size)) return false;
    if (!alloc(&m.noise_embed,       (size_t)C::block_size * C::hidden_size)) return false;
    if (!alloc(&m.h_buf,             (size_t)C::block_size * C::hidden_size)) return false;
    if (!alloc(&m.q_buf,             (size_t)C::block_size * C::q_dim)) return false;
    if (!alloc(&m.k_buf,             (size_t)Kb * C::kv_dim)) return false;
    if (!alloc(&m.v_buf,             (size_t)Kb * C::kv_dim)) return false;
    if (!alloc(&m.attn_out_buf,      (size_t)C::block_size * C::q_dim)) return false;
    if (!alloc(&m.gate_buf,          (size_t)C::block_size * C::intermediate_size)) return false;
    if (!alloc(&m.up_buf,            (size_t)C::block_size * C::intermediate_size)) return false;

    // Incremental context K/V cache rings (one per draft layer), sized to the window.
    m.cache_window   = K;
    m.cache_last_pos = -1;
    for (int il = 0; il < C::num_layers; il++) {
        if (!alloc(&m.kctx_cache[il], (size_t)K * C::kv_dim)) return false;
        if (!alloc(&m.vctx_cache[il], (size_t)K * C::kv_dim)) return false;
    }

    // Q8_1 input scratch — sized for the largest GEMM input we'll quantize:
    //   - target_hidden_cat → fc:  N=ctx_len, K=5*hidden,           blocks/row = 5*hidden/32 = 800
    //   - target_feat → wk/wv:     N=ctx_len, K=hidden,             blocks/row = 160
    //   - h_buf → wq/wk/wv:        N=block_size, K=hidden,          blocks/row = 160
    //   - h_buf → w_gate/w_up:     N=block_size, K=hidden,          blocks/row = 160
    //   - hf → w_down:             N=block_size, K=intermediate,    blocks/row = 17408/32 = 544
    //   - attn → wo:               N=block_size, K=q_dim,           blocks/row = 4096/32 = 128
    // The largest is target_hidden_cat (ctx_len rows × 800 blocks/row).
    {
        size_t bpr_max = (size_t)(C::n_target_layers * C::hidden_size) / 32; // 800
        size_t rows_max = (size_t)K;  // ctx_len
        m.xq_scratch_blocks = rows_max * bpr_max;
        cudaError_t e = cudaMalloc(&m.xq_scratch, m.xq_scratch_blocks * sizeof(block_q8_1));
        if (e != cudaSuccess) {
            fprintf(stderr, "[dflash] cudaMalloc xq_scratch (%zu blocks): %s\n",
                    m.xq_scratch_blocks, cudaGetErrorString(e));
            return false;
        }
    }

    m.loaded = true;

    // Print memory footprint
    size_t draft_q8 = ((size_t)C::hidden_size * C::n_target_layers * C::hidden_size / 32
                     + 5 * ((size_t)C::q_dim * C::hidden_size / 32 * 2
                          + (size_t)C::kv_dim * C::hidden_size / 32 * 2
                          + (size_t)C::hidden_size * C::q_dim / 32
                          + (size_t)C::intermediate_size * C::hidden_size / 32 * 2
                          + (size_t)C::hidden_size * C::intermediate_size / 32))
                    * sizeof(block_q8_0_aligned);
    printf("[dflash] draft loaded on GPU %d, max_ctx=%d, weight Q8_0 ~%.1f MB\n",
           gpu_id, max_ctx_len, draft_q8 / 1024.0 / 1024.0);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// Draft forward kernels
// ─────────────────────────────────────────────────────────────────────────

// Per-head RMSNorm: x reshaped [head_dim, n_head, n_tokens]; norm over head_dim,
// then multiply by 1D weight [head_dim] (broadcast over heads & tokens).
// Layout: x is contiguous [n_tokens][n_head][head_dim] (row-major: tok·head·dim).
__global__ void per_head_rms_norm_kernel(
    half* __restrict__ x,
    const half* __restrict__ weight,  // [head_dim]
    int n_tokens,
    int n_heads,
    int head_dim,
    float eps
) {
    int tid    = threadIdx.x;
    int head   = blockIdx.x;
    int tok    = blockIdx.y;
    if (head >= n_heads || tok >= n_tokens) return;

    half* row = x + ((size_t)tok * n_heads + head) * head_dim;

    // sum of squares
    float ss = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = __half2float(row[i]);
        ss += v * v;
    }
    __shared__ float ssh[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffff, ss, o);
    if (lane == 0) ssh[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        ss = (tid < (blockDim.x + 31) / 32) ? ssh[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffff, ss, o);
        if (lane == 0) ssh[0] = ss;
    }
    __syncthreads();
    float rms = rsqrtf(ssh[0] / head_dim + eps);

    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = __half2float(row[i]) * rms * __half2float(weight[i]);
        row[i] = __float2half(v);
    }
}

inline void launch_per_head_rms_norm(
    half* x, const half* weight, int n_tokens, int n_heads, int head_dim, float eps,
    cudaStream_t stream = 0
) {
    int threads = 128;
    dim3 grid(n_heads, n_tokens);
    per_head_rms_norm_kernel<<<grid, threads, 0, stream>>>(x, weight, n_tokens, n_heads, head_dim, eps);
}

// NEOX-style RoPE for a tensor [n_tokens, n_heads, head_dim].
// Applies rotation to the first `rope_dim` dimensions of head_dim (NEOX layout:
// pairs are (i, i + rope_dim/2) for i in [0..rope_dim/2)). For DFlash draft,
// rope_dim == head_dim (full RoPE on all 128 dims). theta = 10M.
// positions: [n_tokens] int32
__global__ void rope_neox_kernel(
    half* __restrict__ x,
    const int32_t* __restrict__ positions,  // [n_tokens]
    int n_tokens, int n_heads, int head_dim, int rope_dim,
    float rope_theta
) {
    int tid    = threadIdx.x;
    int head   = blockIdx.x;
    int tok    = blockIdx.y;
    if (head >= n_heads || tok >= n_tokens) return;

    int half_rope = rope_dim / 2;
    if (tid >= half_rope) return;

    half* row = x + ((size_t)tok * n_heads + head) * head_dim;
    float pos = (float)positions[tok];

    // freq: theta^{-2*tid/rope_dim}
    float freq = expf(-((float)tid * 2.0f / (float)rope_dim) * logf(rope_theta));
    float angle = pos * freq;
    float c = cosf(angle);
    float s = sinf(angle);

    // NEOX: pair (i, i + half_rope)
    int idx_a = tid;
    int idx_b = tid + half_rope;
    float xa = __half2float(row[idx_a]);
    float xb = __half2float(row[idx_b]);
    row[idx_a] = __float2half(xa * c - xb * s);
    row[idx_b] = __float2half(xa * s + xb * c);
}

inline void launch_rope_neox(
    half* x, const int32_t* positions, int n_tokens, int n_heads, int head_dim, int rope_dim,
    float rope_theta, cudaStream_t stream = 0
) {
    int threads = (rope_dim / 2 + 31) / 32 * 32;  // round up to warp
    if (threads > 1024) threads = 1024;
    dim3 grid(n_heads, n_tokens);
    rope_neox_kernel<<<grid, threads, 0, stream>>>(x, positions, n_tokens, n_heads, head_dim, rope_dim, rope_theta);
}

// Naive non-causal attention for small q_len (16) and any ctx_len.
// Q: [q_len, n_q_heads, head_dim]
// K: [total_k, n_kv_heads, head_dim]
// V: [total_k, n_kv_heads, head_dim]
// Y: [q_len, n_q_heads, head_dim]
// GQA: q_head → kv_head = q_head / (n_q_heads / n_kv_heads)
__global__ void attn_full_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ Y,
    int q_len, int total_k, int n_q_heads, int n_kv_heads, int head_dim,
    float scale
) {
    int q_tok = blockIdx.x;          // 0..q_len-1
    int q_head = blockIdx.y;         // 0..n_q_heads-1
    int tid = threadIdx.x;
    int kv_head = q_head * n_kv_heads / n_q_heads;

    extern __shared__ float smem[];
    float* scores = smem;            // [total_k]
    float* q_local = smem + total_k; // [head_dim] cache Q

    const half* qp = Q + ((size_t)q_tok * n_q_heads + q_head) * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        q_local[i] = __half2float(qp[i]);
    }
    __syncthreads();

    // 1) scores[k] = (Q · K[k]) * scale
    for (int k = tid; k < total_k; k += blockDim.x) {
        const half* kp = K + ((size_t)k * n_kv_heads + kv_head) * head_dim;
        float s = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            s += q_local[i] * __half2float(kp[i]);
        }
        scores[k] = s * scale;
    }
    __syncthreads();

    // 2) softmax across k
    __shared__ float sh_max, sh_sum;
    if (tid == 0) {
        float mx = -INFINITY;
        for (int k = 0; k < total_k; k++) if (scores[k] > mx) mx = scores[k];
        sh_max = mx;
    }
    __syncthreads();

    float sum = 0.0f;
    for (int k = tid; k < total_k; k += blockDim.x) {
        scores[k] = expf(scores[k] - sh_max);
        sum += scores[k];
    }
    // warp/block reduce sum
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    __shared__ float warp_sums[32];
    int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
        if (tid == 0) sh_sum = sum;
    }
    __syncthreads();
    float inv_sum = 1.0f / sh_sum;

    // 3) output[i] = sum_k scores[k] * V[k][i]
    half* yp = Y + ((size_t)q_tok * n_q_heads + q_head) * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < total_k; k++) {
            const half* vp = V + ((size_t)k * n_kv_heads + kv_head) * head_dim;
            acc += scores[k] * inv_sum * __half2float(vp[i]);
        }
        yp[i] = __float2half(acc);
    }
}

// FA-tiled drafter attention. The legacy attn_full_kernel reads K with one
// thread per key row (each thread streams its own 256B row -> fully
// uncoalesced, ~8-16x HBM amplification + L2 thrash). At the drafter's 4096
// window that kernel dominates the whole draft forward. This version stages
// K/V in BM-row smem tiles with float4 coalesced loads and runs an online-
// softmax accumulation:
//   grid  = (q_len, n_kv_heads), block = 128 threads = 4 warps
//   warp g owns q-head (kv_head*GQA + g): lane r scores tile row r
//   acc:  thread t owns output dim t for all GQA heads
// Row stride is padded (+2 halfs) so lane-r smem reads are conflict-free.
// fp32 math, same softmax up to fp reorder (greedy-equivalent).
template<int HD, int GQA, int BM>
__global__ void attn_full_fa_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ Y,
    int q_len, int total_k, int n_q_heads, int n_kv_heads, float scale
) {
    constexpr int PAD    = 2;
    constexpr int STRIDE = HD + PAD;
    int q_tok   = blockIdx.x;
    int kv_head = blockIdx.y;
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    __shared__ half  q_s[GQA * HD];
    __shared__ half  k_tile[BM * STRIDE];
    __shared__ half  v_tile[BM * STRIDE];
    __shared__ float p_smem[GQA * BM];
    __shared__ float fct_s[GQA];     // online-softmax rescale per head
    __shared__ float linv_s[GQA];    // final 1/l per head

    // Load the GQA query heads for this q_tok (coalesced).
    for (int i = tid; i < GQA * HD; i += blockDim.x) {
        int g = i / HD, c = i - g * HD;
        int q_head = kv_head * GQA + g;
        q_s[i] = Q[((size_t)q_tok * n_q_heads + q_head) * HD + c];
    }
    __syncthreads();

    float m_w = -INFINITY;           // per (warp=head) running max (lanes agree)
    float l_w = 0.0f;                // running denom
    float acc[GQA];                  // thread owns dim=tid for each head
    #pragma unroll
    for (int g = 0; g < GQA; g++) acc[g] = 0.0f;

    constexpr int H2       = HD / 2;             // half2s per row
    constexpr int ROWS_PER = 128 / H2;           // rows loaded per pass

    for (int tile = 0; tile < total_k; tile += BM) {
        int tile_len = min(BM, total_k - tile);
        // ---- cooperative K/V tile load (half2: coalesced 128B/warp, and
        //      4B-aligned for any even STRIDE) ----
        #pragma unroll
        for (int pass = 0; pass < BM / ROWS_PER; pass++) {
            int r  = pass * ROWS_PER + tid / H2;
            int c2 = (tid % H2) * 2;
            if (r < tile_len) {
                size_t base = ((size_t)(tile + r) * n_kv_heads + kv_head) * HD + c2;
                *(half2*)&k_tile[r * STRIDE + c2] = *(const half2*)&K[base];
                *(half2*)&v_tile[r * STRIDE + c2] = *(const half2*)&V[base];
            }
        }
        __syncthreads();

        // ---- scores: warp g, lane r ----
        if (warp < GQA) {
            int g = warp;
            float s = -INFINITY;
            if (lane < tile_len) {
                float acc_s = 0.0f;
                const half* qp = q_s + g * HD;
                const half* kp = k_tile + lane * STRIDE;
                #pragma unroll
                for (int i = 0; i < HD; i += 2) {
                    half2 qv = *(const half2*)&qp[i];
                    half2 kv = *(const half2*)&kp[i];
                    float2 qf = __half22float2(qv);
                    float2 kf = __half22float2(kv);
                    acc_s += qf.x * kf.x + qf.y * kf.y;
                }
                s = acc_s * scale;
            }
            // tile max
            float m_tile = s;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, o));
            float m_new  = fmaxf(m_w, m_tile);
            float factor = (m_w == -INFINITY) ? 0.0f : expf(m_w - m_new);
            float p = (lane < tile_len) ? expf(s - m_new) : 0.0f;
            float p_sum = p;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                p_sum += __shfl_xor_sync(0xffffffff, p_sum, o);
            l_w = l_w * factor + p_sum;
            m_w = m_new;
            p_smem[g * BM + lane] = p;
            if (lane == 0) fct_s[g] = factor;
        }
        __syncthreads();

        // ---- accumulate output: thread owns dim d = tid ----
        if (tid < HD) {
            #pragma unroll
            for (int g = 0; g < GQA; g++) {
                float a = acc[g] * fct_s[g];
                const float* pr = p_smem + g * BM;
                for (int r = 0; r < tile_len; r++)
                    a += pr[r] * __half2float(v_tile[r * STRIDE + tid]);
                acc[g] = a;
            }
        }
        __syncthreads();   // before next tile overwrites k/v_tile
    }

    if (warp < GQA && lane == 0) linv_s[warp] = 1.0f / l_w;
    __syncthreads();

    if (tid < HD) {
        #pragma unroll
        for (int g = 0; g < GQA; g++) {
            int q_head = kv_head * GQA + g;
            Y[((size_t)q_tok * n_q_heads + q_head) * HD + tid] =
                __float2half(acc[g] * linv_s[g]);
        }
    }
}

inline void launch_attn_full(
    const half* Q, const half* K, const half* V, half* Y,
    int q_len, int total_k, int n_q_heads, int n_kv_heads, int head_dim, float scale,
    cudaStream_t stream = 0
) {
    // FA-tiled fast path (drafter shape: HD=128, GQA=4). DFLASH_ATTN_FA=0
    // reverts to the legacy kernel for A/B.
    static const bool fa_on = !(getenv("DFLASH_ATTN_FA") && atoi(getenv("DFLASH_ATTN_FA")) == 0);
    if (fa_on && head_dim == 128 && n_q_heads == 4 * n_kv_heads) {
        dim3 grid(q_len, n_kv_heads);
        attn_full_fa_kernel<128, 4, 32><<<grid, 128, 0, stream>>>(
            Q, K, V, Y, q_len, total_k, n_q_heads, n_kv_heads, scale);
        return;
    }
    int threads = 128;
    dim3 grid(q_len, n_q_heads);
    size_t smem = (total_k + head_dim) * sizeof(float);
    attn_full_kernel<<<grid, threads, smem, stream>>>(
        Q, K, V, Y, q_len, total_k, n_q_heads, n_kv_heads, head_dim, scale);
}

// SiLU * elementwise multiply: gate[i] = silu(gate[i]) * up[i]
__global__ void silu_mul_kernel(half* gate, const half* up, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    g = g / (1.0f + expf(-g));
    gate[i] = __float2half(g * u);
}

inline void launch_silu_mul(half* gate, const half* up, size_t n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks  = (int)((n + threads - 1) / threads);
    silu_mul_kernel<<<blocks, threads, 0, stream>>>(gate, up, n);
}

// h += delta (residual add) elementwise on n elements.
__global__ void residual_add_kernel(half* h, const half* delta, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    h[i] = __float2half(__half2float(h[i]) + __half2float(delta[i]));
}

inline void launch_residual_add(half* h, const half* delta, size_t n, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks  = (int)((n + threads - 1) / threads);
    residual_add_kernel<<<blocks, threads, 0, stream>>>(h, delta, n);
}

// ─────────────────────────────────────────────────────────────────────────
// Draft forward (one call per draft step)
//
// Inputs:
//   m              : DraftModel with weights loaded
//   target_hidden_cat[ctx_len, 5*hidden] fp16 — concat of target layers
//                                              {1,16,31,46,61} hidden states.
//                                              Already on m.device.
//   noise_embed   [block_size, hidden]   fp16 — embed of [last_tok, MASK*15]
//   positions_q   [block_size]           int32 device — values [ctx_len..ctx_len+block_size-1]
//   positions_k   [ctx_len+block_size]   int32 device — values [0..ctx_len+block_size-1]
//   ctx_len       : current committed length
//
// Output:
//   m.h_buf       [block_size, hidden]   fp16 — final hidden states (caller projects through target lm_head)
// ─────────────────────────────────────────────────────────────────────────
inline void draft_forward(
    DraftModel& m,
    const half* target_hidden_cat,
    const half* noise_embed,
    const int32_t* positions_q,
    const int32_t* positions_k,
    int ctx_len,
    cudaStream_t stream = 0
) {
    using C = DraftConfig;
    int q_len   = C::block_size;
    int total_k = ctx_len + q_len;
    cudaSetDevice(m.device);

    // Step 1: target_feat = rms_norm(fc @ target_hidden_cat) * hidden_norm
    //   target_hidden_cat: [ctx_len, 5*hidden]
    //   fc Q8_0:           [hidden, 5*hidden]   (M=hidden, K=5*hidden)
    //   target_feat:       [ctx_len, hidden]
    {
        int N = ctx_len, K = C::n_target_layers * C::hidden_size, M = C::hidden_size;
        q8gemm::launch_quantize_input_q8_1(target_hidden_cat, m.xq_scratch, N, K, stream);
        q8gemm::launch_gemm_q8_0_q8_1(m.fc, m.xq_scratch, m.target_feat, M, N, K, stream);
        rms_norm(m.target_feat, m.target_feat, m.hidden_norm, N, M, C::rms_eps, stream);
    }

    // Step 2: 5 decoder layers
    // h starts as a copy of noise_embed (we'll modify in place).
    cudaMemcpyAsync(m.h_buf, noise_embed, (size_t)q_len * C::hidden_size * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);

    for (int il = 0; il < C::num_layers; il++) {
        const DraftLayer& L = m.layers[il];

        // 2a. attn pre-norm: hn = rms_norm(h) * attn_norm     — store back to h_buf temp
        // We keep h in m.h_buf and use a scratch (reuse attn_out_buf since no attn computed yet).
        // Actually we need 2 buffers: original h (for residual) + normed.
        // Store normed in attn_out_buf reused for hn.
        half* hn = m.attn_out_buf;  // borrow as scratch; will be overwritten by attn output later
        rms_norm(hn, m.h_buf, L.attn_norm, q_len, C::hidden_size, C::rms_eps, stream);

        // 2b. Q from noise (hn): wq @ hn → q_buf[q_len, q_dim]
        {
            q8gemm::launch_quantize_input_q8_1(hn, m.xq_scratch, q_len, C::hidden_size, stream);
            q8gemm::launch_gemm_q8_0_q8_1(L.wq, m.xq_scratch, m.q_buf, C::q_dim, q_len, C::hidden_size, stream);
            // q_norm: per-head over head_dim
            launch_per_head_rms_norm(m.q_buf, L.q_norm, q_len, C::num_q_heads, C::head_dim, C::rms_eps, stream);
        }

        // 2c. K and V from concat[target_feat (ctx_len), hn (q_len)]
        //   K_ctx = wk @ target_feat → k_buf[0..ctx_len)
        //   K_n   = wk @ hn          → k_buf[ctx_len..total_k)
        //   V_ctx = wv @ target_feat → v_buf[0..ctx_len)
        //   V_n   = wv @ hn          → v_buf[ctx_len..total_k)
        // Reuse xq_scratch for both: quantize target_feat once, then quantize hn once.
        {
            // K_ctx, V_ctx from target_feat
            q8gemm::launch_quantize_input_q8_1(m.target_feat, m.xq_scratch, ctx_len, C::hidden_size, stream);
            q8gemm::launch_gemm_q8_0_q8_1(L.wk, m.xq_scratch, m.k_buf, C::kv_dim, ctx_len, C::hidden_size, stream);
            q8gemm::launch_gemm_q8_0_q8_1(L.wv, m.xq_scratch, m.v_buf, C::kv_dim, ctx_len, C::hidden_size, stream);
            // K_n, V_n from hn — write into k_buf[ctx_len..], v_buf[ctx_len..]
            q8gemm::launch_quantize_input_q8_1(hn, m.xq_scratch, q_len, C::hidden_size, stream);
            half* k_n = m.k_buf + (size_t)ctx_len * C::kv_dim;
            half* v_n = m.v_buf + (size_t)ctx_len * C::kv_dim;
            q8gemm::launch_gemm_q8_0_q8_1(L.wk, m.xq_scratch, k_n, C::kv_dim, q_len, C::hidden_size, stream);
            q8gemm::launch_gemm_q8_0_q8_1(L.wv, m.xq_scratch, v_n, C::kv_dim, q_len, C::hidden_size, stream);

            // k_norm: per-head
            launch_per_head_rms_norm(m.k_buf, L.k_norm, total_k, C::num_kv_heads, C::head_dim, C::rms_eps, stream);
        }

        // 2d. RoPE NEOX θ=10M on Q (positions_q) and K (positions_k)
        launch_rope_neox(m.q_buf, positions_q, q_len,   C::num_q_heads,  C::head_dim, C::head_dim, C::rope_theta, stream);
        launch_rope_neox(m.k_buf, positions_k, total_k, C::num_kv_heads, C::head_dim, C::head_dim, C::rope_theta, stream);

        // 2e. Non-causal full attention (no mask)
        float scale = 1.0f / sqrtf((float)C::head_dim);
        launch_attn_full(m.q_buf, m.k_buf, m.v_buf, m.attn_out_buf,
                         q_len, total_k, C::num_q_heads, C::num_kv_heads, C::head_dim,
                         scale, stream);

        // 2f. Output projection + residual: h += wo @ attn
        {
            // Reuse hn buffer (=attn_out_buf) source — but attn_out_buf already holds attn output.
            // We need a fresh quantized buffer for attn — quantize attn_out_buf to xq_scratch.
            q8gemm::launch_quantize_input_q8_1(m.attn_out_buf, m.xq_scratch, q_len, C::q_dim, stream);
            // wo[hidden, q_dim] @ attn[q_len, q_dim]^T → [q_len, hidden] but we want token-major.
            // gemm produces Y[N, M], so M=hidden, N=q_len, K=q_dim → Y[q_len, hidden]. Good.
            half* attn_proj = m.q_buf;  // reuse q_buf as scratch
            q8gemm::launch_gemm_q8_0_q8_1(L.wo, m.xq_scratch, attn_proj, C::hidden_size, q_len, C::q_dim, stream);
            launch_residual_add(m.h_buf, attn_proj, (size_t)q_len * C::hidden_size, stream);
        }

        // 2g. FFN pre-norm: hf = rms_norm(h) * post_attn_norm
        half* hf = m.attn_out_buf;  // reuse
        rms_norm(hf, m.h_buf, L.post_attn_norm, q_len, C::hidden_size, C::rms_eps, stream);

        // 2h. SwiGLU: down(silu(gate(hf)) * up(hf))
        {
            q8gemm::launch_quantize_input_q8_1(hf, m.xq_scratch, q_len, C::hidden_size, stream);
            q8gemm::launch_gemm_q8_0_q8_1(L.w_gate, m.xq_scratch, m.gate_buf, C::intermediate_size, q_len, C::hidden_size, stream);
            q8gemm::launch_gemm_q8_0_q8_1(L.w_up,   m.xq_scratch, m.up_buf,   C::intermediate_size, q_len, C::hidden_size, stream);
            launch_silu_mul(m.gate_buf, m.up_buf, (size_t)q_len * C::intermediate_size, stream);

            // down: w_down @ silu_mul → [q_len, hidden]
            q8gemm::launch_quantize_input_q8_1(m.gate_buf, m.xq_scratch, q_len, C::intermediate_size, stream);
            half* ffn_out = m.q_buf;  // reuse
            q8gemm::launch_gemm_q8_0_q8_1(L.w_down, m.xq_scratch, ffn_out, C::hidden_size, q_len, C::intermediate_size, stream);
            launch_residual_add(m.h_buf, ffn_out, (size_t)q_len * C::hidden_size, stream);
        }
    }

    // Step 3: final norm — h_buf = rms_norm(h_buf) * out_norm
    rms_norm(m.h_buf, m.h_buf, m.out_norm, q_len, C::hidden_size, C::rms_eps, stream);
}

// ─────────────────────────────────────────────────────────────────────────
// Incremental draft forward — bit-identical to draft_forward but O(n_new)
// instead of O(ctx_len). Caches PRE-RoPE/post-k_norm context K and context V
// per layer (indexed by absolute_pos % cache_window). Only the newest `n_new`
// context rows are projected each call; the rest are served from the cache.
// RoPE is still applied fresh over the whole working buffer (window-relative
// positions), so output matches the full-recompute path exactly.
//
//   target_hidden_cat : C_view [ctx_len, 5*hidden] contiguous, chronological
//                       (oldest-in-window first), as returned by dflash_window_view.
//   last_pos          : the REAL absolute position of the newest context row
//                       (= step-1 in the fold path). n_new is derived from
//                       (last_pos - cache_last_pos); on the first call of a
//                       request cache_last_pos is -1 → full cache fill.
//                       cache_last_pos must be reset to -1 per request.
// ─────────────────────────────────────────────────────────────────────────
inline void draft_forward_cached(
    DraftModel& m,
    const half* target_hidden_cat,
    const half* noise_embed,
    const int32_t* positions_q,
    const int32_t* positions_k,
    int ctx_len,
    int last_pos,
    cudaStream_t stream = 0
) {
    using C = DraftConfig;
    int q_len   = C::block_size;
    int total_k = ctx_len + q_len;
    int W       = m.cache_window;
    cudaSetDevice(m.device);

    // n_new from the REAL absolute positions. First call (cache empty) or any
    // non-monotonic jump (new request) → recompute the whole window.
    int n_new = (m.cache_last_pos < 0) ? ctx_len : (last_pos - m.cache_last_pos);
    if (n_new < 0 || n_new > ctx_len) n_new = ctx_len;
    int oldest        = last_pos - ctx_len + 1;        // absolute pos of C_view row 0
    int first_new_abs = last_pos - n_new + 1;          // absolute pos of first new row

    // ── (A) project the new context rows' K/V into the ring ───────────────
    if (n_new > 0) {
        // target_feat for new rows = LAST n_new rows of C_view (newest = newly committed)
        const half* C_new = target_hidden_cat
                          + (size_t)(ctx_len - n_new) * C::n_target_layers * C::hidden_size;
        half* feat_new = m.target_feat;   // reuse [<=W, hidden] scratch (n_new <= ctx_len <= W)
        {
            int N = n_new, K5 = C::n_target_layers * C::hidden_size, M = C::hidden_size;
            q8gemm::launch_quantize_input_q8_1(C_new, m.xq_scratch, N, K5, stream);
            q8gemm::launch_gemm_q8_0_q8_1(m.fc, m.xq_scratch, feat_new, M, N, K5, stream);
            rms_norm(feat_new, feat_new, m.hidden_norm, N, M, C::rms_eps, stream);
        }
        // quantize feat_new ONCE — same input feeds wk/wv across all layers
        q8gemm::launch_quantize_input_q8_1(feat_new, m.xq_scratch, n_new, C::hidden_size, stream);
        int start = ((first_new_abs % W) + W) % W;
        int seg1  = (start + n_new <= W) ? n_new : (W - start);
        int seg2  = n_new - seg1;
        for (int il = 0; il < C::num_layers; il++) {
            const DraftLayer& L = m.layers[il];
            half* kp = m.k_buf;   // transient scratch for the new rows
            half* vp = m.v_buf;
            q8gemm::launch_gemm_q8_0_q8_1(L.wk, m.xq_scratch, kp, C::kv_dim, n_new, C::hidden_size, stream);
            q8gemm::launch_gemm_q8_0_q8_1(L.wv, m.xq_scratch, vp, C::kv_dim, n_new, C::hidden_size, stream);
            launch_per_head_rms_norm(kp, L.k_norm, n_new, C::num_kv_heads, C::head_dim, C::rms_eps, stream);
            // scatter to ring[start .. start+n_new) % W  (1 or 2 contiguous segments)
            cudaMemcpyAsync(m.kctx_cache[il] + (size_t)start * C::kv_dim, kp,
                            (size_t)seg1 * C::kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(m.vctx_cache[il] + (size_t)start * C::kv_dim, vp,
                            (size_t)seg1 * C::kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            if (seg2 > 0) {
                cudaMemcpyAsync(m.kctx_cache[il], kp + (size_t)seg1 * C::kv_dim,
                                (size_t)seg2 * C::kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(m.vctx_cache[il], vp + (size_t)seg1 * C::kv_dim,
                                (size_t)seg2 * C::kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            }
        }
    }
    m.cache_last_pos = last_pos;

    // ── (B) 5-layer forward of the 16 noise tokens over cached context ────
    cudaMemcpyAsync(m.h_buf, noise_embed, (size_t)q_len * C::hidden_size * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
    int ring_start = ((oldest % W) + W) % W;             // ring slot of C_view row 0
    int u_seg1 = (ring_start + ctx_len <= W) ? ctx_len : (W - ring_start);
    int u_seg2 = ctx_len - u_seg1;
    for (int il = 0; il < C::num_layers; il++) {
        const DraftLayer& L = m.layers[il];

        half* hn = m.attn_out_buf;
        rms_norm(hn, m.h_buf, L.attn_norm, q_len, C::hidden_size, C::rms_eps, stream);

        // Q from noise
        q8gemm::launch_quantize_input_q8_1(hn, m.xq_scratch, q_len, C::hidden_size, stream);
        q8gemm::launch_gemm_q8_0_q8_1(L.wq, m.xq_scratch, m.q_buf, C::q_dim, q_len, C::hidden_size, stream);
        launch_per_head_rms_norm(m.q_buf, L.q_norm, q_len, C::num_q_heads, C::head_dim, C::rms_eps, stream);

        // unwrap cached context K/V (pre-RoPE) into k_buf[0..ctx_len), v_buf[0..ctx_len)
        cudaMemcpyAsync(m.k_buf, m.kctx_cache[il] + (size_t)ring_start * C::kv_dim,
                        (size_t)u_seg1 * C::kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(m.v_buf, m.vctx_cache[il] + (size_t)ring_start * C::kv_dim,
                        (size_t)u_seg1 * C::kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        if (u_seg2 > 0) {
            cudaMemcpyAsync(m.k_buf + (size_t)u_seg1 * C::kv_dim, m.kctx_cache[il],
                            (size_t)u_seg2 * C::kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(m.v_buf + (size_t)u_seg1 * C::kv_dim, m.vctx_cache[il],
                            (size_t)u_seg2 * C::kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        }

        // noise K/V appended at [ctx_len..total_k); k_norm the noise rows only
        q8gemm::launch_quantize_input_q8_1(hn, m.xq_scratch, q_len, C::hidden_size, stream);
        half* k_n = m.k_buf + (size_t)ctx_len * C::kv_dim;
        half* v_n = m.v_buf + (size_t)ctx_len * C::kv_dim;
        q8gemm::launch_gemm_q8_0_q8_1(L.wk, m.xq_scratch, k_n, C::kv_dim, q_len, C::hidden_size, stream);
        q8gemm::launch_gemm_q8_0_q8_1(L.wv, m.xq_scratch, v_n, C::kv_dim, q_len, C::hidden_size, stream);
        launch_per_head_rms_norm(k_n, L.k_norm, q_len, C::num_kv_heads, C::head_dim, C::rms_eps, stream);

        // RoPE Q and the FULL K (context pre-RoPE + noise pre-RoPE) — window-relative positions
        launch_rope_neox(m.q_buf, positions_q, q_len,   C::num_q_heads,  C::head_dim, C::head_dim, C::rope_theta, stream);
        launch_rope_neox(m.k_buf, positions_k, total_k, C::num_kv_heads, C::head_dim, C::head_dim, C::rope_theta, stream);

        // attention (non-causal full)
        float scale = 1.0f / sqrtf((float)C::head_dim);
        launch_attn_full(m.q_buf, m.k_buf, m.v_buf, m.attn_out_buf,
                         q_len, total_k, C::num_q_heads, C::num_kv_heads, C::head_dim,
                         scale, stream);

        // output projection + residual
        {
            q8gemm::launch_quantize_input_q8_1(m.attn_out_buf, m.xq_scratch, q_len, C::q_dim, stream);
            half* attn_proj = m.q_buf;
            q8gemm::launch_gemm_q8_0_q8_1(L.wo, m.xq_scratch, attn_proj, C::hidden_size, q_len, C::q_dim, stream);
            launch_residual_add(m.h_buf, attn_proj, (size_t)q_len * C::hidden_size, stream);
        }

        // FFN pre-norm + SwiGLU
        half* hf = m.attn_out_buf;
        rms_norm(hf, m.h_buf, L.post_attn_norm, q_len, C::hidden_size, C::rms_eps, stream);
        {
            q8gemm::launch_quantize_input_q8_1(hf, m.xq_scratch, q_len, C::hidden_size, stream);
            q8gemm::launch_gemm_q8_0_q8_1(L.w_gate, m.xq_scratch, m.gate_buf, C::intermediate_size, q_len, C::hidden_size, stream);
            q8gemm::launch_gemm_q8_0_q8_1(L.w_up,   m.xq_scratch, m.up_buf,   C::intermediate_size, q_len, C::hidden_size, stream);
            launch_silu_mul(m.gate_buf, m.up_buf, (size_t)q_len * C::intermediate_size, stream);
            q8gemm::launch_quantize_input_q8_1(m.gate_buf, m.xq_scratch, q_len, C::intermediate_size, stream);
            half* ffn_out = m.q_buf;
            q8gemm::launch_gemm_q8_0_q8_1(L.w_down, m.xq_scratch, ffn_out, C::hidden_size, q_len, C::intermediate_size, stream);
            launch_residual_add(m.h_buf, ffn_out, (size_t)q_len * C::hidden_size, stream);
        }
    }

    rms_norm(m.h_buf, m.h_buf, m.out_norm, q_len, C::hidden_size, C::rms_eps, stream);
}

inline void free_draft(DraftModel& m) {
    if (!m.loaded) return;
    cudaSetDevice(m.device);
    auto F  = [](half*& p)               { if (p) { cudaFree(p); p = nullptr; } };
    auto FQ = [](block_q8_0_aligned*& p) { if (p) { cudaFree(p); p = nullptr; } };
    auto F1 = [](block_q8_1*& p)         { if (p) { cudaFree(p); p = nullptr; } };
    FQ(m.fc); F(m.hidden_norm); F(m.out_norm);
    for (auto& L : m.layers) {
        F(L.attn_norm); F(L.post_attn_norm);
        FQ(L.wq); FQ(L.wk); FQ(L.wv); FQ(L.wo);
        F(L.q_norm); F(L.k_norm);
        FQ(L.w_gate); FQ(L.w_up); FQ(L.w_down);
    }
    F(m.target_hidden_cat); F(m.target_feat);
    F(m.noise_embed); F(m.h_buf);
    F(m.q_buf); F(m.k_buf); F(m.v_buf);
    F(m.attn_out_buf); F(m.gate_buf); F(m.up_buf);
    for (int il = 0; il < DraftConfig::num_layers; il++) { F(m.kctx_cache[il]); F(m.vctx_cache[il]); }
    F1(m.xq_scratch);
    m.loaded = false;
}

} // namespace dflash
