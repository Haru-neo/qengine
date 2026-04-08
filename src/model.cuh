#pragma once
#include "gpu_loader.h"
#include "quant_gemv.cuh"
#include "ops.cuh"
#include "gdn_kernels.cuh"
#include "attention.cuh"
#include <string>
#include <cstdio>

struct QwenConfig {
    int hidden_size;
    int intermediate_size;
    int num_layers;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    int vocab_size;
    int linear_k_heads;
    int linear_v_heads;
    int linear_k_dim;
    int linear_v_dim;
    float rms_norm_eps;
    int rope_dim;
};

struct LayerBuffers {
    half* norm_out;      // [hidden_size]
    half* attn_out;      // [hidden_size] or larger
    half* mlp_gate;      // [intermediate_size]
    half* mlp_up;        // [intermediate_size]
    half* mlp_down;      // [hidden_size]
    half* residual;      // [hidden_size]

    // Chunked prefill MLP buffers (token-major, [CHUNK_SIZE * dim]).
    // Allocated lazily on first chunked prefill.
    half* mlp_chunk_norm = nullptr;  // [CHUNK_SIZE * H]  RMSNorm output
    half* mlp_chunk_gate = nullptr;  // [CHUNK_SIZE * I]  ffn_gate proj
    half* mlp_chunk_up   = nullptr;  // [CHUNK_SIZE * I]  ffn_up proj
    half* mlp_chunk_down = nullptr;  // [CHUNK_SIZE * H]  ffn_down proj
};

struct QwenModel {
    QwenConfig cfg;
    GPUModel* gpu;
    LayerBuffers bufs[4];
    QuantInput gpu_qi[4];  // per-GPU reusable Q8 buffer (hidden_size)
    QuantInput gpu_qi_inter[4];  // per-GPU for intermediate_size  // one per GPU
    // Second-token buffer set used by the speculative-decoding path
    // (forward_*_n2). Allocated lazily in alloc_buffers_n2() — only the
    // MLP/attn fields are duplicated here because GDN's per-call temp
    // buffers (gdn_bufs[g].conv_out etc.) are reused across the two
    // sequential GDN forwards within a spec iter.
    LayerBuffers bufs2[4];
    QuantInput gpu_qi2[4];
    QuantInput gpu_qi_inter2[4];
    bool n2_buffers_ready = false;

    // Second-token GDN intermediate buffers (conv_out / core_out / normed_out
    // / proj_out) used by the batched forward_gdn_n2.
    struct GDNBuffers2 {
        float* conv_out;
        half*  core_out;
        half*  normed_out;
        half*  proj_out;
    };
    GDNBuffers2 gdn_bufs2[4];

    void init_config(GGUFFile& gguf) {
        auto arch = gguf.get_str("general.architecture");
        cfg.hidden_size = gguf.get_u32(arch + ".embedding_length");
        cfg.num_layers = gguf.get_u32(arch + ".block_count");
        cfg.num_q_heads = gguf.get_u32(arch + ".attention.head_count");
        cfg.num_kv_heads = gguf.get_u32(arch + ".attention.head_count_kv");
        cfg.head_dim = gguf.get_u32(arch + ".attention.key_length", 256);
        cfg.rms_norm_eps = gguf.get_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        
        // Get intermediate size from first MLP weight
        auto* gate = gpu->get("blk.0.ffn_gate.weight");
        cfg.intermediate_size = gate ? gate->dims[1] : 0;
        
        // Get vocab size from output weight
        auto* out = gpu->get("output.weight");
        cfg.vocab_size = out ? out->dims[1] : 0;
        
        cfg.rope_dim = gguf.get_u32(arch + ".rope.dimension_count", cfg.head_dim / 2);
        printf("Config: hidden=%d, inter=%d, layers=%d, heads=%d/%d, vocab=%d, rope_dim=%d\n",
            cfg.hidden_size, cfg.intermediate_size, cfg.num_layers,
            cfg.num_q_heads, cfg.num_kv_heads, cfg.vocab_size, cfg.rope_dim);
    }

    // Chunk size for parallel scan during prompt processing.
    // 128 doubles the per-chunk batched-GEMM utilisation versus the
    // original 64 (each NB=16 GEMM tile gets 8 column blocks instead of
    // 4) and amortises chunk-loop launch overhead. The chunked attention
    // compute path internally splits CHUNK_SIZE into ATTN_NB-token
    // sub-chunks (see ATTN_NB) so the value kernel's register-resident
    // accumulator stays bounded.
    static constexpr int CHUNK_SIZE = 128;
    // Sub-chunk size used by attn_score / softmax / attn_value chunked
    // kernels. Picked so attn_value's per-thread accumulator
    // (ATTN_NB × gqa_max=8 floats) fits in registers without spill on Volta.
    static constexpr int ATTN_NB = 16;

    // GDN temp buffers per GPU
    struct GDNBuffers {
        float* conv_out;    // [qkv_dim] FP32
        half* core_out;     // [num_v * v_dim]
        half* normed_out;   // [num_v * v_dim]
        half* proj_out;     // [hidden_size]

        // Chunk buffers (per-token data accumulated for chunked GDN)
        float* chunk_qkv;     // [CHUNK_SIZE * qkv_dim] FP32 conv1d outputs
        half*  chunk_qkv_half = nullptr;  // [CHUNK_SIZE * qkv_dim] fp16 staging for batched QKV proj
        half*  chunk_a_proj;  // [CHUNK_SIZE * num_v]
        half*  chunk_b_proj;  // [CHUNK_SIZE * num_v]
        half*  chunk_z_out;   // [CHUNK_SIZE * num_v * v_dim]
        half*  chunk_core_out;// [CHUNK_SIZE * num_v * v_dim]
        half*  chunk_normed;  // [CHUNK_SIZE * num_v * v_dim]
        half*  chunk_proj_out;// [CHUNK_SIZE * hidden_size]
        half*  chunk_norm_out;// [CHUNK_SIZE * hidden_size]
    };
    GDNBuffers gdn_bufs[4];

    void alloc_buffers() {
        int H = cfg.hidden_size;
        int I = cfg.intermediate_size;
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&bufs[g].norm_out, H * sizeof(half));
            cudaMalloc(&bufs[g].attn_out, std::max(H * 4, cfg.num_q_heads * cfg.head_dim * 2) * sizeof(half));
            cudaMalloc(&bufs[g].mlp_gate, I * sizeof(half));
            cudaMalloc(&bufs[g].mlp_up, I * sizeof(half));
            cudaMalloc(&bufs[g].mlp_down, H * sizeof(half));
            cudaMalloc(&bufs[g].residual, H * sizeof(half));

            // Chunked prefill MLP buffers (token-major, [CHUNK_SIZE * dim]).
            cudaMalloc(&bufs[g].mlp_chunk_norm, (size_t)CHUNK_SIZE * H * sizeof(half));
            cudaMalloc(&bufs[g].mlp_chunk_gate, (size_t)CHUNK_SIZE * I * sizeof(half));
            cudaMalloc(&bufs[g].mlp_chunk_up,   (size_t)CHUNK_SIZE * I * sizeof(half));
            cudaMalloc(&bufs[g].mlp_chunk_down, (size_t)CHUNK_SIZE * H * sizeof(half));

            // GDN buffers (over-allocate for 27B max)
            int qkv_dim = 2 * 16 * 128 + 48 * 128;  // 10240
            int v_total = 48 * 128;  // 6144
            int num_v_max = 48;
            cudaMalloc(&gdn_bufs[g].conv_out, qkv_dim * sizeof(float));
            cudaMalloc(&gdn_bufs[g].core_out, v_total * sizeof(half));
            cudaMalloc(&gdn_bufs[g].normed_out, v_total * sizeof(half));
            cudaMalloc(&gdn_bufs[g].proj_out, H * sizeof(half));

            // Chunk buffers
            cudaMalloc(&gdn_bufs[g].chunk_qkv,      CHUNK_SIZE * qkv_dim * sizeof(float));
            cudaMalloc(&gdn_bufs[g].chunk_a_proj,   CHUNK_SIZE * num_v_max * sizeof(half));
            cudaMalloc(&gdn_bufs[g].chunk_b_proj,   CHUNK_SIZE * num_v_max * sizeof(half));
            cudaMalloc(&gdn_bufs[g].chunk_z_out,    CHUNK_SIZE * v_total * sizeof(half));
            cudaMalloc(&gdn_bufs[g].chunk_core_out, CHUNK_SIZE * v_total * sizeof(half));
            cudaMalloc(&gdn_bufs[g].chunk_normed,   CHUNK_SIZE * v_total * sizeof(half));
            cudaMalloc(&gdn_bufs[g].chunk_proj_out, CHUNK_SIZE * H * sizeof(half));
            cudaMalloc(&gdn_bufs[g].chunk_norm_out, CHUNK_SIZE * H * sizeof(half));
        }
    }

    void alloc_buffers_n2(int max_seq = 4096) {
        if (n2_buffers_ready) return;
        int H = cfg.hidden_size;
        int I = cfg.intermediate_size;
        int qkv_dim = 2 * 16 * 128 + 48 * 128;  // GDN qkv_dim
        int v_total = 48 * 128;
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&bufs2[g].norm_out, H * sizeof(half));
            cudaMalloc(&bufs2[g].attn_out, std::max(H * 4, cfg.num_q_heads * cfg.head_dim * 2) * sizeof(half));
            cudaMalloc(&bufs2[g].mlp_gate, I * sizeof(half));
            cudaMalloc(&bufs2[g].mlp_up,   I * sizeof(half));
            cudaMalloc(&bufs2[g].mlp_down, H * sizeof(half));
            cudaMalloc(&bufs2[g].residual, H * sizeof(half));
            // GDN second-token intermediates
            cudaMalloc(&gdn_bufs2[g].conv_out,   qkv_dim * sizeof(float));
            cudaMalloc(&gdn_bufs2[g].core_out,   v_total * sizeof(half));
            cudaMalloc(&gdn_bufs2[g].normed_out, v_total * sizeof(half));
            cudaMalloc(&gdn_bufs2[g].proj_out,   H * sizeof(half));
            // Attention second-token intermediates
            int num_q  = cfg.num_q_heads;
            int num_kv = cfg.num_kv_heads;
            int hd     = cfg.head_dim;
            cudaMalloc(&attn_bufs2[g].q_proj,      num_q  * hd * 2 * sizeof(half));
            cudaMalloc(&attn_bufs2[g].k_proj,      num_kv * hd * sizeof(half));
            cudaMalloc(&attn_bufs2[g].v_proj,      num_kv * hd * sizeof(half));
            cudaMalloc(&attn_bufs2[g].attn_scores, (size_t)num_q * max_seq * sizeof(float));
            cudaMalloc(&attn_bufs2[g].attn_out,    num_q  * hd * sizeof(half));
            cudaMalloc(&attn_bufs2[g].gate_buf,    num_q  * hd * sizeof(half));
        }
        n2_buffers_ready = true;
        printf("[SPEC] N=2 buffers allocated for speculative decoding\n");
    }

    // Get tensor helper
    GPUTensor* t(const std::string& name) { return gpu->get(name); }
    std::string blk(int layer, const std::string& suffix) {
        return "blk." + std::to_string(layer) + "." + suffix;
    }

    // MLP: gate_proj(SiLU) * up_proj -> down_proj
    void forward_mlp(int layer, float* hidden, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& buf = bufs[g];
        int H = cfg.hidden_size;
        int I = cfg.intermediate_size;

        auto* norm_w = t(blk(layer, "post_attention_norm.weight"));
        auto* gate_w = t(blk(layer, "ffn_gate.weight"));
        auto* up_w   = t(blk(layer, "ffn_up.weight"));
        auto* down_w = t(blk(layer, "ffn_down.weight"));

        if (!norm_w || !gate_w || !up_w || !down_w) {
            printf("MLP L%d: missing weights!\n", layer);
            return;
        }

        // RMSNorm: read FP32 hidden, output FP16 norm_out
        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w(buf.norm_out, hidden, (float*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);
        } else {
            rms_norm_f32in(buf.norm_out, hidden, (half*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);
        }

        gpu_qi[g].quantize(buf.norm_out, H, stream);
        quant_gemv(gate_w->data, gate_w->type, buf.norm_out, buf.mlp_gate, H, I, &gpu_qi[g], stream);
        quant_gemv(up_w->data, up_w->type, buf.norm_out, buf.mlp_up, H, I, &gpu_qi[g], stream);
        silu_mul_kernel<<<(I+255)/256, 256, 0, stream>>>(buf.mlp_gate, buf.mlp_up, I);
        gpu_qi_inter[g].quantize(buf.mlp_gate, I, stream);
        quant_gemv(down_w->data, down_w->type, buf.mlp_gate, buf.mlp_down, I, H, &gpu_qi_inter[g], stream);

        // Residual add into FP32 hidden
        add_kernel_f32<<<(H+255)/256, 256, 0, stream>>>(hidden, buf.mlp_down, H);
    }

    // N=2 batched MLP. Processes two hidden states sharing the same weight
    // loads (gate_proj, up_proj, down_proj). Memory traffic stays at the
    // N=1 cost since each weight word is read once and dp4a runs against
    // both inputs. The two RMSNorms, two silu_muls, and two residual adds
    // are sequential but cheap (no GEMV).
    void forward_mlp_n2(int layer, float* hidden_a, float* hidden_b, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& bA = bufs[g];
        auto& bB = bufs2[g];
        int H = cfg.hidden_size;
        int I = cfg.intermediate_size;

        auto* norm_w = t(blk(layer, "post_attention_norm.weight"));
        auto* gate_w = t(blk(layer, "ffn_gate.weight"));
        auto* up_w   = t(blk(layer, "ffn_up.weight"));
        auto* down_w = t(blk(layer, "ffn_down.weight"));

        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w_n2(bA.norm_out, bB.norm_out,
                                   hidden_a, hidden_b,
                                   (float*)norm_w->data, H, cfg.rms_norm_eps, stream);
        } else {
            rms_norm_f32in_n2(bA.norm_out, bB.norm_out,
                              hidden_a, hidden_b,
                              (half*)norm_w->data, H, cfg.rms_norm_eps, stream);
        }

        gpu_qi[g].quantize(bA.norm_out, H, stream);
        gpu_qi2[g].quantize(bB.norm_out, H, stream);

        quant_gemv_n2(gate_w->data, gate_w->type,
                      bA.norm_out, bB.norm_out,
                      bA.mlp_gate, bB.mlp_gate,
                      H, I, &gpu_qi[g], &gpu_qi2[g], stream);
        quant_gemv_n2(up_w->data, up_w->type,
                      bA.norm_out, bB.norm_out,
                      bA.mlp_up, bB.mlp_up,
                      H, I, &gpu_qi[g], &gpu_qi2[g], stream);

        { dim3 sg((I+255)/256, 2);
          silu_mul_n2_kernel<<<sg, 256, 0, stream>>>(
              bA.mlp_gate, bA.mlp_up,
              bB.mlp_gate, bB.mlp_up, I); }

        gpu_qi_inter[g].quantize(bA.mlp_gate, I, stream);
        gpu_qi_inter2[g].quantize(bB.mlp_gate, I, stream);

        quant_gemv_n2(down_w->data, down_w->type,
                      bA.mlp_gate, bB.mlp_gate,
                      bA.mlp_down, bB.mlp_down,
                      I, H, &gpu_qi_inter[g], &gpu_qi_inter2[g], stream);

        { dim3 ag((H+255)/256, 2);
          add_kernel_f32_n2<<<ag, 256, 0, stream>>>(
              hidden_a, bA.mlp_down,
              hidden_b, bB.mlp_down, H); }
    }

    // ============ Attention state ============
    RoPETable rope;
    TurboQuantCache tq_cache;
    int cur_seq_len = 0;
    
    // Attention temp buffers (per GPU)
    struct AttnBuffers {
        half* q_proj;      // [num_q * head_dim * 2] (Q + gate)
        half* k_proj;      // [num_kv * head_dim]
        half* v_proj;      // [num_kv * head_dim]
        float* attn_scores; // [num_q * max_seq]
        half* gate_buf;     // [num_q * head_dim] for output gate
        half* attn_out;     // [num_q * head_dim]
        // Chunked-prefill staging: batched Q/K/V projection outputs.
        // Allocated lazily on first chunked attn forward.
        half* attn_chunk_q = nullptr;   // [CHUNK_SIZE * q_out_dim]  raw QKV proj output
        half* attn_chunk_k = nullptr;   // [CHUNK_SIZE * kv_dim]
        half* attn_chunk_v = nullptr;   // [CHUNK_SIZE * kv_dim]
        // Chunked-attention compute scratch:
        half*  attn_chunk_q_post = nullptr;  // [CHUNK_SIZE * num_q * head_dim] post deinterleave + head_rms + RoPE
        half*  attn_chunk_gate   = nullptr;  // [CHUNK_SIZE * num_q * head_dim] gate slice from QKV
        float* attn_chunk_scores = nullptr;  // [CHUNK_SIZE * num_q * max_seq] softmax scratch
        half*  attn_chunk_out    = nullptr;  // [CHUNK_SIZE * num_q * head_dim] V-weighted output
        half*  attn_chunk_oproj  = nullptr;  // [CHUNK_SIZE * H] o_proj output staging
    };
    AttnBuffers attn_bufs[4];
    AttnBuffers attn_bufs2[4];  // second-token attn buffers, for forward_attn_n2
    
    void init_attention(int max_seq) {
        int num_q = cfg.num_q_heads;    // 24
        int num_kv = cfg.num_kv_heads;  // 4
        int hd = cfg.head_dim;          // 256
        float theta = 10000000.0f;
        
        // RoPE on each GPU that has attention layers
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            // Alloc attention buffers
            cudaMalloc(&attn_bufs[g].q_proj, num_q * hd * 2 * sizeof(half));
            cudaMalloc(&attn_bufs[g].k_proj, num_kv * hd * sizeof(half));
            cudaMalloc(&attn_bufs[g].v_proj, num_kv * hd * sizeof(half));
            cudaMalloc(&attn_bufs[g].attn_scores, num_q * max_seq * sizeof(float));
            cudaMalloc(&attn_bufs[g].attn_out, num_q * hd * sizeof(half));
            cudaMalloc(&attn_bufs[g].gate_buf, num_q * hd * sizeof(half));
        }
        
        // RoPE table (same on all GPUs — just put on GPU 0 for now, copy as needed)
        rope.init(max_seq, hd, cfg.rope_dim, theta, gpu->num_gpus);
        
        // TurboQuant KV cache — only for attention layers, distributed across GPUs
        // For simplicity, per-GPU cache for layers on that GPU
        // Actually for now, each attn layer gets its own cache on its GPU
        // We'll manage it per-layer in forward
        
        // Count attention layers per GPU
        int attn_count = 0;
        for (int l = 0; l < cfg.num_layers; l++)
            if (is_attn_layer(l)) attn_count++;
        
        printf("Attention: %d layers, %d Q heads, %d KV heads, head_dim=%d\n",
            attn_count, num_q, num_kv, hd);
        printf("Max context: %d tokens\n", max_seq);
        init_kv_cache(max_seq);
    }
    
    // Attention forward (seq_len=1, token generation)
    // ============ FP16 KV Cache (memory + speed) ============
    // Precision is recovered by the attn_score_kernel cross-warp fix +
    // FP32 attention compute (accumulators, softmax). The cache itself
    // can stay fp16: the K/V tensors come straight from fp16 head_norm
    // outputs and storing them in fp16 has no extra cast loss.
    struct KVCache {
        half* k;  // [max_seq, num_kv * head_dim]
        half* v;  // [max_seq, num_kv * head_dim]
    };
    std::unordered_map<int, KVCache> kv_caches;
    int kv_max_seq = 0;

    // ============ TurboQuant 3-bit KV cache (optional, MTP_TQ=1) ============
    // Drop-in replacement for the fp16 KV cache that compresses K/V to ~52
    // bytes per 128-element block (~4.7x smaller than fp16). On the read
    // path the active range [0..seq_len) is dequantized into
    // tq_k_buf / tq_v_buf which are then fed to the existing attn_score and
    // attn_value kernels — no kernel rewrite required.
    struct TQKVCache {
        block_tq3* k;       // [max_seq, blocks_per_token]
        block_tq3* v;
        int blocks_per_token;
    };
    std::unordered_map<int, TQKVCache> tq_kv_caches;
    half* tq_k_buf[4] = {nullptr,nullptr,nullptr,nullptr};  // per-GPU dequant scratch
    half* tq_v_buf[4] = {nullptr,nullptr,nullptr,nullptr};
    bool use_turboquant = false;

    void init_kv_cache(int max_seq) {
        kv_max_seq = max_seq;
        int num_kv = cfg.num_kv_heads;
        int hd = cfg.head_dim;
        int kv_dim = num_kv * hd;
        int kv_size = max_seq * kv_dim * sizeof(half);
        use_turboquant = (getenv("MTP_TQ") != nullptr);

        if (use_turboquant) {
            // 3-bit TurboQuant cache. blocks_per_token = kv_dim / 128.
            // For Qwen35: 4*256 / 128 = 8 blocks per token, 52 B each
            // = 416 B per token (vs 2048 B fp16) → 4.92x compression.
            int bpt = kv_dim / TQ_BLOCK_SIZE;
            for (int g = 0; g < gpu->num_gpus; g++) tq3_init_signs(g);
            size_t per_layer_bytes = (size_t)max_seq * bpt * sizeof(block_tq3);
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                if (!is_attn_layer(layer)) continue;
                int g = gpu->layer_gpu[layer];
                cudaSetDevice(g);
                TQKVCache kv;
                kv.blocks_per_token = bpt;
                cudaMalloc(&kv.k, per_layer_bytes);
                cudaMalloc(&kv.v, per_layer_bytes);
                cudaMemset(kv.k, 0, per_layer_bytes);
                cudaMemset(kv.v, 0, per_layer_bytes);
                tq_kv_caches[layer] = kv;
            }
            // Per-GPU dequant scratch (size = max_seq * kv_dim half)
            for (int g = 0; g < gpu->num_gpus; g++) {
                cudaSetDevice(g);
                cudaMalloc(&tq_k_buf[g], (size_t)max_seq * kv_dim * sizeof(half));
                cudaMalloc(&tq_v_buf[g], (size_t)max_seq * kv_dim * sizeof(half));
            }
            float total_mb = (float)tq_kv_caches.size() * per_layer_bytes * 2 / 1e6;
            float fp16_mb  = (float)tq_kv_caches.size() * kv_size * 2 / 1e6;
            printf("KV cache (TurboQuant 3-bit): %d layers, %d max_seq, %.1f MB total (vs %.1f MB fp16, %.2fx compression)\n",
                (int)tq_kv_caches.size(), max_seq, total_mb, fp16_mb, fp16_mb / total_mb);
            return;
        }

        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (!is_attn_layer(layer)) continue;
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            KVCache kv;
            cudaMalloc(&kv.k, kv_size);
            cudaMalloc(&kv.v, kv_size);
            cudaMemset(kv.k, 0, kv_size);
            cudaMemset(kv.v, 0, kv_size);
            kv_caches[layer] = kv;
        }
        printf("KV cache (FP16): %d layers, %d max_seq, %.1f MB total\n",
            (int)kv_caches.size(), max_seq,
            (float)kv_caches.size() * kv_size * 2 / 1e6);
    }

    void reset_kv_cache() {
        int num_kv = cfg.num_kv_heads;
        int hd = cfg.head_dim;
        int kv_dim = num_kv * hd;
        if (use_turboquant) {
            int bpt = kv_dim / TQ_BLOCK_SIZE;
            size_t sz = (size_t)kv_max_seq * bpt * sizeof(block_tq3);
            for (auto& [layer, kv] : tq_kv_caches) {
                int g = gpu->layer_gpu[layer];
                cudaSetDevice(g);
                cudaMemset(kv.k, 0, sz);
                cudaMemset(kv.v, 0, sz);
            }
            return;
        }
        int kv_size = kv_max_seq * kv_dim * sizeof(half);
        for (auto& [layer, kv] : kv_caches) {
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            cudaMemset(kv.k, 0, kv_size);
            cudaMemset(kv.v, 0, kv_size);
        }
    }

    // forward_attn: full per-token attention forward.
    //
    // When `external_proj` is true the caller has ALREADY filled
    // ab.q_proj / ab.k_proj / ab.v_proj for this token (e.g. via a batched
    // chunked Q/K/V projection in forward_attn_chunk). We then skip the
    // RMSNorm + Q/K/V GEMVs and pick up at the deinterleave step. The
    // residual still uses `hidden` so the caller passes the matching token's
    // fp32 hidden slice.
    void forward_attn(int layer, float* hidden, int pos, cudaStream_t stream,
                      bool external_proj = false) {
        int g = gpu->layer_gpu[layer];
        int H = cfg.hidden_size;
        int num_q = cfg.num_q_heads;
        int num_kv = cfg.num_kv_heads;
        int hd = cfg.head_dim;
        int gqa_ratio = num_q / num_kv;
        float eps = cfg.rms_norm_eps;
        float scale = 1.0f / sqrtf((float)hd);
        auto& ab = attn_bufs[g];

        auto* norm_w = t(blk(layer, "attn_norm.weight"));
        auto* q_w = t(blk(layer, "attn_q.weight"));
        auto* k_w = t(blk(layer, "attn_k.weight"));
        auto* v_w = t(blk(layer, "attn_v.weight"));
        auto* q_norm_w = t(blk(layer, "attn_q_norm.weight"));
        auto* k_norm_w = t(blk(layer, "attn_k_norm.weight"));
        auto* o_w = t(blk(layer, "attn_output.weight"));

        half* norm_out = bufs[g].norm_out;
        int total_qg = num_q * hd;
        int kv_dim = num_kv * hd;
        int seq_len = pos + 1;  // including current token
        int q_out_dim = q_w->dims[1];

        if (!external_proj) {
            // 1. RMSNorm (FP32 hidden in)
            if (norm_w->type == GGML_TYPE_F32)
                rms_norm_f32in_f32w(norm_out, hidden, (float*)norm_w->data, 1, H, eps, stream);
            else
                rms_norm_f32in(norm_out, hidden, (half*)norm_w->data, 1, H, eps, stream);

            // 2. Q/K/V projections
            gpu_qi[g].quantize(norm_out, H, stream);
            quant_gemv(q_w->data, q_w->type, norm_out, ab.q_proj, H, q_out_dim, &gpu_qi[g], stream);
            quant_gemv(k_w->data, k_w->type, norm_out, ab.k_proj, H, kv_dim, &gpu_qi[g], stream);
            quant_gemv(v_w->data, v_w->type, norm_out, ab.v_proj, H, kv_dim, &gpu_qi[g], stream);
        }
        // else: caller has already populated ab.q_proj / ab.k_proj / ab.v_proj.

        // 3. Deinterleave Q and gate
        half* q_buf = ab.attn_out;
        half* gate_buf = ab.gate_buf;
        deinterleave_qg_kernel<<<(total_qg+255)/256, 256, 0, stream>>>(
            ab.q_proj, q_buf, gate_buf, num_q, hd);

        // 4. Head RMSNorm
        int tn = min(hd, 128);
        // Debug: print weight types once
        static bool printed_norm_types = false;
        if (!printed_norm_types && layer == 3) {
            printf("[DBG L3] q_norm type=%d (F32=0,F16=1), k_norm type=%d, attn_norm type=%d\n",
                   q_norm_w->type, k_norm_w->type, norm_w->type);
            printed_norm_types = true;
        }
        head_rms_norm_kernel<<<num_q, tn, tn * sizeof(float), stream>>>(
            q_buf, (float*)q_norm_w->data, num_q, hd, eps);
        head_rms_norm_kernel<<<num_kv, tn, tn * sizeof(float), stream>>>(
            ab.k_proj, (float*)k_norm_w->data, num_kv, hd, eps);

        // 5. RoPE
        int rope_dim = rope.rope_dim;
        int half_rope = rope_dim / 2;
        float* sin_pos = rope.sin_table(g) + pos * half_rope;
        float* cos_pos = rope.cos_table(g) + pos * half_rope;
        apply_rope_kernel<<<(num_q * half_rope + 255)/256, 256, 0, stream>>>(
            q_buf, sin_pos, cos_pos, num_q, hd, rope_dim);
        apply_rope_kernel<<<(num_kv * half_rope + 255)/256, 256, 0, stream>>>(
            ab.k_proj, sin_pos, cos_pos, num_kv, hd, rope_dim);

        // 6. Store K, V at position pos (fp16 or TurboQuant 3-bit)
        if (use_turboquant) {
            auto& tq = tq_kv_caches[layer];
            int bpt = tq.blocks_per_token;
            size_t off = (size_t)pos * bpt;
            tq3_quantize_kernel<<<(bpt+31)/32, 32, 0, stream>>>(
                ab.k_proj, &tq.k[off], bpt);
            tq3_quantize_kernel<<<(bpt+31)/32, 32, 0, stream>>>(
                ab.v_proj, &tq.v[off], bpt);
        } else {
            auto& kv = kv_caches[layer];
            half* k_cache_pos = kv.k + (size_t)pos * kv_dim;
            half* v_cache_pos = kv.v + (size_t)pos * kv_dim;
            cudaMemcpyAsync(k_cache_pos, ab.k_proj, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(v_cache_pos, ab.v_proj, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        }

        // 7. Attention: Q[num_q, hd] @ K[seq_len, num_kv, hd]^T → softmax → @ V
        // TQ path uses fused kernels that read block_tq3 directly and dequant
        // cooperatively in shared memory — full 4.92x bandwidth saving on the
        // K/V reads. fp16 path uses the existing kernels.
        dim3 score_grid(num_q, seq_len);
        if (use_turboquant) {
            auto& tq = tq_kv_caches[layer];
            dim3 sg_tq(num_kv, seq_len);
            attn_score_kernel_tq3<<<sg_tq, hd, hd * sizeof(float), stream>>>(
                q_buf, tq.k, ab.attn_scores,
                num_q, num_kv, hd, seq_len, scale);
        } else {
            auto& kv = kv_caches[layer];
            attn_score_kernel_h<<<score_grid, min(hd, 256), 0, stream>>>(
                q_buf, kv.k, ab.attn_scores,
                num_q, num_kv, hd, seq_len, scale);
        }

        { int st = 1; while(st < seq_len && st < 256) st <<= 1;
        softmax_kernel<<<num_q, st, st * sizeof(float), stream>>>(
            ab.attn_scores, num_q, seq_len); }

        if (use_turboquant) {
            auto& tq = tq_kv_caches[layer];
            int blocks_per_kv_head = hd / TQ_BLOCK_SIZE;
            dim3 vg(num_kv, blocks_per_kv_head);
            attn_value_kernel_tq3<<<vg, TQ_BLOCK_SIZE, TQ_BLOCK_SIZE * sizeof(float), stream>>>(
                ab.attn_scores, tq.v, q_buf,
                num_q, num_kv, hd, seq_len);
        } else {
            auto& kv = kv_caches[layer];
            attn_value_kernel_h<<<num_q, min(hd, 256), 0, stream>>>(
                ab.attn_scores, kv.v, q_buf,
                num_q, num_kv, hd, seq_len);
        }

        // 8. Output gate: out *= sigmoid(gate)
        apply_gate_sigmoid<<<(total_qg+255)/256, 256, 0, stream>>>(
            q_buf, gate_buf, total_qg);

        // 9. Output projection
        half* proj_out = bufs[g].mlp_down;
        gpu_qi_inter[g].quantize(q_buf, num_q * hd, stream);
        quant_gemv(o_w->data, o_w->type, q_buf, proj_out, num_q * hd, H, &gpu_qi_inter[g], stream);

        // 10. Residual into FP32 hidden
        add_kernel_f32<<<(H+255)/256, 256, 0, stream>>>(hidden, proj_out, H);
    }

    // ============ GDN Forward ============
    // conv1d state: [qkv_dim, kernel_width] per layer
    // recurrent state: [num_v_heads, k_head_dim, v_head_dim] per layer
    
    struct GDNState {
        float* conv_state;    // [qkv_dim, 4] FP32
        float* rec_state;    // [num_v_heads, k_dim, v_dim] FP32 (Volta has 1:32 fp64)
    };
    std::vector<GDNState> gdn_states;
    
    int gdn_qkv_dim() { return 0; } // computed from tensor
    
    void init_gdn_states() {
        // Get dimensions from first GDN layer
        auto* qkv = t("blk.0.attn_qkv.weight");
        auto* ssm_out = t("blk.0.ssm_out.weight");
        if (!qkv || !ssm_out) { printf("Missing GDN tensors!\n"); return; }
        
        int qkv_dim = qkv->dims[1];  // output dim of QKV
        int v_total = ssm_out->dims[0]; // input dim of out_proj = num_v * v_dim
        
        // Get GDN config from tensor shapes
        auto* a_t = t("blk.0.ssm_alpha.weight");
        int num_v = a_t ? a_t->dims[1] : 48;
        int v_dim = v_total / num_v;
        auto* ssm_a_t = t("blk.0.ssm_a");
        int k_dim = 128; // head_dim for GDN keys (Qwen3.5 = 128)
        // num_k = qkv_dim / k_dim - ... calculate from qkv
        // qkv_dim = 2*num_k*k_dim + num_v*v_dim = 2*16*128 + 48*128 = 10240
        int num_k = (qkv_dim - num_v * v_dim) / (2 * k_dim);
        
        printf("GDN config: qkv_dim=%d, num_k=%d, num_v=%d, k_dim=%d, v_dim=%d\n",
            qkv_dim, num_k, num_v, k_dim, v_dim);
        
        cfg.linear_k_heads = num_k;
        cfg.linear_v_heads = num_v;
        cfg.linear_k_dim = k_dim;
        cfg.linear_v_dim = v_dim;
        
        gdn_states.resize(cfg.num_layers);
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            // Only GDN layers need state (every layer except 3,7,11,...,63)
            bool is_attn = ((layer + 1) % 4 == 0);
            if (is_attn) {
                gdn_states[layer].conv_state = nullptr;
                gdn_states[layer].rec_state = nullptr;
                continue;
            }
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            cudaMalloc(&gdn_states[layer].conv_state, qkv_dim * 4 * sizeof(float));
            cudaMemset(gdn_states[layer].conv_state, 0, qkv_dim * 4 * sizeof(float));
            cudaMalloc(&gdn_states[layer].rec_state, num_v * k_dim * v_dim * sizeof(float));
            cudaMemset(gdn_states[layer].rec_state, 0, num_v * k_dim * v_dim * sizeof(float));
        }
        printf("GDN states allocated for %d layers\n", cfg.num_layers);
    }
    
    void reset_all_states() {
        reset_kv_cache();
        reset_gdn_states_inner();
    }

    void reset_gdn_states_inner() {
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (!gdn_states[layer].conv_state) continue;
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            auto* qkv = t("blk.0.attn_qkv.weight");
            int qkv_dim = qkv->dims[1];
            int num_v = cfg.linear_v_heads;
            int k_dim = cfg.linear_k_dim;
            int v_dim = cfg.linear_v_dim;
            cudaMemset(gdn_states[layer].conv_state, 0, qkv_dim * 4 * sizeof(float));
            cudaMemset(gdn_states[layer].rec_state, 0, num_v * k_dim * v_dim * sizeof(float));
        }
    }
    
    bool is_attn_layer(int layer) { return ((layer + 1) % 4 == 0); }


    // GDN forward (seq_len=1, with cache)
    void forward_gdn(int layer, float* hidden, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& buf = bufs[g];
        int H = cfg.hidden_size;
        int num_k = cfg.linear_k_heads;
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        int qkv_dim = 2 * num_k * k_dim + num_v * v_dim;

        auto* norm_w  = t(blk(layer, "attn_norm.weight"));
        auto* qkv_w   = t(blk(layer, "attn_qkv.weight"));
        auto* gate_w  = t(blk(layer, "attn_gate.weight"));  // Z projection
        auto* alpha_w = t(blk(layer, "ssm_alpha.weight"));
        auto* beta_w  = t(blk(layer, "ssm_beta.weight"));
        auto* conv_w  = t(blk(layer, "ssm_conv1d.weight"));
        auto* a_log_t = t(blk(layer, "ssm_a"));
        auto* dt_bias_t = t(blk(layer, "ssm_dt.bias"));
        auto* ssm_norm_w = t(blk(layer, "ssm_norm.weight"));
        auto* out_w   = t(blk(layer, "ssm_out.weight"));

        // Temp buffers (reuse attn_out which is large enough)
        half* norm_out = buf.norm_out;
        half* qkv_out = buf.attn_out;  // [qkv_dim]
        half* z_out = buf.mlp_gate;    // reuse, [num_v * v_dim]
        half* a_out = buf.mlp_up;      // reuse, [num_v]  
        half* b_out = buf.mlp_down;    // reuse, [num_v]

        // 1. RMSNorm (FP32 hidden in)
        if (norm_w->type == GGML_TYPE_F32)
            rms_norm_f32in_f32w(norm_out, hidden, (float*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);
        else
            rms_norm_f32in(norm_out, hidden, (half*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);

        // 2. Projections: QKV, Z(gate), alpha, beta — quantize once, reuse
        gpu_qi[g].quantize(norm_out, H, stream);
        quant_gemv(qkv_w->data, qkv_w->type, norm_out, qkv_out, H, qkv_dim, &gpu_qi[g], stream);
        quant_gemv(gate_w->data, gate_w->type, norm_out, z_out, H, num_v * v_dim, &gpu_qi[g], stream);
        quant_gemv(alpha_w->data, alpha_w->type, norm_out, a_out, H, num_v, &gpu_qi[g], stream);
        quant_gemv(beta_w->data, beta_w->type, norm_out, b_out, H, num_v, &gpu_qi[g], stream);

        // 3. Conv1d update (FP32)
        float* conv_out = gdn_bufs[g].conv_out;
        int kw = 4;
        int threads_conv = min(qkv_dim, 256);
        int blocks_conv = (qkv_dim + threads_conv - 1) / threads_conv;
        conv1d_update_silu<<<blocks_conv, threads_conv, 0, stream>>>(
            gdn_states[layer].conv_state,
            qkv_out,
            (float*)conv_w->data,  // conv weight is F32
            conv_out,
            qkv_dim, kw
        );

        // 4. Recurrent GDN step
        half* core_out = gdn_bufs[g].core_out;
        // shared mem: sQ[k_dim+1] + sK[k_dim] floats (last sQ slot stores attn_score)
        int gdn_smem = (2 * cfg.linear_k_dim + 1) * sizeof(float);
        gdn_recurrent_step<<<num_v, min(v_dim, 128), gdn_smem, stream>>>(
            conv_out,
            (float*)a_log_t->data,
            (float*)dt_bias_t->data,
            a_out,
            b_out,
            gdn_states[layer].rec_state,
            core_out,
            num_k, num_v, k_dim, v_dim
        );

        // 5. RMSNorm Gated
        half* normed_out = gdn_bufs[g].normed_out;
        rms_norm_gated_kernel<<<num_v, min(v_dim, 128), 128 * sizeof(float), stream>>>(
            core_out, z_out, (float*)ssm_norm_w->data, normed_out,
            num_v, v_dim, 1e-6f
        );

        // 6. Output projection
        half* proj_out = gdn_bufs[g].proj_out;
        gpu_qi_inter[g].quantize(normed_out, num_v * v_dim, stream);
        quant_gemv(out_w->data, out_w->type, normed_out, proj_out, num_v * v_dim, H, &gpu_qi_inter[g], stream);

        // 7. Residual add into FP32 hidden
        add_kernel_f32<<<(H+255)/256, 256, 0, stream>>>(hidden, proj_out, H);
    }

    // ============ Chunked GDN forward (process N tokens together) ============
    // hidden_chunk: [n_tokens, H] FP32 — read & updated in-place
    void forward_gdn_chunk(int layer, float* hidden_chunk, int n_tokens, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& buf = bufs[g];
        auto& gb = gdn_bufs[g];
        int H = cfg.hidden_size;
        int num_k = cfg.linear_k_heads;
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        int qkv_dim = 2 * num_k * k_dim + num_v * v_dim;
        int v_total = num_v * v_dim;

        auto* norm_w  = t(blk(layer, "attn_norm.weight"));
        auto* qkv_w   = t(blk(layer, "attn_qkv.weight"));
        auto* gate_w  = t(blk(layer, "attn_gate.weight"));
        auto* alpha_w = t(blk(layer, "ssm_alpha.weight"));
        auto* beta_w  = t(blk(layer, "ssm_beta.weight"));
        auto* conv_w  = t(blk(layer, "ssm_conv1d.weight"));
        auto* a_log_t = t(blk(layer, "ssm_a"));
        auto* dt_bias_t = t(blk(layer, "ssm_dt.bias"));
        auto* ssm_norm_w = t(blk(layer, "ssm_norm.weight"));
        auto* out_w   = t(blk(layer, "ssm_out.weight"));

        // ============ Batched RMSNorm + Q/K/V/Z/α/β projection ============
        // All four projections (qkv, gate, alpha, beta) share the same
        // pre-quantized norm output. The N-token batched GEMV reuses each
        // weight word across n_tokens dp4a accumulators, which is the actual
        // prefill speedup vs the per-token loop.
        bool can_batch_gdn = (qkv_w->type   == GGML_TYPE_Q8_0)
                          && (gate_w->type  == GGML_TYPE_Q8_0)
                          && (alpha_w->type == GGML_TYPE_Q8_0)
                          && (beta_w->type  == GGML_TYPE_Q8_0);

        if (can_batch_gdn) {
            // 1. Batched RMSNorm: rows=n_tokens, write into chunk_norm_out
            if (norm_w->type == GGML_TYPE_F32) {
                rms_norm_f32in_f32w(gb.chunk_norm_out, hidden_chunk, (float*)norm_w->data,
                                    n_tokens, H, cfg.rms_norm_eps, stream);
            } else {
                rms_norm_f32in(gb.chunk_norm_out, hidden_chunk, (half*)norm_w->data,
                               n_tokens, H, cfg.rms_norm_eps, stream);
            }

            // 2. Quantize all n_tokens × H normed values once
            gpu_qi[g].quantize_chunk(gb.chunk_norm_out, H, n_tokens, stream);

            // 3. Batched QKV projection. Output is [n_tokens, qkv_dim] half,
            //    we write into buf.attn_out as scratch and immediately
            //    half→float into chunk_qkv. Need scratch sized n_tokens*qkv_dim.
            //    buf.attn_out is sized max(H*4, num_q*hd*2) ≥ 12288, which
            //    is < n_tokens*qkv_dim for chunks > 1. We instead reuse
            //    chunk_z_out as fp16 staging since it's [CHUNK*v_total]
            //    (v_total=6144) — that's only 6144 elem per token but qkv_dim
            //    is 10240. So allocate a dedicated qkv_chunk buffer instead.
            //    For now: write directly to chunk_qkv as fp16 then convert.
            //    Actually the simplest: do batched GEMV writing into a fp16
            //    chunk staging, then half→float.
            //
            //    Simpler still: write the batched GEMV output into the FRONT
            //    of chunk_qkv (which is fp32 [CHUNK*qkv_dim]). We need fp16
            //    staging though, because gemv_q8_0_q8_nN writes half outputs.
            //    Use chunk_proj_out as fp16 staging (it's CHUNK*H half = at
            //    least CHUNK*5120; for 27B qkv_dim=10240 that's not enough).
            //
            //    Allocate a dedicated fp16 staging if not already there.
            if (!gb.chunk_qkv_half) {
                cudaMalloc(&gb.chunk_qkv_half, (size_t)CHUNK_SIZE * qkv_dim * sizeof(half));
            }

            quant_gemv_chunk(qkv_w->data, qkv_w->type,
                             gpu_qi[g].q8_buf, gb.chunk_qkv_half,
                             H, qkv_dim, n_tokens, stream);

            // Convert chunk_qkv_half [n_tokens, qkv_dim] → chunk_qkv [n_tokens, qkv_dim] fp32
            {
                int n_elem = n_tokens * qkv_dim;
                half_to_float_kernel<<<(n_elem + 255) / 256, 256, 0, stream>>>(
                    gb.chunk_qkv_half, gb.chunk_qkv, n_elem);
            }

            // Z (gate) projection — direct fp16 output to chunk_z_out
            quant_gemv_chunk(gate_w->data, gate_w->type,
                             gpu_qi[g].q8_buf, gb.chunk_z_out,
                             H, num_v * v_dim, n_tokens, stream);

            // alpha, beta projections — direct fp16 outputs
            quant_gemv_chunk(alpha_w->data, alpha_w->type,
                             gpu_qi[g].q8_buf, gb.chunk_a_proj,
                             H, num_v, n_tokens, stream);
            quant_gemv_chunk(beta_w->data, beta_w->type,
                             gpu_qi[g].q8_buf, gb.chunk_b_proj,
                             H, num_v, n_tokens, stream);
        } else {
            // Fallback: original per-token path for non-Q8_0 quantizations.
            for (int t = 0; t < n_tokens; t++) {
                float* h_t = hidden_chunk + (size_t)t * H;
                half* nrm = gb.chunk_norm_out + (size_t)t * H;

                if (norm_w->type == GGML_TYPE_F32)
                    rms_norm_f32in_f32w(nrm, h_t, (float*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);
                else
                    rms_norm_f32in(nrm, h_t, (half*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);

                gpu_qi[g].quantize(nrm, H, stream);
                quant_gemv(qkv_w->data, qkv_w->type, nrm, buf.attn_out, H, qkv_dim, &gpu_qi[g], stream);
                half_to_float_kernel<<<(qkv_dim+255)/256, 256, 0, stream>>>(
                    buf.attn_out, gb.chunk_qkv + (size_t)t * qkv_dim, qkv_dim);
                quant_gemv(gate_w->data, gate_w->type, nrm,
                           gb.chunk_z_out + (size_t)t * v_total, H, num_v * v_dim, &gpu_qi[g], stream);
                quant_gemv(alpha_w->data, alpha_w->type, nrm,
                           gb.chunk_a_proj + (size_t)t * num_v, H, num_v, &gpu_qi[g], stream);
                quant_gemv(beta_w->data, beta_w->type, nrm,
                           gb.chunk_b_proj + (size_t)t * num_v, H, num_v, &gpu_qi[g], stream);
            }
        }

        // Conv1d update — single chunked launch instead of n_tokens
        // per-token launches. conv1d_update_silu_chunk walks the n_tokens
        // dimension inside one block per dim element, sharing the conv
        // state register accumulator across the whole sub-sequence.
        //
        // Input format: needs fp16 [n_tokens, qkv_dim]. We have fp32
        // [n_tokens, qkv_dim] in gb.chunk_qkv from the projection above
        // (the batched path) or from the fallback. Convert in one shot
        // into chunk_qkv_half (already alloc'd by the batched projection
        // path; if the fallback was used, allocate it on demand).
        {
            int kw = 4;
            if (!gb.chunk_qkv_half) {
                cudaMalloc(&gb.chunk_qkv_half, (size_t)CHUNK_SIZE * qkv_dim * sizeof(half));
            }
            int n_elem = n_tokens * qkv_dim;
            float_to_half_kernel<<<(n_elem + 255) / 256, 256, 0, stream>>>(
                gb.chunk_qkv, gb.chunk_qkv_half, n_elem);

            int threads = min(qkv_dim, 256);
            int blocks  = (qkv_dim + threads - 1) / threads;
            conv1d_update_silu_chunk<<<blocks, threads, 0, stream>>>(
                gdn_states[layer].conv_state, gb.chunk_qkv_half,
                (float*)conv_w->data, gb.chunk_qkv, qkv_dim, kw, n_tokens);
        }

        // Chunked GDN recurrent step (single kernel call for all N tokens)
        {
            int gdn_smem = (2 * k_dim + 1) * sizeof(float);
            gdn_chunk_step<<<num_v, min(v_dim, 128), gdn_smem, stream>>>(
                gb.chunk_qkv,
                (float*)a_log_t->data,
                (float*)dt_bias_t->data,
                gb.chunk_a_proj,
                gb.chunk_b_proj,
                gdn_states[layer].rec_state,
                gb.chunk_core_out,
                n_tokens, num_k, num_v, k_dim, v_dim
            );
        }

        // RMSNorm gated (chunked) + batched output projection + batched residual
        {
            // Chunked rms_norm_gated → gb.chunk_normed [n_tokens * v_total] half
            int total_blocks = num_v * n_tokens;
            rms_norm_gated_chunk_kernel<<<total_blocks, min(v_dim, 128), 128*sizeof(float), stream>>>(
                gb.chunk_core_out, gb.chunk_z_out, (float*)ssm_norm_w->data,
                gb.chunk_normed, num_v, v_dim, n_tokens, 1e-6f);

            if (out_w->type == GGML_TYPE_Q8_0) {
                // Batched output projection: quantize all n_tokens × v_total
                // values once, then a single chunked GEMM call for the
                // [v_total → H] projection. This was the largest leftover
                // per-token GEMV in the GDN path (out_w is the same size
                // class as MLP down_proj).
                gpu_qi_inter[g].quantize_chunk(gb.chunk_normed, num_v * v_dim, n_tokens, stream);
                quant_gemv_chunk(out_w->data, out_w->type,
                                 gpu_qi_inter[g].q8_buf, gb.chunk_proj_out,
                                 num_v * v_dim, H, n_tokens, stream);

                // Batched residual into FP32 hidden (one launch).
                int n_elem = n_tokens * H;
                add_kernel_f32<<<(n_elem + 255) / 256, 256, 0, stream>>>(
                    hidden_chunk, gb.chunk_proj_out, n_elem);
            } else {
                // Non-Q8_0 fallback: per-token (rare).
                for (int t = 0; t < n_tokens; t++) {
                    half* normed_t = gb.chunk_normed + (size_t)t * v_total;
                    half* proj_t   = gb.chunk_proj_out + (size_t)t * H;
                    gpu_qi_inter[g].quantize(normed_t, num_v * v_dim, stream);
                    quant_gemv(out_w->data, out_w->type, normed_t, proj_t,
                               num_v * v_dim, H, &gpu_qi_inter[g], stream);
                    add_kernel_f32<<<(H+255)/256, 256, 0, stream>>>(
                        hidden_chunk + (size_t)t * H, proj_t, H);
                }
            }
        }
    }

    // ============ Chunked attention forward (process N tokens) ============
    // For attention layers in prompt phase: process tokens sequentially through
    // forward_attn (KV cache requires sequential append). hidden_chunk is FP32.
    void forward_attn_chunk(int layer, float* hidden_chunk, int start_pos, int n_tokens, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& ab = attn_bufs[g];
        auto& buf = bufs[g];
        int H = cfg.hidden_size;
        int num_q  = cfg.num_q_heads;
        int num_kv = cfg.num_kv_heads;
        int hd     = cfg.head_dim;
        int kv_dim = num_kv * hd;
        int total_qg = num_q * hd;
        float eps = cfg.rms_norm_eps;
        float scale = 1.0f / sqrtf((float)hd);

        auto* norm_w   = t(blk(layer, "attn_norm.weight"));
        auto* q_w      = t(blk(layer, "attn_q.weight"));
        auto* k_w      = t(blk(layer, "attn_k.weight"));
        auto* v_w      = t(blk(layer, "attn_v.weight"));
        auto* q_norm_w = t(blk(layer, "attn_q_norm.weight"));
        auto* k_norm_w = t(blk(layer, "attn_k_norm.weight"));
        auto* o_w      = t(blk(layer, "attn_output.weight"));
        if (!norm_w || !q_w || !k_w || !v_w || !o_w) {
            for (int tt = 0; tt < n_tokens; tt++)
                forward_attn(layer, hidden_chunk + (size_t)tt * H, start_pos + tt, stream);
            return;
        }
        int q_out_dim = q_w->dims[1];

        bool can_batch_attn = (q_w->type == GGML_TYPE_Q8_0)
                           && (k_w->type == GGML_TYPE_Q8_0)
                           && (v_w->type == GGML_TYPE_Q8_0)
                           && (o_w->type == GGML_TYPE_Q8_0)
                           && !use_turboquant;  // TQ KV cache uses different score path
        if (!can_batch_attn) {
            for (int tt = 0; tt < n_tokens; tt++)
                forward_attn(layer, hidden_chunk + (size_t)tt * H, start_pos + tt, stream);
            return;
        }

        // Lazy alloc per-GPU chunked attention buffers.
        if (!ab.attn_chunk_q) {
            cudaMalloc(&ab.attn_chunk_q,        (size_t)CHUNK_SIZE * q_out_dim * sizeof(half));
            cudaMalloc(&ab.attn_chunk_k,        (size_t)CHUNK_SIZE * kv_dim    * sizeof(half));
            cudaMalloc(&ab.attn_chunk_v,        (size_t)CHUNK_SIZE * kv_dim    * sizeof(half));
            cudaMalloc(&ab.attn_chunk_q_post,   (size_t)CHUNK_SIZE * total_qg  * sizeof(half));
            cudaMalloc(&ab.attn_chunk_gate,     (size_t)CHUNK_SIZE * total_qg  * sizeof(half));
            cudaMalloc(&ab.attn_chunk_scores,   (size_t)CHUNK_SIZE * num_q * kv_max_seq * sizeof(float));
            cudaMalloc(&ab.attn_chunk_out,      (size_t)CHUNK_SIZE * total_qg  * sizeof(half));
            cudaMalloc(&ab.attn_chunk_oproj,    (size_t)CHUNK_SIZE * H         * sizeof(half));
        }

        // 1. Batched RMSNorm: rows=n_tokens, output → mlp_chunk_norm
        //    (mlp_chunk_norm is reused; MLP runs after attn and re-norms).
        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w(buf.mlp_chunk_norm, hidden_chunk, (float*)norm_w->data,
                                n_tokens, H, eps, stream);
        } else {
            rms_norm_f32in(buf.mlp_chunk_norm, hidden_chunk, (half*)norm_w->data,
                           n_tokens, H, eps, stream);
        }

        // 2. Quantize all n_tokens × H normed values once
        gpu_qi[g].quantize_chunk(buf.mlp_chunk_norm, H, n_tokens, stream);

        // 3. Batched Q/K/V projections — 3 GEMV calls instead of 3*n_tokens.
        quant_gemv_chunk(q_w->data, q_w->type, gpu_qi[g].q8_buf, ab.attn_chunk_q,
                         H, q_out_dim, n_tokens, stream);
        quant_gemv_chunk(k_w->data, k_w->type, gpu_qi[g].q8_buf, ab.attn_chunk_k,
                         H, kv_dim, n_tokens, stream);
        quant_gemv_chunk(v_w->data, v_w->type, gpu_qi[g].q8_buf, ab.attn_chunk_v,
                         H, kv_dim, n_tokens, stream);

        // 4. Per-token: deinterleave Q/gate, head-RMS, RoPE, KV cache append.
        //    These are small kernels — keeping them per-token in a single
        //    fused loop. KV cache append is collapsed into TWO contiguous
        //    cudaMemcpyAsync calls per chunk (chunk_n × kv_dim K and V),
        //    which is far cheaper than 2*n_tokens individual copies.
        int rope_dim = rope.rope_dim;
        int half_rope = rope_dim / 2;
        int tn = min(hd, 128);
        auto& kv = kv_caches[layer];
        for (int tt = 0; tt < n_tokens; tt++) {
            int pos = start_pos + tt;
            half* q_slice    = ab.attn_chunk_q      + (size_t)tt * q_out_dim;
            half* qpost      = ab.attn_chunk_q_post + (size_t)tt * total_qg;
            half* gate_slice = ab.attn_chunk_gate   + (size_t)tt * total_qg;
            half* k_slice    = ab.attn_chunk_k      + (size_t)tt * kv_dim;

            deinterleave_qg_kernel<<<(total_qg + 255) / 256, 256, 0, stream>>>(
                q_slice, qpost, gate_slice, num_q, hd);

            head_rms_norm_kernel<<<num_q, tn, tn * sizeof(float), stream>>>(
                qpost, (float*)q_norm_w->data, num_q, hd, eps);
            head_rms_norm_kernel<<<num_kv, tn, tn * sizeof(float), stream>>>(
                k_slice, (float*)k_norm_w->data, num_kv, hd, eps);

            float* sin_pos = rope.sin_table(g) + pos * half_rope;
            float* cos_pos = rope.cos_table(g) + pos * half_rope;
            apply_rope_kernel<<<(num_q  * half_rope + 255) / 256, 256, 0, stream>>>(
                qpost,  sin_pos, cos_pos, num_q,  hd, rope_dim);
            apply_rope_kernel<<<(num_kv * half_rope + 255) / 256, 256, 0, stream>>>(
                k_slice, sin_pos, cos_pos, num_kv, hd, rope_dim);
        }
        // Bulk KV cache append: contiguous slabs of n_tokens × kv_dim each
        // (attn_chunk_k / attn_chunk_v) into kv.k / kv.v at offset start_pos.
        {
            half* k_dst = kv.k + (size_t)start_pos * kv_dim;
            half* v_dst = kv.v + (size_t)start_pos * kv_dim;
            size_t bytes = (size_t)n_tokens * kv_dim * sizeof(half);
            cudaMemcpyAsync(k_dst, ab.attn_chunk_k, bytes, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(v_dst, ab.attn_chunk_v, bytes, cudaMemcpyDeviceToDevice, stream);
        }

        // 5–7. Chunked attention compute, processed in ATTN_NB-token
        //      sub-chunks so the value kernel's register-resident
        //      accumulator stays bounded (see ATTN_NB / acc[16][8]).
        //      `kv` already aliased above for the bulk K/V append.
        int sub_processed = 0;
        while (sub_processed < n_tokens) {
            int sub_n = std::min(ATTN_NB, n_tokens - sub_processed);
            int sub_start_pos = start_pos + sub_processed;
            int sub_seq_total = sub_start_pos + sub_n;

            // Pointers into the chunk-major Q-post / scores / out buffers,
            // offset by sub_processed tokens.
            half*  q_post_sub = ab.attn_chunk_q_post + (size_t)sub_processed * total_qg;
            float* scores_sub = ab.attn_chunk_scores + (size_t)sub_processed * num_q * kv_max_seq;
            half*  out_sub    = ab.attn_chunk_out    + (size_t)sub_processed * total_qg;

            // 5. Score: each (kv_head, pos) block loads K[pos] once and
            //    dots against sub_n × gqa Q lanes.
            {
                int smem_bytes = (hd + 8) * sizeof(float);
                dim3 sg(num_kv, sub_seq_total);
                attn_score_kernel_h_chunk<<<sg, hd, smem_bytes, stream>>>(
                    q_post_sub, kv.k, scores_sub,
                    num_q, num_kv, hd, sub_seq_total, sub_start_pos, sub_n, scale,
                    /*row_stride=*/kv_max_seq);
            }

            // 6. Softmax (one block per (t_idx, q_head) row in this sub-chunk).
            //    The chunked softmax assumes scores stride = (num_q * seq_len),
            //    which matches the per-sub layout because each sub_chunk's
            //    scores buffer is [sub_n][num_q][kv_max_seq] but only
            //    [..][..][0..sub_seq_total) is meaningful — softmax only
            //    touches the active range.
            //
            //    NOTE: chunked softmax indexes its row as
            //    scores + t_idx*gridDim.y*seq_len + q_head*seq_len.
            //    So each row is `kv_max_seq` apart in the per-sub layout
            //    (because we keep the full max-seq stride). Pass
            //    seq_len = kv_max_seq to honor that stride, with
            //    start_pos = sub_start_pos so causal range stays correct.
            {
                int st = 1;
                while (st < sub_seq_total && st < 256) st <<= 1;
                dim3 sm_grid(sub_n, num_q);
                softmax_kernel_chunk<<<sm_grid, st, st * sizeof(float), stream>>>(
                    scores_sub, num_q, kv_max_seq, sub_start_pos,
                    /*wipe_end=*/sub_seq_total);
            }

            // 7. Value multiply.
            {
                int threads = min(hd, 128);
                int d_blocks = (hd + threads - 1) / threads;
                dim3 vg(num_kv, d_blocks);
                attn_value_kernel_h_chunk<<<vg, threads, 0, stream>>>(
                    scores_sub, kv.v, out_sub,
                    num_q, num_kv, hd, sub_seq_total, sub_start_pos, sub_n,
                    /*row_stride=*/kv_max_seq);
            }

            sub_processed += sub_n;
        }
        int seq_len_total = start_pos + n_tokens;  // for downstream code (unused)
        (void)seq_len_total;

        // 8. Per-token output gate (sigmoid) + 9. batched output projection.
        for (int tt = 0; tt < n_tokens; tt++) {
            half* out_slice  = ab.attn_chunk_out  + (size_t)tt * total_qg;
            half* gate_slice = ab.attn_chunk_gate + (size_t)tt * total_qg;
            apply_gate_sigmoid<<<(total_qg + 255) / 256, 256, 0, stream>>>(
                out_slice, gate_slice, total_qg);
        }

        // 9. Batched output projection: quantize attn_chunk_out → q8_1 chunk,
        //    then chunked GEMV for o_proj.
        gpu_qi_inter[g].quantize_chunk(ab.attn_chunk_out, total_qg, n_tokens, stream);
        quant_gemv_chunk(o_w->data, o_w->type,
                         gpu_qi_inter[g].q8_buf, ab.attn_chunk_oproj,
                         total_qg, H, n_tokens, stream);

        // 10. Residual add into the fp32 hidden chunk (one launch).
        {
            int n_elem = n_tokens * H;
            add_kernel_f32<<<(n_elem + 255) / 256, 256, 0, stream>>>(
                hidden_chunk, ab.attn_chunk_oproj, n_elem);
        }
    }

    // ============ Chunked MLP forward (batched, prefill fast path) ============
    // Processes n_tokens hidden states through one MLP layer with shared
    // weight loads. Each weight word in ffn_gate/ffn_up/ffn_down is read
    // ONCE per output row and dp4a'd against up to NB token inputs at the
    // same time (NB=16/8/4 peeling inside quant_gemv_chunk). For Q8_0 27B
    // this collapses 64 sequential GEMVs into ~4 batched calls per
    // projection — same total compute, ~1/16 the memory traffic, which is
    // the actual prefill bottleneck on HBM2-bound Volta.
    void forward_mlp_chunk(int layer, float* hidden_chunk, int n_tokens, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& buf = bufs[g];
        int H = cfg.hidden_size;
        int I = cfg.intermediate_size;

        auto* norm_w = t(blk(layer, "post_attention_norm.weight"));
        auto* gate_w = t(blk(layer, "ffn_gate.weight"));
        auto* up_w   = t(blk(layer, "ffn_up.weight"));
        auto* down_w = t(blk(layer, "ffn_down.weight"));
        if (!norm_w || !gate_w || !up_w || !down_w) {
            printf("MLP L%d: missing weights!\n", layer);
            return;
        }

        // Q8_0 path uses the new chunked dispatch. Q5_K / Q6_K and any
        // mixed quant fall back to the old per-token loop — those models
        // don't use the chunked GEMV anyway, and per-token still works.
        bool can_batch = (gate_w->type == GGML_TYPE_Q8_0)
                      && (up_w->type   == GGML_TYPE_Q8_0)
                      && (down_w->type == GGML_TYPE_Q8_0);
        if (!can_batch) {
            for (int tt = 0; tt < n_tokens; tt++) {
                forward_mlp(layer, hidden_chunk + (size_t)tt * H, stream);
            }
            return;
        }

        // 1. Batched RMSNorm: rows = n_tokens, all writing into mlp_chunk_norm.
        //    rms_norm_f32in_*_kernel uses one block per row, so this is one
        //    launch with n_tokens blocks instead of n_tokens launches.
        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w(buf.mlp_chunk_norm, hidden_chunk, (float*)norm_w->data,
                                n_tokens, H, cfg.rms_norm_eps, stream);
        } else {
            rms_norm_f32in(buf.mlp_chunk_norm, hidden_chunk, (half*)norm_w->data,
                           n_tokens, H, cfg.rms_norm_eps, stream);
        }

        // 2. Quantize all n_tokens × H normed values in one shot. The chunk
        //    is contiguous and H is a multiple of QK8=32, so block boundaries
        //    align with token boundaries.
        gpu_qi[g].quantize_chunk(buf.mlp_chunk_norm, H, n_tokens, stream);

        // 3. Batched ffn_gate and ffn_up projections. Both share the same
        //    pre-quantized input chunk → weight word load is paid twice
        //    (once per matrix), not 2*n_tokens times.
        quant_gemv_chunk(gate_w->data, gate_w->type,
                         gpu_qi[g].q8_buf, buf.mlp_chunk_gate,
                         H, I, n_tokens, stream);
        quant_gemv_chunk(up_w->data,   up_w->type,
                         gpu_qi[g].q8_buf, buf.mlp_chunk_up,
                         H, I, n_tokens, stream);

        // 4. Fused SiLU(gate) * up across the whole chunk. silu_mul_kernel
        //    is element-wise so we just expand n to (n_tokens * I).
        {
            int n_elem = n_tokens * I;
            silu_mul_kernel<<<(n_elem + 255) / 256, 256, 0, stream>>>(
                buf.mlp_chunk_gate, buf.mlp_chunk_up, n_elem);
        }

        // 5. Quantize the (silu_mul'd) intermediate chunk for the down proj.
        gpu_qi_inter[g].quantize_chunk(buf.mlp_chunk_gate, I, n_tokens, stream);

        // 6. Batched ffn_down projection.
        quant_gemv_chunk(down_w->data, down_w->type,
                         gpu_qi_inter[g].q8_buf, buf.mlp_chunk_down,
                         I, H, n_tokens, stream);

        // 7. Residual add: hidden_chunk[t,h] += mlp_chunk_down[t,h]. Both
        //    buffers are contiguous and the same shape, so add_kernel_f32
        //    handles the whole thing in one launch.
        {
            int n_elem = n_tokens * H;
            add_kernel_f32<<<(n_elem + 255) / 256, 256, 0, stream>>>(
                hidden_chunk, buf.mlp_chunk_down, n_elem);
        }
    }

#if 0  // unused — kept for reference, signatures need updating to fp32 hidden
    // ============ Full Pipeline Forward (1 token) ============
    void forward_one_token(half* hidden, int pos, int gpu_start = 0) {
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            int g = gpu->layer_gpu[layer];
            
            // Transfer hidden to correct GPU if needed
            int prev_g = (layer == 0) ? gpu_start : gpu->layer_gpu[layer - 1];
            if (g != prev_g) {
                half* new_hidden;
                cudaSetDevice(g);
                cudaMalloc(&new_hidden, cfg.hidden_size * sizeof(half));
                cudaMemcpy(new_hidden, hidden, cfg.hidden_size * sizeof(half), cudaMemcpyDeviceToDevice);
                if (layer > 0) {
                    cudaSetDevice(prev_g);
                    // Don't free — reuse buffer. Actually we need persistent buffers.
                }
                hidden = new_hidden;
                // TODO: use persistent transfer buffers
            }
            
            cudaSetDevice(g);
            cudaStream_t stream = 0;  // default stream for simplicity
            
            if (is_attn_layer(layer)) {
                // Attention + MLP
                forward_attn(layer, hidden, pos, stream);
            } else {
                // GDN + MLP
                forward_gdn(layer, hidden, stream);
            }
            
            // MLP (shared by both layer types)
            forward_mlp(layer, hidden, stream);
        }
    }

    // Full pipeline benchmark
    void benchmark_pipeline() {
        int H = cfg.hidden_size;
        
        // Allocate persistent hidden state on GPU 0
        cudaSetDevice(0);
        half* hidden;
        cudaMalloc(&hidden, H * sizeof(half));
        
        // Fill with test data
        std::vector<half> h_data(H);
        for (int i = 0; i < H; i++) h_data[i] = __float2half(0.01f * (i % 100 - 50));
        cudaMemcpy(hidden, h_data.data(), H * sizeof(half), cudaMemcpyHostToDevice);
        
        // Allocate transfer buffers on each GPU
        half* gpu_hidden[4];
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&gpu_hidden[g], H * sizeof(half));
        }
        
        // Reset GDN states
        reset_all_states();
        
        printf("\n=== Full Pipeline Benchmark ===\n");
        
        // Warmup: 2 tokens
        for (int tok = 0; tok < 2; tok++) {
            cudaMemcpy(gpu_hidden[0], hidden, H * sizeof(half), cudaMemcpyDeviceToDevice);
            half* h = gpu_hidden[0];
            
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int g = gpu->layer_gpu[layer];
                int prev_g = (layer == 0) ? 0 : gpu->layer_gpu[layer - 1];
                
                if (g != prev_g) {
                    cudaSetDevice(prev_g);
                    cudaDeviceSynchronize();
                    cudaMemcpy(gpu_hidden[g], h, H * sizeof(half), cudaMemcpyDeviceToDevice);
                    h = gpu_hidden[g];
                }
                
                cudaSetDevice(g);
                if (is_attn_layer(layer))
                    forward_attn(layer, h, tok, 0);
                else
                    forward_gdn(layer, h, 0);
                forward_mlp(layer, h, 0);
            }
            for (int g = 0; g < gpu->num_gpus; g++) {
                cudaSetDevice(g);
                cudaDeviceSynchronize();
            }
        }
        
        // Benchmark: N tokens
        int N = 20;
        
        // Sync all GPUs
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaDeviceSynchronize();
        }
        
        auto t0 = std::chrono::high_resolution_clock::now();
        
        for (int tok = 0; tok < N; tok++) {
            cudaMemcpy(gpu_hidden[0], hidden, H * sizeof(half), cudaMemcpyDeviceToDevice);
            half* h = gpu_hidden[0];
            
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int g = gpu->layer_gpu[layer];
                int prev_g = (layer == 0) ? 0 : gpu->layer_gpu[layer - 1];
                
                if (g != prev_g) {
                    // Sync previous GPU before transfer
                    cudaSetDevice(prev_g);
                    cudaDeviceSynchronize();
                    cudaMemcpy(gpu_hidden[g], h, H * sizeof(half), cudaMemcpyDeviceToDevice);
                    h = gpu_hidden[g];
                }
                
                cudaSetDevice(g);
                if (is_attn_layer(layer))
                    forward_attn(layer, h, tok + 2, 0);
                else
                    forward_gdn(layer, h, 0);
                forward_mlp(layer, h, 0);
            }
            
            // Sync all GPUs
            for (int g = 0; g < gpu->num_gpus; g++) {
                cudaSetDevice(g);
                cudaDeviceSynchronize();
            }
        }
        
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_per_tok = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
        double tps = 1000.0 / ms_per_tok;
        
        // Print result on last GPU
        int last_g = gpu->layer_gpu[cfg.num_layers - 1];
        half host_buf[8];
        cudaMemcpy(host_buf, gpu_hidden[last_g], 8 * sizeof(half), cudaMemcpyDeviceToHost);
        
        printf("Pipeline: %.1f ms/token = %.1f t/s\n", ms_per_tok, tps);
        printf("Output[0:4]: %.4f %.4f %.4f %.4f\n",
            __half2float(host_buf[0]), __half2float(host_buf[1]),
            __half2float(host_buf[2]), __half2float(host_buf[3]));
        
        // Breakdown estimate
        printf("\nBreakdown (from individual benchmarks):\n");
        printf("  GDN: ~%.0f us x 48 = %.1f ms\n", 330.0, 330.0 * 48 / 1000);
        printf("  Attn: ~%.0f us x 16 = %.1f ms\n", 335.0, 335.0 * 16 / 1000);
        printf("  MLP: ~%.0f us x 64 = %.1f ms\n", 524.0, 524.0 * 64 / 1000);
        printf("  Sum: ~%.1f ms (vs actual pipeline)\n", 
            (330.0*48 + 335.0*16 + 524.0*64) / 1000);
        
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaFree(gpu_hidden[g]);
        }
    }
#endif

    // test_gdn / test_mlp removed — needed updating to fp32 hidden

    // ============ N=2 sequential wrappers for spec decoding ============
    // forward_attn / forward_gdn are inherently sequential because of KV cache
    // append + GDN recurrent state. We just call them twice with the two
    // positions / hidden states. The "speedup" comes entirely from MLP
    // batching (forward_mlp_n2) which shares weight loads across both tokens.

    // Batched attention forward for two tokens. The 4 projections (q, k, v,
    // o) are run via quant_gemv_n2 sharing weight loads. The per-token bits
    // (deinterleave, head norm, RoPE, KV append, Q@K, softmax, @V, gate
    // sigmoid) are inherently sequential per-position.
    void forward_attn_n2(int layer, float* hidden_a, float* hidden_b,
                         int pos_a, int pos_b, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        int H = cfg.hidden_size;
        int num_q = cfg.num_q_heads;
        int num_kv = cfg.num_kv_heads;
        int hd = cfg.head_dim;
        float eps = cfg.rms_norm_eps;
        float scale = 1.0f / sqrtf((float)hd);
        auto& abA = attn_bufs[g];   auto& abB = attn_bufs2[g];
        auto& bA  = bufs[g];        auto& bB  = bufs2[g];

        auto* norm_w   = t(blk(layer, "attn_norm.weight"));
        auto* q_w      = t(blk(layer, "attn_q.weight"));
        auto* k_w      = t(blk(layer, "attn_k.weight"));
        auto* v_w      = t(blk(layer, "attn_v.weight"));
        auto* q_norm_w = t(blk(layer, "attn_q_norm.weight"));
        auto* k_norm_w = t(blk(layer, "attn_k_norm.weight"));
        auto* o_w      = t(blk(layer, "attn_output.weight"));

        int total_qg = num_q * hd;
        int kv_dim   = num_kv * hd;
        int q_out_dim = q_w->dims[1];

        // 1. RMSNorm both tokens (fused n2)
        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w_n2(bA.norm_out, bB.norm_out,
                                   hidden_a, hidden_b,
                                   (float*)norm_w->data, H, eps, stream);
        } else {
            rms_norm_f32in_n2(bA.norm_out, bB.norm_out,
                              hidden_a, hidden_b,
                              (half*)norm_w->data, H, eps, stream);
        }

        // 2. Quantize both norm outputs
        gpu_qi[g].quantize(bA.norm_out, H, stream);
        gpu_qi2[g].quantize(bB.norm_out, H, stream);

        // 3. Batched Q / K / V projections
        quant_gemv_n2(q_w->data, q_w->type,
                      bA.norm_out, bB.norm_out,
                      abA.q_proj, abB.q_proj,
                      H, q_out_dim, &gpu_qi[g], &gpu_qi2[g], stream);
        quant_gemv_n2(k_w->data, k_w->type,
                      bA.norm_out, bB.norm_out,
                      abA.k_proj, abB.k_proj,
                      H, kv_dim, &gpu_qi[g], &gpu_qi2[g], stream);
        quant_gemv_n2(v_w->data, v_w->type,
                      bA.norm_out, bB.norm_out,
                      abA.v_proj, abB.v_proj,
                      H, kv_dim, &gpu_qi[g], &gpu_qi2[g], stream);

        // 4. Per-token attention compute (deinterleave, norm, RoPE, KV
        //    append, Q@K, softmax, @V, gate sigmoid). Lambda to avoid
        //    duplicating ~30 lines.
        int rope_dim = rope.rope_dim;
        int half_rope = rope_dim / 2;
        auto run_one = [&](AttnBuffers& ab, int pos) {
            half* q_buf    = ab.attn_out;
            half* gate_buf = ab.gate_buf;
            deinterleave_qg_kernel<<<(total_qg+255)/256, 256, 0, stream>>>(
                ab.q_proj, q_buf, gate_buf, num_q, hd);
            int tn = min(hd, 128);
            head_rms_norm_kernel<<<num_q, tn, tn * sizeof(float), stream>>>(
                q_buf, (float*)q_norm_w->data, num_q, hd, eps);
            head_rms_norm_kernel<<<num_kv, tn, tn * sizeof(float), stream>>>(
                ab.k_proj, (float*)k_norm_w->data, num_kv, hd, eps);
            float* sin_pos = rope.sin_table(g) + (size_t)pos * half_rope;
            float* cos_pos = rope.cos_table(g) + (size_t)pos * half_rope;
            apply_rope_kernel<<<(num_q  * half_rope + 255)/256, 256, 0, stream>>>(
                q_buf, sin_pos, cos_pos, num_q, hd, rope_dim);
            apply_rope_kernel<<<(num_kv * half_rope + 255)/256, 256, 0, stream>>>(
                ab.k_proj, sin_pos, cos_pos, num_kv, hd, rope_dim);
            if (use_turboquant) {
                auto& tq = tq_kv_caches[layer];
                int bpt = tq.blocks_per_token;
                size_t off = (size_t)pos * bpt;
                tq3_quantize_kernel<<<(bpt+31)/32, 32, 0, stream>>>(ab.k_proj, &tq.k[off], bpt);
                tq3_quantize_kernel<<<(bpt+31)/32, 32, 0, stream>>>(ab.v_proj, &tq.v[off], bpt);
            } else {
                auto& kv = kv_caches[layer];
                half* k_cache_pos = kv.k + (size_t)pos * kv_dim;
                half* v_cache_pos = kv.v + (size_t)pos * kv_dim;
                cudaMemcpyAsync(k_cache_pos, ab.k_proj, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(v_cache_pos, ab.v_proj, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            }
            int seq_len = pos + 1;
            dim3 score_grid(num_q, seq_len);
            if (use_turboquant) {
                auto& tq = tq_kv_caches[layer];
                attn_score_kernel_tq3<<<score_grid, hd, hd * sizeof(float), stream>>>(
                    q_buf, tq.k, ab.attn_scores, num_q, num_kv, hd, seq_len, scale);
            } else {
                auto& kv = kv_caches[layer];
                attn_score_kernel_h<<<score_grid, min(hd, 256), 0, stream>>>(
                    q_buf, kv.k, ab.attn_scores, num_q, num_kv, hd, seq_len, scale);
            }
            { int st = 1; while(st < seq_len && st < 256) st <<= 1;
            softmax_kernel<<<num_q, st, st * sizeof(float), stream>>>(
                ab.attn_scores, num_q, seq_len); }
            if (use_turboquant) {
                auto& tq = tq_kv_caches[layer];
                int blocks_per_kv_head = hd / TQ_BLOCK_SIZE;
                dim3 vg(num_kv, blocks_per_kv_head);
                attn_value_kernel_tq3<<<vg, TQ_BLOCK_SIZE, TQ_BLOCK_SIZE * sizeof(float), stream>>>(
                    ab.attn_scores, tq.v, q_buf, num_q, num_kv, hd, seq_len);
            } else {
                auto& kv = kv_caches[layer];
                attn_value_kernel_h<<<num_q, min(hd, 256), 0, stream>>>(
                    ab.attn_scores, kv.v, q_buf, num_q, num_kv, hd, seq_len);
            }
            apply_gate_sigmoid<<<(total_qg+255)/256, 256, 0, stream>>>(
                q_buf, gate_buf, total_qg);
        };
        run_one(abA, pos_a);
        run_one(abB, pos_b);

        // 5. Batched output projection
        gpu_qi_inter[g].quantize(abA.attn_out,  num_q * hd, stream);
        gpu_qi_inter2[g].quantize(abB.attn_out, num_q * hd, stream);
        quant_gemv_n2(o_w->data, o_w->type,
                      abA.attn_out, abB.attn_out,
                      bA.mlp_down,  bB.mlp_down,
                      num_q * hd, H, &gpu_qi_inter[g], &gpu_qi_inter2[g], stream);

        // 6. Residual add into FP32 hidden (both tokens, fused n2)
        { dim3 ag((H+255)/256, 2);
          add_kernel_f32_n2<<<ag, 256, 0, stream>>>(
              hidden_a, bA.mlp_down,
              hidden_b, bB.mlp_down, H); }
    }

    // Batched GDN forward for two tokens. The 5 projections (qkv, gate,
    // alpha, beta, output) are run via quant_gemv_n2 sharing weight loads,
    // saving roughly half the GDN GEMV memory traffic. The recurrent path
    // (conv1d, gdn_recurrent_step, rms_norm_gated) is inherently per-token
    // and runs sequentially with a state snapshot taken between the two
    // tokens so a rejected second token can be rolled back exactly.
    void forward_gdn_n2(int layer, float* hidden_a, float* hidden_b, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        auto& bA = bufs[g];      auto& bB = bufs2[g];
        auto& gA = gdn_bufs[g];  auto& gB = gdn_bufs2[g];
        int H = cfg.hidden_size;
        int num_k = cfg.linear_k_heads;
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        int qkv_dim = 2 * num_k * k_dim + num_v * v_dim;
        int v_total = num_v * v_dim;

        auto* norm_w  = t(blk(layer, "attn_norm.weight"));
        auto* qkv_w   = t(blk(layer, "attn_qkv.weight"));
        auto* gate_w  = t(blk(layer, "attn_gate.weight"));
        auto* alpha_w = t(blk(layer, "ssm_alpha.weight"));
        auto* beta_w  = t(blk(layer, "ssm_beta.weight"));
        auto* conv_w  = t(blk(layer, "ssm_conv1d.weight"));
        auto* a_log_t = t(blk(layer, "ssm_a"));
        auto* dt_bias_t = t(blk(layer, "ssm_dt.bias"));
        auto* ssm_norm_w = t(blk(layer, "ssm_norm.weight"));
        auto* out_w   = t(blk(layer, "ssm_out.weight"));

        // 1. RMSNorm both tokens (fused n2)
        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w_n2(bA.norm_out, bB.norm_out,
                                   hidden_a, hidden_b,
                                   (float*)norm_w->data, H, cfg.rms_norm_eps, stream);
        } else {
            rms_norm_f32in_n2(bA.norm_out, bB.norm_out,
                              hidden_a, hidden_b,
                              (half*)norm_w->data, H, cfg.rms_norm_eps, stream);
        }

        // 2. Quantize both norm outputs once
        gpu_qi[g].quantize(bA.norm_out, H, stream);
        gpu_qi2[g].quantize(bB.norm_out, H, stream);

        // 3. Batched projections — qkv (largest), gate, alpha, beta
        // Per-token output buffers reuse the existing aliases used by forward_gdn:
        //   qkv_out → attn_out, z_out → mlp_gate, a_out → mlp_up, b_out → mlp_down
        quant_gemv_n2(qkv_w->data, qkv_w->type,
                      bA.norm_out, bB.norm_out,
                      bA.attn_out, bB.attn_out,
                      H, qkv_dim, &gpu_qi[g], &gpu_qi2[g], stream);
        quant_gemv_n2(gate_w->data, gate_w->type,
                      bA.norm_out, bB.norm_out,
                      bA.mlp_gate, bB.mlp_gate,
                      H, num_v * v_dim, &gpu_qi[g], &gpu_qi2[g], stream);
        quant_gemv_n2(alpha_w->data, alpha_w->type,
                      bA.norm_out, bB.norm_out,
                      bA.mlp_up, bB.mlp_up,
                      H, num_v, &gpu_qi[g], &gpu_qi2[g], stream);
        quant_gemv_n2(beta_w->data, beta_w->type,
                      bA.norm_out, bB.norm_out,
                      bA.mlp_down, bB.mlp_down,
                      H, num_v, &gpu_qi[g], &gpu_qi2[g], stream);

        // 4. Token A: conv1d + recurrent + gated norm  (advances state to post-a)
        int kw = 4;
        int threads_conv = min(qkv_dim, 256);
        int blocks_conv  = (qkv_dim + threads_conv - 1) / threads_conv;
        conv1d_update_silu<<<blocks_conv, threads_conv, 0, stream>>>(
            gdn_states[layer].conv_state, bA.attn_out, (float*)conv_w->data,
            gA.conv_out, qkv_dim, kw);
        int gdn_smem = (2 * cfg.linear_k_dim + 1) * sizeof(float);
        gdn_recurrent_step<<<num_v, min(v_dim, 128), gdn_smem, stream>>>(
            gA.conv_out, (float*)a_log_t->data, (float*)dt_bias_t->data,
            bA.mlp_up, bA.mlp_down,
            gdn_states[layer].rec_state, gA.core_out,
            num_k, num_v, k_dim, v_dim);
        rms_norm_gated_kernel<<<num_v, min(v_dim, 128), 128 * sizeof(float), stream>>>(
            gA.core_out, bA.mlp_gate, (float*)ssm_norm_w->data, gA.normed_out,
            num_v, v_dim, 1e-6f);

        // 5. Snapshot GDN state for THIS layer (post-a, for reject rollback)
        if (snapshots_ready) {
            size_t conv_sz = qkv_dim * 4 * sizeof(float);
            size_t rec_sz  = (size_t)num_v * k_dim * v_dim * sizeof(float);
            cudaMemcpyAsync(gdn_snapshots[layer].conv_state, gdn_states[layer].conv_state,
                            conv_sz, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(gdn_snapshots[layer].rec_state, gdn_states[layer].rec_state,
                            rec_sz, cudaMemcpyDeviceToDevice, stream);
        }

        // 6. Token B: conv1d + recurrent + gated norm  (advances state to post-b)
        conv1d_update_silu<<<blocks_conv, threads_conv, 0, stream>>>(
            gdn_states[layer].conv_state, bB.attn_out, (float*)conv_w->data,
            gB.conv_out, qkv_dim, kw);
        gdn_recurrent_step<<<num_v, min(v_dim, 128), gdn_smem, stream>>>(
            gB.conv_out, (float*)a_log_t->data, (float*)dt_bias_t->data,
            bB.mlp_up, bB.mlp_down,
            gdn_states[layer].rec_state, gB.core_out,
            num_k, num_v, k_dim, v_dim);
        rms_norm_gated_kernel<<<num_v, min(v_dim, 128), 128 * sizeof(float), stream>>>(
            gB.core_out, bB.mlp_gate, (float*)ssm_norm_w->data, gB.normed_out,
            num_v, v_dim, 1e-6f);

        // 7. Batched output projection — both normed_out → proj_out
        gpu_qi_inter[g].quantize(gA.normed_out,  num_v * v_dim, stream);
        gpu_qi_inter2[g].quantize(gB.normed_out, num_v * v_dim, stream);
        quant_gemv_n2(out_w->data, out_w->type,
                      gA.normed_out, gB.normed_out,
                      gA.proj_out,   gB.proj_out,
                      num_v * v_dim, H, &gpu_qi_inter[g], &gpu_qi_inter2[g], stream);

        // 8. Residual add into FP32 hidden (both tokens, fused n2)
        { dim3 ag((H+255)/256, 2);
          add_kernel_f32_n2<<<ag, 256, 0, stream>>>(
              hidden_a, gA.proj_out,
              hidden_b, gB.proj_out, H); }
    }

    // ============ GDN state snapshot/restore (for spec rollback) ============
    // When the MTP draft for the second token is rejected, GDN state has
    // already advanced past the (wrong) draft input. We need to roll back
    // to "after the first token only". Strategy: snapshot state BEFORE the
    // second token forward, run forward, then either commit (do nothing) or
    // restore. KV cache rollback is automatic — we just don't increment the
    // position counter and the next forward overwrites the rejected slot.

    struct GDNSnapshot {
        float* conv_state;  // backup
        float* rec_state;
    };
    std::vector<GDNSnapshot> gdn_snapshots;
    bool snapshots_ready = false;

    void alloc_gdn_snapshots() {
        if (snapshots_ready) return;
        auto* qkv = t("blk.0.attn_qkv.weight");
        if (!qkv) { printf("[SPEC] no qkv tensor for snapshot alloc\n"); return; }
        int qkv_dim = qkv->dims[1];
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        gdn_snapshots.resize(cfg.num_layers);
        size_t total = 0;
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (is_attn_layer(layer)) {
                gdn_snapshots[layer].conv_state = nullptr;
                gdn_snapshots[layer].rec_state = nullptr;
                continue;
            }
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            size_t conv_sz = qkv_dim * 4 * sizeof(float);
            size_t rec_sz  = (size_t)num_v * k_dim * v_dim * sizeof(float);
            cudaMalloc(&gdn_snapshots[layer].conv_state, conv_sz);
            cudaMalloc(&gdn_snapshots[layer].rec_state,  rec_sz);
            total += conv_sz + rec_sz;
        }
        snapshots_ready = true;
        printf("[SPEC] GDN snapshot buffers allocated (%.1f MB total)\n", total / 1e6);
    }

    void snapshot_gdn_states(cudaStream_t stream = 0) {
        auto* qkv = t("blk.0.attn_qkv.weight");
        int qkv_dim = qkv->dims[1];
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        size_t conv_sz = qkv_dim * 4 * sizeof(float);
        size_t rec_sz  = (size_t)num_v * k_dim * v_dim * sizeof(float);
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (!gdn_snapshots[layer].conv_state) continue;
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            cudaMemcpyAsync(gdn_snapshots[layer].conv_state, gdn_states[layer].conv_state,
                            conv_sz, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(gdn_snapshots[layer].rec_state, gdn_states[layer].rec_state,
                            rec_sz, cudaMemcpyDeviceToDevice, stream);
        }
    }

    void restore_gdn_states(cudaStream_t stream = 0) {
        auto* qkv = t("blk.0.attn_qkv.weight");
        int qkv_dim = qkv->dims[1];
        int num_v = cfg.linear_v_heads;
        int k_dim = cfg.linear_k_dim;
        int v_dim = cfg.linear_v_dim;
        size_t conv_sz = qkv_dim * 4 * sizeof(float);
        size_t rec_sz  = (size_t)num_v * k_dim * v_dim * sizeof(float);
        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (!gdn_snapshots[layer].conv_state) continue;
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            cudaMemcpyAsync(gdn_states[layer].conv_state, gdn_snapshots[layer].conv_state,
                            conv_sz, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(gdn_states[layer].rec_state, gdn_snapshots[layer].rec_state,
                            rec_sz, cudaMemcpyDeviceToDevice, stream);
        }
    }
};
