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
    float* norm_out;      // [hidden_size] FP32
    float* attn_out;      // [hidden_size] or larger FP32 (also used for QKV projection output)
    float* mlp_gate;      // [intermediate_size] FP32
    float* mlp_up;        // [intermediate_size] FP32
    float* mlp_down;      // [hidden_size] FP32
    float* residual;      // [hidden_size] FP32 (unused; kept for size compat)
};

struct QwenModel {
    QwenConfig cfg;
    GPUModel* gpu;
    LayerBuffers bufs[4];
    QuantInput gpu_qi[4];  // per-GPU reusable Q8 buffer (hidden_size)
    QuantInput gpu_qi_inter[4];  // per-GPU for intermediate_size  // one per GPU

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

    // Chunk size for parallel scan during prompt processing
    static constexpr int CHUNK_SIZE = 64;

    // GDN temp buffers per GPU (all FP32)
    struct GDNBuffers {
        float* conv_out;    // [qkv_dim] FP32
        float* core_out;    // [num_v * v_dim] FP32
        float* normed_out;  // [num_v * v_dim] FP32
        float* proj_out;    // [hidden_size] FP32

        // Chunk buffers (per-token data accumulated for chunked GDN)
        float* chunk_qkv;     // [CHUNK_SIZE * qkv_dim] FP32 conv1d outputs
        float* chunk_a_proj;  // [CHUNK_SIZE * num_v] FP32
        float* chunk_b_proj;  // [CHUNK_SIZE * num_v] FP32
        float* chunk_z_out;   // [CHUNK_SIZE * num_v * v_dim] FP32
        float* chunk_core_out;// [CHUNK_SIZE * num_v * v_dim] FP32
        float* chunk_normed;  // [CHUNK_SIZE * num_v * v_dim] FP32
        float* chunk_proj_out;// [CHUNK_SIZE * hidden_size] FP32
        float* chunk_norm_out;// [CHUNK_SIZE * hidden_size] FP32
    };
    GDNBuffers gdn_bufs[4];

    void alloc_buffers() {
        int H = cfg.hidden_size;
        int I = cfg.intermediate_size;
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&bufs[g].norm_out, H * sizeof(float));
            cudaMalloc(&bufs[g].attn_out, std::max(H * 4, cfg.num_q_heads * cfg.head_dim * 2) * sizeof(float));
            cudaMalloc(&bufs[g].mlp_gate, I * sizeof(float));
            cudaMalloc(&bufs[g].mlp_up, I * sizeof(float));
            cudaMalloc(&bufs[g].mlp_down, H * sizeof(float));
            cudaMalloc(&bufs[g].residual, H * sizeof(float));

            // GDN buffers (over-allocate for 27B max)
            int qkv_dim = 2 * 16 * 128 + 48 * 128;  // 10240
            int v_total = 48 * 128;  // 6144
            int num_v_max = 48;
            cudaMalloc(&gdn_bufs[g].conv_out, qkv_dim * sizeof(float));
            cudaMalloc(&gdn_bufs[g].core_out, v_total * sizeof(float));
            cudaMalloc(&gdn_bufs[g].normed_out, v_total * sizeof(float));
            cudaMalloc(&gdn_bufs[g].proj_out, H * sizeof(float));

            // Chunk buffers
            cudaMalloc(&gdn_bufs[g].chunk_qkv,      CHUNK_SIZE * qkv_dim * sizeof(float));
            cudaMalloc(&gdn_bufs[g].chunk_a_proj,   CHUNK_SIZE * num_v_max * sizeof(float));
            cudaMalloc(&gdn_bufs[g].chunk_b_proj,   CHUNK_SIZE * num_v_max * sizeof(float));
            cudaMalloc(&gdn_bufs[g].chunk_z_out,    CHUNK_SIZE * v_total * sizeof(float));
            cudaMalloc(&gdn_bufs[g].chunk_core_out, CHUNK_SIZE * v_total * sizeof(float));
            cudaMalloc(&gdn_bufs[g].chunk_normed,   CHUNK_SIZE * v_total * sizeof(float));
            cudaMalloc(&gdn_bufs[g].chunk_proj_out, CHUNK_SIZE * H * sizeof(float));
            cudaMalloc(&gdn_bufs[g].chunk_norm_out, CHUNK_SIZE * H * sizeof(float));
        }
    }

    // Get tensor helper
    GPUTensor* t(const std::string& name) { return gpu->get(name); }
    std::string blk(int layer, const std::string& suffix) {
        return "blk." + std::to_string(layer) + "." + suffix;
    }

    // MLP: gate_proj(SiLU) * up_proj -> down_proj  (all FP32)
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

        // RMSNorm: FP32 in -> FP32 out
        if (norm_w->type == GGML_TYPE_F32) {
            rms_norm_f32_full(buf.norm_out, hidden, (float*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);
        } else {
            rms_norm_f32_full_h_w(buf.norm_out, hidden, (half*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);
        }

        // gate / up / down all in FP32 path
        quant_gemv_f32(gate_w->data, gate_w->type, buf.norm_out, buf.mlp_gate, H, I, &gpu_qi[g], stream);
        quant_gemv_f32(up_w->data,   up_w->type,   buf.norm_out, buf.mlp_up,   H, I, &gpu_qi[g], stream);
        silu_mul_kernel_f32<<<(I+255)/256, 256, 0, stream>>>(buf.mlp_gate, buf.mlp_up, I);
        quant_gemv_f32(down_w->data, down_w->type, buf.mlp_gate, buf.mlp_down, I, H, &gpu_qi_inter[g], stream);

        // Residual add: hidden_fp32 += mlp_down_fp32
        add_kernel_f32_f32<<<(H+255)/256, 256, 0, stream>>>(hidden, buf.mlp_down, H);
    }



    // ============ Attention state ============
    RoPETable rope;
    TurboQuantCache tq_cache;
    int cur_seq_len = 0;
    
    // Attention temp buffers (per GPU) — all FP32
    struct AttnBuffers {
        float* q_proj;      // [num_q * head_dim * 2] (Q + gate concat)
        float* k_proj;      // [num_kv * head_dim]
        float* v_proj;      // [num_kv * head_dim]
        float* attn_scores; // [num_q * max_seq]
        float* gate_buf;    // [num_q * head_dim] for output gate
        float* attn_out;    // [num_q * head_dim]
    };
    AttnBuffers attn_bufs[4];

    void init_attention(int max_seq) {
        int num_q = cfg.num_q_heads;    // 24
        int num_kv = cfg.num_kv_heads;  // 4
        int hd = cfg.head_dim;          // 256
        float theta = 10000000.0f;

        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&attn_bufs[g].q_proj, num_q * hd * 2 * sizeof(float));
            cudaMalloc(&attn_bufs[g].k_proj, num_kv * hd * sizeof(float));
            cudaMalloc(&attn_bufs[g].v_proj, num_kv * hd * sizeof(float));
            cudaMalloc(&attn_bufs[g].attn_scores, num_q * max_seq * sizeof(float));
            cudaMalloc(&attn_bufs[g].attn_out, num_q * hd * sizeof(float));
            cudaMalloc(&attn_bufs[g].gate_buf, num_q * hd * sizeof(float));
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
    // ============ TurboQuant KV Cache (3-bit, ~10x compression) ============
    // Compressed K/V stored as block_tq3. Dequantized on-the-fly into a
    // shared per-GPU dequant buffer for use by the attention kernels.
    struct KVCache {
        block_tq3* k_tq3;  // [max_seq * blocks_per_token]
        block_tq3* v_tq3;
    };
    std::unordered_map<int, KVCache> kv_caches;
    int kv_max_seq = 0;
    int kv_blocks_per_token = 0;  // num_kv * head_dim / TQ_BLOCK_SIZE

    // Per-GPU shared dequant buffers (one set per GPU, reused across layers)
    float* dequant_k_buf[4] = {nullptr, nullptr, nullptr, nullptr};
    float* dequant_v_buf[4] = {nullptr, nullptr, nullptr, nullptr};

    void init_kv_cache(int max_seq) {
        kv_max_seq = max_seq;
        int num_kv = cfg.num_kv_heads;
        int hd = cfg.head_dim;
        int kv_dim = num_kv * hd;

        // Verify kv_dim is multiple of TQ_BLOCK_SIZE (=128)
        if (kv_dim % TQ_BLOCK_SIZE != 0) {
            fprintf(stderr, "TQ KV cache: kv_dim=%d not multiple of %d\n", kv_dim, TQ_BLOCK_SIZE);
            return;
        }
        kv_blocks_per_token = kv_dim / TQ_BLOCK_SIZE;

        size_t tq_bytes_per_layer = (size_t)max_seq * kv_blocks_per_token * sizeof(block_tq3);
        size_t dequant_bytes = (size_t)max_seq * kv_dim * sizeof(float);

        for (int layer = 0; layer < cfg.num_layers; layer++) {
            if (!is_attn_layer(layer)) continue;
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            KVCache kv;
            cudaMalloc(&kv.k_tq3, tq_bytes_per_layer);
            cudaMalloc(&kv.v_tq3, tq_bytes_per_layer);
            cudaMemset(kv.k_tq3, 0, tq_bytes_per_layer);
            cudaMemset(kv.v_tq3, 0, tq_bytes_per_layer);
            kv_caches[layer] = kv;
        }

        // Allocate one shared dequant buffer per GPU (max_seq sized) +
        // initialize the constant random sign vector for randomized Hadamard
        // transform (must match between encode and decode).
        for (int g = 0; g < gpu->num_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&dequant_k_buf[g], dequant_bytes);
            cudaMalloc(&dequant_v_buf[g], dequant_bytes);
            tq3_init_signs(g);
        }

        size_t total_tq = (size_t)kv_caches.size() * tq_bytes_per_layer * 2;  // K+V
        size_t total_dequant = (size_t)gpu->num_gpus * dequant_bytes * 2;
        size_t fp32_equiv = (size_t)kv_caches.size() * dequant_bytes * 2;
        printf("KV cache (TurboQuant 3-bit): %d layers, %d max_seq, %.1f MB compressed (%.1fx vs FP32 %.1f MB) + %.1f MB shared dequant\n",
            (int)kv_caches.size(), max_seq,
            total_tq / 1e6,
            (double)fp32_equiv / total_tq,
            fp32_equiv / 1e6,
            total_dequant / 1e6);
    }

    void reset_kv_cache() {
        int num_kv = cfg.num_kv_heads;
        int hd = cfg.head_dim;
        size_t tq_bytes_per_layer = (size_t)kv_max_seq * kv_blocks_per_token * sizeof(block_tq3);
        for (auto& [layer, kv] : kv_caches) {
            int g = gpu->layer_gpu[layer];
            cudaSetDevice(g);
            cudaMemset(kv.k_tq3, 0, tq_bytes_per_layer);
            cudaMemset(kv.v_tq3, 0, tq_bytes_per_layer);
        }
    }

    void forward_attn(int layer, float* hidden, int pos, cudaStream_t stream) {
        int g = gpu->layer_gpu[layer];
        int H = cfg.hidden_size;
        int num_q = cfg.num_q_heads;
        int num_kv = cfg.num_kv_heads;
        int hd = cfg.head_dim;
        float eps = cfg.rms_norm_eps;
        float scale = 1.0f / sqrtf((float)hd);
        auto& ab = attn_bufs[g];
        auto& kv = kv_caches[layer];

        auto* norm_w = t(blk(layer, "attn_norm.weight"));
        auto* q_w = t(blk(layer, "attn_q.weight"));
        auto* k_w = t(blk(layer, "attn_k.weight"));
        auto* v_w = t(blk(layer, "attn_v.weight"));
        auto* q_norm_w = t(blk(layer, "attn_q_norm.weight"));
        auto* k_norm_w = t(blk(layer, "attn_k_norm.weight"));
        auto* o_w = t(blk(layer, "attn_output.weight"));

        float* norm_out = bufs[g].norm_out;
        int total_qg = num_q * hd;
        int kv_dim = num_kv * hd;
        int seq_len = pos + 1;

        // 1. RMSNorm: FP32 in -> FP32 out
        if (norm_w->type == GGML_TYPE_F32)
            rms_norm_f32_full(norm_out, hidden, (float*)norm_w->data, 1, H, eps, stream);
        else
            rms_norm_f32_full_h_w(norm_out, hidden, (half*)norm_w->data, 1, H, eps, stream);

        // 2. Q/K/V projections (FP32 path)
        int q_out_dim = q_w->dims[1];
        quant_gemv_f32(q_w->data, q_w->type, norm_out, ab.q_proj, H, q_out_dim, &gpu_qi[g], stream);
        quant_gemv_f32(k_w->data, k_w->type, norm_out, ab.k_proj, H, kv_dim, &gpu_qi[g], stream);
        quant_gemv_f32(v_w->data, v_w->type, norm_out, ab.v_proj, H, kv_dim, &gpu_qi[g], stream);

        // 3. Deinterleave Q and gate
        float* q_buf = ab.attn_out;
        float* gate_buf = ab.gate_buf;
        deinterleave_qg_kernel_f32<<<(total_qg+255)/256, 256, 0, stream>>>(
            ab.q_proj, q_buf, gate_buf, num_q, hd);

        // 4. Head RMSNorm (q_norm/k_norm weights are F32)
        int tn = min(hd, 128);
        head_rms_norm_kernel_f32<<<num_q, tn, tn * sizeof(float), stream>>>(
            q_buf, (float*)q_norm_w->data, num_q, hd, eps);
        head_rms_norm_kernel_f32<<<num_kv, tn, tn * sizeof(float), stream>>>(
            ab.k_proj, (float*)k_norm_w->data, num_kv, hd, eps);

        // 5. RoPE (FP32)
        int rope_dim = rope.rope_dim;
        int half_rope = rope_dim / 2;
        float* sin_pos = rope.sin_table(g) + pos * half_rope;
        float* cos_pos = rope.cos_table(g) + pos * half_rope;
        apply_rope_kernel_f32<<<(num_q * half_rope + 255)/256, 256, 0, stream>>>(
            q_buf, sin_pos, cos_pos, num_q, hd, rope_dim);
        apply_rope_kernel_f32<<<(num_kv * half_rope + 255)/256, 256, 0, stream>>>(
            ab.k_proj, sin_pos, cos_pos, num_kv, hd, rope_dim);

        // 6. TurboQuant store: quantize K, V (fp32) → tq3 at position pos
        size_t tq_offset = (size_t)pos * kv_blocks_per_token;
        tq3_quantize_kernel_f32<<<(kv_blocks_per_token+31)/32, 32, 0, stream>>>(
            ab.k_proj, kv.k_tq3 + tq_offset, kv_blocks_per_token);
        tq3_quantize_kernel_f32<<<(kv_blocks_per_token+31)/32, 32, 0, stream>>>(
            ab.v_proj, kv.v_tq3 + tq_offset, kv_blocks_per_token);

        // 6b. Dequantize entire history [0..pos] into the per-GPU shared buffer
        int total_blocks = seq_len * kv_blocks_per_token;
        tq3_dequantize_kernel_f32<<<(total_blocks+31)/32, 32, 0, stream>>>(
            kv.k_tq3, dequant_k_buf[g], total_blocks);
        tq3_dequantize_kernel_f32<<<(total_blocks+31)/32, 32, 0, stream>>>(
            kv.v_tq3, dequant_v_buf[g], total_blocks);

        // 7. Attention scores (FP32 Q, FP32 dequantized K)
        dim3 score_grid(num_q, seq_len);
        attn_score_kernel_f32<<<score_grid, min(hd, 256), 0, stream>>>(
            q_buf, dequant_k_buf[g], ab.attn_scores,
            num_q, num_kv, hd, seq_len, scale);

        // Softmax (already FP32)
        { int st = 1; while(st < seq_len && st < 256) st <<= 1;
        softmax_kernel<<<num_q, st, st * sizeof(float), stream>>>(
            ab.attn_scores, num_q, seq_len); }

        // Weighted sum of V (FP32 output, dequantized V)
        attn_value_kernel_f32<<<num_q, min(hd, 256), 0, stream>>>(
            ab.attn_scores, dequant_v_buf[g], q_buf,
            num_q, num_kv, hd, seq_len);

        // 8. Output gate: out *= sigmoid(gate)  (FP32)
        apply_gate_sigmoid_f32<<<(total_qg+255)/256, 256, 0, stream>>>(
            q_buf, gate_buf, total_qg);

        // 9. Output projection (FP32 path)
        float* proj_out = bufs[g].mlp_down;
        quant_gemv_f32(o_w->data, o_w->type, q_buf, proj_out, num_q * hd, H, &gpu_qi_inter[g], stream);

        // 10. Residual: hidden_fp32 += proj_out_fp32
        add_kernel_f32_f32<<<(H+255)/256, 256, 0, stream>>>(hidden, proj_out, H);
    }

    // ============ GDN Forward ============
    // conv1d state: [qkv_dim, kernel_width] per layer
    // recurrent state: [num_v_heads, k_head_dim, v_head_dim] per layer
    
    struct GDNState {
        float* conv_state;    // [qkv_dim, 4] FP32
        double* rec_state;    // [num_v_heads, k_dim, v_dim] FP64
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
            cudaMalloc(&gdn_states[layer].rec_state, num_v * k_dim * v_dim * sizeof(double));
            cudaMemset(gdn_states[layer].rec_state, 0, num_v * k_dim * v_dim * sizeof(double));
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
            cudaMemset(gdn_states[layer].rec_state, 0, num_v * k_dim * v_dim * sizeof(double));
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

        // Temp buffers (reuse attn_out which is large enough) — all FP32
        float* norm_out = buf.norm_out;
        float* qkv_out = buf.attn_out;  // [qkv_dim]
        float* z_out = buf.mlp_gate;    // reuse, [num_v * v_dim]
        float* a_out = buf.mlp_up;      // reuse, [num_v]
        float* b_out = buf.mlp_down;    // reuse, [num_v]

        // 1. RMSNorm: FP32 in -> FP32 out
        if (norm_w->type == GGML_TYPE_F32)
            rms_norm_f32_full(norm_out, hidden, (float*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);
        else
            rms_norm_f32_full_h_w(norm_out, hidden, (half*)norm_w->data, 1, H, cfg.rms_norm_eps, stream);

        // 2. Projections: QKV, Z(gate), alpha, beta (FP32 path)
        quant_gemv_f32(qkv_w->data, qkv_w->type, norm_out, qkv_out, H, qkv_dim, &gpu_qi[g], stream);
        quant_gemv_f32(gate_w->data, gate_w->type, norm_out, z_out, H, num_v * v_dim, &gpu_qi[g], stream);
        quant_gemv_f32(alpha_w->data, alpha_w->type, norm_out, a_out, H, num_v, &gpu_qi[g], stream);
        quant_gemv_f32(beta_w->data, beta_w->type, norm_out, b_out, H, num_v, &gpu_qi[g], stream);

        // 3. Conv1d update (FP32 input now)
        float* conv_out = gdn_bufs[g].conv_out;
        int kw = 4;
        int threads_conv = min(qkv_dim, 256);
        int blocks_conv = (qkv_dim + threads_conv - 1) / threads_conv;
        conv1d_update_silu_f32<<<blocks_conv, threads_conv, 0, stream>>>(
            gdn_states[layer].conv_state,
            qkv_out,
            (float*)conv_w->data,
            conv_out,
            qkv_dim, kw
        );

        // 4. Recurrent GDN step (FP32 alpha/beta input, FP32 output)
        float* core_out = gdn_bufs[g].core_out;
        int gdn_smem = (2 * cfg.linear_k_dim + 1) * sizeof(float);
        gdn_recurrent_step_f32<<<num_v, min(v_dim, 128), gdn_smem, stream>>>(
            conv_out,
            (float*)a_log_t->data,
            (float*)dt_bias_t->data,
            a_out,
            b_out,
            gdn_states[layer].rec_state,
            core_out,
            num_k, num_v, k_dim, v_dim
        );

        // 5. RMSNorm Gated (FP32 in/out)
        float* normed_out = gdn_bufs[g].normed_out;
        rms_norm_gated_kernel_f32<<<num_v, min(v_dim, 128), 128 * sizeof(float), stream>>>(
            core_out, z_out, (float*)ssm_norm_w->data, normed_out,
            num_v, v_dim, 1e-6f
        );

        // 6. Output projection (FP32)
        float* proj_out = gdn_bufs[g].proj_out;
        quant_gemv_f32(out_w->data, out_w->type, normed_out, proj_out, num_v * v_dim, H, &gpu_qi_inter[g], stream);

        // 7. Residual: hidden_fp32 += proj_out_fp32
        add_kernel_f32_f32<<<(H+255)/256, 256, 0, stream>>>(hidden, proj_out, H);

    }

    // ============ Chunked GDN forward (process N tokens together) ============
    // For now, just iterate per-token through forward_gdn (which is fully FP32).
    // The previous chunked-kernel approach used FP16 intermediates which lost
    // precision; per-token FP32 is more accurate at the cost of some speed.
    void forward_gdn_chunk(int layer, float* hidden_chunk, int n_tokens, cudaStream_t stream) {
        int H = cfg.hidden_size;
        for (int t = 0; t < n_tokens; t++) {
            forward_gdn(layer, hidden_chunk + (size_t)t * H, stream);
        }
    }

    // ============ Chunked attention forward (process N tokens) ============
    // For attention layers in prompt phase: process tokens sequentially through
    // forward_attn (KV cache requires sequential append). hidden_chunk is FP32.
    void forward_attn_chunk(int layer, float* hidden_chunk, int start_pos, int n_tokens, cudaStream_t stream) {
        int H = cfg.hidden_size;
        for (int t = 0; t < n_tokens; t++) {
            forward_attn(layer, hidden_chunk + (size_t)t * H, start_pos + t, stream);
        }
    }

    // ============ Chunked MLP forward ============
    void forward_mlp_chunk(int layer, float* hidden_chunk, int n_tokens, cudaStream_t stream) {
        int H = cfg.hidden_size;
        for (int t = 0; t < n_tokens; t++) {
            forward_mlp(layer, hidden_chunk + (size_t)t * H, stream);
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
};
