// MTP (Multi-Token Prediction) head — Phase 0 (forward + acceptance measurement)
//
// Loads the 15 MTP tensors from mtp_head.bin (produced by mtp_work/convert_mtp.py)
// onto last_gpu and runs the speculative draft pass:
//
//     e  = embed(prev_token)            // 5120 fp16
//     en = rmsnorm(e,  pre_fc_norm_embedding)
//     hn = rmsnorm(h_main_normed, pre_fc_norm_hidden)   // h_main is the
//                                                       // post-final-norm
//                                                       // hidden state from
//                                                       // the main model
//     fused = concat([en, hn])          // 10240
//     h     = fc @ fused                // 5120
//
//     // ── 1 standard transformer layer (full_attention) ──
//     h2 = rmsnorm(h, input_layernorm)
//     qg = q_proj @ h2                  // 12288 = q (6144) + gate (6144) interleaved per head
//     k  = k_proj @ h2                  // 1024
//     v  = v_proj @ h2                  // 1024
//     q, gate = deinterleave(qg, num_q=24, hd=256)
//     q  = head_rmsnorm(q, q_norm)
//     k  = head_rmsnorm(k, k_norm)
//     RoPE(q, position) ; RoPE(k, position)
//     append (k, v) to MTP's own KV cache at position
//     attn_out = attention(q, mtp_kv) ; attn_out *= sigmoid(gate)
//     h = h + o_proj @ attn_out
//
//     h3 = rmsnorm(h, post_attention_layernorm)
//     mlp_gate = gate_proj @ h3
//     mlp_up   = up_proj   @ h3
//     mlp_out  = down_proj @ (silu(mlp_gate) * mlp_up)
//     h = h + mlp_out
//
//     // ── final norm + shared lm_head ──
//     h_final = rmsnorm(h, mtp.norm)
//     logits  = lm_head @ h_final       // shared with main output.weight
//
// Phase 0 only loads + runs the head — speculative verify / KV rollback are
// Phase 1.

#pragma once
#include "quant_gemv.cuh"
#include "ops.cuh"
#include "attention.cuh"
#include "gpu_loader.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <cmath>

static inline void dump_half_stats(const char* tag, const half* dev, int n) {
    cudaDeviceSynchronize();
    std::vector<half> h(n);
    cudaMemcpy(h.data(), dev, n * sizeof(half), cudaMemcpyDeviceToHost);
    int nn = 0, ni = 0; double s = 0, sq = 0;
    for (int i = 0; i < n; i++) {
        float v = __half2float(h[i]);
        if (isnan(v)) nn++;
        if (isinf(v)) ni++;
        s += v; sq += (double)v * v;
    }
    double mean = s / n;
    double var  = sq / n - mean * mean;
    printf("[MTP-DBG] %-25s mean=%+.4f std=%.4f nan=%d inf=%d first6=[%+.4f %+.4f %+.4f %+.4f %+.4f %+.4f]\n",
           tag, mean, sqrt(var > 0 ? var : 0), nn, ni,
           __half2float(h[0]), __half2float(h[1]), __half2float(h[2]),
           __half2float(h[3]), __half2float(h[4]), __half2float(h[5]));
}

static inline void dump_f32_stats(const char* tag, const float* dev, int n) {
    cudaDeviceSynchronize();
    std::vector<float> h(n);
    cudaMemcpy(h.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost);
    int nn = 0, ni = 0; double s = 0, sq = 0;
    for (int i = 0; i < n; i++) {
        float v = h[i];
        if (isnan(v)) nn++;
        if (isinf(v)) ni++;
        s += v; sq += (double)v * v;
    }
    double mean = s / n;
    double var  = sq / n - mean * mean;
    printf("[MTP-DBG] %-25s mean=%+.4f std=%.4f nan=%d inf=%d first6=[%+.4f %+.4f %+.4f %+.4f %+.4f %+.4f]\n",
           tag, mean, sqrt(var > 0 ? var : 0), nn, ni,
           h[0], h[1], h[2], h[3], h[4], h[5]);
}

struct MTPHead {
    // ── Config (mirrors main model) ───────────────────────────────────────
    int H;            // hidden size = 5120
    int num_q;        // 24
    int num_kv;       // 4
    int head_dim;     // 256
    int q_dim;        // num_q * head_dim = 6144
    int kv_dim;       // num_kv * head_dim = 1024
    int qg_dim;       // 2 * q_dim = 12288 (q + gate concatenated per head)
    int I;            // intermediate = 17408
    int V;            // vocab = 248320
    int rope_dim;     // 64 (partial rotary)
    int max_seq;      // KV cache capacity
    float eps;        // 1e-6f
    int gpu_id;       // last_gpu

    // ── Weights (Q8_0 for big mats, F32 for norms) ───────────────────────
    // The big projections live in block_q8_0_aligned layout (post repack),
    // exactly like the main model's quantized weights, so quant_gemv works
    // with them as-is.
    void* fc;          // [H, 2H] Q8_0 — concat-fused projection
    void* q_proj;      // [H, qg_dim] Q8_0
    void* k_proj;      // [H, kv_dim] Q8_0
    void* v_proj;      // [H, kv_dim] Q8_0
    void* o_proj;      // [q_dim, H] Q8_0
    void* gate_proj;   // [H, I] Q8_0
    void* up_proj;     // [H, I] Q8_0
    void* down_proj;   // [I, H] Q8_0

    // Norms — converted from F16 → F32 at load so we can reuse the existing
    // F32-weight RMSNorm kernels (head_rms_norm_kernel etc.).
    float* pre_fc_norm_embedding;     // [H]
    float* pre_fc_norm_hidden;        // [H]
    float* input_layernorm;           // [H]
    float* post_attention_layernorm;  // [H]
    float* q_norm;                    // [head_dim]
    float* k_norm;                    // [head_dim]
    float* final_norm;                // [H]

    // ── Temp buffers (live across calls, reused) ─────────────────────────
    half* embed_tmp;       // [H]      embedding of prev_token
    half* en_tmp;          // [H]      pre_fc_norm_embedding(e)
    half* hn_tmp;          // [H]      pre_fc_norm_hidden(h_main)
    half* fused_tmp;       // [2H]     concat(en, hn)
    half* h_tmp;           // [H]      after fc
    half* h_norm_tmp;      // [H]      after input_layernorm / post_attn norm
    half* qg_tmp;          // [qg_dim] q_proj output
    half* q_tmp;           // [q_dim]
    half* gate_tmp;        // [q_dim]
    half* k_tmp;           // [kv_dim]
    half* v_tmp;           // [kv_dim]
    float* attn_scores;    // [num_q * max_seq]
    half* attn_out;        // [q_dim]
    half* attn_proj_out;   // [H]      o_proj output
    half* mlp_gate_tmp;    // [I]
    half* mlp_up_tmp;      // [I]
    half* mlp_out;         // [H]      down_proj output
    half* h_final;         // [H]      after mtp.norm
    half* logits_buf;      // [V]      MTP logits

    // ── KV cache ─────────────────────────────────────────────────────────
    half* kv_k;            // [max_seq * kv_dim]
    half* kv_v;            // [max_seq * kv_dim]
    int   kv_pos;          // next write position

    // Argmax scratch
    int* d_argmax;
    int* h_argmax;

    // ── Reused dispatch helpers ──────────────────────────────────────────
    QuantInput qi;         // for input quantization on H-sized inputs
    QuantInput qi_qdim;    // for q_dim sized inputs (o_proj input)
    QuantInput qi_inter;   // for I-sized inputs (down_proj input)
    QuantInput qi_2H;      // for 2H sized inputs (fc input)

    // External shared resources owned by the model (we just keep pointers)
    void* lm_head_data = nullptr;
    ggml_type lm_head_type = GGML_TYPE_F16;

    // For RoPE we reuse the main model's table (already on this GPU)
    const float* rope_sin = nullptr;   // [max_seq * (rope_dim/2)]
    const float* rope_cos = nullptr;

    // For embedding lookup of prev_token we point at the main model's
    // token_embd weight on its home GPU and pull just one row each call.
    void* embd_data = nullptr;
    ggml_type embd_type = GGML_TYPE_F16;
    int embd_gpu = 0;
    half* embed_pinned_host = nullptr;  // staging buffer for cross-GPU copy
    half* embed_src_dev = nullptr;      // dev buffer on embd_gpu (size H)

    // ── Loader ───────────────────────────────────────────────────────────
    // The mtp_head.bin format is described in mtp_work/convert_mtp.py.
    bool load(const char* path, int last_gpu, int hidden_size, int vocab_size,
              int num_q_heads, int num_kv_heads, int head_dim_,
              int intermediate, int rope_dim_, int max_seq_, float rms_eps) {
        gpu_id = last_gpu;
        H = hidden_size;
        V = vocab_size;
        num_q = num_q_heads;
        num_kv = num_kv_heads;
        head_dim = head_dim_;
        q_dim = num_q * head_dim;
        kv_dim = num_kv * head_dim;
        qg_dim = 2 * q_dim;
        I = intermediate;
        rope_dim = rope_dim_;
        max_seq = max_seq_;
        eps = rms_eps;
        kv_pos = 0;

        cudaSetDevice(gpu_id);

        int fd = open(path, O_RDONLY);
        if (fd < 0) { perror(path); return false; }
        struct stat st; fstat(fd, &st);
        size_t fsize = st.st_size;
        void* map = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (map == MAP_FAILED) { perror("mmap"); return false; }
        const uint8_t* base = (const uint8_t*)map;

        if (memcmp(base, "MTP1", 4) != 0) {
            fprintf(stderr, "MTP: bad magic in %s\n", path);
            munmap(map, fsize);
            return false;
        }
        uint32_t version = *(const uint32_t*)(base + 4);
        uint32_t n_tensors = *(const uint32_t*)(base + 8);
        printf("[MTP] loading %s: version=%u tensors=%u (%.1f MB)\n",
               path, version, n_tensors, fsize / 1e6);

        const uint8_t* p = base + 12;
        // Parse all tensor records and stage upload
        for (uint32_t i = 0; i < n_tensors; i++) {
            uint16_t name_len = *(const uint16_t*)p; p += 2;
            std::string name((const char*)p, name_len); p += name_len;
            uint8_t dtype = *p++;
            uint8_t n_dims = *p++;
            std::vector<uint64_t> dims(n_dims);
            memcpy(dims.data(), p, 8 * n_dims); p += 8 * n_dims;
            uint64_t off = *(const uint64_t*)p; p += 8;
            uint64_t size = *(const uint64_t*)p; p += 8;

            const uint8_t* blob = base + off;

            if (dtype == 0) {
                // F16 weight — used for the small norm vectors. Convert to
                // F32 on host then upload, so we can use the F32-weight
                // RMSNorm kernels.
                int n_elems = (int)(size / 2);
                std::vector<float> as_f32(n_elems);
                // Qwen3NextRMSNorm uses `output * (1 + weight)`. llama.cpp's
                // GGUF converter bakes the `+1` into stored norm weights, but
                // our mtp_head.bin holds RAW safetensors values, so apply +1
                // here. (See Qwen3NextModel.modify_tensors in convert_hf_to_gguf.py)
                for (int j = 0; j < n_elems; j++)
                    as_f32[j] = __half2float(((const __half*)blob)[j]) + 1.0f;
                float* dev = nullptr;
                cudaMalloc(&dev, n_elems * sizeof(float));
                cudaMemcpy(dev, as_f32.data(), n_elems * sizeof(float), cudaMemcpyHostToDevice);
                if (getenv("MTP_DEBUG_LOAD")) {
                    // Read back from device to verify upload integrity
                    std::vector<float> all(n_elems);
                    cudaMemcpy(all.data(), dev, n_elems * sizeof(float), cudaMemcpyDeviceToHost);
                    int n_nan = 0, n_inf = 0;
                    double sum = 0, sumsq = 0;
                    for (int j = 0; j < n_elems; j++) {
                        float v = all[j];
                        if (isnan(v)) n_nan++;
                        if (isinf(v)) n_inf++;
                        sum += v; sumsq += v * v;
                    }
                    double mean = sum / n_elems;
                    double var  = sumsq / n_elems - mean * mean;
                    printf("[MTP-LOAD] %-50s n=%5d mean=%+.4f std=%.4f nan=%d inf=%d first6=[%+.4f %+.4f %+.4f %+.4f %+.4f %+.4f]\n",
                           name.c_str(), n_elems, mean, sqrt(var), n_nan, n_inf,
                           all[0], all[1], all[2], all[3], all[4], all[5]);
                }
                bind_f32(name, dev);
            } else if (dtype == 1) {
                // Q8_0 — same path as the main model loader: copy the GGUF
                // {half d; int8_t qs[32];} blocks to the GPU then run the
                // existing repack to block_q8_0_aligned (qs at offset 0).
                int n_blocks = (int)(size / 34);
                size_t aligned_size = (size_t)n_blocks * 36;
                void* tmp = nullptr;
                cudaMalloc(&tmp, size);
                cudaMemcpy(tmp, blob, size, cudaMemcpyHostToDevice);
                void* dev = nullptr;
                cudaMalloc(&dev, aligned_size);
                int t = 256;
                int b = (n_blocks + t - 1) / t;
                q8_0_repack_kernel<<<b, t>>>(tmp, dev, n_blocks);
                cudaDeviceSynchronize();
                cudaFree(tmp);
                bind_q8_0(name, dev);
            } else {
                fprintf(stderr, "MTP: unknown dtype %d for %s\n", dtype, name.c_str());
            }
        }

        munmap(map, fsize);

        // Allocate temp + KV buffers
        cudaMalloc(&embed_tmp, H * sizeof(half));
        cudaMalloc(&en_tmp,    H * sizeof(half));
        cudaMalloc(&hn_tmp,    H * sizeof(half));
        cudaMalloc(&fused_tmp, 2 * H * sizeof(half));
        cudaMalloc(&h_tmp,      H * sizeof(half));
        cudaMalloc(&h_norm_tmp, H * sizeof(half));
        cudaMalloc(&qg_tmp,     qg_dim * sizeof(half));
        cudaMalloc(&q_tmp,      q_dim * sizeof(half));
        cudaMalloc(&gate_tmp,   q_dim * sizeof(half));
        cudaMalloc(&k_tmp,      kv_dim * sizeof(half));
        cudaMalloc(&v_tmp,      kv_dim * sizeof(half));
        cudaMalloc(&attn_scores, num_q * max_seq * sizeof(float));
        cudaMalloc(&attn_out,   q_dim * sizeof(half));
        cudaMalloc(&attn_proj_out, H * sizeof(half));
        cudaMalloc(&mlp_gate_tmp, I * sizeof(half));
        cudaMalloc(&mlp_up_tmp,   I * sizeof(half));
        cudaMalloc(&mlp_out,      H * sizeof(half));
        cudaMalloc(&h_final,      H * sizeof(half));
        cudaMalloc(&logits_buf,   V * sizeof(half));

        cudaMalloc(&kv_k, (size_t)max_seq * kv_dim * sizeof(half));
        cudaMalloc(&kv_v, (size_t)max_seq * kv_dim * sizeof(half));

        cudaMalloc(&d_argmax, sizeof(int));
        cudaMallocHost(&h_argmax, sizeof(int));

        // Pinned host staging for cross-GPU embedding row copy
        cudaMallocHost(&embed_pinned_host, H * sizeof(half));

        printf("[MTP] head loaded on GPU %d, KV cache %d slots = %.1f MB\n",
               gpu_id, max_seq, (2.0 * max_seq * kv_dim * 2) / 1e6);
        return true;
    }

    void bind_f32(const std::string& name, float* dev) {
        if      (name == "mtp.pre_fc_norm_embedding.weight")          pre_fc_norm_embedding = dev;
        else if (name == "mtp.pre_fc_norm_hidden.weight")             pre_fc_norm_hidden    = dev;
        else if (name == "mtp.layers.0.input_layernorm.weight")       input_layernorm       = dev;
        else if (name == "mtp.layers.0.post_attention_layernorm.weight") post_attention_layernorm = dev;
        else if (name == "mtp.layers.0.self_attn.q_norm.weight")      q_norm                = dev;
        else if (name == "mtp.layers.0.self_attn.k_norm.weight")      k_norm                = dev;
        else if (name == "mtp.norm.weight")                           final_norm            = dev;
        else fprintf(stderr, "MTP: unbound F32 tensor %s\n", name.c_str());
    }

    void bind_q8_0(const std::string& name, void* dev) {
        if      (name == "mtp.fc.weight")                              fc        = dev;
        else if (name == "mtp.layers.0.self_attn.q_proj.weight")       q_proj    = dev;
        else if (name == "mtp.layers.0.self_attn.k_proj.weight")       k_proj    = dev;
        else if (name == "mtp.layers.0.self_attn.v_proj.weight")       v_proj    = dev;
        else if (name == "mtp.layers.0.self_attn.o_proj.weight")       o_proj    = dev;
        else if (name == "mtp.layers.0.mlp.gate_proj.weight")          gate_proj = dev;
        else if (name == "mtp.layers.0.mlp.up_proj.weight")            up_proj   = dev;
        else if (name == "mtp.layers.0.mlp.down_proj.weight")          down_proj = dev;
        else fprintf(stderr, "MTP: unbound Q8_0 tensor %s\n", name.c_str());
    }

    // Hook the embedding source so forward() can pull a row by token id.
    void set_embed_source(void* embd_data_, ggml_type type, int gpu_for_embd) {
        embd_data = embd_data_;
        embd_type = type;
        embd_gpu  = gpu_for_embd;
        cudaSetDevice(embd_gpu);
        cudaMalloc(&embed_src_dev, H * sizeof(half));
        cudaSetDevice(gpu_id);
    }

    void set_lm_head(void* lm_w, ggml_type t) { lm_head_data = lm_w; lm_head_type = t; }

    void set_rope_tables(const float* sin_t, const float* cos_t) {
        rope_sin = sin_t;
        rope_cos = cos_t;
    }

    void reset_kv() { kv_pos = 0; }

    // Run the MTP head one position. Returns argmax(logits) of the predicted
    // next token. h_main_normed is the post-output_norm hidden state from the
    // main model at the SAME position; prev_token_id is the token currently at
    // that position (i.e. the token main just consumed as input).
    int forward(const half* h_main_normed, int prev_token_id, int position) {
        cudaSetDevice(gpu_id);
        cudaStream_t stream = 0;

        // DEBUG: skip the MTP head entirely and just run lm_head on the
        // (already-normed) main hidden state. Should reproduce main's argmax.
        if (getenv("MTP_LMHEAD_ONLY")) {
            qi.quantize(h_main_normed, H, stream);
            quant_gemv(lm_head_data, lm_head_type, (half*)h_main_normed, logits_buf, H, V, &qi, stream);
            argmax_half_kernel<<<1, 1024, 0, stream>>>(logits_buf, V, d_argmax);
            cudaMemcpy(h_argmax, d_argmax, sizeof(int), cudaMemcpyDeviceToHost);
            return h_argmax[0];
        }

        // DEBUG: only run pre_fc_norm_hidden then lm_head. Should produce a
        // sensible (but different from main) token if the norm weight is OK.
        if (getenv("MTP_NORM_ONLY")) {
            static int call_n = 0;
            bool dbg = (call_n < 2) && getenv("MTP_NORM_DUMP");
            call_n++;
            if (dbg) dump_half_stats("h_main_normed", h_main_normed, H);
            if (dbg) dump_f32_stats("pre_fc_norm_hidden", pre_fc_norm_hidden, H);
            rms_norm_f32w_kernel<<<1, min(H, 1024), min(H, 1024) * sizeof(float), stream>>>(
                h_main_normed, pre_fc_norm_hidden, hn_tmp, H, eps);
            if (dbg) {
                cudaDeviceSynchronize();
                cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess) printf("[MTP-NORMDBG] kernel err: %s\n", cudaGetErrorString(e));
                dump_half_stats("hn_tmp_after_norm", hn_tmp, H);
            }
            qi.quantize(hn_tmp, H, stream);
            quant_gemv(lm_head_data, lm_head_type, hn_tmp, logits_buf, H, V, &qi, stream);
            argmax_half_kernel<<<1, 1024, 0, stream>>>(logits_buf, V, d_argmax);
            cudaMemcpy(h_argmax, d_argmax, sizeof(int), cudaMemcpyDeviceToHost);
            return h_argmax[0];
        }

        // DEBUG: run pre_fc_norm + concat + fc, then lm_head. Tests the fc
        // projection without the transformer layer.
        if (getenv("MTP_FC_ONLY")) {
            // embed lookup
            cudaSetDevice(embd_gpu);
            if (embd_type == GGML_TYPE_Q8_0)
                dequant_embd_q8_0_row<<<(H + 255) / 256, 256>>>(embd_data, embed_src_dev, prev_token_id, H);
            cudaMemcpy(embed_pinned_host, embed_src_dev, H * sizeof(half), cudaMemcpyDeviceToHost);
            cudaSetDevice(gpu_id);
            cudaMemcpy(embed_tmp, embed_pinned_host, H * sizeof(half), cudaMemcpyHostToDevice);

            rms_norm_f32w_kernel<<<1, min(H, 1024), min(H, 1024) * sizeof(float), stream>>>(
                embed_tmp, pre_fc_norm_embedding, en_tmp, H, eps);
            rms_norm_f32w_kernel<<<1, min(H, 1024), min(H, 1024) * sizeof(float), stream>>>(
                h_main_normed, pre_fc_norm_hidden, hn_tmp, H, eps);
            cudaMemcpyAsync(fused_tmp,     en_tmp, H * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(fused_tmp + H, hn_tmp, H * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            qi_2H.quantize(fused_tmp, 2 * H, stream);
            quant_gemv(fc, GGML_TYPE_Q8_0, fused_tmp, h_tmp, 2 * H, H, &qi_2H, stream);

            // skip the transformer layer; lm_head directly on h_tmp (no final norm)
            qi.quantize(h_tmp, H, stream);
            quant_gemv(lm_head_data, lm_head_type, h_tmp, logits_buf, H, V, &qi, stream);
            argmax_half_kernel<<<1, 1024, 0, stream>>>(logits_buf, V, d_argmax);
            cudaMemcpy(h_argmax, d_argmax, sizeof(int), cudaMemcpyDeviceToHost);
            return h_argmax[0];
        }

        // 1. Pull embed(prev_token_id) row from main embd into embed_tmp on this GPU.
        //    The embedding lives on embd_gpu in GGUF Q8_0 layout (after the
        //    gpu_loader.h repack to block_q8_0_aligned). Run dequant on
        //    embd_gpu, copy via host pinned staging, store on this GPU.
        cudaSetDevice(embd_gpu);
        if (embd_type == GGML_TYPE_Q8_0) {
            dequant_embd_q8_0_row<<<(H + 255) / 256, 256>>>(embd_data, embed_src_dev, prev_token_id, H);
        } else if (embd_type == GGML_TYPE_F16) {
            cudaMemcpy(embed_src_dev,
                       (const half*)embd_data + (size_t)prev_token_id * H,
                       H * sizeof(half), cudaMemcpyDeviceToDevice);
        }
        cudaMemcpy(embed_pinned_host, embed_src_dev, H * sizeof(half), cudaMemcpyDeviceToHost);
        cudaSetDevice(gpu_id);
        cudaMemcpy(embed_tmp, embed_pinned_host, H * sizeof(half), cudaMemcpyHostToDevice);

        // 2. pre_fc_norm_embedding(e) and pre_fc_norm_hidden(h_main_normed)
        rms_norm_f32w_kernel<<<1, min(H, 1024), min(H, 1024) * sizeof(float), stream>>>(
            embed_tmp, pre_fc_norm_embedding, en_tmp, H, eps);
        rms_norm_f32w_kernel<<<1, min(H, 1024), min(H, 1024) * sizeof(float), stream>>>(
            h_main_normed, pre_fc_norm_hidden, hn_tmp, H, eps);

        // 3. fused = concat([en, hn]) → just copy contiguously into fused_tmp
        cudaMemcpyAsync(fused_tmp,         en_tmp, H * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(fused_tmp + H,     hn_tmp, H * sizeof(half), cudaMemcpyDeviceToDevice, stream);

        // 4. h = fc @ fused (input dim 2H, output dim H)
        qi_2H.quantize(fused_tmp, 2 * H, stream);
        quant_gemv(fc, GGML_TYPE_Q8_0, fused_tmp, h_tmp, 2 * H, H, &qi_2H, stream);

        // 5. Transformer layer ─────────────────────────────────────────────
        // 5a. input_layernorm
        rms_norm_f32w_kernel<<<1, min(H, 1024), min(H, 1024) * sizeof(float), stream>>>(
            h_tmp, input_layernorm, h_norm_tmp, H, eps);

        // 5b. q_proj, k_proj, v_proj (input quantized once, reused)
        qi.quantize(h_norm_tmp, H, stream);
        quant_gemv(q_proj, GGML_TYPE_Q8_0, h_norm_tmp, qg_tmp, H, qg_dim, &qi, stream);
        quant_gemv(k_proj, GGML_TYPE_Q8_0, h_norm_tmp, k_tmp,  H, kv_dim, &qi, stream);
        quant_gemv(v_proj, GGML_TYPE_Q8_0, h_norm_tmp, v_tmp,  H, kv_dim, &qi, stream);

        // 5c. deinterleave q + gate (each [num_q * head_dim])
        deinterleave_qg_kernel<<<(q_dim + 255) / 256, 256, 0, stream>>>(
            qg_tmp, q_tmp, gate_tmp, num_q, head_dim);

        // 5d. per-head q_norm / k_norm
        int tn = min(head_dim, 128);
        head_rms_norm_kernel<<<num_q,  tn, tn * sizeof(float), stream>>>(
            q_tmp, q_norm, num_q,  head_dim, eps);
        head_rms_norm_kernel<<<num_kv, tn, tn * sizeof(float), stream>>>(
            k_tmp, k_norm, num_kv, head_dim, eps);

        // 5e. RoPE on partial dim (rope_dim total → half_rope sin/cos pairs)
        int half_rope = rope_dim / 2;
        const float* sin_pos = rope_sin + (size_t)position * half_rope;
        const float* cos_pos = rope_cos + (size_t)position * half_rope;
        apply_rope_kernel<<<(num_q  * half_rope + 255) / 256, 256, 0, stream>>>(
            q_tmp, sin_pos, cos_pos, num_q,  head_dim, rope_dim);
        apply_rope_kernel<<<(num_kv * half_rope + 255) / 256, 256, 0, stream>>>(
            k_tmp, sin_pos, cos_pos, num_kv, head_dim, rope_dim);

        // 5f. Append k, v into MTP's own KV cache at slot kv_pos
        if (kv_pos >= max_seq) kv_pos = max_seq - 1;  // safety; phase 0 only
        half* kpos = kv_k + (size_t)kv_pos * kv_dim;
        half* vpos = kv_v + (size_t)kv_pos * kv_dim;
        cudaMemcpyAsync(kpos, k_tmp, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(vpos, v_tmp, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        int seq_len = kv_pos + 1;
        kv_pos++;

        // 5g. attention: Q @ K^T → softmax → @ V
        float scale = rsqrtf((float)head_dim);
        dim3 score_grid(num_q, seq_len);
        attn_score_kernel_h<<<score_grid, min(head_dim, 256), 0, stream>>>(
            q_tmp, kv_k, attn_scores, num_q, num_kv, head_dim, seq_len, scale);
        { int st = 1; while (st < seq_len && st < 256) st <<= 1;
          softmax_kernel<<<num_q, st, st * sizeof(float), stream>>>(
              attn_scores, num_q, seq_len); }
        attn_value_kernel_h<<<num_q, min(head_dim, 256), 0, stream>>>(
            attn_scores, kv_v, attn_out, num_q, num_kv, head_dim, seq_len);

        // 5h. attn_out *= sigmoid(gate)
        apply_gate_sigmoid<<<(q_dim + 255) / 256, 256, 0, stream>>>(
            attn_out, gate_tmp, q_dim);

        // 5i. o_proj
        qi_qdim.quantize(attn_out, q_dim, stream);
        quant_gemv(o_proj, GGML_TYPE_Q8_0, attn_out, attn_proj_out, q_dim, H, &qi_qdim, stream);

        // 5j. residual: h_tmp += attn_proj_out
        add_kernel<<<(H + 255) / 256, 256, 0, stream>>>(h_tmp, attn_proj_out, H);

        // 5k. post_attention_layernorm
        rms_norm_f32w_kernel<<<1, min(H, 1024), min(H, 1024) * sizeof(float), stream>>>(
            h_tmp, post_attention_layernorm, h_norm_tmp, H, eps);

        // 5l. SwiGLU MLP
        qi.quantize(h_norm_tmp, H, stream);
        quant_gemv(gate_proj, GGML_TYPE_Q8_0, h_norm_tmp, mlp_gate_tmp, H, I, &qi, stream);
        quant_gemv(up_proj,   GGML_TYPE_Q8_0, h_norm_tmp, mlp_up_tmp,   H, I, &qi, stream);
        silu_mul_kernel<<<(I + 255) / 256, 256, 0, stream>>>(mlp_gate_tmp, mlp_up_tmp, I);
        qi_inter.quantize(mlp_gate_tmp, I, stream);
        quant_gemv(down_proj, GGML_TYPE_Q8_0, mlp_gate_tmp, mlp_out, I, H, &qi_inter, stream);

        // 5m. residual: h_tmp += mlp_out
        add_kernel<<<(H + 255) / 256, 256, 0, stream>>>(h_tmp, mlp_out, H);

        // 6. final mtp.norm + lm_head
        rms_norm_f32w_kernel<<<1, min(H, 1024), min(H, 1024) * sizeof(float), stream>>>(
            h_tmp, final_norm, h_final, H, eps);
        qi.quantize(h_final, H, stream);
        quant_gemv(lm_head_data, lm_head_type, h_final, logits_buf, H, V, &qi, stream);

        // 7. argmax (reuse the engine's GPU argmax)
        argmax_half_kernel<<<1, 1024, 0, stream>>>(logits_buf, V, d_argmax);
        cudaMemcpy(h_argmax, d_argmax, sizeof(int), cudaMemcpyDeviceToHost);
        return h_argmax[0];
    }

    // MTP forward that also exposes its post-final-norm hidden state. Used by
    // MTP K=2 to self-chain: draft2 = forward(h_final_1, draft1, pos+2).
    // `h_final_out` must be a half[H] buffer on `gpu_id`. Returns argmax.
    int forward_with_state(const half* h_main_normed, int prev_token_id, int position,
                           half* h_final_out) {
        int tok = forward(h_main_normed, prev_token_id, position);
        cudaSetDevice(gpu_id);
        cudaMemcpyAsync(h_final_out, h_final, H * sizeof(half),
                        cudaMemcpyDeviceToDevice, 0);
        return tok;
    }

    // DDTree: copy the last forward()'s post-final-norm hidden into `dst` so
    // the caller can chain a subsequent MTP forward from it. Must be called
    // immediately after forward() / forward_topk() and before any other
    // forward() overwrites h_final. `dst` must be half[H] on gpu_id.
    void copy_h_final_to(half* dst, cudaStream_t stream = 0) {
        cudaSetDevice(gpu_id);
        cudaMemcpyAsync(dst, h_final, H * sizeof(half),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // DDTree: forward followed by top-K argmax instead of top-1. Fills
    // `out_ids[0..K-1]` with the K highest-logit token ids (sorted descending)
    // and, if non-null, `out_logits[0..K-1]` with their fp32 logits. Also
    // returns the top-1 (same as forward()). The MTP KV cache is advanced by
    // one slot as usual; callers that used the non-top-1 branches must
    // `kv_rollback(1)` if they didn't commit that slot.
    int forward_topk(const half* h_main_normed, int prev_token_id, int position,
                     int K, int* out_ids_host, float* out_logits_host = nullptr) {
        // Reuse the standard forward() up through the lm_head projection, but
        // replace the argmax step with a top-K reduction.
        int top1 = forward(h_main_normed, prev_token_id, position);
        // `logits_buf` still holds the post-lm_head logits from the last call.
        cudaSetDevice(gpu_id);
        cudaStream_t stream = 0;
        if (K <= 4) {
            static int*   d_topk     = nullptr;
            static float* d_topk_val = nullptr;
            if (!d_topk)     cudaMalloc(&d_topk,     4 * sizeof(int));
            if (!d_topk_val) cudaMalloc(&d_topk_val, 4 * sizeof(float));
            argmax_topk_half_kernel<4><<<1, 1024, 0, stream>>>(
                logits_buf, V, d_topk, out_logits_host ? d_topk_val : nullptr);
            cudaMemcpy(out_ids_host, d_topk, 4 * sizeof(int), cudaMemcpyDeviceToHost);
            if (out_logits_host) {
                cudaMemcpy(out_logits_host, d_topk_val, 4 * sizeof(float), cudaMemcpyDeviceToHost);
            }
            // Caller asked for K<=4; leave slots out_ids_host[K..3] unused.
            (void)K;
        } else {
            static int*   d_topk     = nullptr;
            static float* d_topk_val = nullptr;
            if (!d_topk)     cudaMalloc(&d_topk,     8 * sizeof(int));
            if (!d_topk_val) cudaMalloc(&d_topk_val, 8 * sizeof(float));
            argmax_topk_half_kernel<8><<<1, 1024, 0, stream>>>(
                logits_buf, V, d_topk, out_logits_host ? d_topk_val : nullptr);
            cudaMemcpy(out_ids_host, d_topk, 8 * sizeof(int), cudaMemcpyDeviceToHost);
            if (out_logits_host) {
                cudaMemcpy(out_logits_host, d_topk_val, 8 * sizeof(float), cudaMemcpyDeviceToHost);
            }
        }
        return top1;
    }

    // Rollback helpers for MTP K=2. If the main model rejects draft1 or
    // draft2, the MTP KV cache has advanced too far. Each forward() appends
    // one K/V slot and increments kv_pos, so 1 draft = 1 slot. Roll back by
    // decrementing kv_pos.
    void kv_rollback(int steps) {
        kv_pos = (kv_pos >= steps) ? (kv_pos - steps) : 0;
    }
};
