#include "gguf.h"
#include "gpu_loader.h"
#include "model.cuh"
#include "mtp_head.cuh"
#include "gemma_model.cuh"
#include "ops.cuh"
#include "turboquant.cuh"
#include "tokenizer.h"
#include "sampling.h"
#include "scheduler.h"
#include "server.h"
#include "vision.cuh"
#include "dflash_decode.cuh"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <csignal>
#include <poll.h>
// Vision hooks (shared via global pointers — see main() wiring). When the CLI
// is invoked with --mmproj + --image-raw, main() runs the ViT on GPU 0 and
// populates g_vision_embeds with 576 fp16 rows of LLM-hidden dim. During
// prefill, any token matching g_image_pad_id uses the next vision embedding
// instead of the normal dequant_embd path.
static half* g_vision_embeds = nullptr;
static int g_vision_n_tokens = 0;
static int g_vision_H = 0;
static int g_image_pad_id = -1;
// Qwen3-VL multimodal RoPE (M-RoPE) per-token position arrays. One device
// buffer per GPU because the layers are spread across 4 cards with no P2P
// (Volta CMP). forward_attn_chunk in model.cuh dereferences
// g_mrope_pos_*[g] for whichever GPU it's running on. nullptr → 1D fallback.
int* g_mrope_pos_t[4] = {nullptr, nullptr, nullptr, nullptr};
int* g_mrope_pos_h[4] = {nullptr, nullptr, nullptr, nullptr};
int* g_mrope_pos_w[4] = {nullptr, nullptr, nullptr, nullptr};
int g_mrope_sec_t = 0;
int g_mrope_sec_h = 0;
int g_mrope_sec_w = 0;
int g_mrope_len = 0;   // valid range [0, g_mrope_len) of g_mrope_pos_*[g]
// Persistent vision encoder for the server path. Lazily loaded once when
// `--vision-mmproj` is passed and reused across requests. Lives for the
// program's lifetime — the GGUF mmap stays open behind it.
static vision::VisionModel* g_vision_model = nullptr;
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM
#include "stb_image.h"
#include <chrono>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <cstring>
#include <iostream>

// Decode an in-memory image (PNG/JPEG/BMP/GIF/...) and bilinear-resize to a
// fixed 768×768 fp32 tensor with the (3,H,W) channel-first layout the ViT
// expects, normalised to [-1,1] (mean=std=0.5 per llama.cpp's qwen3vl
// preprocessor). Returns true on success, false if stb_image rejected the
// bytes.
static bool decode_and_preprocess_image(const std::vector<uint8_t>& bytes,
                                        int target_size,
                                        std::vector<float>& out_chw) {
    int w = 0, h = 0, ch = 0;
    stbi_uc* pix = stbi_load_from_memory(bytes.data(), (int)bytes.size(),
                                         &w, &h, &ch, 3);
    if (!pix) {
        fprintf(stderr, "[vision] stbi_load_from_memory failed: %s\n", stbi_failure_reason());
        return false;
    }
    out_chw.assign((size_t)3 * target_size * target_size, 0.0f);
    const float scale_x = (float)w / target_size;
    const float scale_y = (float)h / target_size;
    for (int dy = 0; dy < target_size; dy++) {
        float fy = (dy + 0.5f) * scale_y - 0.5f;
        int y0 = (int)floorf(fy);
        float ay = fy - y0;
        int y1 = y0 + 1;
        if (y0 < 0) y0 = 0; if (y0 >= h) y0 = h - 1;
        if (y1 < 0) y1 = 0; if (y1 >= h) y1 = h - 1;
        for (int dx = 0; dx < target_size; dx++) {
            float fx = (dx + 0.5f) * scale_x - 0.5f;
            int x0 = (int)floorf(fx);
            float ax = fx - x0;
            int x1 = x0 + 1;
            if (x0 < 0) x0 = 0; if (x0 >= w) x0 = w - 1;
            if (x1 < 0) x1 = 0; if (x1 >= w) x1 = w - 1;
            for (int c = 0; c < 3; c++) {
                float v00 = pix[(y0 * w + x0) * 3 + c];
                float v01 = pix[(y0 * w + x1) * 3 + c];
                float v10 = pix[(y1 * w + x0) * 3 + c];
                float v11 = pix[(y1 * w + x1) * 3 + c];
                float v0 = v00 * (1 - ax) + v01 * ax;
                float v1 = v10 * (1 - ax) + v11 * ax;
                float v  = v0  * (1 - ay) + v1  * ay;
                // Normalise: pixel/255 → (val - 0.5) / 0.5 = 2*val/255 - 1
                v = v * (2.0f / 255.0f) - 1.0f;
                out_chw[((size_t)c * target_size + dy) * target_size + dx] = v;
            }
        }
    }
    stbi_image_free(pix);
    return true;
}

// Build per-token (t,h,w) position arrays from a finalised prompt and upload
// to all GPUs. Each contiguous run of `image_pad_id` tokens is treated as one
// image of `n_merged_per_side` × `n_merged_per_side` patches; the remaining
// tokens get a sequential 1D position so RoPE collapses to its standard
// behaviour for them. Frees any previous arrays first. `gen_reserve` extra
// slots are appended past the prompt so generated tokens still hit the
// vision-aware position table.
static void setup_mrope_positions(const std::vector<int>& prompt_ids,
                                  int n_gpus,
                                  int n_merged_per_side,
                                  int image_pad_id,
                                  int gen_reserve = 4096) {
    if (image_pad_id < 0) return;
    int ng_local = std::min(n_gpus, 4);
    for (int g = 0; g < ng_local; g++) {
        cudaSetDevice(g);
        if (g_mrope_pos_t[g]) { cudaFree(g_mrope_pos_t[g]); g_mrope_pos_t[g] = nullptr; }
        if (g_mrope_pos_h[g]) { cudaFree(g_mrope_pos_h[g]); g_mrope_pos_h[g] = nullptr; }
        if (g_mrope_pos_w[g]) { cudaFree(g_mrope_pos_w[g]); g_mrope_pos_w[g] = nullptr; }
    }
    g_mrope_len = 0;
    int nx = n_merged_per_side, ny = n_merged_per_side;
    size_t total_len = prompt_ids.size() + (size_t)gen_reserve;
    std::vector<int> h_pos_t(total_len), h_pos_h(total_len), h_pos_w(total_len);
    int logical = 0;
    size_t i = 0;
    bool sanity_seq = getenv("VL_MROPE_SANITY") != nullptr;
    bool axis_swap = getenv("VL_AXIS_HW_SWAP") != nullptr;
    bool t_zero = getenv("VL_T_ZERO") != nullptr;
    while (i < prompt_ids.size()) {
        if (!sanity_seq && prompt_ids[i] == image_pad_id) {
            size_t run_start = i;
            while (i < prompt_ids.size() && prompt_ids[i] == image_pad_id) i++;
            size_t run_len = i - run_start;
            int vision_base = logical;
            for (size_t k = 0; k < run_len; k++) {
                int y = (int)k / nx, x = (int)k % nx;
                int hv = axis_swap ? x : y;
                int wv = axis_swap ? y : x;
                h_pos_t[run_start + k] = t_zero ? 0 : vision_base;
                h_pos_h[run_start + k] = vision_base + hv;
                h_pos_w[run_start + k] = vision_base + wv;
            }
            logical += std::max(nx, ny);
        } else {
            h_pos_t[i] = h_pos_h[i] = h_pos_w[i] = logical++;
            i++;
        }
    }
    for (size_t k = prompt_ids.size(); k < total_len; k++) {
        h_pos_t[k] = h_pos_h[k] = h_pos_w[k] = logical++;
    }
    size_t bytes = total_len * sizeof(int);
    for (int g = 0; g < ng_local; g++) {
        cudaSetDevice(g);
        cudaMalloc(&g_mrope_pos_t[g], bytes);
        cudaMalloc(&g_mrope_pos_h[g], bytes);
        cudaMalloc(&g_mrope_pos_w[g], bytes);
        cudaMemcpy(g_mrope_pos_t[g], h_pos_t.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(g_mrope_pos_h[g], h_pos_h.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(g_mrope_pos_w[g], h_pos_w.data(), bytes, cudaMemcpyHostToDevice);
    }
    cudaSetDevice(0);
    g_mrope_len = (int)total_len;
}

// ============ Gemma 4 generation loop ============

int run_gemma(GGUFFile& gguf, GPUModel& gpu_model, int n_gpus, const SamplingParams& sp, const std::vector<int>& prompt_ids_in, const Tokenizer* tok_ptr = nullptr) {
    GemmaModel model;
    model.gpu = &gpu_model;
    model.init_config(gguf);
    model.alloc_buffers();
    model.init_kv_caches(4096, false);  // fp16 KV for debugging
    model.init_rope(4096);

    int H = model.cfg.hidden_size;
    int V = model.cfg.vocab_size;
    int last_gpu = gpu_model.layer_gpu[model.cfg.num_layers - 1];

    std::vector<int> prompt_ids = prompt_ids_in;
    if (prompt_ids.empty()) prompt_ids = {2};  // BOS token for Gemma

    Sampler sampler;
    sampler.init(sp, V);
    sp.print();

    auto* embd_t = gpu_model.get("token_embd.weight");
    auto* out_norm_t = gpu_model.get("output_norm.weight");
    printf("embd: gpu=%d type=%d dims=[%d,%d], out_norm: gpu=%d type=%d, last_gpu=%d\n",
           embd_t->gpu_id, embd_t->type, embd_t->dims[0], embd_t->dims[1],
           out_norm_t->gpu_id, out_norm_t->type, last_gpu);
    // Tied embeddings: output weight = token_embd.weight (on GPU 0)

    half* gpu_hidden[4];
    for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaMalloc(&gpu_hidden[g], H * sizeof(half)); }
    cudaSetDevice(0);
    half* logits_buf; cudaMalloc(&logits_buf, V * sizeof(half));
    half* norm_buf; cudaMalloc(&norm_buf, H * sizeof(half));
    // norm_out on last GPU for final layer norm
    cudaSetDevice(last_gpu);
    half* norm_out_last; cudaMalloc(&norm_out_last, H * sizeof(half));
    half* host_transfer; cudaMallocHost(&host_transfer, H * sizeof(half));
    QuantInput qi_logits;
    float embd_scale = model.cfg.embd_scale;

    model.reset_all();
    std::vector<int> generated;
    int max_gen = sp.max_tokens > 0 ? sp.max_tokens : 4096;

    printf("\n=== Gemma 4 Generation (%zu prompt tokens) ===\n", prompt_ids.size());
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < (int)(prompt_ids.size() + max_gen); step++) {
        int token_id = step < (int)prompt_ids.size() ? prompt_ids[step] : generated.back();
        auto step_start = std::chrono::high_resolution_clock::now();

        // 1. Embedding dequant on GPU 0
        cudaSetDevice(0);
        if (embd_t->type == GGML_TYPE_Q8_0)
            dequant_embd_q8_0_row<<<(H + 255) / 256, 256>>>(embd_t->data, gpu_hidden[0], token_id, H);
        else if (embd_t->type == GGML_TYPE_Q8_K)
            dequant_embd_q8k_row<<<(H + 255) / 256, 256>>>(embd_t->data, gpu_hidden[0], token_id, H);
        else if (embd_t->type == GGML_TYPE_Q5_K)
            dequant_embd_q5k_row_v2<<<(H + 255) / 256, 256>>>(embd_t->data, gpu_hidden[0], token_id, H);
        else if (embd_t->type == GGML_TYPE_Q6_K)
            dequant_embd_q6k_row_v2<<<(H + 255) / 256, 256>>>(embd_t->data, gpu_hidden[0], token_id, H);

        // 2. Scale embedding by sqrt(hidden_size)
        scale_embedding_kernel<<<(H + 255) / 256, 256>>>(gpu_hidden[0], embd_scale, H);

        half* h = gpu_hidden[0];

        // 3. Layer loop
        for (int layer = 0; layer < model.cfg.num_layers; layer++) {
            int g = gpu_model.layer_gpu[layer];
            int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer - 1];
            if (g != prev_g) {
                cudaSetDevice(prev_g); cudaDeviceSynchronize();
                cudaMemcpy(host_transfer, h, H * sizeof(half), cudaMemcpyDeviceToHost);
                cudaSetDevice(g);
                cudaMemcpy(gpu_hidden[g], host_transfer, H * sizeof(half), cudaMemcpyHostToDevice);
                h = gpu_hidden[g];
            }
            cudaSetDevice(g);
            if (model.is_full_attn(layer))
                model.forward_full_attn(layer, h, step, 0);
            else
                model.forward_sliding_attn(layer, h, step, 0);
            if (model.cfg.is_moe)
                model.forward_moe(layer, h, 0);
            else
                model.forward_mlp(layer, h, 0);
            // layer_output_scale now applied inside forward_*_attn and forward_moe/mlp

        }

        // 4. Output: norm + logits (tied embeddings on GPU 0)
        if (step >= (int)prompt_ids.size() - 1) {
            cudaSetDevice(last_gpu); cudaDeviceSynchronize();

            // Final norm on last GPU
            if (out_norm_t->type == GGML_TYPE_F32)
                rms_norm_f32w(norm_out_last, h, (float*)out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
            else
                rms_norm(norm_out_last, h, (half*)out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
            cudaDeviceSynchronize();

            // Transfer normed hidden to GPU 0 for tied-embedding logits
            cudaMemcpy(host_transfer, norm_out_last, H * sizeof(half), cudaMemcpyDeviceToHost);
            cudaSetDevice(0);
            cudaMemcpy(norm_buf, host_transfer, H * sizeof(half), cudaMemcpyHostToDevice);

            // Debug: check normed hidden before logits
            { std::vector<half> d(H); cudaMemcpy(d.data(), norm_buf, H*sizeof(half), cudaMemcpyDeviceToHost);
              float s=0; for(int i=0;i<H;i++) s+=fabsf(__half2float(d[i]));
              printf("[DBG] norm_buf L1=%.2f first=%.4f\n", s, __half2float(d[0])); }

            // Logits = norm_buf @ token_embd.weight^T
            qi_logits.quantize(norm_buf, H, 0);
            quant_gemv(embd_t->data, embd_t->type, norm_buf, logits_buf, H, V, &qi_logits);

            // Softcap
            if (model.cfg.softcap > 0)
                softcap_kernel<<<(V + 255) / 256, 256>>>(logits_buf, model.cfg.softcap, V);
            cudaDeviceSynchronize();

            std::vector<half> h_logits(V);
            cudaMemcpy(h_logits.data(), logits_buf, V * sizeof(half), cudaMemcpyDeviceToHost);

            std::vector<int> ctx = prompt_ids;
            ctx.insert(ctx.end(), generated.begin(), generated.end());
            int max_idx = sampler.sample(h_logits.data(), V, ctx);

            auto step_end = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();

            if (step == (int)prompt_ids.size() - 1)
                printf("Prefill max[%d]=%.2f (%.1f ms)\n", max_idx, __half2float(h_logits[max_idx]), step_ms);

            if (step >= (int)prompt_ids.size()) {
                generated.push_back(max_idx);
                if ((int)generated.size() <= 5 || (int)generated.size() % 50 == 0)
                    printf("Gen %d: tok=%d (%.1f ms)\n", (int)generated.size(), max_idx, step_ms);
                // EOS tokens for Gemma 4: 1 or 106
                if (max_idx == 1 || max_idx == 106) break;
            } else {
                generated.push_back(max_idx);
            }
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    printf("\nGenerated %d tokens in %.0f ms = %.1f t/s\n",
        (int)generated.size(), total_ms, generated.size() * 1000.0 / total_ms);

    printf("Token IDs: ");
    for (int t : generated) printf("%d ", t);
    printf("\n");

    if (tok_ptr) {
        std::string text = tok_ptr->decode(generated);
        printf("Output: %s\n", text.c_str());
    }

    for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaFree(gpu_hidden[g]); }
    cudaSetDevice(0); cudaFree(logits_buf); cudaFree(norm_buf);
    cudaSetDevice(last_gpu); cudaFree(norm_out_last);
    cudaFreeHost(host_transfer);
    return 0;
}

// ============ Chat mode: interactive multi-turn ============

int run_chat(GGUFFile& gguf, GPUModel& gpu_model, int n_gpus, const SamplingParams& sp, const Tokenizer& tok) {
    QwenModel model;
    model.gpu = &gpu_model;
    model.init_config(gguf);
    model.alloc_buffers();
    model.init_gdn_states();
    model.init_attention(4096);

    int H = model.cfg.hidden_size;
    int V = model.cfg.vocab_size;
    int last_gpu = gpu_model.layer_gpu[model.cfg.num_layers - 1];

    auto* embd_t = gpu_model.get("token_embd.weight");
    auto* out_norm_t = gpu_model.get("output_norm.weight");
    auto* out_w = gpu_model.get("output.weight");

    float* gpu_hidden[4];
    half* gpu_hidden_half[4];
    for (int g = 0; g < n_gpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&gpu_hidden[g], H * sizeof(float));
        cudaMalloc(&gpu_hidden_half[g], H * sizeof(half));
    }
    cudaSetDevice(last_gpu);
    half* logits_buf; cudaMalloc(&logits_buf, V * sizeof(half));
    half* norm_buf; cudaMalloc(&norm_buf, H * sizeof(half));
    float* host_transfer; cudaMallocHost(&host_transfer, H * sizeof(float));
    QuantInput qi_logits;

    Sampler sampler;
    sampler.init(sp, V);

    std::vector<std::pair<std::string, std::string>> history;

    printf("\n=== Qwen Chat (type 'quit' to exit, 'clear' to reset) ===\n");
    sp.print();
    printf("\n");

    while (true) {
        printf("\033[1;32mYou:\033[0m ");
        fflush(stdout);
        std::string input;
        if (!std::getline(std::cin, input) || input == "quit") break;
        if (input.empty()) continue;
        if (input == "clear") {
            history.clear();
            model.reset_all_states();
            printf("[Cleared]\n\n");
            continue;
        }

        // Build full prompt from history
        history.push_back({"user", input});
        auto prompt_ids = tok.apply_chat("", history);

        // Reset model state (stateful GDN, KV cache)
        model.reset_all_states();

        // Generate (max_tokens=0 means unlimited, cap at context size)
        std::vector<int> generated;
        int max_gen = sp.max_tokens > 0 ? sp.max_tokens : 4096 - (int)prompt_ids.size();
        if (max_gen < 1) max_gen = 1;
        int total_steps = (int)prompt_ids.size() + max_gen;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto t_first_token = t0;
        bool got_first = false;

        printf("\033[1;34mAssistant:\033[0m ");
        fflush(stdout);

        bool in_think = false;
        std::string utf8_buf;  // buffer for partial UTF-8 sequences
        for (int step = 0; step < total_steps; step++) {
            int token_id = step < (int)prompt_ids.size() ? prompt_ids[step] : generated.back();

            // Embedding (fp16 → fp32 hidden)
            cudaSetDevice(0);
            if (embd_t->type == GGML_TYPE_Q8_0)
                dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
            else if (embd_t->type == GGML_TYPE_Q5_K)
                dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
            else if (embd_t->type == GGML_TYPE_Q6_K)
                dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
            half_to_float_kernel<<<(H+255)/256, 256>>>(gpu_hidden_half[0], gpu_hidden[0], H);

            float* h = gpu_hidden[0];
            for (int layer = 0; layer < model.cfg.num_layers; layer++) {
                int g = model.gpu->layer_gpu[layer];
                int prev_g = (layer == 0) ? 0 : model.gpu->layer_gpu[layer - 1];
                if (g != prev_g) {
                    cudaSetDevice(prev_g);
                    cudaMemcpy(host_transfer, h, H * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaSetDevice(g);
                    cudaMemcpy(gpu_hidden[g], host_transfer, H * sizeof(float), cudaMemcpyHostToDevice);
                    h = gpu_hidden[g];
                } else {
                    cudaSetDevice(g);
                }
                if (model.is_attn_layer(layer))
                    model.forward_attn(layer, h, step, 0);
                else
                    model.forward_gdn(layer, h, 0);
                if (model.layer_is_moe[layer]) model.forward_moe(layer, h, 0);
                else                           model.forward_mlp(layer, h, 0);
            }

            // Decode only after prefill
            if (step >= (int)prompt_ids.size() - 1) {
                cudaSetDevice(last_gpu); cudaDeviceSynchronize();
                float_to_half_kernel<<<(H+255)/256, 256>>>(h, gpu_hidden_half[last_gpu], H);
                if (out_norm_t->type == GGML_TYPE_F32)
                    rms_norm_f32w(norm_buf, gpu_hidden_half[last_gpu], (float*)out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
                else
                    rms_norm(norm_buf, gpu_hidden_half[last_gpu], (half*)out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
                qi_logits.quantize(norm_buf, H, 0);
                quant_gemv(out_w->data, out_w->type, norm_buf, logits_buf, H, V, &qi_logits);
                cudaDeviceSynchronize();

                std::vector<half> h_logits(V);
                cudaMemcpy(h_logits.data(), logits_buf, V * sizeof(half), cudaMemcpyDeviceToHost);

                std::vector<int> ctx = prompt_ids;
                ctx.insert(ctx.end(), generated.begin(), generated.end());
                int max_idx = sampler.sample(h_logits.data(), V, ctx);

                if (step >= (int)prompt_ids.size()) {
                    generated.push_back(max_idx);
                    if (!got_first) { t_first_token = std::chrono::high_resolution_clock::now(); got_first = true; }

                    // Handle <think> display (dim reasoning, skip after first block)
                    if (max_idx == 248068) {  // <think>
                        in_think = true;
                        printf("\033[2m");  // dim
                        fflush(stdout);
                        continue;
                    }
                    if (max_idx == 248069) {  // </think>
                        if (in_think) { printf("\033[0m"); fflush(stdout); }
                        in_think = false;
                        continue;
                    }

                    // Stream decoded text with UTF-8 buffering
                    utf8_buf += tok.decode_token(max_idx);
                    std::string complete = Tokenizer::extract_complete_utf8(utf8_buf);
                    if (!complete.empty()) {
                        printf("%s", complete.c_str());
                        fflush(stdout);
                    }

                    // EOS: im_end, endoftext, or im_start (new turn)
                    if (max_idx == 248046 || max_idx == 248044 || max_idx == 248045) break;
                } else {
                    generated.push_back(max_idx);
                }
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(t_first_token - t0).count();
        double gen_ms = std::chrono::duration<double, std::milli>(t1 - t_first_token).count();
        int gen_count = (int)generated.size();
        printf("\n\033[2m[prefill %zu tok %.1fs | gen %d tok %.1f t/s]\033[0m\n\n",
               prompt_ids.size(), prefill_ms / 1000.0,
               gen_count, gen_count > 1 ? (gen_count - 1) * 1000.0 / gen_ms : 0);

        // Add assistant response to history
        std::string response = tok.decode(generated);
        history.push_back({"assistant", response});
    }

    for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaFree(gpu_hidden[g]); }
    cudaFree(logits_buf); cudaFree(norm_buf); cudaFreeHost(host_transfer);
    return 0;
}

// ============ DFlash drafter training-data extraction (offline) ============
//
// --mode dflash-extract: runs the 27B PREFILL over a pre-tokenized corpus and
// dumps, per sequence, the captured target-layer hidden states C (the DFlash
// "context features") + the token ids, plus a one-time export of the
// token-embedding and lm-head weights dequantized to fp16.
//
// C is captured via the SAME mechanism inference uses (init_dflash_capture +
// dflash_capture_chunk), so the layout matches what the drafter sees at
// inference time exactly:
//   gpu0_buf[(token_pos * n_slots + slot) * H + h]   n_slots=5, H=5120
// C for token t = concat over slot=0..4 of gpu0_buf[t,slot,:] = [5*H]=[25600].
//
// The prefill loop mirrors the simple chunked BENCH_PREFILL path in run_qwen
// (embed → per-layer forward_attn/gdn_chunk + forward_mlp/moe_chunk with the
// cross-GPU host-bridge transfer), and calls dflash_capture_chunk after each
// layer's full forward — exactly as the serve path does (main.cu:1899).
//
// On-disk format (read by the Python trainer — do NOT change):
//   <dir>/meta.json
//   <dir>/embed.fp16     [V, H] fp16 row-major  (token_embd dequantized)
//   <dir>/lm_head.fp16   [V, H] fp16 row-major  (output.weight dequantized)
//   <dir>/chunk_000.bin, ...  each = concat of per-seq records:
//       [int32 magic=0x44464331 ("DFC1")][int32 L][int32 tokens[L]][fp16 C[L*5*H]]
//   <dir>/chunks.txt     listing "<filename> <record_count>" per line.
int run_dflash_extract(GGUFFile& gguf, GPUModel& gpu_model, int n_gpus,
                       const std::string& corpus_path,
                       const std::string& out_dir,
                       size_t chunk_bytes) {
    QwenModel model;
    model.gpu = &gpu_model;
    model.init_config(gguf);
    model.alloc_buffers();

    const int H = model.cfg.hidden_size;
    const int V = model.cfg.vocab_size;
    const int n_layers = model.cfg.num_layers;
    const int last_gpu = gpu_model.layer_gpu[n_layers - 1];
    constexpr int CHUNK = QwenModel::CHUNK_SIZE;

    // DFLASH_EXTRACT_PIPELINE (default ON): cross-sequence pipelined prefill.
    // Each in-flight pipeline buffer is bound to its own KV/GDN slot + its own
    // capture region, so independent corpus sequences prefill concurrently
    // across the 3 GPU stages (GPU0 does seq B while GPU1 does seq A etc.).
    // Set to 0 to fall back to the original single-stream per-sequence loop
    // (used for the byte-identical correctness diff).
    const bool use_pipeline = [](){ const char* e=getenv("DFLASH_EXTRACT_PIPELINE"); return !e || e[0]!='0'; }();
    // Number of concurrent in-flight sequences = pipeline buffers = KV/GDN
    // slots. 6 keeps all 3 GPU stages saturated with margin. DFLASH_EXTRACT_NB
    // overrides for tuning.
    int NB = 6;
    if (const char* e = getenv("DFLASH_EXTRACT_NB")) { int v = atoi(e); if (v >= 1 && v <= 16) NB = v; }
    if (!use_pipeline) NB = 1;
    model.init_gdn_states(NB);
    constexpr int N_TGT = dflash::DraftConfig::n_target_layers;     // 5
    constexpr int BLOCK_SIZE = dflash::DraftConfig::block_size;     // 16
    constexpr int MASK_TOKEN = dflash::DraftConfig::mask_token_id;  // 248070

    // ── Read corpus: one line per sequence, space-separated int token ids ──
    std::vector<std::vector<int>> sequences;
    {
        std::ifstream cf(corpus_path);
        if (!cf) { fprintf(stderr, "[dflash-extract] cannot open corpus %s\n", corpus_path.c_str()); return 1; }
        std::string line;
        while (std::getline(cf, line)) {
            std::vector<int> ids;
            const char* p = line.c_str();
            char* end = nullptr;
            while (*p) {
                while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
                if (!*p) break;
                long v = std::strtol(p, &end, 10);
                if (end == p) break;
                if (v >= 0 && v < V) ids.push_back((int)v);
                p = end;
            }
            if (!ids.empty()) sequences.push_back(std::move(ids));
        }
    }
    if (sequences.empty()) { fprintf(stderr, "[dflash-extract] corpus has no valid sequences\n"); return 1; }
    size_t max_L = 0;
    for (auto& s : sequences) max_L = std::max(max_L, s.size());
    printf("[dflash-extract] %zu sequences, max_len=%zu, H=%d V=%d layers=%d\n",
           sequences.size(), max_L, H, V, n_layers);

    // Attention/KV + GDN are sized per slot. With the pipeline we run NB
    // independent sequences concurrently, one per slot.
    int per_slot_cap = (int)std::max<size_t>(max_L, 1);
    {
        std::vector<int> caps(NB, per_slot_cap);
        model.init_attention_caps(caps);
    }
    // Capture buffer holds NB independent regions of max_L tokens each, so the
    // NB in-flight sequences write to disjoint ranges of gpu0_buf. The single-
    // stream path (NB=1) is unchanged (one region of max_L).
    model.init_dflash_capture((int)(NB * max_L), dflash::kTargetLayerIds, N_TGT);

    // ── Output dir + chunk-stream bookkeeping ──
    {
        std::string mk = "mkdir -p '" + out_dir + "'";
        if (system(mk.c_str()) != 0) { fprintf(stderr, "[dflash-extract] mkdir failed\n"); return 1; }
    }

    auto* embd_t = gpu_model.get("token_embd.weight");
    auto* out_w  = gpu_model.get("output.weight");
    if (!embd_t) { fprintf(stderr, "[dflash-extract] token_embd.weight missing\n"); return 1; }
    if (!out_w)  { fprintf(stderr, "[dflash-extract] output.weight missing — falling back to token_embd\n"); out_w = embd_t; }

    // DFLASH_EXTRACT_TGT=1: also store the 27B greedy argmax per position (DFC2
    // record) so the drafter trains on the TARGET's distribution (knowledge
    // distillation), not the raw training-corpus next-token. Training-only:
    // serving/verify/accept is untouched. MVP = single-stream path only (the
    // pipeline path's per-GPU worker threads would race the shared argmax
    // scratch), so require DFLASH_EXTRACT_PIPELINE=0.
    const bool extract_tgt = (getenv("DFLASH_EXTRACT_TGT") != nullptr);
    auto* out_norm_t = gpu_model.get("output_norm.weight");
    if (extract_tgt && !out_norm_t) {
        fprintf(stderr, "[dflash-extract] DFLASH_EXTRACT_TGT set but output_norm.weight missing\n"); return 1;
    }
    if (extract_tgt && use_pipeline) {
        fprintf(stderr, "[dflash-extract] DFLASH_EXTRACT_TGT requires DFLASH_EXTRACT_PIPELINE=0 (single-stream)\n"); return 1;
    }

    // ── 1. meta.json ──
    {
        std::string meta = out_dir + "/meta.json";
        FILE* mf = fopen(meta.c_str(), "w");
        if (!mf) { fprintf(stderr, "[dflash-extract] cannot write meta.json\n"); return 1; }
        fprintf(mf,
            "{\"hidden\":%d, \"n_tgt\":%d, \"target_layer_ids\":[%d,%d,%d,%d,%d], "
            "\"vocab\":%d, \"block_size\":%d, \"mask_token_id\":%d, \"c_dtype\":\"fp16\", \"format\":\"%s\"}\n",
            H, N_TGT,
            dflash::kTargetLayerIds[0], dflash::kTargetLayerIds[1], dflash::kTargetLayerIds[2],
            dflash::kTargetLayerIds[3], dflash::kTargetLayerIds[4],
            V, BLOCK_SIZE, MASK_TOKEN, extract_tgt ? "DFC2" : "DFC1");
        fclose(mf);
        printf("[dflash-extract] wrote meta.json\n");
    }

    // ── 2/3. Dequantize embed + lm_head weights → fp16 [V,H] row-major ──
    // Both are GPU-resident GGUF tensors. We dequant row-by-row on the source
    // GPU using the existing dequant_embd_q8_0_row kernel, copy to host, write.
    auto dequant_weight_to_file = [&](GPUTensor* w, const std::string& fname) -> bool {
        cudaSetDevice(w->gpu_id);
        half* d_row = nullptr;
        if (cudaMalloc(&d_row, (size_t)H * sizeof(half)) != cudaSuccess) return false;
        std::vector<half> h_row(H);
        std::string path = out_dir + "/" + fname;
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) { cudaFree(d_row); return false; }
        bool ok = true;
        for (int row = 0; row < V; row++) {
            if (w->type == GGML_TYPE_Q8_0)
                dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(w->data, d_row, row, H);
            else if (w->type == GGML_TYPE_Q5_K)
                dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(w->data, d_row, row, H);
            else if (w->type == GGML_TYPE_Q6_K)
                dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(w->data, d_row, row, H);
            else { fprintf(stderr, "[dflash-extract] unsupported weight type %d for %s\n", (int)w->type, fname.c_str()); ok = false; break; }
            cudaMemcpy(h_row.data(), d_row, (size_t)H * sizeof(half), cudaMemcpyDeviceToHost);
            if (fwrite(h_row.data(), sizeof(half), H, f) != (size_t)H) { ok = false; break; }
        }
        fclose(f);
        cudaFree(d_row);
        return ok;
    };
    // DFLASH_SKIP_WEIGHTS=1: skip the embed.fp16 / lm_head.fp16 export (each
    // ~V*H*2 ≈ 2.5 GB). These exports are mode-independent (computed before the
    // prefill loop) and only used for the correctness A/B diff of the C arrays
    // and throughput benchmarking — not by the on-disk record stream. Default
    // OFF so a real extract still produces the full output set.
    const bool skip_weights = (getenv("DFLASH_SKIP_WEIGHTS") != nullptr);
    if (!skip_weights) {
        if (!dequant_weight_to_file(embd_t, "embed.fp16")) {
            fprintf(stderr, "[dflash-extract] embed.fp16 export failed\n"); return 1;
        }
        printf("[dflash-extract] wrote embed.fp16 (%zu bytes)\n", (size_t)V * H * sizeof(half));
        if (!dequant_weight_to_file(out_w, "lm_head.fp16")) {
            fprintf(stderr, "[dflash-extract] lm_head.fp16 export failed\n"); return 1;
        }
        printf("[dflash-extract] wrote lm_head.fp16 (%zu bytes)\n", (size_t)V * H * sizeof(half));
    } else {
        printf("[dflash-extract] DFLASH_SKIP_WEIGHTS=1 — skipping embed.fp16/lm_head.fp16 export\n");
    }

    // ── Chunk-file stream ──
    std::vector<std::pair<std::string,int>> chunk_index;  // (filename, record_count)
    FILE* chunk_f = nullptr;
    int   chunk_idx = -1;
    char  cur_chunk_name[64] = {0};
    size_t cur_chunk_bytes = 0;
    int    cur_chunk_records = 0;
    // DFLASH_NO_WRITE=1: open chunk files on /dev/null instead of out_dir.
    // Benchmark-only — runs the full GPU compute + C readback path but discards
    // the (potentially many-GB) record bytes so throughput can be measured on a
    // large corpus without the disk to hold the output. Capture/compute is
    // untouched. Default OFF.
    const bool no_write = (getenv("DFLASH_NO_WRITE") != nullptr);
    // Roll to a fresh chunk file when the current one would exceed chunk_bytes.
    auto ensure_chunk = [&](size_t need_bytes) {
        if (chunk_f == nullptr || (cur_chunk_bytes + need_bytes > chunk_bytes && cur_chunk_records > 0)) {
            if (chunk_f) {
                fclose(chunk_f);
                chunk_index.push_back({std::string(cur_chunk_name), cur_chunk_records});
            }
            chunk_idx++;
            snprintf(cur_chunk_name, sizeof(cur_chunk_name), "chunk_%03d.bin", chunk_idx);
            std::string path = no_write ? std::string("/dev/null") : (out_dir + "/" + cur_chunk_name);
            chunk_f = fopen(path.c_str(), "wb");
            cur_chunk_bytes = 0;
            cur_chunk_records = 0;
            printf("[dflash-extract] opened %s\n", cur_chunk_name);
        }
    };
    // Write one record from a host-side C buffer (byte-identical layout to the
    // single-stream path). Used by BOTH paths.
    auto write_record = [&](int L, const int* ids_ptr, const half* C, size_t c_count,
                            const int* tgt = nullptr) {
        bool has_tgt = (tgt != nullptr);
        size_t rec_bytes = sizeof(int32_t) * 2 + (size_t)L * sizeof(int32_t)
                         + c_count * sizeof(half)
                         + (has_tgt ? (size_t)L * sizeof(int32_t) : 0);
        ensure_chunk(rec_bytes);
        int32_t magic = has_tgt ? 0x44464332 : 0x44464331;  // "DFC2" : "DFC1"
        int32_t Lw = L;
        fwrite(&magic, sizeof(int32_t), 1, chunk_f);
        fwrite(&Lw,    sizeof(int32_t), 1, chunk_f);
        fwrite(ids_ptr, sizeof(int32_t), L, chunk_f);
        fwrite(C, sizeof(half), c_count, chunk_f);
        if (has_tgt) fwrite(tgt, sizeof(int32_t), L, chunk_f);   // 27B greedy argmax[L]
        cur_chunk_bytes += rec_bytes;
        cur_chunk_records++;
    };

    auto t_start = std::chrono::high_resolution_clock::now();
    size_t total_tokens = 0;

    if (!use_pipeline) {
        // ───────────────────── Single-stream path (NB=1) ────────────────────
        // Original hand-rolled chunked prefill. Kept for the byte-identical
        // correctness diff (DFLASH_EXTRACT_PIPELINE=0).
        half* gpu_hidden_half0 = nullptr;
        cudaSetDevice(0); cudaMalloc(&gpu_hidden_half0, (size_t)H * sizeof(half));
        float* gpu_hidden_chunk[4] = {};
        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&gpu_hidden_chunk[g], (size_t)CHUNK * H * sizeof(float));
        }
        float* host_chunk_transfer = nullptr;
        cudaMallocHost(&host_chunk_transfer, (size_t)CHUNK * H * sizeof(float));
        std::vector<half> host_C;

        // KD (DFLASH_EXTRACT_TGT): per-chunk scratch on last_gpu to compute the
        // 27B greedy argmax (same out_norm + lm_head + argmax the serve path uses)
        // from each chunk's post-final-layer hidden, plus a per-seq host buffer.
        std::vector<int> host_tgt;
        half* tgt_norm_buf = nullptr; half* tgt_logits_buf = nullptr; int* tgt_d_argmax = nullptr;
        QuantInput tgt_qi;
        if (extract_tgt) {
            cudaSetDevice(last_gpu);
            cudaMalloc(&tgt_norm_buf,   (size_t)CHUNK * H * sizeof(half));
            cudaMalloc(&tgt_logits_buf, (size_t)CHUNK * V * sizeof(half));
            cudaMalloc(&tgt_d_argmax,   (size_t)CHUNK * sizeof(int));
        }
        // h_final[chunk_n,H] (fp32, on last_gpu) -> per-row 27B greedy id into out_host.
        auto compute_tgt_chunk = [&](const float* h_final, int chunk_n, int* out_host) {
            cudaSetDevice(last_gpu);
            if (out_norm_t->type == GGML_TYPE_F32)
                rms_norm_f32in_f32w(tgt_norm_buf, h_final, (float*)out_norm_t->data, chunk_n, H, model.cfg.rms_norm_eps, 0);
            else
                rms_norm_f32in(tgt_norm_buf, h_final, (half*)out_norm_t->data, chunk_n, H, model.cfg.rms_norm_eps, 0);
            tgt_qi.quantize_chunk(tgt_norm_buf, H, chunk_n, 0);
            quant_gemv_chunk(out_w->data, out_w->type, tgt_qi.q8_buf, tgt_logits_buf, H, V, chunk_n, 0);
            for (int t = 0; t < chunk_n; t++)
                argmax_half_kernel<<<1, 1024>>>(tgt_logits_buf + (size_t)t * V, V, tgt_d_argmax + t);
            cudaMemcpy(out_host, tgt_d_argmax, (size_t)chunk_n * sizeof(int), cudaMemcpyDeviceToHost);
        };

        for (size_t si = 0; si < sequences.size(); si++) {
            const std::vector<int>& ids = sequences[si];
            int L = (int)ids.size();
            if (L > (int)max_L) L = (int)max_L;

            // Fresh KV/GDN state + zero the capture buffer region for this seq.
            model.reset_slot_states(0);
            cudaSetDevice(0);
            cudaMemset(model.dflash_cap.gpu0_buf, 0,
                       (size_t)L * N_TGT * H * sizeof(half));

            int chunk_pos = 0;
            while (chunk_pos < L) {
                int chunk_n = std::min(CHUNK, L - chunk_pos);
                cudaSetDevice(0);
                for (int t = 0; t < chunk_n; t++) {
                    int token_id = ids[chunk_pos + t];
                    if (embd_t->type == GGML_TYPE_Q8_0)
                        dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half0, token_id, H);
                    else if (embd_t->type == GGML_TYPE_Q5_K)
                        dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half0, token_id, H);
                    else if (embd_t->type == GGML_TYPE_Q6_K)
                        dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half0, token_id, H);
                    half_to_float_kernel<<<(H+255)/256, 256>>>(
                        gpu_hidden_half0, gpu_hidden_chunk[0] + (size_t)t * H, H);
                }
                float* h_chunk = gpu_hidden_chunk[0];
                for (int layer = 0; layer < n_layers; layer++) {
                    int g = gpu_model.layer_gpu[layer];
                    int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer - 1];
                    if (g != prev_g) {
                        cudaSetDevice(prev_g);
                        cudaMemcpy(host_chunk_transfer, h_chunk,
                                   (size_t)chunk_n * H * sizeof(float), cudaMemcpyDeviceToHost);
                        cudaSetDevice(g);
                        cudaMemcpy(gpu_hidden_chunk[g], host_chunk_transfer,
                                   (size_t)chunk_n * H * sizeof(float), cudaMemcpyHostToDevice);
                        h_chunk = gpu_hidden_chunk[g];
                    } else {
                        cudaSetDevice(g);
                    }
                    bool is_attn = model.is_attn_layer(layer);
                    if (is_attn) model.forward_attn_chunk(layer, h_chunk, chunk_pos, chunk_n, 0);
                    else         model.forward_gdn_chunk (layer, h_chunk, chunk_n, 0);
                    if (model.layer_is_moe[layer]) model.forward_moe_chunk(layer, h_chunk, chunk_n, 0);
                    else                           model.forward_mlp_chunk(layer, h_chunk, chunk_n, 0);
                    model.dflash_capture_chunk(layer, h_chunk, chunk_pos, chunk_n, g, 0);
                }
                cudaSetDevice(last_gpu); cudaDeviceSynchronize();
                // KD: h_chunk now holds the post-final-layer hidden for these
                // chunk_n positions on last_gpu → 27B greedy argmax at p=chunk_pos+t.
                if (extract_tgt) {
                    if ((int)host_tgt.size() < L) host_tgt.resize(L);
                    compute_tgt_chunk(h_chunk, chunk_n, host_tgt.data() + chunk_pos);
                }
                chunk_pos += chunk_n;
            }

            cudaSetDevice(0); cudaDeviceSynchronize();
            size_t c_count = (size_t)L * N_TGT * H;
            host_C.resize(c_count);
            cudaMemcpy(host_C.data(), model.dflash_cap.gpu0_buf,
                       c_count * sizeof(half), cudaMemcpyDeviceToHost);
            write_record(L, ids.data(), host_C.data(), c_count,
                         extract_tgt ? host_tgt.data() : nullptr);
            total_tokens += L;

            if ((si + 1) % 64 == 0 || si + 1 == sequences.size()) {
                auto now = std::chrono::high_resolution_clock::now();
                double s = std::chrono::duration<double>(now - t_start).count();
                printf("[dflash-extract] %zu/%zu seqs, %zu tokens, %.1f tok/s\n",
                       si + 1, sequences.size(), total_tokens,
                       s > 0 ? total_tokens / s : 0.0);
            }
        }
        cudaFreeHost(host_chunk_transfer);
        for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaFree(gpu_hidden_chunk[g]); }
        cudaSetDevice(0); cudaFree(gpu_hidden_half0);
        if (extract_tgt) {
            cudaSetDevice(last_gpu);
            if (tgt_norm_buf)   cudaFree(tgt_norm_buf);
            if (tgt_logits_buf) cudaFree(tgt_logits_buf);
            if (tgt_d_argmax)   cudaFree(tgt_d_argmax);
        }
    } else {
        // ──────────────── Cross-sequence pipelined path (NB slots) ──────────
        // Multi-threaded, one CPU launch thread per GPU segment (mirrors the
        // serve PREFILL_PIPELINE_V3 design). Each in-flight sequence owns one
        // KV/GDN slot + one capture region; while sequence A's chunk is on GPU2
        // sequence B's chunk runs on GPU1 and sequence C's on GPU0, so all 3
        // GPU stages stay saturated even though each individual sequence is
        // only 1 chunk (≤256 tok) → no per-sequence pipeline benefit alone.
        // Per-GPU compute scratch is shared, so each GPU runs ONE chunk-segment
        // at a time (FIFO on its compute stream) — exactly like serve.

        // Contiguous (gpu, l_lo, l_hi) layer segments.
        struct GpuSeg { int g; int l_lo; int l_hi; };
        std::vector<GpuSeg> gpu_segs;
        {
            int cur_g = gpu_model.layer_gpu[0]; int seg_start = 0;
            for (int l = 1; l < n_layers; l++) {
                if (gpu_model.layer_gpu[l] != cur_g) {
                    gpu_segs.push_back({cur_g, seg_start, l});
                    cur_g = gpu_model.layer_gpu[l]; seg_start = l;
                }
            }
            gpu_segs.push_back({cur_g, seg_start, n_layers});
        }
        const int nseg = (int)gpu_segs.size();
        const int last_seg = nseg - 1;
        const int g0 = gpu_segs[0].g;

        // Per-GPU streams.
        cudaStream_t comp_stream[4] = {}, d2h_stream[4] = {};
        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g);
            cudaStreamCreate(&comp_stream[g]);
            cudaStreamCreate(&d2h_stream[g]);
        }
        // Per-slot embed scratch (GPU0) + per-(GPU,slot) hidden chunk buffers.
        half*  embed_half[16] = {};
        float* gpu_hidden[4][16] = {};
        cudaSetDevice(g0);
        for (int s = 0; s < NB; s++) cudaMalloc(&embed_half[s], (size_t)H * sizeof(half));
        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g);
            for (int s = 0; s < NB; s++)
                cudaMalloc(&gpu_hidden[g][s], (size_t)CHUNK * H * sizeof(float));
        }
        // Pinned host bridge buffers: one per (slot, hop). hop = source segment.
        float* host_xfer[16][4] = {};
        for (int s = 0; s < NB; s++)
            for (int h = 0; h < nseg; h++)
                cudaMallocHost(&host_xfer[s][h], (size_t)CHUNK * H * sizeof(float));
        // Events: seg-done[s][slot] (compute) and d2h-done[s][slot] (host coherent).
        cudaEvent_t ev_sd[4][16] = {}, ev_dh[4][16] = {};
        for (int s = 0; s < nseg; s++) {
            cudaSetDevice(gpu_segs[s].g);
            for (int sl = 0; sl < NB; sl++) {
                cudaEventCreateWithFlags(&ev_sd[s][sl], cudaEventDisableTiming);
                cudaEventCreateWithFlags(&ev_dh[s][sl], cudaEventDisableTiming);
            }
        }

        // A chunk work item flowing through the stages.
        struct W { int seq; int slot; int ctx_base; int pos; int n; bool first; bool last; };

        // Run segment `s` on its GPU for work `w`. Resets the slot's segment
        // state on the FIRST chunk of the sequence (stream-ordered before the
        // forward kernels), then runs the captured-layer forward exactly like
        // the serve chunked path.
        auto run_seg = [&](int s, const W& w){
            int g = gpu_segs[s].g;
            cudaSetDevice(g);
            cudaStream_t cs = comp_stream[g];
            float* h_chunk = gpu_hidden[g][w.slot];
            if (w.first)
                model.reset_slot_segment_stream(w.slot, gpu_segs[s].l_lo, gpu_segs[s].l_hi, g, cs);
            for (int layer = gpu_segs[s].l_lo; layer < gpu_segs[s].l_hi; layer++) {
                if (model.is_attn_layer(layer)) model.forward_attn_chunk(layer, h_chunk, w.pos, w.n, cs, w.slot);
                else                            model.forward_gdn_chunk (layer, h_chunk, w.n, cs, w.slot);
                if (model.layer_is_moe[layer])  model.forward_moe_chunk(layer, h_chunk, w.n, cs);
                else                            model.forward_mlp_chunk(layer, h_chunk, w.n, cs);
                model.dflash_capture_chunk(layer, h_chunk, w.pos, w.n, g, cs, w.ctx_base);
            }
            cudaEventRecord(ev_sd[s][w.slot], cs);
        };
        auto run_d2h = [&](int s, const W& w){
            int g = gpu_segs[s].g;
            cudaSetDevice(g);
            cudaStreamWaitEvent(d2h_stream[g], ev_sd[s][w.slot], 0);
            cudaMemcpyAsync(host_xfer[w.slot][s], gpu_hidden[g][w.slot],
                (size_t)w.n * H * sizeof(float), cudaMemcpyDeviceToHost, d2h_stream[g]);
            cudaEventRecord(ev_dh[s][w.slot], d2h_stream[g]);
        };
        auto run_embed = [&](const W& w){
            cudaSetDevice(g0);
            cudaStream_t cs = comp_stream[g0];
            float* h_in = gpu_hidden[g0][w.slot];
            const std::vector<int>& ids = sequences[w.seq];
            for (int t = 0; t < w.n; t++) {
                int token_id = ids[w.pos + t];
                if (embd_t->type == GGML_TYPE_Q8_0)
                    dequant_embd_q8_0_row<<<(H+255)/256,256,0,cs>>>(embd_t->data, embed_half[w.slot], token_id, H);
                else if (embd_t->type == GGML_TYPE_Q5_K)
                    dequant_embd_q5k_row_v2<<<(H+255)/256,256,0,cs>>>(embd_t->data, embed_half[w.slot], token_id, H);
                else if (embd_t->type == GGML_TYPE_Q6_K)
                    dequant_embd_q6k_row_v2<<<(H+255)/256,256,0,cs>>>(embd_t->data, embed_half[w.slot], token_id, H);
                half_to_float_kernel<<<(H+255)/256,256,0,cs>>>(embed_half[w.slot], h_in + (size_t)t * H, H);
            }
        };

        // Completed-sequence collection: last stage stores each seq's C into a
        // host buffer; the main thread flushes records in seq order so the
        // on-disk record order is identical to the single-stream path.
        std::vector<std::vector<half>> seq_C(sequences.size());
        std::vector<char> seq_ready(sequences.size(), 0);
        std::mutex done_mtx;

        // Per-stage work queues + slot free-list.
        std::mutex mtx; std::condition_variable cv;
        std::vector<std::deque<W>> q(nseg);
        std::vector<bool> q_done(nseg, false);
        std::deque<int> free_slots; for (int s = 0; s < NB; s++) free_slots.push_back(s);
        // Per-slot count of chunks that have FULLY drained the pipeline (last
        // stage done). A slot's per-(stage,slot) events + host_xfer bridge
        // buffers are reused by every chunk of the sequence bound to that slot,
        // so two chunks of the SAME slot must never be in flight at once
        // (the slot-scoped events would be re-recorded and the cross-stage
        // fences would sync the wrong chunk → corruption, manifesting only on
        // sequences > CHUNK_SIZE tokens). The producer therefore admits a
        // sequence's chunk cp+1 only after chunk cp has been counted here.
        // Independent slots stay fully concurrent, so the common single-chunk
        // case (and cross-sequence overlap) is unaffected.
        std::vector<int> slot_chunk_done(NB, 0);

        // Consumer threads for stages 1..last_seg.
        std::vector<std::thread> workers;
        for (int s = 1; s < nseg; s++) {
            workers.emplace_back([&, s]{
                cudaSetDevice(gpu_segs[s].g);
                for (;;) {
                    W w;
                    { std::unique_lock<std::mutex> lk(mtx);
                      cv.wait(lk, [&]{ return !q[s].empty() || q_done[s]; });
                      if (q[s].empty()) break;
                      w = q[s].front(); q[s].pop_front(); }
                    // Wait for the previous segment's D2H to be host-coherent,
                    // then H2D into this GPU's slot buffer.
                    cudaEventSynchronize(ev_dh[s-1][w.slot]);
                    // Multi-chunk: don't H2D over a buffer whose previous chunk's
                    // stage-s D2H hasn't read it yet (intermediate stages only).
                    if (!w.first && s < last_seg) cudaEventSynchronize(ev_dh[s][w.slot]);
                    cudaSetDevice(gpu_segs[s].g);
                    cudaMemcpyAsync(gpu_hidden[gpu_segs[s].g][w.slot], host_xfer[w.slot][s-1],
                        (size_t)w.n * H * sizeof(float), cudaMemcpyHostToDevice, comp_stream[gpu_segs[s].g]);
                    run_seg(s, w);
                    if (s < last_seg) {
                        run_d2h(s, w);
                        { std::lock_guard<std::mutex> lk(mtx); q[s+1].push_back(w); } cv.notify_all();
                    } else {
                        // Last stage: this chunk's captures are now committed to
                        // gpu0_buf (capture H2D to GPU0 is synchronous inside
                        // run_seg for src_gpu!=0; ev_sd covers the src_gpu==0
                        // case). On the sequence's last chunk, read its whole
                        // C region back and mark it ready + free the slot.
                        cudaEventSynchronize(ev_sd[last_seg][w.slot]);
                        if (w.last) {
                            int L = (int)sequences[w.seq].size();
                            if (L > (int)max_L) L = (int)max_L;
                            size_t c_count = (size_t)L * N_TGT * H;
                            std::vector<half> C(c_count);
                            cudaSetDevice(0);
                            cudaMemcpy(C.data(),
                                model.dflash_cap.gpu0_buf + (size_t)w.ctx_base * N_TGT * H,
                                c_count * sizeof(half), cudaMemcpyDeviceToHost);
                            { std::lock_guard<std::mutex> lk(done_mtx);
                              seq_C[w.seq] = std::move(C); seq_ready[w.seq] = 1; }
                            { std::lock_guard<std::mutex> lk(mtx); free_slots.push_back(w.slot); }
                        }
                        // Signal end-to-end drain of THIS chunk so the producer
                        // may admit the next chunk of the same slot.
                        { std::lock_guard<std::mutex> lk(mtx); slot_chunk_done[w.slot]++; }
                        cv.notify_all();
                    }
                }
                if (s < last_seg) { std::lock_guard<std::mutex> lk(mtx); q_done[s+1] = true; cv.notify_all(); }
            });
        }

        // Producer (this thread) = stage 0. Also drains finished sequences in
        // order to disk while waiting for free slots.
        size_t flush_next = 0;
        auto try_flush = [&](){
            for (;;) {
                bool ready = false; int L = 0;
                {
                    std::lock_guard<std::mutex> lk(done_mtx);
                    if (flush_next < sequences.size() && seq_ready[flush_next]) ready = true;
                }
                if (!ready) break;
                L = (int)sequences[flush_next].size();
                if (L > (int)max_L) L = (int)max_L;
                size_t c_count = (size_t)L * N_TGT * H;
                write_record(L, sequences[flush_next].data(), seq_C[flush_next].data(), c_count);
                total_tokens += L;
                { std::lock_guard<std::mutex> lk(done_mtx); seq_C[flush_next].clear(); seq_C[flush_next].shrink_to_fit(); }
                flush_next++;
                if (flush_next % 256 == 0 || flush_next == sequences.size()) {
                    auto now = std::chrono::high_resolution_clock::now();
                    double sec = std::chrono::duration<double>(now - t_start).count();
                    printf("[dflash-extract] %zu/%zu seqs, %zu tokens, %.1f tok/s\n",
                           flush_next, sequences.size(), total_tokens,
                           sec > 0 ? total_tokens / sec : 0.0);
                    fflush(stdout);
                }
            }
        };

        cudaSetDevice(g0);
        for (size_t si = 0; si < sequences.size(); si++) {
            int L = (int)sequences[si].size();
            if (L > (int)max_L) L = (int)max_L;
            // Acquire a free slot (= capture region = KV/GDN lane).
            int slot;
            { std::unique_lock<std::mutex> lk(mtx);
              cv.wait(lk, [&]{ return !free_slots.empty(); });
              slot = free_slots.front(); free_slots.pop_front(); }
            try_flush();
            int ctx_base = slot * (int)max_L;
            // Reset this slot's drain counter; we admit chunk #k of this slot
            // only once chunk #(k-1) has fully drained (last stage done). This
            // serializes a single sequence's chunks end-to-end through the
            // pipeline so the slot-scoped events / host_xfer bridge buffers are
            // never reused by two in-flight chunks (correctness; see above).
            { std::lock_guard<std::mutex> lk(mtx); slot_chunk_done[slot] = 0; }
            int chunk_idx = 0;
            // Emit this sequence's chunks in order through the pipeline.
            for (int cp = 0; cp < L; cp += CHUNK) {
                int n = std::min(CHUNK, L - cp);
                W w{ (int)si, slot, ctx_base, cp, n, (cp == 0), (cp + n >= L) };
                // Multi-chunk: wait for the previous chunk of THIS slot to fully
                // drain before reusing the slot's buffers/events/KV. Single-chunk
                // sequences (the common case) never enter this branch and keep
                // full cross-sequence overlap.
                if (chunk_idx > 0) {
                    if (nseg > 1) {
                        std::unique_lock<std::mutex> lk(mtx);
                        cv.wait(lk, [&]{ return slot_chunk_done[slot] >= chunk_idx; });
                    }
                    // (single-GPU path drains synchronously below, no wait needed)
                    cudaSetDevice(g0);
                }
                run_embed(w);
                run_seg(0, w);
                if (nseg > 1) {
                    run_d2h(0, w);
                    { std::lock_guard<std::mutex> lk(mtx); q[1].push_back(w); } cv.notify_all();
                } else {
                    // Single-GPU: capture already committed; handle completion here.
                    cudaEventSynchronize(ev_sd[0][slot]);
                    if (w.last) {
                        size_t c_count = (size_t)L * N_TGT * H;
                        std::vector<half> C(c_count);
                        cudaSetDevice(0);
                        cudaMemcpy(C.data(),
                            model.dflash_cap.gpu0_buf + (size_t)ctx_base * N_TGT * H,
                            c_count * sizeof(half), cudaMemcpyDeviceToHost);
                        { std::lock_guard<std::mutex> lk(done_mtx);
                          seq_C[si] = std::move(C); seq_ready[si] = 1; }
                        { std::lock_guard<std::mutex> lk(mtx); free_slots.push_back(slot); }
                        cudaSetDevice(g0);
                    }
                }
                chunk_idx++;
            }
        }
        if (nseg > 1) { std::lock_guard<std::mutex> lk(mtx); q_done[1] = true; cv.notify_all(); }
        for (auto& t : workers) t.join();
        for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaDeviceSynchronize(); }
        try_flush();  // flush any remaining ready sequences

        // Cleanup pipeline resources.
        for (int s = 0; s < NB; s++) { cudaSetDevice(g0); cudaFree(embed_half[s]); }
        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g);
            for (int s = 0; s < NB; s++) cudaFree(gpu_hidden[g][s]);
        }
        for (int s = 0; s < NB; s++)
            for (int h = 0; h < nseg; h++) cudaFreeHost(host_xfer[s][h]);
    }

    // Close last chunk + write chunks.txt.
    if (chunk_f) {
        fclose(chunk_f);
        chunk_index.push_back({std::string(cur_chunk_name), cur_chunk_records});
    }
    {
        std::string ct = out_dir + "/chunks.txt";
        FILE* cf = fopen(ct.c_str(), "w");
        if (cf) {
            for (auto& e : chunk_index) fprintf(cf, "%s %d\n", e.first.c_str(), e.second);
            fclose(cf);
        }
    }
    printf("[dflash-extract] DONE: %zu seqs, %zu tokens, %zu chunk file(s)\n",
           sequences.size(), total_tokens, chunk_index.size());
    return 0;
}

// ============ Qwen generation loop (original) ============

int run_qwen(GGUFFile& gguf, GPUModel& gpu_model, int n_gpus, const SamplingParams& sp, const std::vector<int>& prompt_ids_in, const Tokenizer* tok = nullptr) {
    QwenModel model;
    model.gpu = &gpu_model;
    model.init_config(gguf);
    model.alloc_buffers();
    model.init_gdn_states();
    model.init_attention(4096);

    int H = model.cfg.hidden_size;
    int V = model.cfg.vocab_size;
    int last_gpu = gpu_model.layer_gpu[model.cfg.num_layers - 1];

    std::vector<int> prompt_ids = prompt_ids_in;
    if (prompt_ids.empty()) prompt_ids = {1};

    Sampler sampler;
    sampler.init(sp, V);
    sp.print();

    auto* embd_t = gpu_model.get("token_embd.weight");
    auto* out_norm_t = gpu_model.get("output_norm.weight");
    auto* out_w = gpu_model.get("output.weight");

    float* gpu_hidden[4];
    half* gpu_hidden_half[4];
    for (int g = 0; g < n_gpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&gpu_hidden[g], H * sizeof(float));
        cudaMalloc(&gpu_hidden_half[g], H * sizeof(half));
    }
    cudaSetDevice(last_gpu);
    half* logits_buf; cudaMalloc(&logits_buf, V * sizeof(half));
    half* norm_buf; cudaMalloc(&norm_buf, H * sizeof(half));
    float* host_transfer; cudaMallocHost(&host_transfer, H * sizeof(float));
    QuantInput qi_logits;

    model.reset_all_states();
    std::vector<int> generated;
    int max_gen = sp.max_tokens > 0 ? sp.max_tokens : 4096;

    // BENCH_PREFILL=1: chunked prefill microbench, baseline vs FLASH_ATTN.
    // Runs the chunked pre-fill path twice (cold-loaded weights, stream 0)
    // and prints per-phase timers. Uses the same generate_impl plumbing so
    // forward_attn_chunk / _mlp_chunk / _gdn_chunk take effect. Exits after.
    if (const char* be = getenv("BENCH_PREFILL"); be && be[0] == '1') {
        constexpr int CHUNK = QwenModel::CHUNK_SIZE;
        int prompt_len_local  = (int)prompt_ids.size();
        // Include ALL prompt tokens in prefill so the final hidden = last
        // prompt token's hidden, usable for next-token argmax correctness.
        int prefill_len_local = prompt_len_local;
        float* gpu_hidden_chunk[4] = {};
        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&gpu_hidden_chunk[g], (size_t)CHUNK * H * sizeof(float));
        }
        float* host_chunk_transfer;
        cudaMallocHost(&host_chunk_transfer, (size_t)CHUNK * H * sizeof(float));

        auto run_prefill = [&](const char* label, bool use_fa) -> int {
            g_profile_attn  = true;
            g_use_flash_attn = use_fa;
            g_attn_score_ms = g_attn_softmax_ms = g_attn_value_ms = g_attn_fused_ms = 0.0;
            model.reset_all_states();

            double t_embed=0, t_xfer=0, t_attn=0, t_gdn=0, t_mlp=0;
            auto prof_now = [](){ return std::chrono::high_resolution_clock::now(); };
            auto sync_ms = [&](std::chrono::high_resolution_clock::time_point tb, int dev){
                cudaSetDevice(dev); cudaDeviceSynchronize();
                auto te = std::chrono::high_resolution_clock::now();
                return std::chrono::duration<double, std::milli>(te - tb).count();
            };
            int chunk_pos = 0;
            float* last_final_h = nullptr;
            int    last_chunk_n = 0;
            while (chunk_pos < prefill_len_local) {
                int chunk_n = std::min(CHUNK, prefill_len_local - chunk_pos);
                cudaSetDevice(0);
                auto te0 = prof_now();
                for (int t = 0; t < chunk_n; t++) {
                    int token_id = prompt_ids[chunk_pos + t];
                    if (embd_t->type == GGML_TYPE_Q8_0)
                        dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
                    else if (embd_t->type == GGML_TYPE_Q5_K)
                        dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
                    else if (embd_t->type == GGML_TYPE_Q6_K)
                        dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
                    half_to_float_kernel<<<(H+255)/256, 256>>>(
                        gpu_hidden_half[0], gpu_hidden_chunk[0] + (size_t)t * H, H);
                }
                t_embed += sync_ms(te0, 0);

                float* h_chunk = gpu_hidden_chunk[0];
                for (int layer = 0; layer < model.cfg.num_layers; layer++) {
                    int g = gpu_model.layer_gpu[layer];
                    int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer - 1];
                    if (g != prev_g) {
                        auto tx0 = prof_now();
                        cudaSetDevice(prev_g);
                        cudaMemcpy(host_chunk_transfer, h_chunk,
                                   (size_t)chunk_n * H * sizeof(float), cudaMemcpyDeviceToHost);
                        cudaSetDevice(g);
                        cudaMemcpy(gpu_hidden_chunk[g], host_chunk_transfer,
                                   (size_t)chunk_n * H * sizeof(float), cudaMemcpyHostToDevice);
                        h_chunk = gpu_hidden_chunk[g];
                        t_xfer += sync_ms(tx0, g);
                    } else {
                        cudaSetDevice(g);
                    }
                    auto ta0 = prof_now();
                    bool is_attn = model.is_attn_layer(layer);
                    if (is_attn) model.forward_attn_chunk(layer, h_chunk, chunk_pos, chunk_n, 0);
                    else         model.forward_gdn_chunk (layer, h_chunk, chunk_n, 0);
                    double ms_al = sync_ms(ta0, g);
                    if (is_attn) t_attn += ms_al; else t_gdn += ms_al;

                    auto tm0 = prof_now();
                    if (model.layer_is_moe[layer]) model.forward_moe_chunk(layer, h_chunk, chunk_n, 0);
                    else                           model.forward_mlp_chunk(layer, h_chunk, chunk_n, 0);
                    t_mlp += sync_ms(tm0, g);
                }
                cudaSetDevice(last_gpu); cudaDeviceSynchronize();
                last_final_h = h_chunk;
                last_chunk_n = chunk_n;
                chunk_pos += chunk_n;
            }
            double total = t_embed + t_xfer + t_attn + t_gdn + t_mlp;
            printf("[%s] prompt=%d total=%.1fms (%.1f t/s) "
                   "embed=%.1f xfer=%.1f attn=%.1f gdn=%.1f mlp=%.1f\n",
                   label, prefill_len_local, total, prefill_len_local * 1000.0 / total,
                   t_embed, t_xfer, t_attn, t_gdn, t_mlp);
            double sub = g_attn_score_ms + g_attn_softmax_ms + g_attn_value_ms;
            printf("[%s ATTN] score=%.1f softmax=%.1f value=%.1f (sub=%.1f) fused=%.1f\n",
                   label, g_attn_score_ms, g_attn_softmax_ms, g_attn_value_ms,
                   sub, g_attn_fused_ms);
            // Correctness probe: sample first 8 floats + next-token argmax.
            int argmax_tok = -1;
            if (last_final_h && last_chunk_n > 0) {
                cudaSetDevice(last_gpu);
                std::vector<float> hs(8);
                cudaMemcpy(hs.data(),
                           last_final_h + (size_t)(last_chunk_n - 1) * H,
                           8 * sizeof(float), cudaMemcpyDeviceToHost);
                printf("[%s HSAMPLE] %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                       label, hs[0], hs[1], hs[2], hs[3], hs[4], hs[5], hs[6], hs[7]);

                float_to_half_kernel<<<(H+255)/256, 256>>>(
                    last_final_h + (size_t)(last_chunk_n - 1) * H,
                    gpu_hidden_half[last_gpu], H);
                if (out_norm_t->type == GGML_TYPE_F32)
                    rms_norm_f32w(norm_buf, gpu_hidden_half[last_gpu],
                                  (float*)out_norm_t->data, 1, H,
                                  model.cfg.rms_norm_eps);
                else
                    rms_norm(norm_buf, gpu_hidden_half[last_gpu],
                             (half*)out_norm_t->data, 1, H,
                             model.cfg.rms_norm_eps);
                qi_logits.quantize(norm_buf, H, 0);
                quant_gemv(out_w->data, out_w->type, norm_buf, logits_buf,
                           H, V, &qi_logits);
                int* d_am; cudaMalloc(&d_am, sizeof(int));
                argmax_half_kernel<<<1, 1024>>>(logits_buf, V, d_am);
                cudaDeviceSynchronize();
                cudaMemcpy(&argmax_tok, d_am, sizeof(int), cudaMemcpyDeviceToHost);
                cudaFree(d_am);
                printf("[%s ARGMAX] next_tok=%d\n", label, argmax_tok);
            }
            fflush(stdout);
            return argmax_tok;
        };
        int base_tok = run_prefill("BASELINE", false);
        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g); cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("[post-BASELINE gpu %d] %s\n", g, cudaGetErrorString(err));
        }
        int fa_tok   = run_prefill("FLASH_ATTN", true);
        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g); cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("[post-FA gpu %d] %s\n", g, cudaGetErrorString(err));
        }
        printf("[CORRECTNESS] %s  (baseline=%d  flash_attn=%d)\n",
               (base_tok == fa_tok) ? "MATCH" : "MISMATCH",
               base_tok, fa_tok);
        fflush(stdout);
        for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaFree(gpu_hidden_chunk[g]); }
        cudaFreeHost(host_chunk_transfer);
        return 0;
    }

    printf("\n=== Qwen Generation (%zu prompt tokens) ===\n", prompt_ids.size());
    auto total_start = std::chrono::high_resolution_clock::now();

    int s_vision_idx = 0;  // running index over g_vision_embeds rows
    for (int step = 0; step < (int)(prompt_ids.size() + max_gen); step++) {
        int token_id = step < (int)prompt_ids.size() ? prompt_ids[step] : generated.back();
        auto step_start = std::chrono::high_resolution_clock::now();

        cudaSetDevice(0);
        bool vision_hit = false;
        if (g_vision_embeds && g_image_pad_id >= 0 && token_id == g_image_pad_id
            && s_vision_idx < g_vision_n_tokens && H == g_vision_H) {
            cudaMemcpyAsync(gpu_hidden_half[0],
                            g_vision_embeds + (size_t)s_vision_idx * H,
                            (size_t)H * sizeof(half),
                            cudaMemcpyDeviceToDevice);
            s_vision_idx++;
            vision_hit = true;
        }
        if (!vision_hit) {
            if (embd_t->type == GGML_TYPE_Q8_0)
                dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
            else if (embd_t->type == GGML_TYPE_Q5_K)
                dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
            else if (embd_t->type == GGML_TYPE_Q6_K)
                dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
        }
        half_to_float_kernel<<<(H+255)/256, 256>>>(gpu_hidden_half[0], gpu_hidden[0], H);

        float* h = gpu_hidden[0];
        for (int layer = 0; layer < model.cfg.num_layers; layer++) {
            int g = model.gpu->layer_gpu[layer];
            int prev_g = (layer == 0) ? 0 : model.gpu->layer_gpu[layer - 1];
            if (g != prev_g) {
                cudaSetDevice(prev_g);
                cudaMemcpy(host_transfer, h, H * sizeof(float), cudaMemcpyDeviceToHost);
                cudaSetDevice(g);
                cudaMemcpy(gpu_hidden[g], host_transfer, H * sizeof(float), cudaMemcpyHostToDevice);
                h = gpu_hidden[g];
            } else {
                cudaSetDevice(g);
            }
            if (model.is_attn_layer(layer))
                model.forward_attn(layer, h, step, 0);
            else
                model.forward_gdn(layer, h, 0);
            if (model.layer_is_moe[layer]) model.forward_moe(layer, h, 0);
            else                           model.forward_mlp(layer, h, 0);
        }

        if (step >= (int)prompt_ids.size() - 1) {
            cudaSetDevice(last_gpu); cudaDeviceSynchronize();
            float_to_half_kernel<<<(H+255)/256, 256>>>(h, gpu_hidden_half[last_gpu], H);
            if (out_norm_t->type == GGML_TYPE_F32)
                rms_norm_f32w(norm_buf, gpu_hidden_half[last_gpu], (float*)out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
            else
                rms_norm(norm_buf, gpu_hidden_half[last_gpu], (half*)out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
            qi_logits.quantize(norm_buf, H, 0);
            quant_gemv(out_w->data, out_w->type, norm_buf, logits_buf, H, V, &qi_logits);
            cudaDeviceSynchronize();

            std::vector<half> h_logits(V);
            cudaMemcpy(h_logits.data(), logits_buf, V * sizeof(half), cudaMemcpyDeviceToHost);

            // Build context for repetition penalty (prompt + generated)
            std::vector<int> ctx = prompt_ids;
            ctx.insert(ctx.end(), generated.begin(), generated.end());

            int max_idx = sampler.sample(h_logits.data(), V, ctx);

            auto step_end = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();

            if (step == (int)prompt_ids.size() - 1)
                printf("<think>=%.2f max[%d]=%.2f\n", __half2float(h_logits[248068]), max_idx,
                       __half2float(h_logits[max_idx]));

            if (step >= (int)prompt_ids.size()) {
                generated.push_back(max_idx);
                if ((int)generated.size() <= 5 || (int)generated.size() % 50 == 0)
                    printf("Gen %d: tok=%d (%.1f ms)\n", (int)generated.size(), max_idx, step_ms);
                if (max_idx == 248044 || max_idx == 248046) break;
            } else {
                generated.push_back(max_idx);
                printf("Prefill -> tok=%d (%.1f ms)\n", max_idx, step_ms);
            }
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    printf("\nGenerated %d tokens in %.0f ms = %.1f t/s\n",
        (int)generated.size(), total_ms, generated.size() * 1000.0 / total_ms);

    // Decode and print text output
    if (tok) {
        std::string text = tok->decode(generated);
        printf("\n--- Output ---\n%s\n--- End ---\n", text.c_str());
    } else {
        printf("Token IDs: ");
        for (int t : generated) printf("%d ", t);
        printf("\n");
    }

    for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaFree(gpu_hidden[g]); }
    cudaFree(logits_buf); cudaFree(norm_buf); cudaFreeHost(host_transfer);
    return 0;
}

// ============ Qwen serve mode: OpenAI-compatible HTTP API ============

int serve_qwen(GGUFFile& gguf, GPUModel& gpu_model, int n_gpus, int port, const Tokenizer& tok, const std::string& model_name, const std::string& api_key = "", int max_seq = 262144, int num_slots = 1, const std::vector<int>& slot_caps = {}, const std::string& proxy_embed_url = "", const std::string& proxy_rerank_url = "") {
    QwenModel model;
    model.gpu = &gpu_model;
    model.init_config(gguf);
    model.alloc_buffers();
    // slot_caps wins over --slots if both are given: GDN state count + KV
    // partition must agree, otherwise kv_slot_offset returns 0 for "extra"
    // GDN slots and they silently share slot 0's KV. Override num_slots in
    // the local scope so scheduler / batched-gen loop / /stats see one value.
    if (!slot_caps.empty()) num_slots = (int)slot_caps.size();
    model.init_gdn_states(num_slots);
    if (!slot_caps.empty()) {
        model.init_attention_caps(slot_caps);
    } else {
        model.init_attention(max_seq, num_slots);
    }

    int H = model.cfg.hidden_size;
    int V = model.cfg.vocab_size;
    int last_gpu = gpu_model.layer_gpu[model.cfg.num_layers - 1];

    auto* embd_t = gpu_model.get("token_embd.weight");
    auto* out_norm_t = gpu_model.get("output_norm.weight");
    auto* out_w = gpu_model.get("output.weight");

    // FP32 hidden state through forward pass (precision)
    constexpr int CHUNK_SIZE = QwenModel::CHUNK_SIZE;
    float* gpu_hidden[4];
    half*  gpu_hidden_half[4];  // scratch fp16 buffer for embedding dequant
    float* gpu_hidden_chunk[4]; // [CHUNK_SIZE * H] for chunked prompt processing
    // ── Batched gen-step buffers (Phase C) ───────────────────────────────
    // Sized for the worst case: every slot active in one batched forward.
    // The buffers are reused per-step; the static allocation simplifies the
    // code path vs lazy allocation while costing only num_slots × H × 4
    // bytes per GPU (≈ 80 KB per slot at H=5120, negligible vs KV state).
    int batch_cap = std::max(1, num_slots);
    float* gpu_hidden_batch[4];
    half*  gpu_hidden_half_batch[4];
    int*   slot_ids_dev[4];      // per-GPU copy of slot ids for the batched call
    int*   slot_pos_dev[4];      // per-GPU copy of per-slot logical positions
    int*   dst_kv_pos_dev[4];    // per-GPU copy of slot * slot_max_seq + pos
    for (int g = 0; g < n_gpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&gpu_hidden[g], H * sizeof(float));
        cudaMalloc(&gpu_hidden_half[g], H * sizeof(half));
        cudaMalloc(&gpu_hidden_chunk[g], CHUNK_SIZE * H * sizeof(float));
        cudaMalloc(&gpu_hidden_batch[g],      (size_t)batch_cap * H * sizeof(float));
        cudaMalloc(&gpu_hidden_half_batch[g], (size_t)batch_cap * H * sizeof(half));
        cudaMalloc(&slot_ids_dev[g],   (size_t)batch_cap * sizeof(int));
        cudaMalloc(&slot_pos_dev[g],   (size_t)batch_cap * sizeof(int));
        cudaMalloc(&dst_kv_pos_dev[g], (size_t)batch_cap * sizeof(int));
    }
    cudaSetDevice(last_gpu);
    half* logits_buf; cudaMalloc(&logits_buf, V * sizeof(half));
    half* norm_buf; cudaMalloc(&norm_buf, H * sizeof(half));
    half* logits_batch; cudaMalloc(&logits_batch, (size_t)batch_cap * V * sizeof(half));
    half* norm_batch;   cudaMalloc(&norm_batch,   (size_t)batch_cap * H * sizeof(half));
    int*  d_argmax_batch;       cudaMalloc(&d_argmax_batch, (size_t)batch_cap * sizeof(int));
    int*  h_argmax_batch_pinned; cudaMallocHost(&h_argmax_batch_pinned, (size_t)batch_cap * sizeof(int));
    float* host_transfer; cudaMallocHost(&host_transfer, H * sizeof(float));
    float* host_batch_transfer; cudaMallocHost(&host_batch_transfer, (size_t)batch_cap * H * sizeof(float));
    float* host_chunk_transfer; cudaMallocHost(&host_chunk_transfer, CHUNK_SIZE * H * sizeof(float));
    // ── Cross-GPU prefill pipelining resources ──────────────────────────────
    // The chunked prefill loop overlaps chunk i's D2H/H2D transfers with
    // chunk i+1's GPU0 compute. To do that safely we need:
    //   - a second pinned host buffer (host_chunk_transfer is now buf 0)
    //   - a second per-GPU device hidden chunk (so chunk i and chunk i+1
    //     can hold their hidden state on the same GPU concurrently)
    //   - per-GPU compute / D2H / H2D streams so the three queues run in
    //     parallel rather than serializing on the legacy null-stream
    // Memory cost: 2 pinned host + 1 extra device per GPU = ~5 MB / GPU at
    // CHUNK_SIZE=256 H=5120, negligible vs KV state.
    float* host_chunk_transfer_pp; cudaMallocHost(&host_chunk_transfer_pp, (size_t)CHUNK_SIZE * H * sizeof(float));
    float* host_chunk_xfer_db[2] = { host_chunk_transfer, host_chunk_transfer_pp };
    float* gpu_hidden_chunk_pp[4] = {};
    float* gpu_hidden_chunk_db[4][2] = {};
    cudaStream_t prefill_comp_stream[4] = {};
    cudaStream_t prefill_d2h_stream[4]  = {};
    cudaStream_t prefill_h2d_stream[4]  = {};
    for (int g = 0; g < n_gpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&gpu_hidden_chunk_pp[g], (size_t)CHUNK_SIZE * H * sizeof(float));
        gpu_hidden_chunk_db[g][0] = gpu_hidden_chunk[g];
        gpu_hidden_chunk_db[g][1] = gpu_hidden_chunk_pp[g];
        cudaStreamCreate(&prefill_comp_stream[g]);
        cudaStreamCreate(&prefill_d2h_stream[g]);
        cudaStreamCreate(&prefill_h2d_stream[g]);
    }
    // Contiguous (gpu, layer_lo, layer_hi) segments of the layer pipeline.
    // Each segment runs on its GPU's compute stream; transfers happen
    // between consecutive segments.
    struct GpuSeg { int g; int l_lo; int l_hi; };
    std::vector<GpuSeg> gpu_segs;
    {
        int cur_g = gpu_model.layer_gpu[0]; int seg_start = 0;
        for (int l = 1; l < model.cfg.num_layers; l++) {
            if (gpu_model.layer_gpu[l] != cur_g) {
                gpu_segs.push_back({cur_g, seg_start, l});
                cur_g = gpu_model.layer_gpu[l]; seg_start = l;
            }
        }
        gpu_segs.push_back({cur_g, seg_start, model.cfg.num_layers});
    }
    // Per-buffer events used to chain dependencies across the pipeline:
    //   seg_done[s][b]   : segment s of buf b finished compute
    //   d2h_done[s][b]   : D2H from segment s end of buf b finished
    //   h2d_done[s][b]   : H2D into segment s+1 input of buf b finished
    // Allocated up to MAX_SEGS = 4 (we currently have at most 3 GPUs in
    // service). buf is 0 or 1 (double-buffered chunks).
    constexpr int PP_MAX_SEGS = 4;
    cudaEvent_t pp_seg_done[PP_MAX_SEGS][2] = {};
    cudaEvent_t pp_d2h_done[PP_MAX_SEGS][2] = {};
    cudaEvent_t pp_h2d_done[PP_MAX_SEGS][2] = {};
    for (int s = 0; s < (int)gpu_segs.size(); s++) {
        cudaSetDevice(gpu_segs[s].g);
        for (int b = 0; b < 2; b++) {
            cudaEventCreateWithFlags(&pp_seg_done[s][b], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&pp_d2h_done[s][b], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&pp_h2d_done[s][b], cudaEventDisableTiming);
        }
    }
    // ── PREFILL_PIPELINE_V2 resources ───────────────────────────────────────
    // Deep (N-buffer) event-driven prefill pipeline. The legacy double-buffer
    // path (above) serializes the 3-GPU pipeline because the per-segment host
    // fence (cudaEventSynchronize) blocks the CPU launch loop, so GPUs run one
    // chunk-segment at a time (~48% util, measured). v2 keeps NB chunks in
    // flight and replaces the blocking fence with a NON-blocking cudaEventQuery
    // poll: when a segment's D2H completes (host memory coherent — same
    // completion guarantee as cudaEventSynchronize, unlike the GPU-side
    // cudaStreamWaitEvent which does NOT fence pinned host on no-P2P CMP), the
    // scheduler launches that hop's H2D + the next segment, and spends the wait
    // launching OTHER chunks' segments. All 3 GPUs stay busy on different
    // chunks. Single-threaded: every CUDA call is on the main thread, per-GPU
    // compute-stream FIFO serializes same-GPU chunk-segments so the shared
    // per-GPU chunk scratch is reused safely.
    constexpr int PP_NBUF = 6;
    // Default ON (verified token-identical to the v1 double-buffer pipeline on
    // full + prefix-cache-hit prefills; 1.6-1.9x faster at 18K). Opt out with
    // PREFILL_PIPELINE_V2=0.
    const bool alloc_pp_v2 = [](){ const char* e = getenv("PREFILL_PIPELINE_V2"); return !e || e[0] != '0'; }();
    float* v2_host_xfer[PP_NBUF] = {};
    float* v2_gpu_hidden[4][PP_NBUF] = {};
    cudaEvent_t v2_sd[PP_MAX_SEGS][PP_NBUF] = {};   // seg compute done
    cudaEvent_t v2_dh[PP_MAX_SEGS][PP_NBUF] = {};   // D2H done (host coherent)
    cudaEvent_t v2_hd[PP_MAX_SEGS][PP_NBUF] = {};   // H2D done
    if (alloc_pp_v2) {
        for (int b = 0; b < PP_NBUF; b++)
            cudaMallocHost(&v2_host_xfer[b], (size_t)CHUNK_SIZE * H * sizeof(float));
        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g);
            for (int b = 0; b < PP_NBUF; b++)
                cudaMalloc(&v2_gpu_hidden[g][b], (size_t)CHUNK_SIZE * H * sizeof(float));
        }
        for (int s = 0; s < (int)gpu_segs.size(); s++) {
            cudaSetDevice(gpu_segs[s].g);
            for (int b = 0; b < PP_NBUF; b++) {
                cudaEventCreateWithFlags(&v2_sd[s][b], cudaEventDisableTiming);
                cudaEventCreateWithFlags(&v2_dh[s][b], cudaEventDisableTiming);
                cudaEventCreateWithFlags(&v2_hd[s][b], cudaEventDisableTiming);
            }
        }
    }
    QuantInput qi_logits;
    // GPU argmax fast path: when temp=0 and rep_penalty=1.0 we can pick the
    // winning token entirely on-device and only transfer 4 bytes back over
    // the slow PCIe 1.0 x1 bus instead of V*2 bytes (~500 KB).
    // [0] = top-1 (standard argmax). [1] = top-2 (only populated when
    // MTP_ACCEPT_TOP2=1 and argmax_top2_half_kernel is used). Allocating
    // 2 slots unconditionally is cheap and lets the env flag flip behavior
    // without re-allocating.
    int*  d_argmax;       cudaMalloc(&d_argmax, 2 * sizeof(int));
    int*  h_argmax_pinned; cudaMallocHost(&h_argmax_pinned, 2 * sizeof(int));

    // Speculative decoding (MTP_SPEC=1) buffers — declared here, allocated
    // below after mtp.load() reports success.
    bool spec_enabled = false;
    float* gpu_hidden_b[4]   = {nullptr,nullptr,nullptr,nullptr};
    half*  gpu_hidden_half_b[4] = {nullptr,nullptr,nullptr,nullptr};
    half*  norm_buf_b   = nullptr;
    half*  logits_buf_b = nullptr;
    int*   d_argmax_b   = nullptr;
    int*   h_argmax_pinned_b = nullptr;
    float* host_transfer_b = nullptr;
    QuantInput qi_logits_b;
    // MTP K=2 third-stream buffers — allocated only when MTP_K2 is enabled.
    bool spec_k2_enabled = false;
    float* gpu_hidden_c[4]   = {nullptr,nullptr,nullptr,nullptr};
    half*  gpu_hidden_half_c[4] = {nullptr,nullptr,nullptr,nullptr};
    half*  norm_buf_c   = nullptr;
    half*  logits_buf_c = nullptr;
    int*   d_argmax_c   = nullptr;
    int*   h_argmax_pinned_c = nullptr;
    float* host_transfer_c = nullptr;
    QuantInput qi_logits_c;
    half*  h_final_draft1 = nullptr;   // MTP's post-final-norm hidden after draft1
    long long spec_k2_accept_ab_count = 0;  // both drafts accepted
    long long spec_k2_accept_a_count  = 0;  // only draft1 accepted
    long long spec_k2_reject_count    = 0;  // draft1 rejected

    // MTP_TREE: chain-tree path. Initial shape is chain budget=3 (main +
    // draft1 + draft2), mirroring MTP K=2's semantics via the tree
    // forward_gdn_tree / forward_attn_tree / forward_mlp_tree pipeline so
    // we can compare against MTP K=2 before expanding into real branching.
    bool spec_tree_enabled = false;
    // Chain-tree depth. budget=3 is MTP K=2 equivalent (main + 2 drafts).
    // Larger values extend the chain (e.g. 4 = 3 drafts) for higher per-iter
    // expected tokens at the cost of per-token-forward_attn overhead.
    int   tree_budget = getenv("MTP_TREE_BUDGET") ? atoi(getenv("MTP_TREE_BUDGET")) : 3;
    if (tree_budget < 2) tree_budget = 2;
    if (tree_budget > 8) tree_budget = 8;
    float* tree_hidden[4] = {nullptr,nullptr,nullptr,nullptr};   // [budget*H] fp32
    half*  tree_hidden_half[4] = {nullptr,nullptr,nullptr,nullptr};
    half*  tree_norm_buf   = nullptr;   // [budget*H] on last_gpu
    half*  tree_logits_buf = nullptr;   // [budget*V] on last_gpu
    int*   tree_d_argmax   = nullptr;   // [budget] on last_gpu
    int*   tree_h_argmax   = nullptr;   // [budget] pinned
    float* tree_host_transfer = nullptr;// [budget*H] pinned for GPU→GPU copy
    QuantInput qi_logits_tree;
    half*  h_final_draft2_tree = nullptr;  // second ping-pong buffer for budget > 3
    long long tree_accept_full_count = 0;
    long long tree_accept_partial_count = 0;
    long long tree_reject_count = 0;

    // ── MTP head (Phase 0: forward + accept rate measurement) ─────────────
    MTPHead mtp;
    bool mtp_loaded = false;
    {
        // Try inline GGUF MTP first (v2 models embed nextn in blk.N). The MTP
        // head handles both a dense FFN (ffn_gate/up/down) and a MoE FFN
        // (ffn_gate_inp + *_exps + shared expert) in the MTP layer — qwen3_5_moe
        // makes the MTP layer itself MoE. load_from_gguf detects which and binds
        // the right tensors; num_experts_per_tok is passed through for top-k.
        if (model.mtp_layer_idx >= 0) {
            mtp_loaded = mtp.load_from_gguf(
                gpu_model, model.mtp_layer_idx,
                model.cfg.hidden_size, V,
                model.cfg.num_q_heads, model.cfg.num_kv_heads, model.cfg.head_dim,
                model.cfg.intermediate_size, model.cfg.rope_dim, max_seq, model.cfg.rms_norm_eps,
                model.cfg.num_experts_per_tok);
        }
        // Fallback: external mtp_head_<hidden>.bin
        if (!mtp_loaded) {
            char mtp_path_buf[256];
            snprintf(mtp_path_buf, sizeof(mtp_path_buf),
                     "/home/paru/mtp_work/mtp_head_%d.bin", model.cfg.hidden_size);
            const char* mtp_path = mtp_path_buf;
            if (access(mtp_path, R_OK) != 0)
                mtp_path = "/home/paru/mtp_work/mtp_head.bin";
            if (access(mtp_path, R_OK) == 0) {
                mtp_loaded = mtp.load(
                    mtp_path, last_gpu,
                    model.cfg.hidden_size, V,
                    model.cfg.num_q_heads, model.cfg.num_kv_heads, model.cfg.head_dim,
                    model.cfg.intermediate_size, model.cfg.rope_dim, max_seq, model.cfg.rms_norm_eps);
                if (!mtp_loaded) printf("[MTP] failed to load %s\n", mtp_path);
            }
        }
        if (mtp_loaded) {
            auto* embd_t  = gpu_model.get("token_embd.weight");
            auto* outw_t  = gpu_model.get("output.weight");
            mtp.set_embed_source(embd_t->data, embd_t->type, embd_t->gpu_id);
            mtp.set_lm_head(outw_t->data, outw_t->type);
            mtp.set_rope_tables(model.rope.sin_table(last_gpu), model.rope.cos_table(last_gpu));
            printf("[MTP] head ready, will measure acceptance rate during gen\n");
        }
    }

    // Speculative decoding is the default whenever the MTP head is loaded.
    // Set MTP_SPEC_OFF=1 to fall back to the plain per-token loop. The
    // per-iter MTP measurement (formerly gated by MTP_ON=1) is suppressed
    // automatically when spec_enabled is true to avoid running MTP twice
    // per iter — the spec branch already runs its own draft MTP.
    //
    // MoE (qwen3_5_moe): MTP spec is correct (MoE-aware draft + n2/n3 verify),
    // but it's a net SLOWDOWN on these compute-bound cards. Unlike a dense MLP
    // — where the verify's 2nd token reuses the same weights and is nearly free
    // (forward_mlp_n2 batches the weight load) — MoE routes each token to a
    // different top-8 expert set, so the verify pays full expert GEMV cost for
    // the speculative 2nd token, and ~60% of that is wasted on reject. Measured
    // ~27 t/s plain vs ~22 t/s spec. Default OFF for MoE; MTP_SPEC_ON=1 forces it
    // on (correct, just slower). Dense models keep spec on by default.
    bool moe_spec_default_off = model.cfg.is_moe && getenv("MTP_SPEC_ON") == nullptr;
    spec_enabled = (mtp_loaded && getenv("MTP_SPEC_OFF") == nullptr && !moe_spec_default_off);
    spec_k2_enabled   = (spec_enabled && getenv("MTP_K2")   != nullptr && getenv("MTP_TREE") == nullptr);
    // Tree spec verify (forward_*_tree) has no MoE FFN routing yet; MoE models
    // use the n2/n3 verify paths (which do route to forward_moe). Disable tree
    // for MoE so MTP_TREE silently falls back to basic/K2 spec.
    spec_tree_enabled = (spec_enabled && getenv("MTP_TREE") != nullptr && !model.cfg.is_moe);

    // Continuous batching N>1 keeps the MTP/spec paths available because
    // they're only ever invoked from generate_impl when slot==0, and the
    // legacy fast path (slot 0 alone) holds forward_mutex while running so
    // the gen-loop thread can't race the single-global mtp.kv / gdn-snapshot
    // state. The batched-gen handoff also calls generate_impl, but breaks at
    // step == prompt_len-1 (before any spec branch which only fires at
    // step+1), so spec_enabled stays true with no batched-path leak.
    // DFlash is the exception — it still needs slot-0-only ownership of its
    // capture buffer, handled below.
    if (num_slots > 1 && (spec_enabled || spec_k2_enabled || spec_tree_enabled)) {
        printf("[server] num_slots=%d > 1: MTP/spec available on slot-0 fast path "
               "(serialized via forward_mutex)\n", num_slots);
    }

    // ── DFlash speculative decoding (block-diffusion drafter + DDTree verify) ─
    // Activated when DFLASH=1 AND DFLASH_DRAFT_PATH=<safetensors path>.
    // Disables the MTP-based speculative paths above since DFlash uses its own
    // 5-layer non-causal drafter + target hidden capture hook.
    bool dflash_enabled = false;
    dflash::DecodeState dflash_state;
    if (getenv("DFLASH") && num_slots > 1) {
        printf("[dflash] num_slots=%d > 1: DFlash needs slot-0-only capture "
               "buffer — disabled. Run with --slots 1 for DFlash.\n", num_slots);
    } else if (getenv("DFLASH")) {
        const char* dflash_path = getenv("DFLASH_DRAFT_PATH");
        if (!dflash_path) {
            printf("[dflash] DFLASH=1 but DFLASH_DRAFT_PATH unset — disabled\n");
        } else if (!dflash::dflash_init(dflash_state, dflash_path, model, max_seq)) {
            printf("[dflash] init failed — disabled\n");
        } else {
            dflash_enabled = true;
            // MVP: chain-only verify (budget = block_size = 16). KV slot ↔
            // absolute pos identity holds, so accept_len truncation is free.
            // Tree-branching (budget=22) needs KV/GDN rebuild and lands later.
            // Verify budget = how many of the drafter's 16 predicted tokens we
            // actually forward through the 27B per iteration. The verify is the
            // dominant cost (~76% of gen time, partly compute-bound in the MLP),
            // so over-verifying past the real accept length (AL) is pure waste.
            // DFLASH_BUDGET tunes it (<= block_size=16). Default 8: measured sweet
            // spot for the AL~5 drafter (16.5->24 t/s, lossless — output is byte-
            // identical to budget=16, only faster). Raise it for a higher-AL drafter.
            tree_budget = 8;
            if (const char* e = getenv("DFLASH_BUDGET")) {
                int b = atoi(e);
                if (b >= 2 && b <= dflash::DraftConfig::block_size) tree_budget = b;
            }
            dflash_state.budget = tree_budget;
            // Disable MTP paths — DFlash has its own draft.
            spec_enabled = spec_k2_enabled = spec_tree_enabled = false;
            mtp_loaded = false;
            printf("[dflash] active — chain MVP, budget=%d, MTP paths disabled\n", tree_budget);
        }
    }
    if (spec_enabled) {
        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&gpu_hidden_b[g], H * sizeof(float));
            cudaMalloc(&gpu_hidden_half_b[g], H * sizeof(half));
        }
        cudaSetDevice(last_gpu);
        cudaMalloc(&norm_buf_b,   H * sizeof(half));
        cudaMalloc(&logits_buf_b, V * sizeof(half));
        cudaMalloc(&d_argmax_b,   2 * sizeof(int));
        cudaMallocHost(&h_argmax_pinned_b, 2 * sizeof(int));
        cudaMallocHost(&host_transfer_b, H * sizeof(float));
        model.alloc_buffers_n2(max_seq);
        model.alloc_gdn_snapshots();
        printf("[SPEC] speculative decoding enabled (MTP K=1, MLP batched)\n");
    }
    if (spec_k2_enabled) {
        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&gpu_hidden_c[g], H * sizeof(float));
            cudaMalloc(&gpu_hidden_half_c[g], H * sizeof(half));
        }
        cudaSetDevice(last_gpu);
        cudaMalloc(&norm_buf_c,   H * sizeof(half));
        cudaMalloc(&logits_buf_c, V * sizeof(half));
        cudaMalloc(&d_argmax_c,   2 * sizeof(int));
        cudaMallocHost(&h_argmax_pinned_c, 2 * sizeof(int));
        cudaMallocHost(&host_transfer_c, H * sizeof(float));
        cudaMalloc(&h_final_draft1, H * sizeof(half));
        model.alloc_buffers_n3(max_seq);
        model.alloc_gdn_snapshots_b();
        printf("[SPEC-K2] MTP K=2 speculative decoding enabled (self-chained draft, N=3 batched verify)\n");
    }
    if (spec_tree_enabled || dflash_enabled) {
        // These buffers serve BOTH the draft lm_head (which always uses the full
        // block_size=16 slots) AND the verify (tree_budget <= block_size slots).
        // Size them by the MAX (block_size); using tree_budget under-allocates when
        // DFLASH_BUDGET<16 -> the 16-slot draft lm_head overflows -> draft_chain
        // corrupts to garbage -> accept collapses (the budget<16 bug). tree_budget
        // still controls the verify LOOP counts elsewhere.
        int tb_alloc = dflash_enabled
                     ? std::max(tree_budget, (int)dflash::DraftConfig::block_size)
                     : tree_budget;
        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&tree_hidden[g],      (size_t)tb_alloc * H * sizeof(float));
            cudaMalloc(&tree_hidden_half[g], (size_t)tb_alloc * H * sizeof(half));
        }
        cudaSetDevice(last_gpu);
        cudaMalloc(&tree_norm_buf,   (size_t)tb_alloc * H * sizeof(half));
        cudaMalloc(&tree_logits_buf, (size_t)tb_alloc * V * sizeof(half));
        cudaMalloc(&tree_d_argmax,   (size_t)tb_alloc * sizeof(int));
        cudaMallocHost(&tree_h_argmax, (size_t)tb_alloc * sizeof(int));
        cudaMallocHost(&tree_host_transfer, (size_t)tb_alloc * H * sizeof(float));
        if (!h_final_draft1) cudaMalloc(&h_final_draft1, H * sizeof(half));   // reused for self-chain MTP
        // Second ping-pong buffer for chain depths > 3 (budget > 3).
        // We always alloc one so budget=3 path stays simple.
        cudaMalloc(&h_final_draft2_tree, H * sizeof(half));
        // qi_logits_tree lazily sizes its q8 buffer on first quantize_chunk.
        // tree forward reuses gdn_bufs[g].chunk_* buffers, already alloc'd in
        // model init. No additional buffer pool needed here.
        model.alloc_tree_decode(tree_budget);
        if (dflash_enabled) model.init_dflash_tree_scratch(tree_budget);
        printf("[TREE] tree decoding buffers allocated (budget=%d)\n", tree_budget);
    }
    long long mtp_accept_count = 0;
    long long mtp_total_count  = 0;
    int mtp_pending_draft = -1;       // draft for some future step (MTP_DRAFT mode)
    int mtp_pending_draft_step = -1;
    long long spec_accept_count = 0;
    long long spec_total_count  = 0;

    // ── Dynamic slot-router (RANK 1) ────────────────────────────────
    // When slot 0 runs the legacy MTP single-stream fast path it holds
    // sched.forward_mutex() for its whole run, which blocks the batched
    // gen-loop the moment a 2nd request arrives (concurrency REGRESSES from
    // ~24 t/s single-stream to ~17 t/s). The router lets slot-0's MTP loop
    // cooperatively yield: when any slot 1..N becomes active mid-MTP, run_fn
    // raises `router_yield_signal`; generate_impl polls it (RELAXED, every
    // few tokens, no lock) and breaks the MTP loop CLEANLY at the TOP of a
    // step iter — where slot-0's live GDN+KV state is exactly consistent with
    // the last emitted token (no restore_gdn_states needed). It then hands the
    // continuation (next_tok + slot_pos + the tokens already emitted) back to
    // run_fn through the dyn_yield_* out-params, and run_fn registers slot 0
    // into slot_gen_state so the batched loop drives it alongside the new
    // request. QWEN_DYNAMIC_ROUTER=0 disables (today's behavior — full MTP,
    // no yield) for bisecting. Default ON.
    static const bool dynamic_router_on = []{
        const char* e = getenv("QWEN_DYNAMIC_ROUTER");
        return !e || e[0] != '0';
    }();
    std::atomic<bool> router_yield_signal{false};

    SamplingParams sp;
    sp.rep_penalty = 1.0f;  // match llama.cpp default; users can set via repetition_penalty in request
    Sampler sampler;
    sampler.init(sp, V);

    // Pre-build per-token byte strings + special-token flags for grammar mode.
    // Only built once at startup (V ≈ 250K * a few bytes ≈ 1.5 MB). The
    // sampler reads these by pointer when `response_format` is active.
    std::vector<std::string> tok_bytes(V);
    std::vector<uint8_t>     tok_special(V, 0);
    for (int i = 0; i < V; i++) {
        if (tok.is_special(i)) tok_special[i] = 1;
        else tok_bytes[i] = tok.decode_token(i);
    }
    sampler.tok_bytes = &tok_bytes;
    sampler.tok_special = &tok_special;

    // Unified generate. When `on_token` is provided, it's invoked once per
    // generated token id (in append order, including both tokens accepted by
    // a spec iter). The streaming wrapper below uses it to drive SSE chunks
    // so streaming benefits from the same spec decoding path as non-stream.
    // Slot-aware generate. `slot` selects the per-request KV+GDN partition.
    // Single-request server callers pass slot=0; the continuous-batching
    // scheduler passes the per-Sequence slot id. The Phase B scheduler holds
    // a global forward mutex around the entire call (see run_fn in serve_qwen)
    // so concurrent workers serialize GPU execution while keeping per-slot
    // KV+GDN state isolated. Phase C will replace that mutex with a true
    // batched_step that processes N slots in a single forward.
    // batched_handoff_first_tok / batched_handoff_slot_pos: when non-null,
    // generate_impl runs prefix-restore + chunked prefill + the single per-token
    // forward at step==prompt_len-1 (which samples the FIRST generated token,
    // pushes it to `generated`, and fires on_token for it), then breaks out of
    // the per-token loop. The caller picks up where this left off and drives
    // subsequent tokens via the batched gen-step path. *first_tok is the
    // sampled token, *slot_pos is the next position to write (= prompt_len).
    auto generate_impl = [&](const std::vector<int>& prompt_ids, int max_tokens,
                             int cached_prompt_tokens,
                             int* out_completion_tokens,
                             const std::function<void(int)>& on_token,
                             int slot = 0,
                             int* batched_handoff_first_tok = nullptr,
                             int* batched_handoff_slot_pos = nullptr,
                             std::atomic<bool>* cancelled = nullptr,
                             // Dynamic slot-router yield (RANK 1). When
                             // dyn_yield_tok is non-null AND router_yield_signal
                             // fires mid-generation, the MTP loop breaks at a
                             // step boundary and hands the continuation state
                             // back here: *dyn_yield_tok = last emitted token
                             // (the one forward_step_batched should embed next),
                             // *dyn_yield_pos = its KV write position, and the
                             // already-emitted tokens / think-state / output
                             // count via the dyn_yield_* sinks below. on_token
                             // has ALREADY fired for every token in dyn_yield_gen
                             // — the caller must NOT re-stream them. When the
                             // loop finishes normally these stay untouched
                             // (*dyn_yield_tok keeps its caller-set sentinel).
                             int* dyn_yield_tok = nullptr,
                             int* dyn_yield_pos = nullptr,
                             std::vector<int>* dyn_yield_gen = nullptr,
                             int* dyn_yield_output_tokens = nullptr,
                             bool* dyn_yield_in_think = nullptr) -> std::string {
        // Prefix cache: if caller asked for caching AND a snapshot matches
        // the requested prefix, restore state instead of resetting. Otherwise
        // do a normal full reset.
        // Automatic prefix caching: default ON. The client doesn't need to
        // send cached_prompt_tokens — we transparently reuse the slot's last
        // snapshot if it's still a bit-exact prefix of this prompt, and re-
        // snapshot at the largest chunk boundary each turn so the cache rides
        // the append-only chat history forward. PREFIX_CACHE_AUTO=0 disables.
        static const bool auto_cache = []{
            const char* e = getenv("PREFIX_CACHE_AUTO"); return !e || e[0] != '0';
        }();
        int prefix_skip = 0;
        if (cached_prompt_tokens > 0) {
            prefix_skip = model.try_restore_prefix_cache(prompt_ids, cached_prompt_tokens, slot);
            if (prefix_skip > 0) {
                printf("[CACHE slot=%d] hit: skipped %d tok of prefill\n", slot, prefix_skip);
                fflush(stdout);
            }
        } else if (auto_cache) {
            prefix_skip = model.try_restore_prefix_cache_auto(prompt_ids, slot);
            if (prefix_skip > 0) {
                printf("[CACHE slot=%d] auto hit: skipped %d tok of prefill\n", slot, prefix_skip);
                fflush(stdout);
            }
        }
        if (prefix_skip == 0) {
            // Reset only this slot's state. For slot 0 with num_slots=1
            // this is equivalent to reset_all_states (the previous default).
            model.reset_slot_states(slot);
        }
        // Chunk-aligned snapshot point for THIS request. Only save when the
        // chunked prefill crosses this exact boundary. Capped to prompt_len-1
        // (chunked prefill stops one short of the last prompt token). In auto
        // mode we target the largest boundary below the last token so the next
        // turn can reuse as much as possible.
        int eff_cached = (cached_prompt_tokens > 0)
            ? cached_prompt_tokens
            : (auto_cache ? (int)prompt_ids.size() : 0);
        int snapshot_target = (eff_cached > 0)
            ? (eff_cached / QwenModel::CHUNK_SIZE) * QwenModel::CHUNK_SIZE
            : 0;
        if (snapshot_target > (int)prompt_ids.size() - 1) {
            snapshot_target = ((int)prompt_ids.size() - 1) / QwenModel::CHUNK_SIZE * QwenModel::CHUNK_SIZE;
        }
        std::vector<int> generated;
        // No default cap on response length: if the caller doesn't pass
        // max_tokens we let the model run all the way to the end of the
        // KV context (minus a safety margin for the in-loop bound). It
        // will still stop on EOS or one of the other stop tags. Caller
        // can still set an explicit max_tokens to clip earlier.
        int slot_cap = model.slot_capacity(slot);
        int max_gen = max_tokens > 0
                      ? max_tokens
                      : std::max(64, slot_cap - (int)prompt_ids.size() - 64);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto t_first = t0;
        bool got_first_tok = false;
        bool in_think = false;
        int output_tokens = 0;
        // Reset MTP KV cache at the start of each generate() call so its
        // attention window matches the main model's per-request context.
        // MTP head's KV cache is a single global resource; only the slot-0
        // generate path invokes spec, so the reset/pending-draft state only
        // makes sense for slot 0.
        if (slot == 0) {
            if (mtp_loaded) mtp.reset_kv();
            mtp_pending_draft = -1;
            mtp_pending_draft_step = -1;
        }

        // ============ Phase 1: chunked prompt processing ============
        // Process prompt in CHUNK_SIZE token chunks. Skip the very last prompt
        // token so the existing per-token loop handles logits + sampling for it.
        int prompt_len = (int)prompt_ids.size();
        int prefill_len = prompt_len > 1 ? prompt_len - 1 : 0;
        int chunk_pos = prefix_skip;
        // Per-request running index over g_vision_embeds rows. Used by the
        // image_pad splice below; lives in this scope so it auto-resets to 0
        // for every new request.
        int s_vision_idx = 0;
        // If the cache hit covers the entire chunked-prefill range, skip
        // the loop entirely. Last token still goes through per-token loop.
        if (chunk_pos > prefill_len) chunk_pos = prefill_len;
        // PROFILE_PREFILL=1 env gates per-phase sync+timer. Adds sync overhead
        // so keep OFF for production. Phase totals printed at prefill end.
        const char* prof_env = getenv("PROFILE_PREFILL");
        const bool do_prof = prof_env && prof_env[0] == '1';
        const char* prof_attn_env = getenv("PROFILE_ATTN");
        g_profile_attn = prof_attn_env && prof_attn_env[0] == '1';
        // FlashAttention fused score+softmax+value is default ON for the
        // qwen3 hybrid shapes (HD=256, num_kv=4, num_q ∈ {16, 24}). The strict
        // block-per-score path (`attn_score_kernel_h_chunk_strict`) is bit-
        // exact with the per-token forward_attn, but FA gives ~1.8-2.7× prefill
        // throughput at every length we measured and matches greedy argmax on
        // real Korean coding prompts (memory: feedback_fa_correctness_greedy,
        // project_fa_kernel_verified). Set FLASH_ATTN=0 to opt back into the
        // strict path for regression / bit-exact bisecting.
        //
        // Note: at max_seq > 32 K the strict path is infeasible anyway —
        // attn_chunk_scores wants CHUNK_SIZE × num_q × kv_max_seq × 4 B which
        // exceeds VRAM on 16 GB GPUs (e.g. MTP_TQ=1 + --max-seq 262144 needs
        // 6.4 GB per GPU for that one buffer → cudaMalloc null → token-0
        // spam). FA path skips attn_chunk_scores entirely.
        const char* fa_env = getenv("FLASH_ATTN");
        g_use_flash_attn = (fa_env == nullptr) ? true : (fa_env[0] == '1');
        g_attn_score_ms = g_attn_softmax_ms = g_attn_value_ms
                       = g_attn_fused_ms  = g_attn_other_ms = 0.0;
        g_pt_qkvr_ms = g_pt_kvwrite_ms = g_pt_attn_ms = g_pt_oproj_ms = 0.0;
        g_pt_calls = 0;
        const char* prof_gdn_env = getenv("PROFILE_GDN");
        g_profile_gdn = prof_gdn_env && prof_gdn_env[0] == '1';
        const char* prof_mlp_env = getenv("PROFILE_MLP");
        g_profile_mlp = prof_mlp_env && prof_mlp_env[0] == '1';
        g_mlp_norm_ms = g_mlp_q1_ms = g_mlp_gate_ms = g_mlp_up_ms
                      = g_mlp_silu_ms = g_mlp_q2_ms = g_mlp_down_ms
                      = g_mlp_resi_ms = 0.0;
        g_mlp_calls = 0;
        g_gdn_norm_ms = g_gdn_proj_ms = g_gdn_conv_ms = g_gdn_recur_ms
                     = g_gdn_rmsg_ms = g_gdn_oproj_ms = g_gdn_resi_ms = 0.0;
        g_gdn_calls = 0;
        double t_embed = 0, t_xfer = 0, t_attn = 0, t_gdn = 0, t_mlp = 0;
        auto prof_now = [](){ return std::chrono::high_resolution_clock::now(); };
        auto prof_sync_ms = [&](std::chrono::high_resolution_clock::time_point t_beg, int dev) {
            cudaSetDevice(dev); cudaDeviceSynchronize();
            auto t_end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t_end - t_beg).count();
        };
        // PROFILE_GEN=1 drives the same sync+timer machinery for the per-token
        // loop. Separate totals for gen so prefill numbers stay clean. The
        // breakdown covers embed / xfer / attn / gdn / mlp / logits / mtp so
        // we can see which step is the next target.
        const char* gen_prof_env = getenv("PROFILE_GEN");
        const bool do_gen_prof = gen_prof_env && gen_prof_env[0] == '1';
        double g_embed = 0, g_xfer = 0, g_attn = 0, g_gdn = 0, g_mlp = 0,
               g_logits = 0, g_mtp = 0, g_other = 0;
        // DFlash tree-verify sub-phase timers (the untimed 27B 16-token forward).
        double g_tv_xfer = 0, g_tv_attn = 0, g_tv_gdn = 0, g_tv_mlp = 0, g_tv_cap = 0;
        int g_steps = 0, g_spec_iters = 0;
        // PROFILE_FA=1 zeroes the device-side fused-FA phase cycle counters
        // (g_fa_phase_cyc[0..4]: decode/score/softmax/value/tile_cnt) on each
        // GPU before gen and prints their sum after gen. The kernel always
        // updates them, but reading them is opt-in to keep noise low.
        const char* fa_prof_env = getenv("PROFILE_FA");
        const bool do_fa_prof = fa_prof_env && fa_prof_env[0] == '1';
        if (do_fa_prof) {
            unsigned long long zero[5] = {0,0,0,0,0};
            for (int g = 0; g < n_gpus; g++) {
                cudaSetDevice(g);
                cudaMemcpyToSymbol(g_fa_phase_cyc, zero, sizeof(zero));
            }
        }
        // DISABLE_CHUNKED_PREFILL=1 : chunked path 우회. per-token loop 가
        // 처음부터 prompt 전체를 처리 (batch=1, state update 명시적 누적).
        // 9B 에서 chunked GDN 의 fp16 누적 오차가 언어 classification 을
        // 넘겨버리는지 확인용. 맞으면 chunked kernel 수정 타깃.
        static const bool skip_chunked = getenv("DISABLE_CHUNKED_PREFILL") != nullptr;
        if (skip_chunked) prefill_len = 0;
        // DEBUG envs for bit-exact bisect. Each env forces the chunked
        // control flow to call the per-token forward_* in a loop over
        // n_tokens instead of the batched chunked kernel. Combine with
        // DUMP_LAYERS=1 to diff against DISABLE_CHUNKED_PREFILL=1 path.
        // ATTN chunked kernels: the score kernel now has a strict variant
        // (`attn_score_kernel_h_chunk_strict`, block-per-score with per-
        // token reduction tree) enabled by default. CHUNK_FORCE_PT_ATTN=1
        // still forces per-token (debug/bisect). CHUNK_ATTN_FAST=1 swaps
        // back to the old warp-per-score kernel (fastest but ~1% argmax
        // drift on Korean).
        static const bool force_pt_gdn  = getenv("CHUNK_FORCE_PT_GDN")  != nullptr;
        static const bool force_pt_attn = getenv("CHUNK_FORCE_PT_ATTN") != nullptr;
        static const bool force_pt_mlp  = getenv("CHUNK_FORCE_PT_MLP")  != nullptr;
        static const bool dump_layers_chunk = getenv("DUMP_LAYERS") != nullptr;
        static const bool dump_prefill_hash = getenv("DUMP_PREFILL_HASH") != nullptr;
        // Cross-GPU pipelining (PREFILL_NO_PIPELINE=1 to opt out). Requires:
        //   - ≥2 GPU segments (single-GPU is already optimal)
        //   - no debug env that needs in-loop sync
        //   - PROFILE_PREFILL=0 (the prof timers serialize each phase)
        static const bool no_pipeline_env = getenv("PREFILL_NO_PIPELINE") != nullptr;
        // PREFILL_CHUNKS_SERIAL=1 forces each chunk to wait for the
        // previous chunk's last segment to finish (disables cross-chunk
        // overlap, keeps cross-segment async). Bisect helper for the drift
        // we see at certain prompt lengths.
        static const bool chunks_serial = getenv("PREFILL_CHUNKS_SERIAL") != nullptr;
        const bool common_pipe_ok = (gpu_segs.size() >= 2)
            && !no_pipeline_env && !do_prof
            && !force_pt_gdn && !force_pt_attn && !force_pt_mlp
            && !dump_layers_chunk && !dump_prefill_hash;
        // v3 multi-threaded pipeline (PREFILL_PIPELINE_V3=1, opt-in): one CPU
        // launch thread per GPU segment. Measured root cause of the v2 ~61%
        // util: a single CPU thread serializes launches — heavy GEMM launches
        // BLOCK on the full per-stream queue (~97us vs 6us for a tiny kernel),
        // so the one thread can only feed one GPU at a time. Per-GPU threads
        // each block only on their own GPU's drain → all 3 GPUs run concurrently
        // → wall → max-stage (~GPU-bound). No snapshot support yet → falls back
        // to v2 when a snapshot is required.
        // Default ON (opt out with PREFILL_PIPELINE_V3=0 → falls back to the
        // single-thread v2 pipeline). Verified token-identical to v1/v2 on full
        // + prefix-cache-hit prefills; clean A/B 1.25x faster than v2 at 18K
        // (33.9s vs 42.4s, util 96% vs 79%) because per-GPU launch threads keep
        // all 3 GPUs saturated instead of one CPU thread serializing launches.
        static const bool v3_env = []{ const char* e=getenv("PREFILL_PIPELINE_V3"); return !e || e[0]!='0'; }();
        const bool use_pipeline_v3 = v3_env && alloc_pp_v2 && common_pipe_ok;
        // v2 deep pipeline: opt-in. Handles the prefix-cache snapshot via a
        // one-time drain barrier at snapshot_target (see below).
        const bool use_pipeline_v2 = alloc_pp_v2 && common_pipe_ok && !use_pipeline_v3;
        const bool use_pipeline = common_pipe_ok && !use_pipeline_v2 && !use_pipeline_v3;

        if (use_pipeline_v3) {
            // ── Multi-threaded chunked prefill (PREFILL_PIPELINE_V3) ───────────
            const int nseg = (int)gpu_segs.size();
            const int last_seg = nseg - 1;
            const int g0 = gpu_segs[0].g;
            const int NB = PP_NBUF;
            // Stage lambdas — each sets its own device, so they are correct when
            // invoked from the per-stage worker thread that owns that GPU.
            auto v3_embed = [&](int buf, int pos, int n){
                cudaSetDevice(g0);
                cudaStream_t s0 = prefill_comp_stream[g0];
                float* h_in = v2_gpu_hidden[g0][buf];
                for (int t = 0; t < n; t++) {
                    int token_id = prompt_ids[pos + t];
                    bool vision_hit = false;
                    if (g_vision_embeds && g_image_pad_id >= 0 && token_id == g_image_pad_id) {
                        if (s_vision_idx < g_vision_n_tokens && H == g_vision_H) {
                            cudaMemcpyAsync(gpu_hidden_half[0], g_vision_embeds + (size_t)s_vision_idx * H,
                                (size_t)H * sizeof(half), cudaMemcpyDeviceToDevice, s0);
                            s_vision_idx++; vision_hit = true;
                        }
                    }
                    if (!vision_hit) {
                        if (embd_t->type == GGML_TYPE_Q8_0)
                            dequant_embd_q8_0_row<<<(H+255)/256,256,0,s0>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
                        else if (embd_t->type == GGML_TYPE_Q5_K)
                            dequant_embd_q5k_row_v2<<<(H+255)/256,256,0,s0>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
                        else if (embd_t->type == GGML_TYPE_Q6_K)
                            dequant_embd_q6k_row_v2<<<(H+255)/256,256,0,s0>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
                    }
                    half_to_float_kernel<<<(H+255)/256,256,0,s0>>>(gpu_hidden_half[0], h_in + (size_t)t * H, H);
                }
            };
            auto v3_seg = [&](int s, int buf, int pos, int n){
                int g = gpu_segs[s].g;
                cudaSetDevice(g);
                cudaStream_t cs = prefill_comp_stream[g];
                float* h_chunk = v2_gpu_hidden[g][buf];
                for (int layer = gpu_segs[s].l_lo; layer < gpu_segs[s].l_hi; layer++) {
                    if (model.is_attn_layer(layer)) model.forward_attn_chunk(layer, h_chunk, pos, n, cs, slot);
                    else                            model.forward_gdn_chunk(layer, h_chunk, n, cs, slot);
                    if (model.layer_is_moe[layer]) model.forward_moe_chunk(layer, h_chunk, n, cs);
                    else                           model.forward_mlp_chunk(layer, h_chunk, n, cs);
                    model.dflash_capture_chunk(layer, h_chunk, pos, n, g, cs);
                }
                cudaEventRecord(v2_sd[s][buf], cs);
            };
            auto v3_d2h = [&](int s, int buf, int n){
                int g = gpu_segs[s].g;
                cudaSetDevice(g);
                cudaStreamWaitEvent(prefill_d2h_stream[g], v2_sd[s][buf], 0);
                cudaMemcpyAsync(v2_host_xfer[buf], v2_gpu_hidden[g][buf],
                    (size_t)n * H * sizeof(float), cudaMemcpyDeviceToHost, prefill_d2h_stream[g]);
                cudaEventRecord(v2_dh[s][buf], prefill_d2h_stream[g]);
            };
            // Buffer free-list + per-stage work queues.
            struct W { int buf, pos, n; };
            std::mutex mtx; std::condition_variable cv;
            std::deque<int> free_bufs; for (int b=0;b<NB;b++) free_bufs.push_back(b);
            std::vector<std::deque<W>> q(nseg);
            std::vector<bool> q_done(nseg, false);
            int v3_final_buf = -1, v3_final_n = 0;
            auto pv_t0 = std::chrono::high_resolution_clock::now();

            // Consumer threads for stages 1..last_seg.
            std::vector<std::thread> workers;
            for (int s = 1; s < nseg; s++) {
                workers.emplace_back([&, s]{
                    cudaSetDevice(gpu_segs[s].g);
                    for (;;) {
                        W w;
                        { std::unique_lock<std::mutex> lk(mtx);
                          cv.wait(lk, [&]{ return !q[s].empty() || q_done[s]; });
                          if (q[s].empty()) break;
                          w = q[s].front(); q[s].pop_front(); }
                        cudaEventSynchronize(v2_dh[s-1][w.buf]);  // host-coherent: prev D2H done
                        cudaMemcpyAsync(v2_gpu_hidden[gpu_segs[s].g][w.buf], v2_host_xfer[w.buf],
                            (size_t)w.n * H * sizeof(float), cudaMemcpyHostToDevice, prefill_comp_stream[gpu_segs[s].g]);
                        v3_seg(s, w.buf, w.pos, w.n);
                        if (s < last_seg) {
                            v3_d2h(s, w.buf, w.n);
                            { std::lock_guard<std::mutex> lk(mtx); q[s+1].push_back(w); } cv.notify_all();
                        } else {
                            cudaEventSynchronize(v2_sd[last_seg][w.buf]);
                            v3_final_buf = w.buf; v3_final_n = w.n;
                            { std::lock_guard<std::mutex> lk(mtx); free_bufs.push_back(w.buf); } cv.notify_all();
                        }
                    }
                    if (s < last_seg) { std::lock_guard<std::mutex> lk(mtx); q_done[s+1] = true; cv.notify_all(); }
                });
            }
            // Producer (this thread) = stage 0.
            cudaSetDevice(g0);
            int v3_pos = chunk_pos; bool v3_cancel = false;
            bool v3_snap_pending = (snapshot_target > prefix_skip);
            while (v3_pos < prefill_len) {
                if (cancelled && cancelled->load(std::memory_order_relaxed)) { v3_cancel = true; break; }
                // Snapshot drain barrier: when all tokens < snapshot_target are
                // admitted, wait for the pipeline to fully drain (every buffer
                // back in the free-list = all in-flight chunks done), save a
                // consistent KV/GDN snapshot, then resume. One-time, ~= tail.
                if (v3_snap_pending && v3_pos >= snapshot_target) {
                    { std::unique_lock<std::mutex> lk(mtx);
                      cv.wait(lk, [&]{ return (int)free_bufs.size() == NB; }); }
                    for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaStreamSynchronize(prefill_comp_stream[g]); }
                    model.save_prefix_snapshot(prompt_ids, snapshot_target, slot);
                    printf("[CACHE] snapshot saved at pos=%d (v3)\n", snapshot_target);
                    fflush(stdout);
                    cudaSetDevice(g0);
                    v3_snap_pending = false;
                }
                int n = std::min(CHUNK_SIZE, prefill_len - v3_pos);
                int buf;
                { std::unique_lock<std::mutex> lk(mtx);
                  cv.wait(lk, [&]{ return !free_bufs.empty(); });
                  buf = free_bufs.front(); free_bufs.pop_front(); }
                v3_embed(buf, v3_pos, n);
                v3_seg(0, buf, v3_pos, n);
                v3_d2h(0, buf, n);
                { std::lock_guard<std::mutex> lk(mtx); q[1].push_back({buf, v3_pos, n}); } cv.notify_all();
                v3_pos += n;
            }
            { std::lock_guard<std::mutex> lk(mtx); q_done[1] = true; cv.notify_all(); }
            for (auto& t : workers) t.join();
            for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaDeviceSynchronize(); }
            {
                double wall = std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-pv_t0).count();
                printf("[PIPE_V3] wall=%.1fms (multi-thread, %d stages, NB=%d)\n", wall, nseg, NB); fflush(stdout);
            }
            if (v3_cancel) {
                if (out_completion_tokens) *out_completion_tokens = (int)generated.size();
                return tok.decode(generated);
            }
            if (v3_final_buf >= 0 && v3_final_n > 0) {
                int g = gpu_model.layer_gpu[model.cfg.num_layers - 1];
                cudaSetDevice(g);
                cudaMemcpy(gpu_hidden_chunk[g], v2_gpu_hidden[g][v3_final_buf],
                           (size_t)v3_final_n * H * sizeof(float), cudaMemcpyDeviceToDevice);
            }
            cudaSetDevice(last_gpu);
        } else if (use_pipeline_v2) {
            // ── Deep event-driven chunked prefill (PREFILL_PIPELINE_V2) ────────
            const int NB = PP_NBUF;
            const int nseg = (int)gpu_segs.size();
            const int last_seg = nseg - 1;
            const int g0 = gpu_segs[0].g;
            // Embed n tokens of a chunk onto buffer `buf` of GPU0, on GPU0's
            // compute stream (FIFO-serialized vs other chunks' embeds, so the
            // shared gpu_hidden_half[0] scratch is reused safely).
            auto launch_embed = [&](int buf, int pos, int n){
                cudaSetDevice(g0);
                cudaStream_t s0 = prefill_comp_stream[g0];
                float* h_in = v2_gpu_hidden[g0][buf];
                for (int t = 0; t < n; t++) {
                    int token_id = prompt_ids[pos + t];
                    bool vision_hit = false;
                    if (g_vision_embeds && g_image_pad_id >= 0 && token_id == g_image_pad_id) {
                        if (s_vision_idx < g_vision_n_tokens && H == g_vision_H) {
                            cudaMemcpyAsync(gpu_hidden_half[0],
                                g_vision_embeds + (size_t)s_vision_idx * H,
                                (size_t)H * sizeof(half), cudaMemcpyDeviceToDevice, s0);
                            s_vision_idx++; vision_hit = true;
                        }
                    }
                    if (!vision_hit) {
                        if (embd_t->type == GGML_TYPE_Q8_0)
                            dequant_embd_q8_0_row<<<(H+255)/256,256,0,s0>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
                        else if (embd_t->type == GGML_TYPE_Q5_K)
                            dequant_embd_q5k_row_v2<<<(H+255)/256,256,0,s0>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
                        else if (embd_t->type == GGML_TYPE_Q6_K)
                            dequant_embd_q6k_row_v2<<<(H+255)/256,256,0,s0>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
                    }
                    half_to_float_kernel<<<(H+255)/256,256,0,s0>>>(gpu_hidden_half[0], h_in + (size_t)t * H, H);
                }
            };
            // Run segment s's layers for chunk (buf,pos,n) on its GPU's compute
            // stream; record seg-done.
            auto launch_seg = [&](int s, int buf, int pos, int n){
                int g = gpu_segs[s].g;
                cudaSetDevice(g);
                cudaStream_t cs = prefill_comp_stream[g];
                float* h_chunk = v2_gpu_hidden[g][buf];
                for (int layer = gpu_segs[s].l_lo; layer < gpu_segs[s].l_hi; layer++) {
                    if (model.is_attn_layer(layer)) model.forward_attn_chunk(layer, h_chunk, pos, n, cs, slot);
                    else                            model.forward_gdn_chunk(layer, h_chunk, n, cs, slot);
                    if (model.layer_is_moe[layer]) model.forward_moe_chunk(layer, h_chunk, n, cs);
                    else                           model.forward_mlp_chunk(layer, h_chunk, n, cs);
                    model.dflash_capture_chunk(layer, h_chunk, pos, n, g, cs);
                }
                cudaEventRecord(v2_sd[s][buf], cs);
            };
            // Issue the D2H at the end of segment s (depends seg-done[s]).
            auto launch_d2h = [&](int s, int buf, int n){
                int g = gpu_segs[s].g;
                cudaSetDevice(g);
                cudaStreamWaitEvent(prefill_d2h_stream[g], v2_sd[s][buf], 0);
                cudaMemcpyAsync(v2_host_xfer[buf], v2_gpu_hidden[g][buf],
                    (size_t)n * H * sizeof(float), cudaMemcpyDeviceToHost, prefill_d2h_stream[g]);
                cudaEventRecord(v2_dh[s][buf], prefill_d2h_stream[g]);
            };
            // After D2H[s] is host-complete: H2D into seg s+1's GPU, then launch
            // seg s+1 (gated on H2D).
            auto launch_h2d_and_next = [&](int s, int buf, int pos, int n){
                int gnext = gpu_segs[s+1].g;
                cudaSetDevice(gnext);
                cudaMemcpyAsync(v2_gpu_hidden[gnext][buf], v2_host_xfer[buf],
                    (size_t)n * H * sizeof(float), cudaMemcpyHostToDevice, prefill_h2d_stream[gnext]);
                cudaEventRecord(v2_hd[s][buf], prefill_h2d_stream[gnext]);
                cudaStreamWaitEvent(prefill_comp_stream[gnext], v2_hd[s][buf], 0);
                launch_seg(s+1, buf, pos, n);
                if (s+1 < last_seg) launch_d2h(s+1, buf, n);
            };

            struct IFC { int buf, pos, n, stage; };  // stage: 2*s=awaiting seg_done[s]; 2*s+1=polling d2h[s]
            std::vector<IFC> inflight;
            bool buf_busy[PP_NBUF] = {};
            int v2_pos = chunk_pos, v2_last_buf = -1, v2_last_n = 0;  // resume from prefix-cache hit (chunk_pos = prefix_skip)
            bool v2_cancelled = false;
            // Prefix-cache snapshot: save a consistent KV/GDN state at
            // snapshot_target. v2 enforces a one-time drain barrier there —
            // admit only up to snapshot_target, let the pipeline empty, save,
            // then resume. snapshot_target is chunk-aligned and ~= prompt end,
            // so the barrier costs one pipeline-depth bubble near the tail.
            bool snap_pending = (snapshot_target > prefix_skip);
            int admit_limit = snap_pending ? snapshot_target : prefill_len;

            // PREFILL_PIPELINE_V2_PROF=1 : measure where the wall-time goes —
            // CPU launch (admit+advance kernel launches) vs poll-spin (GPU idle
            // waiting on events). Localizes the ~39% idle (CPU-bound vs GPU dep).
            static const bool prof_v2 = getenv("PREFILL_PIPELINE_V2_PROF") != nullptr;
            auto pv_now = []{ return std::chrono::high_resolution_clock::now(); };
            auto pv_t0 = pv_now();
            double pv_launch_ms = 0; long pv_iters = 0;

            while (v2_pos < prefill_len || !inflight.empty()) {
                pv_iters++;
                if (cancelled && cancelled->load(std::memory_order_relaxed)) { v2_cancelled = true; break; }
                // Snapshot drain barrier: all tokens < snapshot_target fully
                // processed (pipeline empty) → save, then lift the limit.
                if (snap_pending && v2_pos >= admit_limit && inflight.empty()) {
                    for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaStreamSynchronize(prefill_comp_stream[g]); }
                    model.save_prefix_snapshot(prompt_ids, snapshot_target, slot);
                    printf("[CACHE] snapshot saved at pos=%d (v2)\n", snapshot_target);
                    fflush(stdout);
                    snap_pending = false; admit_limit = prefill_len;
                }
                // 1) Admit new chunks into any free buffer.
                for (int b = 0; b < NB && v2_pos < admit_limit; b++) {
                    if (buf_busy[b]) continue;
                    int n = std::min(CHUNK_SIZE, prefill_len - v2_pos);
                    buf_busy[b] = true;
                    auto _la = pv_now();
                    launch_embed(b, v2_pos, n);
                    launch_seg(0, b, v2_pos, n);
                    if (nseg > 1) { launch_d2h(0, b, n); inflight.push_back({b, v2_pos, n, 1}); }
                    else          { inflight.push_back({b, v2_pos, n, 2*last_seg}); }
                    pv_launch_ms += std::chrono::duration<double,std::milli>(pv_now()-_la).count();
                    v2_last_buf = b; v2_last_n = n;
                    v2_pos += n;
                }
                // 2) Advance any in-flight chunk whose pending event is ready.
                bool progressed = false;
                for (size_t i = 0; i < inflight.size(); ) {
                    IFC& c = inflight[i];
                    int s = c.stage >> 1;
                    bool polling_d2h = c.stage & 1;
                    bool done = false, advanced = false;
                    if (polling_d2h) {
                        if (cudaEventQuery(v2_dh[s][c.buf]) == cudaSuccess) {
                            auto _la = pv_now();
                            launch_h2d_and_next(s, c.buf, c.pos, c.n);
                            pv_launch_ms += std::chrono::duration<double,std::milli>(pv_now()-_la).count();
                            c.stage = (s+1 < last_seg) ? (2*(s+1)+1) : (2*last_seg);
                            advanced = true;
                        }
                    } else { // awaiting seg_done[s]; only the last segment lands here
                        if (cudaEventQuery(v2_sd[s][c.buf]) == cudaSuccess) {
                            buf_busy[c.buf] = false; done = true; advanced = true;
                        }
                    }
                    if (advanced) progressed = true;
                    if (done) inflight.erase(inflight.begin() + i);
                    else      i++;
                }
                (void)progressed;  // busy-poll; events fire fast and we mostly have work to launch
            }
            if (prof_v2) {
                double wall = std::chrono::duration<double,std::milli>(pv_now()-pv_t0).count();
                printf("[PIPE_V2 PROF] wall=%.1fms  cpu_launch=%.1fms (%.1f%%)  poll/idle=%.1fms (%.1f%%)  iters=%ld\n",
                       wall, pv_launch_ms, 100.0*pv_launch_ms/wall,
                       wall-pv_launch_ms, 100.0*(wall-pv_launch_ms)/wall, pv_iters);
                fflush(stdout);
            }
            // Drain all pipeline streams before the per-token loop reads KV/state.
            for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaDeviceSynchronize(); }
            if (v2_cancelled) {
                if (out_completion_tokens) *out_completion_tokens = (int)generated.size();
                return tok.decode(generated);
            }
            // Hand the final chunk's hidden to where the per-token loop expects it.
            if (v2_last_buf >= 0 && v2_last_n > 0) {
                int g = gpu_model.layer_gpu[model.cfg.num_layers - 1];
                cudaSetDevice(g);
                cudaMemcpy(gpu_hidden_chunk[g], v2_gpu_hidden[g][v2_last_buf],
                           (size_t)v2_last_n * H * sizeof(float), cudaMemcpyDeviceToDevice);
            }
            cudaSetDevice(last_gpu);
        } else if (use_pipeline) {
            // ── Pipelined chunked prefill ──────────────────────────────────
            // Per-segment compute streams + per-GPU D2H/H2D streams + double-
            // buffered (host transfer, per-GPU hidden chunk). chunk i's D2H
            // and H2D run while chunk i+1's GPU0 segment is computing on the
            // other buffer; chunk i+2 then reuses chunk i's buffer once the
            // last segment has finished consuming it.
            int chunk_idx = 0;
            int last_buf = -1;
            int last_chunk_n = 0;
            const int last_seg = (int)gpu_segs.size() - 1;
            while (chunk_pos < prefill_len) {
                if (cancelled && cancelled->load(std::memory_order_relaxed)) {
                    for (auto& seg : gpu_segs) {
                        cudaSetDevice(seg.g);
                        cudaStreamSynchronize(prefill_comp_stream[seg.g]);
                    }
                    if (out_completion_tokens) *out_completion_tokens = (int)generated.size();
                    return tok.decode(generated);
                }
                int chunk_n = std::min(CHUNK_SIZE, prefill_len - chunk_pos);
                int buf = chunk_idx & 1;

                // Reusing host_xfer[buf] / gpu_hidden_chunk_db[*][buf]: wait
                // for the last segment of the previous tenant of this buffer
                // to be done consuming it.
                if (chunk_idx >= 2) {
                    cudaSetDevice(gpu_segs[0].g);
                    cudaStreamWaitEvent(prefill_comp_stream[gpu_segs[0].g],
                                        pp_seg_done[last_seg][buf], 0);
                }
                // Bisect: force serial chunks (waits for previous chunk's
                // last segment, not just chunk_idx-2's). Disables cross-
                // chunk overlap entirely.
                if (chunks_serial && chunk_idx >= 1) {
                    int prev_buf = (chunk_idx - 1) & 1;
                    cudaSetDevice(gpu_segs[0].g);
                    cudaStreamWaitEvent(prefill_comp_stream[gpu_segs[0].g],
                                        pp_seg_done[last_seg][prev_buf], 0);
                }

                // 1. Embed onto buf-side hidden of segment 0's GPU.
                cudaSetDevice(gpu_segs[0].g);
                float* h_in = gpu_hidden_chunk_db[gpu_segs[0].g][buf];
                cudaStream_t s0 = prefill_comp_stream[gpu_segs[0].g];
                for (int t = 0; t < chunk_n; t++) {
                    int token_id = prompt_ids[chunk_pos + t];
                    bool vision_hit = false;
                    if (g_vision_embeds && g_image_pad_id >= 0
                        && token_id == g_image_pad_id) {
                        if (s_vision_idx < g_vision_n_tokens && H == g_vision_H) {
                            cudaMemcpyAsync(gpu_hidden_half[0],
                                            g_vision_embeds + (size_t)s_vision_idx * H,
                                            (size_t)H * sizeof(half),
                                            cudaMemcpyDeviceToDevice, s0);
                            s_vision_idx++;
                            vision_hit = true;
                        }
                    }
                    if (!vision_hit) {
                        if (embd_t->type == GGML_TYPE_Q8_0)
                            dequant_embd_q8_0_row<<<(H+255)/256, 256, 0, s0>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
                        else if (embd_t->type == GGML_TYPE_Q5_K)
                            dequant_embd_q5k_row_v2<<<(H+255)/256, 256, 0, s0>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
                        else if (embd_t->type == GGML_TYPE_Q6_K)
                            dequant_embd_q6k_row_v2<<<(H+255)/256, 256, 0, s0>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
                    }
                    half_to_float_kernel<<<(H+255)/256, 256, 0, s0>>>(
                        gpu_hidden_half[0], h_in + (size_t)t * H, H);
                }

                // 2. For each segment: compute its layers on its compute
                //    stream, record done-event, then chain D2H + H2D into
                //    the next segment's input buffer.
                for (int s = 0; s < (int)gpu_segs.size(); s++) {
                    int g = gpu_segs[s].g;
                    cudaSetDevice(g);
                    cudaStream_t cs = prefill_comp_stream[g];
                    float* h_chunk = gpu_hidden_chunk_db[g][buf];
                    for (int layer = gpu_segs[s].l_lo; layer < gpu_segs[s].l_hi; layer++) {
                        bool is_attn = model.is_attn_layer(layer);
                        if (is_attn) {
                            model.forward_attn_chunk(layer, h_chunk, chunk_pos, chunk_n, cs, slot);
                        } else {
                            model.forward_gdn_chunk(layer, h_chunk, chunk_n, cs, slot);
                        }
                        if (model.layer_is_moe[layer]) model.forward_moe_chunk(layer, h_chunk, chunk_n, cs);
                        else                           model.forward_mlp_chunk(layer, h_chunk, chunk_n, cs);
                        model.dflash_capture_chunk(layer, h_chunk, chunk_pos, chunk_n, g, cs);
                    }
                    cudaEventRecord(pp_seg_done[s][buf], cs);
                    if (s < last_seg) {
                        cudaStream_t d2h_s = prefill_d2h_stream[g];
                        cudaStreamWaitEvent(d2h_s, pp_seg_done[s][buf], 0);
                        cudaMemcpyAsync(host_chunk_xfer_db[buf], h_chunk,
                                        (size_t)chunk_n * H * sizeof(float),
                                        cudaMemcpyDeviceToHost, d2h_s);
                        cudaEventRecord(pp_d2h_done[s][buf], d2h_s);
                        int g_next = gpu_segs[s + 1].g;
                        cudaSetDevice(g_next);
                        cudaStream_t h2d_s = prefill_h2d_stream[g_next];
                        // Cross-device cudaStreamWaitEvent does NOT fence
                        // pinned host memory between the source GPU's D2H
                        // and the destination GPU's H2D on CMP 100-210 (or
                        // any PCIe 1.0 / no-P2P setup we tested). The H2D
                        // reads stale host bytes and sampled tokens diverge
                        // from the sequential path on certain prompt
                        // lengths. cudaEventSynchronize is the cheapest
                        // available host-side fence and chunks-internal
                        // overlap is already serialized via stream FIFOs,
                        // so the perf cost is ≤3% (measured at 18 K). Set
                        // PREFILL_NO_HOST_FENCE=1 to revert to the (racy)
                        // event-only path for benchmarking.
                        static const bool no_host_fence = getenv("PREFILL_NO_HOST_FENCE") != nullptr;
                        if (no_host_fence) {
                            cudaStreamWaitEvent(h2d_s, pp_d2h_done[s][buf], 0);
                        } else {
                            cudaEventSynchronize(pp_d2h_done[s][buf]);
                        }
                        cudaMemcpyAsync(gpu_hidden_chunk_db[g_next][buf],
                                        host_chunk_xfer_db[buf],
                                        (size_t)chunk_n * H * sizeof(float),
                                        cudaMemcpyHostToDevice, h2d_s);
                        cudaEventRecord(pp_h2d_done[s][buf], h2d_s);
                        cudaStreamWaitEvent(prefill_comp_stream[g_next],
                                            pp_h2d_done[s][buf], 0);
                    }
                }

                last_buf = buf;
                last_chunk_n = chunk_n;
                chunk_idx++;
                chunk_pos += chunk_n;

                // Snapshot needs all in-flight pipeline stages to be done so
                // the saved KV state is consistent. Drain & restart cycle.
                // Save when we cross the target boundary AND it extends past
                // whatever prefix we restored (snapshot_target > prefix_skip),
                // so an auto-cache hit still advances the snapshot forward.
                if (snapshot_target > prefix_skip
                    && chunk_pos == snapshot_target) {
                    for (auto& seg : gpu_segs) {
                        cudaSetDevice(seg.g);
                        cudaStreamSynchronize(prefill_comp_stream[seg.g]);
                    }
                    model.save_prefix_snapshot(prompt_ids, snapshot_target, slot);
                    printf("[CACHE] snapshot saved at pos=%d\n", snapshot_target);
                    fflush(stdout);
                    chunk_idx = 0;
                }
            }
            // Drain pipeline before the per-token loop reads the final
            // hidden / KV cache. cudaDeviceSynchronize on every GPU drains
            // ALL streams (compute + d2h + h2d) so the per-token forward
            // (which uses the default stream) sees consistent KV/GDN state.
            for (int g = 0; g < n_gpus; g++) {
                cudaSetDevice(g);
                cudaDeviceSynchronize();
            }
            // Per-token loop reads from gpu_hidden_chunk[last_gpu] (slot 0 of
            // the double-buffer); copy back when last chunk landed in slot 1.
            if (last_buf == 1 && last_chunk_n > 0) {
                int g = gpu_model.layer_gpu[model.cfg.num_layers - 1];
                cudaSetDevice(g);
                cudaMemcpy(gpu_hidden_chunk[g],
                           gpu_hidden_chunk_db[g][1],
                           (size_t)last_chunk_n * H * sizeof(float),
                           cudaMemcpyDeviceToDevice);
            }
            cudaSetDevice(last_gpu);
        } else {
            // ── Sequential chunked prefill (legacy, debug, single-GPU) ───
            while (chunk_pos < prefill_len) {
                if (cancelled && cancelled->load(std::memory_order_relaxed)) {
                    if (out_completion_tokens) *out_completion_tokens = (int)generated.size();
                    return tok.decode(generated);
                }
                int chunk_n = std::min(CHUNK_SIZE, prefill_len - chunk_pos);

                cudaSetDevice(0);
                auto te0 = prof_now();
                for (int t = 0; t < chunk_n; t++) {
                    int token_id = prompt_ids[chunk_pos + t];
                    bool vision_hit = false;
                    if (g_vision_embeds && g_image_pad_id >= 0 && token_id == g_image_pad_id) {
                        if (s_vision_idx < g_vision_n_tokens && H == g_vision_H) {
                            cudaMemcpyAsync(gpu_hidden_half[0],
                                            g_vision_embeds + (size_t)s_vision_idx * H,
                                            (size_t)H * sizeof(half),
                                            cudaMemcpyDeviceToDevice);
                            s_vision_idx++;
                            vision_hit = true;
                        }
                    }
                    if (!vision_hit) {
                        if (embd_t->type == GGML_TYPE_Q8_0)
                            dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
                        else if (embd_t->type == GGML_TYPE_Q5_K)
                            dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
                        else if (embd_t->type == GGML_TYPE_Q6_K)
                            dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
                    }
                    half_to_float_kernel<<<(H+255)/256, 256>>>(
                        gpu_hidden_half[0], gpu_hidden_chunk[0] + (size_t)t * H, H);
                }
                if (do_prof) t_embed += prof_sync_ms(te0, 0);

                float* h_chunk = gpu_hidden_chunk[0];
                for (int layer = 0; layer < model.cfg.num_layers; layer++) {
                    int g = gpu_model.layer_gpu[layer];
                    int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer - 1];
                    if (g != prev_g) {
                        auto tx0 = prof_now();
                        cudaSetDevice(prev_g);
                        cudaMemcpy(host_chunk_transfer, h_chunk, (size_t)chunk_n * H * sizeof(float), cudaMemcpyDeviceToHost);
                        cudaSetDevice(g);
                        cudaMemcpy(gpu_hidden_chunk[g], host_chunk_transfer, (size_t)chunk_n * H * sizeof(float), cudaMemcpyHostToDevice);
                        h_chunk = gpu_hidden_chunk[g];
                        if (do_prof) t_xfer += prof_sync_ms(tx0, g);
                    } else { cudaSetDevice(g); }

                    auto ta0 = prof_now();
                    bool is_attn = model.is_attn_layer(layer);
                    if (is_attn) {
                        if (force_pt_attn) {
                            for (int tt = 0; tt < chunk_n; tt++) {
                                float* h_t = h_chunk + (size_t)tt * H;
                                model.forward_attn(layer, h_t, chunk_pos + tt, 0,
                                                   /*external_proj=*/false, /*slot_pos=*/-1,
                                                   /*mask_start=*/-1, /*mask_len=*/0,
                                                   /*mask_bits=*/0xffffffffu, slot);
                            }
                        } else {
                            model.forward_attn_chunk(layer, h_chunk, chunk_pos, chunk_n, 0, slot);
                        }
                    } else {
                        if (force_pt_gdn) {
                            for (int tt = 0; tt < chunk_n; tt++) {
                                float* h_t = h_chunk + (size_t)tt * H;
                                model.forward_gdn(layer, h_t, 0, slot);
                            }
                        } else {
                            model.forward_gdn_chunk(layer, h_chunk, chunk_n, 0, slot);
                        }
                    }
                    if (do_prof) {
                        double ms = prof_sync_ms(ta0, g);
                        if (is_attn) t_attn += ms; else t_gdn += ms;
                    }

                    auto tm0 = prof_now();
                    if (model.layer_is_moe[layer]) {
                        model.forward_moe_chunk(layer, h_chunk, chunk_n, 0);
                    } else if (force_pt_mlp) {
                        for (int tt = 0; tt < chunk_n; tt++) {
                            float* h_t = h_chunk + (size_t)tt * H;
                            model.forward_mlp(layer, h_t, 0);
                        }
                    } else {
                        model.forward_mlp_chunk(layer, h_chunk, chunk_n, 0);
                    }
                    if (do_prof) t_mlp += prof_sync_ms(tm0, g);

                    model.dflash_capture_chunk(layer, h_chunk, chunk_pos, chunk_n, g, 0);

                    if (dump_layers_chunk && chunk_pos + chunk_n == prefill_len) {
                        cudaSetDevice(g); cudaDeviceSynchronize();
                        float sample[8];
                        float* last_h = h_chunk + (size_t)(chunk_n - 1) * H;
                        cudaMemcpy(sample, last_h, 8 * sizeof(float), cudaMemcpyDeviceToHost);
                        fprintf(stderr, "[L%02d %s] %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
                                layer, is_attn ? "attn" : "gdn ",
                                sample[0], sample[1], sample[2], sample[3],
                                sample[4], sample[5], sample[6], sample[7]);
                        fflush(stderr);
                    }
                    if (dump_prefill_hash) {
                        cudaSetDevice(g); cudaDeviceSynchronize();
                        static thread_local std::vector<float> host_buf;
                        host_buf.resize((size_t)chunk_n * H);
                        cudaMemcpy(host_buf.data(), h_chunk,
                                   (size_t)chunk_n * H * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        for (int t = 0; t < chunk_n; t++) {
                            uint64_t h = 0xcbf29ce484222325ULL;
                            const float* p = host_buf.data() + (size_t)t * H;
                            for (int i = 0; i < H; i++) {
                                uint32_t b; memcpy(&b, &p[i], 4);
                                h = (h ^ (uint64_t)b) * 0x100000001b3ULL;
                            }
                            fprintf(stderr, "[CHK L%02d t%03d %s] hash=%016lx\n",
                                    layer, chunk_pos + t,
                                    is_attn ? "attn" : "gdn ", h);
                        }
                        fflush(stderr);
                    }
                }
                cudaSetDevice(last_gpu); cudaDeviceSynchronize();

                chunk_pos += chunk_n;

                if (snapshot_target > prefix_skip
                    && chunk_pos == snapshot_target) {
                    model.save_prefix_snapshot(prompt_ids, snapshot_target, slot);
                    printf("[CACHE] snapshot saved at pos=%d\n", snapshot_target);
                    fflush(stdout);
                }
            }
        }
        if (do_prof) {
            double total = t_embed + t_xfer + t_attn + t_gdn + t_mlp;
            printf("[PREFILL PROF] prompt=%d tok  total=%.1fms  embed=%.1f(%.0f%%) xfer=%.1f(%.0f%%) attn=%.1f(%.0f%%) gdn=%.1f(%.0f%%) mlp=%.1f(%.0f%%)\n",
                   prefill_len, total,
                   t_embed, 100.0*t_embed/total, t_xfer, 100.0*t_xfer/total,
                   t_attn, 100.0*t_attn/total, t_gdn, 100.0*t_gdn/total,
                   t_mlp, 100.0*t_mlp/total);
            fflush(stdout);
        }
        if (g_profile_attn) {
            double sub = g_attn_score_ms + g_attn_softmax_ms + g_attn_value_ms;
            printf("[ATTN PROF]  score=%.1fms(%.0f%%) softmax=%.1fms(%.0f%%) value=%.1fms(%.0f%%)  sub_sum=%.1fms  fused=%.1fms\n",
                   g_attn_score_ms, sub > 0 ? 100.0*g_attn_score_ms/sub : 0,
                   g_attn_softmax_ms, sub > 0 ? 100.0*g_attn_softmax_ms/sub : 0,
                   g_attn_value_ms, sub > 0 ? 100.0*g_attn_value_ms/sub : 0,
                   sub, g_attn_fused_ms);
            fflush(stdout);
        }

        // ============ Phase 2: per-token loop (handles last prompt token + generation) ============
        // The +4096 used to be a "think buffer" so reasoning chains
        // weren't counted against max_gen, but for unlimited responses
        // we just cap at the actual KV context capacity instead.
        //
        // emit_tok: single-source-of-truth for "append a token to the
        // client-visible stream". Stop markers (<|im_end|>, <|endoftext|>,
        // <|im_start|>) are *not* emitted — we just signal stop by returning
        // true so the outer loop breaks before the marker reaches the
        // transcript. Also handles the <think>/</think> bookkeeping and
        // max_gen cap.
        auto emit_tok = [&](int tok) -> bool {
            if (tok == 248046 || tok == 248044 || tok == 248045) return true;
            generated.push_back(tok);
            if (on_token) on_token(tok);
            if (tok == 248068) in_think = true;
            if (tok == 248069) in_think = false;
            if (!in_think) output_tokens++;
            return output_tokens >= max_gen;
        };
        int step_cap = std::min((int)prompt_ids.size() + max_gen + 4096, max_seq);
        int dyn_poll_ctr = 0;   // throttle the router-yield atomic poll
        for (int step = prefill_len; step < step_cap; step++) {
            if (cancelled && cancelled->load(std::memory_order_relaxed)) break;

            // ── Dynamic slot-router cooperative yield (RANK 1) ─────────────
            // Poll the yield signal at the TOP of a step iter, BEFORE any
            // forward for `step` runs. Here slot-0's live GDN + KV state is
            // fully consistent with the last emitted token (every spec
            // accept/reject already applied its own rollback at the end of
            // the previous iter), and `generated.back()` is the token whose
            // forward would write KV at position `step`. That is exactly the
            // (next_tok, slot_pos) contract forward_step_batched expects, so
            // we can hand off WITHOUT touching GDN state (NO restore_gdn_states
            // — the advanced state IS what the batched path consumes). We only
            // yield once we are past prefill (step >= prompt_len, so `generated`
            // is non-empty and holds the token to embed next).
            // Poll EVERY iteration (relaxed atomic load ~1ns). The earlier
            // "every 4 iters" throttle delayed the yield by ~8-12 tokens under
            // MTP K=2 (2-3 tokens/iter), leaving slot-1 contending and dropping
            // 2-concurrent aggregate to ~19 t/s; polling each step yields within
            // one iter so K=2 recovers to the batched ~28.8. dyn_poll_ctr kept
            // (unused throttle removed) for clarity.
            (void)dyn_poll_ctr;
            if (dyn_yield_tok && dynamic_router_on
                && step >= (int)prompt_ids.size()
                && router_yield_signal.load(std::memory_order_relaxed)) {
                *dyn_yield_tok = generated.back();          // token at logical pos `step`
                if (dyn_yield_pos)           *dyn_yield_pos = step;
                if (dyn_yield_gen)           *dyn_yield_gen = generated;
                if (dyn_yield_output_tokens) *dyn_yield_output_tokens = output_tokens;
                if (dyn_yield_in_think)      *dyn_yield_in_think = in_think;
                break;
            }

            int token_id = step < (int)prompt_ids.size() ? prompt_ids[step] : generated.back();

            // ===================== DFlash PIPELINED-FOLD (opt-in) =====================
            // Removes the standalone per-token forward of the just-committed token by
            // making it tree-verify SLOT 0: slot0 (anchor_tok=generated.back()@step) is
            // forwarded inside the batched verify, capturing its own C[step] and yielding
            // posterior[0] = the 27B greedy at step+1 (the max_idx the per-token forward
            // used to produce). Draft conditions on C[0..step-1] (ctx_len_draft=step) since
            // the bonus's C isn't available pre-verify -> needs the ctx-drop1 fold drafter.
            // Lossless: emit = 27B greedy posterior[0..accept_drafts]. Byte-identical to the
            // non-fold path's output for ANY drafter (only AL/speed differ) -> logic-checkable.
            // Pipelined-fold default ON (opt out with DFLASH_FOLD=0). +15-22% t/s,
            // quality-equivalent (verified: identical C++ coding pass-rate + reasoning
            // accuracy vs the non-fold path) but NOT bit-identical — the anchor is
            // forwarded via the batched tree-verify kernel instead of the single-token
            // kernel, so near-tie argmax can differ. DFLASH_FOLD=0 = bit-exact (slower).
            static const bool dflash_fold_enabled = !getenv("DFLASH_FOLD") || atoi(getenv("DFLASH_FOLD")) != 0;
            if (dflash_fold_enabled && slot == 0 && dflash_enabled
                && sampler.grammar == nullptr && step >= (int)prompt_ids.size()) {
                auto df0 = prof_now();
                int B = dflash::DraftConfig::block_size;       // 16
                int ctx_len_draft = step;                      // C[0..step-1] (NO bonus C)
                int anchor_tok = token_id;                     // committed bonus @ position step

                // (1) noise = [embed(anchor_tok), embed(MASK)*15]
                cudaSetDevice(0);
                half* noise = dflash_state.d_noise_embed;
                auto fold_embed_row = [&](half* dst, int tok) {
                    if (embd_t->type == GGML_TYPE_Q8_0)      dequant_embd_q8_0_row<<<(H+255)/256,256>>>(embd_t->data, dst, tok, H);
                    else if (embd_t->type == GGML_TYPE_Q5_K) dequant_embd_q5k_row_v2<<<(H+255)/256,256>>>(embd_t->data, dst, tok, H);
                    else if (embd_t->type == GGML_TYPE_Q6_K) dequant_embd_q6k_row_v2<<<(H+255)/256,256>>>(embd_t->data, dst, tok, H);
                };
                fold_embed_row(noise, anchor_tok);
                for (int i = 1; i < B; i++) fold_embed_row(noise + (size_t)i*H, dflash::DraftConfig::mask_token_id);

                // (2) positions + (3) draft forward over windowed C[..step-1]
                int W_df = model.dflash_cap.window;
                int ctx_used = (ctx_len_draft < W_df) ? ctx_len_draft : W_df;
                const half* C_view = model.dflash_window_view(step - 1, ctx_used);
                dflash::prepare_positions(dflash_state, ctx_used);
                dflash::draft_forward(dflash_state.draft, C_view, noise,
                                      dflash_state.d_pos_q, dflash_state.d_pos_k, ctx_used, 0);

                // (4) draft lm_head -> draft_chain[d] = draft pred @ step+1+d (slots 1..15)
                static half* host_pinned_draft_f = nullptr;
                if (!host_pinned_draft_f) cudaMallocHost(&host_pinned_draft_f, (size_t)B*H*sizeof(half));
                cudaSetDevice(0); cudaDeviceSynchronize();
                cudaMemcpy(host_pinned_draft_f, dflash_state.draft.h_buf, (size_t)B*H*sizeof(half), cudaMemcpyDeviceToHost);
                cudaSetDevice(last_gpu);
                cudaMemcpy(tree_norm_buf, host_pinned_draft_f, (size_t)B*H*sizeof(half), cudaMemcpyHostToDevice);
                qi_logits_tree.quantize_chunk(tree_norm_buf, H, B, 0);
                quant_gemv_chunk(out_w->data, out_w->type, qi_logits_tree.q8_buf, tree_logits_buf, H, V, B, 0);
                for (int b = 0; b < B; b++)
                    argmax_half_kernel<<<1,1024>>>(tree_logits_buf + (size_t)b*V, V, tree_d_argmax + b);
                cudaMemcpy(tree_h_argmax, tree_d_argmax, (size_t)B*sizeof(int), cudaMemcpyDeviceToHost);
                int fold_chain[16];
                for (int d = 0; d < B-1; d++) fold_chain[d] = tree_h_argmax[d+1];
                if (do_gen_prof) g_mtp += prof_sync_ms(df0, last_gpu);

                // (5) verify: tokens=[anchor_tok, fold_chain[0..14]] @ pos_base=step
                std::vector<int> tokens_h(tree_budget), host_parents(tree_budget);
                tokens_h[0] = anchor_tok; host_parents[0] = -1;
                for (int t = 1; t < tree_budget; t++) { tokens_h[t] = fold_chain[t-1]; host_parents[t] = t-1; }
                cudaSetDevice(0);
                auto se0 = prof_now();
                for (int b = 0; b < tree_budget; b++) {
                    half* dst = tree_hidden_half[0] + (size_t)b*H;
                    if (embd_t->type == GGML_TYPE_Q8_0)      dequant_embd_q8_0_row<<<(H+255)/256,256>>>(embd_t->data, dst, tokens_h[b], H);
                    else if (embd_t->type == GGML_TYPE_Q5_K) dequant_embd_q5k_row_v2<<<(H+255)/256,256>>>(embd_t->data, dst, tokens_h[b], H);
                    else if (embd_t->type == GGML_TYPE_Q6_K) dequant_embd_q6k_row_v2<<<(H+255)/256,256>>>(embd_t->data, dst, tokens_h[b], H);
                }
                half_to_float_kernel<<<(tree_budget*H+255)/256,256>>>(tree_hidden_half[0], tree_hidden[0], tree_budget*H);
                if (do_gen_prof) g_embed += prof_sync_ms(se0, 0);
                model.upload_parent_ids(host_parents.data(), 0);

                float* h_tree = tree_hidden[0];
                for (int layer = 0; layer < model.cfg.num_layers; layer++) {
                    int g_l = gpu_model.layer_gpu[layer];
                    int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer-1];
                    if (g_l != prev_g) {
                        auto tvx = prof_now();
                        cudaSetDevice(prev_g);
                        cudaMemcpy(tree_host_transfer, h_tree, (size_t)tree_budget*H*sizeof(float), cudaMemcpyDeviceToHost);
                        cudaSetDevice(g_l);
                        cudaMemcpy(tree_hidden[g_l], tree_host_transfer, (size_t)tree_budget*H*sizeof(float), cudaMemcpyHostToDevice);
                        h_tree = tree_hidden[g_l];
                        if (do_gen_prof) g_tv_xfer += prof_sync_ms(tvx, g_l);
                    } else cudaSetDevice(g_l);
                    bool is_attn_t = model.is_attn_layer(layer);
                    auto tvl = prof_now();
                    if (is_attn_t) { model.forward_attn_tree(layer, h_tree, step, tree_budget, model.tree_parent_ids_d[g_l], 0); if (do_gen_prof) g_tv_attn += prof_sync_ms(tvl, g_l); }
                    else           { model.forward_gdn_tree(layer, h_tree, tree_budget, model.tree_parent_ids_d[g_l], 0);        if (do_gen_prof) g_tv_gdn  += prof_sync_ms(tvl, g_l); }
                    auto tvm = prof_now();
                    model.forward_mlp_tree(layer, h_tree, tree_budget, model.tree_parent_ids_d[g_l], 0);
                    if (do_gen_prof) g_tv_mlp += prof_sync_ms(tvm, g_l);
                    auto tvc = prof_now();
                    model.dflash_capture_tree_layer(layer, h_tree, tree_budget, g_l, 0);
                    if (do_gen_prof) g_tv_cap += prof_sync_ms(tvc, g_l);
                }
                cudaSetDevice(last_gpu); cudaDeviceSynchronize();
                auto sl0 = prof_now();
                if (out_norm_t->type == GGML_TYPE_F32)
                    rms_norm_f32in_f32w(tree_norm_buf, h_tree, (float*)out_norm_t->data, tree_budget, H, model.cfg.rms_norm_eps, 0);
                else
                    rms_norm_f32in(tree_norm_buf, h_tree, (half*)out_norm_t->data, tree_budget, H, model.cfg.rms_norm_eps, 0);
                qi_logits_tree.quantize_chunk(tree_norm_buf, H, tree_budget, 0);
                quant_gemv_chunk(out_w->data, out_w->type, qi_logits_tree.q8_buf, tree_logits_buf, H, V, tree_budget, 0);
                for (int b = 0; b < tree_budget; b++)
                    argmax_half_kernel<<<1,1024>>>(tree_logits_buf + (size_t)b*V, V, tree_d_argmax + b);
                cudaMemcpy(tree_h_argmax, tree_d_argmax, (size_t)tree_budget*sizeof(int), cudaMemcpyDeviceToHost);
                if (do_gen_prof) { g_logits += prof_sync_ms(sl0, last_gpu); g_spec_iters++; g_steps++; }

                // (6) accept: posterior[i]=tree_h_argmax[i]=27B greedy @ step+1+i.
                //     accept_drafts = largest m s.t. fold_chain[i]==posterior[i] for i<m.
                int accept_drafts = 0;
                for (int i = 0; i < tree_budget - 1; i++) {
                    if (tree_h_argmax[i] == fold_chain[i]) accept_drafts++;
                    else break;
                }
                int accept_len_slots = accept_drafts + 1;     // slots 0..accept_drafts (positions step..step+accept_drafts)
                model.commit_tree_gdn_chain(accept_len_slots);
                std::vector<int> commit_slots(accept_len_slots);
                for (int i = 0; i < accept_len_slots; i++) commit_slots[i] = i;
                model.dflash_commit_tree_capture(commit_slots.data(), accept_len_slots, step);  // C[step..step+accept]

                if (accept_drafts == tree_budget - 1) tree_accept_full_count++;
                else if (accept_drafts > 0)            tree_accept_partial_count++;
                else                                    tree_reject_count++;
                spec_accept_count += accept_drafts;

                // (7) emit posterior[0..accept_drafts] = 27B greedy tokens @ step+1..step+1+accept_drafts.
                if (!got_first_tok) { t_first = std::chrono::high_resolution_clock::now(); got_first_tok = true; }
                bool stopped_fold = false;
                for (int i = 0; i <= accept_drafts && !stopped_fold; i++) {
                    if (emit_tok(tree_h_argmax[i])) stopped_fold = true;
                }
                // (8) step advance. New slot0 = posterior[accept_drafts] @ step+1+accept_drafts.
                //     step += accept_drafts; outer ++ -> next step = step+accept_drafts+1 = step+1+accept_drafts.
                step += accept_drafts;
                if (stopped_fold) break;
                continue;
            }

            cudaSetDevice(0);
            auto ge0 = prof_now();
            // Dequant embedding into fp16 scratch, then convert to fp32 hidden
            if (embd_t->type == GGML_TYPE_Q8_0)
                dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
            else if (embd_t->type == GGML_TYPE_Q5_K)
                dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
            else if (embd_t->type == GGML_TYPE_Q6_K)
                dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
            half_to_float_kernel<<<(H+255)/256, 256>>>(gpu_hidden_half[0], gpu_hidden[0], H);
            if (do_gen_prof) g_embed += prof_sync_ms(ge0, 0);

            {
                static const bool dump_emb = getenv("DUMP_LAYERS") != nullptr;
                if (dump_emb && step == (int)prompt_ids.size() - 1) {
                    cudaDeviceSynchronize();
                    float sample[8];
                    cudaMemcpy(sample, gpu_hidden[0], 8 * sizeof(float), cudaMemcpyDeviceToHost);
                    fprintf(stderr, "[EMB tok=%d] %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
                            token_id, sample[0], sample[1], sample[2], sample[3],
                            sample[4], sample[5], sample[6], sample[7]);
                    fflush(stderr);
                }
            }

            float* h = gpu_hidden[0];
            for (int layer = 0; layer < model.cfg.num_layers; layer++) {
                int g = gpu_model.layer_gpu[layer];
                int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer - 1];
                if (g != prev_g) {
                    auto gx0 = prof_now();
                    cudaSetDevice(prev_g);
                    cudaMemcpy(host_transfer, h, H * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaSetDevice(g);
                    cudaMemcpy(gpu_hidden[g], host_transfer, H * sizeof(float), cudaMemcpyHostToDevice);
                    h = gpu_hidden[g];
                    if (do_gen_prof) g_xfer += prof_sync_ms(gx0, g);
                } else { cudaSetDevice(g); }
                auto ga0 = prof_now();
                bool is_attn = model.is_attn_layer(layer);
                if (is_attn)
                    model.forward_attn(layer, h, step, 0,
                                       /*external_proj=*/false, /*slot_pos=*/-1,
                                       /*mask_start=*/-1, /*mask_len=*/0,
                                       /*mask_bits=*/0xffffffffu, slot);
                else
                    model.forward_gdn(layer, h, 0, slot);
                if (do_gen_prof) {
                    double ms = prof_sync_ms(ga0, g);
                    if (is_attn) g_attn += ms; else g_gdn += ms;
                }
                auto gm0 = prof_now();
                if (model.layer_is_moe[layer]) model.forward_moe(layer, h, 0);
                else                           model.forward_mlp(layer, h, 0);
                if (do_gen_prof) g_mlp += prof_sync_ms(gm0, g);

                // DFlash hidden capture (no-op unless dflash mode enabled).
                // Slot-0 only — dflash uses a single global hidden buffer.
                if (slot == 0) {
                    model.dflash_capture_layer(layer, h, step, g, 0);
                }

                // DUMP_LAYERS=1: 각 layer 후 hidden state 처음 8 floats 출력.
                // 마지막 prompt token (= prefill 직후 첫 per-token step) 에서만
                // 찍어서 llama.cpp eval-callback 출력과 layer-wise 비교 가능.
                static const bool dump_layers = getenv("DUMP_LAYERS") != nullptr;
                if (dump_layers && step == (int)prompt_ids.size() - 1) {
                    cudaSetDevice(g); cudaDeviceSynchronize();
                    float sample[8];
                    cudaMemcpy(sample, h, 8 * sizeof(float), cudaMemcpyDeviceToHost);
                    fprintf(stderr, "[L%02d %s] %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
                            layer, is_attn ? "attn" : "gdn ",
                            sample[0], sample[1], sample[2], sample[3],
                            sample[4], sample[5], sample[6], sample[7]);
                    fflush(stderr);
                }
                // DUMP_PREFILL_HASH=1 (per-token side): for each step <
                // prompt_ids.size() (prefill phase via DISABLE_CHUNKED_PREFILL),
                // emit the same per-(layer,token) hash format as the chunked
                // path for diff. Only emit while we're still processing prompt
                // tokens; once past prompt the hash is just generation noise.
                static const bool dump_prefill_hash = getenv("DUMP_PREFILL_HASH") != nullptr;
                if (dump_prefill_hash && step < (int)prompt_ids.size()) {
                    cudaSetDevice(g); cudaDeviceSynchronize();
                    static thread_local std::vector<float> host_buf;
                    host_buf.resize(H);
                    cudaMemcpy(host_buf.data(), h, H * sizeof(float), cudaMemcpyDeviceToHost);
                    uint64_t hh = 0xcbf29ce484222325ULL;
                    for (int i = 0; i < H; i++) {
                        uint32_t b; memcpy(&b, &host_buf[i], 4);
                        hh = (hh ^ (uint64_t)b) * 0x100000001b3ULL;
                    }
                    fprintf(stderr, "[CHK L%02d t%03d %s] hash=%016lx\n",
                            layer, step, is_attn ? "attn" : "gdn ", hh);
                    fflush(stderr);
                }
            }

            if (step >= (int)prompt_ids.size() - 1) {
                cudaSetDevice(last_gpu); cudaDeviceSynchronize();
                auto gl0 = prof_now();
                // Convert fp32 hidden to fp16 for output norm + projection
                float_to_half_kernel<<<(H+255)/256, 256>>>(h, gpu_hidden_half[last_gpu], H);
                if (out_norm_t->type == GGML_TYPE_F32)
                    rms_norm_f32w(norm_buf, gpu_hidden_half[last_gpu], (float*)out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
                else
                    rms_norm(norm_buf, gpu_hidden_half[last_gpu], (half*)out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
                qi_logits.quantize(norm_buf, H, 0);
                quant_gemv(out_w->data, out_w->type, norm_buf, logits_buf, H, V, &qi_logits);
                if (do_gen_prof) g_logits += prof_sync_ms(gl0, last_gpu);
                if (do_gen_prof) g_steps++;

                std::vector<int> ctx = prompt_ids;
                ctx.insert(ctx.end(), generated.begin(), generated.end());

                // Greedy fast path: temp<=0 + no rep penalty → argmax on the
                // GPU and only ship 4 bytes home over the slow PCIe x1 bus
                // instead of V*2 ≈ 500 KB.
                int max_idx;
                const auto& sp_now = sampler.params();
                bool greedy_fast = (sp_now.temperature <= 0.0f)
                                && (sp_now.rep_penalty == 1.0f)
                                && (sp_now.freq_penalty == 0.0f)
                                && (sp_now.pres_penalty == 0.0f)
                                && (sampler.grammar == nullptr);
                if (greedy_fast) {
                    argmax_half_kernel<<<1, 1024>>>(logits_buf, V, d_argmax);
                    cudaMemcpy(h_argmax_pinned, d_argmax, sizeof(int), cudaMemcpyDeviceToHost);
                    max_idx = h_argmax_pinned[0];
                } else {
                    cudaDeviceSynchronize();
                    std::vector<half> h_logits(V);
                    cudaMemcpy(h_logits.data(), logits_buf, V * sizeof(half), cudaMemcpyDeviceToHost);
                    max_idx = sampler.sample(h_logits.data(), V, ctx);
                }

                // Phase 0 MTP measurement: at the same step run the MTP head
                // with the post-output_norm hidden state and compare its argmax
                // to main's. Both predict the token at position step+1.
                //
                // MTP_DRAFT=1 switches to TRUE draft mode (one-step ahead).
                // In that mode at iter for step we call MTP with prev_token_id
                // = max_idx (the token main JUST produced) and position=step+1,
                // so MTP predicts the token at step+2. The verify is then done
                // at the NEXT iter by comparing that iter's main max_idx to the
                // draft we saved here. mtp_pending_draft / _step persist across
                // iters for that purpose.
                if (slot == 0 && mtp_loaded && !spec_enabled && step >= (int)prompt_ids.size() - 1 && getenv("MTP_ON")) {
                    cudaError_t err_pre = cudaGetLastError();
                    if (err_pre != cudaSuccess && mtp_total_count == 0)
                        printf("[MTP] CUDA error BEFORE forward: %s\n", cudaGetErrorString(err_pre));
                    bool draft_mode = getenv("MTP_DRAFT") != nullptr;
                    int mtp_pred;
                    if (draft_mode) {
                        if (mtp_pending_draft_step == step && mtp_pending_draft >= 0) {
                            if (mtp_pending_draft == max_idx) mtp_accept_count++;
                            mtp_total_count++;
                            if (mtp_total_count <= 3)
                                printf("[MTP-DRAFT] verify step=%d main=%d draft=%d %s\n",
                                       step, max_idx, mtp_pending_draft,
                                       (mtp_pending_draft == max_idx) ? "ACCEPT" : "reject");
                        }
                        mtp_pred = mtp.forward(norm_buf, max_idx, step + 1);
                        mtp_pending_draft = mtp_pred;
                        mtp_pending_draft_step = step + 1;
                        cudaError_t err_post = cudaGetLastError();
                        if (err_post != cudaSuccess)
                            printf("[MTP-DRAFT] CUDA error step=%d: %s\n", step, cudaGetErrorString(err_post));
                        cudaSetDevice(last_gpu);
                    } else {
                        mtp_pred = mtp.forward(norm_buf, token_id, step);
                        cudaError_t err_post = cudaGetLastError();
                        if (err_post != cudaSuccess && mtp_total_count < 3)
                            printf("[MTP] CUDA error AFTER forward step=%d: %s\n", step, cudaGetErrorString(err_post));
                        if (mtp_total_count < 3)
                            printf("[MTP] step=%d token_id=%d main=%d mtp=%d\n", step, token_id, max_idx, mtp_pred);
                        if (mtp_pred == max_idx) mtp_accept_count++;
                        mtp_total_count++;
                        cudaSetDevice(last_gpu);
                    }
                }

                if (step >= (int)prompt_ids.size()) {
                    if (!got_first_tok) { t_first = std::chrono::high_resolution_clock::now(); got_first_tok = true; }
                    // Stop tokens are NOT emitted to the client: check before
                    // push/stream so the final response text doesn't include
                    // raw `<|im_end|>` / `<|endoftext|>` markers.
                    if (max_idx == 248046 || max_idx == 248044 || max_idx == 248045) break;
                    generated.push_back(max_idx);
                    if (on_token) on_token(max_idx);

                    // Track think state: tokens inside <think>...</think> don't count against max_tokens
                    if (max_idx == 248068) in_think = true;   // <think>
                    if (max_idx == 248069) in_think = false;  // </think>
                    if (!in_think) output_tokens++;

                    if (output_tokens >= max_gen) break;  // max_tokens only counts non-think output
                    // Stop on tool call end tags
                    if (generated.size() >= 4) {
                        std::string tail = tok.decode(std::vector<int>(generated.end()-4, generated.end()));
                        if (tail.find("</tool_call>") != std::string::npos) break;
                    }

                    // ===================== DFlash speculative decode =====================
                    // Step 6c-1 (diagnostic only): predict the next 16-token block
                    // with the DFlash draft, lm_head it, dump the chain. No accept
                    // commit yet — main keeps producing tokens normally. Validates
                    // that draft.forward + capture buffer + lm_head all wire up.
                    // (Accept + KV/GDN rollback land in 6c-2 / 6c-3.)
                    if (slot == 0 && dflash_enabled && sampler.grammar == nullptr) {
                        auto df0 = prof_now();
                        int B = dflash::DraftConfig::block_size;          // 16
                        int ctx_len_draft = step + 1;                     // capture[0..step] filled

                        // (1) noise_embed = [embed(max_idx), embed(MASK)*15]
                        cudaSetDevice(0);
                        half* noise = dflash_state.d_noise_embed;
                        auto embed_row = [&](half* dst, int tok) {
                            if (embd_t->type == GGML_TYPE_Q8_0)
                                dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(embd_t->data, dst, tok, H);
                            else if (embd_t->type == GGML_TYPE_Q5_K)
                                dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, dst, tok, H);
                            else if (embd_t->type == GGML_TYPE_Q6_K)
                                dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, dst, tok, H);
                        };
                        embed_row(noise, max_idx);
                        for (int i = 1; i < B; i++)
                            embed_row(noise + (size_t)i * H, dflash::DraftConfig::mask_token_id);

                        // (2) positions + windowed C view (cap drafter ctx to the ring window)
                        int W_df = model.dflash_cap.window;
                        int ctx_used = (ctx_len_draft < W_df) ? ctx_len_draft : W_df;
                        const half* C_view = model.dflash_window_view(step, ctx_used);
                        dflash::prepare_positions(dflash_state, ctx_used);

                        // (3) draft forward → dflash_state.draft.h_buf [B, H] fp16
                        dflash::draft_forward(
                            dflash_state.draft,
                            C_view,
                            noise,
                            dflash_state.d_pos_q,
                            dflash_state.d_pos_k,
                            ctx_used, 0);

                        // (4) lm_head: draft hidden (GPU 0) → last_gpu via host pinned bridge
                        static half* host_pinned_draft = nullptr;
                        if (!host_pinned_draft)
                            cudaMallocHost(&host_pinned_draft, (size_t)B * H * sizeof(half));
                        cudaSetDevice(0); cudaDeviceSynchronize();
                        cudaMemcpy(host_pinned_draft, dflash_state.draft.h_buf,
                                   (size_t)B * H * sizeof(half),
                                   cudaMemcpyDeviceToHost);
                        cudaSetDevice(last_gpu);
                        cudaMemcpy(tree_norm_buf, host_pinned_draft,
                                   (size_t)B * H * sizeof(half),
                                   cudaMemcpyHostToDevice);
                        // Draft already applied final out_norm; skip rms_norm here.
                        qi_logits_tree.quantize_chunk(tree_norm_buf, H, B, 0);
                        quant_gemv_chunk(out_w->data, out_w->type,
                                         qi_logits_tree.q8_buf,
                                         tree_logits_buf, H, V, B, 0);
                        for (int b = 0; b < B; b++)
                            argmax_half_kernel<<<1, 1024>>>(
                                tree_logits_buf + (size_t)b * V, V,
                                tree_d_argmax + b);
                        cudaMemcpy(tree_h_argmax, tree_d_argmax,
                                   (size_t)B * sizeof(int),
                                   cudaMemcpyDeviceToHost);

                        // Slot 1..15 are drafts for positions step+2..step+16.
                        // (Slot 0 is the draft's prediction for pos step+1, which
                        //  duplicates main's max_idx — discarded.)
                        int draft_chain[16];  // up to B-1 = 15 used
                        for (int d = 0; d < B - 1; d++) draft_chain[d] = tree_h_argmax[d + 1];

                        static int dflash_diag_count = 0;
                        if (dflash_diag_count < 5) {
                            printf("[dflash-diag] step=%d main=%d draft slot0=%d chain:",
                                   step, max_idx, tree_h_argmax[0]);
                            for (int i = 0; i < B - 1; i++) printf(" %d", draft_chain[i]);
                            printf("\n");
                            fflush(stdout);
                            dflash_diag_count++;
                        }
                        if (do_gen_prof) g_mtp += prof_sync_ms(df0, last_gpu);

                        // (6) Batched tree verify path (6c-3).
                        // tokens = [max_idx, chain[0], ..., chain[14]] (16 slots).
                        // parent_ids = [-1, 0, 1, ..., 14] (chain shape).
                        // pos_base = step + 1 (slot t at absolute pos step+1+t).
                        // We forward all 16 slots through every target layer, capturing
                        // configured layer outputs into the GPU 0 scratch as we go.
                        // After lm_head per slot, chain accept_drafts is the largest d
                        // such that posterior[i] == chain[i] for all i < d.
                        std::vector<int> tokens_h(tree_budget);
                        std::vector<int> host_parents(tree_budget);
                        tokens_h[0] = max_idx;
                        host_parents[0] = -1;
                        for (int t = 1; t < tree_budget; t++) {
                            tokens_h[t]      = draft_chain[t - 1];
                            host_parents[t]  = t - 1;
                        }

                        // (6a) Embed all 16 tokens onto GPU 0
                        cudaSetDevice(0);
                        auto se0 = prof_now();
                        for (int b = 0; b < tree_budget; b++) {
                            half* dst = tree_hidden_half[0] + (size_t)b * H;
                            if (embd_t->type == GGML_TYPE_Q8_0)
                                dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(embd_t->data, dst, tokens_h[b], H);
                            else if (embd_t->type == GGML_TYPE_Q5_K)
                                dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, dst, tokens_h[b], H);
                            else if (embd_t->type == GGML_TYPE_Q6_K)
                                dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, dst, tokens_h[b], H);
                        }
                        half_to_float_kernel<<<(tree_budget*H+255)/256, 256>>>(
                            tree_hidden_half[0], tree_hidden[0], tree_budget * H);
                        if (do_gen_prof) g_embed += prof_sync_ms(se0, 0);

                        model.upload_parent_ids(host_parents.data(), 0);

                        // (6b) Tree forward 64 layers; capture per-layer hiddens.
                        float* h_tree = tree_hidden[0];
                        for (int layer = 0; layer < model.cfg.num_layers; layer++) {
                            int g_l = gpu_model.layer_gpu[layer];
                            int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer - 1];
                            if (g_l != prev_g) {
                                auto tvx = prof_now();
                                cudaSetDevice(prev_g);
                                cudaMemcpy(tree_host_transfer, h_tree,
                                           (size_t)tree_budget * H * sizeof(float),
                                           cudaMemcpyDeviceToHost);
                                cudaSetDevice(g_l);
                                cudaMemcpy(tree_hidden[g_l], tree_host_transfer,
                                           (size_t)tree_budget * H * sizeof(float),
                                           cudaMemcpyHostToDevice);
                                h_tree = tree_hidden[g_l];
                                if (do_gen_prof) g_tv_xfer += prof_sync_ms(tvx, g_l);
                            } else cudaSetDevice(g_l);

                            bool is_attn_t = model.is_attn_layer(layer);
                            auto tvl = prof_now();
                            if (is_attn_t) {
                                model.forward_attn_tree(layer, h_tree, step + 1, tree_budget,
                                                        model.tree_parent_ids_d[g_l], 0);
                                if (do_gen_prof) g_tv_attn += prof_sync_ms(tvl, g_l);
                            } else {
                                model.forward_gdn_tree(layer, h_tree, tree_budget,
                                                       model.tree_parent_ids_d[g_l], 0);
                                if (do_gen_prof) g_tv_gdn += prof_sync_ms(tvl, g_l);
                            }
                            auto tvm = prof_now();
                            model.forward_mlp_tree(layer, h_tree, tree_budget,
                                                   model.tree_parent_ids_d[g_l], 0);
                            if (do_gen_prof) g_tv_mlp += prof_sync_ms(tvm, g_l);
                            // Capture this layer's per-slot hidden into scratch
                            // (no-op unless layer ∈ {1, 16, 31, 46, 61}).
                            auto tvc = prof_now();
                            model.dflash_capture_tree_layer(layer, h_tree, tree_budget, g_l, 0);
                            if (do_gen_prof) g_tv_cap += prof_sync_ms(tvc, g_l);
                        }

                        // (6c) Batched lm_head per slot.
                        cudaSetDevice(last_gpu); cudaDeviceSynchronize();
                        auto sl0 = prof_now();
                        if (out_norm_t->type == GGML_TYPE_F32)
                            rms_norm_f32in_f32w(tree_norm_buf, h_tree, (float*)out_norm_t->data,
                                                tree_budget, H, model.cfg.rms_norm_eps, 0);
                        else
                            rms_norm_f32in(tree_norm_buf, h_tree, (half*)out_norm_t->data,
                                           tree_budget, H, model.cfg.rms_norm_eps, 0);
                        qi_logits_tree.quantize_chunk(tree_norm_buf, H, tree_budget, 0);
                        quant_gemv_chunk(out_w->data, out_w->type, qi_logits_tree.q8_buf,
                                         tree_logits_buf, H, V, tree_budget, 0);
                        for (int b = 0; b < tree_budget; b++)
                            argmax_half_kernel<<<1, 1024>>>(
                                tree_logits_buf + (size_t)b * V, V,
                                tree_d_argmax + b);
                        cudaMemcpy(tree_h_argmax, tree_d_argmax,
                                   (size_t)tree_budget * sizeof(int), cudaMemcpyDeviceToHost);
                        if (do_gen_prof) {
                            g_logits += prof_sync_ms(sl0, last_gpu);
                            g_spec_iters++;
                        }

                        // (6d) Chain accept: largest d such that posterior[i] == chain[i] for i<d.
                        // posterior[i] = tree_h_argmax[i] = prediction for pos step+2+i.
                        // DFLASH_POS_ACCEPT=1: bucket per-position conditional accept
                        // (reached[i] = chain survived to draft pos i; matched[i] = matched at i).
                        // p_{i+1} = matched[i]/reached[i] = the decay curve.
                        static const bool pos_accept = getenv("DFLASH_POS_ACCEPT") != nullptr;
                        static long pos_reached[16] = {0}, pos_matched[16] = {0};
                        int accept_drafts = 0;
                        for (int i = 0; i < tree_budget - 1; i++) {
                            if (pos_accept) pos_reached[i]++;
                            if (tree_h_argmax[i] == draft_chain[i]) { accept_drafts++; if (pos_accept) pos_matched[i]++; }
                            else break;
                        }
                        if (pos_accept) {
                            static int pa_cnt = 0;
                            if (++pa_cnt % 40 == 0) {  // periodic dump
                                printf("[POS-ACCEPT] p_k (k=1..%d): ", tree_budget - 1);
                                for (int i = 0; i < tree_budget - 1; i++)
                                    printf("%.2f ", pos_reached[i] ? (double)pos_matched[i] / pos_reached[i] : 0.0);
                                printf("\n"); fflush(stdout);
                            }
                        }
                        // Bonus = posterior at slot accept_drafts (= chain[accept_drafts] if matched, or tree_h_argmax[accept_drafts] if rejected).
                        int bonus = tree_h_argmax[accept_drafts];

                        // (6e) Commit GDN state for chain prefix [slot 0..accept_drafts].
                        int accept_len_slots = accept_drafts + 1;  // includes root
                        model.commit_tree_gdn_chain(accept_len_slots);

                        // (6f) Commit captured hiddens for accepted slots into main capture buffer.
                        // Slot 0 (max_idx) at pos step+1, slot t at pos step+1+t.
                        std::vector<int> commit_slots(accept_len_slots);
                        for (int i = 0; i < accept_len_slots; i++) commit_slots[i] = i;
                        model.dflash_commit_tree_capture(commit_slots.data(), accept_len_slots, step + 1);

                        // (6g) Counters.
                        if (accept_drafts == tree_budget - 1) tree_accept_full_count++;
                        else if (accept_drafts > 0)            tree_accept_partial_count++;
                        else                                    tree_reject_count++;
                        spec_accept_count += accept_drafts;

                        // (6h) Emit accepted drafts + bonus.
                        bool stopped_dflash = false;
                        for (int i = 0; i < accept_drafts && !stopped_dflash; i++) {
                            if (emit_tok(draft_chain[i])) stopped_dflash = true;
                        }
                        if (!stopped_dflash) {
                            if (emit_tok(bonus)) stopped_dflash = true;
                        }

                        // (6i) Step advance. Tokens emitted this iter (post-max_idx) = accept_drafts + 1.
                        // Outer step++ adds 1; want next iter step = X + 1 + emitted + 1 - 1 = X + emitted + 1.
                        // step += emitted; outer ++ → step + 1 = X + emitted + 1.
                        step += accept_drafts + 1;
                        if (stopped_dflash) break;
                        continue;
                    } else

                    // ===================== MTP_TREE chain-tree path (Phase 1) =====================
                    // Chain tree with parent_ids=[-1,0,1] is MTP K=2-equivalent in
                    // terms of sampled argmax outputs; we route through the
                    // forward_*_tree pipeline to shake out the DDTree kernels
                    // end-to-end. Once this matches K2, we'll widen to real
                    // branching by growing budget and adding ancestor-mask attn.
                    if (slot == 0 && spec_tree_enabled && sampler.grammar == nullptr) {
                        auto sd0 = prof_now();
                        // MTP_TREE_BRANCH=1: root fan-out of top-2 drafts.
                        //   budget=3 → depth 1 (tree [root, a, b], parents [-1,0,0])
                        //   budget=5 → depth 2 (tree [root, a, b, aa, bb],
                        //                       parents [-1, 0, 0, 1, 2])
                        //   Grandchildren aa, bb come from a per-branch MTP
                        //   forward(h_root, a|b, step+2); each is rolled back
                        //   immediately so MTP KV keeps only the root slot.
                        // Default (chain): budget=N chain, drafts = [d0, d1, ...]
                        //   with parents = [-1, 0, 1, ..., N-2].
                        static const bool use_branch = getenv("MTP_TREE_BRANCH") != nullptr;
                        // Supported branching shapes:
                        //   budget=3: fanout=2, depth=1 → [root, a, b]
                        //   budget=4: fanout=3, depth=1 → [root, a, b, c]
                        //   budget=5: fanout=2, depth=2 → [root, a, b, aa, bb]
                        //   budget=7: fanout=2, depth=3 → [root, a, b, aa, bb, aaa, bbb]
                        bool branch_mode = use_branch &&
                                           (tree_budget == 3 || tree_budget == 4 ||
                                            tree_budget == 5 || tree_budget == 7);
                        int  branch_fanout = (tree_budget == 4) ? 3 : 2;
                        int  branch_depth  = (tree_budget == 5) ? 2 :
                                             (tree_budget == 7) ? 3 : 1;
                        std::vector<int> drafts(tree_budget - 1);
                        std::vector<int> host_parents(tree_budget);
                        host_parents[0] = -1;
                        if (branch_mode) {
                            // Root: MTP top-K at step+1 (K == branch_fanout).
                            int top_ids[4] = {-1,-1,-1,-1};
                            mtp.forward_topk(norm_buf, max_idx, step + 1, branch_fanout,
                                             top_ids, nullptr);
                            for (int k = 0; k < branch_fanout; k++) {
                                drafts[k] = top_ids[k];
                                host_parents[k + 1] = 0;
                            }
                            if (branch_depth == 2) {
                                // fanout=2, depth=2. Save the root's h_final so both
                                // branches can chain off it.
                                half* h_root = h_final_draft1;
                                mtp.copy_h_final_to(h_root, 0);
                                drafts[2] = mtp.forward(h_root, drafts[0], step + 2);
                                mtp.kv_rollback(1);
                                drafts[3] = mtp.forward(h_root, drafts[1], step + 2);
                                mtp.kv_rollback(1);
                                host_parents[3] = 1;
                                host_parents[4] = 2;
                            } else if (branch_depth == 3) {
                                // fanout=2, depth=3. Tree slots:
                                //   0:root  1:a   2:b   3:aa  4:bb  5:aaa 6:bbb
                                // drafts indices: 0=a 1=b 2=aa 3=bb 4=aaa 5=bbb
                                half* h_root = h_final_draft1;
                                mtp.copy_h_final_to(h_root, 0);
                                // Branch a chain: forward(root, a) → aa,
                                // then forward(h_aa, aa) → aaa.
                                half* h_chain = h_final_draft2_tree;
                                drafts[2] = mtp.forward_with_state(h_root, drafts[0], step + 2, h_chain);
                                drafts[4] = mtp.forward(h_chain, drafts[2], step + 3);
                                mtp.kv_rollback(2);
                                // Branch b chain (reuses h_chain buffer).
                                drafts[3] = mtp.forward_with_state(h_root, drafts[1], step + 2, h_chain);
                                drafts[5] = mtp.forward(h_chain, drafts[3], step + 3);
                                mtp.kv_rollback(2);
                                host_parents[3] = 1;  // aa  ← a
                                host_parents[4] = 2;  // bb  ← b
                                host_parents[5] = 3;  // aaa ← aa
                                host_parents[6] = 4;  // bbb ← bb
                            }
                        } else {
                            // Chain fallback.
                            half* h_buf[2] = {h_final_draft1, h_final_draft2_tree};
                            const half* h_prev_in = norm_buf;
                            int prev_tok = max_idx;
                            for (int d = 0; d < tree_budget - 1; d++) {
                                if (d + 1 < tree_budget - 1) {
                                    drafts[d] = mtp.forward_with_state(
                                        h_prev_in, prev_tok, step + 1 + d, h_buf[d & 1]);
                                    h_prev_in = h_buf[d & 1];
                                } else {
                                    drafts[d] = mtp.forward(
                                        h_prev_in, prev_tok, step + 1 + d);
                                }
                                prev_tok = drafts[d];
                            }
                            for (int i = 1; i < tree_budget; i++) host_parents[i] = i - 1;
                        }
                        cudaSetDevice(last_gpu);
                        if (do_gen_prof) g_mtp += prof_sync_ms(sd0, last_gpu);

                        // 2. Upload parent_ids (also triggers host-side depth +
                        //    ancestor-bitmask derivation for attn tree path).
                        model.upload_parent_ids(host_parents.data(), 0);

                        // 3. Embed all budget tokens into contiguous slots on GPU 0.
                        cudaSetDevice(0);
                        auto se0 = prof_now();
                        std::vector<int> tokens(tree_budget);
                        tokens[0] = max_idx;
                        for (int d = 0; d < tree_budget - 1; d++) tokens[d + 1] = drafts[d];
                        for (int b = 0; b < tree_budget; b++) {
                            half* dst = tree_hidden_half[0] + (size_t)b * H;
                            if (embd_t->type == GGML_TYPE_Q8_0)
                                dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(embd_t->data, dst, tokens[b], H);
                            else if (embd_t->type == GGML_TYPE_Q5_K)
                                dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, dst, tokens[b], H);
                            else if (embd_t->type == GGML_TYPE_Q6_K)
                                dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, dst, tokens[b], H);
                        }
                        half_to_float_kernel<<<(tree_budget*H+255)/256, 256>>>(
                            tree_hidden_half[0], tree_hidden[0], tree_budget * H);
                        if (do_gen_prof) g_embed += prof_sync_ms(se0, 0);

                        // 4. Run all layers in tree mode.
                        float* h_tree = tree_hidden[0];
                        for (int layer = 0; layer < model.cfg.num_layers; layer++) {
                            int g_l = gpu_model.layer_gpu[layer];
                            int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer - 1];
                            if (g_l != prev_g) {
                                auto sx0 = prof_now();
                                cudaSetDevice(prev_g);
                                cudaMemcpy(tree_host_transfer, h_tree,
                                           (size_t)tree_budget * H * sizeof(float),
                                           cudaMemcpyDeviceToHost);
                                cudaSetDevice(g_l);
                                cudaMemcpy(tree_hidden[g_l], tree_host_transfer,
                                           (size_t)tree_budget * H * sizeof(float),
                                           cudaMemcpyHostToDevice);
                                h_tree = tree_hidden[g_l];
                                if (do_gen_prof) g_xfer += prof_sync_ms(sx0, g_l);
                            } else {
                                cudaSetDevice(g_l);
                            }
                            auto sa0 = prof_now();
                            bool is_attn_t = model.is_attn_layer(layer);
                            if (is_attn_t) {
                                model.forward_attn_tree(layer, h_tree, step + 1, tree_budget,
                                                        model.tree_parent_ids_d[g_l], 0);
                            } else {
                                model.forward_gdn_tree(layer, h_tree, tree_budget,
                                                       model.tree_parent_ids_d[g_l], 0);
                            }
                            if (do_gen_prof) {
                                double ms = prof_sync_ms(sa0, g_l);
                                if (is_attn_t) g_attn += ms; else g_gdn += ms;
                            }
                            auto sm0 = prof_now();
                            model.forward_mlp_tree(layer, h_tree, tree_budget,
                                                   model.tree_parent_ids_d[g_l], 0);
                            if (do_gen_prof) g_mlp += prof_sync_ms(sm0, g_l);
                        }

                        // 5. Output norm + batched lm_head for all budget nodes.
                        cudaSetDevice(last_gpu);
                        cudaDeviceSynchronize();
                        auto sl0 = prof_now();
                        if (out_norm_t->type == GGML_TYPE_F32) {
                            rms_norm_f32in_f32w(tree_norm_buf, h_tree,
                                                (float*)out_norm_t->data,
                                                tree_budget, H,
                                                model.cfg.rms_norm_eps, 0);
                        } else {
                            rms_norm_f32in(tree_norm_buf, h_tree,
                                           (half*)out_norm_t->data,
                                           tree_budget, H,
                                           model.cfg.rms_norm_eps, 0);
                        }
                        qi_logits_tree.quantize_chunk(tree_norm_buf, H, tree_budget, 0);
                        quant_gemv_chunk(out_w->data, out_w->type,
                                         qi_logits_tree.q8_buf,
                                         tree_logits_buf, H, V, tree_budget, 0);
                        for (int b = 0; b < tree_budget; b++) {
                            argmax_half_kernel<<<1, 1024>>>(
                                tree_logits_buf + (size_t)b * V, V,
                                tree_d_argmax + b);
                        }
                        cudaMemcpy(tree_h_argmax, tree_d_argmax,
                                   (size_t)tree_budget * sizeof(int),
                                   cudaMemcpyDeviceToHost);
                        if (do_gen_prof) {
                            g_logits += prof_sync_ms(sl0, last_gpu);
                            g_spec_iters++;
                        }

                        // Accept logic — differs for chain vs branch mode.
                        int accept_len = 1;
                        int accepted_final_slot = 0;  // slot index of last accepted node
                        std::vector<int> accepted_path = {0};  // slot indices of accepted chain
                        int emit_final_verify_slot = 0;  // which verify[] to emit as next-main
                        if (branch_mode) {
                            // Root fan-out. verify[0] is main's slot-0
                            // prediction (next token at step+2).
                            // Depth 1, fanout K: verify[0]==drafts[k] → branch k accept.
                            // Depth 2, fanout 2: further checks verify[1]==drafts[2] (aa)
                            //                    or verify[2]==drafts[3] (bb) on the
                            //                    selected branch for one extra accepted token.
                            // Depth 3, fanout 2: same chain extended one more level
                            //                    via verify[grand_slot]==drafts[great_idx].
                            accept_len = 1;
                            accepted_path = {0};
                            int matched_branch = -1;
                            for (int k = 0; k < branch_fanout; k++) {
                                if (tree_h_argmax[0] == drafts[k]) {
                                    matched_branch = k;
                                    break;
                                }
                            }
                            if (matched_branch >= 0) {
                                accept_len = 2;
                                accepted_path = {0, matched_branch + 1};
                                if (branch_depth >= 2) {
                                    // depth ≥ 2 path is fanout=2 only.
                                    // Slots: branch a (m=0): {1,3,5}, b (m=1): {2,4,6}
                                    // Drafts: a→aa,aaa = {2,4}; b→bb,bbb = {3,5}
                                    int grand_slot  = matched_branch + 3;
                                    int grand_draft = matched_branch + 2;
                                    if (tree_h_argmax[matched_branch + 1] == drafts[grand_draft]) {
                                        accept_len = 3;
                                        accepted_path = {0, matched_branch + 1, grand_slot};
                                        if (branch_depth >= 3) {
                                            int great_slot  = matched_branch + 5;
                                            int great_draft = matched_branch + 4;
                                            if (tree_h_argmax[grand_slot] == drafts[great_draft]) {
                                                accept_len = 4;
                                                accepted_path.push_back(great_slot);
                                            }
                                        }
                                    }
                                }
                            }
                            accepted_final_slot = accepted_path.back();
                            emit_final_verify_slot = accepted_path.back();
                        } else {
                            // Chain: longest matching prefix.
                            for (int d = 0; d < tree_budget - 1; d++) {
                                if (tree_h_argmax[d] == drafts[d]) accept_len = d + 2;
                                else break;
                            }
                            accepted_path.resize(accept_len);
                            for (int i = 0; i < accept_len; i++) accepted_path[i] = i;
                            accepted_final_slot = accept_len - 1;
                            emit_final_verify_slot = accept_len - 1;
                        }

                        if (spec_total_count < 3) {
                            printf("[TREE] step=%d budget=%d mode=%s main=%d drafts=[",
                                   step, tree_budget, branch_mode ? "branch" : "chain", max_idx);
                            for (int d = 0; d < tree_budget - 1; d++)
                                printf("%s%d", d ? "," : "", drafts[d]);
                            printf("] verify=[");
                            for (int b = 0; b < tree_budget; b++)
                                printf("%s%d", b ? "," : "", tree_h_argmax[b]);
                            printf("] accept_len=%d\n", accept_len);
                        }
                        spec_total_count++;

                        // 6. Commit accepted path into GDN rec_state / conv_state.
                        if (branch_mode) {
                            model.commit_tree_gdn_path(accepted_path.data(), (int)accepted_path.size());
                        } else {
                            model.commit_tree_gdn_chain(accept_len);
                        }

                        // Counters.
                        if (accept_len == tree_budget)      tree_accept_full_count++;
                        else if (accept_len > 1)            tree_accept_partial_count++;
                        else                                 tree_reject_count++;
                        spec_accept_count += (accept_len - 1);

                        // 7. Emit accepted drafts (in slot order along the path),
                        //    then verify[emit_final_verify_slot] as the next-main
                        //    prediction.
                        bool stopped = false;
                        for (int i = 1; i < (int)accepted_path.size() && !stopped; i++) {
                            int slot = accepted_path[i];
                            // drafts[slot - 1] is the token that occupied slot i
                            // (drafts[0] occupies slot 1, drafts[1] occupies slot 2, ...)
                            if (emit_tok(drafts[slot - 1])) stopped = true;
                        }
                        if (!stopped) {
                            if (emit_tok(tree_h_argmax[emit_final_verify_slot])) stopped = true;
                        }

                        // MTP KV rollback.
                        //   chain: uploaded (tree_budget-1) drafts; accepted
                        //          (accept_len-1). Unused = tree_budget - accept_len.
                        //   branch: root forward_topk committed exactly 1 slot;
                        //          always roll it back so MTP KV is clean at the
                        //          next step (cold-start the following iter's
                        //          MTP). This avoids the hole problem when the
                        //          accepted branch didn't come from the KV-cached
                        //          top-1.
                        if (branch_mode) {
                            mtp.kv_rollback(1);
                        } else {
                            int unused_drafts = tree_budget - accept_len;
                            if (unused_drafts > 0) mtp.kv_rollback(unused_drafts);
                        }
                        step += accept_len;
                        if (stopped) break;
                    } else

                    // ===================== MTP K=2 speculative decoding =====================
                    // After main produced max_idx (= t_{step+1}) and we hold h[step] in norm_buf,
                    // draft two tokens by self-chaining the MTP head (draft1 from h_main, draft2
                    // from h_final_draft1). Then run a three-stream N=3 batched forward at
                    // positions [step+1, step+2, step+3] with inputs [max_idx, draft1, draft2].
                    // Accept logic:
                    //   verify_a == draft1 && verify_b == draft2 → accept both, provisional verify_c
                    //   verify_a == draft1                       → accept draft1 only
                    //   else                                      → reject both
                    if (slot == 0 && spec_k2_enabled && sampler.grammar == nullptr) {
                        auto sd0 = prof_now();
                        // 1. Self-chain draft: MTP(norm_buf, max_idx, step+1) → draft1, h_final_draft1
                        //    Then MTP(h_final_draft1, draft1, step+2) → draft2.
                        int draft1 = mtp.forward_with_state(norm_buf, max_idx, step + 1, h_final_draft1);
                        int draft2 = mtp.forward(h_final_draft1, draft1, step + 2);
                        cudaSetDevice(last_gpu);
                        if (do_gen_prof) g_mtp += prof_sync_ms(sd0, last_gpu);

                        // 2. Embed all three tokens onto GPU 0
                        cudaSetDevice(0);
                        auto se0 = prof_now();
                        if (embd_t->type == GGML_TYPE_Q8_0) {
                            dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0],   max_idx, H);
                            dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half_b[0], draft1,  H);
                            dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half_c[0], draft2,  H);
                        } else if (embd_t->type == GGML_TYPE_Q5_K) {
                            dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0],   max_idx, H);
                            dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half_b[0], draft1,  H);
                            dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half_c[0], draft2,  H);
                        } else if (embd_t->type == GGML_TYPE_Q6_K) {
                            dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0],   max_idx, H);
                            dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half_b[0], draft1,  H);
                            dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half_c[0], draft2,  H);
                        }
                        half_to_float_kernel<<<(H+255)/256, 256>>>(gpu_hidden_half[0],   gpu_hidden[0],   H);
                        half_to_float_kernel<<<(H+255)/256, 256>>>(gpu_hidden_half_b[0], gpu_hidden_b[0], H);
                        half_to_float_kernel<<<(H+255)/256, 256>>>(gpu_hidden_half_c[0], gpu_hidden_c[0], H);
                        if (do_gen_prof) g_embed += prof_sync_ms(se0, 0);

                        // 3. N=3 batched forward across all 64 layers
                        float* h_a = gpu_hidden[0];
                        float* h_b = gpu_hidden_b[0];
                        float* h_c = gpu_hidden_c[0];
                        for (int layer = 0; layer < model.cfg.num_layers; layer++) {
                            int g_l = gpu_model.layer_gpu[layer];
                            int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer - 1];
                            if (g_l != prev_g) {
                                auto sx0 = prof_now();
                                cudaSetDevice(prev_g);
                                cudaMemcpy(host_transfer,   h_a, H * sizeof(float), cudaMemcpyDeviceToHost);
                                cudaMemcpy(host_transfer_b, h_b, H * sizeof(float), cudaMemcpyDeviceToHost);
                                cudaMemcpy(host_transfer_c, h_c, H * sizeof(float), cudaMemcpyDeviceToHost);
                                cudaSetDevice(g_l);
                                cudaMemcpy(gpu_hidden[g_l],   host_transfer,   H * sizeof(float), cudaMemcpyHostToDevice);
                                cudaMemcpy(gpu_hidden_b[g_l], host_transfer_b, H * sizeof(float), cudaMemcpyHostToDevice);
                                cudaMemcpy(gpu_hidden_c[g_l], host_transfer_c, H * sizeof(float), cudaMemcpyHostToDevice);
                                h_a = gpu_hidden[g_l];
                                h_b = gpu_hidden_b[g_l];
                                h_c = gpu_hidden_c[g_l];
                                if (do_gen_prof) g_xfer += prof_sync_ms(sx0, g_l);
                            } else {
                                cudaSetDevice(g_l);
                            }
                            auto sa0 = prof_now();
                            bool is_attn_s = model.is_attn_layer(layer);
                            if (is_attn_s)
                                model.forward_attn_n3(layer, h_a, h_b, h_c, step + 1, step + 2, step + 3, 0);
                            else
                                model.forward_gdn_n3(layer, h_a, h_b, h_c, 0);
                            if (do_gen_prof) {
                                double ms = prof_sync_ms(sa0, g_l);
                                if (is_attn_s) g_attn += ms; else g_gdn += ms;
                            }
                            auto sm0 = prof_now();
                            if (model.layer_is_moe[layer]) {
                                model.forward_moe(layer, h_a, 0);
                                model.forward_moe(layer, h_b, 0);
                                model.forward_moe(layer, h_c, 0);
                            } else {
                                model.forward_mlp_n3(layer, h_a, h_b, h_c, 0);
                            }
                            if (do_gen_prof) g_mlp += prof_sync_ms(sm0, g_l);
                        }

                        // 4. Output norm + lm_head for all three via n3 helpers
                        cudaSetDevice(last_gpu);
                        cudaDeviceSynchronize();
                        auto sl0 = prof_now();
                        float_to_half_kernel<<<(H+255)/256, 256>>>(h_a, gpu_hidden_half[last_gpu],   H);
                        float_to_half_kernel<<<(H+255)/256, 256>>>(h_b, gpu_hidden_half_b[last_gpu], H);
                        float_to_half_kernel<<<(H+255)/256, 256>>>(h_c, gpu_hidden_half_c[last_gpu], H);
                        if (out_norm_t->type == GGML_TYPE_F32) {
                            rms_norm_f32w_n3(norm_buf, norm_buf_b, norm_buf_c,
                                             gpu_hidden_half[last_gpu], gpu_hidden_half_b[last_gpu], gpu_hidden_half_c[last_gpu],
                                             (float*)out_norm_t->data, H, model.cfg.rms_norm_eps, 0);
                        } else {
                            rms_norm_n3(norm_buf, norm_buf_b, norm_buf_c,
                                        gpu_hidden_half[last_gpu], gpu_hidden_half_b[last_gpu], gpu_hidden_half_c[last_gpu],
                                        (half*)out_norm_t->data, H, model.cfg.rms_norm_eps, 0);
                        }
                        qi_logits.quantize(norm_buf,     H, 0);
                        qi_logits_b.quantize(norm_buf_b, H, 0);
                        qi_logits_c.quantize(norm_buf_c, H, 0);
                        quant_gemv_n3(out_w->data, out_w->type,
                                      norm_buf,   norm_buf_b, norm_buf_c,
                                      logits_buf, logits_buf_b, logits_buf_c,
                                      H, V, &qi_logits, &qi_logits_b, &qi_logits_c, 0);

                        // 5. Argmax all three (top-2 if MTP_ACCEPT_TOP2 set)
                        static const bool mtp_accept_top2 = getenv("MTP_ACCEPT_TOP2") != nullptr;
                        if (mtp_accept_top2) {
                            argmax_top2_half_kernel<<<1, 1024>>>(logits_buf,   V, d_argmax);
                            argmax_top2_half_kernel<<<1, 1024>>>(logits_buf_b, V, d_argmax_b);
                            argmax_top2_half_kernel<<<1, 1024>>>(logits_buf_c, V, d_argmax_c);
                            cudaMemcpy(h_argmax_pinned,   d_argmax,   2*sizeof(int), cudaMemcpyDeviceToHost);
                            cudaMemcpy(h_argmax_pinned_b, d_argmax_b, 2*sizeof(int), cudaMemcpyDeviceToHost);
                            cudaMemcpy(h_argmax_pinned_c, d_argmax_c, 2*sizeof(int), cudaMemcpyDeviceToHost);
                        } else {
                            argmax_half_kernel<<<1, 1024>>>(logits_buf,   V, d_argmax);
                            argmax_half_kernel<<<1, 1024>>>(logits_buf_b, V, d_argmax_b);
                            argmax_half_kernel<<<1, 1024>>>(logits_buf_c, V, d_argmax_c);
                            cudaMemcpy(h_argmax_pinned,   d_argmax,   sizeof(int), cudaMemcpyDeviceToHost);
                            cudaMemcpy(h_argmax_pinned_b, d_argmax_b, sizeof(int), cudaMemcpyDeviceToHost);
                            cudaMemcpy(h_argmax_pinned_c, d_argmax_c, sizeof(int), cudaMemcpyDeviceToHost);
                        }
                        int verify_a = h_argmax_pinned[0];    // main's prediction at step+2
                        int verify_b = h_argmax_pinned_b[0];  // main's prediction at step+3
                        int verify_c = h_argmax_pinned_c[0];  // provisional at step+4
                        if (do_gen_prof) {
                            g_logits += prof_sync_ms(sl0, last_gpu);
                            g_spec_iters++;
                        }

                        // Top-2 softer accept: draft is accepted if it matches
                        // main's top-1 OR top-2 argmax (opt-in via MTP_ACCEPT_TOP2).
                        bool accept_draft1 = (verify_a == draft1)
                            || (mtp_accept_top2 && h_argmax_pinned[1]   == draft1);
                        bool accept_draft2 = (verify_b == draft2)
                            || (mtp_accept_top2 && h_argmax_pinned_b[1] == draft2);
                        if (spec_total_count < 3)
                            printf("[SPEC-K2] step=%d main=%d d1=%d d2=%d va=%d vb=%d vc=%d %s\n",
                                   step, max_idx, draft1, draft2, verify_a, verify_b, verify_c,
                                   accept_draft1 ? (accept_draft2 ? "ACCEPT_BOTH" : "ACCEPT_A") : "REJECT");
                        spec_total_count++;

                        if (accept_draft1 && accept_draft2) {
                            spec_k2_accept_ab_count++;
                            spec_accept_count += 2;  // two drafts accepted this iter
                            if (emit_tok(draft1)) break;
                            if (emit_tok(draft2)) break;
                            if (emit_tok(verify_c)) break;
                            step += 3;  // skip step+1..step+3 (all consumed)
                        } else if (accept_draft1) {
                            spec_k2_accept_a_count++;
                            spec_accept_count += 1;
                            // Keep draft1 + main's verify_b (real prediction at step+3).
                            if (emit_tok(draft1)) break;
                            if (emit_tok(verify_b)) break;
                            // Roll back GDN to post-draft1 (=post-token-B). KV: draft2 slot stays
                            // stale (not read since next main forward runs at step+3 onwards).
                            // MTP KV: one draft (draft2) was appended after draft1; roll back one.
                            model.restore_gdn_states_b(0);
                            mtp.kv_rollback(1);
                            step += 2;  // consumed step+1 (main) and step+2 (draft1 accepted)
                        } else {
                            spec_k2_reject_count++;
                            // Reject both drafts — commit main's verify_a (= prediction at step+2).
                            if (emit_tok(verify_a)) break;
                            // Roll back GDN to post-main (slot A). MTP: two drafts appended; roll both back.
                            model.restore_gdn_states(0);
                            mtp.kv_rollback(2);
                            step += 1;
                        }
                    } else if (slot == 0 && spec_enabled && sampler.grammar == nullptr) {
                        auto sd0 = prof_now();
                        // 1. MTP draft
                        int mtp_draft = mtp.forward(norm_buf, max_idx, step + 1);
                        cudaSetDevice(last_gpu);
                        if (do_gen_prof) g_mtp += prof_sync_ms(sd0, last_gpu);

                        // 2. Embed both tokens onto GPU 0
                        cudaSetDevice(0);
                        auto se0 = prof_now();
                        if (embd_t->type == GGML_TYPE_Q8_0) {
                            dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0],   max_idx,   H);
                            dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half_b[0], mtp_draft, H);
                        } else if (embd_t->type == GGML_TYPE_Q5_K) {
                            dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0],   max_idx,   H);
                            dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half_b[0], mtp_draft, H);
                        } else if (embd_t->type == GGML_TYPE_Q6_K) {
                            dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0],   max_idx,   H);
                            dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half_b[0], mtp_draft, H);
                        }
                        half_to_float_kernel<<<(H+255)/256, 256>>>(gpu_hidden_half[0],   gpu_hidden[0],   H);
                        half_to_float_kernel<<<(H+255)/256, 256>>>(gpu_hidden_half_b[0], gpu_hidden_b[0], H);
                        if (do_gen_prof) g_embed += prof_sync_ms(se0, 0);

                        // 3. Run all 64 layers in N=2 batched mode
                        float* h_a = gpu_hidden[0];
                        float* h_b = gpu_hidden_b[0];
                        for (int layer = 0; layer < model.cfg.num_layers; layer++) {
                            int g_l = gpu_model.layer_gpu[layer];
                            int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer - 1];
                            if (g_l != prev_g) {
                                auto sx0 = prof_now();
                                cudaSetDevice(prev_g);
                                cudaMemcpy(host_transfer,   h_a, H * sizeof(float), cudaMemcpyDeviceToHost);
                                cudaMemcpy(host_transfer_b, h_b, H * sizeof(float), cudaMemcpyDeviceToHost);
                                cudaSetDevice(g_l);
                                cudaMemcpy(gpu_hidden[g_l],   host_transfer,   H * sizeof(float), cudaMemcpyHostToDevice);
                                cudaMemcpy(gpu_hidden_b[g_l], host_transfer_b, H * sizeof(float), cudaMemcpyHostToDevice);
                                h_a = gpu_hidden[g_l];
                                h_b = gpu_hidden_b[g_l];
                                if (do_gen_prof) g_xfer += prof_sync_ms(sx0, g_l);
                            } else {
                                cudaSetDevice(g_l);
                            }
                            auto sa0 = prof_now();
                            bool is_attn_s = model.is_attn_layer(layer);
                            if (is_attn_s)
                                model.forward_attn_n2(layer, h_a, h_b, step + 1, step + 2, 0);
                            else
                                model.forward_gdn_n2(layer, h_a, h_b, 0);
                            if (do_gen_prof) {
                                double ms = prof_sync_ms(sa0, g_l);
                                if (is_attn_s) g_attn += ms; else g_gdn += ms;
                            }
                            auto sm0 = prof_now();
                            if (model.layer_is_moe[layer]) {
                                model.forward_moe(layer, h_a, 0);
                                model.forward_moe(layer, h_b, 0);
                            } else {
                                model.forward_mlp_n2(layer, h_a, h_b, 0);
                            }
                            if (do_gen_prof) g_mlp += prof_sync_ms(sm0, g_l);
                        }

                        // 4. Output norm + lm_head for both
                        cudaSetDevice(last_gpu);
                        cudaDeviceSynchronize();
                        auto sl0 = prof_now();
                        float_to_half_kernel<<<(H+255)/256, 256>>>(h_a, gpu_hidden_half[last_gpu],   H);
                        float_to_half_kernel<<<(H+255)/256, 256>>>(h_b, gpu_hidden_half_b[last_gpu], H);
                        if (out_norm_t->type == GGML_TYPE_F32) {
                            rms_norm_f32w_n2(norm_buf, norm_buf_b,
                                             gpu_hidden_half[last_gpu], gpu_hidden_half_b[last_gpu],
                                             (float*)out_norm_t->data, H, model.cfg.rms_norm_eps, 0);
                        } else {
                            rms_norm_n2(norm_buf, norm_buf_b,
                                        gpu_hidden_half[last_gpu], gpu_hidden_half_b[last_gpu],
                                        (half*)out_norm_t->data, H, model.cfg.rms_norm_eps, 0);
                        }
                        qi_logits.quantize(norm_buf,     H, 0);
                        qi_logits_b.quantize(norm_buf_b, H, 0);
                        quant_gemv_n2(out_w->data, out_w->type,
                                      norm_buf,   norm_buf_b,
                                      logits_buf, logits_buf_b,
                                      H, V, &qi_logits, &qi_logits_b, 0);

                        // 5. Argmax both
                        argmax_half_kernel<<<1, 1024>>>(logits_buf,   V, d_argmax);
                        argmax_half_kernel<<<1, 1024>>>(logits_buf_b, V, d_argmax_b);
                        cudaMemcpy(h_argmax_pinned,   d_argmax,   sizeof(int), cudaMemcpyDeviceToHost);
                        cudaMemcpy(h_argmax_pinned_b, d_argmax_b, sizeof(int), cudaMemcpyDeviceToHost);
                        int max_idx_a = h_argmax_pinned[0];   // verify of mtp_draft (= main's pred at step+2)
                        int max_idx_b = h_argmax_pinned_b[0]; // provisional t_{step+3}
                        if (do_gen_prof) {
                            g_logits += prof_sync_ms(sl0, last_gpu);
                            g_spec_iters++;
                        }

                        bool spec_accept = (max_idx_a == mtp_draft);
                        if (spec_total_count < 3)
                            printf("[SPEC] step=%d main=%d draft=%d verify=%d %s prov=%d\n",
                                   step, max_idx, mtp_draft, max_idx_a,
                                   spec_accept ? "ACCEPT" : "reject", max_idx_b);
                        spec_total_count++;
                        if (spec_accept) spec_accept_count++;

                        if (spec_accept) {
                            // Append the accepted draft + the provisional next token.
                            if (emit_tok(mtp_draft)) break;
                            if (emit_tok(max_idx_b)) break;
                            // Skip outer loop iters at step+1 and step+2 (we processed them
                            // via the batched verify). norm_buf gets re-derived next iter
                            // from main's forward at the new step.
                            step += 2;
                        } else {
                            // Reject: keep main's verify (max_idx_a = t_{step+2}_main).
                            if (emit_tok(max_idx_a)) break;
                            // Rollback GDN past the second (rejected) token. The snapshot
                            // was taken inside forward_gdn_n2 right after the first token's
                            // GDN update, so restoring brings state to "post-(step+1)".
                            model.restore_gdn_states(0);
                            // KV[step+2] in main's attention cache is wrong but will be
                            // overwritten when next iter's main runs at step+2. No explicit
                            // KV rollback needed (positional addressing).
                            // Skip outer loop iter at step+1 (we processed it correctly via
                            // the verify).
                            step += 1;
                            // norm_buf now needs to be h[step+1] = norm_buf (the verify path's
                            // hidden), which is already in norm_buf. Done.
                        }
                    }
                } else {
                    // step == prompt_ids.size() - 1: this is actually the
                    // first sampled output token (the prediction made from
                    // the last prompt token's hidden). It's a real
                    // generation token, so fire on_token for streaming.
                    generated.push_back(max_idx);
                    if (on_token) on_token(max_idx);
                }
            }

            // Batched gen handoff: caller wants control after the very first
            // token. State is now: KV+GDN at slot_pos=prompt_len, first_tok in
            // generated[0] and on_token already fired.
            if (batched_handoff_first_tok && step == (int)prompt_ids.size() - 1
                && !generated.empty()) {
                *batched_handoff_first_tok = generated.back();
                if (batched_handoff_slot_pos) *batched_handoff_slot_pos = (int)prompt_ids.size();
                break;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double prefill_ms = got_first_tok ? std::chrono::duration<double, std::milli>(t_first - t0).count() : 0;
        double gen_ms = got_first_tok ? std::chrono::duration<double, std::milli>(t1 - t_first).count() : std::chrono::duration<double, std::milli>(t1 - t0).count();
        int gen_count = (int)generated.size();
        printf("[API] prefill %zu tok %.1fs | gen %d tok %.1f t/s (%.1fs)\n",
               prompt_ids.size(), prefill_ms / 1000.0,
               gen_count, gen_count > 1 ? (gen_count - 1) * 1000.0 / gen_ms : 0, gen_ms / 1000.0);
        // Dump the full decoded output (think + answer) so operators can
        // inspect model chain-of-thought + final reply from the server log.
        if (!generated.empty()) {
            std::string full_text = tok.decode(generated);
            printf("[API FULL TEXT]\n%s\n[/API FULL TEXT]\n", full_text.c_str());
            // Diag: dump raw token IDs (first 16, last 8) so we can see
            // when decode produces empty text but generated has entries.
            int n = (int)generated.size();
            printf("[API TOK_IDS n=%d] head:", n);
            for (int i = 0; i < std::min(16, n); i++) printf(" %d", generated[i]);
            if (n > 24) {
                printf(" ... tail:");
                for (int i = std::max(16, n - 8); i < n; i++) printf(" %d", generated[i]);
            }
            printf("\n");
        }
        if (mtp_loaded && mtp_total_count > 0) {
            printf("[MTP] accept %lld / %lld = %.1f%%\n",
                   mtp_accept_count, mtp_total_count,
                   100.0 * mtp_accept_count / mtp_total_count);
        }
        if (spec_enabled && spec_total_count > 0) {
            double avg = 1.0 + (double)spec_accept_count / spec_total_count;
            printf("[SPEC] accept %lld / %lld = %.1f%%  (per-iter avg tokens %.2f)\n",
                   spec_accept_count, spec_total_count,
                   100.0 * spec_accept_count / spec_total_count, avg);
            if (spec_k2_enabled) {
                long long tot_k2 = spec_k2_accept_ab_count + spec_k2_accept_a_count + spec_k2_reject_count;
                if (tot_k2 > 0) {
                    printf("[SPEC-K2] accept_both=%lld (%.1f%%) accept_a=%lld (%.1f%%) reject=%lld (%.1f%%)\n",
                           spec_k2_accept_ab_count, 100.0 * spec_k2_accept_ab_count / tot_k2,
                           spec_k2_accept_a_count,  100.0 * spec_k2_accept_a_count / tot_k2,
                           spec_k2_reject_count,    100.0 * spec_k2_reject_count / tot_k2);
                }
            }
        }
        if (dflash_enabled) {
            long long iters = tree_accept_full_count + tree_accept_partial_count + tree_reject_count;
            double al = iters ? 1.0 + (double)spec_accept_count / (double)iters : 0.0;
            printf("[DFLASH] budget=%d iters=%lld accepted_drafts=%lld  AL(tokens/iter)=%.2f  "
                   "full=%lld partial=%lld reject=%lld\n",
                   tree_budget, iters, spec_accept_count, al,
                   tree_accept_full_count, tree_accept_partial_count, tree_reject_count);
        }
        if (do_gen_prof) {
            double total = g_embed + g_xfer + g_attn + g_gdn + g_mlp + g_logits + g_mtp;
            printf("[GEN PROF] steps=%d spec_iters=%d  total=%.1fms  "
                   "embed=%.1f(%.0f%%) xfer=%.1f(%.0f%%) attn=%.1f(%.0f%%) "
                   "gdn=%.1f(%.0f%%) mlp=%.1f(%.0f%%) logits=%.1f(%.0f%%) mtp=%.1f(%.0f%%)\n",
                   g_steps, g_spec_iters, total,
                   g_embed, 100.0*g_embed/total, g_xfer, 100.0*g_xfer/total,
                   g_attn, 100.0*g_attn/total, g_gdn, 100.0*g_gdn/total,
                   g_mlp, 100.0*g_mlp/total, g_logits, 100.0*g_logits/total,
                   g_mtp, 100.0*g_mtp/total);
            double tv = g_tv_xfer + g_tv_attn + g_tv_gdn + g_tv_mlp + g_tv_cap;
            printf("[GEN PROF TREE-VERIFY] total=%.1fms  xfer=%.1f(%.0f%%) attn=%.1f(%.0f%%) "
                   "gdn=%.1f(%.0f%%) mlp=%.1f(%.0f%%) capture=%.1f(%.0f%%)  (= the 16-tok 27B verify, was untimed)\n",
                   tv, g_tv_xfer, 100.0*g_tv_xfer/tv, g_tv_attn, 100.0*g_tv_attn/tv,
                   g_tv_gdn, 100.0*g_tv_gdn/tv, g_tv_mlp, 100.0*g_tv_mlp/tv,
                   g_tv_cap, 100.0*g_tv_cap/tv);
            fflush(stdout);
        }
        if (g_profile_attn && g_pt_calls > 0) {
            double pt_sub = g_pt_qkvr_ms + g_pt_kvwrite_ms + g_pt_attn_ms + g_pt_oproj_ms;
            printf("[ATTN PT PROF] calls=%ld  sub_sum=%.1fms  "
                   "qkvr=%.1fms(%.0f%%) kvwrite=%.1fms(%.0f%%) "
                   "attn_compute=%.1fms(%.0f%%) oproj=%.1fms(%.0f%%)\n",
                   g_pt_calls, pt_sub,
                   g_pt_qkvr_ms,    pt_sub > 0 ? 100.0*g_pt_qkvr_ms/pt_sub    : 0,
                   g_pt_kvwrite_ms, pt_sub > 0 ? 100.0*g_pt_kvwrite_ms/pt_sub : 0,
                   g_pt_attn_ms,    pt_sub > 0 ? 100.0*g_pt_attn_ms/pt_sub    : 0,
                   g_pt_oproj_ms,   pt_sub > 0 ? 100.0*g_pt_oproj_ms/pt_sub   : 0);
            fflush(stdout);
        }
        if (g_profile_mlp && g_mlp_calls > 0) {
            double mlp_sub = g_mlp_norm_ms + g_mlp_q1_ms + g_mlp_gate_ms +
                             g_mlp_up_ms + g_mlp_silu_ms + g_mlp_q2_ms +
                             g_mlp_down_ms + g_mlp_resi_ms;
            printf("[MLP PROF] calls=%ld sub_sum=%.1fms  "
                   "norm=%.1f(%.0f%%) q1=%.1f(%.0f%%) gate=%.1f(%.0f%%) "
                   "up=%.1f(%.0f%%) silu=%.1f(%.0f%%) q2=%.1f(%.0f%%) "
                   "down=%.1f(%.0f%%) resi=%.1f(%.0f%%)\n",
                   g_mlp_calls, mlp_sub,
                   g_mlp_norm_ms, mlp_sub > 0 ? 100.0*g_mlp_norm_ms/mlp_sub : 0,
                   g_mlp_q1_ms,   mlp_sub > 0 ? 100.0*g_mlp_q1_ms/mlp_sub   : 0,
                   g_mlp_gate_ms, mlp_sub > 0 ? 100.0*g_mlp_gate_ms/mlp_sub : 0,
                   g_mlp_up_ms,   mlp_sub > 0 ? 100.0*g_mlp_up_ms/mlp_sub   : 0,
                   g_mlp_silu_ms, mlp_sub > 0 ? 100.0*g_mlp_silu_ms/mlp_sub : 0,
                   g_mlp_q2_ms,   mlp_sub > 0 ? 100.0*g_mlp_q2_ms/mlp_sub   : 0,
                   g_mlp_down_ms, mlp_sub > 0 ? 100.0*g_mlp_down_ms/mlp_sub : 0,
                   g_mlp_resi_ms, mlp_sub > 0 ? 100.0*g_mlp_resi_ms/mlp_sub : 0);
            fflush(stdout);
        }
        if (g_profile_gdn && g_gdn_calls > 0) {
            double gdn_sub = g_gdn_norm_ms + g_gdn_proj_ms + g_gdn_conv_ms +
                             g_gdn_recur_ms + g_gdn_rmsg_ms + g_gdn_oproj_ms + g_gdn_resi_ms;
            printf("[GDN PROF] calls=%ld sub_sum=%.1fms  "
                   "norm=%.1f(%.0f%%) proj4gemv=%.1f(%.0f%%) conv1d=%.1f(%.0f%%) "
                   "recur=%.1f(%.0f%%) rmsg=%.1f(%.0f%%) oproj=%.1f(%.0f%%) resi=%.1f(%.0f%%)\n",
                   g_gdn_calls, gdn_sub,
                   g_gdn_norm_ms,  gdn_sub > 0 ? 100.0*g_gdn_norm_ms/gdn_sub  : 0,
                   g_gdn_proj_ms,  gdn_sub > 0 ? 100.0*g_gdn_proj_ms/gdn_sub  : 0,
                   g_gdn_conv_ms,  gdn_sub > 0 ? 100.0*g_gdn_conv_ms/gdn_sub  : 0,
                   g_gdn_recur_ms, gdn_sub > 0 ? 100.0*g_gdn_recur_ms/gdn_sub : 0,
                   g_gdn_rmsg_ms,  gdn_sub > 0 ? 100.0*g_gdn_rmsg_ms/gdn_sub  : 0,
                   g_gdn_oproj_ms, gdn_sub > 0 ? 100.0*g_gdn_oproj_ms/gdn_sub : 0,
                   g_gdn_resi_ms,  gdn_sub > 0 ? 100.0*g_gdn_resi_ms/gdn_sub  : 0);
            fflush(stdout);
        }
        if (do_fa_prof) {
            unsigned long long sum[5] = {0,0,0,0,0};
            for (int g = 0; g < n_gpus; g++) {
                cudaSetDevice(g);
                unsigned long long buf[5];
                cudaMemcpyFromSymbol(buf, g_fa_phase_cyc, sizeof(buf));
                for (int i = 0; i < 5; i++) sum[i] += buf[i];
            }
            unsigned long long phase_sum = sum[0] + sum[1] + sum[2] + sum[3];
            if (phase_sum > 0) {
                printf("[FA PHASE] tiles=%llu  total_cyc=%llu  "
                       "decode=%.0f%%  score=%.0f%%  softmax=%.0f%%  value=%.0f%%\n",
                       sum[4], phase_sum,
                       100.0 * sum[0] / phase_sum,
                       100.0 * sum[1] / phase_sum,
                       100.0 * sum[2] / phase_sum,
                       100.0 * sum[3] / phase_sum);
                fflush(stdout);
            }
        }

        if (out_completion_tokens) *out_completion_tokens = (int)generated.size();
        return tok.decode(generated);
    };

    // ──── Phase C: forward_step_batched ────────────────────────────────────
    // Process one new token from each of N slots in a single forward pass.
    // Returns the sampled token id (greedy / temp=0) for each slot in
    // out_tokens[N]. Only handles plain greedy gen — no spec/MTP/DDTree —
    // since the spec scaffolding is still single-slot. Caller is responsible
    // for embedding decisions, stop-tag detection, and KV slot lifetime.
    //
    // Inputs (host-side):
    //   N            : number of active slots (1 ≤ N ≤ batch_cap)
    //   slot_ids[N]  : which slot each entry occupies (0..num_slots-1)
    //   token_ids[N] : the input token to embed for each slot's next forward
    //   slot_pos[N]  : the logical position in each slot at which to write
    //                  the new K/V (i.e. the # tokens already in the slot;
    //                  the K/V for this token lands at this index).
    //
    // Outputs:
    //   out_tokens[N] : sampled next token id per slot (host array)
    auto forward_step_batched = [&](int N,
                                    const int* slot_ids,
                                    const int* token_ids,
                                    const int* slot_pos,
                                    int* out_tokens) {
        if (N <= 0) return;
        // Build per-GPU host index buffers for slot_ids / slot_pos / dst_kv_pos.
        // dst_kv_pos[i] = kv_slot_offset(slot) + slot_pos[i]  (physical
        // KV index for the new token's K/V — cumulative for asymmetric caps).
        std::vector<int> dst_kv_host(N);
        for (int i = 0; i < N; i++) {
            dst_kv_host[i] = (int)model.kv_slot_offset(slot_ids[i]) + slot_pos[i];
        }
        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g);
            cudaMemcpyAsync(slot_ids_dev[g], slot_ids, N * sizeof(int),
                            cudaMemcpyHostToDevice);
            cudaMemcpyAsync(slot_pos_dev[g], slot_pos, N * sizeof(int),
                            cudaMemcpyHostToDevice);
            cudaMemcpyAsync(dst_kv_pos_dev[g], dst_kv_host.data(), N * sizeof(int),
                            cudaMemcpyHostToDevice);
        }
        cudaSetDevice(0);

        // 1. Embed N tokens onto GPU 0 → gpu_hidden_half_batch[0] [N, H] half,
        //    then convert to fp32 → gpu_hidden_batch[0] [N, H] fp32.
        for (int i = 0; i < N; i++) {
            int tid = token_ids[i];
            half* dst = gpu_hidden_half_batch[0] + (size_t)i * H;
            if (embd_t->type == GGML_TYPE_Q8_0)
                dequant_embd_q8_0_row<<<(H + 255) / 256, 256>>>(embd_t->data, dst, tid, H);
            else if (embd_t->type == GGML_TYPE_Q5_K)
                dequant_embd_q5k_row_v2<<<(H + 255) / 256, 256>>>(embd_t->data, dst, tid, H);
            else if (embd_t->type == GGML_TYPE_Q6_K)
                dequant_embd_q6k_row_v2<<<(H + 255) / 256, 256>>>(embd_t->data, dst, tid, H);
        }
        {
            int total = N * H;
            half_to_float_kernel<<<(total + 255) / 256, 256>>>(
                gpu_hidden_half_batch[0], gpu_hidden_batch[0], total);
        }

        // 2. Layer loop with cross-GPU transfers of the [N, H] fp32 block.
        float* h = gpu_hidden_batch[0];
        for (int layer = 0; layer < model.cfg.num_layers; layer++) {
            int g = gpu_model.layer_gpu[layer];
            int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer - 1];
            if (g != prev_g) {
                cudaSetDevice(prev_g); cudaDeviceSynchronize();
                cudaMemcpy(host_batch_transfer, h,
                           (size_t)N * H * sizeof(float), cudaMemcpyDeviceToHost);
                cudaSetDevice(g);
                cudaMemcpy(gpu_hidden_batch[g], host_batch_transfer,
                           (size_t)N * H * sizeof(float), cudaMemcpyHostToDevice);
                h = gpu_hidden_batch[g];
            } else {
                cudaSetDevice(g);
            }
            bool is_attn = model.is_attn_layer(layer);
            if (is_attn) {
                model.forward_attn_step_batched(layer, h, N,
                                                slot_ids, slot_pos,
                                                dst_kv_pos_dev[g], slot_pos_dev[g], 0);
            } else {
                model.forward_gdn_step_batched(layer, h, N, slot_ids, 0);
            }
            // forward_mlp_chunk handles N rows directly (stateless across slots).
            if (model.layer_is_moe[layer]) model.forward_moe_chunk(layer, h, N, 0);
            else                           model.forward_mlp_chunk(layer, h, N, 0);
        }

        // 3. Final RMSNorm + LM head + per-slot argmax on last_gpu.
        cudaSetDevice(last_gpu);
        cudaDeviceSynchronize();
        // hidden_batch lives on last_gpu (pipeline ended there).
        // RMSNorm rows=N → norm_batch [N, H] half.
        if (out_norm_t->type == GGML_TYPE_F32) {
            rms_norm_f32in_f32w(norm_batch, h, (float*)out_norm_t->data,
                                N, H, model.cfg.rms_norm_eps, 0);
        } else {
            rms_norm_f32in(norm_batch, h, (half*)out_norm_t->data,
                           N, H, model.cfg.rms_norm_eps, 0);
        }
        // Quantize all N normed rows once, then batched LM head GEMV.
        // qi_logits is a serve_qwen-local QuantInput; quantize_chunk handles
        // the N×H flat span (block boundaries align with token boundaries).
        // LM head weights live on last_gpu (output.weight, not the embedding
        // tensor which is on GPU 0).
        qi_logits.quantize_chunk(norm_batch, H, N, 0);
        quant_gemv_chunk(out_w->data, out_w->type,
                         qi_logits.q8_buf,
                         logits_batch, H, V, N, 0);
        // Per-slot greedy argmax. The reduce inside argmax_half_kernel is
        // single-block, so launching N kernels is cheap (each is ~50 µs).
        for (int i = 0; i < N; i++) {
            argmax_half_kernel<<<1, 1024>>>(logits_batch + (size_t)i * V, V,
                                            d_argmax_batch + i);
        }
        cudaMemcpy(h_argmax_batch_pinned, d_argmax_batch, N * sizeof(int),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++) out_tokens[i] = h_argmax_batch_pinned[i];
    };

    // ──── Continuous batching scheduler ────────────────────────────────────
    // Each HTTP request is wrapped in a Sequence and submitted to the
    // GenScheduler, which maintains a worker thread per slot. Workers pop
    // pending sequences, allocate a slot, and call run_fn(seq, slot) — which
    // forwards into generate_impl with that slot threaded through the model.
    //
    // GPU execution is currently serialized behind `sched.forward_mutex()`
    // so the per-GPU scratch buffers (bufs[g], attn_bufs[g], gdn_bufs[g])
    // remain race-free. Phase B will replace this with a batched_step that
    // processes N slots in a single forward and amortizes weight loads,
    // unlocking real concurrent throughput.
    // ──── Per-slot batched gen state + dedicated gen-loop thread ──────────
    // When num_slots > 1 we want concurrent requests to share a single
    // batched forward per token. The split:
    //   - Prefill (chunked) + first-token sample : single-slot, runs under the
    //     scheduler's forward_mutex (legacy generate_impl with handoff break).
    //   - Per-token gen loop                    : driven by the gen-loop
    //     thread below, which collects all "active" slots each iter and
    //     calls forward_step_batched once across them. One token per slot
    //     per iter; per-slot stop conditions handled in the distribute step.
    // Spec / MTP / DFlash / DDTree paths only apply to slot 0 single-stream
    // mode and are skipped here (the batched gen loop is plain greedy).
    struct SlotGenState {
        bool active = false;
        int  next_tok = -1;       // token to embed at the next batched step
        int  slot_pos = 0;        // physical KV write position for that step
        int  output_tokens = 0;
        int  max_gen = 0;
        bool in_think = false;
        std::vector<int>          generated;
        std::function<void(int)>  on_token;
        // Pointer to the owning Sequence's cancelled atomic. The gen loop
        // checks this each iter and stops the slot if the client is gone,
        // so a queued-up burst of disconnected requests doesn't burn GPU
        // generating tokens nobody will see.
        std::atomic<bool>*        cancelled = nullptr;
        // Completion signaling (the run_fn waits on this after handing the
        // sequence off to the gen loop).
        std::mutex                done_mu;
        std::condition_variable   done_cv;
        bool                      done = false;
    };
    std::vector<std::unique_ptr<SlotGenState>> slot_gen_state;
    slot_gen_state.reserve(num_slots);
    for (int i = 0; i < num_slots; i++) slot_gen_state.emplace_back(new SlotGenState());

    std::mutex              gen_loop_mu;
    std::condition_variable gen_loop_cv;
    std::atomic<bool>       gen_loop_stop{false};
    auto any_active = [&]() {
        for (auto& s : slot_gen_state) if (s->active) return true;
        return false;
    };

    // Dynamic slot-router (RANK 1): count of in-flight requests on slots
    // OTHER than slot 0. While >0, slot-0's legacy MTP fast path must yield
    // (router_yield_signal raised) so the batched loop can run both. Raised
    // at the TOP of run_fn for slot!=0 (before any forward_mutex acquisition),
    // lowered on that request's completion. When it drops back to 0 the next
    // lone slot-0 request goes back to full MTP single-stream.
    std::atomic<int> router_others_inflight{0};
    auto router_enter_other = [&]() {
        if (!dynamic_router_on) return;
        router_others_inflight.fetch_add(1, std::memory_order_acq_rel);
        router_yield_signal.store(true, std::memory_order_release);
    };
    auto router_leave_other = [&]() {
        if (!dynamic_router_on) return;
        if (router_others_inflight.fetch_sub(1, std::memory_order_acq_rel) == 1)
            router_yield_signal.store(false, std::memory_order_release);
    };

    std::thread gen_loop_thread;  // started below, after `sched` is in scope.

    qwen_engine::GenScheduler sched(num_slots,
        /*run_fn=*/[&](qwen_engine::Sequence& seq, int slot) {
            auto on_tok = [&](int tok_id) {
                if (seq.on_token) seq.on_token(tok_id);
            };
            int completion_tokens = 0;
            std::string final_text;

            // Dynamic slot-router (RANK 1): any request NOT on slot 0 raises
            // the yield signal so slot-0's legacy MTP loop cooperatively bails
            // and lets the batched loop drive both. Raise BEFORE acquiring any
            // forward_mutex (slot-0 holds it for its whole MTP run, so we must
            // signal it to release before we can make progress). RAII guard
            // lowers the count when this request finishes.
            struct RouterGuard {
                bool engaged = false;
                std::function<void()> leave;
                ~RouterGuard() { if (engaged) leave(); }
            } router_guard;
            if (slot != 0) {
                router_enter_other();
                router_guard.engaged = true;
                router_guard.leave = router_leave_other;
            }

            // Grammar-constrained requests force the legacy single-stream path:
            // sampler is shared (one Sampler instance) so we can only bind one
            // grammar at a time, and the batched gen loop has its own argmax
            // that does not consult grammar. Forward mutex serializes other
            // requests while we generate.
            if (seq.grammar) {
                std::lock_guard<std::mutex> lk(sched.forward_mutex());
                sampler.grammar = seq.grammar.get();
                try {
                    final_text = generate_impl(seq.prompt_ids, seq.max_tokens,
                                               seq.cached_prompt_tokens,
                                               &completion_tokens, on_tok, slot,
                                               nullptr, nullptr, &seq.cancelled);
                } catch (...) { sampler.grammar = nullptr; throw; }
                sampler.grammar = nullptr;
                if (seq.on_done) seq.on_done(std::move(final_text), completion_tokens);
                return;
            }

            if (num_slots <= 1) {
                // Single-slot legacy path: full generate_impl under mutex.
                std::lock_guard<std::mutex> lk(sched.forward_mutex());
                final_text = generate_impl(seq.prompt_ids, seq.max_tokens,
                                           seq.cached_prompt_tokens,
                                           &completion_tokens, on_tok, slot,
                                           nullptr, nullptr, &seq.cancelled);
                if (seq.on_done) seq.on_done(std::move(final_text), completion_tokens);
                return;
            }

            // Multi-slot fast path: when this is slot 0 AND no other slots are
            // currently active, route through the legacy generate_impl path so
            // we get MTP/spec speedup (~28 t/s instead of 20 t/s for batched
            // N=1). SlotManager allocates lowest-index first, so the first
            // arriving request always gets slot 0. Other slots only run
            // batched gen — they use forward_step_batched via the gen loop.
            //
            // Race: if another slot becomes active while we're mid-legacy, the
            // gen loop will block on forward_mutex until our legacy completes,
            // serializing them. Acceptable: MTP-accelerated slot-0 finishes
            // faster than a batched N=2 step would, and other slots queue.
            // QWEN_NO_LEGACY_FAST_PATH=1 disables this optimization (always
            // batched) — useful for benchmarking pure batched throughput.
            // Dynamic-router continuation state. When slot-0's legacy MTP loop
            // yields mid-generation (a 2nd request arrived), generate_impl hands
            // back the tokens it already emitted + the live (next_tok, slot_pos)
            // through these. We then seed the batched gen-loop with the FULL
            // history instead of clearing it. dyn_yielded gates that path.
            bool             dyn_yielded = false;
            int              dyn_yield_tok = -1;       // sentinel: not yielded
            int              dyn_yield_pos = 0;
            std::vector<int> dyn_yield_gen;
            int              dyn_yield_output_tokens = 0;
            bool             dyn_yield_in_think = false;

            if (slot == 0 && getenv("QWEN_NO_LEGACY_FAST_PATH") == nullptr) {
                bool alone;
                {
                    std::lock_guard<std::mutex> g(gen_loop_mu);
                    alone = true;
                    for (int s = 1; s < num_slots; s++) {
                        if (slot_gen_state[s]->active) { alone = false; break; }
                    }
                }
                if (alone) {
                    {
                        std::lock_guard<std::mutex> lk(sched.forward_mutex());
                        // Pass the dynamic-router yield hooks. generate_impl runs
                        // full MTP; if router_yield_signal fires mid-gen it breaks
                        // cleanly at a step boundary and fills dyn_yield_*.
                        final_text = generate_impl(seq.prompt_ids, seq.max_tokens,
                                                   seq.cached_prompt_tokens,
                                                   &completion_tokens, on_tok, slot,
                                                   nullptr, nullptr, &seq.cancelled,
                                                   &dyn_yield_tok, &dyn_yield_pos,
                                                   &dyn_yield_gen,
                                                   &dyn_yield_output_tokens,
                                                   &dyn_yield_in_think);
                    }
                    if (dyn_yield_tok < 0) {
                        // MTP ran to completion (no yield) — same as before.
                        if (seq.on_done) seq.on_done(std::move(final_text), completion_tokens);
                        return;
                    }
                    // Yielded mid-generation: fall through to register slot 0
                    // into the batched gen loop, seeded with the emitted history
                    // and the live continuation point. NO GDN restore — the live
                    // state already matches dyn_yield_tok @ dyn_yield_pos.
                    dyn_yielded = true;
                }
            }

            // Batched path: prefill + first-token sample under the mutex via
            // generate_impl's handoff hook, then drive remaining tokens via
            // the gen-loop thread. (When dyn_yielded, prefill+MTP already ran;
            // we just hand the continuation to the gen loop.)
            int first_tok = -1, slot_pos_after = 0;
            if (!dyn_yielded) {
                std::lock_guard<std::mutex> lk(sched.forward_mutex());
                generate_impl(seq.prompt_ids, seq.max_tokens,
                              seq.cached_prompt_tokens,
                              &completion_tokens, on_tok, slot,
                              &first_tok, &slot_pos_after, &seq.cancelled);
            } else {
                first_tok      = dyn_yield_tok;
                slot_pos_after = dyn_yield_pos;
            }
            // Reset per-slot gen state.
            auto& st = *slot_gen_state[slot];
            {
                std::lock_guard<std::mutex> lk(gen_loop_mu);
                st.active = false;
                st.done = false;
                st.on_token = on_tok;
                st.cancelled = &seq.cancelled;
                int prompt_len = (int)seq.prompt_ids.size();
                st.max_gen = seq.max_tokens > 0
                             ? seq.max_tokens
                             : std::max(64, model.slot_capacity(slot) - prompt_len - 64);
                if (dyn_yielded) {
                    // Seed with the FULL emitted history. on_token already fired
                    // for every token in dyn_yield_gen, so the gen loop must NOT
                    // re-stream them — it appends only NEW tokens from here.
                    st.generated     = std::move(dyn_yield_gen);
                    st.output_tokens = dyn_yield_output_tokens;
                    st.in_think      = dyn_yield_in_think;
                } else {
                    st.in_think = false;
                    st.output_tokens = 0;
                    st.generated.clear();
                }
            }
            // first_tok already pushed into generate_impl's local `generated`
            // and on_token fired. Replicate the bookkeeping here so our
            // SlotGenState matches the model's side.
            bool stop = false;
            if (dyn_yielded) {
                // dyn_yield_tok (== first_tok) is the LAST already-emitted token
                // and is ALREADY in st.generated + counted in output_tokens. The
                // gen loop embeds it next (next_tok) and writes its KV at
                // slot_pos; the token it samples becomes the NEXT new token. So
                // do not re-push or re-count here — just check the gen budget.
                if (st.output_tokens >= st.max_gen) stop = true;
            } else if (first_tok < 0) {
                // generate_impl took an early exit before sampling (max_tokens=0
                // or empty prompt). Treat as immediate stop.
                stop = true;
            } else if (first_tok == 248046 || first_tok == 248044 || first_tok == 248045) {
                stop = true;  // EOS as first sampled token: nothing to gen.
            } else {
                st.generated.push_back(first_tok);
                if (first_tok == 248068) st.in_think = true;
                if (first_tok == 248069) st.in_think = false;
                if (!st.in_think) st.output_tokens++;
                if (st.output_tokens >= st.max_gen) stop = true;
            }
            if (!stop) {
                {
                    std::lock_guard<std::mutex> lk(gen_loop_mu);
                    st.next_tok = first_tok;
                    st.slot_pos = slot_pos_after;
                    st.active = true;
                }
                gen_loop_cv.notify_one();
                // Wait until the gen loop signals completion for this slot.
                std::unique_lock<std::mutex> lk(st.done_mu);
                st.done_cv.wait(lk, [&]() { return st.done; });
            }
            final_text = tok.decode(st.generated);
            int n_completion = (int)st.generated.size();
            if (seq.on_done) seq.on_done(std::move(final_text), n_completion);
        });
    printf("[server] continuous batching: %d concurrent slot(s)%s\n",
           num_slots, num_slots > 1 ? " (batched gen loop active)" : "");
    // Backpressure: cap pending queue length so a flood of requests can't
    // pile up with no way to drain. Default 16x slots, override via env.
    int max_queue = std::max(num_slots * 16, 32);
    if (const char* e = getenv("QWEN_MAX_QUEUE")) max_queue = std::max(0, atoi(e));
    sched.set_max_queue(max_queue);
    printf("[server] queue cap: %d (set QWEN_MAX_QUEUE=0 for unbounded)\n", max_queue);

    // Now that `sched` exists, spawn the batched gen-loop thread. It pulls
    // active slots from slot_gen_state and runs forward_step_batched under
    // sched.forward_mutex() so prefill/single-slot forwards stay race-free.
    if (num_slots > 1) {
        gen_loop_thread = std::thread([&]() {
            std::vector<int> a_slots, a_toks, a_pos, a_out;
            a_slots.reserve(num_slots); a_toks.reserve(num_slots);
            a_pos.reserve(num_slots);   a_out.reserve(num_slots);
            while (!gen_loop_stop.load(std::memory_order_acquire)) {
                a_slots.clear(); a_toks.clear(); a_pos.clear();
                {
                    std::unique_lock<std::mutex> lk(gen_loop_mu);
                    gen_loop_cv.wait_for(lk, std::chrono::milliseconds(2),
                        [&]() { return gen_loop_stop.load() || any_active(); });
                    if (gen_loop_stop.load()) return;
                    for (int s = 0; s < num_slots; s++) {
                        if (slot_gen_state[s]->active) {
                            a_slots.push_back(s);
                            a_toks.push_back(slot_gen_state[s]->next_tok);
                            a_pos.push_back(slot_gen_state[s]->slot_pos);
                        }
                    }
                }
                if (a_slots.empty()) continue;
                int N = (int)a_slots.size();
                a_out.assign(N, 0);
                {
                    std::lock_guard<std::mutex> lk(sched.forward_mutex());
                    forward_step_batched(N, a_slots.data(), a_toks.data(),
                                         a_pos.data(), a_out.data());
                }
                std::vector<int> finished_slots;
                {
                    std::lock_guard<std::mutex> lk(gen_loop_mu);
                    for (int i = 0; i < N; i++) {
                        int s = a_slots[i];
                        auto& st = *slot_gen_state[s];
                        int tok_id = a_out[i];
                        st.slot_pos += 1;
                        bool stop = false;
                        // Client gone: stop immediately, drop the token.
                        if (st.cancelled && st.cancelled->load(std::memory_order_relaxed)) {
                            stop = true;
                        } else if (tok_id == 248046 || tok_id == 248044 || tok_id == 248045) {
                            stop = true;
                        } else {
                            st.generated.push_back(tok_id);
                            if (st.on_token) st.on_token(tok_id);
                            if (tok_id == 248068) st.in_think = true;
                            if (tok_id == 248069) st.in_think = false;
                            if (!st.in_think) st.output_tokens++;
                            if (st.output_tokens >= st.max_gen) stop = true;
                            if (!stop && st.generated.size() >= 4) {
                                std::vector<int> tail(st.generated.end()-4, st.generated.end());
                                std::string ts = tok.decode(tail);
                                if (ts.find("</tool_call>") != std::string::npos) stop = true;
                            }
                            if (!stop && st.slot_pos >= model.slot_capacity(s) - 1) stop = true;
                        }
                        if (stop) {
                            st.active = false;
                            finished_slots.push_back(s);
                        } else {
                            st.next_tok = tok_id;
                        }
                    }
                }
                for (int s : finished_slots) {
                    auto& st = *slot_gen_state[s];
                    {
                        std::lock_guard<std::mutex> lk(st.done_mu);
                        st.done = true;
                    }
                    st.done_cv.notify_all();
                }
            }
        });
    }

    HttpServer server;
    server.port = port;
    server.model_name = model_name;
    server.api_key = api_key;
    server.proxy_embed_url  = proxy_embed_url;
    server.proxy_rerank_url = proxy_rerank_url;
    server.stats_fn = [&]() {
        int active = 0;
        if (num_slots > 1) {
            std::lock_guard<std::mutex> lk(gen_loop_mu);
            for (int s = 0; s < num_slots; s++)
                if (slot_gen_state[s]->active) active++;
        }
        std::ostringstream os;
        os << "{\"model\":\"" << model_name << "\""
           << ",\"slots\":" << num_slots
           << ",\"active\":" << active
           << ",\"queued\":" << sched.queued_count()
           << ",\"queue_cap\":" << max_queue
           << "}";
        return os.str();
    };

    // Non-streaming: submit a Sequence, wait on a future for the result.
    // While waiting we periodically poll(client_fd, POLLRDHUP) so a client
    // that closes mid-generation flips seq->cancelled and the gen loop
    // stops. Without this, non-streaming has no per-token send to fail and
    // would burn the slot until natural completion.
    server.generate_fn = [&](const std::vector<int>& prompt_ids, int max_tokens,
                             int cached_prompt_tokens,
                             const ResponseFormat& rf,
                             int client_fd,
                             int requested_slot,
                             int* out_completion_tokens) -> std::string {
        auto seq = std::make_shared<qwen_engine::Sequence>();
        seq->prompt_ids           = prompt_ids;
        seq->max_tokens           = max_tokens;
        seq->cached_prompt_tokens = cached_prompt_tokens;
        seq->requested_slot       = requested_slot;
        if (rf.json_mode) seq->grammar = std::make_shared<qwen_engine::JsonGrammar>();

        std::mutex done_mu;
        std::condition_variable done_cv;
        bool done = false;
        std::string final_text;
        int comp = 0;
        seq->on_done = [&](std::string text, int n) {
            {
                std::lock_guard<std::mutex> lk(done_mu);
                final_text = std::move(text);
                comp = n;
                done = true;
            }
            done_cv.notify_all();
        };
        if (!sched.submit(seq)) {
            if (out_completion_tokens) *out_completion_tokens = 0;
            return std::string("[server overloaded — queue full]");
        }
        {
            std::unique_lock<std::mutex> lk(done_mu);
            while (!done) {
                done_cv.wait_for(lk, std::chrono::milliseconds(100), [&]() { return done; });
                if (done) break;
                if (client_fd >= 0 && !seq->cancelled.load(std::memory_order_acquire)) {
                    struct pollfd pfd { client_fd, POLLRDHUP | POLLHUP | POLLERR, 0 };
                    if (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLRDHUP | POLLHUP | POLLERR))) {
                        seq->cancelled.store(true, std::memory_order_release);
                    }
                }
            }
        }
        if (out_completion_tokens) *out_completion_tokens = comp;
        return final_text;
    };
    // Match llama.cpp: no <think> prefill. 모델이 자율적으로 필요 시
    // `<think>` 열고 닫음. 웹 UI splitThink 는 `<think>` 로 시작하는
    // stream 만 think block 으로 접고, 바로 답 오는 경우는 all-answer
    // 로 표시 — 둘 다 자연스럽게 처리.
    server.prefills_think_tag = false;
    server.sampling_params = &sp;

    // Streaming wrapper. The scheduler invokes on_token on the worker thread
    // that owns this sequence, so we decode + flush directly into the SSE
    // callback. on_done finalizes the trailing UTF-8 buffer (in case a final
    // multi-byte char straddled the last token boundary).
    server.stream_generate_fn = [&](const std::vector<int>& prompt_ids, int max_tokens,
                                    int cached_prompt_tokens,
                                    const ResponseFormat& rf,
                                    int requested_slot,
                                    StreamCallback cb) {
        auto seq = std::make_shared<qwen_engine::Sequence>();
        seq->prompt_ids           = prompt_ids;
        seq->max_tokens           = max_tokens;
        seq->cached_prompt_tokens = cached_prompt_tokens;
        seq->requested_slot       = requested_slot;
        if (rf.json_mode) seq->grammar = std::make_shared<qwen_engine::JsonGrammar>();

        // utf8_buf is owned by the lambda captures — the worker thread runs
        // on_token, then on_done; both fire before the request handler
        // returns thanks to the cv wait below.
        auto utf8_buf = std::make_shared<std::string>();
        // Capture seq raw pointer for the on_token closure so it can mark
        // the Sequence as cancelled when SSE delivery fails. We hold a
        // shared_ptr above so the raw pointer stays valid for the duration
        // of generate_impl + gen_loop activity (both finish before this
        // function returns, thanks to the cv wait below).
        auto* seq_raw = seq.get();
        seq->on_token = [seq_raw, utf8_buf, cb](int tok_id) {
            // Tokenization happens elsewhere — but the closure needs `tok`
            // for decode. We can't capture `tok` by reference here because
            // this lambda outlives the enclosing serve_qwen frame... wait,
            // it doesn't: serve_qwen runs the HTTP loop in start(port) and
            // owns `tok`. So tok& by capture is fine via the [&] above.
            // Re-add the [&] capture for tok and friends:
        };
        // Replace with the proper capture (need [&] for tok decode + cb).
        seq->on_token = [&, utf8_buf, seq_raw](int tok_id) {
            if (seq_raw->cancelled.load(std::memory_order_acquire)) return;
            *utf8_buf += tok.decode_token(tok_id);
            std::string complete = Tokenizer::extract_complete_utf8(*utf8_buf);
            if (!complete.empty()) {
                if (!cb(complete, false)) {
                    seq_raw->cancelled.store(true, std::memory_order_release);
                }
            }
        };

        std::mutex done_mu;
        std::condition_variable done_cv;
        bool done = false;
        seq->on_done = [&, utf8_buf, seq_raw](std::string /*final_text*/, int /*n*/) {
            if (!seq_raw->cancelled.load(std::memory_order_acquire)) {
                if (!utf8_buf->empty()) { cb(*utf8_buf, false); utf8_buf->clear(); }
                cb("", true);
            }
            {
                std::lock_guard<std::mutex> lk(done_mu);
                done = true;
            }
            done_cv.notify_all();
        };

        if (!sched.submit(seq)) {
            // Queue at cap. Tell the client and bail.
            cb("", true);
            return;
        }
        {
            std::unique_lock<std::mutex> lk(done_mu);
            done_cv.wait(lk, [&]() { return done; });
        }
    };

    server.chat_encode_fn = [&](const std::vector<std::pair<std::string, std::string>>& msgs,
                                int force_think) {
        // Separate system prompt from conversation messages
        std::string sys_msg;
        std::vector<std::pair<std::string, std::string>> conv;
        for (auto& [role, content] : msgs) {
            if (role == "system") {
                if (!sys_msg.empty()) sys_msg += "\n\n";
                sys_msg += content;
            } else {
                conv.push_back({role, content});
            }
        }
        // Do not force <think>\n prefill by default (llama.cpp parity);
        // response_format=json_object passes force_think=-1 so the model
        // can't open a reasoning block whose body bytes the JSON grammar
        // would reject.
        auto ids = tok.apply_chat(sys_msg, conv, force_think);
        if (getenv("DUMP_PROMPT_IDS")) {
            fprintf(stderr, "[PROMPT_IDS %zu]", ids.size());
            for (int id : ids) fprintf(stderr, " %d", id);
            fprintf(stderr, "\n");
            fflush(stderr);
        }
        return ids;
    };
    server.encode_fn = [&](const std::string& text) {
        return tok.encode(text);
    };

    // Vision wiring. Only effective if `--vision-mmproj` was passed and the
    // GGUF carries the M-RoPE dimension_sections. We keep one mutex around
    // the per-request pipeline because the encoder writes into shared
    // globals (g_vision_embeds, g_mrope_pos_*) that the gen loop reads.
    if (g_vision_model && g_image_pad_id >= 0) {
        // Resolve the M-RoPE sections once, here, so every request gets a
        // consistent view (the LLM gguf is the source of truth, not the
        // mmproj). If they're missing, vision still runs but RoPE collapses
        // to 1D — predictably worse but not catastrophic.
        auto sec_it = gguf.meta_i32_arr.find("qwen35.rope.dimension_sections");
        if (sec_it != gguf.meta_i32_arr.end() && sec_it->second.size() >= 3) {
            g_mrope_sec_t = sec_it->second[0];
            g_mrope_sec_h = sec_it->second[1];
            g_mrope_sec_w = sec_it->second[2];
        }
        static std::mutex vision_mu;

        server.vision_reset_fn = [&]() {
            std::lock_guard<std::mutex> lk(vision_mu);
            if (g_vision_embeds) {
                cudaSetDevice(0);
                cudaFree(g_vision_embeds);
                g_vision_embeds = nullptr;
            }
            g_vision_n_tokens = 0;
            // Free per-GPU M-RoPE position arrays so the next request — vision
            // or text-only — starts from a clean slate. forward_attn falls back
            // to standard 1D RoPE when g_mrope_pos_t[g] is nullptr.
            int ng_local = std::min(n_gpus, 4);
            for (int g = 0; g < ng_local; g++) {
                cudaSetDevice(g);
                if (g_mrope_pos_t[g]) { cudaFree(g_mrope_pos_t[g]); g_mrope_pos_t[g] = nullptr; }
                if (g_mrope_pos_h[g]) { cudaFree(g_mrope_pos_h[g]); g_mrope_pos_h[g] = nullptr; }
                if (g_mrope_pos_w[g]) { cudaFree(g_mrope_pos_w[g]); g_mrope_pos_w[g] = nullptr; }
            }
            cudaSetDevice(0);
            g_mrope_len = 0;
        };
        server.vision_encode_fn = [&](const std::vector<uint8_t>& bytes) -> int {
            std::lock_guard<std::mutex> lk(vision_mu);
            int target = g_vision_model->cfg.image_size;
            std::vector<float> chw;
            if (!decode_and_preprocess_image(bytes, target, chw)) return 0;
            cudaSetDevice(0);
            float* d_img = nullptr;
            size_t img_bytes = chw.size() * sizeof(float);
            if (cudaMalloc(&d_img, img_bytes) != cudaSuccess) {
                fprintf(stderr, "[vision] cudaMalloc d_img failed (%zu bytes)\n", img_bytes);
                return 0;
            }
            cudaMemcpy(d_img, chw.data(), img_bytes, cudaMemcpyHostToDevice);
            int Nm = g_vision_model->cfg.num_merged();
            int P  = g_vision_model->cfg.proj_dim;
            // Grow the global embed buffer to hold prior images plus this one.
            half* d_new = nullptr;
            size_t new_bytes = (size_t)(g_vision_n_tokens + Nm) * P * sizeof(half);
            if (cudaMalloc(&d_new, new_bytes) != cudaSuccess) {
                fprintf(stderr, "[vision] cudaMalloc d_new failed (%zu bytes)\n", new_bytes);
                cudaFree(d_img);
                return 0;
            }
            if (g_vision_embeds && g_vision_n_tokens > 0) {
                cudaMemcpy(d_new, g_vision_embeds,
                           (size_t)g_vision_n_tokens * P * sizeof(half),
                           cudaMemcpyDeviceToDevice);
                cudaFree(g_vision_embeds);
            }
            half* d_dst = d_new + (size_t)g_vision_n_tokens * P;
            auto t0 = std::chrono::high_resolution_clock::now();
            g_vision_model->forward(d_img, d_dst, 0);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            cudaFree(d_img);
            g_vision_embeds = d_new;
            g_vision_n_tokens += Nm;
            printf("[vision] ViT forward: %.1f ms → %d tokens (total %d)\n", ms, Nm, g_vision_n_tokens);
            return Nm;
        };
        server.vision_setup_mrope_fn = [&](const std::vector<int>& prompt_ids) {
            std::lock_guard<std::mutex> lk(vision_mu);
            if (g_vision_n_tokens <= 0) return;
            int n_mps = (int)std::sqrt((double)g_vision_n_tokens);
            // If we have multiple images, n_vision_n_tokens = K*576, and the
            // square-root above gives an irrational answer. Each image is
            // independent in the prompt anyway (separate <|vision_start|>
            // ... <|vision_end|> blocks), so the merged-per-side is still 24
            // for our 768x768 fixed input.
            n_mps = g_vision_model->cfg.num_merged_per_side();
            setup_mrope_positions(prompt_ids, n_gpus, n_mps, g_image_pad_id);
            printf("[mrope] sections=[%d,%d,%d] logical_len=%d, %d image tokens (n_mps=%d)\n",
                   g_mrope_sec_t, g_mrope_sec_h, g_mrope_sec_w, g_mrope_len,
                   g_vision_n_tokens, n_mps);
        };
    }

    bool ok = server.start(port);
    // Stop + join the batched gen loop before tearing down captures.
    if (gen_loop_thread.joinable()) {
        gen_loop_stop.store(true, std::memory_order_release);
        gen_loop_cv.notify_all();
        gen_loop_thread.join();
    }
    // Cross-GPU prefill pipelining cleanup.
    for (int s = 0; s < (int)gpu_segs.size(); s++) {
        cudaSetDevice(gpu_segs[s].g);
        for (int b = 0; b < 2; b++) {
            cudaEventDestroy(pp_seg_done[s][b]);
            cudaEventDestroy(pp_d2h_done[s][b]);
            cudaEventDestroy(pp_h2d_done[s][b]);
        }
    }
    for (int g = 0; g < n_gpus; g++) {
        cudaSetDevice(g);
        cudaStreamDestroy(prefill_comp_stream[g]);
        cudaStreamDestroy(prefill_d2h_stream[g]);
        cudaStreamDestroy(prefill_h2d_stream[g]);
        cudaFree(gpu_hidden_chunk_pp[g]);
    }
    cudaFreeHost(host_chunk_transfer_pp);
    return ok ? 0 : 1;
}

// ============ Qwen3 dense Embedding/Reranker serve mode ============
// Loads a Qwen3 dense model (4B Q8_0 typical) for one of two services:
//   embed  → /v1/embeddings (last-token pool + L2 normalize)
//   rerank → /v1/rerank (query+doc score via lm_head logit softmax)
//
// Both sit on a single GPU (4B Q8_0 ≈ 4 GB → fits 16 GB CMP comfortably,
// even with the small KV cache for the input sequence). The forward path
// mirrors run_qwen's per-token prefill loop: walk tokens 0..N-1 through
// forward_attn + forward_mlp, then read the final position's hidden, apply
// output_norm, and either L2-normalize (embed) or run lm_head + yes/no
// softmax (rerank).
//
// Shared helper that returns the post-output_norm hidden vector for the
// LAST token of `prompt_ids`. Caller then handles the embed/rerank-specific
// post-processing.
struct EmbedForwardCtx {
    QwenModel*       model = nullptr;
    GPUModel*        gpu_model = nullptr;
    int              n_gpus = 0;
    int              H = 0, V = 0;
    int              last_gpu = 0;
    half*            gpu_hidden_half[4] = {};
    float*           gpu_hidden_fp32[4] = {};
    float*           gpu_hidden_chunk[4] = {};   // [CHUNK_SIZE * H] per GPU (chunked prefill)
    float*           host_chunk_transfer = nullptr; // [CHUNK_SIZE * H] pinned cross-GPU bridge
    half*            host_transfer = nullptr;
    half*            norm_buf = nullptr;
    half*            logits_buf = nullptr;
    GPUTensor*       embd_t = nullptr;
    GPUTensor*       out_norm_t = nullptr;
    GPUTensor*       out_w = nullptr;
    QuantInput       qi_logits;
};

// RERANK_PROF accumulators (per-forward; reset after the print).
static double g_rr_attn = 0.0, g_rr_mlp = 0.0;

// Run a single-pass prefill and return the post-output_norm hidden for the
// last token (length H, fp32 on host).
static bool run_embed_forward(EmbedForwardCtx& C,
                              const std::vector<int>& prompt_ids,
                              std::vector<float>& out_hidden) {
    QwenModel& model = *C.model;
    GPUModel& gpu_model = *C.gpu_model;
    int H = C.H;
    int last_gpu = C.last_gpu;
    if (prompt_ids.empty()) return false;

    // RERANK_PER_TOKEN=1 forces the legacy one-token-at-a-time prefill (for
    // A/B correctness checks against the chunked path).
    static const bool per_token = getenv("RERANK_PER_TOKEN") != nullptr;
    // STAGE 0: skip the ~1.15 GB per-doc KV memset on the fp16 dense sidecar
    // path (safe: causal attention + fresh KV writes never read stale pos>=plen
    // from a longer previous doc). EMBED_SKIP_KV_RESET=0 reverts to the full
    // reset for A/B / bisecting.
    static const bool skip_kv_reset = []{ const char* e=getenv("EMBED_SKIP_KV_RESET"); return !e || e[0]!='0'; }();
    auto embed_reset = [&]{ if (skip_kv_reset) model.reset_states_no_kv_memset(); else model.reset_all_states(); };
    if (per_token) {
        embed_reset();
        for (int step = 0; step < (int)prompt_ids.size(); step++) {
            int tok = prompt_ids[step];
            cudaSetDevice(0);
            if (C.embd_t->type == GGML_TYPE_Q8_0)
                dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(C.embd_t->data, C.gpu_hidden_half[0], tok, H);
            else if (C.embd_t->type == GGML_TYPE_Q5_K)
                dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(C.embd_t->data, C.gpu_hidden_half[0], tok, H);
            else if (C.embd_t->type == GGML_TYPE_Q6_K)
                dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(C.embd_t->data, C.gpu_hidden_half[0], tok, H);
            else return false;
            half_to_float_kernel<<<(H+255)/256, 256>>>(C.gpu_hidden_half[0], C.gpu_hidden_fp32[0], H);
            float* h = C.gpu_hidden_fp32[0];
            for (int layer = 0; layer < model.cfg.num_layers; layer++) {
                int g = gpu_model.layer_gpu[layer];
                int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer - 1];
                if (g != prev_g) {
                    cudaSetDevice(prev_g); cudaDeviceSynchronize();
                    cudaMemcpy(C.host_transfer, h, H * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaSetDevice(g);
                    cudaMemcpy(C.gpu_hidden_fp32[g], C.host_transfer, H * sizeof(float), cudaMemcpyHostToDevice);
                    h = C.gpu_hidden_fp32[g];
                }
                cudaSetDevice(g);
                model.forward_attn(layer, h, step, /*stream=*/0);
                model.forward_mlp_chunk(layer, h, /*n_tokens=*/1, /*stream=*/0);
            }
        }
        cudaSetDevice(last_gpu); cudaDeviceSynchronize();
        float_to_half_kernel<<<(H+255)/256, 256>>>(
            C.gpu_hidden_fp32[last_gpu], C.gpu_hidden_half[last_gpu], H);
        if (C.out_norm_t->type == GGML_TYPE_F32)
            rms_norm_f32w(C.norm_buf, C.gpu_hidden_half[last_gpu], (float*)C.out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
        else
            rms_norm(C.norm_buf, C.gpu_hidden_half[last_gpu], (half*)C.out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
        cudaDeviceSynchronize();
        std::vector<half> hh(H);
        cudaMemcpy(hh.data(), C.norm_buf, H * sizeof(half), cudaMemcpyDeviceToHost);
        out_hidden.resize(H);
        for (int i = 0; i < H; i++) out_hidden[i] = __half2float(hh[i]);
        return true;
    }

    // Fresh KV per request — single slot, no continuous batching for sidecars.
    // Chunked prefill (CHUNK_SIZE tokens per pass through the layer stack)
    // instead of one-token-at-a-time: amortises the per-layer cross-GPU host
    // bridge and uses the batched chunk kernels, the same path the chat server
    // uses for prompt processing. For an N-token rerank/embed prompt this turns
    // N sequential single-token forwards into ceil(N/256) batched ones.
    constexpr int CHUNK = QwenModel::CHUNK_SIZE;
    embed_reset();   // STAGE 0: skip the full KV memset on the fp16 dense path
    int plen = (int)prompt_ids.size();
    int chunk_pos = 0;
    float* last_h = nullptr;   // chunk buffer holding the final chunk (on last_gpu)
    int    last_n = 0;
    while (chunk_pos < plen) {
        int chunk_n = std::min(CHUNK, plen - chunk_pos);
        cudaSetDevice(0);
        for (int t = 0; t < chunk_n; t++) {
            int tok = prompt_ids[chunk_pos + t];
            if (C.embd_t->type == GGML_TYPE_Q8_0)
                dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(C.embd_t->data, C.gpu_hidden_half[0], tok, H);
            else if (C.embd_t->type == GGML_TYPE_Q5_K)
                dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(C.embd_t->data, C.gpu_hidden_half[0], tok, H);
            else if (C.embd_t->type == GGML_TYPE_Q6_K)
                dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(C.embd_t->data, C.gpu_hidden_half[0], tok, H);
            else { fprintf(stderr, "[embed] unsupported embd type %d\n", C.embd_t->type); return false; }
            half_to_float_kernel<<<(H+255)/256, 256>>>(
                C.gpu_hidden_half[0], C.gpu_hidden_chunk[0] + (size_t)t * H, H);
        }

        float* h_chunk = C.gpu_hidden_chunk[0];
        for (int layer = 0; layer < model.cfg.num_layers; layer++) {
            int g = gpu_model.layer_gpu[layer];
            int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer - 1];
            if (g != prev_g) {
                cudaSetDevice(prev_g); cudaDeviceSynchronize();
                cudaMemcpy(C.host_chunk_transfer, h_chunk,
                           (size_t)chunk_n * H * sizeof(float), cudaMemcpyDeviceToHost);
                cudaSetDevice(g);
                cudaMemcpy(C.gpu_hidden_chunk[g], C.host_chunk_transfer,
                           (size_t)chunk_n * H * sizeof(float), cudaMemcpyHostToDevice);
                h_chunk = C.gpu_hidden_chunk[g];
            }
            cudaSetDevice(g);
            static const bool prof = getenv("RERANK_PROF") != nullptr;
            auto p0 = std::chrono::high_resolution_clock::now();
            if (model.is_attn_layer(layer))
                model.forward_attn_chunk(layer, h_chunk, chunk_pos, chunk_n, /*stream=*/0);
            else
                model.forward_gdn_chunk(layer, h_chunk, chunk_n, /*stream=*/0);
            if (prof) { cudaSetDevice(g); cudaDeviceSynchronize(); }
            auto p1 = std::chrono::high_resolution_clock::now();
            model.forward_mlp_chunk(layer, h_chunk, chunk_n, /*stream=*/0);
            if (prof) {
                cudaSetDevice(g); cudaDeviceSynchronize();
                auto p2 = std::chrono::high_resolution_clock::now();
                g_rr_attn += std::chrono::duration<double,std::milli>(p1-p0).count();
                g_rr_mlp  += std::chrono::duration<double,std::milli>(p2-p1).count();
            }
        }
        cudaSetDevice(last_gpu); cudaDeviceSynchronize();
        last_h = h_chunk;
        last_n = chunk_n;
        chunk_pos += chunk_n;
    }
    {
        static const bool prof = getenv("RERANK_PROF") != nullptr;
        if (prof) {
            fprintf(stderr, "[rr-prof] tokens=%d attn=%.1fms mlp=%.1fms\n",
                    plen, g_rr_attn, g_rr_mlp);
            g_rr_attn = 0; g_rr_mlp = 0;
        }
    }

    // Last token's hidden is the final row of the final chunk. Convert fp32 →
    // fp16, apply output_norm, copy back to host.
    cudaSetDevice(last_gpu); cudaDeviceSynchronize();
    float_to_half_kernel<<<(H+255)/256, 256>>>(
        last_h + (size_t)(last_n - 1) * H, C.gpu_hidden_half[last_gpu], H);
    if (C.out_norm_t->type == GGML_TYPE_F32)
        rms_norm_f32w(C.norm_buf, C.gpu_hidden_half[last_gpu],
                      (float*)C.out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
    else
        rms_norm(C.norm_buf, C.gpu_hidden_half[last_gpu],
                 (half*)C.out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
    cudaDeviceSynchronize();
    std::vector<half> host_h(H);
    cudaMemcpy(host_h.data(), C.norm_buf, H * sizeof(half), cudaMemcpyDeviceToHost);
    out_hidden.resize(H);
    for (int i = 0; i < H; i++) out_hidden[i] = __half2float(host_h[i]);
    return true;
}

static void l2_normalize(std::vector<float>& v) {
    double ss = 0.0;
    for (float x : v) ss += (double)x * (double)x;
    float inv = (ss > 0.0) ? (float)(1.0 / std::sqrt(ss)) : 0.0f;
    for (float& x : v) x *= inv;
}

// Allocate all per-process buffers needed by run_embed_forward.
static void init_embed_ctx(EmbedForwardCtx& C, QwenModel& model, GPUModel& gpu_model, int n_gpus) {
    C.model = &model;
    C.gpu_model = &gpu_model;
    C.n_gpus = n_gpus;
    C.H = model.cfg.hidden_size;
    C.V = model.cfg.vocab_size;
    C.last_gpu = gpu_model.layer_gpu[model.cfg.num_layers - 1];
    C.embd_t     = gpu_model.get("token_embd.weight");
    C.out_norm_t = gpu_model.get("output_norm.weight");
    C.out_w      = gpu_model.get("output.weight");  // may be null (tied embeddings)

    for (int g = 0; g < n_gpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&C.gpu_hidden_half[g], C.H * sizeof(half));
        cudaMalloc(&C.gpu_hidden_fp32[g], C.H * sizeof(float));
        cudaMalloc(&C.gpu_hidden_chunk[g], (size_t)QwenModel::CHUNK_SIZE * C.H * sizeof(float));
    }
    cudaSetDevice(C.last_gpu);
    cudaMalloc(&C.norm_buf, C.H * sizeof(half));
    cudaMalloc(&C.logits_buf, C.V * sizeof(half));
    cudaMallocHost(&C.host_transfer, C.H * sizeof(float));
    cudaMallocHost(&C.host_chunk_transfer, (size_t)QwenModel::CHUNK_SIZE * C.H * sizeof(float));
}

int serve_embed(GGUFFile& gguf, GPUModel& gpu_model, int n_gpus, int port,
                const Tokenizer& tok, const std::string& model_name) {
    QwenModel model;
    model.gpu = &gpu_model;
    model.init_config(gguf);
    model.alloc_buffers();
    model.init_gdn_states(1);                  // no-op for dense (no ssm tensors)
    model.init_attention(8192, 1);             // 8K context plenty for embed inputs
    // Enable FlashAttention for the dense embed shape (HD=128, GQA=4, num_kv=8).
    // Default ON; FLASH_ATTN=0 opts back into the strict score/softmax/value path.
    {
        const char* fa_env = getenv("FLASH_ATTN");
        g_use_flash_attn = (fa_env == nullptr) ? true : (fa_env[0] == '1');
    }
    printf("[embed] model loaded: H=%d, layers=%d, V=%d\n",
           model.cfg.hidden_size, model.cfg.num_layers, model.cfg.vocab_size);

    EmbedForwardCtx C;
    init_embed_ctx(C, model, gpu_model, n_gpus);

    std::mutex fwd_mu;  // single-flight forward (sidecar = low QPS)
    auto embed_fn = [&](const std::vector<std::string>& inputs) -> std::vector<std::vector<float>> {
        std::lock_guard<std::mutex> lk(fwd_mu);
        std::vector<std::vector<float>> out;
        out.reserve(inputs.size());
        for (const auto& s : inputs) {
            auto ids = tok.encode(s);
            std::vector<float> h;
            if (!run_embed_forward(C, ids, h)) { out.push_back({}); continue; }
            l2_normalize(h);
            out.push_back(std::move(h));
        }
        return out;
    };

    HttpServer server;
    server.port = port;
    server.model_name = model_name;
    server.embed_fn = embed_fn;
    printf("[embed] listening on :%d\n", port);
    return server.start(port) ? 0 : 1;
}

int serve_rerank(GGUFFile& gguf, GPUModel& gpu_model, int n_gpus, int port,
                 const Tokenizer& tok, const std::string& model_name) {
    QwenModel model;
    model.gpu = &gpu_model;
    model.init_config(gguf);
    model.alloc_buffers();
    model.init_gdn_states(1);
    model.init_attention(8192, 1);
    // Enable FlashAttention for the dense rerank shape (HD=128, GQA=4, num_kv=8).
    // Default ON; FLASH_ATTN=0 opts back into the strict path.
    {
        const char* fa_env = getenv("FLASH_ATTN");
        g_use_flash_attn = (fa_env == nullptr) ? true : (fa_env[0] == '1');
    }
    printf("[rerank] model loaded: H=%d, layers=%d, V=%d\n",
           model.cfg.hidden_size, model.cfg.num_layers, model.cfg.vocab_size);

    EmbedForwardCtx C;
    init_embed_ctx(C, model, gpu_model, n_gpus);

    // Classifier head path. ggml-org's reranker GGUFs ship a [H, 2] classifier
    // (`cls.output.weight`) trained on the reranker's yes/no objective — much
    // more discriminative than reading raw yes/no token logits from the
    // generic LM head. We prefer it when present; otherwise we fall back to
    // the yes/no logit path further down.
    GPUTensor* cls_w = gpu_model.get("cls.output.weight");
    half* cls_out_buf = nullptr;
    int cls_dim = 0;
    std::vector<half> cls_w_host_f16;  // [H * cls_dim] row-major [cls_dim, H]
    if (cls_w) {
        // Expect [H, num_classes]. dims[0] = H (input), dims[1] = num_classes.
        // For Qwen3-Reranker 0.6B this is [1024, 2]. We score on the "yes"
        // class — convention from llama.cpp's reranker path: index 0=false,
        // index 1=true (sigmoid-or-softmax over both).
        cls_dim = (int)cls_w->dims[1];
        cudaSetDevice(C.last_gpu);
        cudaMalloc(&cls_out_buf, cls_dim * sizeof(half));
        printf("[rerank] classifier head detected: %s [%d, %d]\n",
               "cls.output.weight", (int)cls_w->dims[0], cls_dim);
        // Mirror the (tiny [H, cls_dim]) classifier head to host so we can do
        // the 2-class projection on CPU. The on-GPU gemv path was hitting an
        // intermittent CUDA fault for this F16 [2560,2] shape; a host dot
        // product over cls_dim=2 rows is trivially cheap and removes that
        // failure mode entirely.
        if (cls_w->type == GGML_TYPE_F16) {
            cls_w_host_f16.resize((size_t)cls_w->dims[0] * cls_dim);
            cudaMemcpy(cls_w_host_f16.data(), cls_w->data,
                       cls_w_host_f16.size() * sizeof(half), cudaMemcpyDeviceToHost);
        }
    } else {
        printf("[rerank] no cls.output.weight — using yes/no logit path "
               "(less reliable for 4B GGUFs whose fine-tune didn't survive)\n");
    }

    // Qwen3-Reranker uses the standard Qwen3 chat template; the model is
    // trained to emit "yes" or "no" as the FIRST assistant token (after the
    // assistant prefix + empty <think> block). Score = softmax([yes, no])[0]
    // on the logits at the LAST input position (= the position right before
    // the model would generate that yes/no).
    //
    // Template (verbatim from the Qwen3-Reranker HF model card):
    //   <|im_start|>system
    //   Judge whether the Document meets the requirements based on the Query
    //   and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
    //   <|im_start|>user
    //   <Instruct>: …
    //   <Query>: …
    //   <Document>: …<|im_end|>
    //   <|im_start|>assistant
    //   <think>
    //
    //   </think>
    //
    //   ← model emits "yes" or "no" here
    const std::string system_msg =
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        "Note that the answer can only be \"yes\" or \"no\".";
    const std::string default_instruct =
        "Given a web search query, retrieve relevant passages that answer the query";

    auto build_prompt = [&](const std::string& instruct, const std::string& q, const std::string& d) {
        std::string user_msg =
            "<Instruct>: " + (instruct.empty() ? default_instruct : instruct) +
            "\n<Query>: " + q +
            "\n<Document>: " + d;
        // Standard Qwen3-Reranker template (HF model card). Note: the empty
        // `<think>\n\n</think>\n\n` block IS part of the official suffix —
        // the Voodisss GGUF inherits it from the base. The model emits
        // yes/no right after.
        const char* suffix = std::getenv("RERANK_NO_THINK")
            ? "<|im_end|>\n<|im_start|>assistant\n"
            : "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        return std::string("<|im_start|>system\n") + system_msg + "<|im_end|>\n"
             + "<|im_start|>user\n" + user_msg + suffix;
    };

    // Look up token IDs by encoding short strings and picking the LAST token
    // — this is robust to whether the BPE merges "yes"/"no" into a single
    // token or splits them. We want the token the model would emit RIGHT
    // after the prefix above, which (per the Qwen3-Reranker spec) is a bare
    // "yes" or "no" with no leading space (because the template ends in
    // "\n\n" so the next token continues a fresh line).
    auto find_token_id = [&](const std::vector<const char*>& candidates) -> int {
        for (const char* w : candidates) {
            auto ids = tok.encode(w);
            if (ids.size() == 1) return ids[0];
        }
        // Fallback: take the LAST token of "... yes" minus the prefix ids.
        return -1;
    };
    int yes_id = find_token_id({"yes", " yes", "Yes", " Yes", "YES"});
    int no_id  = find_token_id({"no",  " no",  "No",  " No",  "NO"});
    if (yes_id < 0 || no_id < 0) {
        // Last-resort fallback: encode "answer yes" / "answer no" and grab
        // the trailing single-token answer.
        auto try_tail = [&](const std::string& full) -> int {
            auto ids = tok.encode(full);
            return ids.empty() ? -1 : ids.back();
        };
        if (yes_id < 0) yes_id = try_tail("yes");
        if (no_id  < 0) no_id  = try_tail("no");
    }
    if (yes_id < 0 || no_id < 0) {
        fprintf(stderr, "[rerank] failed to find yes/no token IDs (yes=%d no=%d)\n", yes_id, no_id);
        return 1;
    }
    printf("[rerank] yes_id=%d ('%s'), no_id=%d ('%s')\n",
           yes_id, tok.decode({yes_id}).c_str(),
           no_id,  tok.decode({no_id}).c_str());

    std::mutex fwd_mu;
    auto rerank_fn = [&](const std::string& instruction, const std::string& query,
                         const std::vector<std::string>& docs) -> std::vector<float> {
        std::lock_guard<std::mutex> lk(fwd_mu);
        std::vector<float> scores;
        scores.reserve(docs.size());
        for (const auto& d : docs) {
            auto ids = tok.encode(build_prompt(instruction, query, d));
            static const bool dbg_tok = getenv("RERANK_DEBUG") != nullptr;
            if (dbg_tok) {
                fprintf(stderr, "[rerank-tok] n=%zu head:", ids.size());
                for (size_t i = 0; i < ids.size() && i < 12; i++) fprintf(stderr, " %d", ids[i]);
                fprintf(stderr, " tail:");
                for (size_t i = (ids.size()>6?ids.size()-6:0); i < ids.size(); i++) fprintf(stderr, " %d", ids[i]);
                fprintf(stderr, "\n"); fflush(stderr);
            }
            std::vector<float> h;
            if (!run_embed_forward(C, ids, h)) { scores.push_back(0.0f); continue; }

            // Prefer the classifier head when present (the converted Qwen3
            // reranker GGUFs include it). The projection is [cls_dim, H] @ h
            // computed on host — cls_dim is 2, so this is two dot products,
            // and it sidesteps an intermittent CUDA fault we saw running the
            // F16 gemv for this tiny shape. No device round-trip needed.
            if (cls_w && !cls_w_host_f16.empty()) {
                std::vector<float> cls_logits(cls_dim, 0.0f);
                for (int c = 0; c < cls_dim; c++) {
                    const half* row = cls_w_host_f16.data() + (size_t)c * C.H;
                    double s = 0.0;
                    for (int k = 0; k < C.H; k++)
                        s += (double)__half2float(row[k]) * (double)h[k];
                    cls_logits[c] = (float)s;
                }
                // Row 0 = "yes"/true, row 1 = "no"/false (convert_hf_to_gguf.py
                // `_get_cls_out_tensor` stacks [true_row, false_row]).
                // RERANK_CLS_YES overrides which row is yes for GGUFs built
                // with the opposite stack order.
                static const int yes_row = []{ const char* e = getenv("RERANK_CLS_YES"); return e ? atoi(e) : 0; }();
                float lyes = cls_logits[yes_row == 0 ? 0 : (cls_dim > 1 ? 1 : 0)];
                float lno  = cls_logits[yes_row == 0 ? (cls_dim > 1 ? 1 : 0) : 0];
                static const bool dbg = getenv("RERANK_DEBUG") != nullptr;
                if (dbg) {
                    fprintf(stderr, "[rerank-dbg cls] lyes=%.3f lno=%.3f doc='%.40s'\n",
                            lyes, lno, d.c_str()); fflush(stderr);
                }
                float m = std::max(lyes, lno);
                float pyes = std::exp(lyes - m);
                float pno  = std::exp(lno  - m);
                scores.push_back(pyes / (pyes + pno));
                continue;
            }

            // Fallback: lm_head + yes/no token logits. Needs the hidden on
            // device + quantized for the big vocab GEMV.
            cudaSetDevice(C.last_gpu);
            std::vector<half> h16(C.H);
            for (int i = 0; i < C.H; i++) h16[i] = __float2half(h[i]);
            cudaMemcpy(C.norm_buf, h16.data(), C.H * sizeof(half), cudaMemcpyHostToDevice);
            C.qi_logits.quantize(C.norm_buf, C.H, 0);
            auto* lmh = C.out_w ? C.out_w : C.embd_t;  // tied embeddings fallback
            quant_gemv(lmh->data, lmh->type, C.norm_buf, C.logits_buf, C.H, C.V, &C.qi_logits, 0);
            cudaDeviceSynchronize();
            // Read just the two logits we care about.
            half lyes_h, lno_h;
            cudaMemcpy(&lyes_h, C.logits_buf + yes_id, sizeof(half), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lno_h,  C.logits_buf + no_id,  sizeof(half), cudaMemcpyDeviceToHost);
            float lyes = __half2float(lyes_h);
            float lno  = __half2float(lno_h);
            // Optional: RERANK_DEBUG=1 dumps yes/no + argmax to stderr for
            // diagnosing model/quant mismatches (a healthy reranker has yes
            // or no as the argmax with a much larger logit than other tokens).
            static const bool dbg = getenv("RERANK_DEBUG") != nullptr;
            if (dbg) {
                int* d_arg = nullptr; cudaMalloc(&d_arg, sizeof(int));
                argmax_half_kernel<<<1, 1024, 0, 0>>>(C.logits_buf, C.V, d_arg);
                int argmax_tok = -1;
                cudaMemcpy(&argmax_tok, d_arg, sizeof(int), cudaMemcpyDeviceToHost);
                cudaFree(d_arg);
                half max_h;
                cudaMemcpy(&max_h, C.logits_buf + argmax_tok, sizeof(half), cudaMemcpyDeviceToHost);
                std::string argmax_str = tok.decode({argmax_tok});
                fprintf(stderr, "[rerank-dbg] lyes=%.3f lno=%.3f argmax=%d('%s' %.3f) doc='%.40s'\n",
                        lyes, lno, argmax_tok, argmax_str.c_str(),
                        __half2float(max_h), d.c_str()); fflush(stderr);
            }
            float m = std::max(lyes, lno);
            float pyes = std::exp(lyes - m);
            float pno  = std::exp(lno  - m);
            scores.push_back(pyes / (pyes + pno));
        }
        return scores;
    };

    HttpServer server;
    server.port = port;
    server.model_name = model_name;
    server.rerank_fn = rerank_fn;
    printf("[rerank] listening on :%d\n", port);
    return server.start(port) ? 0 : 1;
}

// ============ Gemma serve mode ============

int serve_gemma(GGUFFile& gguf, GPUModel& gpu_model, int n_gpus, int port) {
    GemmaModel model;
    model.gpu = &gpu_model;
    model.init_config(gguf);
    model.alloc_buffers();
    model.init_kv_caches(4096);
    model.init_rope(4096);

    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(gguf)) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }

    int H = model.cfg.hidden_size;
    int V = model.cfg.vocab_size;
    int last_gpu = gpu_model.layer_gpu[model.cfg.num_layers - 1];

    auto* embd_t = gpu_model.get("token_embd.weight");
    auto* out_norm_t = gpu_model.get("output_norm.weight");

    half* gpu_hidden[4];
    for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaMalloc(&gpu_hidden[g], H * sizeof(half)); }
    cudaSetDevice(0);
    half* logits_buf; cudaMalloc(&logits_buf, V * sizeof(half));
    half* norm_buf; cudaMalloc(&norm_buf, H * sizeof(half));
    cudaSetDevice(last_gpu);
    half* norm_out_last; cudaMalloc(&norm_out_last, H * sizeof(half));
    half* host_transfer; cudaMallocHost(&host_transfer, H * sizeof(half));
    QuantInput qi_logits;
    float embd_scale = model.cfg.embd_scale;

    // Generate callback: takes prompt token IDs, returns generated text
    auto generate = [&](const std::vector<int>& prompt_ids, int max_tokens,
                        int /*cached_prompt_tokens*/,
                        const ResponseFormat& /*rf*/,
                        int /*client_fd*/,
                        int /*requested_slot*/,
                        int* out_completion_tokens) -> std::string {
        model.reset_all();
        std::vector<int> generated;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < (int)(prompt_ids.size() + max_tokens); step++) {
            int token_id = step < (int)prompt_ids.size() ? prompt_ids[step] : generated.back();

            cudaSetDevice(0);
            if (embd_t->type == GGML_TYPE_Q8_K)
                dequant_embd_q8k_row<<<(H + 255) / 256, 256>>>(embd_t->data, gpu_hidden[0], token_id, H);
            else if (embd_t->type == GGML_TYPE_Q5_K)
                dequant_embd_q5k_row_v2<<<(H + 255) / 256, 256>>>(embd_t->data, gpu_hidden[0], token_id, H);
            else if (embd_t->type == GGML_TYPE_Q6_K)
                dequant_embd_q6k_row_v2<<<(H + 255) / 256, 256>>>(embd_t->data, gpu_hidden[0], token_id, H);
            scale_embedding_kernel<<<(H + 255) / 256, 256>>>(gpu_hidden[0], embd_scale, H);

            half* h = gpu_hidden[0];
            for (int layer = 0; layer < model.cfg.num_layers; layer++) {
                int g = gpu_model.layer_gpu[layer];
                int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer - 1];
                if (g != prev_g) {
                    cudaSetDevice(prev_g); cudaDeviceSynchronize();
                    cudaMemcpy(host_transfer, h, H * sizeof(half), cudaMemcpyDeviceToHost);
                    cudaSetDevice(g);
                    cudaMemcpy(gpu_hidden[g], host_transfer, H * sizeof(half), cudaMemcpyHostToDevice);
                    h = gpu_hidden[g];
                }
                cudaSetDevice(g);
                if (model.is_full_attn(layer))
                    model.forward_full_attn(layer, h, step, 0);
                else
                    model.forward_sliding_attn(layer, h, step, 0);
                if (model.cfg.is_moe)
                    model.forward_moe(layer, h, 0);
                else
                    model.forward_mlp(layer, h, 0);
                // layer_output_scale now applied inside forward_*_attn and forward_moe/mlp
            }

            if (step >= (int)prompt_ids.size() - 1) {
                cudaSetDevice(last_gpu); cudaDeviceSynchronize();
                if (out_norm_t->type == GGML_TYPE_F32)
                    rms_norm_f32w(norm_out_last, h, (float*)out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
                else
                    rms_norm(norm_out_last, h, (half*)out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
                cudaDeviceSynchronize();

                cudaMemcpy(host_transfer, norm_out_last, H * sizeof(half), cudaMemcpyDeviceToHost);
                cudaSetDevice(0);
                cudaMemcpy(norm_buf, host_transfer, H * sizeof(half), cudaMemcpyHostToDevice);
                qi_logits.quantize(norm_buf, H, 0);
                quant_gemv(embd_t->data, embd_t->type, norm_buf, logits_buf, H, V, &qi_logits);
                if (model.cfg.softcap > 0)
                    softcap_kernel<<<(V + 255) / 256, 256>>>(logits_buf, model.cfg.softcap, V);
                cudaDeviceSynchronize();

                std::vector<half> h_logits(V);
                cudaMemcpy(h_logits.data(), logits_buf, V * sizeof(half), cudaMemcpyDeviceToHost);

                float max_val = -1e30f; int max_idx = 0;
                for (int i = 0; i < V; i++) {
                    float v = __half2float(h_logits[i]);
                    if (v > max_val) { max_val = v; max_idx = i; }
                }

                if (step >= (int)prompt_ids.size()) {
                    generated.push_back(max_idx);
                    if (max_idx == tokenizer.eos_id || max_idx == 106) break;
                } else {
                    generated.push_back(max_idx);
                }
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("[API] Generated %zu tokens in %.0f ms (%.1f t/s)\n",
               generated.size(), ms, generated.size() * 1000.0 / ms);

        if (out_completion_tokens) *out_completion_tokens = (int)generated.size();
        return tokenizer.decode(generated);
    };

    HttpServer server;
    server.port = port;
    server.model_name = "gemma-4-31B";
    server.generate_fn = generate;
    server.chat_encode_fn = [&](const std::vector<std::pair<std::string, std::string>>& msgs,
                                int /*force_think*/) {
        return tokenizer.apply_chat("", msgs);
    };
    server.encode_fn = [&](const std::string& text) {
        return tokenizer.encode(text);
    };

    return server.start(port) ? 0 : 1;
}

// ============ Main: auto-detect architecture ============

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);  // disable stdout buffering for logging
    signal(SIGPIPE, SIG_IGN);  // dead-client socket writes return EPIPE instead of killing process
    if (argc < 2) {
        printf("Usage: %s <model.gguf> [options] [token_ids...]\n", argv[0]);
        printf("Options:\n");
        printf("  --temp <f>        Temperature (0=greedy, default)\n");
        printf("  --top-k <n>       Top-K sampling (0=disabled)\n");
        printf("  --top-p <f>       Nucleus sampling (1.0=disabled)\n");
        printf("  --min-p <f>       Min-P sampling (0.0=disabled)\n");
        printf("  --rep-pen <f>     Repetition penalty (1.0=disabled)\n");
        printf("  --rep-win <n>     Repetition penalty window (default 64)\n");
        printf("  --freq-pen <f>    Frequency penalty (0.0=disabled)\n");
        printf("  --pres-pen <f>    Presence penalty (0.0=disabled)\n");
        printf("  --max-tokens <n>  Max tokens to generate (default 500)\n");
        printf("  --seed <n>        RNG seed (0=random)\n");
        printf("  --serve <port>    Start HTTP API server\n");
        return 1;
    }

    int n_gpus; cudaGetDeviceCount(&n_gpus);
    GGUFFile gguf;
    if (!gguf.open(argv[1])) return 1;

    GPUModel gpu_model;
    if (!gpu_model.load(gguf, n_gpus)) return 1;

    auto arch = gguf.get_str("general.architecture");
    printf("Architecture: %s\n", arch.c_str());

    // Parse sampling params and flags
    SamplingParams sp;
    int serve_port = 0;
    // Default max_seq: 4096 for fp16 KV (safe VRAM budget). When MTP_TQ=1 is
    // set (TurboQuant 3-bit KV cache), the fp16 cache is not allocated and a
    // 256K context fits comfortably, so bump the default accordingly. Can
    // always be overridden with --max-seq / -c.
    int max_seq = getenv("MTP_TQ") ? 262144 : 4096;
    // Continuous batching: number of concurrent request slots. Each slot has
    // its own KV+GDN state. 1 = legacy single-request behavior. Override via
    // --slots N or QWEN_SLOTS env var. With slots=N, each slot gets max_seq
    // tokens of context (so total physical KV is N×max_seq).
    int num_slots = 1;
    if (const char* e = getenv("QWEN_SLOTS")) num_slots = std::max(1, atoi(e));
    // Asymmetric per-slot context caps. Comma-separated list overrides --slots
    // and --max-seq when set (e.g. --slot-caps "262144,65536"). Each entry is
    // that slot's max context in tokens; KV/GDN buffers allocate the sum.
    std::vector<int> slot_caps;
    // Service mode for the Qwen3 dense sidecar models. "chat" (default) goes
    // through serve_qwen / run_qwen as before; "embed" loads a Qwen3 dense
    // model and serves /v1/embeddings; "rerank" likewise for /v1/rerank.
    std::string service_mode = "chat";
    // Proxy targets for the chat server to forward embedding/rerank requests
    // to (e.g. localhost:8001 + localhost:8002). Empty = endpoint disabled.
    std::string proxy_embed_url, proxy_rerank_url;
    std::string prompt_text, api_key;
    std::string vision_mmproj_path, vision_test_image, image_raw_path;
    // DFlash drafter training-data extraction (--mode dflash-extract).
    std::string dflash_corpus_path, dflash_out_dir;
    size_t dflash_chunk_bytes = 18000000000ULL;  // ~18 GB default
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--serve") == 0 && i + 1 < argc) {
            serve_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt_text = argv[++i];
        } else if (strcmp(argv[i], "--api-key") == 0 && i + 1 < argc) {
            api_key = argv[++i];
        } else if (strcmp(argv[i], "--max-seq") == 0 && i + 1 < argc) {
            max_seq = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            max_seq = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--slots") == 0 && i + 1 < argc) {
            num_slots = std::max(1, atoi(argv[++i]));
        } else if (strcmp(argv[i], "--slot-caps") == 0 && i + 1 < argc) {
            slot_caps.clear();
            std::string s = argv[++i];
            size_t pos = 0;
            while (pos < s.size()) {
                size_t comma = s.find(',', pos);
                std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (!tok.empty()) slot_caps.push_back(std::max(1, atoi(tok.c_str())));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
            if (!slot_caps.empty()) {
                num_slots = (int)slot_caps.size();
                int max_c = 0; for (int c : slot_caps) if (c > max_c) max_c = c;
                max_seq = max_c;
            }
        } else if (strcmp(argv[i], "--vision-mmproj") == 0 && i + 1 < argc) {
            vision_mmproj_path = argv[++i];
        } else if (strcmp(argv[i], "--vision-test") == 0 && i + 1 < argc) {
            vision_test_image = argv[++i];
        } else if (strcmp(argv[i], "--image-raw") == 0 && i + 1 < argc) {
            image_raw_path = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            service_mode = argv[++i];
        } else if (strcmp(argv[i], "--proxy-embed") == 0 && i + 1 < argc) {
            proxy_embed_url = argv[++i];
        } else if (strcmp(argv[i], "--proxy-rerank") == 0 && i + 1 < argc) {
            proxy_rerank_url = argv[++i];
        } else if (strcmp(argv[i], "--dflash-corpus") == 0 && i + 1 < argc) {
            dflash_corpus_path = argv[++i];
        } else if (strcmp(argv[i], "--dflash-out") == 0 && i + 1 < argc) {
            dflash_out_dir = argv[++i];
        } else if (strcmp(argv[i], "--dflash-chunk-bytes") == 0 && i + 1 < argc) {
            dflash_chunk_bytes = strtoull(argv[++i], nullptr, 10);
        }
    }

    // Vision smoke test: --vision-mmproj <mmproj.gguf> --vision-test <3x768x768 fp32.raw>
    // Loads mmproj, runs ViT on the preprocessed image, dumps first embeddings.
    if (!vision_mmproj_path.empty() && !vision_test_image.empty()) {
        GGUFFile vgg;
        if (!vgg.open(vision_mmproj_path.c_str())) {
            fprintf(stderr, "vision mmproj open failed\n");
            return 1;
        }
        vision::VisionModel vm;
        if (!vm.load(vgg, 0)) {
            fprintf(stderr, "vision load failed\n");
            return 1;
        }
        // Read fp32 raw image: 3*768*768 = 1769472 floats = 6.75 MB
        int H = vm.cfg.image_size;
        size_t need = (size_t)3 * H * H * sizeof(float);
        FILE* f = fopen(vision_test_image.c_str(), "rb");
        if (!f) { perror("image open"); return 1; }
        std::vector<float> host_img(3 * H * H);
        size_t got = fread(host_img.data(), 1, need, f);
        fclose(f);
        if (got != need) {
            fprintf(stderr, "image size mismatch: got %zu need %zu\n", got, need);
            return 1;
        }
        float* d_img;
        cudaMalloc(&d_img, need);
        cudaMemcpy(d_img, host_img.data(), need, cudaMemcpyHostToDevice);
        half* d_out;
        int Nm = vm.cfg.num_merged();
        int P  = vm.cfg.proj_dim;
        cudaMalloc(&d_out, (size_t)Nm * P * sizeof(half));
        auto t0 = std::chrono::high_resolution_clock::now();
        vm.forward(d_img, d_out, 0);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("[vision] forward pass: %.1f ms → %d tokens × %d dim\n", ms, Nm, P);
        std::vector<half> host_out(Nm * P);
        cudaMemcpy(host_out.data(), d_out, host_out.size() * sizeof(half), cudaMemcpyDeviceToHost);
        printf("[vision] token 0 first 12 embeddings: ");
        for (int i = 0; i < 12; i++) printf("%.4f ", __half2float(host_out[i]));
        printf("\n[vision] token 288 (center) first 12: ");
        for (int i = 0; i < 12; i++) printf("%.4f ", __half2float(host_out[288*P + i]));
        printf("\n");
        // Sanity: check for NaN/Inf
        int nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < host_out.size(); i++) {
            float v = __half2float(host_out[i]);
            if (v != v) nan_count++;
            else if (v > 1e30 || v < -1e30) inf_count++;
        }
        printf("[vision] NaN: %d / %zu, Inf: %d / %zu\n",
               nan_count, host_out.size(), inf_count, host_out.size());
        // Per-token magnitude — useful to compare runs across different images.
        double s_abs = 0.0;
        float t0_max = 0.0f, t0_mean = 0.0f, t288_max = 0.0f, t288_mean = 0.0f;
        for (int i = 0; i < P; i++) {
            float v = __half2float(host_out[i]);
            t0_mean += v / P; if (fabsf(v) > t0_max) t0_max = fabsf(v);
            v = __half2float(host_out[288*P + i]);
            t288_mean += v / P; if (fabsf(v) > t288_max) t288_max = fabsf(v);
        }
        for (size_t i = 0; i < host_out.size(); i++) s_abs += fabsf(__half2float(host_out[i]));
        printf("[vision] stats: tot_abs=%.1f tot_mean_abs=%.4f  t0_mean=%.4f t0_maxabs=%.4f  t288_mean=%.4f t288_maxabs=%.4f\n",
               s_abs, s_abs / host_out.size(), t0_mean, t0_max, t288_mean, t288_max);
        // Print a wide channel slice from token 0 — useful to A/B color encoding.
        printf("[vision] tok0 ch[1000..1011]: ");
        for (int i = 1000; i < 1012; i++) printf("%.4f ", __half2float(host_out[i]));
        printf("\n[vision] tok0 ch[3000..3011]: ");
        for (int i = 3000; i < 3012; i++) printf("%.4f ", __half2float(host_out[i]));
        printf("\n[vision] tok0 ch[4990..5001]: ");
        for (int i = 4990; i < 5002; i++) printf("%.4f ", __half2float(host_out[i]));
        printf("\n");
        cudaFree(d_img); cudaFree(d_out);
        return 0;
    }
    int tok_start = parse_sampling_args(argc, argv, sp);

    // Load tokenizer from GGUF
    Tokenizer tokenizer;
    tokenizer.load_from_gguf(gguf);

    // Vision encoder: load whenever `--vision-mmproj` is passed, regardless of
    // whether `--image-raw` is also given. CLI-mode runs ViT on the raw image
    // up front; server-mode keeps the encoder dormant and runs it per request.
    static GGUFFile vgg;   // static: lifetime = program (mmap stays open)
    if (!vision_mmproj_path.empty()) {
        if (!vgg.open(vision_mmproj_path.c_str())) {
            fprintf(stderr, "vision mmproj open failed\n");
            return 1;
        }
        static vision::VisionModel vm;
        if (!vm.load(vgg, 0)) {
            fprintf(stderr, "vision load failed\n");
            return 1;
        }
        g_vision_model = &vm;
        auto it = tokenizer.token_to_id.find("<|image_pad|>");
        if (it == tokenizer.token_to_id.end()) {
            fprintf(stderr, "<|image_pad|> token not found in vocab\n");
            return 1;
        }
        g_image_pad_id = it->second;
        g_vision_H = vm.cfg.proj_dim;
        printf("[vision] mmproj loaded (image=%d patch=%d proj=%d), image_pad token id=%d\n",
               vm.cfg.image_size, vm.cfg.patch_size, vm.cfg.proj_dim, g_image_pad_id);
    }
    if (!vision_mmproj_path.empty() && !image_raw_path.empty()) {
        vision::VisionModel& vm = *g_vision_model;
        int Hv = vm.cfg.image_size;
        size_t need = (size_t)3 * Hv * Hv * sizeof(float);
        FILE* f = fopen(image_raw_path.c_str(), "rb");
        if (!f) { perror("image open"); return 1; }
        std::vector<float> host_img(3 * Hv * Hv);
        size_t got = fread(host_img.data(), 1, need, f);
        fclose(f);
        if (got != need) {
            fprintf(stderr, "image size mismatch: got %zu need %zu\n", got, need);
            return 1;
        }
        float* d_img;
        cudaSetDevice(0);
        cudaMalloc(&d_img, need);
        cudaMemcpy(d_img, host_img.data(), need, cudaMemcpyHostToDevice);
        int Nm = vm.cfg.num_merged();
        int P  = vm.cfg.proj_dim;
        half* d_vis;
        cudaMalloc(&d_vis, (size_t)Nm * P * sizeof(half));
        auto vt0 = std::chrono::high_resolution_clock::now();
        vm.forward(d_img, d_vis, 0);
        cudaDeviceSynchronize();
        auto vt1 = std::chrono::high_resolution_clock::now();
        double vms = std::chrono::duration<double, std::milli>(vt1 - vt0).count();
        printf("[vision] ViT forward: %.1f ms → %d tokens × %d dim\n", vms, Nm, P);
        cudaFree(d_img);
        g_vision_embeds = d_vis;
        g_vision_n_tokens = Nm;
        printf("[vision] %d placeholders needed\n", Nm);
    }

    // Build prompt token IDs
    std::vector<int> prompt_ids;
    if (!prompt_text.empty()) {
        std::string effective_prompt = prompt_text;
        if (g_vision_embeds) {
            // Wrap the user text with the Qwen-VL vision block. The tokenizer
            // recognises the <|vision_start|>/<|image_pad|>/<|vision_end|>
            // special tokens so they encode as single token IDs.
            std::string vblock = "<|vision_start|>";
            vblock.reserve(16 + g_vision_n_tokens * 16);
            for (int i = 0; i < g_vision_n_tokens; i++) vblock += "<|image_pad|>";
            vblock += "<|vision_end|>";
            effective_prompt = vblock + prompt_text;
        }
        // Text prompt: encode with appropriate chat template
        if (tokenizer.is_sentencepiece)
            prompt_ids = tokenizer.apply_chat_gemma("", {{"user", effective_prompt}});
        else
            prompt_ids = tokenizer.encode_chat(effective_prompt);
        printf("Prompt: %zu tokens total [", prompt_ids.size());
        for (size_t i = 0; i < std::min(prompt_ids.size(), (size_t)20); i++) printf("%d ", prompt_ids[i]);
        printf("...]\n");

        // Build M-RoPE per-token position arrays for Qwen3-VL multimodal prompts.
        // Text tokens get pos_t = pos_h = pos_w = sequential (recovers 1D RoPE);
        // each contiguous run of <|image_pad|> tokens gets (vision_base,
        // vision_base+y, vision_base+x) on the (t, h, w) axes, then logical
        // position advances by max(nx, ny) per llama.cpp's mtmd convention.
        if (g_vision_embeds && g_image_pad_id >= 0) {
            auto sec_it = gguf.meta_i32_arr.find("qwen35.rope.dimension_sections");
            if (sec_it == gguf.meta_i32_arr.end() || sec_it->second.size() < 3) {
                fprintf(stderr, "[mrope] missing qwen35.rope.dimension_sections — cannot enable M-RoPE\n");
            } else {
                g_mrope_sec_t = sec_it->second[0];
                g_mrope_sec_h = sec_it->second[1];
                g_mrope_sec_w = sec_it->second[2];
                int n_merged_per_side = (int)std::sqrt((double)g_vision_n_tokens);
                setup_mrope_positions(prompt_ids, n_gpus, n_merged_per_side, g_image_pad_id);
                printf("[mrope] sections=[%d,%d,%d] logical_len=%d, %zu image tokens, mirrored to %d GPUs\n",
                       g_mrope_sec_t, g_mrope_sec_h, g_mrope_sec_w, g_mrope_len,
                       (size_t)g_vision_n_tokens, std::min(n_gpus, 4));
            }
        }
    } else {
        // Raw token IDs from CLI
        for (int i = tok_start; i < argc; i++) {
            if (argv[i][0] == '-' && !isdigit(argv[i][1])) break;
            prompt_ids.push_back(atoi(argv[i]));
        }
    }

    // Check for --chat flag
    bool chat_mode = false;
    for (int i = 2; i < argc; i++)
        if (strcmp(argv[i], "--chat") == 0) chat_mode = true;

    int ret;
    // Model name from filename
    std::string model_name = argv[1];
    { size_t s = model_name.rfind('/'); if (s != std::string::npos) model_name = model_name.substr(s+1);
      size_t d = model_name.find(".gguf"); if (d != std::string::npos) model_name = model_name.substr(0, d); }

    if (service_mode == "dflash-extract") {
        if (dflash_corpus_path.empty() || dflash_out_dir.empty()) {
            fprintf(stderr, "[dflash-extract] requires --dflash-corpus <file> and --dflash-out <dir>\n");
            ret = 1;
        } else {
            ret = run_dflash_extract(gguf, gpu_model, n_gpus,
                                     dflash_corpus_path, dflash_out_dir, dflash_chunk_bytes);
        }
    } else if (serve_port > 0 && service_mode == "embed") {
        ret = serve_embed(gguf, gpu_model, n_gpus, serve_port, tokenizer, model_name);
    } else if (serve_port > 0 && service_mode == "rerank") {
        ret = serve_rerank(gguf, gpu_model, n_gpus, serve_port, tokenizer, model_name);
    } else if (serve_port > 0 && (arch == "qwen35" || arch == "qwen35moe")) {
        ret = serve_qwen(gguf, gpu_model, n_gpus, serve_port, tokenizer, model_name, api_key, max_seq, num_slots, slot_caps,
                         proxy_embed_url, proxy_rerank_url);
    } else if (serve_port > 0) {
        ret = serve_gemma(gguf, gpu_model, n_gpus, serve_port);
    } else if (chat_mode && (arch == "qwen35" || arch == "qwen35moe")) {
        ret = run_chat(gguf, gpu_model, n_gpus, sp, tokenizer);
    } else if (arch == "gemma4" || arch == "gemma2" || arch == "gemma3") {
        ret = run_gemma(gguf, gpu_model, n_gpus, sp, prompt_ids, &tokenizer);
    } else {
        ret = run_qwen(gguf, gpu_model, n_gpus, sp, prompt_ids, &tokenizer);
    }

    gpu_model.unload(); gguf.close();
    return ret;
}
