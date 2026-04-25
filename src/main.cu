#include "gguf.h"
#include "gpu_loader.h"
#include "model.cuh"
#include "mtp_head.cuh"
#include "gemma_model.cuh"
#include "ops.cuh"
#include "turboquant.cuh"
#include "tokenizer.h"
#include "sampling.h"
#include "server.h"
#include "vision.cuh"
#include <cstdio>
// Vision hooks (shared via global pointers — see main() wiring). When the CLI
// is invoked with --mmproj + --image-raw, main() runs the ViT on GPU 0 and
// populates g_vision_embeds with 576 fp16 rows of LLM-hidden dim. During
// prefill, any token matching g_image_pad_id uses the next vision embedding
// instead of the normal dequant_embd path.
static half* g_vision_embeds = nullptr;
static int g_vision_n_tokens = 0;
static int g_vision_H = 0;
static int g_image_pad_id = -1;
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstring>
#include <iostream>

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
                model.forward_mlp(layer, h, 0);
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
                    model.forward_mlp_chunk(layer, h_chunk, chunk_n, 0);
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

    for (int step = 0; step < (int)(prompt_ids.size() + max_gen); step++) {
        int token_id = step < (int)prompt_ids.size() ? prompt_ids[step] : generated.back();
        auto step_start = std::chrono::high_resolution_clock::now();

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
            model.forward_mlp(layer, h, 0);
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

int serve_qwen(GGUFFile& gguf, GPUModel& gpu_model, int n_gpus, int port, const Tokenizer& tok, const std::string& model_name, const std::string& api_key = "", int max_seq = 262144) {
    QwenModel model;
    model.gpu = &gpu_model;
    model.init_config(gguf);
    model.alloc_buffers();
    model.init_gdn_states();
    model.init_attention(max_seq);

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
    for (int g = 0; g < n_gpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&gpu_hidden[g], H * sizeof(float));
        cudaMalloc(&gpu_hidden_half[g], H * sizeof(half));
        cudaMalloc(&gpu_hidden_chunk[g], CHUNK_SIZE * H * sizeof(float));
    }
    cudaSetDevice(last_gpu);
    half* logits_buf; cudaMalloc(&logits_buf, V * sizeof(half));
    half* norm_buf; cudaMalloc(&norm_buf, H * sizeof(half));
    float* host_transfer; cudaMallocHost(&host_transfer, H * sizeof(float));
    float* host_chunk_transfer; cudaMallocHost(&host_chunk_transfer, CHUNK_SIZE * H * sizeof(float));
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
    // Loaded only if /home/paru/mtp_work/mtp_head.bin exists.
    MTPHead mtp;
    bool mtp_loaded = false;
    {
        // Load the MTP head whose hidden size matches this model. The
        // 27B and 9B Qwen3.5 share the same MTP architecture but have
        // different dims, so we keep one bin per hidden size and pick
        // by name. (Falls back to legacy mtp_head.bin if the sized
        // file is not present, for backwards compat with prior fetches.)
        char mtp_path_buf[256];
        snprintf(mtp_path_buf, sizeof(mtp_path_buf),
                 "/home/paru/mtp_work/mtp_head_%d.bin", model.cfg.hidden_size);
        const char* mtp_path = mtp_path_buf;
        if (access(mtp_path, R_OK) != 0) {
            mtp_path = "/home/paru/mtp_work/mtp_head.bin";
        }
        if (access(mtp_path, R_OK) == 0) {
            // Pull RoPE table for last_gpu, embd source from GPU 0, lm_head from last_gpu.
            mtp_loaded = mtp.load(
                mtp_path, last_gpu,
                model.cfg.hidden_size, V,
                model.cfg.num_q_heads, model.cfg.num_kv_heads, model.cfg.head_dim,
                model.cfg.intermediate_size, model.cfg.rope_dim, max_seq, model.cfg.rms_norm_eps);
            if (mtp_loaded) {
                auto* embd_t  = gpu_model.get("token_embd.weight");
                auto* outw_t  = gpu_model.get("output.weight");
                mtp.set_embed_source(embd_t->data, embd_t->type, embd_t->gpu_id);
                mtp.set_lm_head(outw_t->data, outw_t->type);
                mtp.set_rope_tables(model.rope.sin_table(last_gpu), model.rope.cos_table(last_gpu));
                printf("[MTP] head ready, will measure acceptance rate during gen\n");
            } else {
                printf("[MTP] failed to load %s\n", mtp_path);
            }
        }
    }

    // Speculative decoding is the default whenever the MTP head is loaded.
    // Set MTP_SPEC_OFF=1 to fall back to the plain per-token loop. The
    // per-iter MTP measurement (formerly gated by MTP_ON=1) is suppressed
    // automatically when spec_enabled is true to avoid running MTP twice
    // per iter — the spec branch already runs its own draft MTP.
    spec_enabled = (mtp_loaded && getenv("MTP_SPEC_OFF") == nullptr);
    spec_k2_enabled   = (spec_enabled && getenv("MTP_K2")   != nullptr && getenv("MTP_TREE") == nullptr);
    spec_tree_enabled = (spec_enabled && getenv("MTP_TREE") != nullptr);
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
    if (spec_tree_enabled) {
        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&tree_hidden[g],      (size_t)tree_budget * H * sizeof(float));
            cudaMalloc(&tree_hidden_half[g], (size_t)tree_budget * H * sizeof(half));
        }
        cudaSetDevice(last_gpu);
        cudaMalloc(&tree_norm_buf,   (size_t)tree_budget * H * sizeof(half));
        cudaMalloc(&tree_logits_buf, (size_t)tree_budget * V * sizeof(half));
        cudaMalloc(&tree_d_argmax,   (size_t)tree_budget * sizeof(int));
        cudaMallocHost(&tree_h_argmax, (size_t)tree_budget * sizeof(int));
        cudaMallocHost(&tree_host_transfer, (size_t)tree_budget * H * sizeof(float));
        if (!h_final_draft1) cudaMalloc(&h_final_draft1, H * sizeof(half));   // reused for self-chain MTP
        // Second ping-pong buffer for chain depths > 3 (budget > 3).
        // We always alloc one so budget=3 path stays simple.
        cudaMalloc(&h_final_draft2_tree, H * sizeof(half));
        // qi_logits_tree lazily sizes its q8 buffer on first quantize_chunk.
        // tree forward reuses gdn_bufs[g].chunk_* buffers, already alloc'd in
        // model init. No additional buffer pool needed here.
        model.alloc_tree_decode(tree_budget);
        printf("[TREE] chain-tree decoding enabled (budget=%d, depth=chain)\n", tree_budget);
    }
    long long mtp_accept_count = 0;
    long long mtp_total_count  = 0;
    int mtp_pending_draft = -1;       // draft for some future step (MTP_DRAFT mode)
    int mtp_pending_draft_step = -1;
    long long spec_accept_count = 0;
    long long spec_total_count  = 0;

    SamplingParams sp;
    sp.rep_penalty = 1.0f;  // match llama.cpp default; users can set via repetition_penalty in request
    Sampler sampler;
    sampler.init(sp, V);

    // Unified generate. When `on_token` is provided, it's invoked once per
    // generated token id (in append order, including both tokens accepted by
    // a spec iter). The streaming wrapper below uses it to drive SSE chunks
    // so streaming benefits from the same spec decoding path as non-stream.
    auto generate_impl = [&](const std::vector<int>& prompt_ids, int max_tokens,
                             int* out_completion_tokens,
                             const std::function<void(int)>& on_token) -> std::string {
        model.reset_all_states();
        std::vector<int> generated;
        // No default cap on response length: if the caller doesn't pass
        // max_tokens we let the model run all the way to the end of the
        // KV context (minus a safety margin for the in-loop bound). It
        // will still stop on EOS or one of the other stop tags. Caller
        // can still set an explicit max_tokens to clip earlier.
        int max_gen = max_tokens > 0
                      ? max_tokens
                      : std::max(64, max_seq - (int)prompt_ids.size() - 64);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto t_first = t0;
        bool got_first_tok = false;
        bool in_think = false;
        int output_tokens = 0;
        // Reset MTP KV cache at the start of each generate() call so its
        // attention window matches the main model's per-request context.
        if (mtp_loaded) mtp.reset_kv();
        mtp_pending_draft = -1;
        mtp_pending_draft_step = -1;

        // ============ Phase 1: chunked prompt processing ============
        // Process prompt in CHUNK_SIZE token chunks. Skip the very last prompt
        // token so the existing per-token loop handles logits + sampling for it.
        int prompt_len = (int)prompt_ids.size();
        int prefill_len = prompt_len > 1 ? prompt_len - 1 : 0;
        int chunk_pos = 0;
        // PROFILE_PREFILL=1 env gates per-phase sync+timer. Adds sync overhead
        // so keep OFF for production. Phase totals printed at prefill end.
        const char* prof_env = getenv("PROFILE_PREFILL");
        const bool do_prof = prof_env && prof_env[0] == '1';
        const char* prof_attn_env = getenv("PROFILE_ATTN");
        g_profile_attn = prof_attn_env && prof_attn_env[0] == '1';
        // FlashAttention fused score+softmax+value is now default ON for the
        // 27B shape (head_dim=256, num_kv=4, num_q=24). ~1.8× prefill speedup
        // vs strict block-per-score, code-value accurate (Korean Rust
        // Singleton `value: 42` preserved). Set FLASH_ATTN=0 to force off.
        const char* fa_env = getenv("FLASH_ATTN");
        g_use_flash_attn = (fa_env == nullptr) ? true : (fa_env[0] == '1');
        g_attn_score_ms = g_attn_softmax_ms = g_attn_value_ms
                       = g_attn_fused_ms  = g_attn_other_ms = 0.0;
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
        int g_steps = 0, g_spec_iters = 0;
        // DISABLE_CHUNKED_PREFILL=1 : chunked path 우회. per-token loop 가
        // 처음부터 prompt 전체를 처리 (batch=1, state update 명시적 누적).
        // 9B 에서 chunked GDN 의 fp16 누적 오차가 언어 classification 을
        // 넘겨버리는지 확인용. 맞으면 chunked kernel 수정 타깃.
        static const bool skip_chunked = getenv("DISABLE_CHUNKED_PREFILL") != nullptr;
        if (skip_chunked) prefill_len = 0;
        while (chunk_pos < prefill_len) {
            int chunk_n = std::min(CHUNK_SIZE, prefill_len - chunk_pos);

            // 1. Embed all chunk tokens to gpu_hidden_chunk[0] (fp32)
            cudaSetDevice(0);
            auto te0 = prof_now();
            for (int t = 0; t < chunk_n; t++) {
                int token_id = prompt_ids[chunk_pos + t];
                // Vision override: substitute the pre-computed ViT embedding
                // for each <|image_pad|> token, in order of appearance. Falls
                // through to the normal dequant path when the token is not
                // an image placeholder or when vision is disabled.
                bool vision_hit = false;
                if (g_vision_embeds && g_image_pad_id >= 0 && token_id == g_image_pad_id) {
                    static int s_vision_idx = 0;
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

            // 2. Process chunk through all layers (transfer between GPUs as needed)
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
                // DEBUG envs for bit-exact bisect. Each env forces the chunked
                // control flow to call the per-token forward_* in a loop over
                // n_tokens instead of the batched chunked kernel. Combine with
                // DUMP_LAYERS=1 to diff against DISABLE_CHUNKED_PREFILL=1 path.
                static const bool force_pt_gdn  = getenv("CHUNK_FORCE_PT_GDN")  != nullptr;
                // ATTN chunked kernels: the score kernel now has a strict
                // variant (`attn_score_kernel_h_chunk_strict`, block-per-score
                // with per-token reduction tree) enabled by default. It's
                // argmax-stable with the per-token path, so per-token fallback
                // is no longer needed for correctness. CHUNK_FORCE_PT_ATTN=1
                // still forces per-token (debug/bisect). CHUNK_ATTN_FAST=1
                // swaps back to the old warp-per-score kernel (fastest but
                // ~1% argmax drift on Korean).
                static const bool force_pt_attn = getenv("CHUNK_FORCE_PT_ATTN") != nullptr;
                static const bool force_pt_mlp  = getenv("CHUNK_FORCE_PT_MLP")  != nullptr;
                if (is_attn) {
                    if (force_pt_attn) {
                        for (int tt = 0; tt < chunk_n; tt++) {
                            float* h_t = h_chunk + (size_t)tt * H;
                            model.forward_attn(layer, h_t, chunk_pos + tt, 0);
                        }
                    } else {
                        model.forward_attn_chunk(layer, h_chunk, chunk_pos, chunk_n, 0);
                    }
                } else {
                    if (force_pt_gdn) {
                        for (int tt = 0; tt < chunk_n; tt++) {
                            float* h_t = h_chunk + (size_t)tt * H;
                            model.forward_gdn(layer, h_t, 0);
                        }
                    } else {
                        model.forward_gdn_chunk(layer, h_chunk, chunk_n, 0);
                    }
                }
                if (do_prof) {
                    double ms = prof_sync_ms(ta0, g);
                    if (is_attn) t_attn += ms; else t_gdn += ms;
                }

                auto tm0 = prof_now();
                if (force_pt_mlp) {
                    for (int tt = 0; tt < chunk_n; tt++) {
                        float* h_t = h_chunk + (size_t)tt * H;
                        model.forward_mlp(layer, h_t, 0);
                    }
                } else {
                    model.forward_mlp_chunk(layer, h_chunk, chunk_n, 0);
                }
                if (do_prof) t_mlp += prof_sync_ms(tm0, g);
                // DUMP_LAYERS=1: chunked prefill 경로에서도 마지막 prompt
                // token 의 layer output hidden 처음 8 float 를 덤프한다.
                // per-token 경로 (DISABLE_CHUNKED_PREFILL=1) 와 같은 포맷
                // "[L%02d gdn/attn] ..." 로 찍어 diff 로 drift 레이어 bisect.
                static const bool dump_layers_chunk = getenv("DUMP_LAYERS") != nullptr;
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
            }
            cudaSetDevice(last_gpu); cudaDeviceSynchronize();

            chunk_pos += chunk_n;
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
        for (int step = prefill_len; step < step_cap; step++) {
            int token_id = step < (int)prompt_ids.size() ? prompt_ids[step] : generated.back();

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
                    model.forward_attn(layer, h, step, 0);
                else
                    model.forward_gdn(layer, h, 0);
                if (do_gen_prof) {
                    double ms = prof_sync_ms(ga0, g);
                    if (is_attn) g_attn += ms; else g_gdn += ms;
                }
                auto gm0 = prof_now();
                model.forward_mlp(layer, h, 0);
                if (do_gen_prof) g_mlp += prof_sync_ms(gm0, g);
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
                                && (sp_now.pres_penalty == 0.0f);
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
                if (mtp_loaded && !spec_enabled && step >= (int)prompt_ids.size() - 1 && getenv("MTP_ON")) {
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

                    // ===================== MTP_TREE chain-tree path (Phase 1) =====================
                    // Chain tree with parent_ids=[-1,0,1] is MTP K=2-equivalent in
                    // terms of sampled argmax outputs; we route through the
                    // forward_*_tree pipeline to shake out the DDTree kernels
                    // end-to-end. Once this matches K2, we'll widen to real
                    // branching by growing budget and adding ancestor-mask attn.
                    if (spec_tree_enabled) {
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
                    if (spec_k2_enabled) {
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
                            model.forward_mlp_n3(layer, h_a, h_b, h_c, 0);
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
                    } else if (spec_enabled) {
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
                            model.forward_mlp_n2(layer, h_a, h_b, 0);
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
            fflush(stdout);
        }

        if (out_completion_tokens) *out_completion_tokens = (int)generated.size();
        return tok.decode(generated);
    };

    // Non-streaming wrapper: call the unified generate with no per-token cb.
    auto generate = [&](const std::vector<int>& prompt_ids, int max_tokens,
                        int* out_completion_tokens) -> std::string {
        return generate_impl(prompt_ids, max_tokens, out_completion_tokens, nullptr);
    };

    HttpServer server;
    server.port = port;
    server.model_name = model_name;
    server.api_key = api_key;
    server.generate_fn = generate;
    // Match llama.cpp: no <think> prefill. 모델이 자율적으로 필요 시
    // `<think>` 열고 닫음. 웹 UI splitThink 는 `<think>` 로 시작하는
    // stream 만 think block 으로 접고, 바로 답 오는 경우는 all-answer
    // 로 표시 — 둘 다 자연스럽게 처리.
    server.prefills_think_tag = false;
    server.sampling_params = &sp;

    // Streaming wrapper: re-uses the unified generate_impl so it gets the
    // same spec decoding speedup as non-streaming. The on_token callback
    // decodes each token, buffers partial UTF-8 sequences, and pushes
    // complete chunks through the SSE callback.
    server.stream_generate_fn = [&](const std::vector<int>& prompt_ids, int max_tokens, StreamCallback cb) {
        std::string utf8_buf;
        auto on_tok = [&](int tok_id) {
            utf8_buf += tok.decode_token(tok_id);
            std::string complete = Tokenizer::extract_complete_utf8(utf8_buf);
            if (!complete.empty()) cb(complete, false);
        };
        int dummy_count = 0;
        generate_impl(prompt_ids, max_tokens, &dummy_count, on_tok);
        if (!utf8_buf.empty()) { cb(utf8_buf, false); utf8_buf.clear(); }
        cb("", true);
    };

    server.chat_encode_fn = [&](const std::vector<std::pair<std::string, std::string>>& msgs) {
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
        // Do not force <think>\n prefill: llama.cpp 참조 구현과 동일하게
        // 모델이 자율적으로 reasoning 여부 결정. 짧은 greeting 에 강제
        // reasoning 을 걸면 언어 drift / 환각 증폭 발생함 (실측 확인).
        auto ids = tok.apply_chat(sys_msg, conv, /*force_think=*/0);
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
    auto generate = [&](const std::vector<int>& prompt_ids, int max_tokens, int* out_completion_tokens) -> std::string {
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
    server.chat_encode_fn = [&](const std::vector<std::pair<std::string, std::string>>& msgs) {
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
    std::string prompt_text, api_key;
    std::string vision_mmproj_path, vision_test_image, image_raw_path;
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
        } else if (strcmp(argv[i], "--vision-mmproj") == 0 && i + 1 < argc) {
            vision_mmproj_path = argv[++i];
        } else if (strcmp(argv[i], "--vision-test") == 0 && i + 1 < argc) {
            vision_test_image = argv[++i];
        } else if (strcmp(argv[i], "--image-raw") == 0 && i + 1 < argc) {
            image_raw_path = argv[++i];
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
        cudaFree(d_img); cudaFree(d_out);
        return 0;
    }
    int tok_start = parse_sampling_args(argc, argv, sp);

    // Load tokenizer from GGUF
    Tokenizer tokenizer;
    tokenizer.load_from_gguf(gguf);

    // Vision + text inference: --vision-mmproj <mmproj.gguf> --image-raw <fp32.raw> -p "prompt"
    // Loads mmproj, runs ViT on the preprocessed image, stores embeddings in
    // g_vision_embeds. Later the prefill loop substitutes these for each
    // <|image_pad|> token in the prompt.
    if (!vision_mmproj_path.empty() && !image_raw_path.empty()) {
        GGUFFile vgg;
        if (!vgg.open(vision_mmproj_path.c_str())) {
            fprintf(stderr, "vision mmproj open failed\n");
            return 1;
        }
        static vision::VisionModel vm;   // static: lifetime = program
        if (!vm.load(vgg, 0)) {
            fprintf(stderr, "vision load failed\n");
            return 1;
        }
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
        // Publish to global hooks
        g_vision_embeds = d_vis;
        g_vision_n_tokens = Nm;
        g_vision_H = P;
        auto it = tokenizer.token_to_id.find("<|image_pad|>");
        if (it == tokenizer.token_to_id.end()) {
            fprintf(stderr, "<|image_pad|> token not found in vocab\n");
            return 1;
        }
        g_image_pad_id = it->second;
        printf("[vision] image_pad token id = %d, %d placeholders needed\n",
               g_image_pad_id, Nm);
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

    if (serve_port > 0 && arch == "qwen35") {
        ret = serve_qwen(gguf, gpu_model, n_gpus, serve_port, tokenizer, model_name, api_key, max_seq);
    } else if (serve_port > 0) {
        ret = serve_gemma(gguf, gpu_model, n_gpus, serve_port);
    } else if (chat_mode && arch == "qwen35") {
        ret = run_chat(gguf, gpu_model, n_gpus, sp, tokenizer);
    } else if (arch == "gemma4" || arch == "gemma2" || arch == "gemma3") {
        ret = run_gemma(gguf, gpu_model, n_gpus, sp, prompt_ids, &tokenizer);
    } else {
        ret = run_qwen(gguf, gpu_model, n_gpus, sp, prompt_ids, &tokenizer);
    }

    gpu_model.unload(); gguf.close();
    return ret;
}
