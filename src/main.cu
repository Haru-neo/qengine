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
#include <cstdio>
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
    int*  d_argmax;       cudaMalloc(&d_argmax, sizeof(int));
    int*  h_argmax_pinned; cudaMallocHost(&h_argmax_pinned, sizeof(int));

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
    if (spec_enabled) {
        for (int g = 0; g < n_gpus; g++) {
            cudaSetDevice(g);
            cudaMalloc(&gpu_hidden_b[g], H * sizeof(float));
            cudaMalloc(&gpu_hidden_half_b[g], H * sizeof(half));
        }
        cudaSetDevice(last_gpu);
        cudaMalloc(&norm_buf_b,   H * sizeof(half));
        cudaMalloc(&logits_buf_b, V * sizeof(half));
        cudaMalloc(&d_argmax_b,   sizeof(int));
        cudaMallocHost(&h_argmax_pinned_b, sizeof(int));
        cudaMallocHost(&host_transfer_b, H * sizeof(float));
        model.alloc_buffers_n2(max_seq);
        model.alloc_gdn_snapshots();
        printf("[SPEC] speculative decoding enabled (MTP K=1, MLP batched)\n");
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
        while (chunk_pos < prefill_len) {
            int chunk_n = std::min(CHUNK_SIZE, prefill_len - chunk_pos);

            // 1. Embed all chunk tokens to gpu_hidden_chunk[0] (fp32)
            cudaSetDevice(0);
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

            // 2. Process chunk through all layers (transfer between GPUs as needed)
            float* h_chunk = gpu_hidden_chunk[0];
            for (int layer = 0; layer < model.cfg.num_layers; layer++) {
                int g = gpu_model.layer_gpu[layer];
                int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer - 1];
                if (g != prev_g) {
                    cudaSetDevice(prev_g);
                    cudaMemcpy(host_chunk_transfer, h_chunk, (size_t)chunk_n * H * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaSetDevice(g);
                    cudaMemcpy(gpu_hidden_chunk[g], host_chunk_transfer, (size_t)chunk_n * H * sizeof(float), cudaMemcpyHostToDevice);
                    h_chunk = gpu_hidden_chunk[g];
                } else { cudaSetDevice(g); }

                if (model.is_attn_layer(layer))
                    model.forward_attn_chunk(layer, h_chunk, chunk_pos, chunk_n, 0);
                else
                    model.forward_gdn_chunk(layer, h_chunk, chunk_n, 0);
                model.forward_mlp_chunk(layer, h_chunk, chunk_n, 0);
            }
            cudaSetDevice(last_gpu); cudaDeviceSynchronize();

            chunk_pos += chunk_n;
        }

        // ============ Phase 2: per-token loop (handles last prompt token + generation) ============
        // The +4096 used to be a "think buffer" so reasoning chains
        // weren't counted against max_gen, but for unlimited responses
        // we just cap at the actual KV context capacity instead.
        int step_cap = std::min((int)prompt_ids.size() + max_gen + 4096, max_seq);
        for (int step = prefill_len; step < step_cap; step++) {
            int token_id = step < (int)prompt_ids.size() ? prompt_ids[step] : generated.back();

            cudaSetDevice(0);
            // Dequant embedding into fp16 scratch, then convert to fp32 hidden
            if (embd_t->type == GGML_TYPE_Q8_0)
                dequant_embd_q8_0_row<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
            else if (embd_t->type == GGML_TYPE_Q5_K)
                dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
            else if (embd_t->type == GGML_TYPE_Q6_K)
                dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden_half[0], token_id, H);
            half_to_float_kernel<<<(H+255)/256, 256>>>(gpu_hidden_half[0], gpu_hidden[0], H);

            float* h = gpu_hidden[0];
            for (int layer = 0; layer < model.cfg.num_layers; layer++) {
                int g = gpu_model.layer_gpu[layer];
                int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer - 1];
                if (g != prev_g) {
                    cudaSetDevice(prev_g);
                    cudaMemcpy(host_transfer, h, H * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaSetDevice(g);
                    cudaMemcpy(gpu_hidden[g], host_transfer, H * sizeof(float), cudaMemcpyHostToDevice);
                    h = gpu_hidden[g];
                } else { cudaSetDevice(g); }
                if (model.is_attn_layer(layer))
                    model.forward_attn(layer, h, step, 0);
                else
                    model.forward_gdn(layer, h, 0);
                model.forward_mlp(layer, h, 0);
            }

            if (step >= (int)prompt_ids.size() - 1) {
                cudaSetDevice(last_gpu); cudaDeviceSynchronize();
                // Convert fp32 hidden to fp16 for output norm + projection
                float_to_half_kernel<<<(H+255)/256, 256>>>(h, gpu_hidden_half[last_gpu], H);
                if (out_norm_t->type == GGML_TYPE_F32)
                    rms_norm_f32w(norm_buf, gpu_hidden_half[last_gpu], (float*)out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
                else
                    rms_norm(norm_buf, gpu_hidden_half[last_gpu], (half*)out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
                qi_logits.quantize(norm_buf, H, 0);
                quant_gemv(out_w->data, out_w->type, norm_buf, logits_buf, H, V, &qi_logits);

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
                    generated.push_back(max_idx);
                    if (on_token) on_token(max_idx);

                    // Track think state: tokens inside <think>...</think> don't count against max_tokens
                    if (max_idx == 248068) in_think = true;   // <think>
                    if (max_idx == 248069) in_think = false;  // </think>
                    if (!in_think) output_tokens++;

                    // Stop conditions
                    if (max_idx == 248046 || max_idx == 248044 || max_idx == 248045) break;
                    if (output_tokens >= max_gen) break;  // max_tokens only counts non-think output
                    // Stop on tool call end tags
                    if (generated.size() >= 4) {
                        std::string tail = tok.decode(std::vector<int>(generated.end()-4, generated.end()));
                        if (tail.find("</tool_call>") != std::string::npos) break;
                    }

                    // ===================== MTP_SPEC speculative decoding =====================
                    // After main has produced max_idx (= t_{step+1}) and we hold h[step] in
                    // norm_buf, draft t_{step+2} via MTP, then run a batched main forward at
                    // [step+1, step+2] with inputs [max_idx, mtp_draft] using the N=2 batched
                    // MLP path. Accept the draft if main's verify (logit at step+1) matches.
                    if (spec_enabled) {
                        // 1. MTP draft
                        int mtp_draft = mtp.forward(norm_buf, max_idx, step + 1);
                        cudaSetDevice(last_gpu);

                        // 2. Embed both tokens onto GPU 0
                        cudaSetDevice(0);
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

                        // 3. Run all 64 layers in N=2 batched mode
                        float* h_a = gpu_hidden[0];
                        float* h_b = gpu_hidden_b[0];
                        for (int layer = 0; layer < model.cfg.num_layers; layer++) {
                            int g_l = gpu_model.layer_gpu[layer];
                            int prev_g = (layer == 0) ? 0 : gpu_model.layer_gpu[layer - 1];
                            if (g_l != prev_g) {
                                cudaSetDevice(prev_g);
                                cudaMemcpy(host_transfer,   h_a, H * sizeof(float), cudaMemcpyDeviceToHost);
                                cudaMemcpy(host_transfer_b, h_b, H * sizeof(float), cudaMemcpyDeviceToHost);
                                cudaSetDevice(g_l);
                                cudaMemcpy(gpu_hidden[g_l],   host_transfer,   H * sizeof(float), cudaMemcpyHostToDevice);
                                cudaMemcpy(gpu_hidden_b[g_l], host_transfer_b, H * sizeof(float), cudaMemcpyHostToDevice);
                                h_a = gpu_hidden[g_l];
                                h_b = gpu_hidden_b[g_l];
                            } else {
                                cudaSetDevice(g_l);
                            }
                            if (model.is_attn_layer(layer))
                                model.forward_attn_n2(layer, h_a, h_b, step + 1, step + 2, 0);
                            else
                                model.forward_gdn_n2(layer, h_a, h_b, 0);
                            model.forward_mlp_n2(layer, h_a, h_b, 0);
                        }

                        // 4. Output norm + lm_head for both
                        cudaSetDevice(last_gpu);
                        cudaDeviceSynchronize();
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

                        bool spec_accept = (max_idx_a == mtp_draft);
                        if (spec_total_count < 3)
                            printf("[SPEC] step=%d main=%d draft=%d verify=%d %s prov=%d\n",
                                   step, max_idx, mtp_draft, max_idx_a,
                                   spec_accept ? "ACCEPT" : "reject", max_idx_b);
                        spec_total_count++;
                        if (spec_accept) spec_accept_count++;

                        if (spec_accept) {
                            // Append the accepted draft + the provisional next token.
                            generated.push_back(mtp_draft);   // = t_{step+2}
                            if (on_token) on_token(mtp_draft);
                            if (mtp_draft == 248068) in_think = true;
                            if (mtp_draft == 248069) in_think = false;
                            if (!in_think) output_tokens++;
                            bool stop_now = false;
                            if (mtp_draft == 248046 || mtp_draft == 248044 || mtp_draft == 248045) stop_now = true;

                            if (!stop_now && output_tokens < max_gen) {
                                generated.push_back(max_idx_b);   // = t_{step+3}
                                if (on_token) on_token(max_idx_b);
                                if (max_idx_b == 248068) in_think = true;
                                if (max_idx_b == 248069) in_think = false;
                                if (!in_think) output_tokens++;
                                if (max_idx_b == 248046 || max_idx_b == 248044 || max_idx_b == 248045) stop_now = true;
                            }
                            if (stop_now || output_tokens >= max_gen) break;
                            // Skip outer loop iters at step+1 and step+2 (we processed them
                            // via the batched verify). norm_buf gets re-derived next iter
                            // from main's forward at the new step.
                            step += 2;
                        } else {
                            // Reject: keep main's verify (max_idx_a = t_{step+2}_main).
                            generated.push_back(max_idx_a);
                            if (on_token) on_token(max_idx_a);
                            if (max_idx_a == 248068) in_think = true;
                            if (max_idx_a == 248069) in_think = false;
                            if (!in_think) output_tokens++;
                            if (max_idx_a == 248046 || max_idx_a == 248044 || max_idx_a == 248045) break;
                            if (output_tokens >= max_gen) break;
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
        if (mtp_loaded && mtp_total_count > 0) {
            printf("[MTP] accept %lld / %lld = %.1f%%\n",
                   mtp_accept_count, mtp_total_count,
                   100.0 * mtp_accept_count / mtp_total_count);
        }
        if (spec_enabled && spec_total_count > 0) {
            printf("[SPEC] accept %lld / %lld = %.1f%%  (per-iter avg tokens %.2f)\n",
                   spec_accept_count, spec_total_count,
                   100.0 * spec_accept_count / spec_total_count,
                   1.0 + (double)spec_accept_count / spec_total_count);
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
    server.prefills_think_tag = true;
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
        // Prefill <think>\n (matches gist template default)
        return tok.apply_chat(sys_msg, conv, /*force_think=*/1);
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
        }
    }
    int tok_start = parse_sampling_args(argc, argv, sp);

    // Load tokenizer from GGUF
    Tokenizer tokenizer;
    tokenizer.load_from_gguf(gguf);

    // Build prompt token IDs
    std::vector<int> prompt_ids;
    if (!prompt_text.empty()) {
        // Text prompt: encode with appropriate chat template
        if (tokenizer.is_sentencepiece)
            prompt_ids = tokenizer.apply_chat_gemma("", {{"user", prompt_text}});
        else
            prompt_ids = tokenizer.encode_chat(prompt_text);
        printf("Prompt: \"%s\" → %zu tokens [", prompt_text.c_str(), prompt_ids.size());
        for (size_t i = 0; i < std::min(prompt_ids.size(), (size_t)20); i++) printf("%d ", prompt_ids[i]);
        printf("]\n");
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
