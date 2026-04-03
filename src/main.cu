#include "gguf.h"
#include "gpu_loader.h"
#include "model.cuh"
#include "ops.cuh"
#include "turboquant.cuh"
#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 2) { printf("Usage: %s <model.gguf> [token_ids...]\n", argv[0]); return 1; }

    int n_gpus; cudaGetDeviceCount(&n_gpus);
    GGUFFile gguf;
    if (!gguf.open(argv[1])) return 1;

    GPUModel gpu_model;
    if (!gpu_model.load(gguf, n_gpus)) return 1;

    QwenModel model;
    model.gpu = &gpu_model;
    model.init_config(gguf);
    model.alloc_buffers();
    model.init_gdn_states();
    model.init_attention(4096);

    int H = model.cfg.hidden_size;
    int V = model.cfg.vocab_size;
    int last_gpu = gpu_model.layer_gpu[model.cfg.num_layers - 1];

    std::vector<int> prompt_ids;
    for (int i = 2; i < argc; i++) prompt_ids.push_back(atoi(argv[i]));
    if (prompt_ids.empty()) prompt_ids = {1};

    auto* embd_t = gpu_model.get("token_embd.weight");
    auto* out_norm_t = gpu_model.get("output_norm.weight");
    auto* out_w = gpu_model.get("output.weight");

    half* gpu_hidden[4];
    for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaMalloc(&gpu_hidden[g], H * sizeof(half)); }
    cudaSetDevice(last_gpu);
    half* logits_buf; cudaMalloc(&logits_buf, V * sizeof(half));
    half* norm_buf; cudaMalloc(&norm_buf, H * sizeof(half));
    half* host_transfer; cudaMallocHost(&host_transfer, H * sizeof(half));
    QuantInput qi_logits;

    model.reset_all_states();
    std::vector<int> generated;
    int max_gen = 500;

    printf("\n=== Generation (%zu prompt tokens) ===\n", prompt_ids.size());
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < (int)(prompt_ids.size() + max_gen); step++) {
        int token_id = step < (int)prompt_ids.size() ? prompt_ids[step] : generated.back();
        auto step_start = std::chrono::high_resolution_clock::now();

        cudaSetDevice(0);
        if (embd_t->type == GGML_TYPE_Q5_K)
            dequant_embd_q5k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden[0], token_id, H);
        else if (embd_t->type == GGML_TYPE_Q6_K)
            dequant_embd_q6k_row_v2<<<(H+255)/256, 256>>>(embd_t->data, gpu_hidden[0], token_id, H);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess && step == 0) printf("CUDA error after embd: %s\n", cudaGetErrorString(err));
        half* h = gpu_hidden[0];
        for (int layer = 0; layer < model.cfg.num_layers; layer++) {
            int g = model.gpu->layer_gpu[layer];
            int prev_g = (layer == 0) ? 0 : model.gpu->layer_gpu[layer - 1];
            if (g != prev_g) {
                cudaSetDevice(prev_g); cudaDeviceSynchronize();
                cudaMemcpy(host_transfer, h, H * sizeof(half), cudaMemcpyDeviceToHost);
                cudaSetDevice(g);
                cudaMemcpy(gpu_hidden[g], host_transfer, H * sizeof(half), cudaMemcpyHostToDevice);
                h = gpu_hidden[g];
            }
            cudaSetDevice(g);
            if (model.is_attn_layer(layer))
                model.forward_attn(layer, h, step, 0);
            else
                model.forward_gdn(layer, h, 0);
            model.forward_mlp(layer, h, 0);
        }

        if (step >= (int)prompt_ids.size() - 1) {
            cudaSetDevice(last_gpu); cudaDeviceSynchronize();


            if (out_norm_t->type == GGML_TYPE_F32)
                rms_norm_f32w(norm_buf, h, (float*)out_norm_t->data, 1, H, model.cfg.rms_norm_eps);
            else
                rms_norm(norm_buf, h, (half*)out_norm_t->data, 1, H, model.cfg.rms_norm_eps);

            qi_logits.quantize(norm_buf, H, 0);
            quant_gemv(out_w->data, out_w->type, norm_buf, logits_buf, H, V, &qi_logits);
            cudaDeviceSynchronize();

            std::vector<half> h_logits(V);
            cudaMemcpy(h_logits.data(), logits_buf, V * sizeof(half), cudaMemcpyDeviceToHost);

            float max_val = -1e30f; int max_idx = 0;
            for (int i = 0; i < V; i++) {
                float v = __half2float(h_logits[i]);
                if (v > max_val) { max_val = v; max_idx = i; }
            }

            auto step_end = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();

            if (step == (int)prompt_ids.size() - 1)
                printf("<think>=%.2f max[%d]=%.2f\n", __half2float(h_logits[248068]), max_idx, max_val);

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

    printf("Token IDs: ");
    for (int t : generated) printf("%d ", t);
    printf("\n");

    for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaFree(gpu_hidden[g]); }
    cudaFree(logits_buf); cudaFree(norm_buf); cudaFreeHost(host_transfer);
    gpu_model.unload(); gguf.close();
    return 0;
}
