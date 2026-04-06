#pragma once
// Stub - gemma support temporarily disabled while testing FP32 hidden state path
#include "gguf.h"
#include "gpu_loader.h"
struct GemmaModel {
    GPUModel* gpu;
    struct {
        int hidden_size = 0; int vocab_size = 0; int num_layers = 0;
        bool is_moe = false; float embd_scale = 1.0f;
        float softcap = 0.0f; float rms_norm_eps = 1e-6f;
    } cfg;
    void init_config(GGUFFile&) {}
    void alloc_buffers() {}
    void init_kv_caches(int, bool=false) {}
    void init_rope(int) {}
    void reset_all() {}
    bool is_full_attn(int) { return false; }
    void forward_full_attn(int, half*, int, cudaStream_t) {}
    void forward_sliding_attn(int, half*, int, cudaStream_t) {}
    void forward_moe(int, half*, cudaStream_t) {}
    void forward_mlp(int, half*, cudaStream_t) {}
};
