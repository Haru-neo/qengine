#include "gguf.h"
#include "gpu_loader.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    
    int n_gpus = 0;
    cudaGetDeviceCount(&n_gpus);
    printf("Found %d GPUs\n", n_gpus);
    
    GGUFFile gguf;
    if (!gguf.open(argv[1])) return 1;
    
    auto arch = gguf.get_str("general.architecture");
    printf("Model: %s (%s)\n", gguf.get_str("general.name").c_str(), arch.c_str());
    printf("Layers: %u, Hidden: %u\n", 
        gguf.get_u32(arch + ".block_count"),
        gguf.get_u32(arch + ".embedding_length"));
    
    GPUModel model;
    if (!model.load(gguf, n_gpus)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    // Verify a few tensors
    printf("\n=== Verification ===\n");
    auto* embd = model.get("token_embd.weight");
    if (embd) printf("token_embd: GPU %d, %.1f MB\n", embd->gpu_id, embd->byte_size/1e6);
    auto* out = model.get("output.weight");
    if (out) printf("output: GPU %d, %.1f MB\n", out->gpu_id, out->byte_size/1e6);
    auto* l0 = model.get("blk.0.attn_qkv.weight");
    if (l0) printf("blk.0.attn_qkv: GPU %d, %.1f MB\n", l0->gpu_id, l0->byte_size/1e6);
    auto* l63 = model.get("blk.63.attn_q.weight");
    if (l63) printf("blk.63.attn_q: GPU %d, %.1f MB\n", l63->gpu_id, l63->byte_size/1e6);
    
    model.unload();
    gguf.close();
    printf("\nDone!\n");
    return 0;
}
