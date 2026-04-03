#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include "gguf.h"

struct GPUTensor {
    void* data = nullptr;
    int gpu_id = -1;
    ggml_type type;
    uint64_t dims[4];
    uint32_t n_dims;
    uint64_t byte_size;
    
    uint64_t num_elements() const {
        uint64_t n = 1;
        for (uint32_t i = 0; i < n_dims; i++) n *= dims[i];
        return n;
    }
};

struct GPUModel {
    int num_gpus = 0;
    int num_layers = 0;
    
    std::vector<int> layer_gpu;
    std::unordered_map<std::string, GPUTensor> tensors;
    
    bool load(GGUFFile& gguf, int n_gpus) {
        num_gpus = n_gpus;
        auto arch = gguf.get_str("general.architecture");
        num_layers = gguf.get_u32(arch + ".block_count");
        int layers_per_gpu = (num_layers + n_gpus - 1) / n_gpus;
        
        layer_gpu.resize(num_layers);
        for (int i = 0; i < num_layers; i++) {
            layer_gpu[i] = i / layers_per_gpu;
            if (layer_gpu[i] >= n_gpus) layer_gpu[i] = n_gpus - 1;
        }
        
        printf("\n=== GPU Assignment ===\n");
        for (int g = 0; g < n_gpus; g++) {
            int first = -1, last = -1;
            for (int i = 0; i < num_layers; i++) {
                if (layer_gpu[i] == g) {
                    if (first < 0) first = i;
                    last = i;
                }
            }
            printf("  GPU %d: layers %d-%d\n", g, first, last);
        }
        
        // Group tensors by target GPU
        std::vector<std::vector<std::pair<std::string, TensorInfo*>>> gpu_tensors(n_gpus);
        
        for (auto& [name, ti] : gguf.tensors) {
            int target_gpu = 0;
            if (name.substr(0, 4) == "blk.") {
                int layer_idx = std::stoi(name.substr(4, name.find('.', 4) - 4));
                target_gpu = layer_gpu[layer_idx];
            } else if (name == "token_embd.weight") {
                target_gpu = 0;
            } else if (name == "output.weight" || name == "output_norm.weight") {
                target_gpu = n_gpus - 1;
            }
            gpu_tensors[target_gpu].push_back({name, &ti});
        }
        
        // Parallel load — one thread per GPU
        std::atomic<size_t> total_bytes{0};
        std::vector<size_t> gpu_bytes(n_gpus, 0);
        std::mutex mtx;
        std::vector<std::unordered_map<std::string, GPUTensor>> per_gpu_tensors(n_gpus);
        std::atomic<bool> failed{false};
        
        printf("\nLoading to %d GPUs in parallel...\n", n_gpus);
        
        auto load_fn = [&](int gpu_id) {
            cudaSetDevice(gpu_id);
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            size_t local_bytes = 0;
            
            for (auto& [name, ti] : gpu_tensors[gpu_id]) {
                GPUTensor gt;
                gt.gpu_id = gpu_id;
                gt.type = ti->type;
                gt.n_dims = ti->n_dims;
                memcpy(gt.dims, ti->dims, sizeof(gt.dims));
                gt.byte_size = ti->byte_size();
                
                cudaError_t err = cudaMalloc(&gt.data, gt.byte_size);
                if (err != cudaSuccess) {
                    fprintf(stderr, "GPU %d: alloc failed for %s: %s\n",
                        gpu_id, name.c_str(), cudaGetErrorString(err));
                    failed = true;
                    break;
                }
                cudaMemcpyAsync(gt.data, ti->data, gt.byte_size, cudaMemcpyHostToDevice, stream);
                per_gpu_tensors[gpu_id][name] = gt;
                local_bytes += gt.byte_size;
            }
            
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
            
            std::lock_guard<std::mutex> lock(mtx);
            gpu_bytes[gpu_id] = local_bytes;
            total_bytes += local_bytes;
        };
        
        auto t0 = std::chrono::high_resolution_clock::now();
        
        std::vector<std::thread> threads;
        for (int g = 0; g < n_gpus; g++) {
            threads.emplace_back(load_fn, g);
        }
        for (auto& t : threads) t.join();
        
        auto t1 = std::chrono::high_resolution_clock::now();
        double load_secs = std::chrono::duration<double>(t1 - t0).count();
        
        if (failed) return false;
        
        // Merge per-GPU maps
        for (int g = 0; g < n_gpus; g++) {
            for (auto& [k, v] : per_gpu_tensors[g]) {
                tensors[k] = v;
            }
        }
        
        printf("\n=== VRAM Usage ===\n");
        for (int g = 0; g < n_gpus; g++) {
            printf("  GPU %d: %.1f MB (%zu tensors)\n", g, gpu_bytes[g] / 1e6, gpu_tensors[g].size());
        }
        printf("  Total: %.1f MB in %.2fs (%.1f GB/s)\n", 
            total_bytes.load() / 1e6, load_secs, total_bytes.load() / 1e9 / load_secs);
        
        return true;
    }
    
    GPUTensor* get(const std::string& name) {
        auto it = tensors.find(name);
        return it != tensors.end() ? &it->second : nullptr;
    }
    
    void unload() {
        for (auto& [name, gt] : tensors) {
            if (gt.data) {
                cudaSetDevice(gt.gpu_id);
                cudaFree(gt.data);
            }
        }
        tensors.clear();
    }
};
