#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include "gguf.h"
#include "quant_gemv.cuh"  // for q8_0_repack_kernel and block_q8_0_aligned

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

        layer_gpu.resize(num_layers);
        // PP_LAYER_BOUNDS="b1,b2,b3" overrides the 4-GPU split (cumulative
        // boundaries): [0,b1)->GPU0, [b1,b2)->GPU1, [b2,b3)->GPU2, [b3,N)->GPU3.
        // Lets us rebalance the prefill pipeline (the default 15/17/17/16 was
        // chosen for load-time repack-OOM avoidance, not runtime balance).
        const char* bounds_env = getenv("PP_LAYER_BOUNDS");
        if (n_gpus == 4 && num_layers == 65 && bounds_env) {
            int b1=15,b2=32,b3=49; sscanf(bounds_env, "%d,%d,%d", &b1,&b2,&b3);
            for (int i = 0; i < num_layers; i++)
                layer_gpu[i] = (i<b1)?0 : (i<b2)?1 : (i<b3)?2 : 3;
        } else if (n_gpus == 3 && num_layers == 65 && bounds_env) {
            // 3-GPU rebalance: PP_LAYER_BOUNDS="b1,b2" -> [0,b1)->GPU0,
            // [b1,b2)->GPU1, [b2,N)->GPU2. Used to thin GPU 0 (token_embd +
            // its layers) so the DFlash drafter (~1.8 GB) fits alongside a
            // 256K KV cache. e.g. "17,42".
            int b1=17, b2=42; sscanf(bounds_env, "%d,%d", &b1, &b2);
            for (int i = 0; i < num_layers; i++)
                layer_gpu[i] = (i<b1)?0 : (i<b2)?1 : 2;
        } else if (n_gpus == 4 && num_layers == 65) {
            // v2-MTP Q8_0: GPU 0 has token_embd, GPU 3 has output+MTP.
            // Shift layers away from GPU 0/3 to avoid repack-peak OOM.
            // 15 / 17 / 17 / 16
            for (int i = 0; i < num_layers; i++) {
                if      (i < 15) layer_gpu[i] = 0;
                else if (i < 32) layer_gpu[i] = 1;
                else if (i < 49) layer_gpu[i] = 2;
                else             layer_gpu[i] = 3;
            }
        } else {
            int layers_per_gpu = (num_layers + n_gpus - 1) / n_gpus;
            for (int i = 0; i < num_layers; i++) {
                layer_gpu[i] = i / layers_per_gpu;
                if (layer_gpu[i] >= n_gpus) layer_gpu[i] = n_gpus - 1;
            }
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
        
        // Group tensors by target GPU.
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

        // Sort each bucket by file offset so the per-GPU loader thread
        // walks the mmap region monotonically. gguf.tensors is an
        // unordered_map (random iteration order), and the random pattern
        // would defeat the kernel read-ahead we just hinted with
        // POSIX_FADV_SEQUENTIAL. With this sort each thread now produces
        // one sequential disk read stream over its layer range — four
        // sequential streams in total, which an SSD handles roughly as
        // well as one stream.
        for (int g = 0; g < n_gpus; g++) {
            std::sort(gpu_tensors[g].begin(), gpu_tensors[g].end(),
                      [](const auto& a, const auto& b) {
                          return a.second->data < b.second->data;
                      });
        }
        
        // Parallel load — one thread per GPU
        std::atomic<size_t> total_bytes{0};
        std::vector<size_t> gpu_bytes(n_gpus, 0);
        std::mutex mtx;
        std::vector<std::unordered_map<std::string, GPUTensor>> per_gpu_tensors(n_gpus);
        std::atomic<bool> failed{false};

        // NOTE: We deliberately do NOT call cudaHostRegister on the mmap
        // region. On this hardware (7.6 GB total RAM, 28 GB model file)
        // the bottleneck is disk read, not PCIe — pinning would only
        // matter if the cudaMemcpyAsync H2D path were the slow part, but
        // here the file pages are getting faulted in from the SATA SSD
        // and that's where the 200 MB/s ceiling comes from. We instead
        // help the disk path by hinting POSIX_FADV_SEQUENTIAL +
        // MADV_SEQUENTIAL (in gguf.h) and walking each per-GPU bucket
        // in increasing file-offset order (above), so the four loader
        // threads each produce one sequential read stream over its
        // layer range instead of an unordered scatter.

        printf("\nLoading to %d GPUs in parallel...\n", n_gpus);

        // Per-thread pinned staging buffer size. The worst-case single
        // tensor on Qwen3.5-27B is the lm_head at ~1.34 GB; bigger tensors
        // are chunked. Allocating 256 MB per thread × 4 threads = 1 GB
        // pinned host memory, which is well within the system RLIMIT_MEMLOCK
        // (we measured 970 MB on this box, so we use 200 MB instead — see
        // STAGE_BYTES below) and far smaller than the page-locked-whole-mmap
        // approach that hit RLIMIT_MEMLOCK / OOM.
        const size_t STAGE_BYTES = 200ull * 1024 * 1024;  // 200 MB

        const uint8_t* mmap_base = (const uint8_t*)gguf.mmap_addr;
        int gguf_fd = gguf.fd;

        auto load_fn = [&](int gpu_id) {
            cudaSetDevice(gpu_id);
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            size_t local_bytes = 0;

            // Per-thread pinned staging buffer. cudaMemcpyAsync from
            // pinned host memory runs as direct DMA at full PCIe speed
            // and the transfer can overlap with subsequent CPU work
            // (the next pread). Without pinning the H2D path is slower
            // and the cudaMemcpyAsync degenerates to a blocking copy.
            void* stage = nullptr;
            cudaError_t st_err = cudaMallocHost(&stage, STAGE_BYTES);
            if (st_err != cudaSuccess) {
                fprintf(stderr, "GPU %d: pinned staging alloc failed: %s\n",
                        gpu_id, cudaGetErrorString(st_err));
                failed = true;
                return;
            }

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

                // Copy this tensor in chunks: pread from the file straight
                // into pinned staging, then cudaMemcpyAsync staging → GPU.
                // We sync per chunk so we can reuse the staging buffer for
                // the next chunk; the per-chunk overhead is small compared
                // to the disk read.
                size_t file_off = (size_t)((const uint8_t*)ti->data - mmap_base);
                size_t copied = 0;
                while (copied < gt.byte_size) {
                    size_t chunk = std::min(STAGE_BYTES, gt.byte_size - copied);
                    ssize_t got = pread(gguf_fd, stage, chunk, file_off + copied);
                    if (got != (ssize_t)chunk) {
                        fprintf(stderr, "GPU %d: pread short read on %s (off=%zu want=%zu got=%zd)\n",
                                gpu_id, name.c_str(), file_off + copied, chunk, got);
                        failed = true;
                        break;
                    }
                    cudaMemcpyAsync((uint8_t*)gt.data + copied, stage, chunk,
                                    cudaMemcpyHostToDevice, stream);
                    cudaStreamSynchronize(stream);  // reuse stage for next chunk
                    copied += chunk;
                }
                if (failed) break;

                // Q8_0 repack: convert GGUF {half d; int8_t qs[32]} (34 B) to
                // GPU-resident block_q8_0_aligned {qs[32]; pad; d} (36 B) so the
                // dp4a GEMV kernel can use 4-byte aligned int loads. Halves the
                // weight memory transactions vs the u16-load fallback.
                if (ti->type == GGML_TYPE_Q8_0) {
                    int n_blocks = (int)(gt.byte_size / 34);
                    size_t new_bytes = (size_t)n_blocks * 36;
                    void* repacked = nullptr;
                    cudaError_t err2 = cudaMalloc(&repacked, new_bytes);
                    if (err2 != cudaSuccess) {
                        fprintf(stderr, "GPU %d: q8_0 repack alloc failed for %s: %s\n",
                            gpu_id, name.c_str(), cudaGetErrorString(err2));
                        failed = true;
                        break;
                    }
                    int rt = 256;
                    int rb = (n_blocks + rt - 1) / rt;
                    q8_0_repack_kernel<<<rb, rt, 0, stream>>>(gt.data, repacked, n_blocks);
                    cudaStreamSynchronize(stream);
                    cudaFree(gt.data);
                    gt.data = repacked;
                    gt.byte_size = new_bytes;
                }

                per_gpu_tensors[gpu_id][name] = gt;
                local_bytes += gt.byte_size;
            }

            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
            cudaFreeHost(stage);

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
