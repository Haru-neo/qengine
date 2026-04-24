#pragma once
// Vision encoder for Qwopus3.6-27B mmproj (qwen3vl_merger).
//
// Input: preprocessed image tensor (3, 768, 768) already normalized with
//        v.image_mean / v.image_std on the caller side.
// Output: 576 × 5120 fp16 embeddings — 24×24 merged tokens ready to splice
//         into the LLM's token stream in place of an <image> placeholder.
//
// Architecture (from mmproj GGUF metadata):
//   patch_embd (conv 3×16×16 stride 16) → 48×48 tokens × 1152
//   + learned absolute position embedding (2304 × 1152)
//   × 27 SigLIP-style blocks (pre-LN + MHA + residual, pre-LN + FFN(GELU) + residual)
//   post_ln
//   spatial merge 2×2 → 24×24 tokens × 4608
//   projector: linear(4608→4608) → GELU → linear(4608→5120)
//
// All vision weights live on GPU0 alongside the LLM's embedding table. The
// 461 MB mmproj fits comfortably (Q8_0 quantized matrices + F32 norms).

#include "gguf.h"
#include "quant_gemv.cuh"
#include "ops.cuh"
#include <vector>
#include <string>
#include <cstring>

namespace vision {

// ============ Config ============
struct VisionConfig {
    int image_size = 768;
    int patch_size = 16;
    int embed_dim = 1152;
    int ffn_dim = 4304;
    int num_blocks = 27;
    int num_heads = 16;
    int head_dim = 72;        // 1152 / 16
    int proj_dim = 5120;      // LLM hidden
    int merge_size = 2;
    float ln_eps = 1e-6f;
    float img_mean[3] = {0.5f, 0.5f, 0.5f};
    float img_std[3]  = {0.5f, 0.5f, 0.5f};

    int num_patches_per_side() const { return image_size / patch_size; }       // 48
    int num_patches() const { return num_patches_per_side() * num_patches_per_side(); }  // 2304
    int num_merged_per_side() const { return num_patches_per_side() / merge_size; }      // 24
    int num_merged() const { return num_merged_per_side() * num_merged_per_side(); }     // 576
    int merge_flat_dim() const { return merge_size * merge_size * embed_dim; }           // 4608
};

// BF16 → Q8_0 (aligned) conversion. mmproj weight matrices are stored as BF16
// (ggml type 30); our GEMM kernels expect Q8_0. One thread per 32-element
// block: compute max-abs scale, quantize to int8, pack into
// block_q8_0_aligned (36 B = 32 qs + 2 pad + 2 fp16 scale).
__global__ void bf16_to_q8_0_aligned_kernel(
    const uint16_t* __restrict__ bf16,
    block_q8_0_aligned* __restrict__ out,
    int n_blocks
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const uint16_t* src = bf16 + (size_t)b * 32;
    float vals[32];
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        uint32_t u = ((uint32_t)src[i]) << 16;
        float f = __int_as_float(u);
        vals[i] = f;
        float a = fabsf(f);
        if (a > max_abs) max_abs = a;
    }
    float scale = max_abs / 127.0f;
    float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    block_q8_0_aligned* ob = &out[b];
    ob->d = __float2half(scale);
    ob->pad = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int q = (int)roundf(vals[i] * inv);
        q = max(-127, min(127, q));
        ob->qs[i] = (int8_t)q;
    }
}

// ============ Kernels ============

// LayerNorm with bias: out = ((x - mean) / sqrt(var + eps)) * weight + bias
// Block per row. `tokens` rows × `dim` cols.
__global__ void layer_norm_kernel(
    half* __restrict__ out,
    const half* __restrict__ in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int dim, float eps
) {
    int row = blockIdx.x;
    const half* xr = in + (size_t)row * dim;
    half* yr = out + (size_t)row * dim;

    // 1) mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        sum += __half2float(xr[i]);
    __shared__ float sm[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if (lane == 0) sm[warp] = sum;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    if (warp == 0) {
        sum = (lane < n_warps) ? sm[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane == 0) sm[0] = sum / (float)dim;
    }
    __syncthreads();
    float mean = sm[0];

    // 2) variance
    float vs = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = __half2float(xr[i]) - mean;
        vs += d * d;
    }
    for (int off = 16; off > 0; off >>= 1) vs += __shfl_xor_sync(0xffffffff, vs, off);
    if (lane == 0) sm[warp] = vs;
    __syncthreads();
    if (warp == 0) {
        vs = (lane < n_warps) ? sm[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) vs += __shfl_xor_sync(0xffffffff, vs, off);
        if (lane == 0) sm[1] = rsqrtf(vs / (float)dim + eps);
    }
    __syncthreads();
    float rstd = sm[1];

    // 3) normalize + scale/bias
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float x = __half2float(xr[i]);
        float n = (x - mean) * rstd;
        float y = n * weight[i] + bias[i];
        yr[i] = __float2half(y);
    }
}

// Patch embedding: conv 3×kh×kw stride kh/kw → dense patch tokens.
// input : [3, H, W] fp32 pre-normalized
// weight: [out_ch, in_ch, kh, kw] (dims[0]=kh, dims[1]=kw, dims[2]=in_ch, dims[3]=out_ch in ggml order → out_ch * (kh*kw*in_ch) + ky*(kw*in_ch)… wait, actually check below)
// output: [n_patches, out_ch] in fp16
// Each block handles ONE patch (py, px) and cooperatively computes all
// `embed_dim` output channels. Threads share the 768-value image patch via
// shared memory.
//
// Weight memory order (ggml, d0 innermost → outermost):
//   w[d3=out_ch][d2=in_ch][d1=kw][d0=ky]  ... wait need to verify from dump
// Dump said dims=[16,16,3,1152]. ggml convention: dims[0] = innermost.
// So flat idx = oc*(kh*kw*ic_max) + ic*(kh*kw) + kw_idx*kh + ky
// Wait actually since d0 is innermost (ky), the offset is
//    oc*(16*16*3) + ic*(16*16) + kw_idx*16 + ky
// Let's just match what llama.cpp does — treat it as im2col GEMM where the
// (ky, kw, ic) dims are unrolled per output channel.
__global__ void patch_embd_conv_kernel(
    half* __restrict__ out,                // [n_patches, out_ch] fp16
    const float* __restrict__ img,         // [3, H, W] fp32
    const float* __restrict__ w,           // [out_ch, in_ch, kh, kw] ggml order — see above
    const float* __restrict__ w1,          // second conv weight (qwen3vl, optional)
    const float* __restrict__ b,           // [out_ch]
    int H, int W, int patch_sz, int out_ch
) {
    int pps = H / patch_sz;                // patches per side
    int py = blockIdx.y;
    int px = blockIdx.x;
    if (py >= pps || px >= pps) return;

    int in_ch = 3;
    int k_total = in_ch * patch_sz * patch_sz;  // 768

    extern __shared__ float patch_shmem[];       // [k_total]
    // Load the 3x16x16 patch (768 floats) cooperatively.
    for (int i = threadIdx.x; i < k_total; i += blockDim.x) {
        int ic = i / (patch_sz * patch_sz);
        int ipy = (i % (patch_sz * patch_sz)) / patch_sz;
        int ipx = i % patch_sz;
        int y = py * patch_sz + ipy;
        int x = px * patch_sz + ipx;
        patch_shmem[i] = img[ic * H * W + y * W + x];
    }
    __syncthreads();

    int patch_idx = py * pps + px;
    // Each thread handles a stripe of out channels.
    for (int oc = threadIdx.x; oc < out_ch; oc += blockDim.x) {
        const float* wrow  = w  + (size_t)oc * k_total;
        const float* wrow1 = w1 ? w1 + (size_t)oc * k_total : nullptr;
        float acc = b ? b[oc] : 0.0f;
        // Match ggml order: weight mem layout is oc outermost, then ic, then kw,
        // then ky innermost.
        //   patch_shmem[i] where i = ic*(kh*kw) + ipy*kw + ipx (row-major HW)
        //   weight w[oc][ic][kx][ky]  (ggml d0=ky = innermost)
        //   i.e. w[oc * (kh*kw*ic_max) + ic*(kh*kw) + ipx*kh + ipy]
        // Qwen3-VL mmproj has BOTH patch_embd.weight and patch_embd.weight.1;
        // the build graph adds their outputs element-wise. When w1==nullptr
        // this term contributes 0.
        for (int ic = 0; ic < in_ch; ic++) {
            for (int ipx = 0; ipx < patch_sz; ipx++) {
                for (int ipy = 0; ipy < patch_sz; ipy++) {
                    int pi = ic * (patch_sz * patch_sz) + ipy * patch_sz + ipx;
                    int wi = ic * (patch_sz * patch_sz) + ipx * patch_sz + ipy;
                    float px = patch_shmem[pi];
                    acc += px * wrow[wi];
                    if (wrow1) acc += px * wrow1[wi];
                }
            }
        }
        out[(size_t)patch_idx * out_ch + oc] = __float2half(acc);
    }
}

// Add learned position embedding (fp32) to hidden (fp16) in place.
// hidden [n_tokens, dim], pos [n_tokens, dim]
__global__ void add_pos_embd_kernel(
    half* __restrict__ hidden,
    const float* __restrict__ pos,
    int n_tokens, int dim
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_tokens || col >= dim) return;
    size_t idx = (size_t)row * dim + col;
    float v = __half2float(hidden[idx]) + pos[idx];
    hidden[idx] = __float2half(v);
}

// Add broadcasted bias (fp32) to [n_tokens, dim] fp16 in place.
__global__ void add_bias_kernel(
    half* __restrict__ h, const float* __restrict__ b,
    int n_tokens, int dim
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_tokens || col >= dim) return;
    size_t idx = (size_t)row * dim + col;
    float v = __half2float(h[idx]) + b[col];
    h[idx] = __float2half(v);
}

// Fused bias add + GELU (tanh approx) in place.
__global__ void add_bias_gelu_kernel(
    half* __restrict__ h, const float* __restrict__ b,
    int n_tokens, int dim
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_tokens || col >= dim) return;
    size_t idx = (size_t)row * dim + col;
    float v = __half2float(h[idx]) + b[col];
    float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
    float g = 0.5f * v * (1.0f + tanhf(inner));
    h[idx] = __float2half(g);
}

// Residual add: dst += src (fp16)
__global__ void residual_add_kernel_h(
    half* __restrict__ dst, const half* __restrict__ src, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = __half2float(dst[i]);
    float b = __half2float(src[i]);
    dst[i] = __float2half(a + b);
}

// Deinterleave QKV for a single token: qkv[head*head_dim*3 + ...] → q/k/v [head*head_dim]
// Standard transformer QKV has output layout: row-major [n_tokens, 3*embed_dim]
// Want: q [n_tokens, num_heads, head_dim], k, v same.
// Assume qkv output is [n_tokens, num_heads, 3, head_dim] (common in SigLIP).
// Actually check: qkv_w output dim is 3456 = 3*1152. Typical convention is
// [n_tokens, 3 * embed_dim] split at the 3*embed_dim boundary (contiguous QKV).
// Or [n_tokens, num_heads * 3 * head_dim] interleaved by head.
// Qwen3-VL / SigLIP style typically is contiguous [Q | K | V].
// We pick contiguous-by-group (QQQ...|KKK...|VVV...).
__global__ void split_qkv_kernel(
    const half* __restrict__ qkv,  // [n_tokens, 3*embed_dim]
    half* __restrict__ q,          // [n_tokens, num_heads, head_dim]
    half* __restrict__ k,
    half* __restrict__ v,
    int n_tokens, int num_heads, int head_dim
) {
    int embed_dim = num_heads * head_dim;
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_tokens || col >= embed_dim) return;
    const half* src = qkv + (size_t)row * 3 * embed_dim;
    q[(size_t)row * embed_dim + col] = src[col];
    k[(size_t)row * embed_dim + col] = src[embed_dim + col];
    v[(size_t)row * embed_dim + col] = src[2 * embed_dim + col];
}

// Non-causal attention score.
// Grid (n_tokens_q, num_heads), block 256 threads.
// Each block caches Q[i, h] in shared memory and each thread loops over a
// stripe of K rows, computing scores[h, i, j] = dot(Q[i,h], K[j,h]) * scale.
// For 27B ViT at 2304 tokens × 16 heads = 37K blocks — no launch explosion.
__global__ void vit_attn_score_kernel(
    const half* __restrict__ q,    // [n_tokens, num_heads, head_dim]
    const half* __restrict__ k,    // same layout
    float* __restrict__ scores,    // [num_heads, n_tokens, n_tokens]
    int n_tokens, int num_heads, int head_dim, float scale
) {
    int i = blockIdx.x;
    int h = blockIdx.y;
    if (i >= n_tokens || h >= num_heads) return;

    const half* qi = q + ((size_t)i * num_heads + h) * head_dim;
    extern __shared__ float q_smem[];   // [head_dim]
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        q_smem[d] = __half2float(qi[d]);
    __syncthreads();

    float* out_row = scores + ((size_t)h * n_tokens + i) * n_tokens;
    for (int j = threadIdx.x; j < n_tokens; j += blockDim.x) {
        const half* kj = k + ((size_t)j * num_heads + h) * head_dim;
        float s = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < head_dim; d++)
            s += q_smem[d] * __half2float(kj[d]);
        out_row[j] = s * scale;
    }
}

// Softmax along last dim (n_tokens). grid (num_heads, n_tokens). Reuses the
// blockDim = power-of-2 covering n_tokens pattern from the main engine.
__global__ void vit_softmax_kernel(
    float* __restrict__ scores, int num_heads, int n_tokens
) {
    int h = blockIdx.x;
    int i = blockIdx.y;
    if (h >= num_heads || i >= n_tokens) return;
    float* row = scores + ((size_t)h * n_tokens + i) * n_tokens;
    extern __shared__ float sh[];
    float mv = -1e30f;
    for (int j = threadIdx.x; j < n_tokens; j += blockDim.x) mv = fmaxf(mv, row[j]);
    sh[threadIdx.x] = mv; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    mv = sh[0]; __syncthreads();
    float sm = 0.0f;
    for (int j = threadIdx.x; j < n_tokens; j += blockDim.x) {
        float e = expf(row[j] - mv);
        row[j] = e;
        sm += e;
    }
    sh[threadIdx.x] = sm; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.0f / sh[0];
    for (int j = threadIdx.x; j < n_tokens; j += blockDim.x) row[j] *= inv;
}

// Attention value multiply: out[i, h, d] = sum_j scores[h, i, j] * V[j, h, d]
__global__ void vit_attn_value_kernel(
    const float* __restrict__ scores,  // [h, Nq, Nk]
    const half* __restrict__ v,        // [Nk, h, d]
    half* __restrict__ out,            // [Nq, h, d]
    int n_tokens, int num_heads, int head_dim
) {
    int h = blockIdx.x;
    int i = blockIdx.y;
    if (h >= num_heads || i >= n_tokens) return;
    const float* sr = scores + ((size_t)h * n_tokens + i) * n_tokens;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float s = 0.0f;
        for (int j = 0; j < n_tokens; j++) {
            s += sr[j] * __half2float(v[((size_t)j * num_heads + h) * head_dim + d]);
        }
        out[((size_t)i * num_heads + h) * head_dim + d] = __float2half(s);
    }
}

// Spatial merge: reshape (pps, pps, embed_dim) patches into (mps, mps, merge²·embed_dim)
// pps = patches per side (48), ms = 2, mps = 24. Each merged token aggregates
// a 2×2 patch block by concatenation along the feature dim.
// Input order [row, col, ch]; output order [mrow, mcol, sub_row, sub_col, ch]
// where (sub_row, sub_col) ∈ {0,1}² flattens into the 4× feature block.
__global__ void spatial_merge_kernel(
    const half* __restrict__ in,  // [pps, pps, embed_dim]
    half* __restrict__ out,       // [mps, mps, merge*merge*embed_dim]
    int pps, int ms, int embed_dim
) {
    int mps = pps / ms;
    int mrow = blockIdx.y;
    int mcol = blockIdx.x;
    if (mrow >= mps || mcol >= mps) return;
    int merge_dim = ms * ms * embed_dim;
    half* orow = out + ((size_t)mrow * mps + mcol) * merge_dim;
    for (int i = threadIdx.x; i < merge_dim; i += blockDim.x) {
        int sub_flat = i / embed_dim;
        int ch = i % embed_dim;
        int sr = sub_flat / ms;
        int sc = sub_flat % ms;
        int pr = mrow * ms + sr;
        int pc = mcol * ms + sc;
        orow[i] = in[((size_t)pr * pps + pc) * embed_dim + ch];
    }
}

// ============ Weights ============
struct Block {
    float* ln1_w;   float* ln1_b;
    void* qkv_w;    ggml_type qkv_type; float* qkv_b;
    void* out_w;    ggml_type out_type; float* out_b;
    float* ln2_w;   float* ln2_b;
    void* up_w;     ggml_type up_type;  float* up_b;
    void* down_w;   ggml_type down_type; float* down_b;
};

struct VisionModel {
    VisionConfig cfg;
    int gpu_id = 0;

    float* patch_embd_w = nullptr;   // [out_ch, in_ch*kh*kw] fp32 on GPU
    float* patch_embd_w1 = nullptr;  // second conv (qwen3vl: x + conv0 + conv1)
    float* patch_embd_b = nullptr;
    float* position_embd = nullptr;
    std::vector<Block> blocks;
    float* post_ln_w = nullptr; float* post_ln_b = nullptr;
    void* mm0_w = nullptr; ggml_type mm0_type; float* mm0_b = nullptr;
    void* mm2_w = nullptr; ggml_type mm2_type; float* mm2_b = nullptr;

    // Scratch buffers
    half* buf_hidden = nullptr;      // [2304, 1152]
    half* buf_hidden_res = nullptr;  // residual pre-LN keep
    half* buf_qkv = nullptr;         // [2304, 3456]
    half* buf_q = nullptr;
    half* buf_k = nullptr;
    half* buf_v = nullptr;
    half* buf_attn_out = nullptr;    // [2304, 1152]
    half* buf_proj_in = nullptr;     // [2304, 1152]
    half* buf_ffn_in = nullptr;      // [2304, 4304]
    float* buf_scores = nullptr;     // [num_heads, 2304, 2304]
    half* buf_merged = nullptr;      // [576, 4608]
    half* buf_mm0_out = nullptr;     // [576, 4608]
    half* buf_mm2_out = nullptr;     // [576, 5120]
    QuantInput qi;
    QuantInput qi_merged;

    bool load(GGUFFile& gguf, int gpu) {
        gpu_id = gpu;
        cudaSetDevice(gpu_id);

        // Pull config from metadata
        cfg.image_size        = gguf.get_u32("clip.vision.image_size", 768);
        cfg.patch_size        = gguf.get_u32("clip.vision.patch_size", 16);
        cfg.embed_dim         = gguf.get_u32("clip.vision.embedding_length", 1152);
        cfg.ffn_dim           = gguf.get_u32("clip.vision.feed_forward_length", 4304);
        cfg.num_blocks        = gguf.get_u32("clip.vision.block_count", 27);
        cfg.num_heads         = gguf.get_u32("clip.vision.attention.head_count", 16);
        cfg.head_dim          = cfg.embed_dim / cfg.num_heads;
        cfg.proj_dim          = gguf.get_u32("clip.vision.projection_dim", 5120);
        cfg.merge_size        = gguf.get_u32("clip.vision.spatial_merge_size", 2);
        cfg.ln_eps            = gguf.get_f32("clip.vision.attention.layer_norm_epsilon", 1e-6f);

        printf("[vision] cfg: image=%d patch=%d embed=%d ffn=%d blocks=%d heads=%d proj=%d merge=%d\n",
               cfg.image_size, cfg.patch_size, cfg.embed_dim, cfg.ffn_dim,
               cfg.num_blocks, cfg.num_heads, cfg.proj_dim, cfg.merge_size);

        auto copy_to_gpu = [&](const TensorInfo* t) -> void* {
            if (!t) return nullptr;
            size_t bytes = t->byte_size();
            void* dev;
            cudaMalloc(&dev, bytes);
            cudaMemcpy(dev, t->data, bytes, cudaMemcpyHostToDevice);
            return dev;
        };
        // Quantized weights in mmproj are BF16 (ggml type 30). Convert to
        // Q8_0-aligned on-device so the existing GEMV/GEMM kernels work.
        // Sets out_type=GGML_TYPE_Q8_0 and returns a device pointer to
        // block_q8_0_aligned. Non-BF16 inputs passthrough unchanged.
        auto copy_weight_to_gpu = [&](const TensorInfo* t, ggml_type* out_type) -> void* {
            if (!t) { *out_type = GGML_TYPE_F32; return nullptr; }
            if (t->type != GGML_TYPE_BF16) {
                *out_type = t->type;
                return copy_to_gpu(t);
            }
            size_t n = t->num_elements();
            if (n % 32 != 0) {
                fprintf(stderr, "[vision] BF16 tensor %s n=%zu not 32-aligned\n",
                        t->name.c_str(), n);
                *out_type = GGML_TYPE_F32;
                return nullptr;
            }
            // Upload BF16 to temp device buffer, convert to aligned Q8_0.
            void* bf16_dev;
            cudaMalloc(&bf16_dev, n * 2);
            cudaMemcpy(bf16_dev, t->data, n * 2, cudaMemcpyHostToDevice);
            size_t n_blocks = n / 32;
            size_t q8_bytes = n_blocks * sizeof(block_q8_0_aligned);
            void* q8_dev;
            cudaMalloc(&q8_dev, q8_bytes);
            int bt = 128;
            int bg = (int)((n_blocks + bt - 1) / bt);
            bf16_to_q8_0_aligned_kernel<<<bg, bt>>>(
                (const uint16_t*)bf16_dev, (block_q8_0_aligned*)q8_dev, (int)n_blocks);
            cudaDeviceSynchronize();
            cudaFree(bf16_dev);
            *out_type = GGML_TYPE_Q8_0;
            return q8_dev;
        };

        // --- Global tensors ---
        patch_embd_w  = (float*)copy_to_gpu(gguf.get_tensor("v.patch_embd.weight"));
        patch_embd_w1 = (float*)copy_to_gpu(gguf.get_tensor("v.patch_embd.weight.1"));
        patch_embd_b  = (float*)copy_to_gpu(gguf.get_tensor("v.patch_embd.bias"));
        position_embd = (float*)copy_to_gpu(gguf.get_tensor("v.position_embd.weight"));
        post_ln_w = (float*)copy_to_gpu(gguf.get_tensor("v.post_ln.weight"));
        post_ln_b = (float*)copy_to_gpu(gguf.get_tensor("v.post_ln.bias"));

        if (!patch_embd_w || !position_embd || !post_ln_w) {
            fprintf(stderr, "[vision] missing core tensors\n");
            return false;
        }

        // --- Per-block tensors ---
        blocks.resize(cfg.num_blocks);
        for (int i = 0; i < cfg.num_blocks; i++) {
            auto name = [&](const char* suffix) -> std::string {
                return "v.blk." + std::to_string(i) + "." + suffix;
            };
            auto* ln1w = gguf.get_tensor(name("ln1.weight"));
            auto* ln1b = gguf.get_tensor(name("ln1.bias"));
            auto* qkvw = gguf.get_tensor(name("attn_qkv.weight"));
            auto* qkvb = gguf.get_tensor(name("attn_qkv.bias"));
            auto* outw = gguf.get_tensor(name("attn_out.weight"));
            auto* outb = gguf.get_tensor(name("attn_out.bias"));
            auto* ln2w = gguf.get_tensor(name("ln2.weight"));
            auto* ln2b = gguf.get_tensor(name("ln2.bias"));
            auto* upw  = gguf.get_tensor(name("ffn_up.weight"));
            auto* upb  = gguf.get_tensor(name("ffn_up.bias"));
            auto* dw   = gguf.get_tensor(name("ffn_down.weight"));
            auto* db   = gguf.get_tensor(name("ffn_down.bias"));
            if (!ln1w || !qkvw || !outw || !ln2w || !upw || !dw) {
                fprintf(stderr, "[vision] block %d missing tensors\n", i);
                return false;
            }
            Block& b = blocks[i];
            b.ln1_w = (float*)copy_to_gpu(ln1w);
            b.ln1_b = (float*)copy_to_gpu(ln1b);
            b.qkv_w = copy_weight_to_gpu(qkvw, &b.qkv_type);
            b.qkv_b = (float*)copy_to_gpu(qkvb);
            b.out_w = copy_weight_to_gpu(outw, &b.out_type);
            b.out_b = (float*)copy_to_gpu(outb);
            b.ln2_w = (float*)copy_to_gpu(ln2w);
            b.ln2_b = (float*)copy_to_gpu(ln2b);
            b.up_w  = copy_weight_to_gpu(upw, &b.up_type);
            b.up_b  = (float*)copy_to_gpu(upb);
            b.down_w= copy_weight_to_gpu(dw,  &b.down_type);
            b.down_b= (float*)copy_to_gpu(db);
        }

        // --- Projector ---
        auto* mm0w = gguf.get_tensor("mm.0.weight");
        auto* mm0b = gguf.get_tensor("mm.0.bias");
        auto* mm2w = gguf.get_tensor("mm.2.weight");
        auto* mm2b = gguf.get_tensor("mm.2.bias");
        if (!mm0w || !mm2w) { fprintf(stderr, "[vision] missing projector\n"); return false; }
        mm0_w = copy_weight_to_gpu(mm0w, &mm0_type);
        mm0_b = (float*)copy_to_gpu(mm0b);
        mm2_w = copy_weight_to_gpu(mm2w, &mm2_type);
        mm2_b = (float*)copy_to_gpu(mm2b);

        // --- Scratch buffers ---
        int Np = cfg.num_patches();        // 2304
        int Nm = cfg.num_merged();         // 576
        int E  = cfg.embed_dim;            // 1152
        int I  = cfg.ffn_dim;              // 4304
        int M  = cfg.merge_flat_dim();     // 4608
        int P  = cfg.proj_dim;             // 5120
        int Nh = cfg.num_heads;            // 16

        cudaMalloc(&buf_hidden,     (size_t)Np * E * sizeof(half));
        cudaMalloc(&buf_hidden_res, (size_t)Np * E * sizeof(half));
        cudaMalloc(&buf_qkv,        (size_t)Np * 3 * E * sizeof(half));
        cudaMalloc(&buf_q,          (size_t)Np * E * sizeof(half));
        cudaMalloc(&buf_k,          (size_t)Np * E * sizeof(half));
        cudaMalloc(&buf_v,          (size_t)Np * E * sizeof(half));
        cudaMalloc(&buf_attn_out,   (size_t)Np * E * sizeof(half));
        cudaMalloc(&buf_proj_in,    (size_t)Np * E * sizeof(half));
        cudaMalloc(&buf_ffn_in,     (size_t)Np * I * sizeof(half));
        cudaMalloc(&buf_scores,     (size_t)Nh * Np * Np * sizeof(float));
        cudaMalloc(&buf_merged,     (size_t)Nm * M * sizeof(half));
        cudaMalloc(&buf_mm0_out,    (size_t)Nm * M * sizeof(half));
        cudaMalloc(&buf_mm2_out,    (size_t)Nm * P * sizeof(half));

        size_t scratch_bytes = (size_t)Nh * Np * Np * 4
                             + (size_t)Np * (E * 8 + 3 * E + I) * 2
                             + (size_t)Nm * (M * 2 + P) * 2;
        printf("[vision] scratch buffers allocated (~%.1f MB on GPU %d)\n",
               scratch_bytes / (1024.0 * 1024.0), gpu_id);
        return true;
    }

    // Debug helper: snapshot N elements from a half buffer at position `row`
    // and print them as floats.
    static void dbg_print(const char* tag, half* buf, int row, int dim, int n=5) {
        std::vector<half> h(n);
        cudaMemcpy(h.data(), buf + (size_t)row * dim, n * sizeof(half), cudaMemcpyDeviceToHost);
        printf("[vl-dbg] %s row=%d: ", tag, row);
        for (int i = 0; i < n; i++) printf("%.4f ", __half2float(h[i]));
        printf("\n");
    }

    // Run ViT on a pre-normalized fp32 image [3, H, W] (device pointer) and
    // write 576 × proj_dim fp16 embeddings into `out`.
    void forward(const float* d_image, half* out, cudaStream_t stream = 0) {
        cudaSetDevice(gpu_id);
        bool dbg = getenv("VL_DEBUG") != nullptr;
        int pps = cfg.num_patches_per_side();   // 48
        int Np  = cfg.num_patches();             // 2304
        int E   = cfg.embed_dim;
        int I   = cfg.ffn_dim;
        int Nh  = cfg.num_heads;
        int Hd  = cfg.head_dim;
        int M   = cfg.merge_flat_dim();
        int Nm  = cfg.num_merged();
        int P   = cfg.proj_dim;
        float scale = 1.0f / sqrtf((float)Hd);

        // 1. Patch embed conv (qwen3vl: sum of two parallel convs)
        {
            dim3 grid(pps, pps);
            int smem = 3 * cfg.patch_size * cfg.patch_size * sizeof(float);
            patch_embd_conv_kernel<<<grid, 256, smem, stream>>>(
                buf_hidden, d_image, patch_embd_w, patch_embd_w1, patch_embd_b,
                cfg.image_size, cfg.image_size, cfg.patch_size, E);
        }
        if (dbg) {
            cudaStreamSynchronize(stream);
            dbg_print("patch row=0", buf_hidden, 0, E);
            dbg_print("patch row=1152", buf_hidden, 1152, E);
            dbg_print("patch row=2000", buf_hidden, 2000, E);
        }

        // 2. + position embedding
        {
            dim3 grid((E + 255)/256, Np);
            add_pos_embd_kernel<<<grid, 256, 0, stream>>>(buf_hidden, position_embd, Np, E);
        }
        if (dbg) {
            cudaStreamSynchronize(stream);
            dbg_print("pos   row=0", buf_hidden, 0, E);
            dbg_print("pos   row=1152", buf_hidden, 1152, E);
            dbg_print("pos   row=2000", buf_hidden, 2000, E);
        }

        // 3. 27 transformer blocks
        for (int l = 0; l < cfg.num_blocks; l++) {
            Block& b = blocks[l];
            bool deep = dbg && l == 0;
            // Keep residual copy
            cudaMemcpyAsync(buf_hidden_res, buf_hidden, (size_t)Np*E*sizeof(half),
                            cudaMemcpyDeviceToDevice, stream);
            // LN1
            layer_norm_kernel<<<Np, 256, 0, stream>>>(
                buf_proj_in, buf_hidden, b.ln1_w, b.ln1_b, E, cfg.ln_eps);
            if (deep) { cudaStreamSynchronize(stream); dbg_print("L0 ln1 row=0", buf_proj_in, 0, E); }
            // QKV = proj_in @ qkv_w + qkv_b
            qi.quantize_chunk(buf_proj_in, E, Np, stream);
            quant_gemv_chunk(b.qkv_w, b.qkv_type, qi.q8_buf, buf_qkv, E, 3*E, Np, stream);
            if (deep) { cudaStreamSynchronize(stream); dbg_print("L0 qkv pre-bias row=0", buf_qkv, 0, 3*E); }
            {
                dim3 grid((3*E + 255)/256, Np);
                add_bias_kernel<<<grid, 256, 0, stream>>>(buf_qkv, b.qkv_b, Np, 3*E);
            }
            if (deep) { cudaStreamSynchronize(stream); dbg_print("L0 qkv+bias row=0", buf_qkv, 0, 3*E); }
            // Split QKV
            {
                dim3 grid((E + 255)/256, Np);
                split_qkv_kernel<<<grid, 256, 0, stream>>>(buf_qkv, buf_q, buf_k, buf_v, Np, Nh, Hd);
            }
            if (deep) {
                cudaStreamSynchronize(stream);
                dbg_print("L0 q row=0", buf_q, 0, E);
                dbg_print("L0 k row=0", buf_k, 0, E);
                dbg_print("L0 v row=0", buf_v, 0, E);
            }
            // Score
            {
                dim3 grid(Np, Nh);
                int smem = Hd * sizeof(float);
                vit_attn_score_kernel<<<grid, 256, smem, stream>>>(
                    buf_q, buf_k, buf_scores, Np, Nh, Hd, scale);
            }
            // Softmax. n_tokens=2304 exceeds Volta's 1024-thread block limit,
            // so cap at 1024 and let threads loop over the row.
            {
                int st = 1; while (st < Np && st < 1024) st <<= 1;
                dim3 grid(Nh, Np);
                vit_softmax_kernel<<<grid, st, st*sizeof(float), stream>>>(
                    buf_scores, Nh, Np);
            }
            // Value
            {
                dim3 grid(Nh, Np);
                vit_attn_value_kernel<<<grid, 128, 0, stream>>>(
                    buf_scores, buf_v, buf_attn_out, Np, Nh, Hd);
            }
            if (deep) { cudaStreamSynchronize(stream); dbg_print("L0 attn_out row=0", buf_attn_out, 0, E); }
            // Output projection + bias
            qi.quantize_chunk(buf_attn_out, E, Np, stream);
            quant_gemv_chunk(b.out_w, b.out_type, qi.q8_buf, buf_proj_in, E, E, Np, stream);
            {
                dim3 grid((E + 255)/256, Np);
                add_bias_kernel<<<grid, 256, 0, stream>>>(buf_proj_in, b.out_b, Np, E);
            }
            // residual: buf_hidden = buf_hidden_res + buf_proj_in
            residual_add_kernel_h<<<(Np*E+255)/256, 256, 0, stream>>>(
                buf_proj_in, buf_hidden_res, Np*E);
            cudaMemcpyAsync(buf_hidden, buf_proj_in, (size_t)Np*E*sizeof(half),
                            cudaMemcpyDeviceToDevice, stream);

            // --- FFN ---
            cudaMemcpyAsync(buf_hidden_res, buf_hidden, (size_t)Np*E*sizeof(half),
                            cudaMemcpyDeviceToDevice, stream);
            layer_norm_kernel<<<Np, 256, 0, stream>>>(
                buf_proj_in, buf_hidden, b.ln2_w, b.ln2_b, E, cfg.ln_eps);
            qi.quantize_chunk(buf_proj_in, E, Np, stream);
            quant_gemv_chunk(b.up_w, b.up_type, qi.q8_buf, buf_ffn_in, E, I, Np, stream);
            {
                dim3 grid((I + 255)/256, Np);
                add_bias_gelu_kernel<<<grid, 256, 0, stream>>>(buf_ffn_in, b.up_b, Np, I);
            }
            qi.quantize_chunk(buf_ffn_in, I, Np, stream);
            quant_gemv_chunk(b.down_w, b.down_type, qi.q8_buf, buf_proj_in, I, E, Np, stream);
            {
                dim3 grid((E + 255)/256, Np);
                add_bias_kernel<<<grid, 256, 0, stream>>>(buf_proj_in, b.down_b, Np, E);
            }
            residual_add_kernel_h<<<(Np*E+255)/256, 256, 0, stream>>>(
                buf_proj_in, buf_hidden_res, Np*E);
            cudaMemcpyAsync(buf_hidden, buf_proj_in, (size_t)Np*E*sizeof(half),
                            cudaMemcpyDeviceToDevice, stream);
            if (dbg && (l == 0 || l == 13 || l == 26)) {
                cudaStreamSynchronize(stream);
                printf("[vl-dbg] after block %d:\n", l);
                dbg_print("  row=0   ", buf_hidden, 0, E);
                dbg_print("  row=1152", buf_hidden, 1152, E);
                dbg_print("  row=2000", buf_hidden, 2000, E);
            }
        }

        // 4. Post LN
        layer_norm_kernel<<<Np, 256, 0, stream>>>(
            buf_proj_in, buf_hidden, post_ln_w, post_ln_b, E, cfg.ln_eps);

        // 5. Spatial merge
        {
            int mps = cfg.num_merged_per_side();
            dim3 grid(mps, mps);
            spatial_merge_kernel<<<grid, 256, 0, stream>>>(
                buf_proj_in, buf_merged, pps, cfg.merge_size, E);
        }

        // 6. Projector: mm.0 (linear 4608→4608) + GELU + mm.2 (linear 4608→5120)
        qi_merged.quantize_chunk(buf_merged, M, Nm, stream);
        quant_gemv_chunk(mm0_w, mm0_type, qi_merged.q8_buf, buf_mm0_out, M, M, Nm, stream);
        {
            dim3 grid((M + 255)/256, Nm);
            add_bias_gelu_kernel<<<grid, 256, 0, stream>>>(buf_mm0_out, mm0_b, Nm, M);
        }
        qi_merged.quantize_chunk(buf_mm0_out, M, Nm, stream);
        quant_gemv_chunk(mm2_w, mm2_type, qi_merged.q8_buf, buf_mm2_out, M, P, Nm, stream);
        {
            dim3 grid((P + 255)/256, Nm);
            add_bias_kernel<<<grid, 256, 0, stream>>>(buf_mm2_out, mm2_b, Nm, P);
        }
        // Copy to output
        cudaMemcpyAsync(out, buf_mm2_out, (size_t)Nm*P*sizeof(half),
                        cudaMemcpyDeviceToDevice, stream);
    }
};

} // namespace vision
