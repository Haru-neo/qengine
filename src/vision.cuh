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
#include <cublas_v2.h>

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

// BF16 → fp16 conversion. mmproj weights are tiny (max_abs ~0.5) so the cast
// is essentially lossless and avoids the Q8_0 quantization noise that — across
// 27 ViT blocks — was drowning out the color signal in the projector output.
__global__ void bf16_to_fp16_kernel(
    const uint16_t* __restrict__ bf16,
    half* __restrict__ out,
    size_t n
) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t u = ((uint32_t)bf16[i]) << 16;
    float f = __int_as_float(u);
    out[i] = __float2half(f);
}

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

// Map (py, px) patch coords to the zig-zag spatial-merge-friendly token
// index that llama.cpp's qwen3vl graph produces via its reshape/permute
// chain. Adjacent 2×2 patches end up as four contiguous tokens, so the
// later `spatial_merge` is just a 4-row concat.
//
// llama.cpp reshape (n_patches_x = pps along width, n_patches_y = pps along height):
//   conv  : [w, h, c]
//   permute(1,2,0,3)         → [c, w, h]
//   cont_4d(c*2, w/2, h)     → groups (col_pair, sub_col) into d0
//   reshape_4d(c*2, w/2, 2, h/2)
//   permute(0,2,1,3)         → [c*2, sub_row(2), w/2, h/2]
//   cont_3d(c, w*h, 1)       → token index i = ((h_pair*pps/2 + col_pair)*2 + sub_row)*2 + sub_col
__device__ __forceinline__ int spatial_zigzag_idx(int py, int px, int pps) {
    int h_pair   = py >> 1;
    int sub_row  = py & 1;
    int col_pair = px >> 1;
    int sub_col  = px & 1;
    int mps      = pps >> 1;
    return ((h_pair * mps + col_pair) << 2) + (sub_row << 1) + sub_col;
}

// VL_NO_ZIGZAG=1 disables the spatial zig-zag patch reorder — useful for
// A/B-ing accuracy against the row-major Stage-3 baseline.
__device__ __forceinline__ int spatial_zigzag_or_rowmajor(int py, int px, int pps, bool zigzag) {
    return zigzag ? spatial_zigzag_idx(py, px, pps) : (py * pps + px);
}

// Patch embedding: conv 3×kh×kw stride kh/kw → dense patch tokens.
// input : [3, H, W] fp32 pre-normalized
// weight: [out_ch, in_ch, kh, kw] (dims[0]=kh, dims[1]=kw, dims[2]=in_ch, dims[3]=out_ch in ggml order → out_ch * (kh*kw*in_ch) + ky*(kw*in_ch)… wait, actually check below)
// output: [n_patches, out_ch] in fp16, with tokens reordered into the
//         zig-zag spatial layout (see spatial_zigzag_idx).
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

    int patch_idx = spatial_zigzag_idx(py, px, pps);
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
        // ggml memory layout for patch_embd.weight: dims=[d0=kw, d1=kh, d2=ic, d3=oc]
        // d0 innermost → flat = oc*(ic*kh*kw) + ic*(kh*kw) + kh*kw + kw_idx
        // patch_shmem packs the cropped patch as [ic, ipy(kh), ipx(kw)] row-major.
        for (int ic = 0; ic < in_ch; ic++) {
            for (int ipy = 0; ipy < patch_sz; ipy++) {           // kh
                for (int ipx = 0; ipx < patch_sz; ipx++) {       // kw
                    int pi = ic * (patch_sz * patch_sz) + ipy * patch_sz + ipx;
                    int wi = ic * (patch_sz * patch_sz) + ipy * patch_sz + ipx;
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
// hidden [n_tokens, dim] is in zig-zag spatial-merge token order, while
// pos_embd in the GGUF is stored in raw row-major (py*pps+px) patch order.
// Map each output token back to its (py, px) so we sample the right pos row.
__global__ void add_pos_embd_kernel(
    half* __restrict__ hidden,            // [n_tokens, dim] zig-zag layout
    const float* __restrict__ pos,        // [pps*pps, dim] row-major (py, px)
    int pps, int dim
) {
    int mps      = pps >> 1;
    int h_pair   = blockIdx.z;
    int col_pair = blockIdx.y;
    int sub      = blockIdx.x;            // 0..3 = sub_row*2 + sub_col
    int sub_row  = sub >> 1;
    int sub_col  = sub & 1;
    int py       = h_pair * 2 + sub_row;
    int px       = col_pair * 2 + sub_col;
    int tok_idx  = ((h_pair * mps + col_pair) << 2) + sub;
    int pos_row  = py * pps + px;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        size_t hidx = (size_t)tok_idx * dim + c;
        size_t pidx = (size_t)pos_row * dim + c;
        float v = __half2float(hidden[hidx]) + pos[pidx];
        hidden[hidx] = __float2half(v);
    }
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

// Vision M-RoPE (matches ggml's rope_vision kernel for GGML_ROPE_TYPE_VISION).
// Q/K layout: [n_tokens, num_heads, head_dim]. NeoX-style pairing (k, k+n_dims/2)
// across the *full* head_dim. n_dims here = head_dim/2; sect_dims = sec_h + sec_w
// = head_dim/4 + head_dim/4 = head_dim/2. For k in [0..head_dim/2):
//   sector = k % sect_dims          // = k since k < sect_dims
//   if sector < sec_h:   theta = pos_h * theta_scale^k
//   else:                theta = pos_w * theta_scale^(k - sec_h)
//   x_low' = x_low*cos - x_high*sin    (low = idx k, high = idx k + head_dim/2)
//   x_high' = x_low*sin + x_high*cos
__global__ void vit_mrope_kernel(
    half* __restrict__ qk,                 // [n_tokens, num_heads, head_dim]
    const int* __restrict__ pos_h,         // [n_tokens]
    const int* __restrict__ pos_w,         // [n_tokens]
    int n_tokens, int num_heads, int head_dim,
    int sec_h, int sec_w, float theta_scale
) {
    int tok  = blockIdx.x;
    int head = blockIdx.y;
    if (tok >= n_tokens || head >= num_heads) return;
    int n_pairs = head_dim >> 1;            // = head_dim/2
    int ph = pos_h[tok];
    int pw = pos_w[tok];
    half* base = qk + ((size_t)tok * num_heads + head) * head_dim;
    int sect_dims = sec_h + sec_w;
    for (int k = threadIdx.x; k < n_pairs; k += blockDim.x) {
        int sector = k % sect_dims;         // = k for vision (k < sect_dims)
        float theta;
        if (sector < sec_h) {
            theta = (float)ph * powf(theta_scale, (float)sector);
        } else {
            theta = (float)pw * powf(theta_scale, (float)(sector - sec_h));
        }
        float c = cosf(theta);
        float s = sinf(theta);
        float x0 = __half2float(base[k]);
        float x1 = __half2float(base[k + n_pairs]);
        base[k]           = __float2half(x0 * c - x1 * s);
        base[k + n_pairs] = __float2half(x0 * s + x1 * c);
    }
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

// Spatial merge: in zig-zag layout each merged-token's 4 sub-tokens are
// already consecutive, so this is a straight 4-row concat. Input
// `in` rows are ordered ((h_pair, col_pair, sub_row, sub_col), embed_dim);
// output rows pack 4 consecutive input rows along the feature dim.
__global__ void spatial_merge_kernel(
    const half* __restrict__ in,   // [Nm * 4, embed_dim]
    half* __restrict__ out,        // [Nm, 4*embed_dim]
    int Nm, int embed_dim
) {
    int m = blockIdx.x;
    if (m >= Nm) return;
    int merge_dim = 4 * embed_dim;
    const half* in_base = in + (size_t)m * merge_dim;       // 4 rows, contiguous
    half* orow = out + (size_t)m * merge_dim;
    for (int i = threadIdx.x; i < merge_dim; i += blockDim.x) {
        orow[i] = in_base[i];
    }
}

// ============ Weights ============
struct Block {
    float* ln1_w;   float* ln1_b;
    half* qkv_w;    float* qkv_b;
    half* out_w;    float* out_b;
    float* ln2_w;   float* ln2_b;
    half* up_w;     float* up_b;
    half* down_w;   float* down_b;
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
    half* mm0_w = nullptr; float* mm0_b = nullptr;
    half* mm2_w = nullptr; float* mm2_b = nullptr;
    cublasHandle_t cublas = nullptr;

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
    int* d_pos_h = nullptr;          // [n_patches] zig-zag-ordered py
    int* d_pos_w = nullptr;          // [n_patches] zig-zag-ordered px
    QuantInput qi;

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
        // mmproj BF16 weights → fp16 (lossless cast — max_abs ~0.5 well within
        // fp16 normal range). Used by cuBLAS Hgemm in forward(). Earlier we
        // round-tripped through Q8_0 but the per-block scale quantization noise
        // accumulated across 27 ViT blocks and washed out the color signal in
        // the projector output (verified vs llama.cpp reference).
        auto copy_weight_to_gpu = [&](const TensorInfo* t) -> half* {
            if (!t) return nullptr;
            if (t->type == GGML_TYPE_F16) {
                half* dev;
                cudaMalloc(&dev, t->byte_size());
                cudaMemcpy(dev, t->data, t->byte_size(), cudaMemcpyHostToDevice);
                return dev;
            }
            if (t->type != GGML_TYPE_BF16) {
                fprintf(stderr, "[vision] unexpected weight type %d for %s\n",
                        (int)t->type, t->name.c_str());
                return nullptr;
            }
            size_t n = t->num_elements();
            void* bf16_dev;
            cudaMalloc(&bf16_dev, n * 2);
            cudaMemcpy(bf16_dev, t->data, n * 2, cudaMemcpyHostToDevice);
            half* fp16_dev;
            cudaMalloc(&fp16_dev, n * sizeof(half));
            int bt = 256;
            int bg = (int)((n + bt - 1) / bt);
            bf16_to_fp16_kernel<<<bg, bt>>>(
                (const uint16_t*)bf16_dev, fp16_dev, n);
            cudaDeviceSynchronize();
            cudaFree(bf16_dev);
            return fp16_dev;
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
            b.qkv_w = copy_weight_to_gpu(qkvw);
            b.qkv_b = (float*)copy_to_gpu(qkvb);
            b.out_w = copy_weight_to_gpu(outw);
            b.out_b = (float*)copy_to_gpu(outb);
            b.ln2_w = (float*)copy_to_gpu(ln2w);
            b.ln2_b = (float*)copy_to_gpu(ln2b);
            b.up_w  = copy_weight_to_gpu(upw);
            b.up_b  = (float*)copy_to_gpu(upb);
            b.down_w= copy_weight_to_gpu(dw);
            b.down_b= (float*)copy_to_gpu(db);
        }

        // --- Projector ---
        auto* mm0w = gguf.get_tensor("mm.0.weight");
        auto* mm0b = gguf.get_tensor("mm.0.bias");
        auto* mm2w = gguf.get_tensor("mm.2.weight");
        auto* mm2b = gguf.get_tensor("mm.2.bias");
        if (!mm0w || !mm2w) { fprintf(stderr, "[vision] missing projector\n"); return false; }
        mm0_w = copy_weight_to_gpu(mm0w);
        mm0_b = (float*)copy_to_gpu(mm0b);
        mm2_w = copy_weight_to_gpu(mm2w);
        mm2_b = (float*)copy_to_gpu(mm2b);

        // cuBLAS handle for fp16 GEMM
        if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[vision] cublasCreate failed\n");
            return false;
        }

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

        // Build M-RoPE position id buffers in zig-zag token order.
        int pps = cfg.num_patches_per_side();
        int mps = pps / 2;
        std::vector<int> h_ids(Np), w_ids(Np);
        for (int py = 0; py < pps; py++) {
            for (int px = 0; px < pps; px++) {
                int h_pair  = py >> 1;
                int sub_row = py & 1;
                int col_pair = px >> 1;
                int sub_col  = px & 1;
                int tok = ((h_pair * mps + col_pair) << 2) + (sub_row << 1) + sub_col;
                h_ids[tok] = py;
                w_ids[tok] = px;
            }
        }
        cudaMalloc(&d_pos_h, Np * sizeof(int));
        cudaMalloc(&d_pos_w, Np * sizeof(int));
        cudaMemcpy(d_pos_h, h_ids.data(), Np * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_w, w_ids.data(), Np * sizeof(int), cudaMemcpyHostToDevice);
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

        // 2. + position embedding (pos_embd in raw row-major patch order →
        //    re-mapped to zig-zag tokens via spatial_zigzag_idx).
        {
            int mps = pps >> 1;
            dim3 grid(4, mps, mps);   // (sub, col_pair, h_pair)
            add_pos_embd_kernel<<<grid, 256, 0, stream>>>(buf_hidden, position_embd, pps, E);
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
            gemm_fp16(cublas, buf_qkv, buf_proj_in, b.qkv_w, Np, 3*E, E, stream);
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
            // Vision M-RoPE on Q and K (matches ggml_rope_multi w/ TYPE_VISION).
            // n_dims = head_dim/2; sec_h = sec_w = head_dim/4. Bypass with
            // VL_NO_MROPE=1 for ablation against pre-rope baseline.
            if (!getenv("VL_NO_MROPE")) {
                int sec_h = Hd >> 2;             // head_dim/4
                int sec_w = Hd >> 2;
                int n_dims_rope = Hd >> 1;       // head_dim/2 = pair count (=36)
                // ggml convention: theta_scale = freq_base^(-2/n_dims). For
                // n_dims=36 this is 10000^(-1/18) — sec_h=18 covers freq exponents
                // 0..17, sec_w=18 covers 0..17 of the second axis.
                float theta_scale = powf(10000.0f, -2.0f / (float)n_dims_rope);
                dim3 grid(Np, Nh);
                int tx = (n_dims_rope < 64) ? n_dims_rope : 64;
                vit_mrope_kernel<<<grid, tx, 0, stream>>>(
                    buf_q, d_pos_h, d_pos_w, Np, Nh, Hd, sec_h, sec_w, theta_scale);
                vit_mrope_kernel<<<grid, tx, 0, stream>>>(
                    buf_k, d_pos_h, d_pos_w, Np, Nh, Hd, sec_h, sec_w, theta_scale);
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
            gemm_fp16(cublas, buf_proj_in, buf_attn_out, b.out_w, Np, E, E, stream);
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
            gemm_fp16(cublas, buf_ffn_in, buf_proj_in, b.up_w, Np, I, E, stream);
            {
                dim3 grid((I + 255)/256, Np);
                add_bias_gelu_kernel<<<grid, 256, 0, stream>>>(buf_ffn_in, b.up_b, Np, I);
            }
            gemm_fp16(cublas, buf_proj_in, buf_ffn_in, b.down_w, Np, E, I, stream);
            {
                dim3 grid((E + 255)/256, Np);
                add_bias_kernel<<<grid, 256, 0, stream>>>(buf_proj_in, b.down_b, Np, E);
            }
            residual_add_kernel_h<<<(Np*E+255)/256, 256, 0, stream>>>(
                buf_proj_in, buf_hidden_res, Np*E);
            cudaMemcpyAsync(buf_hidden, buf_proj_in, (size_t)Np*E*sizeof(half),
                            cudaMemcpyDeviceToDevice, stream);
            if (dbg) {
                cudaStreamSynchronize(stream);
                // Compact magnitude probe per block — track explosion/saturation.
                std::vector<half> hb(64);
                cudaMemcpy(hb.data(), buf_hidden, 64*sizeof(half), cudaMemcpyDeviceToHost);
                float mx = 0; for (auto h : hb) { float v = fabsf(__half2float(h)); if (v>mx) mx=v; }
                printf("[vl-dbg] block %2d row=0 maxabs(0..63)=%.3f sample=", l, mx);
                for (int i = 0; i < 5; i++) printf("%.3f ", __half2float(hb[i]));
                printf("\n");
            }
        }

        // 4. Post LN
        layer_norm_kernel<<<Np, 256, 0, stream>>>(
            buf_proj_in, buf_hidden, post_ln_w, post_ln_b, E, cfg.ln_eps);

        // 5. Spatial merge (zig-zag tokens → 4-row concat per merged token)
        {
            spatial_merge_kernel<<<Nm, 256, 0, stream>>>(
                buf_proj_in, buf_merged, Nm, E);
        }

        // 6. Projector: mm.0 (linear 4608→4608) + GELU + mm.2 (linear 4608→5120)
        gemm_fp16(cublas, buf_mm0_out, buf_merged, mm0_w, Nm, M, M, stream);
        {
            dim3 grid((M + 255)/256, Nm);
            add_bias_gelu_kernel<<<grid, 256, 0, stream>>>(buf_mm0_out, mm0_b, Nm, M);
        }
        gemm_fp16(cublas, buf_mm2_out, buf_mm0_out, mm2_w, Nm, P, M, stream);
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
