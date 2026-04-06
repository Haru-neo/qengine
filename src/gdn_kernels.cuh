#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

// ============ Conv1d Update (seq_len=1) ============
// conv_state: [dim, kernel_width=4] 
// input: [dim] (new token's qkv)
// output: [dim] (conv1d result with SiLU)
// weight: [kernel_width=4, dim] stored as [dim, kernel_width] in GGUF

__global__ void conv1d_update_silu(
    float* __restrict__ conv_state,  // [dim, kw] FP32
    const half* __restrict__ input,  // [dim]
    const float* __restrict__ weight, // [kw, dim] F32
    float* __restrict__ output,      // [dim] FP32
    int dim, int kw
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    float* st = conv_state + d * kw;
    for (int i = 0; i < kw - 1; i++)
        st[i] = st[i + 1];
    st[kw - 1] = __half2float(input[d]);

    float sum = 0.0f;
    for (int i = 0; i < kw; i++)
        sum += weight[d * kw + i] * st[i];

    float silu = sum / (1.0f + expf(-sum));
    output[d] = silu;
}

// ============ L2 Norm (per head) ============

__device__ float l2norm_head(const half* data, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float v = __half2float(data[i]);
        sum += v * v;
    }
    return rsqrtf(sum + 1e-6f);
}

// ============ GDN Recurrent Step (seq_len=1) ============
// Formula: S = exp(g) * S + k ⊗ beta * (v - exp(g) * S^T k)
//          o = q^T * S
// g = -exp(A_log) * softplus(alpha + dt_bias)  (negative, so exp(g) < 1 = decay)
// beta = sigmoid(b_proj)
// Q, K = L2 normalized

__global__ void gdn_recurrent_step(
    const float* __restrict__ qkv,   // FP32 input from conv1d
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const half* __restrict__ a_proj,
    const half* __restrict__ b_proj,
    double* __restrict__ rec_state,  // FP64 state for precision
    half* __restrict__ core_out,
    int num_k, int num_v, int k_dim, int v_dim
) {
    int head = blockIdx.x;
    if (head >= num_v) return;
    // qwen35 uses ggml_repeat_4d (tile pattern): v_head h → k_head h%num_k
    int k_head = head % num_k;

    const float* q_raw = qkv + k_head * k_dim;
    const float* k_raw = qkv + num_k * k_dim + k_head * k_dim;
    const float* v_raw = qkv + 2 * num_k * k_dim + head * v_dim;

    extern __shared__ float smem[];
    float* sQ = smem;
    float* sK = smem + k_dim + 1;

    // 1. L2 norm Q and K
    if (threadIdx.x == 0) {
        float q_norm_sq = 0.0f, k_norm_sq = 0.0f;
        for (int i = 0; i < k_dim; i++) {
            float qv = q_raw[i];
            float kv = k_raw[i];
            q_norm_sq += qv * qv;
            k_norm_sq += kv * kv;
        }
        float q_inv = rsqrtf(q_norm_sq + 1e-6f);
        float k_inv = rsqrtf(k_norm_sq + 1e-6f);
        float scale = rsqrtf((float)k_dim);
        float attn_score = 0.0f;
        for (int i = 0; i < k_dim; i++) {
            float q_n = q_raw[i] * q_inv;
            float k_n = k_raw[i] * k_inv;
            sQ[i] = q_n * scale;
            sK[i] = k_n;
            attn_score += q_n * k_n;
        }
        attn_score *= scale;
        sQ[k_dim] = attn_score;
    }
    __syncthreads();

    float attn_score = sQ[k_dim];

    // 2. Compute gate and beta in FP64
    double alpha = (double)__half2float(a_proj[head]);
    double dt = (double)dt_bias[head];
    double ssm_a_val = (double)a_log[head];
    double g_log = ssm_a_val * log(1.0 + exp(alpha + dt));
    double g = exp(fmin(g_log, 50.0));
    double beta = 1.0 / (1.0 + exp(-(double)__half2float(b_proj[head])));

    double* state = rec_state + head * k_dim * v_dim;

    // 3. State update in FP64
    for (int vi = threadIdx.x; vi < v_dim; vi += blockDim.x) {
        double sum1 = 0.0, sum2 = 0.0;
        for (int kd = 0; kd < k_dim; kd++) {
            double s = state[kd * v_dim + vi];
            sum1 += s * (double)sK[kd];
            sum2 += s * (double)sQ[kd];
        }
        double sv_new = beta * ((double)v_raw[vi] - sum1 * g);
        double out_val = sum2 * g + sv_new * (double)attn_score;
        core_out[head * v_dim + vi] = __float2half((float)out_val);

        for (int kd = 0; kd < k_dim; kd++) {
            double s_new = g * state[kd * v_dim + vi] + sv_new * (double)sK[kd];
            if (s_new > 1e6) s_new = 1e6;
            else if (s_new < -1e6) s_new = -1e6;
            state[kd * v_dim + vi] = s_new;
        }
    }
}

// ============ Chunked Conv1d Update (multi-token) ============
// Updates conv_state and produces conv_out for N tokens sequentially.
// One thread per channel processes all N tokens.
__global__ void conv1d_update_silu_chunk(
    float* __restrict__ conv_state,    // [dim, kw] FP32
    const half* __restrict__ chunk_in, // [N, dim] (qkv proj outputs as fp16)
    const float* __restrict__ weight,  // [kw, dim]
    float* __restrict__ chunk_out,     // [N, dim] FP32
    int dim, int kw, int n_tokens
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float* st = conv_state + d * kw;
    float w0 = weight[d * kw + 0];
    float w1 = weight[d * kw + 1];
    float w2 = weight[d * kw + 2];
    float w3 = weight[d * kw + 3];
    for (int t = 0; t < n_tokens; t++) {
        // Shift state, insert new
        st[0] = st[1];
        st[1] = st[2];
        st[2] = st[3];
        st[3] = __half2float(chunk_in[(size_t)t * dim + d]);
        float sum = w0*st[0] + w1*st[1] + w2*st[2] + w3*st[3];
        float silu = sum / (1.0f + expf(-sum));
        chunk_out[(size_t)t * dim + d] = silu;
    }
}

// ============ Chunked GDN Recurrent Step ============
// Processes N tokens through GDN with FP64 state.
// Implements the standard recurrence in a single kernel call (state stays
// in shared/global memory across tokens, no kernel-launch overhead per token).
//
// Future work: replace this with HF chunked-matrix form using a decay_mask
// matrix and (I+L)^-1 transformation for true precision benefit.
__global__ void gdn_chunk_step(
    const float* __restrict__ chunk_qkv,  // [N, qkv_dim] FP32
    const float* __restrict__ a_log,      // [num_v]
    const float* __restrict__ dt_bias,    // [num_v]
    const half* __restrict__ chunk_a,     // [N, num_v]
    const half* __restrict__ chunk_b,     // [N, num_v]
    double* __restrict__ rec_state,       // FP64 state, updated in-place
    half* __restrict__ chunk_out,         // [N, num_v * v_dim]
    int n_tokens,
    int num_k, int num_v, int k_dim, int v_dim
) {
    int head = blockIdx.x;
    if (head >= num_v) return;
    // qwen35 uses ggml_repeat_4d (tile pattern): v_head h → k_head h%num_k
    int k_head = head % num_k;
    int qkv_dim = 2 * num_k * k_dim + num_v * v_dim;
    int v_total = num_v * v_dim;

    extern __shared__ float smem[];
    float* sQ = smem;             // [k_dim+1] q_norm scaled, last slot = attn_score
    float* sK = smem + k_dim + 1; // [k_dim] k_norm

    double* state = rec_state + head * k_dim * v_dim;

    for (int t = 0; t < n_tokens; t++) {
        const float* qkv = chunk_qkv + (size_t)t * qkv_dim;
        const float* q_raw = qkv + k_head * k_dim;
        const float* k_raw = qkv + num_k * k_dim + k_head * k_dim;
        const float* v_raw = qkv + 2 * num_k * k_dim + head * v_dim;
        const half* a_proj = chunk_a + (size_t)t * num_v;
        const half* b_proj = chunk_b + (size_t)t * num_v;

        // L2 norm Q, K and compute attn_score (single-thread setup)
        if (threadIdx.x == 0) {
            float q_norm_sq = 0.0f, k_norm_sq = 0.0f;
            for (int i = 0; i < k_dim; i++) {
                float qv = q_raw[i];
                float kv = k_raw[i];
                q_norm_sq += qv * qv;
                k_norm_sq += kv * kv;
            }
            float q_inv = rsqrtf(q_norm_sq + 1e-6f);
            float k_inv = rsqrtf(k_norm_sq + 1e-6f);
            float scale = rsqrtf((float)k_dim);
            float attn_score = 0.0f;
            for (int i = 0; i < k_dim; i++) {
                float q_n = q_raw[i] * q_inv;
                float k_n = k_raw[i] * k_inv;
                sQ[i] = q_n * scale;
                sK[i] = k_n;
                attn_score += q_n * k_n;
            }
            attn_score *= scale;
            sQ[k_dim] = attn_score;
        }
        __syncthreads();

        float attn_score = sQ[k_dim];

        // Gate and beta in FP64
        double alpha = (double)__half2float(a_proj[head]);
        double dt = (double)dt_bias[head];
        double ssm_a_val = (double)a_log[head];
        double g_log = ssm_a_val * log(1.0 + exp(alpha + dt));
        double g = exp(fmin(g_log, 50.0));
        double beta = 1.0 / (1.0 + exp(-(double)__half2float(b_proj[head])));

        // Per-v state update + output
        for (int vi = threadIdx.x; vi < v_dim; vi += blockDim.x) {
            double sum1 = 0.0, sum2 = 0.0;
            for (int kd = 0; kd < k_dim; kd++) {
                double s = state[kd * v_dim + vi];
                sum1 += s * (double)sK[kd];
                sum2 += s * (double)sQ[kd];
            }
            double sv_new = beta * ((double)v_raw[vi] - sum1 * g);
            double out_val = sum2 * g + sv_new * (double)attn_score;
            chunk_out[(size_t)t * v_total + head * v_dim + vi] = __float2half((float)out_val);

            for (int kd = 0; kd < k_dim; kd++) {
                double s_new = g * state[kd * v_dim + vi] + sv_new * (double)sK[kd];
                if (s_new > 1e6) s_new = 1e6;
                else if (s_new < -1e6) s_new = -1e6;
                state[kd * v_dim + vi] = s_new;
            }
        }
        __syncthreads();
    }
}

// ============ Chunked RMSNorm Gated (multi-token) ============
__global__ void rms_norm_gated_chunk_kernel(
    const half* __restrict__ chunk_x,    // [N, num_v, v_dim]
    const half* __restrict__ chunk_gate, // [N, num_v, v_dim]
    const float* __restrict__ weight,    // [v_dim]
    half* __restrict__ chunk_out,        // [N, num_v, v_dim]
    int num_v, int v_dim, int n_tokens, float eps
) {
    int head = blockIdx.x % num_v;
    int t    = blockIdx.x / num_v;
    if (t >= n_tokens) return;
    size_t base = (size_t)t * num_v * v_dim + (size_t)head * v_dim;
    const half* x_h = chunk_x + base;
    const half* g_h = chunk_gate + base;
    half* out_h = chunk_out + base;
    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < v_dim; i += blockDim.x) {
        float v = __half2float(x_h[i]);
        sum += v * v;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / v_dim + eps);
    for (int i = threadIdx.x; i < v_dim; i += blockDim.x) {
        float normed = __half2float(x_h[i]) * rms * weight[i];
        float gv = __half2float(g_h[i]);
        float silu_g = gv / (1.0f + expf(-gv));
        out_h[i] = __float2half(normed * silu_g);
    }
}

// ============ RMSNorm Gated (GDN output norm) ============
// out = RMSNorm(x) * SiLU(gate)
// x: [num_v, v_dim], gate: [num_v, v_dim], weight: [v_dim]

__global__ void rms_norm_gated_kernel(
    const half* __restrict__ x,
    const half* __restrict__ gate,
    const float* __restrict__ weight,  // F32 norm weight
    half* __restrict__ out,
    int num_v, int v_dim, float eps
) {
    int head = blockIdx.x;
    if (head >= num_v) return;
    
    const half* x_h = x + head * v_dim;
    const half* g_h = gate + head * v_dim;
    half* out_h = out + head * v_dim;
    
    // Compute RMS
    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < v_dim; i += blockDim.x) {
        float v = __half2float(x_h[i]);
        sum += v * v;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / v_dim + eps);
    
    // Apply: norm * weight * silu(gate)
    for (int i = threadIdx.x; i < v_dim; i += blockDim.x) {
        float normed = __half2float(x_h[i]) * rms * weight[i];
        float gv = __half2float(g_h[i]);
        float silu_g = gv / (1.0f + expf(-gv));
        out_h[i] = __float2half(normed * silu_g);
    }
}
