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
    half* __restrict__ conv_state,   // [dim, kw]
    const half* __restrict__ input,  // [dim]
    const float* __restrict__ weight, // [kw, dim] F32
    half* __restrict__ output,       // [dim]
    int dim, int kw
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    
    // Shift state left: state[d][0..kw-2] = state[d][1..kw-1]
    half* st = conv_state + d * kw;
    for (int i = 0; i < kw - 1; i++)
        st[i] = st[i + 1];
    // Insert new input at end
    st[kw - 1] = input[d];
    
    // Conv1d dot product
    float sum = 0.0f;
    for (int i = 0; i < kw; i++)
        sum += weight[d * kw + i] * __half2float(st[i]);
    
    // SiLU
    float silu = sum / (1.0f + expf(-sum));
    output[d] = __float2half(silu);
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
    const half* __restrict__ qkv,      // [2*num_k*k_dim + num_v*v_dim] post-conv+silu
    const float* __restrict__ a_log,   // [num_v] A_log values
    const float* __restrict__ dt_bias, // [num_v] F32
    const half* __restrict__ a_proj,   // [num_v] alpha projection
    const half* __restrict__ b_proj,   // [num_v] beta projection
    float* __restrict__ rec_state,     // [num_v, k_dim, v_dim] updated inplace
    half* __restrict__ core_out,       // [num_v, v_dim] output
    int num_k, int num_v, int k_dim, int v_dim
) {
    int head = blockIdx.x;  // v-head index
    if (head >= num_v) return;

    // Qwen3.5 uses modulo mapping (repeat_type=1): head % num_k
    int k_head = head % num_k;

    // Get Q, K, V pointers
    const half* q_raw = qkv + k_head * k_dim;
    const half* k_raw = qkv + num_k * k_dim + k_head * k_dim;
    const half* v_raw = qkv + 2 * num_k * k_dim + head * v_dim;

    // L2 normalize Q and K
    float q_norm_sq = 0.0f, k_norm_sq = 0.0f;
    for (int i = 0; i < k_dim; i++) {
        float qv = __half2float(q_raw[i]);
        float kv = __half2float(k_raw[i]);
        q_norm_sq += qv * qv;
        k_norm_sq += kv * kv;
    }
    float q_scale = rsqrtf(q_norm_sq + 1e-6f) * rsqrtf((float)k_dim);  // L2 norm + 1/sqrt(d)
    float k_inv_norm = rsqrtf(k_norm_sq + 1e-6f);

    // Compute gate: g_log = ssm_a * softplus(alpha + dt_bias)
    // ssm_a is already negative (stores -exp(A_log) or similar)
    float alpha = __half2float(a_proj[head]);
    float dt = dt_bias[head];
    float ssm_a_val = a_log[head];  // already negative
    float g_log = ssm_a_val * logf(1.0f + expf(alpha + dt));  // negative
    float g = expf(g_log);  // 0 < g < 1, decay factor

    // Beta = sigmoid(b_proj)
    float beta = 1.0f / (1.0f + expf(-__half2float(b_proj[head])));

    float* state = rec_state + head * k_dim * v_dim;

    // For each v dimension
    for (int vi = threadIdx.x; vi < v_dim; vi += blockDim.x) {
        // Compute S^T k for this v-dimension
        // S^T k = sum_kd(state[kd, vi] * k_normed[kd])
        float stk = 0.0f;
        for (int kd = 0; kd < k_dim; kd++) {
            stk += state[kd * v_dim + vi] * __half2float(k_raw[kd]) * k_inv_norm;
        }

        // v value
        float v_val = __half2float(v_raw[vi]);

        // delta = beta * (v - g * S^T k)
        float delta = beta * (v_val - g * stk);

        // Update state and compute output: o = q^T * S_new
        float out_val = 0.0f;
        for (int kd = 0; kd < k_dim; kd++) {
            float k_val = __half2float(k_raw[kd]) * k_inv_norm;
            float q_val = __half2float(q_raw[kd]) * q_scale;
            float s_old = state[kd * v_dim + vi];
            float s_new = g * s_old + k_val * delta;
            s_new = fminf(fmaxf(s_new, -1e6f), 1e6f);
            state[kd * v_dim + vi] = s_new;
            out_val += q_val * s_new;
        }
        core_out[head * v_dim + vi] = __float2half(out_val);
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
