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

// FP32 input variant (no fp16 cast)
__global__ void conv1d_update_silu_f32(
    float* __restrict__ conv_state,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int dim, int kw
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    float* st = conv_state + d * kw;
    for (int i = 0; i < kw - 1; i++)
        st[i] = st[i + 1];
    st[kw - 1] = input[d];

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
    float* __restrict__ rec_state,   // FP32 state (Volta has 1:32 fp64 throughput)
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

    // 1. L2 norm Q and K — parallel reduction across all threads in the block.
    // Previous version had thread 0 do this serially (~256 FP ops in flight),
    // wasting the other (blockDim.x - 1) threads. With blockDim.x = 128 and
    // k_dim = 128 we get exactly 1 element per thread for the norm pass.
    float q_norm_sq = 0.0f, k_norm_sq = 0.0f;
    for (int i = threadIdx.x; i < k_dim; i += blockDim.x) {
        float qv = q_raw[i];
        float kv = k_raw[i];
        q_norm_sq += qv * qv;
        k_norm_sq += kv * kv;
    }
    // Warp + cross-warp reduce
    for (int off = 16; off > 0; off >>= 1) {
        q_norm_sq += __shfl_xor_sync(0xffffffff, q_norm_sq, off);
        k_norm_sq += __shfl_xor_sync(0xffffffff, k_norm_sq, off);
    }
    __shared__ float warp_q[32], warp_k[32];
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int n_warps = (blockDim.x + 31) >> 5;
    if (lane == 0) { warp_q[warp_id] = q_norm_sq; warp_k[warp_id] = k_norm_sq; }
    __syncthreads();
    if (warp_id == 0) {
        q_norm_sq = (lane < n_warps) ? warp_q[lane] : 0.0f;
        k_norm_sq = (lane < n_warps) ? warp_k[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            q_norm_sq += __shfl_xor_sync(0xffffffff, q_norm_sq, off);
            k_norm_sq += __shfl_xor_sync(0xffffffff, k_norm_sq, off);
        }
        if (lane == 0) { warp_q[0] = q_norm_sq; warp_k[0] = k_norm_sq; }
    }
    __syncthreads();
    float q_inv = rsqrtf(warp_q[0] + 1e-6f);
    float k_inv = rsqrtf(warp_k[0] + 1e-6f);
    float scale = rsqrtf((float)k_dim);

    // Normalize Q and K, write to shared mem, accumulate attention score in parallel.
    float my_attn = 0.0f;
    for (int i = threadIdx.x; i < k_dim; i += blockDim.x) {
        float q_n = q_raw[i] * q_inv;
        float k_n = k_raw[i] * k_inv;
        sQ[i] = q_n * scale;
        sK[i] = k_n;
        my_attn += q_n * k_n;
    }
    // Reduce attention score across the block
    for (int off = 16; off > 0; off >>= 1)
        my_attn += __shfl_xor_sync(0xffffffff, my_attn, off);
    __shared__ float warp_a[32];
    if (lane == 0) warp_a[warp_id] = my_attn;
    __syncthreads();
    if (warp_id == 0) {
        my_attn = (lane < n_warps) ? warp_a[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            my_attn += __shfl_xor_sync(0xffffffff, my_attn, off);
        if (lane == 0) sQ[k_dim] = my_attn * scale;
    }
    __syncthreads();

    float attn_score = sQ[k_dim];

    // 2. Compute gate and beta in FP32
    float alpha = __half2float(a_proj[head]);
    float dt = dt_bias[head];
    float ssm_a_val = a_log[head];
    float g_log = ssm_a_val * logf(1.0f + expf(alpha + dt));
    float g = expf(fminf(g_log, 50.0f));
    float beta = 1.0f / (1.0f + expf(-__half2float(b_proj[head])));

    float* state = rec_state + head * k_dim * v_dim;

    // 3. State update in FP32
    for (int vi = threadIdx.x; vi < v_dim; vi += blockDim.x) {
        float sum1 = 0.0f, sum2 = 0.0f;
        for (int kd = 0; kd < k_dim; kd++) {
            float s = state[kd * v_dim + vi];
            sum1 += s * sK[kd];
            sum2 += s * sQ[kd];
        }
        float sv_new = beta * (v_raw[vi] - sum1 * g);
        float out_val = sum2 * g + sv_new * attn_score;
        core_out[head * v_dim + vi] = __float2half(out_val);

        for (int kd = 0; kd < k_dim; kd++) {
            float s_new = g * state[kd * v_dim + vi] + sv_new * sK[kd];
            if (s_new > 1e6f) s_new = 1e6f;
            else if (s_new < -1e6f) s_new = -1e6f;
            state[kd * v_dim + vi] = s_new;
        }
    }
}

// FP32 output variant — eliminates the fp64→fp16 cast loss at output.
// alpha/beta projection inputs are also FP32 now (eliminate fp16 cast on a_proj/b_proj).
__global__ void gdn_recurrent_step_f32(
    const float* __restrict__ qkv,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_proj,   // FP32 alpha projection
    const float* __restrict__ b_proj,   // FP32 beta projection
    double* __restrict__ rec_state,
    float* __restrict__ core_out,        // FP32 output
    int num_k, int num_v, int k_dim, int v_dim
) {
    int head = blockIdx.x;
    if (head >= num_v) return;
    int k_head = head % num_k;

    const float* q_raw = qkv + k_head * k_dim;
    const float* k_raw = qkv + num_k * k_dim + k_head * k_dim;
    const float* v_raw = qkv + 2 * num_k * k_dim + head * v_dim;

    extern __shared__ float smem[];
    float* sQ = smem;
    float* sK = smem + k_dim + 1;

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

    double alpha = (double)a_proj[head];
    double dt = (double)dt_bias[head];
    double ssm_a_val = (double)a_log[head];
    double g_log = ssm_a_val * log(1.0 + exp(alpha + dt));
    double g = exp(fmin(g_log, 50.0));
    double beta = 1.0 / (1.0 + exp(-(double)b_proj[head]));

    double* state = rec_state + head * k_dim * v_dim;

    for (int vi = threadIdx.x; vi < v_dim; vi += blockDim.x) {
        double sum1 = 0.0, sum2 = 0.0;
        for (int kd = 0; kd < k_dim; kd++) {
            double s = state[kd * v_dim + vi];
            sum1 += s * (double)sK[kd];
            sum2 += s * (double)sQ[kd];
        }
        double sv_new = beta * ((double)v_raw[vi] - sum1 * g);
        double out_val = sum2 * g + sv_new * (double)attn_score;
        core_out[head * v_dim + vi] = (float)out_val;

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
    float* __restrict__ rec_state,        // FP32 state, updated in-place
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

    // SMEM layout:
    //   sState [k_dim*v_dim]   — hot recurrent state, loaded once per chunk
    //   sQ     [k_dim+1]
    //   sK     [k_dim]
    //   sWQ    [32]            — warp-partial for q_norm (padded to 32 for tree reduce)
    //   sWK    [32]            — warp-partial for k_norm
    //   sWA    [32]            — warp-partial for attn_score
    //
    // The three 32-wide scratch arrays are sized to let warp 0 run the exact
    // same __shfl_xor_sync tree reduction as `gdn_recurrent_step` (per-token).
    // Previously this kernel summed warp-partials serially which produced a
    // different fp32 accumulation order than the per-token kernel, causing
    // chunked-vs-per-token divergence that argmax-flipped tokens in 27B
    // Korean coding generation. Matching the reduction tree makes the two
    // paths bit-exact on this step.
    extern __shared__ float smem[];
    const int state_len = k_dim * v_dim;
    float* sState = smem;
    float* sQ     = sState + state_len;
    float* sK     = sQ + k_dim + 1;
    float* sWQ    = sK + k_dim;
    float* sWK    = sWQ + 32;
    float* sWA    = sWK + 32;

    float* gState = rec_state + head * k_dim * v_dim;
    // Cooperative load of state into SMEM — done once, amortized over n_tokens.
    for (int i = threadIdx.x; i < state_len; i += blockDim.x) {
        sState[i] = gState[i];
    }
    __syncthreads();

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;

    auto warp_sum = [](float v) -> float {
        v += __shfl_xor_sync(0xffffffff, v, 16);
        v += __shfl_xor_sync(0xffffffff, v,  8);
        v += __shfl_xor_sync(0xffffffff, v,  4);
        v += __shfl_xor_sync(0xffffffff, v,  2);
        v += __shfl_xor_sync(0xffffffff, v,  1);
        return v;
    };

    for (int t = 0; t < n_tokens; t++) {
        const float* qkv = chunk_qkv + (size_t)t * qkv_dim;
        const float* q_raw = qkv + k_head * k_dim;
        const float* k_raw = qkv + num_k * k_dim + k_head * k_dim;
        const float* v_raw = qkv + 2 * num_k * k_dim + head * v_dim;
        const half* a_proj = chunk_a + (size_t)t * num_v;
        const half* b_proj = chunk_b + (size_t)t * num_v;

        // L2 norm Q, K + attn_score — warp-parallel. blockDim = min(v_dim,128)
        // which for k_dim=128 gives one thread per k_dim element (or a strided
        // loop when blockDim < k_dim).
        float q_ss_local = 0.0f, k_ss_local = 0.0f;
        for (int i = tid; i < k_dim; i += blockDim.x) {
            float qv = q_raw[i];
            float kv = k_raw[i];
            q_ss_local += qv * qv;
            k_ss_local += kv * kv;
        }
        float q_ss_w = warp_sum(q_ss_local);
        float k_ss_w = warp_sum(k_ss_local);
        // Cross-warp reduce via warp-0 shfl_xor tree — matches
        // gdn_recurrent_step's order exactly so chunked-vs-per-token fp32
        // accumulation is bit-identical. Initialize the full 32 slots so
        // threads outside [0, nwarp) read 0.
        if (lane == 0) { sWQ[warp] = q_ss_w; sWK[warp] = k_ss_w; }
        if (warp == 0 && lane >= nwarp && lane < 32) { sWQ[lane] = 0.0f; sWK[lane] = 0.0f; }
        __syncthreads();
        if (warp == 0) {
            float q_val = sWQ[lane];
            float k_val = sWK[lane];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                q_val += __shfl_xor_sync(0xffffffff, q_val, off);
                k_val += __shfl_xor_sync(0xffffffff, k_val, off);
            }
            if (lane == 0) { sWQ[0] = q_val; sWK[0] = k_val; }
        }
        __syncthreads();
        float q_norm_sq = sWQ[0];
        float k_norm_sq = sWK[0];
        float q_inv = rsqrtf(q_norm_sq + 1e-6f);
        float k_inv = rsqrtf(k_norm_sq + 1e-6f);
        float scale = rsqrtf((float)k_dim);

        // Pass 2: write sQ/sK, accumulate attn_score.
        float attn_partial = 0.0f;
        for (int i = tid; i < k_dim; i += blockDim.x) {
            float q_n = q_raw[i] * q_inv;
            float k_n = k_raw[i] * k_inv;
            sQ[i] = q_n * scale;
            sK[i] = k_n;
            attn_partial += q_n * k_n;
        }
        float attn_w = warp_sum(attn_partial);
        // Same tree reduce as above for attn_score.
        if (lane == 0) sWA[warp] = attn_w;
        if (warp == 0 && lane >= nwarp && lane < 32) sWA[lane] = 0.0f;
        __syncthreads();
        if (warp == 0) {
            float a_val = sWA[lane];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                a_val += __shfl_xor_sync(0xffffffff, a_val, off);
            }
            if (lane == 0) sWA[0] = a_val * scale;
        }
        __syncthreads();
        float attn_score = sWA[0];

        // Gate and beta in FP32 (Volta has 1:32 fp64 throughput)
        float alpha = __half2float(a_proj[head]);
        float dt = dt_bias[head];
        float ssm_a_val = a_log[head];
        float g_log = ssm_a_val * logf(1.0f + expf(alpha + dt));
        float g = expf(fminf(g_log, 50.0f));
        float beta = 1.0f / (1.0f + expf(-__half2float(b_proj[head])));

        // Per-v state update + output. State lives in SMEM for the duration
        // of this chunk — two passes over kd are cache-local instead of the
        // global-memory re-reads the original kernel was paying.
        for (int vi = threadIdx.x; vi < v_dim; vi += blockDim.x) {
            float sum1 = 0.0f, sum2 = 0.0f;
            for (int kd = 0; kd < k_dim; kd++) {
                float s = sState[kd * v_dim + vi];
                sum1 += s * sK[kd];
                sum2 += s * sQ[kd];
            }
            float sv_new = beta * (v_raw[vi] - sum1 * g);
            float out_val = sum2 * g + sv_new * attn_score;
            chunk_out[(size_t)t * v_total + head * v_dim + vi] = __float2half(out_val);

            for (int kd = 0; kd < k_dim; kd++) {
                float s_new = g * sState[kd * v_dim + vi] + sv_new * sK[kd];
                if (s_new > 1e6f) s_new = 1e6f;
                else if (s_new < -1e6f) s_new = -1e6f;
                sState[kd * v_dim + vi] = s_new;
            }
        }
        __syncthreads();
    }

    // Flush SMEM state back to global so the next chunk picks up where we
    // left off.
    for (int i = threadIdx.x; i < state_len; i += blockDim.x) {
        gState[i] = sState[i];
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

// FP32 input/output variant
__global__ void rms_norm_gated_kernel_f32(
    const float* __restrict__ x,
    const float* __restrict__ gate,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int num_v, int v_dim, float eps
) {
    int head = blockIdx.x;
    if (head >= num_v) return;

    const float* x_h = x + head * v_dim;
    const float* g_h = gate + head * v_dim;
    float* out_h = out + head * v_dim;

    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < v_dim; i += blockDim.x) {
        float v = x_h[i];
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
        float normed = x_h[i] * rms * weight[i];
        float gv = g_h[i];
        float silu_g = gv / (1.0f + expf(-gv));
        out_h[i] = normed * silu_g;
    }
}

// ============ DDTree: conv1d tree-mode kernel ============
// Process n_tokens tree nodes in a single launch. For each node t, walk the
// parent chain (kw-1) times via parent_ids[] to reconstruct the K-wide conv
// window. parent_ids[t] == -1 means "parent is the pre-block state"; walking
// further through negative indices decays into the pre-block `conv_state`
// buffer (slots [0 .. kw-2]).
//
// Inputs / outputs:
//   conv_state: [dim, kw] FP32 — caller supplies the conv state BEFORE the
//               block began. The kernel only reads slots [0 .. kw-2] as the
//               pre-block window; slot kw-1 is ignored (each tree node
//               provides its own "self" value via `inputs`).
//   inputs:     [n_tokens, dim] FP32 — per-node new conv input values (the
//               same value that non-tree conv1d would place into slot kw-1).
//   weight:     [dim, kw] FP32 — same layout as the per-step kernel.
//   parent_ids: [n_tokens] int32 — t's parent node index in the DFS-flattened
//               tree, or -1 if t's parent is the pre-block state.
//   output:     [n_tokens, dim] FP32 — SiLU(conv(window_t)) for each node.
//
// The kernel does NOT update conv_state. After tree verify, the host picks
// the accept-committed suffix and slides the final node's window back into
// conv_state using the standard per-step path.
template <int KW_MAX = 8>
__global__ void conv1d_update_silu_tree(
    const float* __restrict__ conv_state,   // [dim, kw], pre-block window in [0..kw-2]
    const half*  __restrict__ inputs,       // [n_tokens, dim] half
    const float* __restrict__ weight,       // [dim, kw]
    const int*   __restrict__ parent_ids,   // [n_tokens], -1 = pre-block parent
    float*       __restrict__ output,       // [n_tokens, dim]
    int dim, int kw, int n_tokens
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    const float* cs = conv_state + d * kw;
    const float* wr = weight + d * kw;

    float w[KW_MAX];
    #pragma unroll
    for (int k = 0; k < KW_MAX; k++) w[k] = (k < kw) ? wr[k] : 0.0f;

    for (int t = 0; t < n_tokens; t++) {
        int anc[KW_MAX];
        #pragma unroll
        for (int k = 0; k < KW_MAX; k++) anc[k] = 0;
        anc[kw - 1] = t;
        for (int k = kw - 2; k >= 0; k--) {
            int prev = anc[k + 1];
            anc[k] = (prev >= 0) ? parent_ids[prev] : (prev - 1);
        }

        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < KW_MAX; k++) {
            if (k >= kw) break;
            int a = anc[k];
            float x;
            if (a >= 0) {
                x = __half2float(inputs[(size_t)a * dim + d]);
            } else {
                int ps_idx = (kw - 1) + a;       // a in [-1..-(kw-1)] → ps_idx in [kw-2..0]
                x = (ps_idx >= 0) ? cs[ps_idx] : 0.0f;
            }
            sum += w[k] * x;
        }

        float silu = sum / (1.0f + expf(-sum));
        output[(size_t)t * dim + d] = silu;
    }
}

// Explicit instantiation for Qwen3.5 (kw=4). KW_MAX=8 covers 3/4/5.
template __global__ void conv1d_update_silu_tree<8>(
    const float*, const half*, const float*, const int*, float*, int, int, int);

// ============ DDTree: GDN recurrent step tree kernel ============
// Process n_tokens tree nodes over the GDN recurrence in a single launch.
// Each node t reloads its starting state from parent_ids[t]'s saved slot in
// persist_inter (or rec_state_init for root), runs the same state update as
// gdn_recurrent_step, writes the post-token state into persist_inter[t], and
// outputs core_out_tree[t].
//
// Layout:
//   qkv_tree       [n_tokens, qkv_stride]        qkv_stride = 2*num_k*k_dim + num_v*v_dim
//   a_proj_tree    [n_tokens, num_v] half
//   b_proj_tree    [n_tokens, num_v] half
//   rec_state_init [num_v, k_dim, v_dim] f32     PRE-tree GDN state (read-only)
//   persist_inter  [n_tokens, num_v, k_dim, v_dim] f32   per-token state save
//   core_out_tree  [n_tokens, num_v, v_dim] half
//   parent_ids     [n_tokens] int32               -1 = root (parent is rec_state_init)
//
// Grid:    num_v blocks × 128 threads   (same as gdn_recurrent_step)
// SMEM:    (k_dim + 1 + k_dim) floats  (same as gdn_recurrent_step)
//
// Host responsibility: after the tree run, pick the committed suffix and
// copy its last node's state from persist_inter back into rec_state_init.
__global__ void gdn_recurrent_step_tree(
    const float* __restrict__ qkv_tree,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const half*  __restrict__ a_proj_tree,
    const half*  __restrict__ b_proj_tree,
    const float* __restrict__ rec_state_init,
    float*       __restrict__ persist_inter,
    const int*   __restrict__ parent_ids,
    half*        __restrict__ core_out_tree,
    int num_k, int num_v, int k_dim, int v_dim, int n_tokens
) {
    int head = blockIdx.x;
    if (head >= num_v) return;
    int k_head = head % num_k;

    const int qkv_stride = 2 * num_k * k_dim + num_v * v_dim;
    const size_t state_head_size = (size_t)k_dim * v_dim;

    extern __shared__ float smem[];
    float* sQ = smem;
    float* sK = smem + k_dim + 1;

    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int n_warps = (blockDim.x + 31) >> 5;
    __shared__ float warp_q[32], warp_k[32], warp_a[32];

    for (int t = 0; t < n_tokens; t++) {
        // 1. Reload state_t from parent (or root) into persist_inter[t].
        float* state_t = persist_inter
            + ((size_t)t * num_v + head) * state_head_size;
        int parent_t = parent_ids[t];
        const float* src = nullptr;
        if (parent_t < 0) {
            src = rec_state_init + (size_t)head * state_head_size;
        } else {
            src = persist_inter
                + ((size_t)parent_t * num_v + head) * state_head_size;
        }
        for (int i = threadIdx.x; i < (int)state_head_size; i += blockDim.x) {
            state_t[i] = src[i];
        }
        __syncthreads();

        // 2. Per-token inputs.
        const float* qkv_t = qkv_tree + (size_t)t * qkv_stride;
        const float* q_raw = qkv_t + k_head * k_dim;
        const float* k_raw = qkv_t + num_k * k_dim + k_head * k_dim;
        const float* v_raw = qkv_t + 2 * num_k * k_dim + head * v_dim;

        // 3. L2 norm Q and K (parallel reduction) — same as gdn_recurrent_step.
        float q_norm_sq = 0.0f, k_norm_sq = 0.0f;
        for (int i = threadIdx.x; i < k_dim; i += blockDim.x) {
            float qv = q_raw[i];
            float kv = k_raw[i];
            q_norm_sq += qv * qv;
            k_norm_sq += kv * kv;
        }
        for (int off = 16; off > 0; off >>= 1) {
            q_norm_sq += __shfl_xor_sync(0xffffffff, q_norm_sq, off);
            k_norm_sq += __shfl_xor_sync(0xffffffff, k_norm_sq, off);
        }
        if (lane == 0) { warp_q[warp_id] = q_norm_sq; warp_k[warp_id] = k_norm_sq; }
        __syncthreads();
        if (warp_id == 0) {
            q_norm_sq = (lane < n_warps) ? warp_q[lane] : 0.0f;
            k_norm_sq = (lane < n_warps) ? warp_k[lane] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) {
                q_norm_sq += __shfl_xor_sync(0xffffffff, q_norm_sq, off);
                k_norm_sq += __shfl_xor_sync(0xffffffff, k_norm_sq, off);
            }
            if (lane == 0) { warp_q[0] = q_norm_sq; warp_k[0] = k_norm_sq; }
        }
        __syncthreads();
        float q_inv = rsqrtf(warp_q[0] + 1e-6f);
        float k_inv = rsqrtf(warp_k[0] + 1e-6f);
        float scale = rsqrtf((float)k_dim);

        float my_attn = 0.0f;
        for (int i = threadIdx.x; i < k_dim; i += blockDim.x) {
            float q_n = q_raw[i] * q_inv;
            float k_n = k_raw[i] * k_inv;
            sQ[i] = q_n * scale;
            sK[i] = k_n;
            my_attn += q_n * k_n;
        }
        for (int off = 16; off > 0; off >>= 1)
            my_attn += __shfl_xor_sync(0xffffffff, my_attn, off);
        if (lane == 0) warp_a[warp_id] = my_attn;
        __syncthreads();
        if (warp_id == 0) {
            my_attn = (lane < n_warps) ? warp_a[lane] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                my_attn += __shfl_xor_sync(0xffffffff, my_attn, off);
            if (lane == 0) sQ[k_dim] = my_attn * scale;
        }
        __syncthreads();
        float attn_score = sQ[k_dim];

        // 4. Gate and beta (per-token a_proj/b_proj).
        float alpha = __half2float(a_proj_tree[(size_t)t * num_v + head]);
        float dt = dt_bias[head];
        float ssm_a_val = a_log[head];
        float g_log = ssm_a_val * logf(1.0f + expf(alpha + dt));
        float g = expf(fminf(g_log, 50.0f));
        float beta = 1.0f / (1.0f + expf(-__half2float(b_proj_tree[(size_t)t * num_v + head])));

        // 5. State update on state_t (in-place) + output core_out_tree[t].
        half* core_out_t = core_out_tree + ((size_t)t * num_v + head) * v_dim;
        for (int vi = threadIdx.x; vi < v_dim; vi += blockDim.x) {
            float sum1 = 0.0f, sum2 = 0.0f;
            for (int kd = 0; kd < k_dim; kd++) {
                float s = state_t[kd * v_dim + vi];
                sum1 += s * sK[kd];
                sum2 += s * sQ[kd];
            }
            float sv_new = beta * (v_raw[vi] - sum1 * g);
            float out_val = sum2 * g + sv_new * attn_score;
            core_out_t[vi] = __float2half(out_val);

            for (int kd = 0; kd < k_dim; kd++) {
                float s_new = g * state_t[kd * v_dim + vi] + sv_new * sK[kd];
                if (s_new > 1e6f) s_new = 1e6f;
                else if (s_new < -1e6f) s_new = -1e6f;
                state_t[kd * v_dim + vi] = s_new;
            }
        }
        __syncthreads();  // next iter reads state_t from global
    }
}
