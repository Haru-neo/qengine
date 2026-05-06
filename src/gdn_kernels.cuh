#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
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

// V2: holds state row in registers, single-read + single-write, k_dim/v_dim
// templated so the inner loops unroll. 9B and 27B both use 128/128 so a single
// specialization covers production. Saves one full state read pass per token
// per layer (≈ half the HBM traffic of v1).
//
// Launch: gdn_recurrent_step_v2<128,128><<<num_v, V_DIM, smem_bytes>>>(...)
// smem layout matches v1: [sQ:K_DIM+1][sK:K_DIM] = (2K+1) floats.
template <int K_DIM, int V_DIM>
__global__ void gdn_recurrent_step_v2(
    const float* __restrict__ qkv,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const half*  __restrict__ a_proj,
    const half*  __restrict__ b_proj,
    float* __restrict__ rec_state,
    half*  __restrict__ core_out,
    int num_k, int num_v
) {
    int head = blockIdx.x;
    if (head >= num_v) return;
    int k_head = head % num_k;
    int tid = threadIdx.x;

    const float* q_raw = qkv + k_head * K_DIM;
    const float* k_raw = qkv + num_k * K_DIM + k_head * K_DIM;
    const float* v_raw = qkv + 2 * num_k * K_DIM + head * V_DIM;

    extern __shared__ float smem[];
    float* sQ = smem;
    float* sK = smem + K_DIM + 1;

    float q_norm_sq = 0.0f, k_norm_sq = 0.0f;
    if (tid < K_DIM) {
        float qv = q_raw[tid];
        float kv = k_raw[tid];
        q_norm_sq = qv * qv;
        k_norm_sq = kv * kv;
    }
    for (int off = 16; off > 0; off >>= 1) {
        q_norm_sq += __shfl_xor_sync(0xffffffff, q_norm_sq, off);
        k_norm_sq += __shfl_xor_sync(0xffffffff, k_norm_sq, off);
    }
    __shared__ float warp_q[8], warp_k[8];
    int warp_id = tid >> 5;
    int lane    = tid & 31;
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
    float scale = rsqrtf((float)K_DIM);

    float my_attn = 0.0f;
    if (tid < K_DIM) {
        float q_n = q_raw[tid] * q_inv;
        float k_n = k_raw[tid] * k_inv;
        sQ[tid] = q_n * scale;
        sK[tid] = k_n;
        my_attn = q_n * k_n;
    }
    for (int off = 16; off > 0; off >>= 1)
        my_attn += __shfl_xor_sync(0xffffffff, my_attn, off);
    __shared__ float warp_a[8];
    if (lane == 0) warp_a[warp_id] = my_attn;
    __syncthreads();
    if (warp_id == 0) {
        my_attn = (lane < n_warps) ? warp_a[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            my_attn += __shfl_xor_sync(0xffffffff, my_attn, off);
        if (lane == 0) sQ[K_DIM] = my_attn * scale;
    }
    __syncthreads();
    float attn_score = sQ[K_DIM];

    float alpha = __half2float(a_proj[head]);
    float dt = dt_bias[head];
    float ssm_a_val = a_log[head];
    float g_log = ssm_a_val * logf(1.0f + expf(alpha + dt));
    float g = expf(fminf(g_log, 50.0f));
    float beta = 1.0f / (1.0f + expf(-__half2float(b_proj[head])));

    if (tid >= V_DIM) return;
    int vi = tid;
    float* state = rec_state + head * K_DIM * V_DIM;

    float s_row[K_DIM];
    float sum1 = 0.0f, sum2 = 0.0f;
    #pragma unroll
    for (int kd = 0; kd < K_DIM; kd++) {
        float s = state[kd * V_DIM + vi];
        s_row[kd] = s;
        sum1 += s * sK[kd];
        sum2 += s * sQ[kd];
    }
    float sv_new = beta * (v_raw[vi] - sum1 * g);
    float out_val = sum2 * g + sv_new * attn_score;
    core_out[head * V_DIM + vi] = __float2half(out_val);

    #pragma unroll
    for (int kd = 0; kd < K_DIM; kd++) {
        float s_new = g * s_row[kd] + sv_new * sK[kd];
        s_new = fmaxf(-1e6f, fminf(1e6f, s_new));
        state[kd * V_DIM + vi] = s_new;
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
// The kernel does NOT update conv_state. After tree verify, the host slides
// the accepted chain's input values into conv_state via
// `conv_state_commit_chain_kernel`.
//
// conv_state layout (non-tree `conv1d_update_silu` after processing token t):
//   slot 0         = x_{t-(kw-1)}   (oldest)
//   slot kw-1      = x_t            (newest, most recent token processed)
// A new token x_{t+1} then produces window [slot1, slot2, ..., slot_{kw-1},
// x_{t+1}]. So the tree root's ancestors in pre-block state live in slots
// [1 .. kw-1] (not [0 .. kw-2] as an earlier comment claimed).
template <int KW_MAX = 8>
__global__ void conv1d_update_silu_tree(
    const float* __restrict__ conv_state,   // [dim, kw], see layout note above
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
                // a in [-1..-(kw-1)] → ps_idx in [kw-1 .. 1]. Slot 0 is the
                // oldest token; slot kw-1 is the most recent pre-tree input.
                // a=-1 picks kw-1 (most recent), a=-2 picks kw-2, etc.
                int ps_idx = kw + a;
                x = (ps_idx >= 0 && ps_idx < kw) ? cs[ps_idx] : 0.0f;
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

// ============ DDTree: commit accepted chain into conv_state ============
// For chain-shaped accepted prefixes: slide conv_state left by `accept_len`
// slots, then fill the rightmost `accept_len` slots from the corresponding
// nodes' saved post-qkv inputs in `qkv_tree`. Node order in qkv_tree is DFS
// along the accepted chain (node indices in parent order).
//
//   conv_state  [dim, kw] FP32    — in-place updated
//   qkv_tree    [n, dim]  half    — post-qkv-projection values for each tree
//                                   node. We pick slots [node_ids[0..L-1]].
//   node_ids    [L] int32         — tree-node indices of the accepted prefix,
//                                   in append order (root-first, on host).
//   accept_len  L, 1 <= L <= kw-1
//
// Note: if L >= kw-1, only the last kw-1 nodes matter (earlier ones get
//       slid out). Caller should trim node_ids accordingly.
__global__ void conv_state_commit_chain_kernel(
    float*       __restrict__ conv_state,   // [dim, kw]
    const half*  __restrict__ qkv_tree,     // [n_nodes, dim]
    const int*   __restrict__ node_ids,     // [accept_len]
    int dim, int kw, int accept_len
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float* st = conv_state + d * kw;
    // Slide left by accept_len.
    for (int k = 0; k < kw - accept_len; k++) st[k] = st[k + accept_len];
    // Fill rightmost accept_len slots from qkv_tree[node_ids[i]].
    for (int i = 0; i < accept_len; i++) {
        int node = node_ids[i];
        st[(kw - accept_len) + i] =
            __half2float(qkv_tree[(size_t)node * dim + d]);
    }
}

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

// Layer-fuse PoC: combine gdn_recurrent_step + rms_norm_gated_kernel into one
// kernel. Both have the same grid layout (1 block per v_head) so we can fuse
// at the block level. The recurrent output (1 fp16 per (head, vi)) is held in
// a thread-local register and passed to the rmsg phase without touching HBM.
// Saves: one kernel launch, one core_out write+read round-trip.
template <int K_DIM, int V_DIM>
__global__ void gdn_fused_recur_rmsg(
    const float* __restrict__ qkv,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const half*  __restrict__ a_proj,
    const half*  __restrict__ b_proj,
    float* __restrict__ rec_state,
    const half*  __restrict__ z_out,        // gate input for rmsg
    const float* __restrict__ ssm_norm_w,   // RMS weight
    half*  __restrict__ normed_out,         // final output
    int num_k, int num_v, float eps
) {
    int head = blockIdx.x;
    if (head >= num_v) return;
    int k_head = head % num_k;
    int tid = threadIdx.x;

    const float* q_raw = qkv + k_head * K_DIM;
    const float* k_raw = qkv + num_k * K_DIM + k_head * K_DIM;
    const float* v_raw = qkv + 2 * num_k * K_DIM + head * V_DIM;

    extern __shared__ float smem[];
    float* sQ = smem;              // [K_DIM + 1]   (last = attn_score)
    float* sK = smem + K_DIM + 1;  // [K_DIM]
    float* sR = sK + K_DIM;        // [V_DIM]       (rmsg reduction scratch)

    // ---- Phase 1: Q/K L2 norm + attention score ----
    float q_norm_sq = 0.0f, k_norm_sq = 0.0f;
    for (int i = tid; i < K_DIM; i += blockDim.x) {
        float qv = q_raw[i];
        float kv = k_raw[i];
        q_norm_sq += qv * qv;
        k_norm_sq += kv * kv;
    }
    for (int off = 16; off > 0; off >>= 1) {
        q_norm_sq += __shfl_xor_sync(0xffffffff, q_norm_sq, off);
        k_norm_sq += __shfl_xor_sync(0xffffffff, k_norm_sq, off);
    }
    __shared__ float warp_q[8], warp_k[8];
    int warp_id = tid >> 5;
    int lane    = tid & 31;
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
    float scale = rsqrtf((float)K_DIM);

    float my_attn = 0.0f;
    for (int i = tid; i < K_DIM; i += blockDim.x) {
        float q_n = q_raw[i] * q_inv;
        float k_n = k_raw[i] * k_inv;
        sQ[i] = q_n * scale;
        sK[i] = k_n;
        my_attn += q_n * k_n;
    }
    for (int off = 16; off > 0; off >>= 1)
        my_attn += __shfl_xor_sync(0xffffffff, my_attn, off);
    __shared__ float warp_a[8];
    if (lane == 0) warp_a[warp_id] = my_attn;
    __syncthreads();
    if (warp_id == 0) {
        my_attn = (lane < n_warps) ? warp_a[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            my_attn += __shfl_xor_sync(0xffffffff, my_attn, off);
        if (lane == 0) sQ[K_DIM] = my_attn * scale;
    }
    __syncthreads();
    float attn_score = sQ[K_DIM];

    // ---- Phase 2: gate / beta / state update ----
    float alpha = __half2float(a_proj[head]);
    float dt = dt_bias[head];
    float ssm_a_val = a_log[head];
    float g_log = ssm_a_val * logf(1.0f + expf(alpha + dt));
    float g = expf(fminf(g_log, 50.0f));
    float beta = 1.0f / (1.0f + expf(-__half2float(b_proj[head])));

    float* state = rec_state + head * K_DIM * V_DIM;
    int vi = tid;
    float out_val = 0.0f;  // recurrent output for this vi (register-held)

    if (vi < V_DIM) {
        float sum1 = 0.0f, sum2 = 0.0f;
        for (int kd = 0; kd < K_DIM; kd++) {
            float s = state[kd * V_DIM + vi];
            sum1 += s * sK[kd];
            sum2 += s * sQ[kd];
        }
        float sv_new = beta * (v_raw[vi] - sum1 * g);
        out_val = sum2 * g + sv_new * attn_score;

        // State update (re-read state — cache hit expected)
        for (int kd = 0; kd < K_DIM; kd++) {
            float s_new = g * state[kd * V_DIM + vi] + sv_new * sK[kd];
            s_new = fmaxf(-1e6f, fminf(1e6f, s_new));
            state[kd * V_DIM + vi] = s_new;
        }
    }
    // No __syncthreads() needed yet: out_val is register-local per thread.

    // ---- Phase 3: RMS-Gated norm ----
    sR[tid] = (vi < V_DIM) ? (out_val * out_val) : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sR[tid] += sR[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(sR[0] / (float)V_DIM + eps);

    if (vi < V_DIM) {
        float normed = out_val * rms * ssm_norm_w[vi];
        float gv = __half2float(z_out[head * V_DIM + vi]);
        float silu_g = gv / (1.0f + expf(-gv));
        normed_out[head * V_DIM + vi] = __float2half(normed * silu_g);
    }
}

// Layer-wide cooperative kernel: conv1d + recurrent + rmsg in one launch.
// Three phases separated by grid.sync():
//   Phase A: conv1d_update_silu — all (block, thread) cooperatively process
//            qkv_dim channels (block_id * blockDim + tid -> channel).
//   Phase B: recurrent step — first `num_v` blocks act as 1 block per v_head.
//   Phase C: rms-gated norm — same block layout as B, no inter-phase sync
//            needed because each thread keeps `out_val` in a register.
//
// Launch: cooperative, grid = max(qkv_dim/blockDim, num_v), block = 256,
//         smem = (2*K_DIM + 1 + V_DIM) * sizeof(float).
// K_DIM = V_DIM = 128 covers Qwopus 9B (qkv_dim=8192) and 27B (qkv_dim=10240).
//
// Note: this kernel does NOT do the output projection. Caller still needs to
// run quant_gemv on `normed_out` -> `proj_out` after this kernel returns.
template <int K_DIM, int V_DIM>
__global__ void gdn_layer_fused_recur_rmsg(
    // Phase A: conv1d
    float* __restrict__ conv_state,
    const half*  __restrict__ qkv_in,
    const float* __restrict__ conv_w,
    float*       __restrict__ conv_out,
    int qkv_dim, int kw,
    // Phase B: recurrent
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const half*  __restrict__ a_proj,
    const half*  __restrict__ b_proj,
    float* __restrict__ rec_state,
    // Phase C: rmsg
    const half*  __restrict__ z_out,
    const float* __restrict__ ssm_norm_w,
    half*  __restrict__ normed_out,
    int num_k, int num_v, float eps
) {
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();
    int block_id = blockIdx.x;
    int tid = threadIdx.x;

    // ---- Phase A: conv1d (channel-parallel) ----
    int channel = block_id * blockDim.x + tid;
    if (channel < qkv_dim) {
        float* st = conv_state + channel * kw;
        for (int i = 0; i < kw - 1; i++) st[i] = st[i + 1];
        st[kw - 1] = __half2float(qkv_in[channel]);
        float sum = 0.0f;
        for (int i = 0; i < kw; i++) sum += conv_w[channel * kw + i] * st[i];
        float silu = sum / (1.0f + expf(-sum));
        conv_out[channel] = silu;
    }
    grid.sync();

    // ---- Phase B + C: recurrent + rmsg (block per v_head) ----
    int head = block_id;
    if (head >= num_v) return;
    int k_head = head % num_k;

    const float* q_raw = conv_out + k_head * K_DIM;
    const float* k_raw = conv_out + num_k * K_DIM + k_head * K_DIM;
    const float* v_raw = conv_out + 2 * num_k * K_DIM + head * V_DIM;

    extern __shared__ float smem[];
    float* sQ = smem;
    float* sK = smem + K_DIM + 1;
    float* sR = sK + K_DIM;  // rmsg reduction scratch

    // Q/K L2 norm + score (only first 128 threads do useful work, others write 0)
    float q_norm_sq = 0.0f, k_norm_sq = 0.0f;
    if (tid < K_DIM) {
        float qv = q_raw[tid];
        float kv = k_raw[tid];
        q_norm_sq = qv * qv;
        k_norm_sq = kv * kv;
    }
    for (int off = 16; off > 0; off >>= 1) {
        q_norm_sq += __shfl_xor_sync(0xffffffff, q_norm_sq, off);
        k_norm_sq += __shfl_xor_sync(0xffffffff, k_norm_sq, off);
    }
    __shared__ float warp_q[8], warp_k[8];
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int n_warps_active = (K_DIM + 31) >> 5;  // 4 warps for K_DIM=128
    if (lane == 0 && tid < K_DIM) { warp_q[warp_id] = q_norm_sq; warp_k[warp_id] = k_norm_sq; }
    __syncthreads();
    if (warp_id == 0) {
        q_norm_sq = (lane < n_warps_active) ? warp_q[lane] : 0.0f;
        k_norm_sq = (lane < n_warps_active) ? warp_k[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            q_norm_sq += __shfl_xor_sync(0xffffffff, q_norm_sq, off);
            k_norm_sq += __shfl_xor_sync(0xffffffff, k_norm_sq, off);
        }
        if (lane == 0) { warp_q[0] = q_norm_sq; warp_k[0] = k_norm_sq; }
    }
    __syncthreads();
    float q_inv = rsqrtf(warp_q[0] + 1e-6f);
    float k_inv = rsqrtf(warp_k[0] + 1e-6f);
    float scale = rsqrtf((float)K_DIM);

    float my_attn = 0.0f;
    if (tid < K_DIM) {
        float q_n = q_raw[tid] * q_inv;
        float k_n = k_raw[tid] * k_inv;
        sQ[tid] = q_n * scale;
        sK[tid] = k_n;
        my_attn = q_n * k_n;
    }
    for (int off = 16; off > 0; off >>= 1)
        my_attn += __shfl_xor_sync(0xffffffff, my_attn, off);
    __shared__ float warp_a[8];
    if (lane == 0 && tid < K_DIM) warp_a[warp_id] = my_attn;
    __syncthreads();
    if (warp_id == 0) {
        my_attn = (lane < n_warps_active) ? warp_a[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            my_attn += __shfl_xor_sync(0xffffffff, my_attn, off);
        if (lane == 0) sQ[K_DIM] = my_attn * scale;
    }
    __syncthreads();
    float attn_score = sQ[K_DIM];

    // Gate / beta / state update
    float alpha = __half2float(a_proj[head]);
    float dt = dt_bias[head];
    float ssm_a_val = a_log[head];
    float g_log = ssm_a_val * logf(1.0f + expf(alpha + dt));
    float g = expf(fminf(g_log, 50.0f));
    float beta = 1.0f / (1.0f + expf(-__half2float(b_proj[head])));

    float* state = rec_state + head * K_DIM * V_DIM;
    float out_val = 0.0f;
    if (tid < V_DIM) {
        int vi = tid;
        float sum1 = 0.0f, sum2 = 0.0f;
        for (int kd = 0; kd < K_DIM; kd++) {
            float s = state[kd * V_DIM + vi];
            sum1 += s * sK[kd];
            sum2 += s * sQ[kd];
        }
        float sv_new = beta * (v_raw[vi] - sum1 * g);
        out_val = sum2 * g + sv_new * attn_score;

        for (int kd = 0; kd < K_DIM; kd++) {
            float s_new = g * state[kd * V_DIM + vi] + sv_new * sK[kd];
            s_new = fmaxf(-1e6f, fminf(1e6f, s_new));
            state[kd * V_DIM + vi] = s_new;
        }
    }

    // Phase C: rmsg
    sR[tid] = (tid < V_DIM) ? (out_val * out_val) : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sR[tid] += sR[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(sR[0] / (float)V_DIM + eps);
    if (tid < V_DIM) {
        int vi = tid;
        float normed = out_val * rms * ssm_norm_w[vi];
        float gv = __half2float(z_out[head * V_DIM + vi]);
        float silu_g = gv / (1.0f + expf(-gv));
        normed_out[head * V_DIM + vi] = __float2half(normed * silu_g);
    }
}

// Host-side launch wrapper. GDN_V2=1 env switches to the register-hold v2
// kernel for k_dim=v_dim=128 (covers Qwopus 9B + 27B). Otherwise falls back to
// the v1 kernel. Centralizing the env check in one helper means the seven
// per-token / per-chunk launch sites in model.cuh stay identical.
static inline void launch_gdn_recurrent_step(
    int grid_num_v, int v_dim_blockdim, int gdn_smem, cudaStream_t stream,
    const float* conv_out,
    const float* a_log,
    const float* dt_bias,
    const half*  a_proj,
    const half*  b_proj,
    float* rec_state,
    half*  core_out,
    int num_k, int num_v, int k_dim, int v_dim)
{
    static const bool use_v2 = getenv("GDN_V2") != nullptr;
    if (use_v2 && k_dim == 128 && v_dim == 128) {
        gdn_recurrent_step_v2<128, 128><<<grid_num_v, 128, gdn_smem, stream>>>(
            conv_out, a_log, dt_bias, a_proj, b_proj,
            rec_state, core_out, num_k, num_v);
    } else {
        gdn_recurrent_step<<<grid_num_v, v_dim_blockdim, gdn_smem, stream>>>(
            conv_out, a_log, dt_bias, a_proj, b_proj,
            rec_state, core_out, num_k, num_v, k_dim, v_dim);
    }
}
