#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <cuda_fp16.h>

struct SamplingParams {
    float temperature = 0.6f;
    int top_k = 20;
    float top_p = 0.95f;
    float min_p = 0.0f;
    float rep_penalty = 1.0f;    // 1.0 = disabled
    int rep_window = 64;         // how many recent tokens to penalize
    float freq_penalty = 0.0f;   // OpenAI-style frequency penalty
    float pres_penalty = 0.0f;   // OpenAI-style presence penalty
    int max_tokens = 0;  // 0 = no limit (generate until EOS)
    uint64_t seed = 0;           // 0 = random seed

    void print() const {
        printf("Sampling: temp=%.2f top_k=%d top_p=%.2f min_p=%.2f rep_pen=%.2f",
               temperature, top_k, top_p, min_p, rep_penalty);
        if (max_tokens > 0) printf(" max=%d", max_tokens);
        printf("\n");
    }
};

struct Sampler {
    SamplingParams params;
    std::mt19937 rng;
    // Pre-allocated buffers to avoid per-token allocation
    std::vector<float> logits;
    std::vector<int> indices;

    void init(const SamplingParams& p, int vocab_size) {
        params = p;
        logits.resize(vocab_size);
        indices.resize(vocab_size);
        if (p.seed == 0) {
            std::random_device rd;
            rng.seed(rd());
        } else {
            rng.seed(p.seed);
        }
    }

    int sample(const half* logits_half, int vocab_size, const std::vector<int>& context) {
        // Convert to float (reuse buffer)
        for (int i = 0; i < vocab_size; i++)
            logits[i] = __half2float(logits_half[i]);

        // 1. Repetition penalty
        if (params.rep_penalty != 1.0f || params.freq_penalty != 0.0f || params.pres_penalty != 0.0f) {
            apply_rep_penalty(logits, context);
        }

        // 2. Greedy
        if (params.temperature <= 0.0f) {
            return std::max_element(logits.begin(), logits.begin() + vocab_size) - logits.begin();
        }

        // 3. Temperature
        float inv_temp = 1.0f / params.temperature;
        for (int i = 0; i < vocab_size; i++) logits[i] *= inv_temp;

        // 4. Determine effective K for partial sort
        // Even with top_k=0, we only need ~1000 candidates for top_p sampling
        int eff_k = params.top_k > 0 ? params.top_k : 256;
        if (eff_k > vocab_size) eff_k = vocab_size;

        // Build index array and partial sort (top-K only)
        for (int i = 0; i < vocab_size; i++) indices[i] = i;
        std::partial_sort(indices.begin(), indices.begin() + eff_k, indices.begin() + vocab_size,
            [this](int a, int b) { return logits[a] > logits[b]; });

        // 5. Softmax over top-K
        float max_logit = logits[indices[0]];
        float sum = 0.0f;
        // Reuse a small stack buffer for probs
        std::vector<float> probs(eff_k);
        for (int i = 0; i < eff_k; i++) {
            probs[i] = expf(logits[indices[i]] - max_logit);
            sum += probs[i];
        }
        for (int i = 0; i < eff_k; i++) probs[i] /= sum;

        // 6. Min-p
        int n_keep = eff_k;
        if (params.min_p > 0.0f) {
            float threshold = params.min_p * probs[0];
            for (int i = 1; i < eff_k; i++) {
                if (probs[i] < threshold) { n_keep = i; break; }
            }
        }

        // 7. Top-p (nucleus)
        if (params.top_p < 1.0f) {
            float cumsum = 0.0f;
            for (int i = 0; i < n_keep; i++) {
                cumsum += probs[i];
                if (cumsum >= params.top_p) { n_keep = i + 1; break; }
            }
        }

        // 8. Re-normalize and sample
        sum = 0.0f;
        for (int i = 0; i < n_keep; i++) sum += probs[i];
        std::uniform_real_distribution<float> dist(0.0f, sum);
        float r = dist(rng);
        float acc = 0.0f;
        for (int i = 0; i < n_keep; i++) {
            acc += probs[i];
            if (acc >= r) return indices[i];
        }
        return indices[0];
    }

private:
    void apply_rep_penalty(std::vector<float>& logits, const std::vector<int>& context) {
        int start = std::max(0, (int)context.size() - params.rep_window);
        std::unordered_map<int, int> counts;
        for (int i = start; i < (int)context.size(); i++)
            counts[context[i]]++;

        for (auto& [tok, cnt] : counts) {
            if (tok < 0 || tok >= (int)logits.size()) continue;
            if (params.rep_penalty != 1.0f) {
                if (logits[tok] > 0) logits[tok] /= params.rep_penalty;
                else logits[tok] *= params.rep_penalty;
            }
            logits[tok] -= params.freq_penalty * cnt;
            logits[tok] -= params.pres_penalty;
        }
    }
};

// Parse sampling params from argv, return index of first non-flag arg after model path
inline int parse_sampling_args(int argc, char** argv, SamplingParams& sp) {
    int first_token_arg = 2;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            sp.temperature = atof(argv[++i]); first_token_arg = i + 1;
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            sp.top_k = atoi(argv[++i]); first_token_arg = i + 1;
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            sp.top_p = atof(argv[++i]); first_token_arg = i + 1;
        } else if (strcmp(argv[i], "--min-p") == 0 && i + 1 < argc) {
            sp.min_p = atof(argv[++i]); first_token_arg = i + 1;
        } else if (strcmp(argv[i], "--rep-pen") == 0 && i + 1 < argc) {
            sp.rep_penalty = atof(argv[++i]); first_token_arg = i + 1;
        } else if (strcmp(argv[i], "--rep-win") == 0 && i + 1 < argc) {
            sp.rep_window = atoi(argv[++i]); first_token_arg = i + 1;
        } else if (strcmp(argv[i], "--freq-pen") == 0 && i + 1 < argc) {
            sp.freq_penalty = atof(argv[++i]); first_token_arg = i + 1;
        } else if (strcmp(argv[i], "--pres-pen") == 0 && i + 1 < argc) {
            sp.pres_penalty = atof(argv[++i]); first_token_arg = i + 1;
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            sp.max_tokens = atoi(argv[++i]); first_token_arg = i + 1;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            sp.seed = strtoull(argv[++i], nullptr, 10); first_token_arg = i + 1;
        } else if (strcmp(argv[i], "--serve") == 0 && i + 1 < argc) {
            i++; first_token_arg = i + 1;
        } else {
            first_token_arg = i;
            break;
        }
    }
    return first_token_arg;
}
