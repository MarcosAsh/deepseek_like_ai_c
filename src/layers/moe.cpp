#include "layers/moe.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

MoE::MoE(int input_dim, int expert_dim, int num_experts_, int top_k_)
    : gate(input_dim, num_experts_), num_experts(num_experts_), top_k(top_k_) {
    experts.reserve(num_experts);
    for (int i = 0; i < num_experts; ++i) {
        experts.emplace_back(input_dim, expert_dim);
    }
}

Tensor MoE::forward(const Tensor& input, float& aux_loss) {
    int dim = input.rows;
    int seq_len = input.cols;
    Tensor output(dim, seq_len);
    output.fill(0.0f);

    // Track load for auxiliary loss
    std::vector<float> expert_load(num_experts, 0.0f);

    // Process each position
    for (int pos = 0; pos < seq_len; ++pos) {
        // Extract column vector
        Tensor x(dim, 1);
        for (int i = 0; i < dim; ++i) x.data[i] = input(i, pos);

        // Gate logits: [num_experts x 1]
        Tensor gate_logits = gate.forward(x);

        // Softmax over experts
        std::vector<float> probs(num_experts);
        float max_logit = *std::max_element(gate_logits.data.begin(), gate_logits.data.end());
        float sum_exp = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            probs[e] = std::exp(gate_logits.data[e] - max_logit);
            sum_exp += probs[e];
        }
        for (int e = 0; e < num_experts; ++e) probs[e] /= sum_exp;

        // Top-k selection
        std::vector<int> idxs(num_experts);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::partial_sort(idxs.begin(), idxs.begin() + top_k, idxs.end(),
                          [&](int a, int b) { return probs[a] > probs[b]; });

        // Normalize top-k weights
        float topk_sum = 0.0f;
        for (int k = 0; k < top_k; ++k) topk_sum += probs[idxs[k]];
        if (topk_sum < 1e-9f) topk_sum = 1e-9f;

        // Weighted sum of expert outputs
        for (int k = 0; k < top_k; ++k) {
            int eidx = idxs[k];
            float w = probs[eidx] / topk_sum;
            expert_load[eidx] += w;
            Tensor expert_out = experts[eidx].forward(x);
            for (int i = 0; i < dim; ++i) {
                output(i, pos) += w * expert_out.data[i];
            }
        }
    }

    // Load-balancing auxiliary loss: encourage uniform distribution
    // aux = num_experts * sum_e(f_e * P_e) where f_e = fraction of tokens routed to e
    // and P_e = average gate probability for expert e
    float total_load = 0.0f;
    for (int e = 0; e < num_experts; ++e) total_load += expert_load[e];
    if (total_load > 0.0f && seq_len > 0) {
        float aux = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            float f_e = expert_load[e] / total_load;
            aux += f_e * f_e;
        }
        aux_loss += num_experts * aux;
    }

    return output;
}
