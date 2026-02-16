#include "layers/ad_moe.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

ADMoE::ADMoE(int embed_dim_, int hidden_dim, int num_experts_, int top_k_)
    : embed_dim(embed_dim_), num_experts(num_experts_), top_k(top_k_) {
    // Initialize gate weights
    Tensor tW(num_experts, embed_dim);
    Tensor tb(num_experts, 1);
    std::mt19937 gen(std::random_device{}());
    float r = std::sqrt(6.0f / (embed_dim + num_experts));
    std::uniform_real_distribution<float> dist(-r, r);
    for (auto &v : tW.data) v = dist(gen);
    tb.fill(0.0f);
    gate_W = make_ad(tW); register_parameter(gate_W);
    gate_b = make_ad(tb); register_parameter(gate_b);

    // Initialize expert FF networks
    experts.reserve(num_experts);
    for (int i = 0; i < num_experts; ++i) {
        experts.emplace_back(embed_dim, hidden_dim);
    }
}

ADMoE::MoEOutput ADMoE::forward(const std::shared_ptr<ADTensor>& x) {
    int seq_len = x->val.cols;

    // Gate logits: [num_experts x seq_len]
    auto gate_logits = matmul(gate_W, x);
    // Broadcast gate bias
    Tensor ones_t(1, seq_len);
    ones_t.data.assign(seq_len, 1.0f);
    auto ones = make_ad(ones_t);
    auto b_broad = matmul(gate_b, ones);
    gate_logits = add(gate_logits, b_broad);

    // Softmax over experts (row-wise: each column is a distribution over experts)
    // Row-wise max for stability
    Tensor row_max_t(1, seq_len);
    for (int j = 0; j < seq_len; ++j) {
        float mx = gate_logits->val(0, j);
        for (int e = 1; e < num_experts; ++e)
            mx = std::max(mx, gate_logits->val(e, j));
        row_max_t.data[j] = mx;
    }
    // Broadcast max: [num_experts x seq_len]
    Tensor ones_e(num_experts, 1);
    ones_e.data.assign(num_experts, 1.0f);
    auto ones_e_ad = make_ad(ones_e);
    auto max_ad = make_ad(row_max_t);
    auto max_broad = matmul(ones_e_ad, max_ad);
    auto shifted = sub(gate_logits, max_broad);
    auto exp_vals = exp_ad(shifted);

    // Sum over experts per position
    Tensor ones_sum_t(1, num_experts);
    ones_sum_t.data.assign(num_experts, 1.0f);
    auto ones_sum = make_ad(ones_sum_t);
    auto denom = matmul(ones_sum, exp_vals);  // [1 x seq_len]
    auto denom_broad = matmul(ones_e_ad, denom);  // [num_experts x seq_len]
    auto denom_inv = reciprocal(denom_broad);
    auto gate_probs = mul(exp_vals, denom_inv);  // [num_experts x seq_len]

    // For differentiable routing: compute all expert outputs, weight by gate_probs
    // This is the soft MoE approach (all experts process all tokens, weighted by gate)
    // For efficiency, we use top-k masking to zero out non-selected experts

    // Top-k mask: for each position, keep only top_k expert probabilities
    // Create mask from forward values (constant, no gradient needed)
    Tensor mask_t(num_experts, seq_len);
    mask_t.fill(0.0f);
    for (int j = 0; j < seq_len; ++j) {
        std::vector<int> eidxs(num_experts);
        std::iota(eidxs.begin(), eidxs.end(), 0);
        std::partial_sort(eidxs.begin(), eidxs.begin() + top_k, eidxs.end(),
            [&](int a, int b) {
                return gate_probs->val(a, j) > gate_probs->val(b, j);
            });
        for (int k = 0; k < top_k; ++k) {
            mask_t(eidxs[k], j) = 1.0f;
        }
    }
    auto mask_ad = make_ad(mask_t);
    auto masked_probs = mul(gate_probs, mask_ad);  // zero out non-top-k

    // Renormalize masked probs per position
    auto masked_sum = matmul(ones_sum, masked_probs);  // [1 x seq_len]
    auto masked_sum_broad = matmul(ones_e_ad, masked_sum);  // [num_experts x seq_len]
    auto masked_sum_inv = reciprocal(masked_sum_broad);
    auto routing_weights = mul(masked_probs, masked_sum_inv);  // [num_experts x seq_len]

    // Compute weighted combination of expert outputs
    // output = sum_e routing_weights[e,:] * expert_e(x)
    std::shared_ptr<ADTensor> output = nullptr;
    for (int e = 0; e < num_experts; ++e) {
        auto expert_out = experts[e].forward(x);  // [embed_dim x seq_len]
        // Extract routing weight for this expert: [1 x seq_len]
        auto w_e = slice(routing_weights, e, 1);  // [1 x seq_len]
        // Broadcast to [embed_dim x seq_len]
        Tensor ones_dim_t(embed_dim, 1);
        ones_dim_t.data.assign(embed_dim, 1.0f);
        auto ones_dim = make_ad(ones_dim_t);
        auto w_broad = matmul(ones_dim, w_e);  // [embed_dim x seq_len]
        auto weighted = mul(expert_out, w_broad);
        if (output == nullptr) {
            output = weighted;
        } else {
            output = add(output, weighted);
        }
    }

    // Auxiliary load-balancing loss: num_experts * sum_e(f_e^2) where f_e = mean routing weight
    // f_e = (1/seq_len) * sum_j routing_weights[e, j]
    Tensor ones_seq_t(seq_len, 1);
    ones_seq_t.data.assign(seq_len, 1.0f);
    auto ones_seq = make_ad(ones_seq_t);
    auto load_per_expert = matmul(routing_weights, ones_seq);  // [num_experts x 1]
    auto load_scaled = scalar_mul(load_per_expert, 1.0f / seq_len);
    auto load_sq = mul(load_scaled, load_scaled);
    auto aux = sum(load_sq);
    aux = scalar_mul(aux, (float)num_experts);

    return {output, aux};
}
