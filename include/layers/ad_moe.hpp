#pragma once
#include "autodiff.hpp"
#include "layers/ad_linear.hpp"
#include "layers/ad_feed_forward.hpp"
#include <vector>
#include <memory>

// AD-aware Mixture of Experts with differentiable soft routing
class ADMoE {
public:
    // embed_dim: model dim, hidden_dim: expert FF hidden dim
    // num_experts: total experts, top_k: experts per token
    ADMoE(int embed_dim, int hidden_dim, int num_experts, int top_k = 2);

    // x: [embed_dim x seq_len]
    // Returns: (output [embed_dim x seq_len], aux_loss scalar ADTensor)
    struct MoEOutput {
        std::shared_ptr<ADTensor> output;
        std::shared_ptr<ADTensor> aux_loss;
    };
    MoEOutput forward(const std::shared_ptr<ADTensor>& x);

private:
    int embed_dim;
    int num_experts;
    int top_k;
    // Gate: [num_experts x embed_dim] linear
    std::shared_ptr<ADTensor> gate_W;
    std::shared_ptr<ADTensor> gate_b;
    // Expert feed-forward networks
    std::vector<ADFeedForward> experts;
};
