#pragma once
#include "tensor.hpp"
#include "layers/linear.hpp"
#include "layers/feed_forward.hpp"
#include <vector>

// Mixture of Experts layer with top-k routing and load-balancing auxiliary loss
class MoE {
public:
    // input_dim: model dim, expert_dim: FF hidden dim, num_experts: total experts, top_k: experts per token
    MoE(int input_dim, int expert_dim, int num_experts, int top_k = 2);

    // input: [input_dim x seq_len], returns [input_dim x seq_len]
    // aux_loss is accumulated with the load-balancing loss
    Tensor forward(const Tensor& input, float& aux_loss);

    Linear gate;                       // [num_experts x input_dim]
    std::vector<FeedForward> experts;
    int num_experts;
    int top_k;
};