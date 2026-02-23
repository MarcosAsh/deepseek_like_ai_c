#pragma once
#include "tensor.hpp"
#include "layers/linear.hpp"
#include "layers/feed_forward.hpp"
#include <vector>

class MoE {
public:
    MoE(int input_dim, int expert_dim, int num_experts, int top_k = 2);

    Tensor forward(const Tensor& input, float& aux_loss);

    Linear gate;                       // [num_experts x input_dim]
    std::vector<FeedForward> experts;
    int num_experts;
    int top_k;
};