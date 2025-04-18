#include "layers/moe.hpp"
#include <iostream>

// Stub constructor for MoE layer
MoE::MoE(int input_dim, int expert_dim, int num_experts) {
    // No parameters initialized yet
}

Tensor MoE::forward(const Tensor& input) {
    std::cout << "MoE layer forward" << std::endl;
    return input;
}
