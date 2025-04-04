#pragma once
#include "tensor.hpp"

class MoE {
public:
    MoE(int input_dim, int expert_dim, int num_experts);
    Tensor forward(const Tensor& input);
};