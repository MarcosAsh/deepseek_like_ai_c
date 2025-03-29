#pragma once
#include "tensor.hpp"

class MoE {
public:
    Tensor forward(const Tensor& input);
};
