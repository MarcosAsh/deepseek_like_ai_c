#pragma once
#include "tensor.hpp"

class Norm {
public:
    Tensor forward(const Tensor& input);
};
