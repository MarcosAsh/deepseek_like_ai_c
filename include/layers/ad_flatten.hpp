#pragma once
#include "autodiff.hpp"

class ADFlatten {
public:
    int start_dim, end_dim;

    ADFlatten(int start_dim = 1, int end_dim = -1);

    // Input: any shape -> Output: flattened from start_dim to end_dim
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& input);
};
