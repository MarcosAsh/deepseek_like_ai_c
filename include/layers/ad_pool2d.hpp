#pragma once
#include "autodiff.hpp"

class ADMaxPool2D {
public:
    int kernel_size, stride, padding;

    ADMaxPool2D(int kernel_sz, int stride = -1, int padding = 0);

    // Input: [B, C, H, W] -> Output: [B, C, Hout, Wout]
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& input);
};

class ADAvgPool2D {
public:
    int kernel_size, stride, padding;

    ADAvgPool2D(int kernel_sz, int stride = -1, int padding = 0);

    // Input: [B, C, H, W] -> Output: [B, C, Hout, Wout]
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& input);
};
