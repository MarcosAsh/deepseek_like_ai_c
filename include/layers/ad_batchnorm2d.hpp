#pragma once
#include "autodiff.hpp"

class ADBatchNorm2D {
public:
    int num_features;
    float eps;
    float momentum;
    bool training;

    std::shared_ptr<ADTensor> gamma;  // [num_features]
    std::shared_ptr<ADTensor> beta;   // [num_features]
    Tensor running_mean;  // [num_features]
    Tensor running_var;   // [num_features]

    ADBatchNorm2D(int num_features, float eps = 1e-5f, float momentum = 0.1f);

    // Input: [B, C, H, W] -> Output: [B, C, H, W]
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& input);
};
