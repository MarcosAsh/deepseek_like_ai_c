#pragma once
#include <vector>
#include "autodiff.hpp"

// Simple SGD optimizer (no momentum)
class SGD {
public:
    // learning_rate
    explicit SGD(float lr);
    // Perform parameter update: p.val -= lr * p.grad
    void step();
    // Zero out gradients on all parameters
    void zero_grad();

private:
    float lr;
};
// AdamW optimizer with weight decay and gradient clipping
class AdamW {
public:
    // lr: learning rate
    // beta1, beta2: moment coefficients
    // eps: numerical stability
    // weight_decay: L2 penalty multiplier
    // clip_norm: maximum gradient norm (0 to disable)
    AdamW(float lr, float beta1=0.9f, float beta2=0.999f, float eps=1e-8f,
          float weight_decay=0.01f, float clip_norm=1.0f);
    // Update parameters
    void step();
    // Zero all gradients
    void zero_grad();
private:
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float clip_norm;
    int t;
    std::vector<Tensor> m;
    std::vector<Tensor> v;
};