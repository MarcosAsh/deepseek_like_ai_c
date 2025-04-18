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