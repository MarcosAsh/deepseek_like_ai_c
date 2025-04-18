#pragma once
#include "tensor.hpp"
#include <random>

// Dropout layer
class Dropout {
public:
    // drop_prob: probability of dropping each element
    Dropout(float drop_prob);
    // Forward pass: if training=false, no dropout applied
    Tensor forward(const Tensor& input, bool training);

private:
    float drop_prob;
    std::mt19937 gen;
    std::bernoulli_distribution dist;
};