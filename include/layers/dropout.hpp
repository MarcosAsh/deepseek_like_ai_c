#pragma once
#include "tensor.hpp"
#include <random>

class Dropout {
public:
    Dropout(float drop_prob);
    Tensor forward(const Tensor& input, bool training);

private:
    float drop_prob;
    std::mt19937 gen;
    std::bernoulli_distribution dist;
};