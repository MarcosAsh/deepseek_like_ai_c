#pragma once
#include "tensor.hpp"
#include "layers/linear.hpp"

class FeedForward {
public:
    FeedForward(int embed_dim, int hidden_dim, float dropout_prob = 0.0f);
    Tensor forward(const Tensor& input, bool training = false) const;

    Linear fc1;
    Linear fc2;
    float dropout_prob;
};