#pragma once
#include "tensor.hpp"
#include "layers/linear.hpp"

// Position-wise Feed-Forward network: two-layer MLP with activation
class FeedForward {
public:
    // embed_dim: input/output dimension; hidden_dim: inner layer dimension
    FeedForward(int embed_dim, int hidden_dim);
    // input: [embed_dim x seq_len], returns [embed_dim x seq_len]
    Tensor forward(const Tensor& input) const;

private:
    Linear fc1;
    Linear fc2;
};