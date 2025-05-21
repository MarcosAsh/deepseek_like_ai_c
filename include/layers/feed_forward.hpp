#pragma once
#include "tensor.hpp"
#include "layers/linear.hpp"

// Position-wise Feed-Forward network: two-layer MLP with activation
class FeedForward {
public:
    // embed_dim: input/output dimension; hidden_dim: inner layer dimension
    // dropout_prob: probability of dropping activations between layers
    FeedForward(int embed_dim, int hidden_dim, float dropout_prob = 0.0f);
    // input: [embed_dim x seq_len], returns [embed_dim x seq_len]
    Tensor forward(const Tensor& input) const;

private:
    // inner linear layers
    Linear fc1;
    Linear fc2;
    // dropout probability after activation in first layer
    float dropout_prob;
};