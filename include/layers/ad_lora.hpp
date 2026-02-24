#pragma once
#include "autodiff.hpp"

// LoRA: Low-Rank Adaptation
// Wraps an existing weight matrix W with low-rank decomposition: W' = W + alpha * (B @ A)
// Only A and B are trainable; W is frozen
class ADLoRA {
public:
    // input_dim, output_dim: dimensions of the original weight matrix
    // rank: rank of low-rank matrices (typically 4-64)
    // alpha: scaling factor (typically equal to rank)
    ADLoRA(int input_dim, int output_dim, int rank = 8, float alpha = 8.0f);
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& x);

private:
    int input_dim, output_dim, rank;
    float alpha;
    std::shared_ptr<ADTensor> W;       // [output_dim x input_dim] frozen base weight
    std::shared_ptr<ADTensor> A;       // [rank x input_dim] trainable down-projection
    std::shared_ptr<ADTensor> B;       // [output_dim x rank] trainable up-projection
    std::shared_ptr<ADTensor> bias;    // [output_dim x 1]
};
