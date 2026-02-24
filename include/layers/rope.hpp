#pragma once
#include "autodiff.hpp"
#include <vector>
#include <cmath>

// Rotary Position Embeddings (RoPE) as used in LLaMA / DeepSeek
// Applies rotation to pairs of dimensions in Q and K tensors
class RoPE {
public:
    explicit RoPE(int head_dim, int max_len = 4096, float theta = 10000.0f);

    // Apply rotary embeddings to a tensor of shape [head_dim x seq_len]
    // pos_offset is used during cached inference to offset positions
    Tensor apply(const Tensor& x, int pos_offset = 0) const;

    // AD version: wraps apply for autodiff-compatible forward pass
    std::shared_ptr<ADTensor> apply_ad(const std::shared_ptr<ADTensor>& x,
                                        int pos_offset = 0) const;

private:
    int head_dim_;
    int max_len_;
    // Precomputed cos/sin tables: [head_dim/2 x max_len]
    std::vector<float> cos_table_;
    std::vector<float> sin_table_;
};
