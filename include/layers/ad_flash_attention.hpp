#pragma once
#include "autodiff.hpp"
#include <vector>

// Flash Attention: tiled attention computation for memory efficiency
// Computes exact attention but in tiles to reduce peak memory usage
// Based on the FlashAttention algorithm (Dao et al., 2022)
class ADFlashAttention {
public:
    ADFlashAttention(int embed_dim, int num_heads, int tile_size = 32, bool causal = true);
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& input);

private:
    int embed_dim;
    int num_heads;
    int head_dim;
    int tile_size;
    bool causal;
    std::vector<float> alibi_slopes;
    std::shared_ptr<ADTensor> W_q, W_k, W_v, W_o;

    // Tiled softmax-attention for a single head
    // Q, K, V: [head_dim x seq_len]
    // Returns: [head_dim x seq_len]
    std::shared_ptr<ADTensor> tiled_attention(
        const std::shared_ptr<ADTensor>& Q,
        const std::shared_ptr<ADTensor>& K,
        const std::shared_ptr<ADTensor>& V,
        int head_idx);
};
