#pragma once
#include "autodiff.hpp"
#include <vector>

// Grouped Query Attention: uses fewer KV heads than Q heads to save memory
// Q has num_heads heads, K/V have num_kv_heads heads (num_heads must be divisible by num_kv_heads)
class ADGQA {
public:
    ADGQA(int embed_dim, int num_heads, int num_kv_heads, bool causal = true);
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& input);

private:
    int embed_dim;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int kv_group_size;  // num_heads / num_kv_heads
    bool causal;
    std::vector<float> alibi_slopes;
    std::shared_ptr<ADTensor> W_q, W_k, W_v, W_o;
};
