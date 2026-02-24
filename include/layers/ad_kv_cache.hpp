#pragma once
#include "autodiff.hpp"
#include <deque>

// Sliding Window KV Cache: compresses KV cache by keeping only the most recent window_size tokens
class ADKVCache {
public:
    explicit ADKVCache(int window_size = 512);

    // Append new key/value tensors and return the windowed K,V
    // k_new, v_new: [head_dim x new_seq_len]
    // Returns pair of (K_windowed, V_windowed): [head_dim x min(total_len, window_size)]
    struct KVPair {
        std::shared_ptr<ADTensor> keys;
        std::shared_ptr<ADTensor> values;
    };

    KVPair update(const std::shared_ptr<ADTensor>& k_new,
                  const std::shared_ptr<ADTensor>& v_new);

    void clear();
    int cached_length() const;

private:
    int window_size;
    // Store raw tensors for the cache (non-AD for efficiency)
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    int head_dim;
    int current_len;
};
