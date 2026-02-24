#include "layers/ad_kv_cache.hpp"
#include <algorithm>
#include <cstring>

ADKVCache::ADKVCache(int window_size_)
    : window_size(window_size_), head_dim(0), current_len(0) {}

ADKVCache::KVPair ADKVCache::update(
    const std::shared_ptr<ADTensor>& k_new,
    const std::shared_ptr<ADTensor>& v_new) {

    int new_dim = k_new->val.rows;
    int new_len = k_new->val.cols;

    // Initialize head_dim on first call
    if (head_dim == 0) {
        head_dim = new_dim;
    }

    // Append new data to cache
    int old_len = current_len;
    int total_len = old_len + new_len;

    // Resize cache to hold all data
    k_cache.resize(head_dim * total_len);
    v_cache.resize(head_dim * total_len);

    // Copy new K/V data (column-major: each column is a position)
    for (int d = 0; d < head_dim; ++d) {
        for (int t = 0; t < new_len; ++t) {
            k_cache[d * total_len + old_len + t] = k_new->val.data[d * new_len + t];
            v_cache[d * total_len + old_len + t] = v_new->val.data[d * new_len + t];
        }
    }
    current_len = total_len;

    // Apply sliding window: keep only the last window_size positions
    int out_len = std::min(current_len, window_size);
    int start = current_len - out_len;

    Tensor k_out(head_dim, out_len);
    Tensor v_out(head_dim, out_len);

    for (int d = 0; d < head_dim; ++d) {
        for (int t = 0; t < out_len; ++t) {
            k_out.data[d * out_len + t] = k_cache[d * current_len + start + t];
            v_out.data[d * out_len + t] = v_cache[d * current_len + start + t];
        }
    }

    // Compact cache if it grew too large
    if (current_len > window_size) {
        std::vector<float> new_k(head_dim * out_len);
        std::vector<float> new_v(head_dim * out_len);
        for (int d = 0; d < head_dim; ++d) {
            for (int t = 0; t < out_len; ++t) {
                new_k[d * out_len + t] = k_cache[d * current_len + start + t];
                new_v[d * out_len + t] = v_cache[d * current_len + start + t];
            }
        }
        k_cache = std::move(new_k);
        v_cache = std::move(new_v);
        current_len = out_len;
    }

    return {make_ad(k_out), make_ad(v_out)};
}

void ADKVCache::clear() {
    k_cache.clear();
    v_cache.clear();
    current_len = 0;
    head_dim = 0;
}

int ADKVCache::cached_length() const {
    return current_len;
}
