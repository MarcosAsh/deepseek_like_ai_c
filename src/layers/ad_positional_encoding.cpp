#include "layers/ad_positional_encoding.hpp"
#include <random>
#include <cmath>
#include <stdexcept>

ADPositionalEncoding::ADPositionalEncoding(int embed_dim_, int max_len_)
    : embed_dim(embed_dim_), max_len(max_len_) {
    // Initialize learnable positional embeddings (Xavier uniform)
    Tensor pw(embed_dim, max_len);
    std::mt19937 gen(std::random_device{}());
    float r = std::sqrt(6.0f / (embed_dim + max_len));
    std::uniform_real_distribution<float> dist(-r, r);
    for (auto &v : pw.data) v = dist(gen);
    pweights = make_ad(pw);
    register_parameter(pweights);
}

std::shared_ptr<ADTensor> ADPositionalEncoding::forward(int seq_len) const {
    if (seq_len > max_len) throw std::out_of_range("Sequence length exceeds max_len");
    // Build one-hot selector of shape [max_len x seq_len]
    Tensor sel(max_len, seq_len);
    sel.data.assign(max_len * seq_len, 0.0f);
    for (int pos = 0; pos < seq_len; ++pos) {
        sel.data[pos * seq_len + pos] = 1.0f;
    }
    auto sel_ad = make_ad(sel);
    // Select positional embeddings: [embed_dim x max_len] * [max_len x seq_len] -> [embed_dim x seq_len]
    return matmul(pweights, sel_ad);
}