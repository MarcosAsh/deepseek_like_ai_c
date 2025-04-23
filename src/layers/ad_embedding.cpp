#include "layers/ad_embedding.hpp"
#include <random>
#include <stdexcept>
#include <cmath>

ADEmbedding::ADEmbedding(int vocab_size_, int embed_dim_)
    : vocab_size(vocab_size_), embed_dim(embed_dim_) {
    Tensor tW(embed_dim, vocab_size);
    // Xavier uniform initialization: U[-r, r] with r = sqrt(6/(vocab_size + embed_dim))
    std::mt19937 gen(std::random_device{}());
    float r = std::sqrt(6.0f / (vocab_size + embed_dim));
    std::uniform_real_distribution<float> dist(-r, r);
    for (auto &v : tW.data) v = dist(gen);
    weights = make_ad(tW);
    register_parameter(weights);
}

std::shared_ptr<ADTensor> ADEmbedding::forward(const std::vector<int>& tokens) const {
    int seq_len = static_cast<int>(tokens.size());
    // Build one-hot matrix [vocab_size x seq_len]
    Tensor X(vocab_size, seq_len);
    for (int j = 0; j < seq_len; ++j) {
        int id = tokens[j];
        if (id < 0 || id >= vocab_size) throw std::out_of_range("Token ID out of range");
        for (int i = 0; i < vocab_size; ++i) {
            X.data[i * seq_len + j] = (i == id ? 1.0f : 0.0f);
        }
    }
    auto X_ad = make_ad(X);
    // Multiply: [embed_dim x vocab_size] * [vocab_size x seq_len] -> [embed_dim x seq_len]
    return matmul(weights, X_ad);
}