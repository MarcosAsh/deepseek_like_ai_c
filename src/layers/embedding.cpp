#include "layers/embedding.hpp"
#include <random>
#include <stdexcept>
#include <cmath>

Embedding::Embedding(int vocab_size, int embed_dim)
    : weights(embed_dim, vocab_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    float range = std::sqrt(6.0f / (vocab_size + embed_dim));
    std::uniform_real_distribution<float> dist(-range, range);
    for (auto &w : weights.data) {
        w = dist(gen);
    }
}

Tensor Embedding::forward(const std::vector<int>& tokens) const {
    int seq_len = static_cast<int>(tokens.size());
    int embed_dim = weights.rows;
    int vocab_size = weights.cols;
    Tensor output(embed_dim, seq_len);
    for (int pos = 0; pos < seq_len; ++pos) {
        int token_id = tokens[pos];
        if (token_id < 0 || token_id >= vocab_size) {
            throw std::out_of_range("Token ID out of range in Embedding::forward");
        }
        for (int i = 0; i < embed_dim; ++i) {
            output.data[i * seq_len + pos] = weights.data[i * vocab_size + token_id];
        }
    }
    return output;
}