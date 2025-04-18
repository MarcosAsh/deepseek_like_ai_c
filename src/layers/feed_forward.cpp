#include "layers/feed_forward.hpp"
#include <cmath>

FeedForward::FeedForward(int embed_dim, int hidden_dim)
    : fc1(embed_dim, hidden_dim), fc2(hidden_dim, embed_dim) {}

Tensor FeedForward::forward(const Tensor& input) const {
    int embed_dim = input.rows;
    int seq_len = input.cols;
    Tensor output(embed_dim, seq_len);
    // Process each position separately
    for (int pos = 0; pos < seq_len; ++pos) {
        // Slice input column
        Tensor x(embed_dim, 1);
        for (int i = 0; i < embed_dim; ++i) {
            x.data[i] = input.data[i * seq_len + pos];
        }
        // First linear + activation (GELU)
        Tensor h = fc1.forward(x);
        for (auto& v : h.data) {
            double x_val = v;
            // GELU approximation
            v = 0.5f * x_val * (1.0f + std::tanh(0.79788456f * (x_val + 0.044715f * x_val * x_val * x_val)));
        }
        // Second linear
        Tensor y = fc2.forward(h);
        // Write back to output
        for (int i = 0; i < embed_dim; ++i) {
            output.data[i * seq_len + pos] = y.data[i];
        }
    }
    return output;
}