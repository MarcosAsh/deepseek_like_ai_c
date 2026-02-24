#include "layers/feed_forward.hpp"
#include <cmath>
#include <random>

// GELU approximation constants (Hendrycks & Gimpel, 2016)
static constexpr float GELU_SQRT_2_OVER_PI = 0.79788456f;  // sqrt(2/pi)
static constexpr float GELU_COEFF = 0.044715f;

FeedForward::FeedForward(int embed_dim, int hidden_dim, float dropout_prob_)
    : fc1(embed_dim, hidden_dim),
      fc2(hidden_dim, embed_dim),
      dropout_prob(dropout_prob_) {}

Tensor FeedForward::forward(const Tensor& input, bool training) const {
    // Batched: fc1 handles [embed_dim x seq_len] -> [hidden_dim x seq_len]
    Tensor h = fc1.forward(input);

    // GELU activation (applied elementwise)
    for (auto& v : h.data) {
        double x_val = v;
        v = 0.5f * x_val * (1.0f + std::tanh(GELU_SQRT_2_OVER_PI * (x_val + GELU_COEFF * x_val * x_val * x_val)));
    }

    // Apply dropout after activation only during training
    if (training && dropout_prob > 0.0f) {
        static thread_local std::mt19937 _rng(std::random_device{}());
        float _keep_prob = 1.0f - dropout_prob;
        std::bernoulli_distribution _dist(_keep_prob);
        for (auto &v : h.data) {
            bool keep = _dist(_rng);
            v = keep ? (v / _keep_prob) : 0.0f;
        }
    }

    // Second linear: [hidden_dim x seq_len] -> [embed_dim x seq_len]
    return fc2.forward(h);
}