#include "layers/attention.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>

MultiHeadAttention::MultiHeadAttention(int embed_dim_, int num_heads_)
    : embed_dim(embed_dim_), num_heads(num_heads_),
      head_dim( (embed_dim_ % num_heads_ == 0) ? (embed_dim_ / num_heads_) : 0 ),
      W_q(embed_dim_, embed_dim_),
      W_k(embed_dim_, embed_dim_),
      W_v(embed_dim_, embed_dim_),
      W_o(embed_dim_, embed_dim_)
{
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }
    // Xavier initialization for weights
    std::random_device rd;
    std::mt19937 gen(rd());
    float range = std::sqrt(6.0f / (embed_dim + embed_dim));
    std::uniform_real_distribution<float> dist(-range, range);
    for (auto &w : W_q.data) w = dist(gen);
    for (auto &w : W_k.data) w = dist(gen);
    for (auto &w : W_v.data) w = dist(gen);
    for (auto &w : W_o.data) w = dist(gen);
}

Tensor MultiHeadAttention::forward(const Tensor& input) const {
    int seq_len = input.cols;
    // Linear projections
    Tensor Q = W_q.matmul(input); // [embed_dim x seq_len]
    Tensor K = W_k.matmul(input);
    Tensor V = W_v.matmul(input);
    // Prepare output container
    Tensor concat_out(embed_dim, seq_len);
    // Temp storage for scores and attention weights
    std::vector<float> score_mat(seq_len * seq_len);
    std::vector<float> attn_weights(seq_len * seq_len);
    // Iterate over heads
    for (int h = 0; h < num_heads; ++h) {
        int offset = h * head_dim;
        // Compute scaled dot-product scores
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float sum = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    float q = Q.data[(offset + d) * seq_len + i];
                    float k = K.data[(offset + d) * seq_len + j];
                    sum += q * k;
                }
                score_mat[i * seq_len + j] = sum / std::sqrt((float)head_dim);
            }
            // Softmax over j for each i
            float max_score = score_mat[i * seq_len + 0];
            for (int j = 1; j < seq_len; ++j) {
                max_score = std::max(max_score, score_mat[i * seq_len + j]);
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                float e = std::exp(score_mat[i * seq_len + j] - max_score);
                attn_weights[i * seq_len + j] = e;
                sum_exp += e;
            }
            for (int j = 0; j < seq_len; ++j) {
                attn_weights[i * seq_len + j] /= sum_exp;
            }
        }
        // Compute head output [head_dim x seq_len]
        Tensor head_out(head_dim, seq_len);
        for (int d = 0; d < head_dim; ++d) {
            for (int i = 0; i < seq_len; ++i) {
                float val = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    val += attn_weights[i * seq_len + j] *
                           V.data[(offset + d) * seq_len + j];
                }
                head_out.data[d * seq_len + i] = val;
            }
        }
        // Copy head_out into concat_out
        for (int d = 0; d < head_dim; ++d) {
            for (int i = 0; i < seq_len; ++i) {
                concat_out.data[(offset + d) * seq_len + i] =
                    head_out.data[d * seq_len + i];
            }
        }
    }
    // Final output projection
    Tensor output = W_o.matmul(concat_out);
    return output;
}