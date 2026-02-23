#include "layers/attention.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <limits>  // for std::numeric_limits

MultiHeadAttention::MultiHeadAttention(int embed_dim_, int num_heads_, bool causal_, float dropout_prob_)
    : embed_dim(embed_dim_), num_heads(num_heads_),
      head_dim((embed_dim_ % num_heads_ == 0) ? (embed_dim_ / num_heads_) : 0),
      causal(causal_), dropout_prob(dropout_prob_),
      W_q(embed_dim_, embed_dim_),
      W_k(embed_dim_, embed_dim_),
      W_v(embed_dim_, embed_dim_),
      W_o(embed_dim_, embed_dim_),
      k_cache(embed_dim_, 0),
      v_cache(embed_dim_, 0)
{
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    float range = std::sqrt(6.0f / (embed_dim + embed_dim));
    std::uniform_real_distribution<float> dist(-range, range);
    for (auto &w : W_q.data) w = dist(gen);
    for (auto &w : W_k.data) w = dist(gen);
    for (auto &w : W_v.data) w = dist(gen);
    for (auto &w : W_o.data) w = dist(gen);
}

void MultiHeadAttention::clear_cache() {
    k_cache = Tensor(embed_dim, 0);
    v_cache = Tensor(embed_dim, 0);
}

static Tensor hcat(const Tensor& a, const Tensor& b) {
    int rows = a.rows;
    int new_cols = a.cols + b.cols;
    Tensor out(rows, new_cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < a.cols; ++c)
            out.data[r * new_cols + c] = a.data[r * a.cols + c];
        for (int c = 0; c < b.cols; ++c)
            out.data[r * new_cols + a.cols + c] = b.data[r * b.cols + c];
    }
    return out;
}

Tensor MultiHeadAttention::forward(const Tensor& input, bool training, bool use_cache) {
    int q_len = input.cols;
    static thread_local std::mt19937 _rng(std::random_device{}());
    float _keep_prob = 1.0f - dropout_prob;
    std::bernoulli_distribution _dist(_keep_prob);
    Tensor Q = W_q.matmul(input); // [embed_dim x q_len]
    Tensor K_new = W_k.matmul(input);
    Tensor V_new = W_v.matmul(input);

    Tensor K_full = [&]() -> Tensor {
        if (use_cache && k_cache.cols > 0) return hcat(k_cache, K_new);
        return K_new;
    }();
    Tensor V_full = [&]() -> Tensor {
        if (use_cache && v_cache.cols > 0) return hcat(v_cache, V_new);
        return V_new;
    }();
    if (use_cache) {
        k_cache = K_full;
        v_cache = V_full;
    }

    int kv_len = K_full.cols;
    int pos_offset = kv_len - q_len;
    Tensor concat_out(embed_dim, q_len);
    std::vector<float> score_mat(q_len * kv_len);
    std::vector<float> attn_weights(q_len * kv_len);
    for (int h = 0; h < num_heads; ++h) {
        int offset = h * head_dim;
        for (int i = 0; i < q_len; ++i) {
            int abs_i = pos_offset + i;
            for (int j = 0; j < kv_len; ++j) {
                float sum = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    float q = Q.data[(offset + d) * q_len + i];
                    float k = K_full.data[(offset + d) * kv_len + j];
                    sum += q * k;
                }
                float scaled = sum / std::sqrt((float)head_dim);
                if (causal && j > abs_i) {
                    scaled = -std::numeric_limits<float>::infinity();
                }
                score_mat[i * kv_len + j] = scaled;
            }
            float max_score = score_mat[i * kv_len + 0];
            for (int j = 1; j < kv_len; ++j) {
                max_score = std::max(max_score, score_mat[i * kv_len + j]);
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < kv_len; ++j) {
                float e = std::exp(score_mat[i * kv_len + j] - max_score);
                attn_weights[i * kv_len + j] = e;
                sum_exp += e;
            }
            for (int j = 0; j < kv_len; ++j) {
                attn_weights[i * kv_len + j] /= sum_exp;
            }
            if (training && dropout_prob > 0.0f) {
                for (int j = 0; j < kv_len; ++j) {
                    bool keep = _dist(_rng);
                    float& w = attn_weights[i * kv_len + j];
                    w = keep ? (w / _keep_prob) : 0.0f;
                }
            }
        }
        for (int d = 0; d < head_dim; ++d) {
            for (int i = 0; i < q_len; ++i) {
                float val = 0.0f;
                for (int j = 0; j < kv_len; ++j) {
                    val += attn_weights[i * kv_len + j] *
                           V_full.data[(offset + d) * kv_len + j];
                }
                concat_out.data[(offset + d) * q_len + i] = val;
            }
        }
    }
    Tensor output = W_o.matmul(concat_out);
    return output;
}