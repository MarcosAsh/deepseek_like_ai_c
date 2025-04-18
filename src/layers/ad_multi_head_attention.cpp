#include "layers/ad_multi_head_attention.hpp"
#include <random>
#include <stdexcept>
#include <cmath>

ADMultiHeadAttention::ADMultiHeadAttention(int embed_dim_, int num_heads_)
    : embed_dim(embed_dim_), num_heads(num_heads_) {
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }
    head_dim = embed_dim / num_heads;
    // Initialize weight matrices
    Tensor tWq(embed_dim, embed_dim), tWk(embed_dim, embed_dim),
           tWv(embed_dim, embed_dim), tWo(embed_dim, embed_dim);
    std::mt19937 gen(std::random_device{}());
    float range = std::sqrt(6.0f / (2 * embed_dim));
    std::uniform_real_distribution<float> dist(-range, range);
    for (auto &v : tWq.data) v = dist(gen);
    for (auto &v : tWk.data) v = dist(gen);
    for (auto &v : tWv.data) v = dist(gen);
    for (auto &v : tWo.data) v = dist(gen);
    W_q = make_ad(tWq); register_parameter(W_q);
    W_k = make_ad(tWk); register_parameter(W_k);
    W_v = make_ad(tWv); register_parameter(W_v);
    W_o = make_ad(tWo); register_parameter(W_o);
}

std::shared_ptr<ADTensor> ADMultiHeadAttention::forward(
    const std::shared_ptr<ADTensor>& input) {
    // Project to Q, K, V
    auto Q = matmul(W_q, input);
    auto K = matmul(W_k, input);
    auto V = matmul(W_v, input);
    int seq_len = input->val.cols;
    std::vector<std::shared_ptr<ADTensor>> heads;
    heads.reserve(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        int offset = h * head_dim;
        auto Qh = slice(Q, offset, head_dim);
        auto Kh = slice(K, offset, head_dim);
        auto Vh = slice(V, offset, head_dim);
        // Attention scores: [seq_len x seq_len] = Qh^T * Kh
        auto Qh_T = transpose(Qh);
        auto scores = matmul(Qh_T, Kh);
        // Scale
        float scale = 1.0f / std::sqrt((float)head_dim);
        auto scores_scaled = scalar_mul(scores, scale);
        // Softmax
        auto scores_exp = exp_ad(scores_scaled);
        // Sum across keys per query
        Tensor ones_col_t(seq_len, 1);
        for (int i = 0; i < seq_len; ++i) ones_col_t.data[i] = 1.0f;
        auto ones_col = make_ad(ones_col_t);
        Tensor ones_row_t(1, seq_len);
        for (int j = 0; j < seq_len; ++j) ones_row_t.data[j] = 1.0f;
        auto ones_row = make_ad(ones_row_t);
        auto denom_col = matmul(scores_exp, ones_col);      // [seq_len x 1]
        auto denom = matmul(denom_col, ones_row);          // [seq_len x seq_len]
        auto denom_recip = reciprocal(denom);
        auto attn = mul(scores_exp, denom_recip);          // [seq_len x seq_len]
        // Weighted sum: head_out = Vh * attn^T  => [head_dim x seq_len]
        auto attn_T = transpose(attn);
        auto head_out = matmul(Vh, attn_T);
        heads.push_back(head_out);
    }
    // Concatenate heads -> [embed_dim x seq_len]
    auto concat_out = concat(heads);
    // Final linear projection
    auto out = matmul(W_o, concat_out);
    return out;
}