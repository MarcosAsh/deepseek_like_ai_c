#include "layers/ad_flash_attention.hpp"
#include <random>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <algorithm>

ADFlashAttention::ADFlashAttention(int embed_dim_, int num_heads_, int tile_size_, bool causal_)
    : embed_dim(embed_dim_), num_heads(num_heads_), tile_size(tile_size_), causal(causal_) {
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }
    head_dim = embed_dim / num_heads;

    alibi_slopes.resize(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        alibi_slopes[h] = std::pow(2.0f, -8.0f * static_cast<float>(h + 1) / static_cast<float>(num_heads));
    }

    Tensor tWq(embed_dim, embed_dim), tWk(embed_dim, embed_dim),
           tWv(embed_dim, embed_dim), tWo(embed_dim, embed_dim);
    std::mt19937 gen(std::random_device{}());
    float range = std::sqrt(6.0f / (2 * embed_dim));
    std::uniform_real_distribution<float> dist(-range, range);
    for (auto& v : tWq.data) v = dist(gen);
    for (auto& v : tWk.data) v = dist(gen);
    for (auto& v : tWv.data) v = dist(gen);
    for (auto& v : tWo.data) v = dist(gen);

    W_q = make_ad(tWq); register_parameter(W_q);
    W_k = make_ad(tWk); register_parameter(W_k);
    W_v = make_ad(tWv); register_parameter(W_v);
    W_o = make_ad(tWo); register_parameter(W_o);
}

std::shared_ptr<ADTensor> ADFlashAttention::tiled_attention(
    const std::shared_ptr<ADTensor>& Q,
    const std::shared_ptr<ADTensor>& K,
    const std::shared_ptr<ADTensor>& V,
    int head_idx) {

    int seq_len = Q->val.cols;

    // For small sequences, fall back to standard attention
    if (seq_len <= tile_size) {
        auto Qt = transpose(Q);
        auto scores = matmul(Qt, K);
        float scale = 1.0f / std::sqrt((float)head_dim);
        auto scores_scaled = scalar_mul(scores, scale);

        // ALiBi + causal mask
        Tensor bias_t(seq_len, seq_len);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                if (causal && j > i)
                    bias_t.data[i * seq_len + j] = -std::numeric_limits<float>::infinity();
                else
                    bias_t.data[i * seq_len + j] = -std::abs(j - i) * alibi_slopes[head_idx];
            }
        }
        scores_scaled = add(scores_scaled, make_ad(bias_t));

        // Softmax
        Tensor row_max_t(seq_len, 1);
        for (int i = 0; i < seq_len; ++i) {
            float mx = scores_scaled->val.data[i * seq_len];
            for (int j = 1; j < seq_len; ++j)
                mx = std::max(mx, scores_scaled->val.data[i * seq_len + j]);
            row_max_t.data[i] = mx;
        }
        Tensor ones_r(1, seq_len);
        for (int j = 0; j < seq_len; ++j) ones_r.data[j] = 1.0f;
        auto ones_row = make_ad(ones_r);
        auto max_b = matmul(make_ad(row_max_t), ones_row);
        auto shifted = sub(scores_scaled, max_b);
        auto exp_s = exp_ad(shifted);
        Tensor ones_c(seq_len, 1);
        for (int i = 0; i < seq_len; ++i) ones_c.data[i] = 1.0f;
        auto ones_col = make_ad(ones_c);
        auto denom_c = matmul(exp_s, ones_col);
        auto denom = matmul(denom_c, ones_row);
        auto attn = mul(exp_s, reciprocal(denom));
        return matmul(V, transpose(attn));
    }

    // Tiled computation: process query tiles against all key tiles
    int num_tiles = (seq_len + tile_size - 1) / tile_size;
    std::vector<std::shared_ptr<ADTensor>> output_tiles;

    for (int qi = 0; qi < num_tiles; ++qi) {
        int q_start = qi * tile_size;
        int q_len = std::min(tile_size, seq_len - q_start);

        auto Q_tile = slice(transpose(Q), q_start, q_len); // [q_len x head_dim]
        Q_tile = transpose(Q_tile); // [head_dim x q_len]

        // Accumulate attention for this Q tile across all K tiles
        // Using online softmax approach
        Tensor acc(head_dim, q_len);
        acc.fill(0.0f);
        Tensor row_max(q_len, 1);
        for (int i = 0; i < q_len; ++i) row_max.data[i] = -std::numeric_limits<float>::infinity();
        Tensor row_sum(q_len, 1);
        row_sum.fill(0.0f);

        for (int ki = 0; ki < num_tiles; ++ki) {
            int k_start = ki * tile_size;
            int k_len = std::min(tile_size, seq_len - k_start);

            // Check causal: if all keys are after all queries, skip
            if (causal && k_start > q_start + q_len - 1) break;

            auto K_tile = slice(transpose(K), k_start, k_len);
            K_tile = transpose(K_tile); // [head_dim x k_len]
            auto V_tile = slice(transpose(V), k_start, k_len);
            V_tile = transpose(V_tile); // [head_dim x k_len]

            // scores = Q_tile^T @ K_tile: [q_len x k_len]
            auto Qt = transpose(Q_tile);
            auto tile_scores = matmul(Qt, K_tile);
            float scale = 1.0f / std::sqrt((float)head_dim);
            tile_scores = scalar_mul(tile_scores, scale);

            // Apply ALiBi + causal mask for this tile
            Tensor mask_t(q_len, k_len);
            for (int i = 0; i < q_len; ++i) {
                int gi = q_start + i; // global query index
                for (int j = 0; j < k_len; ++j) {
                    int gj = k_start + j; // global key index
                    if (causal && gj > gi)
                        mask_t.data[i * k_len + j] = -std::numeric_limits<float>::infinity();
                    else
                        mask_t.data[i * k_len + j] = -std::abs(gj - gi) * alibi_slopes[head_idx];
                }
            }
            tile_scores = add(tile_scores, make_ad(mask_t));

            // Online softmax accumulation (on raw values for correctness)
            for (int i = 0; i < q_len; ++i) {
                float old_max = row_max.data[i];
                float new_max = old_max;
                for (int j = 0; j < k_len; ++j) {
                    new_max = std::max(new_max, tile_scores->val.data[i * k_len + j]);
                }
                float correction = std::exp(old_max - new_max);

                // Correct running sum and accumulator
                row_sum.data[i] *= correction;
                for (int d = 0; d < head_dim; ++d) {
                    acc.data[d * q_len + i] *= correction;
                }

                // Add contribution of this tile
                for (int j = 0; j < k_len; ++j) {
                    float w = std::exp(tile_scores->val.data[i * k_len + j] - new_max);
                    row_sum.data[i] += w;
                    for (int d = 0; d < head_dim; ++d) {
                        acc.data[d * q_len + i] += w * V_tile->val.data[d * k_len + j];
                    }
                }
                row_max.data[i] = new_max;
            }
        }

        // Normalize accumulated output
        for (int i = 0; i < q_len; ++i) {
            float s = row_sum.data[i];
            if (s < 1e-9f) s = 1e-9f;
            for (int d = 0; d < head_dim; ++d) {
                acc.data[d * q_len + i] /= s;
            }
        }

        output_tiles.push_back(make_ad(acc));
    }

    // Concatenate output tiles along sequence dimension
    if (output_tiles.size() == 1) return output_tiles[0];

    // Manual concat along cols
    Tensor result(head_dim, seq_len);
    int col_offset = 0;
    for (auto& tile : output_tiles) {
        int tile_cols = tile->val.cols;
        for (int d = 0; d < head_dim; ++d) {
            for (int t = 0; t < tile_cols; ++t) {
                result.data[d * seq_len + col_offset + t] = tile->val.data[d * tile_cols + t];
            }
        }
        col_offset += tile_cols;
    }
    return make_ad(result);
}

std::shared_ptr<ADTensor> ADFlashAttention::forward(
    const std::shared_ptr<ADTensor>& input) {
    auto Q = matmul(W_q, input);
    auto K = matmul(W_k, input);
    auto V = matmul(W_v, input);

    std::vector<std::shared_ptr<ADTensor>> heads;
    heads.reserve(num_heads);

    for (int h = 0; h < num_heads; ++h) {
        int offset = h * head_dim;
        auto Qh = slice(Q, offset, head_dim);
        auto Kh = slice(K, offset, head_dim);
        auto Vh = slice(V, offset, head_dim);
        heads.push_back(tiled_attention(Qh, Kh, Vh, h));
    }

    auto concat_out = concat(heads);
    return matmul(W_o, concat_out);
}
