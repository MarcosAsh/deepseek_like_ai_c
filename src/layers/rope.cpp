#include "layers/rope.hpp"
#include <cmath>
#include <stdexcept>

RoPE::RoPE(int head_dim, int max_len, float theta)
    : head_dim_(head_dim), max_len_(max_len) {
    if (head_dim % 2 != 0) {
        throw std::invalid_argument("RoPE requires even head_dim");
    }
    int half = head_dim / 2;
    cos_table_.resize(half * max_len);
    sin_table_.resize(half * max_len);

    for (int d = 0; d < half; ++d) {
        float freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
        for (int pos = 0; pos < max_len; ++pos) {
            float angle = pos * freq;
            cos_table_[d * max_len + pos] = std::cos(angle);
            sin_table_[d * max_len + pos] = std::sin(angle);
        }
    }
}

Tensor RoPE::apply(const Tensor& x, int pos_offset) const {
    int dim = x.rows;
    int seq_len = x.cols;
    if (dim != head_dim_) {
        throw std::invalid_argument("RoPE: input dim must match head_dim");
    }
    if (pos_offset + seq_len > max_len_) {
        throw std::out_of_range("RoPE: sequence exceeds max_len");
    }

    int half = head_dim_ / 2;
    Tensor out(dim, seq_len);

    for (int pos = 0; pos < seq_len; ++pos) {
        int abs_pos = pos_offset + pos;
        for (int d = 0; d < half; ++d) {
            float cos_val = cos_table_[d * max_len_ + abs_pos];
            float sin_val = sin_table_[d * max_len_ + abs_pos];

            float x_even = x.data[d * seq_len + pos];
            float x_odd  = x.data[(d + half) * seq_len + pos];

            // Rotate: [x_even, x_odd] -> [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]
            out.data[d * seq_len + pos]          = x_even * cos_val - x_odd * sin_val;
            out.data[(d + half) * seq_len + pos] = x_even * sin_val + x_odd * cos_val;
        }
    }
    return out;
}

std::shared_ptr<ADTensor> RoPE::apply_ad(const std::shared_ptr<ADTensor>& x,
                                           int pos_offset) const {
    Tensor out_val = apply(x->val, pos_offset);
    auto out = std::make_shared<ADTensor>(out_val);

    int half = head_dim_ / 2;
    int seq_len = x->val.cols;

    // Backward: inverse rotation (transpose of rotation matrix = negate sin)
    out->deps.emplace_back(x, [this, x, out, half, seq_len, pos_offset]() {
        for (int pos = 0; pos < seq_len; ++pos) {
            int abs_pos = pos_offset + pos;
            for (int d = 0; d < half; ++d) {
                float cos_val = cos_table_[d * max_len_ + abs_pos];
                float sin_val = sin_table_[d * max_len_ + abs_pos];

                float g_even = out->grad.data[d * seq_len + pos];
                float g_odd  = out->grad.data[(d + half) * seq_len + pos];

                // Inverse rotation for gradient
                x->grad.data[d * seq_len + pos]          += g_even * cos_val + g_odd * sin_val;
                x->grad.data[(d + half) * seq_len + pos] += -g_even * sin_val + g_odd * cos_val;
            }
        }
    });
    return out;
}
