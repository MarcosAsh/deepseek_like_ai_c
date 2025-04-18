#include "layers/positional_encoding.hpp"
#include <cmath>
#include <stdexcept>

PositionalEncoding::PositionalEncoding(int embed_dim_, int max_len_)
    : embed_dim(embed_dim_), max_len(max_len_), pe(embed_dim_, max_len_)
{
    // Precompute sinusoidal positional encodings
    for (int d = 0; d < embed_dim; ++d) {
        for (int pos = 0; pos < max_len; ++pos) {
            double angle = pos / std::pow(10000.0, (2 * (d / 2)) / static_cast<double>(embed_dim));
            if (d % 2 == 0) {
                pe.data[d * max_len + pos] = std::sin(angle);
            } else {
                pe.data[d * max_len + pos] = std::cos(angle);
            }
        }
    }
}

Tensor PositionalEncoding::forward(int seq_len) const {
    if (seq_len > max_len) {
        throw std::out_of_range("Sequence length exceeds maximum positional encoding length");
    }
    Tensor out(embed_dim, seq_len);
    for (int d = 0; d < embed_dim; ++d) {
        for (int pos = 0; pos < seq_len; ++pos) {
            out.data[d * seq_len + pos] = pe.data[d * max_len + pos];
        }
    }
    return out;
}