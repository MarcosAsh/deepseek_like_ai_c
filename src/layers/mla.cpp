#include "layers/mla.hpp"
#include <cmath>

MLA::MLA(int input_dim, int hidden_dim, int n_heads, int compress_dim)
    : d_in(input_dim), d_hidden(hidden_dim), n_heads(n_heads), d_compress(compress_dim),
      W_dkv(d_compress, d_in),
      W_uk(d_hidden, d_compress),
      W_uv(d_hidden, d_compress),
      W_q(d_hidden, d_in),
      W_o(d_in, d_hidden) {
    W_dkv.fill(0.1f);
    W_uk.fill(0.2f);
    W_uv.fill(0.3f);
    W_q.fill(0.4f);
    W_o.fill(0.5f);
}

Tensor MLA::forward(const Tensor& input) {
    // Project input to latent: c_KV = W_dkv * h
    Tensor c_kv = W_dkv.matmul(input);

    // Reconstruct key and value
    Tensor key = W_uk.matmul(c_kv);
    Tensor value = W_uv.matmul(c_kv);

    // Generate query
    Tensor query = W_q.matmul(input);

    // Attention score
    float score = query.dot(key) / std::sqrt((float)query.data.size());
    float alpha = 1.0f / (1.0f + std::exp(-score)); // sigmoid for simplicity

    // Attention output: output = alpha * value
    Tensor attention_out(value.rows, value.cols);
    for (size_t i = 0; i < value.data.size(); ++i)
        attention_out.data[i] = alpha * value.data[i];

    // Final projection
    Tensor result = W_o.matmul(attention_out);
    return result;
}
