#include "layers/embedding.hpp"
#include "layers/positional_encoding.hpp"
#include "layers/linear.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

// Approximate equality for floats
static bool almost_eq(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // Embedding layer test
    {
        int vocab_size = 5;
        int embed_dim = 3;
        Embedding emb(vocab_size, embed_dim);
        // Test output dimensions and repeated tokens
        std::vector<int> tokens = {0, 2, 0, 4};
        Tensor out = emb.forward(tokens);
        assert(out.rows == embed_dim);
        assert(out.cols == static_cast<int>(tokens.size()));
        // Columns 0 and 2 correspond to token 0 -> identical
        for (int i = 0; i < embed_dim; ++i) {
            assert(almost_eq(out.data[i * out.cols + 0],
                             out.data[i * out.cols + 2]));
        }
        // Out-of-range token should throw
        bool caught = false;
        try {
            emb.forward(std::vector<int>{vocab_size});
        } catch (const std::out_of_range&) {
            caught = true;
        }
        assert(caught);
    }

    // PositionalEncoding layer test
    {
        int embed_dim = 4;
        int max_len = 10;
        PositionalEncoding pe(embed_dim, max_len);
        // seq_len = 3
        int seq_len = 3;
        Tensor out = pe.forward(seq_len);
        assert(out.rows == embed_dim);
        assert(out.cols == seq_len);
        // For pos=0, sin(0)=0, cos(0)=1
        assert(almost_eq(out.data[0 * seq_len + 0], 0.0f));
        assert(almost_eq(out.data[1 * seq_len + 0], 1.0f));
        assert(almost_eq(out.data[2 * seq_len + 0], 0.0f));
        assert(almost_eq(out.data[3 * seq_len + 0], 1.0f));
        // For pos=1: d=0->sin(1), d=1->cos(1)
        assert(almost_eq(out.data[0 * seq_len + 1], std::sin(1.0f)));
        assert(almost_eq(out.data[1 * seq_len + 1], std::cos(1.0f)));
        // Exceeding max_len should throw
        bool caught = false;
        try {
            pe.forward(max_len + 1);
        } catch (const std::out_of_range&) {
            caught = true;
        }
        assert(caught);
    }

    // Linear layer test
    {
        int input_dim = 2;
        int output_dim = 3;
        Linear lin(input_dim, output_dim);
        // Override weights and bias for deterministic test
        // weights: 3x2 matrix [ [1,2], [3,4], [5,6] ]
        lin.weights.data = {1,2,
                           3,4,
                           5,6};
        // bias: [0.0, 1.0, -1.0]
        lin.bias.data = {0.0f, 1.0f, -1.0f};
        // Input vector [7,8]
        Tensor inp(input_dim, 1);
        inp.data = {7.0f, 8.0f};
        Tensor out = lin.forward(inp);
        assert(out.rows == output_dim);
        assert(out.cols == 1);
        // Compute expected: W*x + b
        std::vector<float> expected = {
            1*7 + 2*8 + 0.0f,
            3*7 + 4*8 + 1.0f,
            5*7 + 6*8 - 1.0f
        };
        for (int i = 0; i < output_dim; ++i) {
            assert(almost_eq(out.data[i], expected[i]));
        }
    }

    // Batched embedding lookup
    {
        int vocab_size = 10;
        int embed_dim = 4;
        Embedding emb(vocab_size, embed_dim);
        std::vector<int> tokens = {0, 1, 2, 3, 4, 5};
        Tensor out = emb.forward(tokens);
        assert(out.rows == embed_dim);
        assert(out.cols == 6);
        // Each column should be a valid embedding vector
        for (int j = 0; j < 6; ++j) {
            for (int i = 0; i < embed_dim; ++i) {
                assert(std::isfinite(out(i, j)));
            }
        }
        std::cout << "  [PASS] Batched embedding lookup\n";
    }

    // Multi-seq positional encoding
    {
        int embed_dim = 8;
        int max_len = 100;
        PositionalEncoding pe(embed_dim, max_len);
        for (int seq_len : {1, 5, 50, 100}) {
            Tensor out = pe.forward(seq_len);
            assert(out.rows == embed_dim);
            assert(out.cols == seq_len);
            for (auto& v : out.data) {
                assert(std::isfinite(v));
            }
        }
        std::cout << "  [PASS] Multi-seq positional encoding\n";
    }

    // Linear with batch (multi-column input)
    {
        int in_dim = 3, out_dim = 2;
        Linear lin(in_dim, out_dim);
        lin.weights.data = {1, 0, 0, 0, 1, 0};  // First row selects x[0], second selects x[1]
        lin.bias.data = {0.0f, 0.0f};
        Tensor inp(in_dim, 3);
        inp(0, 0) = 1.0f; inp(1, 0) = 2.0f; inp(2, 0) = 3.0f;
        inp(0, 1) = 4.0f; inp(1, 1) = 5.0f; inp(2, 1) = 6.0f;
        inp(0, 2) = 7.0f; inp(1, 2) = 8.0f; inp(2, 2) = 9.0f;
        Tensor out = lin.forward(inp);
        assert(out.rows == out_dim);
        assert(out.cols == 3);
        assert(almost_eq(out(0, 0), 1.0f));
        assert(almost_eq(out(1, 0), 2.0f));
        assert(almost_eq(out(0, 1), 4.0f));
        assert(almost_eq(out(1, 1), 5.0f));
        assert(almost_eq(out(0, 2), 7.0f));
        assert(almost_eq(out(1, 2), 8.0f));
        std::cout << "  [PASS] Linear with batch\n";
    }

    std::cout << "All layer tests passed." << std::endl;
    return 0;
}