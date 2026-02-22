#include "transformer.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

static bool almost_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // Test 1: TransformerBlock output shape
    {
        int input_dim = 16, hidden_dim = 32, n_heads = 4;
        TransformerBlock block(input_dim, hidden_dim, n_heads);
        Tensor input(input_dim, 4);  // seq_len=4
        for (auto& v : input.data) v = 0.1f;
        Tensor out = block.forward(input);
        assert(out.rows == input_dim);
        assert(out.cols == 4);
        for (auto& v : out.data) {
            assert(std::isfinite(v));
        }
        std::cout << "  [PASS] TransformerBlock output shape\n";
    }

    // Test 2: Transformer multi-layer forward
    {
        int num_layers = 3, input_dim = 16, hidden_dim = 32, n_heads = 4;
        Transformer t(num_layers, input_dim, hidden_dim, n_heads);
        Tensor input(input_dim, 3);
        for (auto& v : input.data) v = 0.5f;
        Tensor out = t.forward(input);
        assert(out.rows == input_dim);
        assert(out.cols == 3);
        for (auto& v : out.data) {
            assert(std::isfinite(v));
        }
        std::cout << "  [PASS] Transformer multi-layer forward\n";
    }

    // Test 3: Single token input
    {
        int num_layers = 2, input_dim = 8, hidden_dim = 16, n_heads = 2;
        Transformer t(num_layers, input_dim, hidden_dim, n_heads);
        Tensor input(input_dim, 1);
        input.fill(1.0f);
        Tensor out = t.forward(input);
        assert(out.rows == input_dim);
        assert(out.cols == 1);
        for (auto& v : out.data) {
            assert(std::isfinite(v));
        }
        std::cout << "  [PASS] Transformer single token\n";
    }

    // Test 4: Transformer with cache clear
    {
        int num_layers = 2, input_dim = 8, hidden_dim = 16, n_heads = 2;
        Transformer t(num_layers, input_dim, hidden_dim, n_heads);
        Tensor input(input_dim, 2);
        for (auto& v : input.data) v = 0.3f;

        // Run forward, clear cache, run again - should not crash
        t.forward(input, false, true);
        t.clear_cache();
        Tensor out = t.forward(input);
        assert(out.rows == input_dim);
        assert(out.cols == 2);
        std::cout << "  [PASS] Transformer cache clear\n";
    }

    // Test 5: Deterministic inference
    {
        int num_layers = 2, input_dim = 8, hidden_dim = 16, n_heads = 2;
        Transformer t(num_layers, input_dim, hidden_dim, n_heads);
        Tensor input(input_dim, 3);
        for (int i = 0; i < (int)input.data.size(); ++i)
            input.data[i] = (float)(i % 5) * 0.1f;

        Tensor out1 = t.forward(input, false);
        Tensor out2 = t.forward(input, false);
        for (size_t i = 0; i < out1.data.size(); ++i) {
            assert(almost_eq(out1.data[i], out2.data[i]));
        }
        std::cout << "  [PASS] Deterministic inference\n";
    }

    std::cout << "All Transformer tests passed." << std::endl;
    return 0;
}
