#include "layers/layer_norm.hpp"
#include "layers/dropout.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

static bool almost_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // Test 1: LayerNorm forward - output columns have mean~0, variance~1
    {
        int dim = 8;
        int seq_len = 4;
        LayerNorm ln(dim);
        Tensor input(dim, seq_len);
        // Fill with varying values
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < seq_len; ++j)
                input(i, j) = (float)(i * seq_len + j) * 0.5f - 2.0f;

        Tensor out = ln.forward(input);
        assert(out.rows == dim);
        assert(out.cols == seq_len);

        // Check each column has mean~0 and variance~1
        for (int j = 0; j < seq_len; ++j) {
            float mean = 0.0f;
            for (int i = 0; i < dim; ++i)
                mean += out(i, j);
            mean /= dim;
            assert(almost_eq(mean, 0.0f, 0.01f));

            float var = 0.0f;
            for (int i = 0; i < dim; ++i)
                var += (out(i, j) - mean) * (out(i, j) - mean);
            var /= dim;
            assert(almost_eq(var, 1.0f, 0.1f));
        }
        std::cout << "  [PASS] LayerNorm output mean~0, variance~1\n";
    }

    // Test 2: Gamma/beta params affect output
    {
        int dim = 4;
        LayerNorm ln(dim);
        Tensor input(dim, 1);
        input.data = {1.0f, 2.0f, 3.0f, 4.0f};

        Tensor out_default = ln.forward(input);

        // Set gamma=2, beta=1
        ln.gamma.fill(2.0f);
        ln.beta.fill(1.0f);
        Tensor out_scaled = ln.forward(input);

        // Scaled output should differ from default
        bool differs = false;
        for (int i = 0; i < dim; ++i) {
            if (std::fabs(out_scaled.data[i] - out_default.data[i]) > 0.01f)
                differs = true;
        }
        assert(differs);
        std::cout << "  [PASS] Gamma/beta affect output\n";
    }

    // Test 3: Shape preserved for batched input
    {
        int dim = 16;
        int seq_len = 10;
        LayerNorm ln(dim);
        Tensor input(dim, seq_len);
        for (auto& v : input.data) v = 0.5f;
        Tensor out = ln.forward(input);
        assert(out.rows == dim);
        assert(out.cols == seq_len);
        std::cout << "  [PASS] Batched shape preserved\n";
    }

    // Test 4: Dropout training - some elements zeroed, rest rescaled
    {
        float p = 0.5f;
        Dropout dropout(p);
        Tensor input(10, 10);
        input.fill(1.0f);

        Tensor out = dropout.forward(input, true);
        int zeros = 0;
        int rescaled = 0;
        float expected_scale = 1.0f / (1.0f - p);
        for (auto& v : out.data) {
            if (v == 0.0f) zeros++;
            else if (almost_eq(v, expected_scale, 0.01f)) rescaled++;
        }
        // With p=0.5, roughly half should be zero
        assert(zeros > 10);  // at least some zeroed
        assert(rescaled > 10);  // at least some rescaled
        assert(zeros + rescaled == (int)out.data.size());
        std::cout << "  [PASS] Dropout training: zeros + rescaled\n";
    }

    // Test 5: Dropout inference - identity
    {
        float p = 0.5f;
        Dropout dropout(p);
        Tensor input(5, 5);
        for (size_t i = 0; i < input.data.size(); ++i)
            input.data[i] = (float)i * 0.1f;

        Tensor out = dropout.forward(input, false);
        for (size_t i = 0; i < input.data.size(); ++i) {
            assert(almost_eq(out.data[i], input.data[i]));
        }
        std::cout << "  [PASS] Dropout inference: identity\n";
    }

    // Test 6: Dropout edge case p=0 - no dropout
    {
        Dropout dropout0(0.0f);
        Tensor input(4, 4);
        input.fill(2.0f);
        Tensor out = dropout0.forward(input, true);
        for (auto& v : out.data) {
            assert(almost_eq(v, 2.0f));
        }
        std::cout << "  [PASS] Dropout p=0: no dropout\n";
    }

    std::cout << "All LayerNorm/Dropout tests passed." << std::endl;
    return 0;
}
