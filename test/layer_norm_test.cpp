#include "layers/layer_norm.hpp"
#include "layers/dropout.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

static bool almost_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // normalized output should have mean~0 and variance~1 per column
    {
        int dim = 8, seq_len = 4;
        LayerNorm ln(dim);
        Tensor input(dim, seq_len);
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < seq_len; ++j)
                input(i, j) = (float)(i * seq_len + j) * 0.5f - 2.0f;

        Tensor out = ln.forward(input);
        assert(out.rows == dim);
        assert(out.cols == seq_len);

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
    }

    // changing gamma/beta should change the output
    {
        int dim = 4;
        LayerNorm ln(dim);
        Tensor input(dim, 1);
        input.data = {1.0f, 2.0f, 3.0f, 4.0f};

        Tensor out_default = ln.forward(input);

        ln.gamma.fill(2.0f);
        ln.beta.fill(1.0f);
        Tensor out_scaled = ln.forward(input);

        bool differs = false;
        for (int i = 0; i < dim; ++i) {
            if (std::fabs(out_scaled.data[i] - out_default.data[i]) > 0.01f)
                differs = true;
        }
        assert(differs);
    }

    // shape preserved for batched input
    {
        LayerNorm ln(16);
        Tensor input(16, 10);
        for (auto& v : input.data) v = 0.5f;
        Tensor out = ln.forward(input);
        assert(out.rows == 16);
        assert(out.cols == 10);
    }

    // dropout during training: some elements zeroed, rest rescaled by 1/(1-p)
    {
        float p = 0.5f;
        Dropout dropout(p);
        Tensor input(10, 10);
        input.fill(1.0f);

        Tensor out = dropout.forward(input, true);
        int zeros = 0, rescaled = 0;
        float scale = 1.0f / (1.0f - p);
        for (auto& v : out.data) {
            if (v == 0.0f) zeros++;
            else if (almost_eq(v, scale, 0.01f)) rescaled++;
        }
        assert(zeros > 10);
        assert(rescaled > 10);
        assert(zeros + rescaled == (int)out.data.size());
    }

    // dropout during inference is identity
    {
        Dropout dropout(0.5f);
        Tensor input(5, 5);
        for (size_t i = 0; i < input.data.size(); ++i)
            input.data[i] = (float)i * 0.1f;

        Tensor out = dropout.forward(input, false);
        for (size_t i = 0; i < input.data.size(); ++i)
            assert(almost_eq(out.data[i], input.data[i]));
    }

    // p=0 means no dropout even during training
    {
        Dropout dropout(0.0f);
        Tensor input(4, 4);
        input.fill(2.0f);
        Tensor out = dropout.forward(input, true);
        for (auto& v : out.data)
            assert(almost_eq(v, 2.0f));
    }

    std::cout << "All LayerNorm/Dropout tests passed." << std::endl;
    return 0;
}
