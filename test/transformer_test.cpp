#include "transformer.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

static bool almost_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // single block: output shape matches input
    {
        TransformerBlock block(16, 32, 4);
        Tensor input(16, 4);
        for (auto& v : input.data) v = 0.1f;
        Tensor out = block.forward(input);
        assert(out.rows == 16 && out.cols == 4);
        for (auto& v : out.data) assert(std::isfinite(v));
    }

    // multi-layer forward
    {
        Transformer t(3, 16, 32, 4);
        Tensor input(16, 3);
        for (auto& v : input.data) v = 0.5f;
        Tensor out = t.forward(input);
        assert(out.rows == 16 && out.cols == 3);
        for (auto& v : out.data) assert(std::isfinite(v));
    }

    // single token
    {
        Transformer t(2, 8, 16, 2);
        Tensor input(8, 1);
        input.fill(1.0f);
        Tensor out = t.forward(input);
        assert(out.rows == 8 && out.cols == 1);
        for (auto& v : out.data) assert(std::isfinite(v));
    }

    // forward -> clear_cache -> forward doesn't crash
    {
        Transformer t(2, 8, 16, 2);
        Tensor input(8, 2);
        for (auto& v : input.data) v = 0.3f;
        t.forward(input, false, true);
        t.clear_cache();
        Tensor out = t.forward(input);
        assert(out.rows == 8 && out.cols == 2);
    }

    // inference is deterministic
    {
        Transformer t(2, 8, 16, 2);
        Tensor input(8, 3);
        for (int i = 0; i < (int)input.data.size(); ++i)
            input.data[i] = (float)(i % 5) * 0.1f;
        Tensor out1 = t.forward(input, false);
        Tensor out2 = t.forward(input, false);
        for (size_t i = 0; i < out1.data.size(); ++i)
            assert(almost_eq(out1.data[i], out2.data[i]));
    }

    std::cout << "All Transformer tests passed." << std::endl;
    return 0;
}
