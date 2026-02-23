#include "layers/feed_forward.hpp"
#include "layers/ad_feed_forward.hpp"
#include "layers/linear.hpp"
#include "autodiff.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

static bool almost_eq(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

static float gelu_ref(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
}

int main() {
    // GELU known values
    {
        assert(almost_eq(gelu_ref(0.0f), 0.0f, 1e-6f));
        assert(almost_eq(gelu_ref(1.0f), 0.8412f, 1e-3f));
        assert(almost_eq(gelu_ref(-1.0f), -0.1588f, 1e-3f));
        assert(gelu_ref(5.0f) > 4.9f);
        assert(std::fabs(gelu_ref(-5.0f)) < 0.01f);
    }

    // batched output dimensions
    {
        FeedForward ff(4, 8);
        Tensor input(4, 5);
        for (auto& v : input.data) v = 0.5f;
        Tensor out = ff.forward(input);
        assert(out.rows == 4 && out.cols == 5);
    }

    // batched == per-position (no dropout)
    {
        FeedForward ff(3, 6, 0.0f);
        Tensor input(3, 4);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                input(i, j) = (float)(i * 4 + j) * 0.1f;

        Tensor batched = ff.forward(input, false);
        for (int pos = 0; pos < 4; ++pos) {
            Tensor col(3, 1);
            for (int i = 0; i < 3; ++i) col.data[i] = input(i, pos);
            Tensor single = ff.forward(col, false);
            for (int i = 0; i < 3; ++i)
                assert(almost_eq(batched(i, pos), single.data[i], 1e-4f));
        }
    }

    // inference is deterministic even with dropout configured
    {
        FeedForward ff(4, 8, 0.5f);
        Tensor input(4, 3);
        for (auto& v : input.data) v = 1.0f;
        Tensor out1 = ff.forward(input, false);
        Tensor out2 = ff.forward(input, false);
        for (size_t i = 0; i < out1.data.size(); ++i)
            assert(almost_eq(out1.data[i], out2.data[i]));
    }

    // batched linear
    {
        Linear lin(2, 3);
        lin.weights.data = {1,2, 3,4, 5,6};
        lin.bias.data = {0.0f, 1.0f, -1.0f};
        Tensor inp(2, 2);
        inp(0,0) = 7; inp(1,0) = 8;
        inp(0,1) = 1; inp(1,1) = 2;
        Tensor out = lin.forward(inp);
        assert(out.rows == 3 && out.cols == 2);
        assert(almost_eq(out(0,0), 23.0f));
        assert(almost_eq(out(1,0), 54.0f));
        assert(almost_eq(out(2,0), 82.0f));
        assert(almost_eq(out(0,1), 5.0f));
        assert(almost_eq(out(1,1), 12.0f));
        assert(almost_eq(out(2,1), 16.0f));
    }

    // AD feed forward: output shape
    {
        clear_parameters();
        ADFeedForward adff(4, 8);
        Tensor t(4, 3);
        for (auto& v : t.data) v = 0.5f;
        auto out = adff.forward(make_ad(t));
        assert(out->val.rows == 4 && out->val.cols == 3);
    }

    // AD feed forward: finite difference gradient check
    {
        clear_parameters();
        int dim = 3;
        ADFeedForward adff(dim, 6);
        Tensor t(dim, 1);
        t.data = {0.5f, -0.3f, 0.8f};

        auto input = make_ad(t);
        register_parameter(input);
        sum(adff.forward(input))->backward();
        std::vector<float> analytical(input->grad.data.begin(), input->grad.data.end());

        float eps = 1e-3f;
        for (int i = 0; i < dim; ++i) {
            clear_parameters();
            Tensor p(dim, 1); p.data = t.data; p.data[i] += eps;
            float fp = sum(adff.forward(make_ad(p)))->val.data[0];

            clear_parameters();
            Tensor m(dim, 1); m.data = t.data; m.data[i] -= eps;
            float fm = sum(adff.forward(make_ad(m)))->val.data[0];

            assert(almost_eq(analytical[i], (fp - fm) / (2 * eps), 0.1f));
        }
    }

    std::cout << "All feed-forward tests passed." << std::endl;
    return 0;
}
