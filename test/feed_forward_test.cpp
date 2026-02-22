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

// Reference GELU implementation
static float gelu_ref(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
}

int main() {
    // Test 1: GELU correctness
    {
        // Test known values
        assert(almost_eq(gelu_ref(0.0f), 0.0f, 1e-6f));
        // GELU(1) ~ 0.8412
        assert(almost_eq(gelu_ref(1.0f), 0.8412f, 1e-3f));
        // GELU(-1) ~ -0.1588
        assert(almost_eq(gelu_ref(-1.0f), -0.1588f, 1e-3f));
        // Large positive: GELU(x) ~ x for large x
        assert(gelu_ref(5.0f) > 4.9f);
        // Large negative: GELU(x) ~ 0 for large negative x
        assert(std::fabs(gelu_ref(-5.0f)) < 0.01f);
        std::cout << "  [PASS] GELU correctness\n";
    }

    // Test 2: Batched FeedForward output dimensions
    {
        int embed_dim = 4;
        int hidden_dim = 8;
        FeedForward ff(embed_dim, hidden_dim);
        // Test with seq_len > 1 (batched)
        Tensor input(embed_dim, 5);
        for (auto& v : input.data) v = 0.5f;
        Tensor out = ff.forward(input);
        assert(out.rows == embed_dim);
        assert(out.cols == 5);
        std::cout << "  [PASS] Batched FeedForward dimensions\n";
    }

    // Test 3: Batched vs single-position equivalence
    {
        int embed_dim = 3;
        int hidden_dim = 6;
        FeedForward ff(embed_dim, hidden_dim, 0.0f);  // no dropout
        int seq_len = 4;

        // Create input with distinct columns
        Tensor input(embed_dim, seq_len);
        for (int i = 0; i < embed_dim; ++i)
            for (int j = 0; j < seq_len; ++j)
                input(i, j) = (float)(i * seq_len + j) * 0.1f;

        // Batched forward
        Tensor batched_out = ff.forward(input, false);

        // Per-position forward (extract each column, process, compare)
        for (int pos = 0; pos < seq_len; ++pos) {
            Tensor single(embed_dim, 1);
            for (int i = 0; i < embed_dim; ++i)
                single.data[i] = input(i, pos);
            Tensor single_out = ff.forward(single, false);
            for (int i = 0; i < embed_dim; ++i) {
                assert(almost_eq(batched_out(i, pos), single_out.data[i], 1e-4f));
            }
        }
        std::cout << "  [PASS] Batched matches per-position\n";
    }

    // Test 4: Dropout off during inference
    {
        int embed_dim = 4;
        int hidden_dim = 8;
        FeedForward ff(embed_dim, hidden_dim, 0.5f);  // 50% dropout

        Tensor input(embed_dim, 3);
        for (auto& v : input.data) v = 1.0f;

        // Run multiple times with training=false -> should be deterministic
        Tensor out1 = ff.forward(input, false);
        Tensor out2 = ff.forward(input, false);
        for (size_t i = 0; i < out1.data.size(); ++i) {
            assert(almost_eq(out1.data[i], out2.data[i]));
        }
        std::cout << "  [PASS] Dropout off during inference (deterministic)\n";
    }

    // Test 5: Linear layer batched support
    {
        int input_dim = 2;
        int output_dim = 3;
        Linear lin(input_dim, output_dim);
        lin.weights.data = {1, 2, 3, 4, 5, 6};
        lin.bias.data = {0.0f, 1.0f, -1.0f};

        // Multi-column input
        Tensor inp(input_dim, 2);
        inp(0, 0) = 7.0f; inp(1, 0) = 8.0f;  // col 0
        inp(0, 1) = 1.0f; inp(1, 1) = 2.0f;  // col 1
        Tensor out = lin.forward(inp);
        assert(out.rows == output_dim);
        assert(out.cols == 2);
        // Col 0: [1*7+2*8+0, 3*7+4*8+1, 5*7+6*8-1] = [23, 54, 82]
        assert(almost_eq(out(0, 0), 23.0f));
        assert(almost_eq(out(1, 0), 54.0f));
        assert(almost_eq(out(2, 0), 82.0f));
        // Col 1: [1*1+2*2+0, 3*1+4*2+1, 5*1+6*2-1] = [5, 12, 16]
        assert(almost_eq(out(0, 1), 5.0f));
        assert(almost_eq(out(1, 1), 12.0f));
        assert(almost_eq(out(2, 1), 16.0f));
        std::cout << "  [PASS] Linear layer batched support\n";
    }

    // Test 6: AD FeedForward forward shape
    {
        clear_parameters();
        int embed_dim = 4, hidden_dim = 8;
        ADFeedForward adff(embed_dim, hidden_dim);
        Tensor input_t(embed_dim, 3);
        for (auto& v : input_t.data) v = 0.5f;
        auto input = make_ad(input_t);
        auto out = adff.forward(input);
        assert(out->val.rows == embed_dim);
        assert(out->val.cols == 3);
        std::cout << "  [PASS] AD FeedForward forward shape\n";
    }

    // Test 7: AD FeedForward backward with gradient check
    {
        clear_parameters();
        int embed_dim = 3, hidden_dim = 6;
        ADFeedForward adff(embed_dim, hidden_dim);
        Tensor input_t(embed_dim, 1);
        input_t.data = {0.5f, -0.3f, 0.8f};

        // Analytical gradient
        auto input = make_ad(input_t);
        register_parameter(input);
        auto out = adff.forward(input);
        auto s = sum(out);
        s->backward();
        std::vector<float> analytical_grad(input->grad.data.begin(), input->grad.data.end());

        // Finite difference gradient check
        float eps = 1e-3f;
        for (int i = 0; i < embed_dim; ++i) {
            clear_parameters();
            Tensor inp_plus(embed_dim, 1);
            inp_plus.data = input_t.data;
            inp_plus.data[i] += eps;
            auto ap = make_ad(inp_plus);
            auto op = adff.forward(ap);
            auto sp = sum(op);
            float f_plus = sp->val.data[0];

            clear_parameters();
            Tensor inp_minus(embed_dim, 1);
            inp_minus.data = input_t.data;
            inp_minus.data[i] -= eps;
            auto am = make_ad(inp_minus);
            auto om = adff.forward(am);
            auto sm = sum(om);
            float f_minus = sm->val.data[0];

            float numerical = (f_plus - f_minus) / (2.0f * eps);
            assert(almost_eq(analytical_grad[i], numerical, 0.1f));
        }
        std::cout << "  [PASS] AD FeedForward backward gradient check\n";
    }

    std::cout << "All feed-forward tests passed." << std::endl;
    return 0;
}
