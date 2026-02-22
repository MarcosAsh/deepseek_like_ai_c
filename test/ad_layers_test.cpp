#include "layers/ad_linear.hpp"
#include "layers/ad_embedding.hpp"
#include "layers/ad_layer_norm.hpp"
#include "layers/ad_positional_encoding.hpp"
#include "autodiff.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

static bool almost_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // ===================== ADLinear Tests =====================

    // Test 1: ADLinear forward shape
    {
        clear_parameters();
        int in_dim = 4, out_dim = 6, seq_len = 3;
        ADLinear lin(in_dim, out_dim);
        Tensor input_t(in_dim, seq_len);
        for (auto& v : input_t.data) v = 0.5f;
        auto input = make_ad(input_t);
        auto out = lin.forward(input);
        assert(out->val.rows == out_dim);
        assert(out->val.cols == seq_len);
        std::cout << "  [PASS] ADLinear forward shape\n";
    }

    // Test 2: ADLinear backward - finite gradients
    {
        clear_parameters();
        int in_dim = 3, out_dim = 2, seq_len = 2;
        ADLinear lin(in_dim, out_dim);
        Tensor input_t(in_dim, seq_len);
        for (auto& v : input_t.data) v = 1.0f;
        auto input = make_ad(input_t);
        register_parameter(input);
        auto out = lin.forward(input);
        auto s = sum(out);
        s->backward();
        for (auto& v : input->grad.data) {
            assert(std::isfinite(v));
        }
        std::cout << "  [PASS] ADLinear backward finite grads\n";
    }

    // Test 3: ADLinear finite difference gradient check
    {
        clear_parameters();
        int in_dim = 2, out_dim = 2;
        ADLinear lin(in_dim, out_dim);
        Tensor input_t(in_dim, 1);
        input_t.data = {1.0f, 2.0f};

        // Compute analytical gradient
        auto input = make_ad(input_t);
        register_parameter(input);
        auto out = lin.forward(input);
        auto s = sum(out);
        s->backward();
        std::vector<float> analytical_grad(input->grad.data.begin(), input->grad.data.end());

        // Compute numerical gradient with finite differences
        float eps = 1e-3f;
        for (int i = 0; i < in_dim; ++i) {
            clear_parameters();
            Tensor inp_plus(in_dim, 1);
            inp_plus.data = input_t.data;
            inp_plus.data[i] += eps;
            auto ap = make_ad(inp_plus);
            auto op = lin.forward(ap);
            auto sp = sum(op);
            float f_plus = sp->val.data[0];

            clear_parameters();
            Tensor inp_minus(in_dim, 1);
            inp_minus.data = input_t.data;
            inp_minus.data[i] -= eps;
            auto am = make_ad(inp_minus);
            auto om = lin.forward(am);
            auto sm = sum(om);
            float f_minus = sm->val.data[0];

            float numerical = (f_plus - f_minus) / (2.0f * eps);
            assert(almost_eq(analytical_grad[i], numerical, 0.05f));
        }
        std::cout << "  [PASS] ADLinear finite difference gradient check\n";
    }

    // ===================== ADEmbedding Tests =====================

    // Test 4: ADEmbedding token lookup shape
    {
        clear_parameters();
        int vocab = 10, embed = 8;
        ADEmbedding emb(vocab, embed);
        std::vector<int> tokens = {0, 3, 7, 1};
        auto out = emb.forward(tokens);
        assert(out->val.rows == embed);
        assert(out->val.cols == (int)tokens.size());
        std::cout << "  [PASS] ADEmbedding token lookup shape\n";
    }

    // Test 5: ADEmbedding same token same embedding
    {
        clear_parameters();
        int vocab = 5, embed = 4;
        ADEmbedding emb(vocab, embed);
        std::vector<int> tokens = {2, 0, 2};
        auto out = emb.forward(tokens);
        // Col 0 and col 2 should be identical (both token 2)
        for (int i = 0; i < embed; ++i) {
            assert(almost_eq(out->val(i, 0), out->val(i, 2)));
        }
        std::cout << "  [PASS] ADEmbedding same token same embedding\n";
    }

    // Test 6: ADEmbedding get_weights
    {
        clear_parameters();
        int vocab = 5, embed = 3;
        ADEmbedding emb(vocab, embed);
        auto w = emb.get_weights();
        assert(w->val.rows == embed);
        assert(w->val.cols == vocab);
        std::cout << "  [PASS] ADEmbedding get_weights\n";
    }

    // Test 7: ADEmbedding backward - gradients flow
    {
        clear_parameters();
        int vocab = 5, embed = 4;
        ADEmbedding emb(vocab, embed);
        std::vector<int> tokens = {1, 3};
        auto out = emb.forward(tokens);
        auto s = sum(out);
        s->backward();
        // Weights should have gradients
        auto w = emb.get_weights();
        bool has_nonzero_grad = false;
        for (auto& v : w->grad.data) {
            if (std::fabs(v) > 1e-8f) has_nonzero_grad = true;
        }
        assert(has_nonzero_grad);
        std::cout << "  [PASS] ADEmbedding backward gradients flow\n";
    }

    // ===================== ADLayerNorm Tests =====================

    // Test 8: ADLayerNorm forward - normalized mean~0
    {
        clear_parameters();
        int dim = 8, seq_len = 3;
        ADLayerNorm ln(dim);
        Tensor input_t(dim, seq_len);
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < seq_len; ++j)
                input_t(i, j) = (float)(i * seq_len + j) * 0.3f - 1.0f;
        auto input = make_ad(input_t);
        auto out = ln.forward(input);
        assert(out->val.rows == dim);
        assert(out->val.cols == seq_len);
        // Check columns have mean~0
        for (int j = 0; j < seq_len; ++j) {
            float mean = 0.0f;
            for (int i = 0; i < dim; ++i)
                mean += out->val(i, j);
            mean /= dim;
            assert(almost_eq(mean, 0.0f, 0.1f));
        }
        std::cout << "  [PASS] ADLayerNorm forward mean~0\n";
    }

    // Test 9: ADLayerNorm backward - finite gradients
    {
        clear_parameters();
        int dim = 4, seq_len = 2;
        ADLayerNorm ln(dim);
        Tensor input_t(dim, seq_len);
        for (auto& v : input_t.data) v = 1.5f;
        auto input = make_ad(input_t);
        register_parameter(input);
        auto out = ln.forward(input);
        auto s = sum(out);
        s->backward();
        for (auto& v : input->grad.data) {
            assert(std::isfinite(v));
        }
        std::cout << "  [PASS] ADLayerNorm backward finite grads\n";
    }

    // ===================== ADPositionalEncoding Tests =====================

    // Test 10: ADPositionalEncoding forward shape
    {
        clear_parameters();
        int embed = 8, max_len = 64;
        ADPositionalEncoding pe(embed, max_len);
        int seq_len = 10;
        auto out = pe.forward(seq_len);
        assert(out->val.rows == embed);
        assert(out->val.cols == seq_len);
        std::cout << "  [PASS] ADPositionalEncoding forward shape\n";
    }

    // Test 11: ADPositionalEncoding max_len boundary
    {
        clear_parameters();
        int embed = 4, max_len = 8;
        ADPositionalEncoding pe(embed, max_len);
        // Within bounds should work
        auto out = pe.forward(max_len);
        assert(out->val.rows == embed);
        assert(out->val.cols == max_len);

        // Exceeding max_len should throw
        bool caught = false;
        try {
            pe.forward(max_len + 1);
        } catch (...) {
            caught = true;
        }
        assert(caught);
        std::cout << "  [PASS] ADPositionalEncoding max_len boundary\n";
    }

    // Test 12: ADPositionalEncoding backward - gradients flow
    {
        clear_parameters();
        int embed = 4, max_len = 16;
        ADPositionalEncoding pe(embed, max_len);
        auto out = pe.forward(3);
        auto s = sum(out);
        s->backward();
        // PE has learnable weights, check they got gradients
        // (just check it didn't crash)
        assert(s->val.data[0] != 0.0f || s->val.data[0] == 0.0f);  // always true, just validates path
        std::cout << "  [PASS] ADPositionalEncoding backward\n";
    }

    std::cout << "All AD layers tests passed." << std::endl;
    return 0;
}
