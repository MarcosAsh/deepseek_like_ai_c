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
    // ADLinear: forward produces correct shape
    {
        clear_parameters();
        ADLinear lin(4, 6);
        Tensor input_t(4, 3);
        for (auto& v : input_t.data) v = 0.5f;
        auto out = lin.forward(make_ad(input_t));
        assert(out->val.rows == 6);
        assert(out->val.cols == 3);
    }

    // ADLinear: backward produces finite gradients
    {
        clear_parameters();
        ADLinear lin(3, 2);
        Tensor input_t(3, 2);
        for (auto& v : input_t.data) v = 1.0f;
        auto input = make_ad(input_t);
        register_parameter(input);
        auto s = sum(lin.forward(input));
        s->backward();
        for (auto& v : input->grad.data)
            assert(std::isfinite(v));
    }

    // ADLinear: finite difference gradient check
    {
        clear_parameters();
        int in_dim = 2;
        ADLinear lin(in_dim, 2);
        Tensor input_t(in_dim, 1);
        input_t.data = {1.0f, 2.0f};

        auto input = make_ad(input_t);
        register_parameter(input);
        auto s = sum(lin.forward(input));
        s->backward();
        std::vector<float> analytical(input->grad.data.begin(), input->grad.data.end());

        float eps = 1e-3f;
        for (int i = 0; i < in_dim; ++i) {
            clear_parameters();
            Tensor p(in_dim, 1); p.data = input_t.data; p.data[i] += eps;
            float f_plus = sum(lin.forward(make_ad(p)))->val.data[0];

            clear_parameters();
            Tensor m(in_dim, 1); m.data = input_t.data; m.data[i] -= eps;
            float f_minus = sum(lin.forward(make_ad(m)))->val.data[0];

            float numerical = (f_plus - f_minus) / (2.0f * eps);
            assert(almost_eq(analytical[i], numerical, 0.05f));
        }
    }

    // ADEmbedding: lookup produces [embed x seq_len]
    {
        clear_parameters();
        ADEmbedding emb(10, 8);
        auto out = emb.forward({0, 3, 7, 1});
        assert(out->val.rows == 8);
        assert(out->val.cols == 4);
    }

    // ADEmbedding: same token gives same embedding
    {
        clear_parameters();
        ADEmbedding emb(5, 4);
        auto out = emb.forward({2, 0, 2});
        for (int i = 0; i < 4; ++i)
            assert(almost_eq(out->val(i, 0), out->val(i, 2)));
    }

    // ADEmbedding: get_weights returns [embed x vocab]
    {
        clear_parameters();
        ADEmbedding emb(5, 3);
        auto w = emb.get_weights();
        assert(w->val.rows == 3);
        assert(w->val.cols == 5);
    }

    // ADEmbedding: backward propagates gradients to weight matrix
    {
        clear_parameters();
        ADEmbedding emb(5, 4);
        auto s = sum(emb.forward({1, 3}));
        s->backward();
        auto w = emb.get_weights();
        bool has_grad = false;
        for (auto& v : w->grad.data)
            if (std::fabs(v) > 1e-8f) has_grad = true;
        assert(has_grad);
    }

    // ADLayerNorm: output columns have mean~0
    {
        clear_parameters();
        int dim = 8, seq_len = 3;
        ADLayerNorm ln(dim);
        Tensor input_t(dim, seq_len);
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < seq_len; ++j)
                input_t(i, j) = (float)(i * seq_len + j) * 0.3f - 1.0f;
        auto out = ln.forward(make_ad(input_t));
        assert(out->val.rows == dim);
        assert(out->val.cols == seq_len);
        for (int j = 0; j < seq_len; ++j) {
            float mean = 0.0f;
            for (int i = 0; i < dim; ++i) mean += out->val(i, j);
            mean /= dim;
            assert(almost_eq(mean, 0.0f, 0.1f));
        }
    }

    // ADLayerNorm: backward produces finite gradients
    {
        clear_parameters();
        ADLayerNorm ln(4);
        Tensor input_t(4, 2);
        for (auto& v : input_t.data) v = 1.5f;
        auto input = make_ad(input_t);
        register_parameter(input);
        auto s = sum(ln.forward(input));
        s->backward();
        for (auto& v : input->grad.data)
            assert(std::isfinite(v));
    }

    // ADPositionalEncoding: forward shape
    {
        clear_parameters();
        ADPositionalEncoding pe(8, 64);
        auto out = pe.forward(10);
        assert(out->val.rows == 8);
        assert(out->val.cols == 10);
    }

    // ADPositionalEncoding: exceeding max_len throws
    {
        clear_parameters();
        ADPositionalEncoding pe(4, 8);
        auto out = pe.forward(8);
        assert(out->val.cols == 8);

        bool caught = false;
        try { pe.forward(9); } catch (...) { caught = true; }
        assert(caught);
    }

    // ADPositionalEncoding: backward doesn't crash
    {
        clear_parameters();
        ADPositionalEncoding pe(4, 16);
        auto s = sum(pe.forward(3));
        s->backward();
    }

    std::cout << "All AD layers tests passed." << std::endl;
    return 0;
}
