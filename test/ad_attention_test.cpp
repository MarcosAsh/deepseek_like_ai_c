#include "layers/ad_multi_head_attention.hpp"
#include "autodiff.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    // output dimensions match input
    {
        ADMultiHeadAttention mha(8, 2, true);
        Tensor t(8, 4);
        for (auto& v : t.data) v = 0.1f;
        auto out = mha.forward(make_ad(t));
        assert(out->val.rows == 8 && out->val.cols == 4);
    }

    // causal and non-causal both produce valid output
    {
        Tensor t(4, 3);
        t.fill(0.0f);
        t(0, 0) = 1.0f; t(1, 1) = 1.0f; t(2, 2) = 1.0f;

        ADMultiHeadAttention causal(4, 1, true);
        auto out1 = causal.forward(make_ad(t));
        assert(out1->val.rows == 4);

        ADMultiHeadAttention nocausal(4, 1, false);
        auto out2 = nocausal.forward(make_ad(t));
        assert(out2->val.rows == 4);
    }

    // ALiBi: output is finite with 4 heads
    {
        ADMultiHeadAttention mha(8, 4, true);
        Tensor t(8, 2);
        for (auto& v : t.data) v = 0.5f;
        auto out = mha.forward(make_ad(t));
        for (auto& v : out->val.data) assert(std::isfinite(v));
    }

    // backward produces finite gradients
    {
        ADMultiHeadAttention mha(4, 2, true);
        Tensor t(4, 3);
        for (auto& v : t.data) v = 0.3f;
        auto input = make_ad(t);
        register_parameter(input);
        sum(mha.forward(input))->backward();
        for (auto& v : input->grad.data) assert(std::isfinite(v));
    }

    // embed_dim % num_heads != 0 should throw
    {
        bool caught = false;
        try { ADMultiHeadAttention mha(7, 3, true); }
        catch (const std::invalid_argument&) { caught = true; }
        assert(caught);
    }

    // non-causal attention output is finite (softmax weights sum to 1)
    {
        ADMultiHeadAttention mha(4, 1, false);
        Tensor t(4, 3);
        for (auto& v : t.data) v = 1.0f;
        auto out = mha.forward(make_ad(t));
        for (auto& v : out->val.data) assert(std::isfinite(v));
    }

    // single-token causal attention works
    {
        ADMultiHeadAttention mha(4, 1, true);
        Tensor t(4, 1);
        for (auto& v : t.data) v = 0.5f;
        auto out = mha.forward(make_ad(t));
        assert(out->val.rows == 4 && out->val.cols == 1);
        for (auto& v : out->val.data) assert(std::isfinite(v));
    }

    // gradient norm is finite
    {
        ADMultiHeadAttention mha(4, 2, true);
        Tensor t(4, 2);
        for (auto& v : t.data) v = 0.01f;
        auto input = make_ad(t);
        register_parameter(input);
        sum(mha.forward(input))->backward();
        float norm = 0.0f;
        for (auto& v : input->grad.data) norm += v * v;
        assert(std::isfinite(norm));
    }

    std::cout << "All AD attention tests passed." << std::endl;
    return 0;
}
