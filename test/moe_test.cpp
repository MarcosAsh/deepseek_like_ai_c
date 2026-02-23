#include "layers/moe.hpp"
#include "layers/ad_moe.hpp"
#include "autodiff.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    // MoE output shape
    {
        MoE moe(8, 16, 4, 2);
        Tensor input(8, 5);
        for (auto& v : input.data) v = 0.3f;
        float aux = 0.0f;
        Tensor out = moe.forward(input, aux);
        assert(out.rows == 8 && out.cols == 5);
    }

    // MoE aux_loss is non-negative
    {
        MoE moe(8, 16, 4, 2);
        Tensor input(8, 4);
        for (auto& v : input.data) v = 0.5f;
        float aux = 0.0f;
        moe.forward(input, aux);
        assert(aux >= 0.0f);
    }

    // works with different top_k values
    {
        for (int top_k : {1, 2, 4}) {
            MoE moe(4, 8, 4, top_k);
            Tensor input(4, 3);
            for (auto& v : input.data) v = 0.2f;
            float aux = 0.0f;
            Tensor out = moe.forward(input, aux);
            assert(out.rows == 4 && out.cols == 3);
            for (auto& v : out.data) assert(std::isfinite(v));
        }
    }

    // works with different expert counts
    {
        for (int n_experts : {2, 4, 8}) {
            MoE moe(4, 8, n_experts, 2);
            Tensor input(4, 2);
            for (auto& v : input.data) v = 0.1f;
            float aux = 0.0f;
            Tensor out = moe.forward(input, aux);
            assert(out.rows == 4 && out.cols == 2);
        }
    }

    // ADMoE output shape
    {
        clear_parameters();
        ADMoE admoe(8, 16, 4, 2);
        Tensor input_t(8, 4);
        for (auto& v : input_t.data) v = 0.3f;
        auto result = admoe.forward(make_ad(input_t));
        assert(result.output->val.rows == 8);
        assert(result.output->val.cols == 4);
    }

    // ADMoE aux_loss is a non-negative scalar
    {
        clear_parameters();
        ADMoE admoe(8, 16, 4, 2);
        Tensor input_t(8, 3);
        for (auto& v : input_t.data) v = 0.2f;
        auto result = admoe.forward(make_ad(input_t));
        assert(result.aux_loss != nullptr);
        assert(result.aux_loss->val.rows == 1 && result.aux_loss->val.cols == 1);
        assert(result.aux_loss->val.data[0] >= 0.0f);
    }

    // ADMoE backward produces finite gradients
    {
        clear_parameters();
        ADMoE admoe(4, 8, 2, 1);
        Tensor input_t(4, 2);
        for (auto& v : input_t.data) v = 0.5f;
        auto input = make_ad(input_t);
        register_parameter(input);
        auto result = admoe.forward(input);
        sum(add(sum(result.output), result.aux_loss))->backward();
        for (auto& v : input->grad.data)
            assert(std::isfinite(v));
    }

    std::cout << "All MoE tests passed." << std::endl;
    return 0;
}
