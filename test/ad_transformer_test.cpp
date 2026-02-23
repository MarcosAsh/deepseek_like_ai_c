#include "layers/ad_transformer.hpp"
#include "autodiff.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    // block output shape matches input
    {
        clear_parameters();
        ADTransformerBlock block(8, 16, 2);
        Tensor input_t(8, 4);
        for (auto& v : input_t.data) v = 0.1f;
        auto out = block.forward(make_ad(input_t));
        assert(out->val.rows == 8);
        assert(out->val.cols == 4);
    }

    // zero input still produces output (from biases via residual path)
    {
        clear_parameters();
        ADTransformerBlock block(4, 8, 2);
        Tensor input_t(4, 2);
        for (auto& v : input_t.data) v = 0.0f;
        auto out = block.forward(make_ad(input_t));
        assert(out->val.rows == 4);
    }

    // backward produces finite, nonzero gradients on input
    {
        clear_parameters();
        ADTransformerBlock block(8, 16, 2);
        Tensor input_t(8, 3);
        for (auto& v : input_t.data) v = 0.2f;
        auto input = make_ad(input_t);
        register_parameter(input);
        sum(block.forward(input))->backward();
        bool has_nonzero = false;
        for (auto& v : input->grad.data) {
            assert(std::isfinite(v));
            if (std::fabs(v) > 1e-8f) has_nonzero = true;
        }
        assert(has_nonzero);
    }

    // MoE block variant produces non-negative aux_loss
    {
        clear_parameters();
        ADTransformerBlock block(8, 16, 2, true, 4, 2);
        Tensor input_t(8, 4);
        for (auto& v : input_t.data) v = 0.3f;
        std::shared_ptr<ADTensor> aux_loss;
        auto out = block.forward(make_ad(input_t), &aux_loss);
        assert(out->val.rows == 8 && out->val.cols == 4);
        assert(aux_loss != nullptr);
        assert(aux_loss->val.data[0] >= 0.0f);
    }

    // multi-layer transformer preserves shape
    {
        clear_parameters();
        ADTransformer transformer(3, 8, 16, 2);
        Tensor input_t(8, 5);
        for (auto& v : input_t.data) v = 0.1f;
        auto out = transformer.forward(make_ad(input_t));
        assert(out->val.rows == 8);
        assert(out->val.cols == 5);
    }

    // multi-layer gradient flow
    {
        clear_parameters();
        ADTransformer transformer(2, 8, 16, 2);
        Tensor input_t(8, 3);
        for (auto& v : input_t.data) v = 0.15f;
        auto input = make_ad(input_t);
        register_parameter(input);
        sum(transformer.forward(input))->backward();
        bool has_nonzero = false;
        for (auto& v : input->grad.data) {
            assert(std::isfinite(v));
            if (std::fabs(v) > 1e-8f) has_nonzero = true;
        }
        assert(has_nonzero);
    }

    // MoE transformer accumulates aux_loss across layers
    {
        clear_parameters();
        ADTransformer transformer(2, 8, 16, 2, true, 4, 2);
        Tensor input_t(8, 4);
        for (auto& v : input_t.data) v = 0.2f;
        std::shared_ptr<ADTensor> aux_loss;
        auto out = transformer.forward(make_ad(input_t), &aux_loss);
        assert(out->val.rows == 8 && out->val.cols == 4);
        assert(aux_loss != nullptr);
        assert(aux_loss->val.data[0] >= 0.0f);
    }

    std::cout << "All AD transformer tests passed." << std::endl;
    return 0;
}
