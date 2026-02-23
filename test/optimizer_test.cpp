#include "optimizer.hpp"
#include "autodiff.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

static bool almost_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // SGD: param = param - lr * grad
    {
        clear_parameters();
        Tensor pt(2, 1);
        pt.data = {5.0f, -3.0f};
        auto param = make_ad(pt);
        register_parameter(param);

        SGD sgd(0.1f);
        param->grad.data = {2.0f, -1.0f};
        sgd.step();

        // [5 - 0.1*2, -3 - 0.1*(-1)] = [4.8, -2.9]
        assert(almost_eq(param->val.data[0], 4.8f));
        assert(almost_eq(param->val.data[1], -2.9f));
    }

    // SGD: zero_grad clears all gradients
    {
        clear_parameters();
        Tensor pt(3, 1);
        pt.data = {1.0f, 2.0f, 3.0f};
        auto param = make_ad(pt);
        register_parameter(param);

        SGD sgd(0.01f);
        param->grad.data = {10.0f, 20.0f, 30.0f};
        sgd.zero_grad();
        for (auto& v : param->grad.data)
            assert(almost_eq(v, 0.0f));
    }

    // SGD: minimizing f(x) = x^2 from x=10 converges near 0
    {
        clear_parameters();
        Tensor pt(1, 1);
        pt.data = {10.0f};
        auto param = make_ad(pt);
        register_parameter(param);

        SGD sgd(0.1f);
        for (int i = 0; i < 50; ++i) {
            sgd.zero_grad();
            param->grad.data = {2.0f * param->val.data[0]};
            sgd.step();
        }
        assert(std::fabs(param->val.data[0]) < 0.1f);
    }

    // AdamW: params move opposite to gradient direction
    {
        clear_parameters();
        Tensor pt(2, 1);
        pt.data = {5.0f, -3.0f};
        auto param = make_ad(pt);
        register_parameter(param);

        AdamW adam(0.01f);
        param->grad.data = {1.0f, -1.0f};
        float before0 = param->val.data[0], before1 = param->val.data[1];
        adam.step();
        assert(param->val.data[0] < before0);
        assert(param->val.data[1] > before1);
    }

    // AdamW: weight decay shrinks params even with zero gradient
    {
        clear_parameters();
        Tensor pt(2, 1);
        pt.data = {10.0f, -10.0f};
        auto param = make_ad(pt);
        register_parameter(param);

        AdamW adam(0.01f, 0.9f, 0.999f, 1e-8f, /*wd=*/0.1f, 0.0f);
        param->grad.data = {0.0f, 0.0f};
        adam.step();
        assert(std::fabs(param->val.data[0]) < 10.0f);
        assert(std::fabs(param->val.data[1]) < 10.0f);
    }

    // AdamW: zero_grad
    {
        clear_parameters();
        Tensor pt(3, 1);
        pt.data = {1.0f, 2.0f, 3.0f};
        auto param = make_ad(pt);
        register_parameter(param);

        AdamW adam(0.001f);
        param->grad.data = {5.0f, 10.0f, 15.0f};
        adam.zero_grad();
        for (auto& v : param->grad.data)
            assert(almost_eq(v, 0.0f));
    }

    // AdamW: minimizing f(x) = x^2 from x=10 converges
    {
        clear_parameters();
        Tensor pt(1, 1);
        pt.data = {10.0f};
        auto param = make_ad(pt);
        register_parameter(param);

        AdamW adam(0.1f, 0.9f, 0.999f, 1e-8f, 0.0f, 0.0f);
        for (int i = 0; i < 200; ++i) {
            adam.zero_grad();
            param->grad.data = {2.0f * param->val.data[0]};
            adam.step();
        }
        assert(std::fabs(param->val.data[0]) < 0.5f);
    }

    // AdamW: gradient clipping bounds the step size
    {
        clear_parameters();
        Tensor pt(2, 1);
        pt.data = {1.0f, 1.0f};
        auto param = make_ad(pt);
        register_parameter(param);

        AdamW adam(0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, /*clip=*/1.0f);
        param->grad.data = {100.0f, 100.0f};
        float before = param->val.data[0];
        adam.step();
        assert(std::fabs(param->val.data[0] - before) < 1.0f);
    }

    std::cout << "All optimizer tests passed." << std::endl;
    return 0;
}
