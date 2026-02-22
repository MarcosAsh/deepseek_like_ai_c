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
    // ===================== SGD Tests =====================

    // Test 1: SGD step reduces param values toward minimum
    {
        clear_parameters();
        // Create a parameter initialized to [5.0, -3.0]
        Tensor pt(2, 1);
        pt.data = {5.0f, -3.0f};
        auto param = make_ad(pt);
        register_parameter(param);

        float lr = 0.1f;
        SGD sgd(lr);

        // Set gradient manually: grad = [2.0, -1.0]
        param->grad.data = {2.0f, -1.0f};

        sgd.step();

        // After step: param = param - lr * grad
        // [5.0 - 0.1*2.0, -3.0 - 0.1*(-1.0)] = [4.8, -2.9]
        assert(almost_eq(param->val.data[0], 4.8f));
        assert(almost_eq(param->val.data[1], -2.9f));
        std::cout << "  [PASS] SGD step updates params\n";
    }

    // Test 2: SGD zero_grad clears gradients
    {
        clear_parameters();
        Tensor pt(3, 1);
        pt.data = {1.0f, 2.0f, 3.0f};
        auto param = make_ad(pt);
        register_parameter(param);

        SGD sgd(0.01f);

        // Set some gradients
        param->grad.data = {10.0f, 20.0f, 30.0f};
        sgd.zero_grad();

        for (auto& v : param->grad.data) {
            assert(almost_eq(v, 0.0f));
        }
        std::cout << "  [PASS] SGD zero_grad clears\n";
    }

    // Test 3: SGD multiple steps converge toward zero gradient
    {
        clear_parameters();
        Tensor pt(1, 1);
        pt.data = {10.0f};
        auto param = make_ad(pt);
        register_parameter(param);

        float lr = 0.1f;
        SGD sgd(lr);

        // Simulate minimizing f(x) = x^2, grad = 2x
        for (int i = 0; i < 50; ++i) {
            sgd.zero_grad();
            param->grad.data = {2.0f * param->val.data[0]};
            sgd.step();
        }
        // After many steps, should be close to 0
        assert(std::fabs(param->val.data[0]) < 0.1f);
        std::cout << "  [PASS] SGD multiple steps converge\n";
    }

    // ===================== AdamW Tests =====================

    // Test 4: AdamW step with momentum
    {
        clear_parameters();
        Tensor pt(2, 1);
        pt.data = {5.0f, -3.0f};
        auto param = make_ad(pt);
        register_parameter(param);

        float lr = 0.01f;
        AdamW adam(lr);

        // Set gradient
        param->grad.data = {1.0f, -1.0f};
        float val_before_0 = param->val.data[0];
        float val_before_1 = param->val.data[1];

        adam.step();

        // After step, params should move in opposite direction of gradient
        assert(param->val.data[0] < val_before_0);  // positive grad -> decrease
        assert(param->val.data[1] > val_before_1);  // negative grad -> increase
        std::cout << "  [PASS] AdamW step with momentum\n";
    }

    // Test 5: AdamW weight decay shrinks params
    {
        clear_parameters();
        Tensor pt(2, 1);
        pt.data = {10.0f, -10.0f};
        auto param = make_ad(pt);
        register_parameter(param);

        float lr = 0.01f;
        float wd = 0.1f;
        AdamW adam(lr, 0.9f, 0.999f, 1e-8f, wd, 0.0f);

        // Zero gradient - only weight decay should act
        param->grad.data = {0.0f, 0.0f};
        adam.step();

        // Weight decay should shrink parameters toward zero
        assert(std::fabs(param->val.data[0]) < 10.0f);
        assert(std::fabs(param->val.data[1]) < 10.0f);
        std::cout << "  [PASS] AdamW weight decay shrinks params\n";
    }

    // Test 6: AdamW zero_grad
    {
        clear_parameters();
        Tensor pt(3, 1);
        pt.data = {1.0f, 2.0f, 3.0f};
        auto param = make_ad(pt);
        register_parameter(param);

        AdamW adam(0.001f);
        param->grad.data = {5.0f, 10.0f, 15.0f};
        adam.zero_grad();

        for (auto& v : param->grad.data) {
            assert(almost_eq(v, 0.0f));
        }
        std::cout << "  [PASS] AdamW zero_grad\n";
    }

    // Test 7: AdamW convergence on simple function
    {
        clear_parameters();
        Tensor pt(1, 1);
        pt.data = {10.0f};
        auto param = make_ad(pt);
        register_parameter(param);

        AdamW adam(0.1f, 0.9f, 0.999f, 1e-8f, 0.0f, 0.0f);

        // Minimize f(x) = x^2
        for (int i = 0; i < 200; ++i) {
            adam.zero_grad();
            param->grad.data = {2.0f * param->val.data[0]};
            adam.step();
        }
        assert(std::fabs(param->val.data[0]) < 0.5f);
        std::cout << "  [PASS] AdamW convergence\n";
    }

    // Test 8: AdamW gradient clipping
    {
        clear_parameters();
        Tensor pt(2, 1);
        pt.data = {1.0f, 1.0f};
        auto param = make_ad(pt);
        register_parameter(param);

        float clip_norm = 1.0f;
        AdamW adam(0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, clip_norm);

        // Set large gradient
        param->grad.data = {100.0f, 100.0f};
        float val_before = param->val.data[0];
        adam.step();

        // Even with huge gradient, the step should be bounded
        float step_size = std::fabs(param->val.data[0] - val_before);
        assert(step_size < 1.0f);  // Step should be reasonable due to clipping
        std::cout << "  [PASS] AdamW gradient clipping\n";
    }

    std::cout << "All optimizer tests passed." << std::endl;
    return 0;
}
