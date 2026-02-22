#include "loss.hpp"
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

// Helper to compare floats
static bool almost_eq(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // Test 1: binary uniform logits
    {
        std::vector<float> logits = {0.0f, 0.0f};
        int target = 0;
        std::vector<float> grad;
        float loss = softmax_cross_entropy(logits, target, grad);
        // p = [0.5, 0.5], loss = -log(0.5) = ln(2)
        float expected_loss = std::log(2.0f);
        assert(almost_eq(loss, expected_loss));
        // grad = [0.5-1, 0.5-0] = [-0.5, 0.5]
        assert(grad.size() == 2);
        assert(almost_eq(grad[0], -0.5f));
        assert(almost_eq(grad[1],  0.5f));
        // sum of grads should be zero
        float sum_grad = grad[0] + grad[1];
        assert(almost_eq(sum_grad, 0.0f));
    }
    // Test 2: 3-class arbitrary logits
    {
        std::vector<float> logits = {1.0f, 2.0f, 3.0f};
        int target = 2;
        std::vector<float> grad;
        float loss = softmax_cross_entropy(logits, target, grad);
        // manual compute
        float m = 3.0f;
        float e0 = std::exp(1.0f - m);
        float e1 = std::exp(2.0f - m);
        float e2 = std::exp(3.0f - m);
        float s = e0 + e1 + e2;
        float p2 = e2 / s;
        float expected_loss = -std::log(p2);
        assert(almost_eq(loss, expected_loss));
        // Check that grad = p_i - y
        assert(grad.size() == 3);
        for (int i = 0; i < 3; ++i) {
            float pi = std::exp(logits[i] - m) / s;
            float yi = (i == target) ? 1.0f : 0.0f;
            assert(almost_eq(grad[i], pi - yi));
        }
        // grads sum to zero
        float sum_grad = grad[0] + grad[1] + grad[2];
        assert(almost_eq(sum_grad, 0.0f));
    }
    // Test 3: Uniform distribution over many classes
    {
        int n_classes = 10;
        std::vector<float> logits(n_classes, 0.0f);
        int target = 5;
        std::vector<float> grad;
        float loss = softmax_cross_entropy(logits, target, grad);
        // Uniform: p = 1/n_classes, loss = -log(1/n_classes) = log(n_classes)
        float expected_loss = std::log((float)n_classes);
        assert(almost_eq(loss, expected_loss, 1e-5f));
        assert(grad.size() == (size_t)n_classes);
        // Gradients sum to zero
        float sum_grad = 0.0f;
        for (auto& g : grad) sum_grad += g;
        assert(almost_eq(sum_grad, 0.0f, 1e-5f));
        std::cout << "  [PASS] Uniform distribution many classes\n";
    }

    // Test 4: Near-perfect prediction (high logit for target)
    {
        std::vector<float> logits = {-10.0f, -10.0f, 20.0f};
        int target = 2;
        std::vector<float> grad;
        float loss = softmax_cross_entropy(logits, target, grad);
        // Loss should be very close to 0 (confident correct prediction)
        assert(loss < 0.01f);
        // Gradient for target should be close to 0 (p ~ 1, so p - 1 ~ 0)
        assert(std::fabs(grad[2]) < 0.01f);
        std::cout << "  [PASS] Near-perfect prediction\n";
    }

    // Test 5: Many classes (100)
    {
        int n = 100;
        std::vector<float> logits(n);
        for (int i = 0; i < n; ++i) logits[i] = (float)i * 0.1f;
        int target = 50;
        std::vector<float> grad;
        float loss = softmax_cross_entropy(logits, target, grad);
        assert(std::isfinite(loss));
        assert(loss > 0.0f);
        assert(grad.size() == (size_t)n);
        float sum_grad = 0.0f;
        for (auto& g : grad) sum_grad += g;
        assert(almost_eq(sum_grad, 0.0f, 1e-4f));
        std::cout << "  [PASS] Many classes (100)\n";
    }

    std::cout << "All loss tests passed." << std::endl;
    return 0;
}