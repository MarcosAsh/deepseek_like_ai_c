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
    std::cout << "All loss tests passed." << std::endl;
    return 0;
}