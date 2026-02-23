#include "loss.hpp"
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

static bool almost_eq(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // binary uniform: p=[0.5, 0.5], loss = ln(2), grads sum to 0
    {
        std::vector<float> logits = {0.0f, 0.0f};
        std::vector<float> grad;
        float loss = softmax_cross_entropy(logits, 0, grad);
        assert(almost_eq(loss, std::log(2.0f)));
        assert(grad.size() == 2);
        assert(almost_eq(grad[0], -0.5f));
        assert(almost_eq(grad[1], 0.5f));
        assert(almost_eq(grad[0] + grad[1], 0.0f));
    }

    // 3-class with known logits
    {
        std::vector<float> logits = {1.0f, 2.0f, 3.0f};
        std::vector<float> grad;
        float loss = softmax_cross_entropy(logits, 2, grad);
        float m = 3.0f;
        float e0 = std::exp(1-m), e1 = std::exp(2-m), e2 = std::exp(3-m);
        float s = e0 + e1 + e2;
        assert(almost_eq(loss, -std::log(e2 / s)));
        for (int i = 0; i < 3; ++i) {
            float pi = std::exp(logits[i] - m) / s;
            float yi = (i == 2) ? 1.0f : 0.0f;
            assert(almost_eq(grad[i], pi - yi));
        }
        assert(almost_eq(grad[0] + grad[1] + grad[2], 0.0f));
    }

    // uniform across 10 classes: loss = log(10)
    {
        int n = 10;
        std::vector<float> logits(n, 0.0f);
        std::vector<float> grad;
        float loss = softmax_cross_entropy(logits, 5, grad);
        assert(almost_eq(loss, std::log((float)n), 1e-5f));
        float gsum = 0.0f;
        for (auto& g : grad) gsum += g;
        assert(almost_eq(gsum, 0.0f, 1e-5f));
    }

    // confident correct prediction: loss near 0
    {
        std::vector<float> logits = {-10.0f, -10.0f, 20.0f};
        std::vector<float> grad;
        float loss = softmax_cross_entropy(logits, 2, grad);
        assert(loss < 0.01f);
        assert(std::fabs(grad[2]) < 0.01f);
    }

    // 100 classes: gradients still sum to 0
    {
        int n = 100;
        std::vector<float> logits(n);
        for (int i = 0; i < n; ++i) logits[i] = (float)i * 0.1f;
        std::vector<float> grad;
        float loss = softmax_cross_entropy(logits, 50, grad);
        assert(std::isfinite(loss) && loss > 0.0f);
        float gsum = 0.0f;
        for (auto& g : grad) gsum += g;
        assert(almost_eq(gsum, 0.0f, 1e-4f));
    }

    std::cout << "All loss tests passed." << std::endl;
    return 0;
}
