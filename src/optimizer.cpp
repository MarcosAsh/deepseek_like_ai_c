#include "optimizer.hpp"

SGD::SGD(float lr_) : lr(lr_) {}

void SGD::step() {
    auto& params = get_parameters();
    for (auto& p : params) {
        Tensor& val = p->val;
        Tensor& grad = p->grad;
        for (size_t i = 0, n = val.data.size(); i < n; ++i) {
            val.data[i] -= lr * grad.data[i];
        }
    }
}

void SGD::zero_grad() {
    auto& params = get_parameters();
    for (auto& p : params) {
        p->grad.fill(0.0f);
    }
}