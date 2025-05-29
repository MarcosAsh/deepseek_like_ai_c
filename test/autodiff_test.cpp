#include "autodiff.hpp"
#include "tensor.hpp"
#include <cassert>
#include <iostream>

int main() {
    // Scalar test: z = x*x + 3*y
    Tensor x_t(1, 1), y_t(1, 1);
    float x_val = 2.5f;
    float y_val = -1.0f;
    x_t.data[0] = x_val;
    y_t.data[0] = y_val;
    auto x = make_ad(x_t);
    auto y = make_ad(y_t);
    // Compute x^2
    auto x2 = mul(x, x);
    // Compute 3*y
    auto three_y = scalar_mul(y, 3.0f);
    // z = x2 + three_y
    auto z = add(x2, three_y);
    // Backprop
    z->backward();
    // Check gradients: dz/dx = 2*x_val, dz/dy = 3
    float grad_x = x->grad.data[0];
    float grad_y = y->grad.data[0];
    assert(fabs(grad_x - (2.0f * x_val)) < 1e-6f);
    assert(fabs(grad_y - 3.0f) < 1e-6f);
    // Check forward value: z = x^2 + 3*y
    float z_val = z->val.data[0];
    assert(fabs(z_val - (x_val * x_val + 3.0f * y_val)) < 1e-6f);

    // Vector sum test: f = sum(a * b)
    Tensor a_t(3, 1), b_t(3, 1);
    a_t.data = {1.0f, 2.0f, 3.0f};
    b_t.data = {4.0f, -1.0f, 0.5f};
    auto a = make_ad(a_t);
    auto b = make_ad(b_t);
    auto prod = mul(a, b); // elementwise
    auto f = sum(prod);
    f->backward();
    // Forward f = 1*4 + 2*(-1) + 3*0.5 = 4 -2 +1.5 = 3.5
    assert(fabs(f->val.data[0] - 3.5f) < 1e-6f);
    // Gradients: df/da = b, df/db = a
    for (size_t i = 0; i < a->grad.data.size(); ++i) {
        assert(fabs(a->grad.data[i] - b_t.data[i]) < 1e-6f);
        assert(fabs(b->grad.data[i] - a_t.data[i]) < 1e-6f);
    }

    std::cout << "All Autodiff tests passed." << std::endl;
    return 0;
}