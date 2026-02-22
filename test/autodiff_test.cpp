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

    // Test sub()
    {
        Tensor at(2, 1), bt(2, 1);
        at.data = {5.0f, 3.0f};
        bt.data = {2.0f, 1.0f};
        auto aa = make_ad(at);
        auto bb = make_ad(bt);
        auto diff = sub(aa, bb);
        auto s = sum(diff);
        s->backward();
        // sub: val = a - b, forward = [3, 2], sum = 5
        assert(fabs(s->val.data[0] - 5.0f) < 1e-6f);
        // grads: d(sum(a-b))/da = [1,1], d/db = [-1,-1]
        for (auto& v : aa->grad.data) assert(fabs(v - 1.0f) < 1e-6f);
        for (auto& v : bb->grad.data) assert(fabs(v - (-1.0f)) < 1e-6f);
        std::cout << "  [PASS] sub()\n";
    }

    // Test tanh_ad()
    {
        Tensor tt(2, 1);
        tt.data = {0.0f, 1.0f};
        auto ta = make_ad(tt);
        auto th = tanh_ad(ta);
        auto s = sum(th);
        s->backward();
        // tanh(0) = 0, tanh(1) ~ 0.7616
        assert(fabs(th->val.data[0] - 0.0f) < 1e-4f);
        assert(fabs(th->val.data[1] - std::tanh(1.0f)) < 1e-4f);
        // grad: 1 - tanh^2(x)
        float g0 = 1.0f - 0.0f;  // 1 - tanh(0)^2 = 1
        float g1 = 1.0f - std::tanh(1.0f) * std::tanh(1.0f);
        assert(fabs(ta->grad.data[0] - g0) < 1e-4f);
        assert(fabs(ta->grad.data[1] - g1) < 1e-4f);
        std::cout << "  [PASS] tanh_ad()\n";
    }

    // Test exp_ad()
    {
        Tensor et(2, 1);
        et.data = {0.0f, 1.0f};
        auto ea = make_ad(et);
        auto ex = exp_ad(ea);
        auto s = sum(ex);
        s->backward();
        assert(fabs(ex->val.data[0] - 1.0f) < 1e-4f);
        assert(fabs(ex->val.data[1] - std::exp(1.0f)) < 1e-4f);
        // grad of exp is exp itself
        assert(fabs(ea->grad.data[0] - 1.0f) < 1e-4f);
        assert(fabs(ea->grad.data[1] - std::exp(1.0f)) < 1e-4f);
        std::cout << "  [PASS] exp_ad()\n";
    }

    // Test log_ad()
    {
        Tensor lt(2, 1);
        lt.data = {1.0f, std::exp(1.0f)};
        auto la = make_ad(lt);
        auto lg = log_ad(la);
        auto s = sum(lg);
        s->backward();
        assert(fabs(lg->val.data[0] - 0.0f) < 1e-4f);
        assert(fabs(lg->val.data[1] - 1.0f) < 1e-4f);
        // grad: 1/x
        assert(fabs(la->grad.data[0] - 1.0f) < 1e-4f);
        assert(fabs(la->grad.data[1] - (1.0f / std::exp(1.0f))) < 1e-4f);
        std::cout << "  [PASS] log_ad()\n";
    }

    // Test sqrt_ad()
    {
        Tensor st(2, 1);
        st.data = {4.0f, 9.0f};
        auto sa = make_ad(st);
        auto sq = sqrt_ad(sa);
        auto s = sum(sq);
        s->backward();
        assert(fabs(sq->val.data[0] - 2.0f) < 1e-4f);
        assert(fabs(sq->val.data[1] - 3.0f) < 1e-4f);
        // grad: 1/(2*sqrt(x))
        assert(fabs(sa->grad.data[0] - 0.25f) < 1e-4f);
        assert(fabs(sa->grad.data[1] - (1.0f / 6.0f)) < 1e-4f);
        std::cout << "  [PASS] sqrt_ad()\n";
    }

    // Test reciprocal()
    {
        Tensor rt(2, 1);
        rt.data = {2.0f, 4.0f};
        auto ra = make_ad(rt);
        auto rc = reciprocal(ra);
        auto s = sum(rc);
        s->backward();
        assert(fabs(rc->val.data[0] - 0.5f) < 1e-4f);
        assert(fabs(rc->val.data[1] - 0.25f) < 1e-4f);
        // grad: -1/x^2
        assert(fabs(ra->grad.data[0] - (-0.25f)) < 1e-4f);
        assert(fabs(ra->grad.data[1] - (-1.0f / 16.0f)) < 1e-4f);
        std::cout << "  [PASS] reciprocal()\n";
    }

    // Test transpose()
    {
        Tensor tt(2, 3);
        tt.data = {1, 2, 3, 4, 5, 6};
        auto ta = make_ad(tt);
        auto tr = transpose(ta);
        assert(tr->val.rows == 3 && tr->val.cols == 2);
        assert(fabs(tr->val(0, 0) - 1.0f) < 1e-6f);
        assert(fabs(tr->val(0, 1) - 4.0f) < 1e-6f);
        assert(fabs(tr->val(1, 0) - 2.0f) < 1e-6f);
        std::cout << "  [PASS] transpose()\n";
    }

    // Test slice()
    {
        Tensor sl(4, 2);
        sl.data = {1, 2, 3, 4, 5, 6, 7, 8};
        auto sla = make_ad(sl);
        auto sliced = slice(sla, 1, 2);  // rows 1-2
        assert(sliced->val.rows == 2 && sliced->val.cols == 2);
        assert(fabs(sliced->val(0, 0) - 3.0f) < 1e-6f);
        assert(fabs(sliced->val(0, 1) - 4.0f) < 1e-6f);
        assert(fabs(sliced->val(1, 0) - 5.0f) < 1e-6f);
        assert(fabs(sliced->val(1, 1) - 6.0f) < 1e-6f);
        std::cout << "  [PASS] slice()\n";
    }

    // Test concat()
    {
        Tensor c1(2, 2), c2(3, 2);
        c1.data = {1, 2, 3, 4};
        c2.data = {5, 6, 7, 8, 9, 10};
        auto ca = make_ad(c1);
        auto cb = make_ad(c2);
        auto cc = concat({ca, cb});
        assert(cc->val.rows == 5 && cc->val.cols == 2);
        assert(fabs(cc->val(0, 0) - 1.0f) < 1e-6f);
        assert(fabs(cc->val(2, 0) - 5.0f) < 1e-6f);
        assert(fabs(cc->val(4, 1) - 10.0f) < 1e-6f);
        std::cout << "  [PASS] concat()\n";
    }

    // Test chained ops backward
    {
        Tensor ct(1, 1);
        ct.data = {3.0f};
        auto ca = make_ad(ct);
        // f = exp(tanh(x)), df/dx = exp(tanh(x)) * (1 - tanh(x)^2)
        auto th = tanh_ad(ca);
        auto ex = exp_ad(th);
        ex->backward();
        float tanh_val = std::tanh(3.0f);
        float expected_grad = std::exp(tanh_val) * (1.0f - tanh_val * tanh_val);
        assert(fabs(ca->grad.data[0] - expected_grad) < 1e-3f);
        std::cout << "  [PASS] Chained ops backward\n";
    }

    // Numerical gradient check with finite differences
    {
        float eps = 1e-3f;
        Tensor nt(1, 1);
        nt.data = {2.0f};

        // f(x) = x^3 = x * x * x, df/dx = 3x^2
        auto na = make_ad(nt);
        auto sq = mul(na, na);
        auto cu = mul(sq, na);
        auto s = sum(cu);
        s->backward();
        float analytical = na->grad.data[0];

        // Finite difference: (f(x+eps) - f(x-eps)) / (2*eps)
        float x_val = 2.0f;
        float f_plus = (x_val + eps) * (x_val + eps) * (x_val + eps);
        float f_minus = (x_val - eps) * (x_val - eps) * (x_val - eps);
        float numerical = (f_plus - f_minus) / (2.0f * eps);
        assert(fabs(analytical - numerical) < 0.1f);
        // Expected: 3 * 4 = 12
        assert(fabs(analytical - 12.0f) < 0.1f);
        std::cout << "  [PASS] Numerical gradient check (finite differences)\n";
    }

    std::cout << "All Autodiff tests passed." << std::endl;
    return 0;
}