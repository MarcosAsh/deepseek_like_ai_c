#include "autodiff.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    // z = x^2 + 3y, dz/dx = 2x, dz/dy = 3
    {
        Tensor x_t(1, 1), y_t(1, 1);
        float xv = 2.5f, yv = -1.0f;
        x_t.data[0] = xv;
        y_t.data[0] = yv;
        auto x = make_ad(x_t);
        auto y = make_ad(y_t);
        auto z = add(mul(x, x), scalar_mul(y, 3.0f));
        z->backward();
        assert(fabs(x->grad.data[0] - 2.0f * xv) < 1e-6f);
        assert(fabs(y->grad.data[0] - 3.0f) < 1e-6f);
        assert(fabs(z->val.data[0] - (xv * xv + 3.0f * yv)) < 1e-6f);
    }

    // f = sum(a * b), df/da = b, df/db = a
    {
        Tensor a_t(3, 1), b_t(3, 1);
        a_t.data = {1.0f, 2.0f, 3.0f};
        b_t.data = {4.0f, -1.0f, 0.5f};
        auto a = make_ad(a_t);
        auto b = make_ad(b_t);
        auto f = sum(mul(a, b));
        f->backward();
        assert(fabs(f->val.data[0] - 3.5f) < 1e-6f);
        for (size_t i = 0; i < 3; ++i) {
            assert(fabs(a->grad.data[i] - b_t.data[i]) < 1e-6f);
            assert(fabs(b->grad.data[i] - a_t.data[i]) < 1e-6f);
        }
    }

    // sub: d(sum(a-b))/da = 1, d/db = -1
    {
        Tensor at(2, 1), bt(2, 1);
        at.data = {5.0f, 3.0f};
        bt.data = {2.0f, 1.0f};
        auto a = make_ad(at), b = make_ad(bt);
        auto s = sum(sub(a, b));
        s->backward();
        assert(fabs(s->val.data[0] - 5.0f) < 1e-6f);
        for (auto& v : a->grad.data) assert(fabs(v - 1.0f) < 1e-6f);
        for (auto& v : b->grad.data) assert(fabs(v + 1.0f) < 1e-6f);
    }

    // tanh: forward values and grad = 1 - tanh^2(x)
    {
        Tensor t(2, 1);
        t.data = {0.0f, 1.0f};
        auto a = make_ad(t);
        auto h = tanh_ad(a);
        sum(h)->backward();
        assert(fabs(h->val.data[0]) < 1e-4f);
        assert(fabs(h->val.data[1] - std::tanh(1.0f)) < 1e-4f);
        assert(fabs(a->grad.data[0] - 1.0f) < 1e-4f);
        float th1 = std::tanh(1.0f);
        assert(fabs(a->grad.data[1] - (1.0f - th1 * th1)) < 1e-4f);
    }

    // exp: grad of exp is exp
    {
        Tensor t(2, 1);
        t.data = {0.0f, 1.0f};
        auto a = make_ad(t);
        auto e = exp_ad(a);
        sum(e)->backward();
        assert(fabs(e->val.data[0] - 1.0f) < 1e-4f);
        assert(fabs(e->val.data[1] - std::exp(1.0f)) < 1e-4f);
        assert(fabs(a->grad.data[0] - 1.0f) < 1e-4f);
        assert(fabs(a->grad.data[1] - std::exp(1.0f)) < 1e-4f);
    }

    // log: grad = 1/x
    {
        Tensor t(2, 1);
        t.data = {1.0f, std::exp(1.0f)};
        auto a = make_ad(t);
        auto l = log_ad(a);
        sum(l)->backward();
        assert(fabs(l->val.data[0]) < 1e-4f);
        assert(fabs(l->val.data[1] - 1.0f) < 1e-4f);
        assert(fabs(a->grad.data[0] - 1.0f) < 1e-4f);
        assert(fabs(a->grad.data[1] - (1.0f / std::exp(1.0f))) < 1e-4f);
    }

    // sqrt: grad = 1/(2*sqrt(x))
    {
        Tensor t(2, 1);
        t.data = {4.0f, 9.0f};
        auto a = make_ad(t);
        auto sq = sqrt_ad(a);
        sum(sq)->backward();
        assert(fabs(sq->val.data[0] - 2.0f) < 1e-4f);
        assert(fabs(sq->val.data[1] - 3.0f) < 1e-4f);
        assert(fabs(a->grad.data[0] - 0.25f) < 1e-4f);
        assert(fabs(a->grad.data[1] - (1.0f / 6.0f)) < 1e-4f);
    }

    // reciprocal: grad = -1/x^2
    {
        Tensor t(2, 1);
        t.data = {2.0f, 4.0f};
        auto a = make_ad(t);
        auto r = reciprocal(a);
        sum(r)->backward();
        assert(fabs(r->val.data[0] - 0.5f) < 1e-4f);
        assert(fabs(r->val.data[1] - 0.25f) < 1e-4f);
        assert(fabs(a->grad.data[0] + 0.25f) < 1e-4f);
        assert(fabs(a->grad.data[1] + 1.0f / 16.0f) < 1e-4f);
    }

    // transpose
    {
        Tensor t(2, 3);
        t.data = {1, 2, 3, 4, 5, 6};
        auto tr = transpose(make_ad(t));
        assert(tr->val.rows == 3 && tr->val.cols == 2);
        assert(fabs(tr->val(0, 1) - 4.0f) < 1e-6f);
    }

    // slice rows [1..3) from a 4x2
    {
        Tensor t(4, 2);
        t.data = {1, 2, 3, 4, 5, 6, 7, 8};
        auto sl = slice(make_ad(t), 1, 2);
        assert(sl->val.rows == 2 && sl->val.cols == 2);
        assert(fabs(sl->val(0, 0) - 3.0f) < 1e-6f);
        assert(fabs(sl->val(1, 1) - 6.0f) < 1e-6f);
    }

    // concat vertically
    {
        Tensor c1(2, 2), c2(3, 2);
        c1.data = {1, 2, 3, 4};
        c2.data = {5, 6, 7, 8, 9, 10};
        auto c = concat({make_ad(c1), make_ad(c2)});
        assert(c->val.rows == 5 && c->val.cols == 2);
        assert(fabs(c->val(2, 0) - 5.0f) < 1e-6f);
        assert(fabs(c->val(4, 1) - 10.0f) < 1e-6f);
    }

    // chained ops: f = exp(tanh(x)), df/dx = exp(tanh(x)) * (1 - tanh(x)^2)
    {
        Tensor t(1, 1);
        t.data = {3.0f};
        auto a = make_ad(t);
        exp_ad(tanh_ad(a))->backward();
        float tv = std::tanh(3.0f);
        assert(fabs(a->grad.data[0] - std::exp(tv) * (1.0f - tv * tv)) < 1e-3f);
    }

    // f(x) = x^3, df/dx = 3x^2 = 12 at x=2, verified via finite differences
    {
        Tensor t(1, 1);
        t.data = {2.0f};
        auto a = make_ad(t);
        sum(mul(mul(a, a), a))->backward();
        float analytical = a->grad.data[0];
        assert(fabs(analytical - 12.0f) < 0.1f);

        float eps = 1e-3f, x = 2.0f;
        float numerical = ((x+eps)*(x+eps)*(x+eps) - (x-eps)*(x-eps)*(x-eps)) / (2*eps);
        assert(fabs(analytical - numerical) < 0.1f);
    }

    std::cout << "All Autodiff tests passed." << std::endl;
    return 0;
}
