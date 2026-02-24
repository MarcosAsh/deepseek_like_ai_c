#include "layers/ad_rmsnorm.hpp"
#include <cmath>

ADRMSNorm::ADRMSNorm(int dim, float eps)
    : dim_(dim), eps_(eps) {
    Tensor tg(dim, 1);
    tg.fill(1.0f);
    gamma = make_ad(tg);
    register_parameter(gamma);
}

std::shared_ptr<ADTensor> ADRMSNorm::forward(const std::shared_ptr<ADTensor>& x) {
    int rows = dim_;
    int cols = x->val.cols;

    if (cols != cached_cols) {
        cached_ones_row = Tensor(1, rows);
        cached_ones_row.data.assign(rows, 1.0f);
        cached_ones_col = Tensor(rows, 1);
        cached_ones_col.data.assign(rows, 1.0f);
        cached_ones_cols = Tensor(1, cols);
        cached_ones_cols.data.assign(cols, 1.0f);
        cached_cols = cols;
    }

    // x^2
    auto x2 = mul(x, x);

    // mean(x^2) per column: [1 x cols]
    auto ones1 = make_ad(cached_ones_row);
    auto sum_x2 = matmul(ones1, x2);
    auto mean_x2 = scalar_mul(sum_x2, 1.0f / rows);

    // add eps
    Tensor eps_t(1, cols);
    eps_t.data.assign(cols, eps_);
    auto eps_ad = make_ad(eps_t);
    auto mean_x2_eps = add(mean_x2, eps_ad);

    // rsqrt = 1 / sqrt(mean(x^2) + eps)
    auto rms = sqrt_ad(mean_x2_eps);
    auto inv_rms = reciprocal(rms);

    // broadcast inv_rms to [rows x cols]
    auto ones2 = make_ad(cached_ones_col);
    auto inv_rms_b = matmul(ones2, inv_rms);

    // normalize: x * inv_rms
    auto normed = mul(x, inv_rms_b);

    // scale by gamma
    auto ones3 = make_ad(cached_ones_cols);
    auto gamma_b = matmul(gamma, ones3);
    return mul(normed, gamma_b);
}
