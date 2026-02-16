#include "layers/ad_layer_norm.hpp"
#include <stdexcept>
#include <cmath>

ADLayerNorm::ADLayerNorm(int dim_, float eps_)
    : dim(dim_), eps(eps_) {
    Tensor tg(dim, 1), tb(dim, 1);
    tg.fill(1.0f);
    tb.fill(0.0f);
    gamma = make_ad(tg); register_parameter(gamma);
    beta  = make_ad(tb); register_parameter(beta);
}

std::shared_ptr<ADTensor> ADLayerNorm::forward(const std::shared_ptr<ADTensor>& x) {
    int rows = dim;
    int cols = x->val.cols;
    // Cache ones tensors when dimensions change
    if (cols != cached_cols) {
        cached_ones_row = Tensor(1, rows);
        cached_ones_row.data.assign(rows, 1.0f);
        cached_ones_col = Tensor(rows, 1);
        cached_ones_col.data.assign(rows, 1.0f);
        cached_ones_cols = Tensor(1, cols);
        cached_ones_cols.data.assign(cols, 1.0f);
        cached_cols = cols;
    }
    // sum over rows: mean = (1/rows) * sum_i x[i, j]
    auto ones1 = make_ad(cached_ones_row);
    auto sum1 = matmul(ones1, x);
    auto mean = scalar_mul(sum1, 1.0f / rows);
    // broadcast mean
    auto ones2 = make_ad(cached_ones_col);
    auto mean_b = matmul(ones2, mean);
    auto x_cent = sub(x, mean_b);
    // variance
    auto x2 = mul(x_cent, x_cent);
    auto sum2 = matmul(ones1, x2);
    auto var = scalar_mul(sum2, 1.0f / rows);
    // add eps
    Tensor eps_t(1, cols);
    eps_t.data.assign(cols, eps);
    auto eps_ad = make_ad(eps_t);
    auto var_eps = add(var, eps_ad);
    auto std = sqrt_ad(var_eps);
    auto inv_std = reciprocal(std);
    auto inv_std_b = matmul(ones2, inv_std);
    auto normed = mul(x_cent, inv_std_b);
    // scale and shift
    auto ones3 = make_ad(cached_ones_cols);
    auto gamma_b = matmul(gamma, ones3);
    auto beta_b  = matmul(beta,  ones3);
    return add(mul(normed, gamma_b), beta_b);
}