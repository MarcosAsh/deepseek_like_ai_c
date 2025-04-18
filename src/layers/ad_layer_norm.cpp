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
    // sum over rows: mean = (1/rows) * sum_i x[i, j]
    // ones1: [1 x rows]
    Tensor ones1_t(1, rows);
    ones1_t.data.assign(rows, 1.0f);
    auto ones1 = make_ad(ones1_t);
    auto sum1 = matmul(ones1, x);          // [1 x cols]
    auto mean = scalar_mul(sum1, 1.0f / rows);
    // broadcast mean: [rows x 1] * [1 x cols]
    Tensor ones2_t(rows, 1);
    ones2_t.data.assign(rows, 1.0f);
    auto ones2 = make_ad(ones2_t);
    auto mean_b = matmul(ones2, mean);     // [rows x cols]
    auto x_cent = sub(x, mean_b);
    // variance: var = (1/rows) * sum over i (x_cent^2)
    auto x2 = mul(x_cent, x_cent);
    auto sum2 = matmul(ones1, x2);         // [1 x cols]
    auto var = scalar_mul(sum2, 1.0f / rows);
    // add eps
    Tensor eps_t(1, cols);
    eps_t.data.assign(cols, eps);
    auto eps_ad = make_ad(eps_t);
    auto var_eps = add(var, eps_ad);       // [1 x cols]
    // stddev and inverse
    auto std = sqrt_ad(var_eps);           // [1 x cols]
    auto inv_std = reciprocal(std);        // [1 x cols]
    auto inv_std_b = matmul(ones2, inv_std); // [rows x cols]
    auto normed = mul(x_cent, inv_std_b);
    // scale and shift: gamma [rows x 1], beta [rows x 1]
    // ones3: [1 x cols]
    Tensor ones3_t(1, cols);
    ones3_t.data.assign(cols, 1.0f);
    auto ones3 = make_ad(ones3_t);
    auto gamma_b = matmul(gamma, ones3);   // [rows x cols]
    auto beta_b  = matmul(beta,  ones3);
    auto out = add(mul(normed, gamma_b), beta_b);
    return out;
}