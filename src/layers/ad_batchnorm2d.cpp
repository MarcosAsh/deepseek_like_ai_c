#include "layers/ad_batchnorm2d.hpp"
#include <cmath>
#include <stdexcept>

ADBatchNorm2D::ADBatchNorm2D(int nf, float eps, float momentum)
    : num_features(nf), eps(eps), momentum(momentum), training(true),
      running_mean(std::vector<int>{nf}),
      running_var(std::vector<int>{nf}) {
    gamma = std::make_shared<ADTensor>(std::vector<int>{nf});
    beta = std::make_shared<ADTensor>(std::vector<int>{nf});
    gamma->val.fill(1.0f);
    beta->val.fill(0.0f);
    running_mean.fill(0.0f);
    running_var.fill(1.0f);

    register_parameter(gamma);
    register_parameter(beta);
}

std::shared_ptr<ADTensor> ADBatchNorm2D::forward(const std::shared_ptr<ADTensor>& input) {
    auto& s = input->val.shape;
    if (s.size() != 4) throw std::runtime_error("BatchNorm2D: input must be [B,C,H,W]");
    int B = s[0], C = s[1], H = s[2], W = s[3];
    if (C != num_features) throw std::runtime_error("BatchNorm2D: channel mismatch");

    int spatial = H * W;
    int n = B * spatial;

    // Compute per-channel mean and variance
    std::vector<float> mean(C, 0.0f), var(C, 0.0f);
    for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        for (int b = 0; b < B; ++b) {
            for (int hw = 0; hw < spatial; ++hw) {
                sum += input->val.data[(b * C + c) * spatial + hw];
            }
        }
        mean[c] = sum / n;

        float vsum = 0.0f;
        for (int b = 0; b < B; ++b) {
            for (int hw = 0; hw < spatial; ++hw) {
                float d = input->val.data[(b * C + c) * spatial + hw] - mean[c];
                vsum += d * d;
            }
        }
        var[c] = vsum / n;
    }

    // Normalize and apply gamma/beta
    Tensor out_val(s);
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            float inv_std = 1.0f / std::sqrt(var[c] + eps);
            for (int hw = 0; hw < spatial; ++hw) {
                int idx = (b * C + c) * spatial + hw;
                float normalized = (input->val.data[idx] - mean[c]) * inv_std;
                out_val.data[idx] = gamma->val.data[c] * normalized + beta->val.data[c];
            }
        }
    }

    // Update running stats
    if (training) {
        for (int c = 0; c < C; ++c) {
            running_mean.data[c] = (1.0f - momentum) * running_mean.data[c] + momentum * mean[c];
            running_var.data[c] = (1.0f - momentum) * running_var.data[c] + momentum * var[c];
        }
    }

    auto out = std::make_shared<ADTensor>(out_val);
    auto g = gamma;
    auto be = beta;
    float e = eps;

    // Backward for input
    out->deps.emplace_back(input, [input, out, g, mean, var, e, B, C, spatial]() {
        for (int c = 0; c < C; ++c) {
            float inv_std = 1.0f / std::sqrt(var[c] + e);
            int n = B * spatial;

            // Compute sum of grad_out and sum of grad_out * (x - mean)
            float sum_grad = 0.0f, sum_grad_x = 0.0f;
            for (int b = 0; b < B; ++b) {
                for (int hw = 0; hw < spatial; ++hw) {
                    int idx = (b * C + c) * spatial + hw;
                    float go = out->grad.data[idx] * g->val.data[c];
                    sum_grad += go;
                    sum_grad_x += go * (input->val.data[idx] - mean[c]);
                }
            }

            for (int b = 0; b < B; ++b) {
                for (int hw = 0; hw < spatial; ++hw) {
                    int idx = (b * C + c) * spatial + hw;
                    float go = out->grad.data[idx] * g->val.data[c];
                    float xhat = (input->val.data[idx] - mean[c]) * inv_std;
                    input->grad.data[idx] += inv_std * (go - sum_grad / n - xhat * sum_grad_x * inv_std / n);
                }
            }
        }
    });

    // Backward for gamma
    out->deps.emplace_back(g, [input, out, g, mean, var, e, B, C, spatial]() {
        for (int c = 0; c < C; ++c) {
            float inv_std = 1.0f / std::sqrt(var[c] + e);
            float sum = 0.0f;
            for (int b = 0; b < B; ++b) {
                for (int hw = 0; hw < spatial; ++hw) {
                    int idx = (b * C + c) * spatial + hw;
                    float xhat = (input->val.data[idx] - mean[c]) * inv_std;
                    sum += out->grad.data[idx] * xhat;
                }
            }
            g->grad.data[c] += sum;
        }
    });

    // Backward for beta
    out->deps.emplace_back(be, [out, be, B, C, spatial]() {
        for (int c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int b = 0; b < B; ++b) {
                for (int hw = 0; hw < spatial; ++hw) {
                    sum += out->grad.data[(b * C + c) * spatial + hw];
                }
            }
            be->grad.data[c] += sum;
        }
    });

    return out;
}
