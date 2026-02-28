#include "layers/ad_pool2d.hpp"
#include <limits>
#include <stdexcept>

// --- MaxPool2D ---

ADMaxPool2D::ADMaxPool2D(int kernel_sz, int s, int p)
    : kernel_size(kernel_sz), stride(s < 0 ? kernel_sz : s), padding(p) {}

std::shared_ptr<ADTensor> ADMaxPool2D::forward(const std::shared_ptr<ADTensor>& input) {
    auto& s = input->val.shape;
    if (s.size() != 4) throw std::runtime_error("MaxPool2D: input must be [B,C,H,W]");
    int B = s[0], C = s[1], H = s[2], W = s[3];

    int Hout = (H + 2 * padding - kernel_size) / stride + 1;
    int Wout = (W + 2 * padding - kernel_size) / stride + 1;

    Tensor out_val(std::vector<int>{B, C, Hout, Wout});
    // Store max indices for backward
    std::vector<int> max_indices(B * C * Hout * Wout, -1);

    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < Hout; ++oh) {
                for (int ow = 0; ow < Wout; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_idx = -1;
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = oh * stride - padding + kh;
                            int iw = ow * stride - padding + kw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int idx = ((b * C + c) * H + ih) * W + iw;
                                if (input->val.data[idx] > max_val) {
                                    max_val = input->val.data[idx];
                                    max_idx = idx;
                                }
                            }
                        }
                    }
                    int out_idx = ((b * C + c) * Hout + oh) * Wout + ow;
                    out_val.data[out_idx] = max_val;
                    max_indices[out_idx] = max_idx;
                }
            }
        }
    }

    auto out = std::make_shared<ADTensor>(out_val);
    out->deps.emplace_back(input, [input, out, max_indices]() {
        for (size_t i = 0; i < max_indices.size(); ++i) {
            if (max_indices[i] >= 0) {
                input->grad.data[max_indices[i]] += out->grad.data[i];
            }
        }
    });

    return out;
}

// --- AvgPool2D ---

ADAvgPool2D::ADAvgPool2D(int kernel_sz, int s, int p)
    : kernel_size(kernel_sz), stride(s < 0 ? kernel_sz : s), padding(p) {}

std::shared_ptr<ADTensor> ADAvgPool2D::forward(const std::shared_ptr<ADTensor>& input) {
    auto& s = input->val.shape;
    if (s.size() != 4) throw std::runtime_error("AvgPool2D: input must be [B,C,H,W]");
    int B = s[0], C = s[1], H = s[2], W = s[3];

    int Hout = (H + 2 * padding - kernel_size) / stride + 1;
    int Wout = (W + 2 * padding - kernel_size) / stride + 1;

    Tensor out_val(std::vector<int>{B, C, Hout, Wout});

    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < Hout; ++oh) {
                for (int ow = 0; ow < Wout; ++ow) {
                    float sum = 0.0f;
                    int count = 0;
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = oh * stride - padding + kh;
                            int iw = ow * stride - padding + kw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                sum += input->val.data[((b * C + c) * H + ih) * W + iw];
                                count++;
                            }
                        }
                    }
                    int out_idx = ((b * C + c) * Hout + oh) * Wout + ow;
                    out_val.data[out_idx] = count > 0 ? sum / count : 0.0f;
                }
            }
        }
    }

    auto out = std::make_shared<ADTensor>(out_val);
    int ks = kernel_size, str = this->stride, pad = padding;
    out->deps.emplace_back(input, [input, out, B, C, H, W, Hout, Wout, ks, str, pad]() {
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                for (int oh = 0; oh < Hout; ++oh) {
                    for (int ow = 0; ow < Wout; ++ow) {
                        int count = 0;
                        for (int kh = 0; kh < ks; ++kh) {
                            for (int kw = 0; kw < ks; ++kw) {
                                int ih = oh * str - pad + kh;
                                int iw = ow * str - pad + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) count++;
                            }
                        }
                        int out_idx = ((b * C + c) * Hout + oh) * Wout + ow;
                        float grad = out->grad.data[out_idx] / (count > 0 ? count : 1);
                        for (int kh = 0; kh < ks; ++kh) {
                            for (int kw = 0; kw < ks; ++kw) {
                                int ih = oh * str - pad + kh;
                                int iw = ow * str - pad + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    input->grad.data[((b * C + c) * H + ih) * W + iw] += grad;
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    return out;
}
