#include "layers/ad_conv2d.hpp"
#include <cmath>
#include <stdexcept>

ADConv2D::ADConv2D(int in_ch, int out_ch, int kernel_sz, int s, int p)
    : in_channels(in_ch), out_channels(out_ch), kernel_size(kernel_sz),
      stride(s), padding(p) {
    // Kaiming/He initialization
    float std_val = std::sqrt(2.0f / (in_ch * kernel_sz * kernel_sz));
    weight = std::make_shared<ADTensor>(std::vector<int>{out_ch, in_ch, kernel_sz, kernel_sz});
    bias = std::make_shared<ADTensor>(std::vector<int>{out_ch});

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, std_val);
    for (auto& v : weight->val.data) v = dist(rng);
    bias->val.fill(0.0f);

    register_parameter(weight);
    register_parameter(bias);
}

Tensor ADConv2D::im2col(const Tensor& input, int B, int C, int H, int W,
                         int kH, int kW, int stride, int padding,
                         int Hout, int Wout) {
    // Output: [B * Hout * Wout, C * kH * kW]
    int col_rows = B * Hout * Wout;
    int col_cols = C * kH * kW;
    Tensor col(col_rows, col_cols);

    for (int b = 0; b < B; ++b) {
        for (int oh = 0; oh < Hout; ++oh) {
            for (int ow = 0; ow < Wout; ++ow) {
                int row = b * Hout * Wout + oh * Wout + ow;
                int col_idx = 0;
                for (int c = 0; c < C; ++c) {
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            int ih = oh * stride - padding + kh;
                            int iw = ow * stride - padding + kw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int input_idx = ((b * C + c) * H + ih) * W + iw;
                                col(row, col_idx) = input.data[input_idx];
                            }
                            col_idx++;
                        }
                    }
                }
            }
        }
    }
    return col;
}

Tensor ADConv2D::col2im(const Tensor& col, int B, int C, int H, int W,
                         int kH, int kW, int stride, int padding,
                         int Hout, int Wout) {
    Tensor result(std::vector<int>{B, C, H, W});

    for (int b = 0; b < B; ++b) {
        for (int oh = 0; oh < Hout; ++oh) {
            for (int ow = 0; ow < Wout; ++ow) {
                int row = b * Hout * Wout + oh * Wout + ow;
                int col_idx = 0;
                for (int c = 0; c < C; ++c) {
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            int ih = oh * stride - padding + kh;
                            int iw = ow * stride - padding + kw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int output_idx = ((b * C + c) * H + ih) * W + iw;
                                result.data[output_idx] += col(row, col_idx);
                            }
                            col_idx++;
                        }
                    }
                }
            }
        }
    }
    return result;
}

std::shared_ptr<ADTensor> ADConv2D::forward(const std::shared_ptr<ADTensor>& input) {
    auto& s = input->val.shape;
    if (s.size() != 4) throw std::runtime_error("Conv2D: input must be [B,C,H,W]");
    int B = s[0], C = s[1], H = s[2], W = s[3];
    if (C != in_channels) throw std::runtime_error("Conv2D: channel mismatch");

    int kH = kernel_size, kW = kernel_size;
    int Hout = (H + 2 * padding - kH) / stride + 1;
    int Wout = (W + 2 * padding - kW) / stride + 1;

    // im2col: [B*Hout*Wout, Cin*kH*kW]
    Tensor col = im2col(input->val, B, C, H, W, kH, kW, stride, padding, Hout, Wout);

    // Reshape weight to [Cout, Cin*kH*kW] then transpose to [Cin*kH*kW, Cout]
    Tensor w_mat(out_channels, in_channels * kH * kW);
    std::copy(weight->val.data.begin(), weight->val.data.end(), w_mat.data.begin());
    Tensor w_matT = w_mat.transpose();

    // matmul: [B*Hout*Wout, Cin*kH*kW] x [Cin*kH*kW, Cout] = [B*Hout*Wout, Cout]
    Tensor out_mat = col.matmul(w_matT);

    // Add bias
    for (int i = 0; i < out_mat.rows; ++i) {
        for (int j = 0; j < out_channels; ++j) {
            out_mat(i, j) += bias->val.data[j];
        }
    }

    // Reshape to [B, Cout, Hout, Wout]
    Tensor out_val(std::vector<int>{B, out_channels, Hout, Wout});
    for (int b = 0; b < B; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < Hout; ++oh) {
                for (int ow = 0; ow < Wout; ++ow) {
                    int mat_row = b * Hout * Wout + oh * Wout + ow;
                    int out_idx = ((b * out_channels + oc) * Hout + oh) * Wout + ow;
                    out_val.data[out_idx] = out_mat(mat_row, oc);
                }
            }
        }
    }

    auto out = std::make_shared<ADTensor>(out_val);

    // Backward for input
    auto self = this;
    auto w = weight;
    auto bi = bias;
    int ic = in_channels, oc = out_channels;
    int str = stride, pad = padding;

    out->deps.emplace_back(input, [input, out, w, ic, oc, kH, kW, str, pad, B, C, H, W, Hout, Wout]() {
        // grad_out shape: [B, Cout, Hout, Wout]
        // Reshape to [B*Hout*Wout, Cout]
        Tensor grad_mat(B * Hout * Wout, oc);
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < oc; ++c) {
                for (int oh = 0; oh < Hout; ++oh) {
                    for (int ow = 0; ow < Wout; ++ow) {
                        int mat_row = b * Hout * Wout + oh * Wout + ow;
                        int grad_idx = ((b * oc + c) * Hout + oh) * Wout + ow;
                        grad_mat(mat_row, c) = out->grad.data[grad_idx];
                    }
                }
            }
        }

        // d_input: grad_mat [B*Hout*Wout, Cout] x weight_mat [Cout, Cin*kH*kW] -> [B*Hout*Wout, Cin*kH*kW]
        Tensor w_mat(oc, ic * kH * kW);
        std::copy(w->val.data.begin(), w->val.data.end(), w_mat.data.begin());
        Tensor d_col = grad_mat.matmul(w_mat);

        // col2im to get input gradient
        Tensor d_input = ADConv2D::col2im(d_col, B, C, H, W, kH, kW, str, pad, Hout, Wout);
        for (size_t i = 0; i < input->grad.data.size(); ++i) {
            input->grad.data[i] += d_input.data[i];
        }
    });

    // Backward for weight
    out->deps.emplace_back(w, [input, out, w, ic, oc, kH, kW, str, pad, B, C, H, W, Hout, Wout]() {
        Tensor grad_mat(B * Hout * Wout, oc);
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < oc; ++c) {
                for (int oh = 0; oh < Hout; ++oh) {
                    for (int ow = 0; ow < Wout; ++ow) {
                        int mat_row = b * Hout * Wout + oh * Wout + ow;
                        int grad_idx = ((b * oc + c) * Hout + oh) * Wout + ow;
                        grad_mat(mat_row, c) = out->grad.data[grad_idx];
                    }
                }
            }
        }

        Tensor col = ADConv2D::im2col(input->val, B, C, H, W, kH, kW, str, pad, Hout, Wout);
        // d_weight: grad_mat^T [Cout, B*Hout*Wout] x col [B*Hout*Wout, Cin*kH*kW] = [Cout, Cin*kH*kW]
        Tensor grad_matT = grad_mat.transpose();
        Tensor dw = grad_matT.matmul(col);
        for (size_t i = 0; i < w->grad.data.size(); ++i) {
            w->grad.data[i] += dw.data[i];
        }
    });

    // Backward for bias
    out->deps.emplace_back(bi, [out, bi, oc, B, Hout, Wout]() {
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < oc; ++c) {
                float sum = 0.0f;
                for (int oh = 0; oh < Hout; ++oh) {
                    for (int ow = 0; ow < Wout; ++ow) {
                        int idx = ((b * oc + c) * Hout + oh) * Wout + ow;
                        sum += out->grad.data[idx];
                    }
                }
                bi->grad.data[c] += sum;
            }
        }
    });

    return out;
}
