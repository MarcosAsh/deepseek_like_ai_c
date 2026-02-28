#pragma once
#include "autodiff.hpp"
#include <random>

class ADConv2D {
public:
    int in_channels, out_channels, kernel_size, stride, padding;
    std::shared_ptr<ADTensor> weight;  // [out_channels, in_channels, kH, kW]
    std::shared_ptr<ADTensor> bias;    // [out_channels]

    ADConv2D(int in_ch, int out_ch, int kernel_sz, int stride = 1, int padding = 0);

    // Input: [B, Cin, H, W] -> Output: [B, Cout, Hout, Wout]
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& input);

private:
    // im2col: unfold input into column matrix for matmul-based convolution
    static Tensor im2col(const Tensor& input, int B, int C, int H, int W,
                         int kH, int kW, int stride, int padding,
                         int Hout, int Wout);
    static Tensor col2im(const Tensor& col, int B, int C, int H, int W,
                         int kH, int kW, int stride, int padding,
                         int Hout, int Wout);
};
