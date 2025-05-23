#pragma once
#include "tensor.hpp"
#include <vector>
#include <cstdint>

namespace quant {
// Flag to enable fake quantization during training
extern bool g_qat_enabled;
// Number of bits for quantization (e.g., 8)
extern int g_qat_bits;

// During training: quantize tensor values to g_qat_bits and dequantize back
void fake_quantize_inplace(Tensor& t);

// After training: quantize tensor to integers and output scale
// out_data will be sized to t.rows * t.cols, each entry in [0, 2^g_qat_bits-1]
void post_training_quantize(const Tensor& t,
                            std::vector<uint8_t>& out_data,
                            float& scale_out);
} // namespace quant