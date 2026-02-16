#include "quantization.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

static bool almost_eq(float a, float b, float eps = 1e-2f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // Test 1: fake_quantize round-trip preserves values approximately
    {
        Tensor t(3, 3);
        // Fill with known values
        t.data = {0.0f, 0.5f, 1.0f,
                  -0.5f, -1.0f, 0.25f,
                  0.75f, -0.75f, 0.125f};
        Tensor original = t;  // copy

        // Enable QAT
        quant::g_qat_enabled = true;
        quant::g_qat_bits = 8;
        quant::fake_quantize_inplace(t);

        // Values should be close to originals (within quantization error)
        for (size_t i = 0; i < t.data.size(); ++i) {
            assert(almost_eq(t.data[i], original.data[i], 0.05f));
        }
        std::cout << "  [PASS] fake_quantize round-trip preserves values\n";
    }

    // Test 2: post_training_quantize output
    {
        Tensor t(2, 2);
        t.data = {0.0f, 1.0f, -1.0f, 0.5f};

        std::vector<uint8_t> qdata;
        float scale;
        quant::g_qat_bits = 8;
        quant::post_training_quantize(t, qdata, scale);

        assert(qdata.size() == 4);
        assert(scale > 0.0f);

        // Verify reconstruction using the actual affine quantization formula:
        // q = round((val - min) * scale), reconstructed = q / scale + min
        float mn = -1.0f;  // min of our data
        for (size_t i = 0; i < qdata.size(); ++i) {
            float reconstructed = float(qdata[i]) / scale + mn;
            assert(almost_eq(reconstructed, t.data[i], 0.02f));
        }
        std::cout << "  [PASS] post_training_quantize produces valid output\n";
    }

    // Test 3: Quantization with different bit widths
    {
        Tensor t(4, 1);
        t.data = {-1.0f, -0.5f, 0.5f, 1.0f};

        for (int bits : {4, 8}) {
            quant::g_qat_bits = bits;
            std::vector<uint8_t> qdata;
            float scale;
            quant::post_training_quantize(t, qdata, scale);
            assert(qdata.size() == 4);

            // All quantized values should be in [0, 2^bits - 1]
            int max_val = (1 << bits) - 1;
            for (auto q : qdata) {
                assert(q <= max_val);
            }
        }
        std::cout << "  [PASS] Quantization works with different bit widths\n";
    }

    // Test 4: Zero tensor
    {
        Tensor t(2, 2);
        t.fill(0.0f);

        quant::g_qat_enabled = true;
        quant::g_qat_bits = 8;
        quant::fake_quantize_inplace(t);
        // All values should remain zero (or very close)
        for (auto v : t.data) {
            assert(almost_eq(v, 0.0f, 0.01f));
        }
        std::cout << "  [PASS] Zero tensor quantization\n";
    }

    // Restore defaults
    quant::g_qat_enabled = false;
    quant::g_qat_bits = 8;

    std::cout << "All quantization tests passed." << std::endl;
    return 0;
}
