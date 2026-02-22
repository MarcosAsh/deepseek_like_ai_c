#include "layers/mla.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

static bool almost_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // Test 1: MLA output shape [d_in x seq_len]
    {
        int d_in = 16, d_hidden = 16, n_heads = 2, compress_dim = 8;
        MLA mla(d_in, d_hidden, n_heads, compress_dim);
        Tensor input(d_in, 4);  // seq_len=4
        for (auto& v : input.data) v = 0.1f;
        Tensor out = mla.forward(input);
        assert(out.rows == d_in);
        assert(out.cols == 4);
        std::cout << "  [PASS] MLA output shape\n";
    }

    // Test 2: MLA finite values
    {
        int d_in = 8, d_hidden = 8, n_heads = 2, compress_dim = 4;
        MLA mla(d_in, d_hidden, n_heads, compress_dim);
        Tensor input(d_in, 3);
        for (auto& v : input.data) v = 0.5f;
        Tensor out = mla.forward(input);
        for (auto& v : out.data) {
            assert(std::isfinite(v));
        }
        std::cout << "  [PASS] MLA finite values\n";
    }

    // Test 3: MLA different compress_dim configs
    {
        int d_in = 16, d_hidden = 16, n_heads = 4;
        for (int compress_dim : {4, 8, 16}) {
            MLA mla(d_in, d_hidden, n_heads, compress_dim);
            Tensor input(d_in, 2);
            for (auto& v : input.data) v = 0.2f;
            Tensor out = mla.forward(input);
            assert(out.rows == d_in);
            assert(out.cols == 2);
            for (auto& v : out.data) {
                assert(std::isfinite(v));
            }
        }
        std::cout << "  [PASS] MLA different compress_dim configs\n";
    }

    // Test 4: MLA single token
    {
        int d_in = 8, d_hidden = 8, n_heads = 2, compress_dim = 4;
        MLA mla(d_in, d_hidden, n_heads, compress_dim);
        Tensor input(d_in, 1);  // single token
        for (auto& v : input.data) v = 1.0f;
        Tensor out = mla.forward(input);
        assert(out.rows == d_in);
        assert(out.cols == 1);
        for (auto& v : out.data) {
            assert(std::isfinite(v));
        }
        std::cout << "  [PASS] MLA single token\n";
    }

    // Test 5: MLA longer sequence
    {
        int d_in = 8, d_hidden = 8, n_heads = 2, compress_dim = 4;
        MLA mla(d_in, d_hidden, n_heads, compress_dim);
        Tensor input(d_in, 16);
        for (int i = 0; i < (int)input.data.size(); ++i)
            input.data[i] = (float)(i % 7) * 0.1f;
        Tensor out = mla.forward(input);
        assert(out.rows == d_in);
        assert(out.cols == 16);
        for (auto& v : out.data) {
            assert(std::isfinite(v));
        }
        std::cout << "  [PASS] MLA longer sequence\n";
    }

    std::cout << "All MLA tests passed." << std::endl;
    return 0;
}
