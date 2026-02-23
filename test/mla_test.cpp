#include "layers/mla.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    // output shape is [d_in x seq_len]
    {
        MLA mla(16, 16, 2, 8);
        Tensor input(16, 4);
        for (auto& v : input.data) v = 0.1f;
        Tensor out = mla.forward(input);
        assert(out.rows == 16 && out.cols == 4);
    }

    // all output values are finite
    {
        MLA mla(8, 8, 2, 4);
        Tensor input(8, 3);
        for (auto& v : input.data) v = 0.5f;
        Tensor out = mla.forward(input);
        for (auto& v : out.data)
            assert(std::isfinite(v));
    }

    // works with different compression dimensions
    {
        for (int cd : {4, 8, 16}) {
            MLA mla(16, 16, 4, cd);
            Tensor input(16, 2);
            for (auto& v : input.data) v = 0.2f;
            Tensor out = mla.forward(input);
            assert(out.rows == 16 && out.cols == 2);
            for (auto& v : out.data) assert(std::isfinite(v));
        }
    }

    // single token
    {
        MLA mla(8, 8, 2, 4);
        Tensor input(8, 1);
        for (auto& v : input.data) v = 1.0f;
        Tensor out = mla.forward(input);
        assert(out.rows == 8 && out.cols == 1);
        for (auto& v : out.data) assert(std::isfinite(v));
    }

    // longer sequence
    {
        MLA mla(8, 8, 2, 4);
        Tensor input(8, 16);
        for (int i = 0; i < (int)input.data.size(); ++i)
            input.data[i] = (float)(i % 7) * 0.1f;
        Tensor out = mla.forward(input);
        assert(out.rows == 8 && out.cols == 16);
        for (auto& v : out.data) assert(std::isfinite(v));
    }

    std::cout << "All MLA tests passed." << std::endl;
    return 0;
}
