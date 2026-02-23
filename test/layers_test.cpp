#include "layers/embedding.hpp"
#include "layers/positional_encoding.hpp"
#include "layers/linear.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

static bool almost_eq(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // embedding: output dims and repeated tokens produce identical columns
    {
        Embedding emb(5, 3);
        std::vector<int> tokens = {0, 2, 0, 4};
        Tensor out = emb.forward(tokens);
        assert(out.rows == 3);
        assert(out.cols == (int)tokens.size());
        for (int i = 0; i < 3; ++i)
            assert(almost_eq(out.data[i * out.cols + 0], out.data[i * out.cols + 2]));

        bool caught = false;
        try { emb.forward({5}); } catch (const std::out_of_range&) { caught = true; }
        assert(caught);
    }

    // positional encoding: sin/cos at pos=0 and pos=1
    {
        PositionalEncoding pe(4, 10);
        Tensor out = pe.forward(3);
        assert(out.rows == 4 && out.cols == 3);
        // pos=0: sin(0)=0, cos(0)=1 alternating per dim pair
        assert(almost_eq(out.data[0*3 + 0], 0.0f));
        assert(almost_eq(out.data[1*3 + 0], 1.0f));
        assert(almost_eq(out.data[2*3 + 0], 0.0f));
        assert(almost_eq(out.data[3*3 + 0], 1.0f));
        // pos=1: sin(1), cos(1) for first dim pair
        assert(almost_eq(out.data[0*3 + 1], std::sin(1.0f)));
        assert(almost_eq(out.data[1*3 + 1], std::cos(1.0f)));

        bool caught = false;
        try { pe.forward(11); } catch (const std::out_of_range&) { caught = true; }
        assert(caught);
    }

    // linear: W*x + b with known weights
    {
        Linear lin(2, 3);
        lin.weights.data = {1,2, 3,4, 5,6};
        lin.bias.data = {0.0f, 1.0f, -1.0f};
        Tensor inp(2, 1);
        inp.data = {7.0f, 8.0f};
        Tensor out = lin.forward(inp);
        assert(out.rows == 3 && out.cols == 1);
        assert(almost_eq(out.data[0], 23.0f));  // 1*7+2*8
        assert(almost_eq(out.data[1], 54.0f));  // 3*7+4*8+1
        assert(almost_eq(out.data[2], 82.0f));  // 5*7+6*8-1
    }

    // embedding: batched lookup
    {
        Embedding emb(10, 4);
        Tensor out = emb.forward({0, 1, 2, 3, 4, 5});
        assert(out.rows == 4 && out.cols == 6);
        for (auto& v : out.data) assert(std::isfinite(v));
    }

    // positional encoding at various lengths
    {
        PositionalEncoding pe(8, 100);
        for (int len : {1, 5, 50, 100}) {
            Tensor out = pe.forward(len);
            assert(out.rows == 8 && out.cols == len);
            for (auto& v : out.data) assert(std::isfinite(v));
        }
    }

    // linear: batched input (selection matrix)
    {
        Linear lin(3, 2);
        lin.weights.data = {1,0,0, 0,1,0}; // selects first two dims
        lin.bias.data = {0.0f, 0.0f};
        Tensor inp(3, 3);
        inp(0,0) = 1; inp(1,0) = 2; inp(2,0) = 3;
        inp(0,1) = 4; inp(1,1) = 5; inp(2,1) = 6;
        inp(0,2) = 7; inp(1,2) = 8; inp(2,2) = 9;
        Tensor out = lin.forward(inp);
        assert(out.rows == 2 && out.cols == 3);
        assert(almost_eq(out(0,0), 1.0f));
        assert(almost_eq(out(1,0), 2.0f));
        assert(almost_eq(out(0,2), 7.0f));
        assert(almost_eq(out(1,2), 8.0f));
    }

    std::cout << "All layer tests passed." << std::endl;
    return 0;
}
