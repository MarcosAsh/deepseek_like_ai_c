#include "tensor.hpp"
#include <cassert>
#include <iostream>

int main() {
    // Test constructor and dimensions
    Tensor a(2, 3);
    assert(a.rows == 2 && a.cols == 3);
    // Test fill
    a.fill(1.5f);
    for (float v : a.data) {
        assert(v == 1.5f);
    }
    // Test element accessor
    a(0, 1) = 2.5f;
    assert(a(0, 1) == 2.5f);
    // Test transpose
    Tensor b = a.transpose();
    assert(b.rows == 3 && b.cols == 2);
    assert(b(1, 0) == a(0, 1));
    // Test addition
    Tensor c = a + a;
    assert(c.rows == a.rows && c.cols == a.cols);
    for (int i = 0; i < c.rows; ++i) {
        for (int j = 0; j < c.cols; ++j) {
            assert(c(i, j) == a(i, j) + a(i, j));
        }
    }
    // Test dot product
    Tensor x(3, 1);
    Tensor y(3, 1);
    x.data = {1.0f, 2.0f, 3.0f};
    y.data = {4.0f, 5.0f, 6.0f};
    float dp = x.dot(y);
    assert(dp == 1.0f*4.0f + 2.0f*5.0f + 3.0f*6.0f);
    // Test matmul
    Tensor m1(2, 3);
    Tensor m2(3, 2);
    m1.data = {1, 2, 3,
               4, 5, 6};
    m2.data = {7,  8,
               9, 10,
              11, 12};
    Tensor m3 = m1.matmul(m2);
    assert(m3.rows == 2 && m3.cols == 2);
    // Validate values
    assert(m3(0, 0) == 58.0f);  // 1*7 + 2*9 + 3*11
    assert(m3(0, 1) == 64.0f);  // 1*8 + 2*10 + 3*12
    assert(m3(1, 0) == 139.0f); // 4*7 + 5*9 + 6*11
    assert(m3(1, 1) == 154.0f); // 4*8 + 5*10 + 6*12
    // Test zero tensor operations
    {
        Tensor z(3, 3);
        z.fill(0.0f);
        Tensor r = z + z;
        for (auto& v : r.data) assert(v == 0.0f);
        // zero matmul
        Tensor z2(3, 2);
        z2.fill(0.0f);
        Tensor zm = z.matmul(z2);
        for (auto& v : zm.data) assert(v == 0.0f);
        std::cout << "  [PASS] Zero tensor operations\n";
    }

    // Test identity-like behavior
    {
        Tensor id(3, 3);
        id.fill(0.0f);
        id(0, 0) = 1.0f;
        id(1, 1) = 1.0f;
        id(2, 2) = 1.0f;
        Tensor v(3, 1);
        v.data = {2.0f, 3.0f, 4.0f};
        Tensor result = id.matmul(v);
        assert(result(0, 0) == 2.0f);
        assert(result(1, 0) == 3.0f);
        assert(result(2, 0) == 4.0f);
        std::cout << "  [PASS] Identity matmul\n";
    }

    // Test scalar multiplication via repeated addition
    {
        Tensor s(2, 2);
        s.data = {1.0f, 2.0f, 3.0f, 4.0f};
        Tensor doubled = s + s;
        assert(doubled(0, 0) == 2.0f);
        assert(doubled(0, 1) == 4.0f);
        assert(doubled(1, 0) == 6.0f);
        assert(doubled(1, 1) == 8.0f);
        std::cout << "  [PASS] Scalar multiplication via addition\n";
    }

    // Test larger matmul
    {
        int N = 16;
        Tensor big1(N, N);
        Tensor big2(N, N);
        // Fill with simple pattern
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                big1(i, j) = (i == j) ? 1.0f : 0.0f;  // identity
                big2(i, j) = (float)(i * N + j);
            }
        Tensor big_result = big1.matmul(big2);
        // Identity * B = B
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                assert(big_result(i, j) == big2(i, j));
        std::cout << "  [PASS] Large matmul (16x16)\n";
    }

    // Test single-element tensor
    {
        Tensor single(1, 1);
        single.data = {42.0f};
        assert(single(0, 0) == 42.0f);
        Tensor t = single.transpose();
        assert(t.rows == 1 && t.cols == 1);
        assert(t(0, 0) == 42.0f);
        std::cout << "  [PASS] Single element tensor\n";
    }

    std::cout << "All Tensor tests passed." << std::endl;
    return 0;
}