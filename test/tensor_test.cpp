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
    std::cout << "All Tensor tests passed." << std::endl;
    return 0;
}