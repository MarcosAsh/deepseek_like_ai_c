#pragma once
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

class Tensor {
public:
    std::vector<float> data;
    int rows, cols;

    Tensor(int r, int c);
    Tensor(int size);

    void fill(float value);
    void print(const std::string& name = "")const;

    Tensor matmul(const Tensor& other) const;
    // Return a transposed copy of this tensor
    Tensor transpose() const;
    float dot(const Tensor& other) const;

    Tensor operator+(const Tensor& other) const;
    // Fast element accessors
    inline float& operator()(int i, int j) { return data[i * cols + j]; }
    inline const float& operator()(int i, int j) const { return data[i * cols + j]; }
};
