#pragma once
#include <vector>
#include "allocator.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

class Tensor {
public:
    int rows, cols;
    std::vector<float, UnifiedMemoryAllocator<float>> data;

    Tensor(int r, int c);
    Tensor(int size);

    void fill(float value);
    void print(const std::string& name = "")const;

    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    float dot(const Tensor& other) const;

    Tensor operator+(const Tensor& other) const;
    inline float& operator()(int i, int j) { return data[i * cols + j]; }
    inline const float& operator()(int i, int j) const { return data[i * cols + j]; }
};
