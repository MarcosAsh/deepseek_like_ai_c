#pragma once
#include <vector>
#include "allocator.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

class Tensor {
public:
    // Tensor data stored via unified memory allocator (on-chip pool or host fallback)
    std::vector<float, UnifiedMemoryAllocator<float>> data;
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
