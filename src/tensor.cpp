#include "tensor.hpp"

Tensor::Tensor(int size) : size(size), data(size, 0.0f) {}

void Tensor::fill(float value) {
    std::fill(data.begin(), data.end(), value);
}
