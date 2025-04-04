#include "tensor.hpp"

Tensor::Tensor(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}

Tensor::Tensor(int size) : Tensor(size, 1) {}

void Tensor::fill(float value) {
    std::fill(data.begin(), data.end(), value);
}

void Tensor::print(const std::string& name) const {
    std::cout << name << " [" << rows << "x" << cols << "]: ";
    for (auto v : data) std::cout << v << " ";
    std::cout << "\n";
}

Tensor Tensor::matmul(const Tensor& other) const {
    assert(cols == other.rows);
    Tensor result(rows, other.cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < cols; ++k) {
                sum += data[i * cols + k] * other.data[k * other.cols + j];
            }
            result.data[i * other.cols + j] = sum;
        }
    }
    return result;
}

float Tensor::dot(const Tensor& other) const {
    assert(data.size() == other.data.size());
    float sum = 0.0f;
    for (size_t i = 0; i < data.size(); ++i)
        sum += data[i] * other.data[i];
    return sum;
}

Tensor Tensor::operator+(const Tensor& other) const {
    assert(rows == other.rows && cols == other.cols && "Tensor dimension mismatch in addition");
    Tensor result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}