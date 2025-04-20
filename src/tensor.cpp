#include "tensor.hpp"
#include <algorithm>  // for std::min

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
    // Cache-blocked matrix multiplication for better locality
    const int B = 32;
    for (int ii = 0; ii < rows; ii += B) {
        int iMax = std::min(ii + B, rows);
        for (int kk = 0; kk < cols; kk += B) {
            int kMax = std::min(kk + B, cols);
            for (int jj = 0; jj < other.cols; jj += B) {
                int jMax = std::min(jj + B, other.cols);
                for (int i = ii; i < iMax; ++i) {
                    for (int k = kk; k < kMax; ++k) {
                        float a = (*this)(i, k);
                        for (int j = jj; j < jMax; ++j) {
                            result(i, j) += a * other(k, j);
                        }
                    }
                }
            }
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

Tensor Tensor::transpose() const {
    Tensor result(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.data[j * rows + i] = data[i * cols + j];
        }
    }
    return result;
}