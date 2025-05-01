#include "tensor.hpp"
#if defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#include <cblas.h>
#endif
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
#ifdef USE_ACCELERATE
    // Use Apple's Accelerate framework for multi-threaded SGEMM
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, other.cols, cols,
                1.0f,
                data.data(), cols,
                other.data.data(), other.cols,
                0.0f,
                result.data.data(), other.cols);
    return result;
#else
    const int B = 32;
    for (int ii = 0; ii < rows; ii += B) {
        int iMax = std::min(ii + B, rows);
        for (int kk = 0; kk < cols; kk += B) {
            int kMax = std::min(kk + B, cols);
            for (int jj = 0; jj < other.cols; jj += B) {
                int jMax = std::min(jj + B, other.cols);
                for (int i = ii; i < iMax; ++i) {
                    for (int k = kk; k < kMax; ++k) {
                        const float a = (*this)(i, k);
                        float* res_ptr = &result.data[i * other.cols + jj];
                        const float* other_ptr = &other.data[k * other.cols + jj];
                        int len = jMax - jj;
                        int j = 0;
#if defined(__ARM_NEON__)
                        float32x4_t va = vdupq_n_f32(a);
                        for (; j + 4 <= len; j += 4) {
                            float32x4_t vb = vld1q_f32(other_ptr + j);
                            float32x4_t vr = vld1q_f32(res_ptr + j);
                            // Multiply-accumulate: vr += vb * va
                            vr = vmlaq_f32(vr, vb, va);
                            vst1q_f32(res_ptr + j, vr);
                        }
#endif
                        for (; j < len; ++j) {
                            res_ptr[j] += a * other_ptr[j];
                        }
                    }
                }
            }
        }
    }
    return result;
#endif  // USE_ACCELERATE
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