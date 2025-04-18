#include "layers/layer_norm.hpp"
#include <cmath>

LayerNorm::LayerNorm(int dim_, float eps_)
    : dim(dim_), eps(eps_), gamma(dim_, 1), beta(dim_, 1) {
    gamma.fill(1.0f);
    beta.fill(0.0f);
}

Tensor LayerNorm::forward(const Tensor& input) const {
    int rows = input.rows;
    int cols = input.cols;
    Tensor output(rows, cols);
    for (int j = 0; j < cols; ++j) {
        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < rows; ++i) {
            mean += input.data[i * cols + j];
        }
        mean /= rows;
        // Compute variance
        float var = 0.0f;
        for (int i = 0; i < rows; ++i) {
            float v = input.data[i * cols + j] - mean;
            var += v * v;
        }
        var /= rows;
        float inv_std = 1.0f / std::sqrt(var + eps);
        // Normalize and apply gain & bias
        for (int i = 0; i < rows; ++i) {
            float x = input.data[i * cols + j];
            float norm = (x - mean) * inv_std;
            output.data[i * cols + j] = gamma.data[i] * norm + beta.data[i];
        }
    }
    return output;
}