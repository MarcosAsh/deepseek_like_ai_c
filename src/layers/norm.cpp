#include "layers/norm.hpp"
#include <cmath>
#include <numeric>

Tensor Norm::forward(const Tensor& input) {
    Tensor output(input.rows, input.cols);
    
    // Calculate RMS (root mean square) per row
    for (int i = 0; i < input.rows; ++i) {
        float sum_sq = 0.0f;
        for (int j = 0; j < input.cols; ++j) {
            float val = input.data[i * input.cols + j];
            sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / input.cols + 1e-5f);
        
        // Normalize
        for (int j = 0; j < input.cols; ++j) {
            output.data[i * input.cols + j] = input.data[i * input.cols + j] / rms;
        }
    }
    
    return output;
}