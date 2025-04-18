#include "layers/dropout.hpp"

Dropout::Dropout(float drop_prob_)
    : drop_prob(drop_prob_),
      gen(std::random_device{}()),
      dist(1.0f - drop_prob_) {}

Tensor Dropout::forward(const Tensor& input, bool training) {
    if (!training || drop_prob <= 0.0f) {
        return input;
    }
    Tensor output(input.rows, input.cols);
    for (size_t i = 0; i < input.data.size(); ++i) {
        bool keep = dist(gen);
        output.data[i] = keep ? input.data[i] / (1.0f - drop_prob) : 0.0f;
    }
    return output;
}