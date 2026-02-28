#include "layers/ad_flatten.hpp"

ADFlatten::ADFlatten(int start_dim, int end_dim)
    : start_dim(start_dim), end_dim(end_dim) {}

std::shared_ptr<ADTensor> ADFlatten::forward(const std::shared_ptr<ADTensor>& input) {
    return flatten_ad(input, start_dim, end_dim);
}
