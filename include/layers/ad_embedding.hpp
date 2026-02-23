#pragma once
#include "autodiff.hpp"
#include <vector>

class ADEmbedding {
public:
    ADEmbedding(int vocab_size, int embed_dim);
    std::shared_ptr<ADTensor> forward(const std::vector<int>& tokens) const;
    const std::shared_ptr<ADTensor>& get_weights() const { return weights; }

private:
    int vocab_size;
    int embed_dim;
    std::shared_ptr<ADTensor> weights; // [embed_dim x vocab_size]
};