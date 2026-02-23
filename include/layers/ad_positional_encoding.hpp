#pragma once
#include "autodiff.hpp"

class ADPositionalEncoding {
public:
    ADPositionalEncoding(int embed_dim, int max_len = 512);
    std::shared_ptr<ADTensor> forward(int seq_len) const;

private:
    int embed_dim;
    int max_len;
    std::shared_ptr<ADTensor> pweights; // learnable [embed_dim x max_len]
};