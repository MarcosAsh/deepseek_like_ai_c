#pragma once
#include "autodiff.hpp"

// Weight Tying: shares weights between input embedding and output projection
// The output projection is the transpose of the embedding matrix
// This reduces parameters and often improves generalization
class ADWeightTying {
public:
    // embedding_weights: the shared weight matrix from ADEmbedding [vocab_size x embed_dim]
    // stored as AD tensor for gradient flow
    explicit ADWeightTying(const std::shared_ptr<ADTensor>& embedding_weights);

    // Projects hidden states to vocabulary logits using transposed embedding weights
    // input: [embed_dim x seq_len]
    // output: [vocab_size x seq_len]
    std::shared_ptr<ADTensor> forward(const std::shared_ptr<ADTensor>& input);

private:
    std::shared_ptr<ADTensor> shared_weights;  // reference to embedding weights
};
