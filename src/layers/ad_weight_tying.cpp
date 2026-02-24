#include "layers/ad_weight_tying.hpp"

ADWeightTying::ADWeightTying(const std::shared_ptr<ADTensor>& embedding_weights)
    : shared_weights(embedding_weights) {}

std::shared_ptr<ADTensor> ADWeightTying::forward(const std::shared_ptr<ADTensor>& input) {
    // Embedding weights shape: [vocab_size x embed_dim]
    // We need: [vocab_size x embed_dim] @ [embed_dim x seq_len] = [vocab_size x seq_len]
    // The embedding weights are stored as an AD tensor, so matmul handles gradients
    return matmul(shared_weights, input);
}
