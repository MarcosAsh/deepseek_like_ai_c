#pragma once
#include "autodiff.hpp"
#include "layers/ad_linear.hpp"
#include "layers/ad_feed_forward.hpp"
#include <vector>
#include <memory>

class ADMoE {
public:
    ADMoE(int embed_dim, int hidden_dim, int num_experts, int top_k = 2);

    struct MoEOutput {
        std::shared_ptr<ADTensor> output;
        std::shared_ptr<ADTensor> aux_loss;
    };
    MoEOutput forward(const std::shared_ptr<ADTensor>& x);

private:
    int embed_dim;
    int num_experts;
    int top_k;
    std::shared_ptr<ADTensor> gate_W;
    std::shared_ptr<ADTensor> gate_b;
    std::vector<ADFeedForward> experts;
};
