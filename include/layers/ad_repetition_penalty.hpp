#pragma once
#include "autodiff.hpp"
#include <vector>

// Repetition Penalty: penalizes repeated tokens in logits
// For each previously generated token, divides its logit by penalty if positive,
// or multiplies by penalty if negative (following the Ctrl paper approach)
class ADRepetitionPenalty {
public:
    explicit ADRepetitionPenalty(float penalty = 1.2f);

    // Apply penalty to logits based on previously generated token IDs
    // logits: [vocab_size x 1] (single position logits)
    // generated_ids: list of previously generated token IDs
    std::shared_ptr<ADTensor> apply(const std::shared_ptr<ADTensor>& logits,
                                     const std::vector<int>& generated_ids);

    float penalty;
};
