#include "layers/ad_repetition_penalty.hpp"
#include <cmath>

ADRepetitionPenalty::ADRepetitionPenalty(float penalty_)
    : penalty(penalty_) {}

std::shared_ptr<ADTensor> ADRepetitionPenalty::apply(
    const std::shared_ptr<ADTensor>& logits,
    const std::vector<int>& generated_ids) {

    int vocab_size = logits->val.rows;
    int seq_len = logits->val.cols;

    // Build penalty mask: 1.0 for untouched tokens, penalty-adjusted for repeated tokens
    Tensor penalty_t(vocab_size, seq_len);
    penalty_t.fill(1.0f);

    for (int col = 0; col < seq_len; ++col) {
        for (int token_id : generated_ids) {
            if (token_id >= 0 && token_id < vocab_size) {
                float logit_val = logits->val.data[token_id * seq_len + col];
                if (logit_val > 0.0f) {
                    // Divide positive logits by penalty (reduce probability)
                    penalty_t.data[token_id * seq_len + col] = 1.0f / penalty;
                } else {
                    // Multiply negative logits by penalty (make more negative)
                    penalty_t.data[token_id * seq_len + col] = penalty;
                }
            }
        }
    }

    // Element-wise multiply: logits * penalty_mask
    // This is a non-AD constant mask applied to AD logits
    auto penalty_ad = make_ad(penalty_t);
    return mul(logits, penalty_ad);
}
