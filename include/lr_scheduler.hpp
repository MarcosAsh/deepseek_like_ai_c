#pragma once
#include <cmath>
#include <algorithm>

// Cosine annealing with linear warmup LR scheduler
class LRScheduler {
public:
    LRScheduler(float base_lr, int warmup_steps, int total_steps, float min_lr = 0.0f)
        : base_lr_(base_lr), warmup_steps_(warmup_steps),
          total_steps_(total_steps), min_lr_(min_lr), step_(0) {}

    // Get the learning rate for the current step and advance
    float get_lr() const {
        if (step_ < warmup_steps_) {
            // Linear warmup
            return base_lr_ * static_cast<float>(step_ + 1) / static_cast<float>(warmup_steps_);
        }
        // Cosine annealing
        int decay_steps = total_steps_ - warmup_steps_;
        if (decay_steps <= 0) return base_lr_;
        int current = step_ - warmup_steps_;
        float progress = static_cast<float>(current) / static_cast<float>(decay_steps);
        progress = std::min(progress, 1.0f);
        return min_lr_ + 0.5f * (base_lr_ - min_lr_) * (1.0f + std::cos(M_PI * progress));
    }

    void step() { ++step_; }
    int current_step() const { return step_; }

private:
    float base_lr_;
    int warmup_steps_;
    int total_steps_;
    float min_lr_;
    int step_;
};
