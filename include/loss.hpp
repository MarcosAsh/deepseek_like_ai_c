#pragma once
#include <vector>
// Compute softmax cross-entropy loss and gradient
// logits: unnormalized scores of size V
// target: index of true class [0..V)
// grad: output gradient vector of size V (must be resized by caller)
// Returns: scalar loss = -log(softmax(logits)[target])
float softmax_cross_entropy(const std::vector<float>& logits,
                            int target,
                            std::vector<float>& grad);