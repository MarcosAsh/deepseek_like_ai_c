#pragma once
#include "tensor.hpp"
#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>


// Automatic differentiation tensor
struct ADTensor {
    Tensor val;                      // forward value
    Tensor grad;                     // gradient w.r.t. this value
    // dependencies: (parent, backprop function)
    std::vector<std::pair<std::shared_ptr<ADTensor>, std::function<void()>>> deps;
    // Constructors
    ADTensor(int rows, int cols);
    ADTensor(const Tensor& t);
    // Perform backward pass to compute gradients
    void backward();
};

// Wrap a raw tensor into an ADTensor
std::shared_ptr<ADTensor> make_ad(const Tensor& t);

// Basic operations with gradient tracking
std::shared_ptr<ADTensor> add(const std::shared_ptr<ADTensor>& a,
                              const std::shared_ptr<ADTensor>& b);
std::shared_ptr<ADTensor> mul(const std::shared_ptr<ADTensor>& a,
                              const std::shared_ptr<ADTensor>& b);
std::shared_ptr<ADTensor> scalar_mul(const std::shared_ptr<ADTensor>& a,
                                     float s);
std::shared_ptr<ADTensor> matmul(const std::shared_ptr<ADTensor>& a,
                                 const std::shared_ptr<ADTensor>& b);
std::shared_ptr<ADTensor> tanh_ad(const std::shared_ptr<ADTensor>& a);
std::shared_ptr<ADTensor> exp_ad(const std::shared_ptr<ADTensor>& a);
// Elementwise square root
std::shared_ptr<ADTensor> sqrt_ad(const std::shared_ptr<ADTensor>& a);
// Elementwise reciprocal (1/x)
std::shared_ptr<ADTensor> reciprocal(const std::shared_ptr<ADTensor>& a);
// Elementwise subtraction: a - b
std::shared_ptr<ADTensor> sub(const std::shared_ptr<ADTensor>& a,
                              const std::shared_ptr<ADTensor>& b);
// Sum all elements to produce a scalar tensor [1x1]
std::shared_ptr<ADTensor> sum(const std::shared_ptr<ADTensor>& a);
// Parameter registry for optimizer
// Register a trainable parameter (ADTensor)
void register_parameter(const std::shared_ptr<ADTensor>& p);
// Retrieve all registered parameters
std::vector<std::shared_ptr<ADTensor>>& get_parameters();
// Transpose tensor (swap rows and cols)
std::shared_ptr<ADTensor> transpose(const std::shared_ptr<ADTensor>& a);
// Slice rows [row_offset .. row_offset+row_count) of a
std::shared_ptr<ADTensor> slice(const std::shared_ptr<ADTensor>& a,
                                 int row_offset, int row_count);
// Concatenate tensors vertically (increasing rows) with same cols
std::shared_ptr<ADTensor> concat(const std::vector<std::shared_ptr<ADTensor>>& parts);