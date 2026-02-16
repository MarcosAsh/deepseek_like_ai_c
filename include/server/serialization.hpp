#pragma once
#include "tensor.hpp"
#include "autodiff.hpp"
#include "third_party/json.hpp"
#include <memory>

namespace server {

using json = nlohmann::json;

// Convert Tensor to JSON with optional data truncation
json tensor_to_json(const Tensor& t, int max_elements = 1000);

// Convert ADTensor to JSON, optionally including gradient info
json ad_tensor_to_json(const std::shared_ptr<ADTensor>& t,
                       int max_elements = 1000,
                       bool include_grad = false);

// Reconstruct a Tensor from JSON
Tensor tensor_from_json(const json& j);

} // namespace server
