#pragma once
#include "tensor.hpp"
#include "autodiff.hpp"
#include "third_party/json.hpp"
#include <memory>

namespace server {

using json = nlohmann::json;

json tensor_to_json(const Tensor& t, int max_elements = 1000);

json ad_tensor_to_json(const std::shared_ptr<ADTensor>& t,
                       int max_elements = 1000,
                       bool include_grad = false);

Tensor tensor_from_json(const json& j);

} // namespace server
