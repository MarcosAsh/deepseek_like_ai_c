#include "server/serialization.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace server {

static json compute_stats(const std::vector<float, UnifiedMemoryAllocator<float>>& data) {
    if (data.empty()) return {{"min", 0}, {"max", 0}, {"mean", 0}, {"std", 0}};

    float mn = data[0], mx = data[0], sum = 0;
    for (float v : data) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum += v;
    }
    float mean = sum / static_cast<float>(data.size());

    float var_sum = 0;
    for (float v : data) {
        float d = v - mean;
        var_sum += d * d;
    }
    float std_dev = std::sqrt(var_sum / static_cast<float>(data.size()));

    return {{"min", mn}, {"max", mx}, {"mean", mean}, {"std", std_dev}};
}

json tensor_to_json(const Tensor& t, int max_elements) {
    json j;
    j["shape"] = {t.rows, t.cols};
    j["stats"] = compute_stats(t.data);

    int total = static_cast<int>(t.data.size());
    int n = std::min(total, max_elements);
    std::vector<float> truncated(t.data.begin(), t.data.begin() + n);
    j["data"] = truncated;
    j["truncated"] = (n < total);

    return j;
}

json ad_tensor_to_json(const std::shared_ptr<ADTensor>& t,
                       int max_elements,
                       bool include_grad) {
    if (!t) return json::object();

    json j = tensor_to_json(t->val, max_elements);

    if (include_grad) {
        j["grad"] = tensor_to_json(t->grad, max_elements);
    }

    return j;
}

Tensor tensor_from_json(const json& j) {
    auto shape = j["shape"];
    int rows = shape[0].get<int>();
    int cols = shape[1].get<int>();
    Tensor t(rows, cols);

    if (j.contains("data")) {
        auto& data = j["data"];
        int n = std::min(static_cast<int>(data.size()), rows * cols);
        for (int i = 0; i < n; i++) {
            t.data[i] = data[i].get<float>();
        }
    }

    return t;
}

} // namespace server
