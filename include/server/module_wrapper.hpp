#pragma once
#include "server/port_types.hpp"
#include "third_party/json.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace server {

using json = nlohmann::json;

class ModuleWrapper {
public:
    virtual ~ModuleWrapper() = default;

    virtual std::string type_name() const = 0;
    virtual std::string category() const = 0;
    virtual std::string description() const = 0;
    virtual std::vector<PortDescriptor> input_ports() const = 0;
    virtual std::vector<PortDescriptor> output_ports() const = 0;
    virtual json default_config() const = 0;

    virtual std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>& inputs) = 0;

    json to_catalog_json() const {
        json j;
        j["type"] = type_name();
        j["category"] = category();
        j["description"] = description();
        j["default_config"] = default_config();

        json inputs = json::array();
        for (auto& p : input_ports()) {
            inputs.push_back({
                {"name", p.name},
                {"type", port_type_name(p.type)},
                {"optional", p.optional}
            });
        }
        j["inputs"] = inputs;

        json outputs = json::array();
        for (auto& p : output_ports()) {
            outputs.push_back({
                {"name", p.name},
                {"type", port_type_name(p.type)},
                {"optional", p.optional}
            });
        }
        j["outputs"] = outputs;

        return j;
    }
};

} // namespace server
