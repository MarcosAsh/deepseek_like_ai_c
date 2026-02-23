#pragma once
#include "server/module_wrapper.hpp"
#include <string>
#include <unordered_map>
#include <functional>
#include <memory>

namespace server {

using json = nlohmann::json;
using ModuleFactory = std::function<std::unique_ptr<ModuleWrapper>(const json& config)>;

class ModuleRegistry {
public:
    static ModuleRegistry& instance();

    void register_module(const std::string& type_name, ModuleFactory factory);

    std::unique_ptr<ModuleWrapper> create(const std::string& type_name,
                                          const json& config = json::object()) const;

    json get_catalog() const;

    bool has(const std::string& type_name) const;

private:
    ModuleRegistry() = default;
    std::unordered_map<std::string, ModuleFactory> factories_;
};

void register_all_modules(ModuleRegistry& registry);

} // namespace server
