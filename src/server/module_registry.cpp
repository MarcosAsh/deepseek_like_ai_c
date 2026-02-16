#include "server/module_registry.hpp"
#include <stdexcept>

namespace server {

ModuleRegistry& ModuleRegistry::instance() {
    static ModuleRegistry reg;
    return reg;
}

void ModuleRegistry::register_module(const std::string& type_name, ModuleFactory factory) {
    factories_[type_name] = std::move(factory);
}

std::unique_ptr<ModuleWrapper> ModuleRegistry::create(const std::string& type_name,
                                                       const json& config) const {
    auto it = factories_.find(type_name);
    if (it == factories_.end())
        throw std::runtime_error("Unknown module type: " + type_name);
    return it->second(config);
}

json ModuleRegistry::get_catalog() const {
    json catalog = json::array();
    for (auto& [name, factory] : factories_) {
        auto module = factory(json::object());
        catalog.push_back(module->to_catalog_json());
    }
    return catalog;
}

bool ModuleRegistry::has(const std::string& type_name) const {
    return factories_.count(type_name) > 0;
}

} // namespace server
