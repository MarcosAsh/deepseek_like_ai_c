#pragma once
#include "server/port_types.hpp"
#include "server/module_registry.hpp"
#include "server/serialization.hpp"
#include "third_party/json.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>

namespace server {

using json = nlohmann::json;

struct NodeDef {
    std::string id;
    std::string type;       // module type name
    json config;            // module config overrides
};

struct EdgeDef {
    std::string source_node;
    std::string source_port;
    std::string target_node;
    std::string target_port;
};

struct GraphDef {
    std::vector<NodeDef> nodes;
    std::vector<EdgeDef> edges;
};

struct NodeResult {
    std::string node_id;
    std::string node_type;
    double execution_time_ms;
    json outputs;           // per-port serialized tensor data
    std::string error;      // non-empty if execution failed
};

struct GraphResult {
    std::vector<NodeResult> node_results;
    std::vector<std::string> execution_order;
    double total_time_ms;
    std::string error;      // non-empty if graph-level error (cycles, etc.)
};

class GraphExecutor {
public:
    explicit GraphExecutor(const ModuleRegistry& registry);

    GraphResult execute(const GraphDef& graph);

    static GraphDef parse_graph(const json& j);

    static json result_to_json(const GraphResult& result);

private:
    const ModuleRegistry& registry_;

    // Topological sort using Kahn's algorithm. Returns empty + error on cycle.
    std::vector<std::string> topological_sort(
        const GraphDef& graph,
        std::string& error) const;

    json serialize_port_value(const PortValue& val) const;
};

} // namespace server
