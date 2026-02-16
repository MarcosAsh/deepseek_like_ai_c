#include "server/graph_executor.hpp"
#include "autodiff.hpp"
#include <queue>
#include <algorithm>
#include <iostream>

namespace server {

GraphExecutor::GraphExecutor(const ModuleRegistry& registry)
    : registry_(registry) {}

std::vector<std::string> GraphExecutor::topological_sort(
    const GraphDef& graph, std::string& error) const {

    // Build adjacency list and in-degree map
    std::unordered_map<std::string, std::vector<std::string>> adj;
    std::unordered_map<std::string, int> in_degree;

    for (auto& node : graph.nodes) {
        adj[node.id] = {};
        in_degree[node.id] = 0;
    }

    for (auto& edge : graph.edges) {
        adj[edge.source_node].push_back(edge.target_node);
        in_degree[edge.target_node]++;
    }

    // Kahn's algorithm
    std::queue<std::string> q;
    for (auto& [id, deg] : in_degree) {
        if (deg == 0) q.push(id);
    }

    std::vector<std::string> order;
    while (!q.empty()) {
        auto node = q.front();
        q.pop();
        order.push_back(node);

        for (auto& neighbor : adj[node]) {
            if (--in_degree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    if (order.size() != graph.nodes.size()) {
        error = "Graph contains a cycle";
        return {};
    }

    error.clear();
    return order;
}

json GraphExecutor::serialize_port_value(const PortValue& val) const {
    if (std::holds_alternative<std::string>(val)) {
        return {{"type", "TEXT"}, {"value", std::get<std::string>(val)}};
    }
    if (std::holds_alternative<std::vector<int>>(val)) {
        auto& v = std::get<std::vector<int>>(val);
        return {{"type", "TOKEN_IDS"}, {"value", v}, {"length", v.size()}};
    }
    if (std::holds_alternative<Tensor>(val)) {
        auto& t = std::get<Tensor>(val);
        json j = tensor_to_json(t);
        j["type"] = "TENSOR";
        return j;
    }
    if (std::holds_alternative<std::shared_ptr<ADTensor>>(val)) {
        auto& t = std::get<std::shared_ptr<ADTensor>>(val);
        json j = ad_tensor_to_json(t, 1000, true);
        j["type"] = "AD_TENSOR";
        return j;
    }
    if (std::holds_alternative<float>(val)) {
        return {{"type", "SCALAR"}, {"value", std::get<float>(val)}};
    }
    if (std::holds_alternative<int>(val)) {
        return {{"type", "INT"}, {"value", std::get<int>(val)}};
    }
    return {{"type", "UNKNOWN"}};
}

GraphResult GraphExecutor::execute(const GraphDef& graph) {
    GraphResult result;
    auto total_start = std::chrono::high_resolution_clock::now();

    // Clear global parameter registry for fresh execution
    clear_parameters();

    // Topological sort
    std::string sort_error;
    auto order = topological_sort(graph, sort_error);
    if (!sort_error.empty()) {
        result.error = sort_error;
        return result;
    }
    result.execution_order = order;

    // Build lookup maps
    std::unordered_map<std::string, const NodeDef*> node_map;
    for (auto& node : graph.nodes)
        node_map[node.id] = &node;

    // edge: target_node.target_port -> (source_node, source_port)
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> incoming;
    for (auto& edge : graph.edges) {
        std::string key = edge.target_node + "." + edge.target_port;
        incoming[key].push_back({edge.source_node, edge.source_port});
    }

    // Store outputs per node
    std::unordered_map<std::string, std::unordered_map<std::string, PortValue>> node_outputs;

    // Execute in topological order
    for (auto& node_id : order) {
        NodeResult nr;
        nr.node_id = node_id;
        auto* node_def = node_map[node_id];
        nr.node_type = node_def->type;

        auto node_start = std::chrono::high_resolution_clock::now();

        try {
            // Create module instance
            auto module = registry_.create(node_def->type, node_def->config);

            // Gather inputs from upstream
            std::unordered_map<std::string, PortValue> inputs;
            for (auto& port : module->input_ports()) {
                std::string key = node_id + "." + port.name;
                auto it = incoming.find(key);
                if (it != incoming.end() && !it->second.empty()) {
                    auto& [src_node, src_port] = it->second[0];
                    auto out_it = node_outputs.find(src_node);
                    if (out_it != node_outputs.end()) {
                        auto port_it = out_it->second.find(src_port);
                        if (port_it != out_it->second.end()) {
                            inputs[port.name] = port_it->second;
                        }
                    }
                }
                if (inputs.find(port.name) == inputs.end() && !port.optional) {
                    throw std::runtime_error("Missing required input: " + port.name);
                }
            }

            // Execute
            auto outputs = module->execute(inputs);

            // Store outputs
            node_outputs[node_id] = outputs;

            // Serialize outputs for response
            nr.outputs = json::object();
            for (auto& [port_name, port_val] : outputs) {
                nr.outputs[port_name] = serialize_port_value(port_val);
            }

        } catch (const std::exception& e) {
            nr.error = e.what();
        }

        auto node_end = std::chrono::high_resolution_clock::now();
        nr.execution_time_ms = std::chrono::duration<double, std::milli>(
            node_end - node_start).count();

        result.node_results.push_back(std::move(nr));
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(
        total_end - total_start).count();

    return result;
}

GraphDef GraphExecutor::parse_graph(const json& j) {
    GraphDef graph;

    for (auto& node_json : j["nodes"]) {
        NodeDef node;
        node.id = node_json["id"].get<std::string>();
        node.type = node_json["type"].get<std::string>();
        node.config = node_json.value("config", json::object());
        graph.nodes.push_back(std::move(node));
    }

    for (auto& edge_json : j["edges"]) {
        EdgeDef edge;
        edge.source_node = edge_json["source_node"].get<std::string>();
        edge.source_port = edge_json["source_port"].get<std::string>();
        edge.target_node = edge_json["target_node"].get<std::string>();
        edge.target_port = edge_json["target_port"].get<std::string>();
        graph.edges.push_back(std::move(edge));
    }

    return graph;
}

json GraphExecutor::result_to_json(const GraphResult& result) {
    json j;
    j["total_time_ms"] = result.total_time_ms;
    j["execution_order"] = result.execution_order;

    if (!result.error.empty()) {
        j["error"] = result.error;
    }

    json nodes = json::object();
    for (auto& nr : result.node_results) {
        json node_j;
        node_j["type"] = nr.node_type;
        node_j["execution_time_ms"] = nr.execution_time_ms;
        node_j["outputs"] = nr.outputs;
        if (!nr.error.empty()) {
            node_j["error"] = nr.error;
        }
        nodes[nr.node_id] = node_j;
    }
    j["nodes"] = nodes;

    return j;
}

} // namespace server
