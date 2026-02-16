#pragma once
#include "server/module_registry.hpp"
#include "server/graph_executor.hpp"
#include "third_party/httplib.h"
#include "third_party/json.hpp"
#include <string>

namespace server {

class NodeServer {
public:
    explicit NodeServer(ModuleRegistry& registry);

    // Start the HTTP server (blocking)
    void start(const std::string& host = "0.0.0.0", int port = 8080);

    // Optionally serve static files from a directory
    void set_static_dir(const std::string& dir);

private:
    ModuleRegistry& registry_;
    GraphExecutor executor_;
    httplib::Server server_;

    void setup_routes();
    void add_cors(httplib::Response& res);

    // Endpoint handlers
    void handle_health(const httplib::Request& req, httplib::Response& res);
    void handle_modules(const httplib::Request& req, httplib::Response& res);
    void handle_execute(const httplib::Request& req, httplib::Response& res);
    void handle_execute_node(const httplib::Request& req, httplib::Response& res);
    void handle_presets(const httplib::Request& req, httplib::Response& res);

    // Preset graph definitions
    static nlohmann::json get_presets();
};

} // namespace server
