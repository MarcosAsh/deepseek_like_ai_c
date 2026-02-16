#include "server/module_registry.hpp"
#include "server/node_server.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char* argv[]) {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string static_dir;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--static" && i + 1 < argc) {
            static_dir = argv[++i];
        }
    }

    // Register all module wrappers
    auto& registry = server::ModuleRegistry::instance();
    server::register_all_modules(registry);

    std::cout << "Registered " << registry.get_catalog().size() << " modules" << std::endl;

    // Create and start server
    server::NodeServer srv(registry);

    if (!static_dir.empty()) {
        srv.set_static_dir(static_dir);
        std::cout << "Serving static files from: " << static_dir << std::endl;
    }

    srv.start(host, port);

    return 0;
}
