#include "server/node_server.hpp"
#include <iostream>

namespace server {

using json = nlohmann::json;

NodeServer::NodeServer(ModuleRegistry& registry)
    : registry_(registry), executor_(registry) {}

void NodeServer::add_cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

void NodeServer::setup_routes() {
    server_.Options(".*", [this](const httplib::Request&, httplib::Response& res) {
        add_cors(res);
        res.status = 204;
    });

    server_.Get("/api/v1/health", [this](const httplib::Request& req, httplib::Response& res) {
        handle_health(req, res);
    });

    server_.Get("/api/v1/modules", [this](const httplib::Request& req, httplib::Response& res) {
        handle_modules(req, res);
    });

    server_.Post("/api/v1/execute", [this](const httplib::Request& req, httplib::Response& res) {
        handle_execute(req, res);
    });

    server_.Post("/api/v1/execute_node", [this](const httplib::Request& req, httplib::Response& res) {
        handle_execute_node(req, res);
    });

    server_.Get("/api/v1/presets", [this](const httplib::Request& req, httplib::Response& res) {
        handle_presets(req, res);
    });
}

void NodeServer::set_static_dir(const std::string& dir) {
    server_.set_mount_point("/", dir);
}

void NodeServer::start(const std::string& host, int port) {
    setup_routes();
    std::cout << "Node server starting on " << host << ":" << port << std::endl;
    server_.listen(host, port);
}

void NodeServer::handle_health(const httplib::Request&, httplib::Response& res) {
    add_cors(res);
    json j = {{"status", "ok"}, {"version", "1.0.0"}};
    res.set_content(j.dump(), "application/json");
}

void NodeServer::handle_modules(const httplib::Request&, httplib::Response& res) {
    add_cors(res);
    json j = {{"modules", registry_.get_catalog()}};
    res.set_content(j.dump(), "application/json");
}

void NodeServer::handle_execute(const httplib::Request& req, httplib::Response& res) {
    add_cors(res);
    try {
        auto body = json::parse(req.body);
        auto graph = GraphExecutor::parse_graph(body);
        auto result = executor_.execute(graph);
        auto j = GraphExecutor::result_to_json(result);
        res.set_content(j.dump(), "application/json");
    } catch (const std::exception& e) {
        json err = {{"error", e.what()}};
        res.status = 400;
        res.set_content(err.dump(), "application/json");
    }
}

void NodeServer::handle_execute_node(const httplib::Request& req, httplib::Response& res) {
    add_cors(res);
    try {
        auto body = json::parse(req.body);
        std::string type = body["type"].get<std::string>();
        json config = body.value("config", json::object());
        json inputs_json = body.value("inputs", json::object());

        auto module = registry_.create(type, config);

        std::unordered_map<std::string, PortValue> inputs;
        for (auto& [key, val] : inputs_json.items()) {
            std::string port_type = val.value("type", "");
            if (port_type == "TEXT") {
                inputs[key] = val["value"].get<std::string>();
            } else if (port_type == "TOKEN_IDS") {
                inputs[key] = val["value"].get<std::vector<int>>();
            } else if (port_type == "INT") {
                inputs[key] = val["value"].get<int>();
            } else if (port_type == "SCALAR") {
                inputs[key] = val["value"].get<float>();
            } else if (port_type == "TENSOR") {
                inputs[key] = tensor_from_json(val);
            }
        }

        auto outputs = module->execute(inputs);

        json result = json::object();
        for (auto& [port_name, port_val] : outputs) {
            if (std::holds_alternative<std::string>(port_val)) {
                result[port_name] = {{"type", "TEXT"}, {"value", std::get<std::string>(port_val)}};
            } else if (std::holds_alternative<std::vector<int>>(port_val)) {
                auto& v = std::get<std::vector<int>>(port_val);
                result[port_name] = {{"type", "TOKEN_IDS"}, {"value", v}};
            } else if (std::holds_alternative<Tensor>(port_val)) {
                auto j = tensor_to_json(std::get<Tensor>(port_val));
                j["type"] = "TENSOR";
                result[port_name] = j;
            } else if (std::holds_alternative<std::shared_ptr<ADTensor>>(port_val)) {
                auto j = ad_tensor_to_json(std::get<std::shared_ptr<ADTensor>>(port_val));
                j["type"] = "AD_TENSOR";
                result[port_name] = j;
            } else if (std::holds_alternative<float>(port_val)) {
                result[port_name] = {{"type", "SCALAR"}, {"value", std::get<float>(port_val)}};
            } else if (std::holds_alternative<int>(port_val)) {
                result[port_name] = {{"type", "INT"}, {"value", std::get<int>(port_val)}};
            }
        }

        res.set_content(result.dump(), "application/json");
    } catch (const std::exception& e) {
        json err = {{"error", e.what()}};
        res.status = 400;
        res.set_content(err.dump(), "application/json");
    }
}

void NodeServer::handle_presets(const httplib::Request&, httplib::Response& res) {
    add_cors(res);
    auto presets = get_presets();
    res.set_content(presets.dump(), "application/json");
}

json NodeServer::get_presets() {
    json presets = json::array();

    {
        json preset;
        preset["name"] = "Embedding + Positional Encoding";
        preset["description"] = "The simplest pipeline: tokenize text, embed tokens, and add positional encoding";
        preset["nodes"] = json::array({
            {{"id", "text_in"}, {"type", "TextInput"}, {"config", {{"text", "Hello world"}}}},
            {{"id", "tokenizer"}, {"type", "Tokenizer"}, {"config", {{"vocab_file", "input_files/vocab.txt"}}}},
            {{"id", "seq_len"}, {"type", "SeqLenExtractor"}, {"config", json::object()}},
            {{"id", "embedding"}, {"type", "ADEmbedding"}, {"config", {{"vocab_size", 10000}, {"embed_dim", 64}}}},
            {{"id", "pos_enc"}, {"type", "ADPositionalEncoding"}, {"config", {{"embed_dim", 64}}}},
            {{"id", "add_pe"}, {"type", "Add"}, {"config", json::object()}}
        });
        preset["edges"] = json::array({
            {{"source_node", "text_in"}, {"source_port", "text"}, {"target_node", "tokenizer"}, {"target_port", "text"}},
            {{"source_node", "tokenizer"}, {"source_port", "tokens"}, {"target_node", "embedding"}, {"target_port", "tokens"}},
            {{"source_node", "tokenizer"}, {"source_port", "tokens"}, {"target_node", "seq_len"}, {"target_port", "tokens"}},
            {{"source_node", "seq_len"}, {"source_port", "seq_len"}, {"target_node", "pos_enc"}, {"target_port", "seq_len"}},
            {{"source_node", "embedding"}, {"source_port", "output"}, {"target_node", "add_pe"}, {"target_port", "a"}},
            {{"source_node", "pos_enc"}, {"source_port", "output"}, {"target_node", "add_pe"}, {"target_port", "b"}}
        });
        presets.push_back(preset);
    }

    {
        json preset;
        preset["name"] = "Single Attention Head";
        preset["description"] = "Decomposed attention mechanism: embedding -> layer norm -> multi-head attention";
        preset["nodes"] = json::array({
            {{"id", "tokens_in"}, {"type", "TokenIDsInput"}, {"config", {{"tokens", {1,2,3,4,5,6,7,8}}}}},
            {{"id", "embedding"}, {"type", "ADEmbedding"}, {"config", {{"vocab_size", 256}, {"embed_dim", 64}}}},
            {{"id", "ln"}, {"type", "ADLayerNorm"}, {"config", {{"dim", 64}}}},
            {{"id", "attention"}, {"type", "ADMultiHeadAttention"}, {"config", {{"embed_dim", 64}, {"num_heads", 4}}}}
        });
        preset["edges"] = json::array({
            {{"source_node", "tokens_in"}, {"source_port", "tokens"}, {"target_node", "embedding"}, {"target_port", "tokens"}},
            {{"source_node", "embedding"}, {"source_port", "output"}, {"target_node", "ln"}, {"target_port", "input"}},
            {{"source_node", "ln"}, {"source_port", "output"}, {"target_node", "attention"}, {"target_port", "input"}}
        });
        presets.push_back(preset);
    }

    {
        json preset;
        preset["name"] = "MoE Routing";
        preset["description"] = "Mixture of Experts layer: see how tokens are routed to different expert FFNs";
        preset["nodes"] = json::array({
            {{"id", "tokens_in"}, {"type", "TokenIDsInput"}, {"config", {{"tokens", {1,2,3,4,5,6,7,8}}}}},
            {{"id", "embedding"}, {"type", "ADEmbedding"}, {"config", {{"vocab_size", 256}, {"embed_dim", 64}}}},
            {{"id", "moe"}, {"type", "ADMoE"}, {"config", {{"embed_dim", 64}, {"hidden_dim", 128}, {"num_experts", 4}, {"top_k", 2}}}}
        });
        preset["edges"] = json::array({
            {{"source_node", "tokens_in"}, {"source_port", "tokens"}, {"target_node", "embedding"}, {"target_port", "tokens"}},
            {{"source_node", "embedding"}, {"source_port", "output"}, {"target_node", "moe"}, {"target_port", "input"}}
        });
        presets.push_back(preset);
    }

    {
        json preset;
        preset["name"] = "Full Transformer Block";
        preset["description"] = "Complete transformer block: LN -> Attention -> Residual -> LN -> FFN -> Residual";
        preset["nodes"] = json::array({
            {{"id", "tokens_in"}, {"type", "TokenIDsInput"}, {"config", {{"tokens", {10,20,30,40,50,60}}}}},
            {{"id", "embedding"}, {"type", "ADEmbedding"}, {"config", {{"vocab_size", 256}, {"embed_dim", 64}}}},
            {{"id", "seq_len"}, {"type", "SeqLenExtractor"}, {"config", json::object()}},
            {{"id", "pos_enc"}, {"type", "ADPositionalEncoding"}, {"config", {{"embed_dim", 64}}}},
            {{"id", "add_pe"}, {"type", "Add"}, {"config", json::object()}},
            {{"id", "transformer"}, {"type", "ADTransformerBlock"}, {"config", {{"embed_dim", 64}, {"hidden_dim", 256}, {"n_heads", 4}}}}
        });
        preset["edges"] = json::array({
            {{"source_node", "tokens_in"}, {"source_port", "tokens"}, {"target_node", "embedding"}, {"target_port", "tokens"}},
            {{"source_node", "tokens_in"}, {"source_port", "tokens"}, {"target_node", "seq_len"}, {"target_port", "tokens"}},
            {{"source_node", "seq_len"}, {"source_port", "seq_len"}, {"target_node", "pos_enc"}, {"target_port", "seq_len"}},
            {{"source_node", "embedding"}, {"source_port", "output"}, {"target_node", "add_pe"}, {"target_port", "a"}},
            {{"source_node", "pos_enc"}, {"source_port", "output"}, {"target_node", "add_pe"}, {"target_port", "b"}},
            {{"source_node", "add_pe"}, {"source_port", "output"}, {"target_node", "transformer"}, {"target_port", "input"}}
        });
        presets.push_back(preset);
    }

    {
        json preset;
        preset["name"] = "Full Training Pipeline";
        preset["description"] = "Complete training pipeline: Tokenize -> Embed -> Transformer -> Logits -> Loss -> Backward";
        preset["nodes"] = json::array({
            {{"id", "tokens_in"}, {"type", "TokenIDsInput"}, {"config", {{"tokens", {1,2,3,4,5,6,7,8}}}}},
            {{"id", "targets"}, {"type", "TokenIDsInput"}, {"config", {{"tokens", {2,3,4,5,6,7,8,9}}}}},
            {{"id", "embedding"}, {"type", "ADEmbedding"}, {"config", {{"vocab_size", 256}, {"embed_dim", 64}}}},
            {{"id", "seq_len"}, {"type", "SeqLenExtractor"}, {"config", json::object()}},
            {{"id", "pos_enc"}, {"type", "ADPositionalEncoding"}, {"config", {{"embed_dim", 64}}}},
            {{"id", "add_pe"}, {"type", "Add"}, {"config", json::object()}},
            {{"id", "transformer"}, {"type", "ADTransformerBlock"}, {"config", {{"embed_dim", 64}, {"hidden_dim", 256}, {"n_heads", 4}}}},
            {{"id", "transpose_emb"}, {"type", "Transpose"}, {"config", json::object()}},
            {{"id", "logits"}, {"type", "MatMul"}, {"config", json::object()}},
            {{"id", "loss"}, {"type", "CrossEntropy"}, {"config", json::object()}},
            {{"id", "backward"}, {"type", "Backward"}, {"config", json::object()}}
        });
        preset["edges"] = json::array({
            {{"source_node", "tokens_in"}, {"source_port", "tokens"}, {"target_node", "embedding"}, {"target_port", "tokens"}},
            {{"source_node", "tokens_in"}, {"source_port", "tokens"}, {"target_node", "seq_len"}, {"target_port", "tokens"}},
            {{"source_node", "seq_len"}, {"source_port", "seq_len"}, {"target_node", "pos_enc"}, {"target_port", "seq_len"}},
            {{"source_node", "embedding"}, {"source_port", "output"}, {"target_node", "add_pe"}, {"target_port", "a"}},
            {{"source_node", "pos_enc"}, {"source_port", "output"}, {"target_node", "add_pe"}, {"target_port", "b"}},
            {{"source_node", "add_pe"}, {"source_port", "output"}, {"target_node", "transformer"}, {"target_port", "input"}},
            {{"source_node", "embedding"}, {"source_port", "weights"}, {"target_node", "transpose_emb"}, {"target_port", "input"}},
            {{"source_node", "transpose_emb"}, {"source_port", "output"}, {"target_node", "logits"}, {"target_port", "a"}},
            {{"source_node", "transformer"}, {"source_port", "output"}, {"target_node", "logits"}, {"target_port", "b"}},
            {{"source_node", "logits"}, {"source_port", "output"}, {"target_node", "loss"}, {"target_port", "logits"}},
            {{"source_node", "targets"}, {"source_port", "tokens"}, {"target_node", "loss"}, {"target_port", "targets"}},
            {{"source_node", "loss"}, {"source_port", "loss"}, {"target_node", "backward"}, {"target_port", "loss"}}
        });
        presets.push_back(preset);
    }

    return {{"presets", presets}};
}

} // namespace server
