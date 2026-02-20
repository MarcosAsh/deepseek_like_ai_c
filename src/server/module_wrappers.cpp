#include "server/module_wrapper.hpp"
#include "server/port_types.hpp"
#include "server/serialization.hpp"
#include "tokenizer.hpp"
#include "layers/ad_embedding.hpp"
#include "layers/ad_positional_encoding.hpp"
#include "layers/ad_layer_norm.hpp"
#include "layers/ad_multi_head_attention.hpp"
#include "layers/ad_feed_forward.hpp"
#include "layers/ad_moe.hpp"
#include "layers/ad_linear.hpp"
#include "layers/ad_transformer.hpp"
#include "autodiff.hpp"
#include <stdexcept>
#include <cmath>

namespace server {

// Helper to get a value from inputs with type checking
template<typename T>
T get_input(const std::unordered_map<std::string, PortValue>& inputs,
            const std::string& name) {
    auto it = inputs.find(name);
    if (it == inputs.end())
        throw std::runtime_error("Missing input: " + name);
    if (!std::holds_alternative<T>(it->second))
        throw std::runtime_error("Type mismatch for input: " + name);
    return std::get<T>(it->second);
}

// ======================== TokenizerWrapper ========================

class TokenizerWrapper : public ModuleWrapper {
    std::unique_ptr<Tokenizer> tok;
    std::string vocab_file, bpe_file;
public:
    TokenizerWrapper(const json& config) {
        vocab_file = config.value("vocab_file", "input_files/vocab.txt");
        bpe_file = config.value("bpe_codes", "");
    }

    std::string type_name() const override { return "Tokenizer"; }
    std::string category() const override { return "preprocessing"; }
    std::string description() const override {
        return "BPE tokenizer: converts text to token IDs";
    }
    std::vector<PortDescriptor> input_ports() const override {
        return {{"text", PortType::TEXT}};
    }
    std::vector<PortDescriptor> output_ports() const override {
        return {{"tokens", PortType::TOKEN_IDS}};
    }
    json default_config() const override {
        return {{"vocab_file", "input_files/vocab.txt"}, {"bpe_codes", ""}};
    }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>& inputs) override {
        if (!tok) {
            tok = std::make_unique<Tokenizer>(vocab_file,
                bpe_file.empty() ? "" : bpe_file);
        }
        auto text = get_input<std::string>(inputs, "text");
        auto tokens = tok->encode(text);
        return {{"tokens", PortValue(tokens)}};
    }
};

// ======================== ADEmbeddingWrapper ========================

class ADEmbeddingWrapper : public ModuleWrapper {
    std::unique_ptr<ADEmbedding> emb;
    int vocab_size, embed_dim;
public:
    ADEmbeddingWrapper(const json& config)
        : vocab_size(config.value("vocab_size", 256)),
          embed_dim(config.value("embed_dim", 64)) {}

    std::string type_name() const override { return "ADEmbedding"; }
    std::string category() const override { return "embedding"; }
    std::string description() const override {
        return "Token embedding lookup: maps token IDs to dense vectors";
    }
    std::vector<PortDescriptor> input_ports() const override {
        return {{"tokens", PortType::TOKEN_IDS}};
    }
    std::vector<PortDescriptor> output_ports() const override {
        return {
            {"output", PortType::AD_TENSOR},
            {"weights", PortType::AD_TENSOR, true}
        };
    }
    json default_config() const override {
        return {{"vocab_size", 256}, {"embed_dim", 64}};
    }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>& inputs) override {
        if (!emb) emb = std::make_unique<ADEmbedding>(vocab_size, embed_dim);
        auto tokens = get_input<std::vector<int>>(inputs, "tokens");
        auto out = emb->forward(tokens);
        return {
            {"output", PortValue(out)},
            {"weights", PortValue(emb->get_weights())}
        };
    }
};

// ======================== ADPosEncWrapper ========================

class ADPosEncWrapper : public ModuleWrapper {
    std::unique_ptr<ADPositionalEncoding> pe;
    int embed_dim, max_len;
public:
    ADPosEncWrapper(const json& config)
        : embed_dim(config.value("embed_dim", 64)),
          max_len(config.value("max_len", 512)) {}

    std::string type_name() const override { return "ADPositionalEncoding"; }
    std::string category() const override { return "embedding"; }
    std::string description() const override {
        return "Learned positional encoding: adds position information to embeddings";
    }
    std::vector<PortDescriptor> input_ports() const override {
        return {{"seq_len", PortType::INT}};
    }
    std::vector<PortDescriptor> output_ports() const override {
        return {{"output", PortType::AD_TENSOR}};
    }
    json default_config() const override {
        return {{"embed_dim", 64}, {"max_len", 512}};
    }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>& inputs) override {
        if (!pe) pe = std::make_unique<ADPositionalEncoding>(embed_dim, max_len);
        int seq_len = get_input<int>(inputs, "seq_len");
        auto out = pe->forward(seq_len);
        return {{"output", PortValue(out)}};
    }
};

// ======================== ADLayerNormWrapper ========================

class ADLayerNormWrapper : public ModuleWrapper {
    std::unique_ptr<ADLayerNorm> ln;
    int dim;
    float eps;
public:
    ADLayerNormWrapper(const json& config)
        : dim(config.value("dim", 64)),
          eps(config.value("eps", 1e-5f)) {}

    std::string type_name() const override { return "ADLayerNorm"; }
    std::string category() const override { return "normalization"; }
    std::string description() const override {
        return "Layer normalization: normalizes activations across features";
    }
    std::vector<PortDescriptor> input_ports() const override {
        return {{"input", PortType::AD_TENSOR}};
    }
    std::vector<PortDescriptor> output_ports() const override {
        return {{"output", PortType::AD_TENSOR}};
    }
    json default_config() const override {
        return {{"dim", 64}, {"eps", 1e-5}};
    }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>& inputs) override {
        if (!ln) ln = std::make_unique<ADLayerNorm>(dim, eps);
        auto input = get_input<std::shared_ptr<ADTensor>>(inputs, "input");
        auto out = ln->forward(input);
        return {{"output", PortValue(out)}};
    }
};

// ======================== ADMHAWrapper ========================

class ADMHAWrapper : public ModuleWrapper {
    std::unique_ptr<ADMultiHeadAttention> mha;
    int embed_dim, num_heads;
public:
    ADMHAWrapper(const json& config)
        : embed_dim(config.value("embed_dim", 64)),
          num_heads(config.value("num_heads", 4)) {}

    std::string type_name() const override { return "ADMultiHeadAttention"; }
    std::string category() const override { return "attention"; }
    std::string description() const override {
        return "Multi-head self-attention with ALiBi position bias and causal masking";
    }
    std::vector<PortDescriptor> input_ports() const override {
        return {{"input", PortType::AD_TENSOR}};
    }
    std::vector<PortDescriptor> output_ports() const override {
        return {{"output", PortType::AD_TENSOR}};
    }
    json default_config() const override {
        return {{"embed_dim", 64}, {"num_heads", 4}};
    }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>& inputs) override {
        if (!mha) mha = std::make_unique<ADMultiHeadAttention>(embed_dim, num_heads);
        auto input = get_input<std::shared_ptr<ADTensor>>(inputs, "input");
        auto out = mha->forward(input);
        return {{"output", PortValue(out)}};
    }
};

// ======================== ADFeedForwardWrapper ========================

class ADFeedForwardWrapper : public ModuleWrapper {
    std::unique_ptr<ADFeedForward> ff;
    int embed_dim, hidden_dim;
public:
    ADFeedForwardWrapper(const json& config)
        : embed_dim(config.value("embed_dim", 64)),
          hidden_dim(config.value("hidden_dim", 256)) {}

    std::string type_name() const override { return "ADFeedForward"; }
    std::string category() const override { return "feedforward"; }
    std::string description() const override {
        return "Position-wise feed-forward network with GELU activation";
    }
    std::vector<PortDescriptor> input_ports() const override {
        return {{"input", PortType::AD_TENSOR}};
    }
    std::vector<PortDescriptor> output_ports() const override {
        return {{"output", PortType::AD_TENSOR}};
    }
    json default_config() const override {
        return {{"embed_dim", 64}, {"hidden_dim", 256}};
    }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>& inputs) override {
        if (!ff) ff = std::make_unique<ADFeedForward>(embed_dim, hidden_dim);
        auto input = get_input<std::shared_ptr<ADTensor>>(inputs, "input");
        auto out = ff->forward(input);
        return {{"output", PortValue(out)}};
    }
};

// ======================== ADMoEWrapper ========================

class ADMoEWrapper : public ModuleWrapper {
    std::unique_ptr<ADMoE> moe;
    int embed_dim, hidden_dim, num_experts, top_k;
public:
    ADMoEWrapper(const json& config)
        : embed_dim(config.value("embed_dim", 64)),
          hidden_dim(config.value("hidden_dim", 256)),
          num_experts(config.value("num_experts", 4)),
          top_k(config.value("top_k", 2)) {}

    std::string type_name() const override { return "ADMoE"; }
    std::string category() const override { return "moe"; }
    std::string description() const override {
        return "Mixture of Experts: routes tokens to top-k expert FFNs with load balancing";
    }
    std::vector<PortDescriptor> input_ports() const override {
        return {{"input", PortType::AD_TENSOR}};
    }
    std::vector<PortDescriptor> output_ports() const override {
        return {
            {"output", PortType::AD_TENSOR},
            {"aux_loss", PortType::AD_TENSOR, true}
        };
    }
    json default_config() const override {
        return {{"embed_dim", 64}, {"hidden_dim", 256},
                {"num_experts", 4}, {"top_k", 2}};
    }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>& inputs) override {
        if (!moe) moe = std::make_unique<ADMoE>(embed_dim, hidden_dim, num_experts, top_k);
        auto input = get_input<std::shared_ptr<ADTensor>>(inputs, "input");
        auto result = moe->forward(input);
        return {
            {"output", PortValue(result.output)},
            {"aux_loss", PortValue(result.aux_loss)}
        };
    }
};

// ======================== ADLinearWrapper ========================

class ADLinearWrapper : public ModuleWrapper {
    std::unique_ptr<ADLinear> linear;
    int input_dim, output_dim;
public:
    ADLinearWrapper(const json& config)
        : input_dim(config.value("input_dim", 64)),
          output_dim(config.value("output_dim", 64)) {}

    std::string type_name() const override { return "ADLinear"; }
    std::string category() const override { return "linear"; }
    std::string description() const override {
        return "Linear projection: y = Wx + b";
    }
    std::vector<PortDescriptor> input_ports() const override {
        return {{"input", PortType::AD_TENSOR}};
    }
    std::vector<PortDescriptor> output_ports() const override {
        return {{"output", PortType::AD_TENSOR}};
    }
    json default_config() const override {
        return {{"input_dim", 64}, {"output_dim", 64}};
    }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>& inputs) override {
        if (!linear) linear = std::make_unique<ADLinear>(input_dim, output_dim);
        auto input = get_input<std::shared_ptr<ADTensor>>(inputs, "input");
        auto out = linear->forward(input);
        return {{"output", PortValue(out)}};
    }
};

// ======================== ADTransBlockWrapper ========================

class ADTransBlockWrapper : public ModuleWrapper {
    std::unique_ptr<ADTransformerBlock> block;
    int embed_dim, hidden_dim, n_heads;
    bool use_moe;
    int num_experts, moe_top_k;
public:
    ADTransBlockWrapper(const json& config)
        : embed_dim(config.value("embed_dim", 64)),
          hidden_dim(config.value("hidden_dim", 256)),
          n_heads(config.value("n_heads", 4)),
          use_moe(config.value("use_moe", false)),
          num_experts(config.value("num_experts", 4)),
          moe_top_k(config.value("moe_top_k", 2)) {}

    std::string type_name() const override { return "ADTransformerBlock"; }
    std::string category() const override { return "transformer"; }
    std::string description() const override {
        return "Full transformer block: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN/MoE -> Residual";
    }
    std::vector<PortDescriptor> input_ports() const override {
        return {{"input", PortType::AD_TENSOR}};
    }
    std::vector<PortDescriptor> output_ports() const override {
        return {{"output", PortType::AD_TENSOR}};
    }
    json default_config() const override {
        return {{"embed_dim", 64}, {"hidden_dim", 256}, {"n_heads", 4},
                {"use_moe", false}, {"num_experts", 4}, {"moe_top_k", 2}};
    }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>& inputs) override {
        if (!block) block = std::make_unique<ADTransformerBlock>(
            embed_dim, hidden_dim, n_heads, use_moe, num_experts, moe_top_k);
        auto input = get_input<std::shared_ptr<ADTensor>>(inputs, "input");
        auto out = block->forward(input);
        return {{"output", PortValue(out)}};
    }
};

// ======================== AddWrapper ========================

class AddWrapper : public ModuleWrapper {
public:
    AddWrapper(const json&) {}

    std::string type_name() const override { return "Add"; }
    std::string category() const override { return "math"; }
    std::string description() const override {
        return "Element-wise tensor addition: output = a + b";
    }
    std::vector<PortDescriptor> input_ports() const override {
        return {{"a", PortType::AD_TENSOR}, {"b", PortType::AD_TENSOR}};
    }
    std::vector<PortDescriptor> output_ports() const override {
        return {{"output", PortType::AD_TENSOR}};
    }
    json default_config() const override { return json::object(); }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>& inputs) override {
        auto a = get_input<std::shared_ptr<ADTensor>>(inputs, "a");
        auto b = get_input<std::shared_ptr<ADTensor>>(inputs, "b");
        auto out = add(a, b);
        return {{"output", PortValue(out)}};
    }
};

// ======================== MatMulWrapper ========================

class MatMulWrapper : public ModuleWrapper {
public:
    MatMulWrapper(const json&) {}

    std::string type_name() const override { return "MatMul"; }
    std::string category() const override { return "math"; }
    std::string description() const override {
        return "Matrix multiplication: output = a @ b";
    }
    std::vector<PortDescriptor> input_ports() const override {
        return {{"a", PortType::AD_TENSOR}, {"b", PortType::AD_TENSOR}};
    }
    std::vector<PortDescriptor> output_ports() const override {
        return {{"output", PortType::AD_TENSOR}};
    }
    json default_config() const override { return json::object(); }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>& inputs) override {
        auto a = get_input<std::shared_ptr<ADTensor>>(inputs, "a");
        auto b = get_input<std::shared_ptr<ADTensor>>(inputs, "b");
        auto out = matmul(a, b);
        return {{"output", PortValue(out)}};
    }
};

// ======================== TransposeWrapper ========================

class TransposeWrapper : public ModuleWrapper {
public:
    TransposeWrapper(const json&) {}

    std::string type_name() const override { return "Transpose"; }
    std::string category() const override { return "math"; }
    std::string description() const override {
        return "Matrix transpose: swaps rows and columns";
    }
    std::vector<PortDescriptor> input_ports() const override {
        return {{"input", PortType::AD_TENSOR}};
    }
    std::vector<PortDescriptor> output_ports() const override {
        return {{"output", PortType::AD_TENSOR}};
    }
    json default_config() const override { return json::object(); }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>& inputs) override {
        auto input = get_input<std::shared_ptr<ADTensor>>(inputs, "input");
        auto out = transpose(input);
        return {{"output", PortValue(out)}};
    }
};

// ======================== CrossEntropyWrapper ========================

class CrossEntropyWrapper : public ModuleWrapper {
public:
    CrossEntropyWrapper(const json&) {}

    std::string type_name() const override { return "CrossEntropy"; }
    std::string category() const override { return "loss"; }
    std::string description() const override {
        return "Cross-entropy loss with log-sum-exp stability. Computes loss between logits and target token IDs.";
    }
    std::vector<PortDescriptor> input_ports() const override {
        return {
            {"logits", PortType::AD_TENSOR},
            {"targets", PortType::TOKEN_IDS}
        };
    }
    std::vector<PortDescriptor> output_ports() const override {
        return {{"loss", PortType::AD_TENSOR}};
    }
    json default_config() const override { return json::object(); }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>& inputs) override {
        auto logits = get_input<std::shared_ptr<ADTensor>>(inputs, "logits");
        auto targets = get_input<std::vector<int>>(inputs, "targets");

        // logits: [vocab_size x seq_len], targets: length seq_len
        int vocab_size = logits->val.rows;
        int seq_len = logits->val.cols;

        // Build one-hot target tensor [vocab_size x seq_len]
        Tensor target_tensor(vocab_size, seq_len);
        target_tensor.fill(0.0f);
        for (int t = 0; t < seq_len && t < static_cast<int>(targets.size()); t++) {
            int target_id = targets[t];
            if (target_id >= 0 && target_id < vocab_size)
                target_tensor(target_id, t) = 1.0f;
        }
        auto target_ad = make_ad(target_tensor);

        // Log-sum-exp for numerical stability
        // max per column
        Tensor max_vals(1, seq_len);
        for (int j = 0; j < seq_len; j++) {
            float mx = logits->val(0, j);
            for (int i = 1; i < vocab_size; i++)
                mx = std::max(mx, logits->val(i, j));
            max_vals(0, j) = mx;
        }

        // Broadcast max and subtract
        Tensor ones_col(vocab_size, 1);
        ones_col.fill(1.0f);
        auto max_ad = make_ad(ones_col.matmul(max_vals));
        auto shifted = sub(logits, max_ad);
        auto exp_vals = exp_ad(shifted);

        // Sum exp per column
        Tensor ones_row(1, vocab_size);
        ones_row.fill(1.0f);
        auto ones_ad = make_ad(ones_row);
        auto sum_exp = matmul(ones_ad, exp_vals); // [1 x seq_len]

        auto log_sum = log_ad(sum_exp); // [1 x seq_len]

        // log_probs = shifted - log_sum (broadcast)
        auto log_sum_broadcast = matmul(make_ad(ones_col), log_sum); // [vocab x seq_len]
        auto log_probs = sub(shifted, log_sum_broadcast);

        // loss = -sum(target * log_probs) / seq_len
        auto target_log_probs = mul(target_ad, log_probs);
        auto total = sum(target_log_probs);
        auto neg_loss = scalar_mul(total, -1.0f / static_cast<float>(seq_len));

        return {{"loss", PortValue(neg_loss)}};
    }
};

// ======================== BackwardWrapper ========================

class BackwardWrapper : public ModuleWrapper {
public:
    BackwardWrapper(const json&) {}

    std::string type_name() const override { return "Backward"; }
    std::string category() const override { return "training"; }
    std::string description() const override {
        return "Triggers reverse-mode backpropagation from the loss tensor to compute all gradients";
    }
    std::vector<PortDescriptor> input_ports() const override {
        return {{"loss", PortType::AD_TENSOR}};
    }
    std::vector<PortDescriptor> output_ports() const override {
        return {};
    }
    json default_config() const override { return json::object(); }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>& inputs) override {
        auto loss = get_input<std::shared_ptr<ADTensor>>(inputs, "loss");
        loss->backward();
        return {};
    }
};

// ======================== TextInputWrapper ========================

class TextInputWrapper : public ModuleWrapper {
    std::string text;
public:
    TextInputWrapper(const json& config)
        : text(config.value("text", "Hello world")) {}

    std::string type_name() const override { return "TextInput"; }
    std::string category() const override { return "input"; }
    std::string description() const override {
        return "Provides a text string as input to the graph";
    }
    std::vector<PortDescriptor> input_ports() const override { return {}; }
    std::vector<PortDescriptor> output_ports() const override {
        return {{"text", PortType::TEXT}};
    }
    json default_config() const override {
        return {{"text", "Hello world"}};
    }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>&) override {
        return {{"text", PortValue(text)}};
    }
};

// ======================== IntInputWrapper ========================

class IntInputWrapper : public ModuleWrapper {
    int value;
public:
    IntInputWrapper(const json& config)
        : value(config.value("value", 8)) {}

    std::string type_name() const override { return "IntInput"; }
    std::string category() const override { return "input"; }
    std::string description() const override {
        return "Provides an integer value as input (e.g., sequence length)";
    }
    std::vector<PortDescriptor> input_ports() const override { return {}; }
    std::vector<PortDescriptor> output_ports() const override {
        return {{"value", PortType::INT}};
    }
    json default_config() const override {
        return {{"value", 8}};
    }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>&) override {
        return {{"value", PortValue(value)}};
    }
};

// ======================== TokenIDsInputWrapper ========================

class TokenIDsInputWrapper : public ModuleWrapper {
    std::vector<int> tokens;
public:
    TokenIDsInputWrapper(const json& config) {
        if (config.contains("tokens")) {
            tokens = config["tokens"].get<std::vector<int>>();
        } else {
            tokens = {1, 2, 3, 4};
        }
    }

    std::string type_name() const override { return "TokenIDsInput"; }
    std::string category() const override { return "input"; }
    std::string description() const override {
        return "Provides a sequence of token IDs as input";
    }
    std::vector<PortDescriptor> input_ports() const override { return {}; }
    std::vector<PortDescriptor> output_ports() const override {
        return {{"tokens", PortType::TOKEN_IDS}};
    }
    json default_config() const override {
        return {{"tokens", {1, 2, 3, 4}}};
    }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>&) override {
        return {{"tokens", PortValue(tokens)}};
    }
};

// ======================== SeqLenExtractor ========================

class SeqLenExtractorWrapper : public ModuleWrapper {
public:
    SeqLenExtractorWrapper(const json&) {}

    std::string type_name() const override { return "SeqLenExtractor"; }
    std::string category() const override { return "utility"; }
    std::string description() const override {
        return "Extracts sequence length from a token ID list";
    }
    std::vector<PortDescriptor> input_ports() const override {
        return {{"tokens", PortType::TOKEN_IDS}};
    }
    std::vector<PortDescriptor> output_ports() const override {
        return {{"seq_len", PortType::INT}};
    }
    json default_config() const override { return json::object(); }

    std::unordered_map<std::string, PortValue> execute(
        const std::unordered_map<std::string, PortValue>& inputs) override {
        auto tokens = get_input<std::vector<int>>(inputs, "tokens");
        return {{"seq_len", PortValue(static_cast<int>(tokens.size()))}};
    }
};

} // namespace server

// ======================== Registration helpers ========================

#include "server/module_registry.hpp"

namespace server {

void register_all_modules(ModuleRegistry& registry) {
    registry.register_module("TextInput",
        [](const json& c) { return std::make_unique<TextInputWrapper>(c); });
    registry.register_module("IntInput",
        [](const json& c) { return std::make_unique<IntInputWrapper>(c); });
    registry.register_module("TokenIDsInput",
        [](const json& c) { return std::make_unique<TokenIDsInputWrapper>(c); });
    registry.register_module("SeqLenExtractor",
        [](const json& c) { return std::make_unique<SeqLenExtractorWrapper>(c); });
    registry.register_module("Tokenizer",
        [](const json& c) { return std::make_unique<TokenizerWrapper>(c); });
    registry.register_module("ADEmbedding",
        [](const json& c) { return std::make_unique<ADEmbeddingWrapper>(c); });
    registry.register_module("ADPositionalEncoding",
        [](const json& c) { return std::make_unique<ADPosEncWrapper>(c); });
    registry.register_module("ADLayerNorm",
        [](const json& c) { return std::make_unique<ADLayerNormWrapper>(c); });
    registry.register_module("ADMultiHeadAttention",
        [](const json& c) { return std::make_unique<ADMHAWrapper>(c); });
    registry.register_module("ADFeedForward",
        [](const json& c) { return std::make_unique<ADFeedForwardWrapper>(c); });
    registry.register_module("ADMoE",
        [](const json& c) { return std::make_unique<ADMoEWrapper>(c); });
    registry.register_module("ADLinear",
        [](const json& c) { return std::make_unique<ADLinearWrapper>(c); });
    registry.register_module("ADTransformerBlock",
        [](const json& c) { return std::make_unique<ADTransBlockWrapper>(c); });
    registry.register_module("Add",
        [](const json& c) { return std::make_unique<AddWrapper>(c); });
    registry.register_module("MatMul",
        [](const json& c) { return std::make_unique<MatMulWrapper>(c); });
    registry.register_module("Transpose",
        [](const json& c) { return std::make_unique<TransposeWrapper>(c); });
    registry.register_module("CrossEntropy",
        [](const json& c) { return std::make_unique<CrossEntropyWrapper>(c); });
    registry.register_module("Backward",
        [](const json& c) { return std::make_unique<BackwardWrapper>(c); });
}

} // namespace server
