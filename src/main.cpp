#include "tokenizer.hpp"
#include "layers/ad_embedding.hpp"
#include "layers/ad_positional_encoding.hpp"
#include "layers/ad_transformer.hpp"
#include "layers/ad_linear.hpp"
#include "optimizer.hpp"
#include "autodiff.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <limits>
#include <cstdint>
#include "timer.hpp"
// Unified memory pool manager
// Unified memory pool manager
#include "memory_pool.hpp"
#include "quantization.hpp"
#include "loss.hpp"
// Inference modules (non-AD)
#include "layers/embedding.hpp"
#include "layers/positional_encoding.hpp"
#include "transformer.hpp"
#include "layers/linear.hpp"

// Checkpoint utilities: save/load model parameters
static bool save_checkpoint(const std::string& path) {
    auto& params = get_parameters();
    std::ofstream out(path, std::ios::binary);
    if (!out) { std::cerr << "Error: cannot open checkpoint file for writing: " << path << "\n"; return false; }
    uint32_t num = params.size();
    out.write(reinterpret_cast<const char*>(&num), sizeof(num));
    for (auto& p : params) {
        uint32_t r = (uint32_t)p->val.rows;
        uint32_t c = (uint32_t)p->val.cols;
        out.write(reinterpret_cast<const char*>(&r), sizeof(r));
        out.write(reinterpret_cast<const char*>(&c), sizeof(c));
        out.write(reinterpret_cast<const char*>(p->val.data.data()), r * c * sizeof(float));
    }
    return true;
}
static bool load_checkpoint(const std::string& path) {
    auto& params = get_parameters();
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "Error: cannot open checkpoint file for reading: " << path << "\n"; return false; }
    uint32_t num = 0;
    in.read(reinterpret_cast<char*>(&num), sizeof(num));
    if (num != params.size()) {
        std::cerr << "Warning: checkpoint parameter count mismatch (" << num << " vs " << params.size() << "), attempting partial load\n";
    }
    // Load available parameters
    uint32_t to_load = std::min<uint32_t>(num, (uint32_t)params.size());
    for (uint32_t i = 0; i < to_load; ++i) {
        auto& p = params[i];
        uint32_t r = 0, c = 0;
        in.read(reinterpret_cast<char*>(&r), sizeof(r));
        in.read(reinterpret_cast<char*>(&c), sizeof(c));
        if (r != (uint32_t)p->val.rows || c != (uint32_t)p->val.cols) {
            std::cerr << "Warning: checkpoint param shape mismatch (" << r << "x" << c
                      << " vs " << p->val.rows << "x" << p->val.cols << "), skipping\n";
            // skip data
            in.seekg((std::streamoff)r * c * sizeof(float), std::ios::cur);
        } else {
            in.read(reinterpret_cast<char*>(p->val.data.data()), r * c * sizeof(float));
        }
    }
    return true;
}

// Simple ASCII sparkline for loss visualization
static std::string sparkline(const std::vector<float>& data) {
    if (data.empty()) return std::string();
    static const std::vector<std::string> levels = {"▁","▂","▃","▄","▅","▆","▇","█"};
    float mn = *std::min_element(data.begin(), data.end());
    float mx = *std::max_element(data.begin(), data.end());
    float range = mx - mn;
    std::string s;
    for (auto v : data) {
        int idx = 0;
        if (range > 0.0f) {
            float norm = (v - mn) / range;
            idx = std::min<int>((int)levels.size()-1, std::max<int>(0, int(norm * (levels.size()-1) + 0.5f)));
        }
        s += levels[idx];
    }
    return s;
}

int main(int argc, char** argv) {
    // Default configuration
    std::string mode;
    std::string data_file;
    std::string vocab_file = "input_files/vocab.txt";
    std::string bpe_codes_file;
    int embed_dim = 64;
    int hidden_dim = 64;
    int n_heads = 4;
    int num_layers = 3;
    int max_len = 128;
    int seq_len = 32;
    int epochs = 5;
    int batch_size = 16;
    float lr = 1e-3f;
    std::string resume_file;
    std::string save_file = "checkpoint.bin";
    std::string valid_file;
    int patience = 2;
    // On-chip unified memory pool size (MB, 0 to disable)
    long pool_size_mb = 0;
    // Quantization parameters
    bool qat_enabled = false;
    int qat_bits = 8;
    std::string ptq_out;
    // Inference mode parameters
    std::string generate_file;
    int max_new_tokens = 32;
    // Sampling strategy: top-k or top-p (nucleus) sampling; defaults to greedy if both disabled
    int top_k = 0;
    float top_p = 0.0f;

    // Parse command-line args (manual parser)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--train" && i + 1 < argc) {
            mode = "train";
            data_file = argv[++i];
        } else if (arg == "--vocab" && i + 1 < argc) {
            vocab_file = argv[++i];
        } else if (arg == "--embed_dim" && i + 1 < argc) {
            embed_dim = std::stoi(argv[++i]);
        } else if (arg == "--hidden_dim" && i + 1 < argc) {
            hidden_dim = std::stoi(argv[++i]);
        } else if (arg == "--n_heads" && i + 1 < argc) {
            n_heads = std::stoi(argv[++i]);
        } else if (arg == "--num_layers" && i + 1 < argc) {
            num_layers = std::stoi(argv[++i]);
        } else if (arg == "--max_len" && i + 1 < argc) {
            max_len = std::stoi(argv[++i]);
        } else if (arg == "--seq_len" && i + 1 < argc) {
            seq_len = std::stoi(argv[++i]);
        } else if (arg == "--batch_size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            lr = std::stof(argv[++i]);
        } else if (arg == "--resume" && i + 1 < argc) {
            resume_file = argv[++i];
        } else if (arg == "--save" && i + 1 < argc) {
            save_file = argv[++i];
        } else if (arg == "--valid" && i + 1 < argc) {
            valid_file = argv[++i];
        } else if (arg == "--patience" && i + 1 < argc) {
            patience = std::stoi(argv[++i]);
        } else if (arg == "--generate" && i + 1 < argc) {
            mode = "generate";
            generate_file = argv[++i];
        } else if (arg == "--cli") {
            // Interactive CLI REPL mode
            mode = "cli";
        } else if (arg == "--max_new_tokens" && i + 1 < argc) {
            max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--top_k" && i + 1 < argc) {
            // Top-k sampling (greedy if 0)
            top_k = std::stoi(argv[++i]);
        } else if (arg == "--top_p" && i + 1 < argc) {
            // Top-p (nucleus) sampling (greedy if 0)
            top_p = std::stof(argv[++i]);
        } else if (arg == "--bpe-codes" && i + 1 < argc) {
            // Path to BPE merge rules file
            bpe_codes_file = argv[++i];
        } else if (arg == "--qat") {
            // Enable quantization-aware training
            qat_enabled = true;
        } else if (arg == "--qat-bits" && i + 1 < argc) {
            // Number of bits for quantization (default 8)
            qat_bits = std::stoi(argv[++i]);
        } else if (arg == "--ptq-out" && i + 1 < argc) {
            // Path to dump post-training quantized model
            ptq_out = argv[++i];
        } else if (arg == "--pool_size_mb" && i + 1 < argc) {
            // On-chip unified memory pool size in megabytes
            pool_size_mb = std::stol(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: deepseek_ai [--train data.txt] [--generate prompt.txt] [options]\n"
                      << "Modes:\n"
                      << "  --train PATH         train model on text data\n"
                      << "  --generate PATH      generate from prompt file (one-shot)\n"
                      << "Options:\n"
                      << "  --vocab PATH         vocabulary file (default: input_files/vocab.txt)\n"
                      << "  --embed_dim N        embedding dimension (default: 64)\n"
                      << "  --hidden_dim N       hidden dimension (default: 64)\n"
                      << "  --n_heads N          number of attention heads (default: 4)\n"
                      << "  --num_layers N       number of transformer layers (default: 3)\n"
                      << "  --max_len N          maximum sequence length (default: 128)\n"
                      << "  --seq_len N          training sequence length (default: 32)\n"
                      << "  --batch_size N       mini-batch size (default: 16)\n"
                      << "  --epochs N           number of training epochs (default: 5)\n"
                      << "  --lr FLOAT           learning rate (default: 1e-3)\n"
                      << "  --resume PATH        checkpoint file to load (default: none)\n"
                      << "  --save PATH          checkpoint file to save (default: checkpoint.bin)\n"
                      << "  --valid PATH         validation data file (default: none)\n"
                      << "  --patience N         early stopping patience (default: 2 epochs)\n"
                      << "  --bpe-codes PATH     BPE merges file for true BPE (optional)\n"
                      << "  --max_new_tokens N   maximum tokens to generate (default: 32)\n"
                      << "  --top_k N            top-k sampling (0=greedy)\n"
                      << "  --top_p FLOAT        top-p (nucleus) sampling (0=greedy)\n"
                      << "  --qat                enable quantization-aware training (fake quant)\n"
                      << "  --qat-bits N         bits for quantization (default: 8)\n"
                      << "  --ptq-out PATH       output path for post-training quantized model\n";
            return 0;
        } else {
            std::cerr << "Unknown option or missing argument: " << arg << "\n";
            return 1;
        }
    }

    // Initialize quantization settings
    quant::g_qat_enabled = qat_enabled;
    quant::g_qat_bits = qat_bits;
    if (quant::g_qat_enabled) {
        std::cout << "Quantization-aware training enabled (" << qat_bits << " bits)\n";
    }
    // Initialize unified memory pool if requested
    if (pool_size_mb > 0) {
        UnifiedMemoryManager::instance().init(pool_size_mb * 1024 * 1024);
        std::cout << "Initialized on-chip memory pool of size " << pool_size_mb << " MB\n";
    }
    // Interactive CLI REPL mode
    if (mode == "cli") {
        // Load tokenizer for CLI
        Tokenizer tokenizer(vocab_file, bpe_codes_file);
        // Build AD modules once
        ADEmbedding ad_embed(tokenizer.vocab_size(), embed_dim);
        ADPositionalEncoding ad_posenc(embed_dim, max_len);
        ADTransformer ad_transformer(num_layers, embed_dim, hidden_dim, n_heads);
        // Tied Embedding-LM head: share embedding weights
        auto W_embed = ad_embed.get_weights(); // [embed_dim x vocab_size]
        // Bias for LM head
        Tensor tb_lm((int)tokenizer.vocab_size(), 1);
        tb_lm.data.assign(tokenizer.vocab_size(), 0.0f);
        auto b_lm = make_ad(tb_lm);
        register_parameter(b_lm);
        // Load checkpoint
        if (!resume_file.empty()) {
            if (!load_checkpoint(resume_file)) return 1;
            std::cout << "Loaded checkpoint from " << resume_file << "\n";
        } else {
            if (!load_checkpoint(save_file)) return 1;
            std::cout << "Loaded checkpoint from " << save_file << "\n";
        }
        // Prepare RNG and EOS
        std::mt19937 gen(std::random_device{}());
        int eos_id = tokenizer.to_id("</s>");
        // REPL loop
        std::string line;
        while (true) {
            std::cout << ">> " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (line.empty() || line == "exit") break;
            // Tokenize input
            auto tokens = tokenizer.encode(line);
            std::vector<int> output_tokens = tokens;
            // Generate up to max_new_tokens
            for (int step = 0; step < max_new_tokens; ++step) {
                int context_len = std::min((int)output_tokens.size(), seq_len);
                std::vector<int> input_ids(output_tokens.end() - context_len, output_tokens.end());
                auto embed_ad = ad_embed.forward(input_ids);
                auto pos_ad = ad_posenc.forward(context_len);
                auto x_ad = add(embed_ad, pos_ad);
                auto h_ad = ad_transformer.forward(x_ad);
                // Compute logits via tied embedding weights: logits = W_embed^T * h_ad + b_lm
                auto Wt = transpose(W_embed);  // [vocab_size x embed_dim]
                auto logits_ad = matmul(Wt, h_ad);  // [vocab_size x seq_len]
                // Broadcast bias using existing context_len
                Tensor ones_bias_t(1, context_len);
                ones_bias_t.data.assign(context_len, 1.0f);
                auto ones_bias = make_ad(ones_bias_t);
                auto b_mat = matmul(b_lm, ones_bias);  // [vocab_size x seq_len]
                logits_ad = add(logits_ad, b_mat);
                Tensor logits = logits_ad->val;
                int V = (int)tokenizer.vocab_size();
                int last_idx = context_len - 1;
                std::vector<float> logit_v(V);
                for (int i = 0; i < V; ++i) logit_v[i] = logits(i, last_idx);
                int next_id = 0;
                if (top_k > 0) {
                    // top-k sampling
                    int k = std::min(top_k, V);
                    std::vector<int> idxs(V);
                    std::iota(idxs.begin(), idxs.end(), 0);
                    std::partial_sort(idxs.begin(), idxs.begin() + k, idxs.end(),
                                      [&](int a, int b) { return logit_v[a] > logit_v[b]; });
                    std::vector<char> mask(V, false);
                    for (int j = 0; j < k; ++j) mask[idxs[j]] = true;
                    std::vector<float> weights(V, 0.0f);
                    for (int i = 0; i < V; ++i) if (mask[i]) weights[i] = std::exp(logit_v[i]);
                    std::discrete_distribution<int> dist(weights.begin(), weights.end());
                    next_id = dist(gen);
                } else if (top_p > 0.0f) {
                    // top-p sampling
                    std::vector<float> probs(V);
                    for (int i = 0; i < V; ++i) probs[i] = std::exp(logit_v[i]);
                    float sum_probs = std::accumulate(probs.begin(), probs.end(), 0.0f);
                    std::vector<int> idxs(V);
                    std::iota(idxs.begin(), idxs.end(), 0);
                    std::sort(idxs.begin(), idxs.end(), [&](int a, int b) { return probs[a] > probs[b]; });
                    std::vector<char> mask(V, false);
                    float cum = 0.0f;
                    for (int j = 0; j < V; ++j) {
                        int i = idxs[j]; cum += probs[i]; mask[i] = true;
                        if (cum / sum_probs >= top_p) break;
                    }
                    std::vector<float> weights(V, 0.0f);
                    for (int i = 0; i < V; ++i) if (mask[i]) weights[i] = probs[i];
                    std::discrete_distribution<int> dist(weights.begin(), weights.end());
                    next_id = dist(gen);
                } else {
                    // greedy
                    next_id = std::max_element(logit_v.begin(), logit_v.end()) - logit_v.begin();
                }
                output_tokens.push_back(next_id);
                if (eos_id >= 0 && next_id == eos_id) break;
            }
            // Decode and print
            std::cout << tokenizer.decode(output_tokens) << std::endl;
        }
        return 0;
    }
    // One-shot generate mode
    if (mode == "generate") {
        // Load tokenizer and prompt
        Tokenizer tokenizer(vocab_file, bpe_codes_file);
        std::string prompt;
        if (!generate_file.empty()) {
            std::ifstream gin(generate_file);
            if (!gin) { std::cerr << "Cannot open prompt file: " << generate_file << "\n"; return 1; }
            std::ostringstream gss;
            gss << gin.rdbuf();
            prompt = gss.str();
        } else {
            std::cerr << "No prompt file provided for generation\n";
            return 1;
        }
        auto tokens = tokenizer.encode(prompt);
        // Build AD modules for inference and register parameters
        ADEmbedding ad_embed(tokenizer.vocab_size(), embed_dim);
        ADPositionalEncoding ad_posenc(embed_dim, max_len);
        ADTransformer ad_transformer(num_layers, embed_dim, hidden_dim, n_heads);
        // Tied Embedding-LM head: share embedding weights
        auto W_embed = ad_embed.get_weights(); // [embed_dim x vocab_size]
        // Bias for LM head
        Tensor tb_lm((int)tokenizer.vocab_size(), 1);
        tb_lm.data.assign(tokenizer.vocab_size(), 0.0f);
        auto b_lm = make_ad(tb_lm);
        register_parameter(b_lm);
        // Load checkpoint
        if (!resume_file.empty()) {
            if (!load_checkpoint(resume_file)) return 1;
            std::cout << "Loaded checkpoint from " << resume_file << "\n";
        } else {
            if (!load_checkpoint(save_file)) return 1;
            std::cout << "Loaded checkpoint from " << save_file << "\n";
        }
        // Generate tokens (sampling or greedy)
        // Random generator for sampling
        std::mt19937 gen(std::random_device{}());
        // EOS detection
        int eos_id = tokenizer.to_id("</s>");
        std::vector<int> output_tokens = tokens;
        for (int step = 0; step < max_new_tokens; ++step) {
            int context_len = std::min((int)output_tokens.size(), seq_len);
            std::vector<int> input_ids(output_tokens.end() - context_len, output_tokens.end());
            auto embed_ad = ad_embed.forward(input_ids);
            auto pos_ad = ad_posenc.forward(context_len);
            auto x_ad = add(embed_ad, pos_ad);
            auto h_ad = ad_transformer.forward(x_ad);
            // Compute logits via tied embedding weights
            auto Wt = transpose(W_embed);  // [vocab_size x embed_dim]
            auto logits_ad = matmul(Wt, h_ad);  // [vocab_size x seq_len]
            // Broadcast bias using existing context_len
            Tensor ones_bias_t(1, context_len);
            ones_bias_t.data.assign(context_len, 1.0f);
            auto ones_bias = make_ad(ones_bias_t);
            auto b_mat = matmul(b_lm, ones_bias);  // [vocab_size x seq_len]
            logits_ad = add(logits_ad, b_mat);
            Tensor logits = logits_ad->val;  // [vocab_size x seq_len]
            int V = (int)tokenizer.vocab_size();
            int last_idx = context_len - 1;
            // Extract logits for last position
            std::vector<float> logit_v(V);
            for (int i = 0; i < V; ++i) logit_v[i] = logits(i, last_idx);
            int next_id = 0;
            if (top_k > 0) {
                // Top-k sampling
                int k = std::min(top_k, V);
                std::vector<int> idxs(V);
                std::iota(idxs.begin(), idxs.end(), 0);
                std::partial_sort(idxs.begin(), idxs.begin() + k, idxs.end(),
                                  [&](int a, int b) { return logit_v[a] > logit_v[b]; });
                std::vector<char> mask(V, false);
                for (int j = 0; j < k; ++j) mask[idxs[j]] = true;
                std::vector<float> weights(V, 0.0f);
                for (int i = 0; i < V; ++i) if (mask[i]) weights[i] = std::exp(logit_v[i]);
                std::discrete_distribution<int> dist(weights.begin(), weights.end());
                next_id = dist(gen);
            } else if (top_p > 0.0f) {
                // Nucleus (top-p) sampling
                std::vector<float> probs(V);
                for (int i = 0; i < V; ++i) probs[i] = std::exp(logit_v[i]);
                float sum_probs = std::accumulate(probs.begin(), probs.end(), 0.0f);
                std::vector<int> idxs(V);
                std::iota(idxs.begin(), idxs.end(), 0);
                std::sort(idxs.begin(), idxs.end(),
                          [&](int a, int b) { return probs[a] > probs[b]; });
                std::vector<char> mask(V, false);
                float cum = 0.0f;
                for (int j = 0; j < V; ++j) {
                    int i = idxs[j];
                    cum += probs[i];
                    mask[i] = true;
                    if (cum / sum_probs >= top_p) break;
                }
                std::vector<float> weights(V, 0.0f);
                for (int i = 0; i < V; ++i) if (mask[i]) weights[i] = probs[i];
                std::discrete_distribution<int> dist(weights.begin(), weights.end());
                next_id = dist(gen);
            } else {
                // Greedy sampling
                next_id = std::max_element(logit_v.begin(), logit_v.end()) - logit_v.begin();
            }
            output_tokens.push_back(next_id);
            // Stop if EOS token generated
            if (eos_id >= 0 && next_id == eos_id) break;
        }
        // Decode and print generated tokens
        std::string result = tokenizer.decode(output_tokens);
        std::cout << result << std::endl;
        return 0;
    }
    // Training mode must be explicitly set
    if (mode != "train" || data_file.empty()) {
        std::cerr << "Usage: deepseek_ai --train data.txt [--vocab vocab.txt] [--seq_len N] ...\n";
        return 1;
    }
    // Print configuration
    std::cout << "Training on: " << data_file << "\n"
              << "Vocab file: " << vocab_file << "\n"
              << "embed_dim=" << embed_dim << " hidden_dim=" << hidden_dim
              << " n_heads=" << n_heads << " num_layers=" << num_layers << "\n"
              << "max_len=" << max_len << " seq_len=" << seq_len
              << " batch_size=" << batch_size << " epochs=" << epochs
              << " lr=" << lr << "\n"
              << "Validation file: " << (valid_file.empty() ? std::string("none") : valid_file) << "\n"
              << "Early stopping patience: " << patience << "\n";

    // Load tokenizer and dataset
    Tokenizer tokenizer(vocab_file, bpe_codes_file);
    std::ifstream in(data_file);
    if (!in) {
        std::cerr << "Cannot open data file: " << data_file << "\n";
        return 1;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();
    auto data_tokens = tokenizer.encode(text);
    int N = data_tokens.size();
    if (N < seq_len + 1) {
        std::cerr << "Not enough tokens in data (need > " << seq_len + 1 << ")\n";
        return 1;
    }
    // Prepare validation data if provided
    std::vector<int> val_tokens;
    std::vector<int> val_starts;
    if (!valid_file.empty()) {
        std::ifstream in_v(valid_file);
        if (!in_v) {
            std::cerr << "Cannot open validation file: " << valid_file << "\n";
            return 1;
        }
        std::ostringstream vss;
        vss << in_v.rdbuf();
        std::string vtext = vss.str();
        val_tokens = tokenizer.encode(vtext);
        int M = val_tokens.size();
        if (M < seq_len + 1) {
            std::cerr << "Not enough tokens in validation data (need > " << seq_len + 1 << ")\n";
            return 1;
        }
        for (int s = 0; s + seq_len < M; s += seq_len) {
            val_starts.push_back(s);
        }
    }

    // AD Model components
    ADEmbedding ad_embed(tokenizer.vocab_size(), embed_dim);
    ADPositionalEncoding ad_posenc(embed_dim, max_len);
    ADTransformer ad_transformer(num_layers, embed_dim, hidden_dim, n_heads);
    // Tied Embedding-LM head weights
    auto W_embed = ad_embed.get_weights(); // [embed_dim x vocab_size]
    // Bias for LM head
    int Vocab = (int)tokenizer.vocab_size();
    Tensor tb_lm(Vocab, 1);
    tb_lm.data.assign(Vocab, 0.0f);
    auto b_lm = make_ad(tb_lm);
    register_parameter(b_lm);
    // Optimizer: AdamW with default betas, eps, weight_decay=0.01, clip_norm=1.0
    AdamW optimizer(lr);
    // Load checkpoint if requested
    if (!resume_file.empty()) {
        if (!load_checkpoint(resume_file)) return 1;
        std::cout << "Loaded checkpoint from " << resume_file << "\n";
    }

    // Training loop with mini-batching and data shuffling
    int steps_per_epoch = (N - 1) / seq_len;
    // Prepare sequence start indices
    std::vector<int> starts;
    for (int s = 0; s + seq_len < N; s += seq_len) {
        starts.push_back(s);
    }
    std::vector<float> loss_history;
    std::vector<float> val_history;
    std::vector<float> grad_z;
    grad_z.reserve(tokenizer.vocab_size());
    // Random generator for shuffling
    std::mt19937 rng(1234);
    // Early stopping state
    int no_improve = 0;
    float best_val_loss = std::numeric_limits<float>::infinity();
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // Shuffle data indices
        std::shuffle(starts.begin(), starts.end(), rng);
        float total_loss = 0.0f;
        int count = 0;
        // Iterate over mini-batches
        for (size_t batch_start = 0; batch_start < starts.size(); batch_start += batch_size) {
            optimizer.zero_grad();
            size_t batch_end = std::min(batch_start + batch_size, starts.size());
            // Process each sequence in the batch
            for (size_t idx = batch_start; idx < batch_end; ++idx) {
                int start = starts[idx];
                // Prepare input and target
                std::vector<int> input_ids(data_tokens.begin() + start,
                                           data_tokens.begin() + start + seq_len);
                std::vector<int> target_ids(data_tokens.begin() + start + 1,
                                            data_tokens.begin() + start + seq_len + 1);
                // AD forward
                auto embed_ad = ad_embed.forward(input_ids);
                auto pos_ad   = ad_posenc.forward(seq_len);
                // Debug shapes
                std::cout << "DEBUG: embed shape: [" << embed_ad->val.rows
                          << "x" << embed_ad->val.cols << "]\n";
                std::cout << "DEBUG: pos shape:   [" << pos_ad->val.rows
                          << "x" << pos_ad->val.cols << "]\n";

                auto x_ad     = add(embed_ad, pos_ad);
                // Time the transformer forward pass
                std::shared_ptr<ADTensor> h_ad;
                {
                    Timer t("ADTransformer forward");
                    h_ad = ad_transformer.forward(x_ad);
                }
                // Compute logits via tied embedding weights
                auto Wt = transpose(W_embed);  // [vocab_size x embed_dim]
                auto logits_ad = matmul(Wt, h_ad);  // [vocab_size x seq_len]
                // Broadcast bias using fixed seq_len
                Tensor ones_bias_t(1, seq_len);
                ones_bias_t.data.assign(seq_len, 1.0f);
                auto ones_bias = make_ad(ones_bias_t);
                auto b_mat = matmul(b_lm, ones_bias);  // [vocab_size x seq_len]
                logits_ad = add(logits_ad, b_mat);
                // Build target one-hot [vocab x seq_len]
                int V = tokenizer.vocab_size();
                Tensor target_tensor(V, seq_len);
                target_tensor.data.assign(V * seq_len, 0.0f);
                for (int t = 0; t < seq_len; ++t) {
                    int id = target_ids[t];
                    if (id >= 0 && id < V) {
                        target_tensor.data[id * seq_len + t] = 1.0f;
                    }
                }
                auto target_ad = make_ad(target_tensor);
                // Cross-entropy loss via AD:
                // sum1 = sum(logits * target)  (sum of true class logits)
                auto prod_ad = mul(logits_ad, target_ad);
                auto sum1_ad = sum(prod_ad);
                // denom: sum_exp per column
                Tensor ones_row_t(1, V);
                ones_row_t.data.assign(V, 1.0f);
                auto ones_row = make_ad(ones_row_t);
                auto exp_ad_logits = exp_ad(logits_ad);
                auto denom_row = matmul(ones_row, exp_ad_logits);    // [1 x seq_len]
                // log denom per column
                auto log_denoms = log_ad(denom_row);                // [1 x seq_len]
                auto sum2_ad = sum(log_denoms);
                // loss = sum(log_denoms) - sum(true_logits)
                auto loss_ad = sub(sum2_ad, sum1_ad);
                // Backward (accumulate gradients)
                loss_ad->backward();
                // Accumulate loss
                float loss = loss_ad->val.data[0];
                total_loss += loss;
                ++count;
            }
            // Update parameters after processing the batch
            optimizer.step();
        }
        float avg_loss = total_loss / count;
        std::cout << "Epoch " << epoch << ": Avg XEnt loss = " << avg_loss << "\n";
        loss_history.push_back(avg_loss);
        // Save checkpoint after each epoch
        if (!save_file.empty()) {
            if (!save_checkpoint(save_file))
                std::cerr << "Error: failed saving checkpoint to " << save_file << "\n";
            else
                std::cout << "Saved checkpoint to " << save_file << "\n";
        }
        // Validation and early stopping
        if (!valid_file.empty()) {
            float val_loss = 0.0f;
            int val_count = 0;
            for (int vs : val_starts) {
                // Prepare input and target
                std::vector<int> inp(val_tokens.begin() + vs,
                                     val_tokens.begin() + vs + seq_len);
                std::vector<int> tgt(val_tokens.begin() + vs + 1,
                                     val_tokens.begin() + vs + seq_len + 1);
                auto embed_v = ad_embed.forward(inp);
                auto pos_v   = ad_posenc.forward(seq_len);
                auto x_v     = add(embed_v, pos_v);
                auto h_v     = ad_transformer.forward(x_v);
                // Compute validation logits via tied embedding weights
                auto Wt_v = transpose(W_embed);  // [vocab_size x embed_dim]
                auto logits_v = matmul(Wt_v, h_v);  // [vocab_size x seq_len]
                // Broadcast bias
                Tensor ones_row_v_t(1, seq_len);
                ones_row_v_t.data.assign(seq_len, 1.0f);
                auto ones_row_v = make_ad(ones_row_v_t);
                auto b_mat_v = matmul(b_lm, ones_row_v);  // [vocab_size x seq_len]
                logits_v = add(logits_v, b_mat_v);
                // Compute cross-entropy per time step
                int V = tokenizer.vocab_size();
                std::vector<float> temp_grad(V);
                for (int t = 0; t < seq_len; ++t) {
                    std::vector<float> logit_t(V);
                    for (int i = 0; i < V; ++i) {
                        logit_t[i] = logits_v->val.data[i * seq_len + t];
                    }
                    val_loss += softmax_cross_entropy(logit_t, tgt[t], temp_grad);
                    ++val_count;
                }
            }
            float avg_val = val_loss / val_count;
            std::cout << "Validation loss = " << avg_val << "\n";
            val_history.push_back(avg_val);
            if (avg_val < best_val_loss) {
                best_val_loss = avg_val;
                no_improve = 0;
            } else {
                no_improve++;
                std::cout << "No improvement (" << no_improve << "/" << patience << ")\n";
            }
            if (no_improve >= patience) {
                std::cout << "Early stopping at epoch " << epoch << "\n";
                break;
            }
        }
        // Display loss trends as sparkline
        std::cout << "Train trend: " << sparkline(loss_history) << "\n";
        if (!val_history.empty()) {
            std::cout << "Valid trend: " << sparkline(val_history) << "\n";
        }
    }
    std::cout << "Training complete.\n";
    // If requested, dump post-training quantized model
    if (!ptq_out.empty()) {
        std::ofstream oq(ptq_out, std::ios::binary);
        if (!oq) {
            std::cerr << "Error: cannot open PTQ output file: " << ptq_out << "\n";
            return 1;
        }
        auto& params = get_parameters();
        uint32_t num = (uint32_t)params.size();
        oq.write(reinterpret_cast<const char*>(&num), sizeof(num));
        for (auto& p : params) {
            uint32_t r = (uint32_t)p->val.rows;
            uint32_t c = (uint32_t)p->val.cols;
            float scale = 1.0f;
            std::vector<uint8_t> qdata;
            quant::post_training_quantize(p->val, qdata, scale);
            oq.write(reinterpret_cast<const char*>(&r), sizeof(r));
            oq.write(reinterpret_cast<const char*>(&c), sizeof(c));
            oq.write(reinterpret_cast<const char*>(&scale), sizeof(scale));
            oq.write(reinterpret_cast<const char*>(qdata.data()), qdata.size());
        }
        std::cout << "Wrote post-training quantized model to " << ptq_out << "\n";
        return 0;
    }
    return 0;
}