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
#include <string>

// Utility: softmax + cross-entropy loss and gradient
static float softmax_cross_entropy(const std::vector<float>& logits,
                                   int target, std::vector<float>& grad) {
    int V = logits.size();
    // compute max for numeric stability
    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> exps(V);
    float sum_exp = 0.0f;
    for (int i = 0; i < V; ++i) {
        exps[i] = std::exp(logits[i] - max_logit);
        sum_exp += exps[i];
    }
    float loss = -std::log(exps[target] / sum_exp);
    // gradient dL/dz = p - y
    for (int i = 0; i < V; ++i) {
        float p = exps[i] / sum_exp;
        grad[i] = p - (i == target ? 1.0f : 0.0f);
    }
    return loss;
}

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--train") {
        std::cout << "Usage: deepseek_ai --train data.txt\n";
        return 1;
    }
    // Training settings
    const std::string data_file = argv[2];
    const int embed_dim = 64;
    const int hidden_dim = 64;
    const int n_heads = 4;
    const int num_layers = 3;
    const int max_len = 128;
    const int seq_len = 32;
    const int epochs = 5;
    const float lr = 1e-3f;

    // Load tokenizer and dataset
    Tokenizer tokenizer("input_files/vocab.txt");
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

    // AD Model components
    ADEmbedding ad_embed(tokenizer.vocab_size(), embed_dim);
    ADPositionalEncoding ad_posenc(embed_dim, max_len);
    ADTransformer ad_transformer(num_layers, embed_dim, hidden_dim, n_heads);
    ADLinear ad_lm_head(embed_dim, (int)tokenizer.vocab_size());
    // Optimizer: AdamW with default betas, eps, weight_decay=0.01, clip_norm=1.0
    AdamW optimizer(lr);

    // Training loop
    int steps_per_epoch = (N - 1) / seq_len;
    std::vector<float> loss_history;
    std::vector<float> grad_z;
    grad_z.reserve(tokenizer.vocab_size());
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float total_loss = 0.0f;
        int count = 0;
        for (int start = 0; start + seq_len < N; start += seq_len) {
            // Zero gradients
            optimizer.zero_grad();
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
            auto h_ad     = ad_transformer.forward(x_ad);
            auto logits_ad = ad_lm_head.forward(h_ad);
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
            // Backward
            loss_ad->backward();
            // Update parameters
            optimizer.step();
            // Accumulate loss
            float loss = loss_ad->val.data[0];
            total_loss += loss;
            ++count;
        }
        float avg_loss = total_loss / count;
        std::cout << "Epoch " << epoch << ": Avg XEnt loss = " << avg_loss << "\n";
        loss_history.push_back(avg_loss);
    }
    std::cout << "Training complete.\n";
    return 0;
}