#include "tokenizer.hpp"
#include "layers/ad_embedding.hpp"
#include "layers/ad_positional_encoding.hpp"
#include "layers/ad_transformer.hpp"
#include "layers/ad_linear.hpp"
#include "optimizer.hpp"
#include "autodiff.hpp"
#include "lr_scheduler.hpp"
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
#include "memory_pool.hpp"
#include "quantization.hpp"
#include "loss.hpp"
#include "layers/embedding.hpp"
#include "layers/positional_encoding.hpp"
#include "transformer.hpp"
#include "layers/linear.hpp"

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
    uint32_t to_load = std::min<uint32_t>(num, (uint32_t)params.size());
    int shape_mismatches = 0;
    for (uint32_t i = 0; i < to_load; ++i) {
        auto& p = params[i];
        uint32_t r = 0, c = 0;
        in.read(reinterpret_cast<char*>(&r), sizeof(r));
        in.read(reinterpret_cast<char*>(&c), sizeof(c));
        if (r != (uint32_t)p->val.rows || c != (uint32_t)p->val.cols) {
            std::cerr << "Warning: checkpoint param shape mismatch (" << r << "x" << c
                      << " vs " << p->val.rows << "x" << p->val.cols << "), skipping\n";
            in.seekg((std::streamoff)r * c * sizeof(float), std::ios::cur);
            ++shape_mismatches;
            if (shape_mismatches > 3) {
                std::cerr << "Error: too many shape mismatches (>3), aborting checkpoint load\n";
                return false;
            }
        } else {
            in.read(reinterpret_cast<char*>(p->val.data.data()), r * c * sizeof(float));
        }
    }
    return true;
}

static int sample_next_token(const std::vector<float>& logits_in,
                             int top_k, float top_p, float temperature,
                             std::mt19937& rng) {
    int V = (int)logits_in.size();
    std::vector<float> logit_v(V);
    for (int i = 0; i < V; ++i) logit_v[i] = logits_in[i] / temperature;

    float max_logit = *std::max_element(logit_v.begin(), logit_v.end());

    if (top_k > 0) {
        int k = std::min(top_k, V);
        std::vector<int> idxs(V);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::partial_sort(idxs.begin(), idxs.begin() + k, idxs.end(),
                          [&](int a, int b) { return logit_v[a] > logit_v[b]; });
        std::vector<float> weights(V, 0.0f);
        for (int j = 0; j < k; ++j)
            weights[idxs[j]] = std::exp(logit_v[idxs[j]] - max_logit);
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        return dist(rng);
    } else if (top_p > 0.0f) {
        std::vector<float> probs(V);
        for (int i = 0; i < V; ++i) probs[i] = std::exp(logit_v[i] - max_logit);
        float sum_probs = std::accumulate(probs.begin(), probs.end(), 0.0f);
        std::vector<int> idxs(V);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::sort(idxs.begin(), idxs.end(), [&](int a, int b) { return probs[a] > probs[b]; });
        std::vector<float> weights(V, 0.0f);
        float cum = 0.0f;
        for (int j = 0; j < V; ++j) {
            int i = idxs[j]; cum += probs[i]; weights[i] = probs[i];
            if (cum / sum_probs >= top_p) break;
        }
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        return dist(rng);
    } else {
        // Greedy
        return std::max_element(logit_v.begin(), logit_v.end()) - logit_v.begin();
    }
}

struct GenerateConfig {
    int max_new_tokens;
    int seq_len;
    int top_k;
    float top_p;
    float temperature;
    int eos_id;
    int beam_width;
};

// Beam search: explore multiple hypotheses in parallel
static std::vector<int> beam_search_cached(
    const std::vector<int>& prompt_tokens,
    Embedding& embed_layer,
    PositionalEncoding& posenc,
    Transformer& transformer,
    const Tensor& out_W,
    const Tensor& out_b,
    int vocab_size,
    const GenerateConfig& cfg) {

    struct Beam {
        std::vector<int> tokens;
        float score;
    };

    int beam_width = cfg.beam_width;
    std::vector<Beam> beams = {{prompt_tokens, 0.0f}};

    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        std::vector<Beam> candidates;
        for (auto& beam : beams) {
            if (cfg.eos_id >= 0 && !beam.tokens.empty() && beam.tokens.back() == cfg.eos_id) {
                candidates.push_back(beam);
                continue;
            }
            // Run inference for this beam's full sequence (non-cached for simplicity)
            int context_len = std::min((int)beam.tokens.size(), cfg.seq_len);
            std::vector<int> input_ids(beam.tokens.end() - context_len, beam.tokens.end());
            Tensor x = embed_layer.forward(input_ids);
            Tensor pos = posenc.forward(context_len);
            for (size_t i = 0; i < x.data.size(); ++i) x.data[i] += pos.data[i];
            Tensor h = transformer.forward(x, false, false);
            int last_idx = context_len - 1;
            Tensor h_last(h.rows, 1);
            for (int r = 0; r < h.rows; ++r) h_last.data[r] = h(r, last_idx);
            Tensor logits = out_W.matmul(h_last);
            for (int i = 0; i < vocab_size; ++i) logits.data[i] += out_b.data[i];

            // Log-softmax
            float max_l = *std::max_element(logits.data.begin(), logits.data.begin() + vocab_size);
            float sum_exp = 0.0f;
            for (int i = 0; i < vocab_size; ++i) sum_exp += std::exp(logits.data[i] - max_l);
            float log_sum = max_l + std::log(sum_exp);

            // Expand top beam_width tokens
            std::vector<int> idxs(vocab_size);
            std::iota(idxs.begin(), idxs.end(), 0);
            std::partial_sort(idxs.begin(), idxs.begin() + beam_width, idxs.end(),
                              [&](int a, int b) { return logits.data[a] > logits.data[b]; });
            for (int k = 0; k < beam_width; ++k) {
                int tok = idxs[k];
                float log_prob = logits.data[tok] - log_sum;
                Beam new_beam;
                new_beam.tokens = beam.tokens;
                new_beam.tokens.push_back(tok);
                new_beam.score = beam.score + log_prob;
                candidates.push_back(std::move(new_beam));
            }
        }
        // Keep top beam_width candidates
        std::partial_sort(candidates.begin(),
                          candidates.begin() + std::min(beam_width, (int)candidates.size()),
                          candidates.end(),
                          [](const Beam& a, const Beam& b) { return a.score > b.score; });
        if ((int)candidates.size() > beam_width) candidates.resize(beam_width);
        beams = std::move(candidates);

        // Check if all beams ended
        bool all_done = true;
        for (auto& b : beams) {
            if (cfg.eos_id < 0 || b.tokens.back() != cfg.eos_id) { all_done = false; break; }
        }
        if (all_done) break;
    }
    return beams[0].tokens;
}

static std::vector<int> generate_tokens(
    const std::vector<int>& prompt_tokens,
    ADEmbedding& ad_embed,
    ADPositionalEncoding& ad_posenc,
    ADTransformer& ad_transformer,
    const std::shared_ptr<ADTensor>& W_embed,
    const std::shared_ptr<ADTensor>& b_lm,
    int vocab_size,
    const GenerateConfig& cfg,
    std::mt19937& rng) {
    std::vector<int> output_tokens = prompt_tokens;
    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        int context_len = std::min((int)output_tokens.size(), cfg.seq_len);
        std::vector<int> input_ids(output_tokens.end() - context_len, output_tokens.end());
        auto embed_ad = ad_embed.forward(input_ids);
        auto pos_ad = ad_posenc.forward(context_len);
        auto x_ad = add(embed_ad, pos_ad);
        auto h_ad = ad_transformer.forward(x_ad);
        auto Wt = transpose(W_embed);
        auto logits_ad = matmul(Wt, h_ad);
        Tensor ones_bias_t(1, context_len);
        ones_bias_t.data.assign(context_len, 1.0f);
        auto ones_bias = make_ad(ones_bias_t);
        auto b_mat = matmul(b_lm, ones_bias);
        logits_ad = add(logits_ad, b_mat);
        Tensor logits = logits_ad->val;
        int last_idx = context_len - 1;
        std::vector<float> logit_v(vocab_size);
        for (int i = 0; i < vocab_size; ++i) logit_v[i] = logits(i, last_idx);
        int next_id = sample_next_token(logit_v, cfg.top_k, cfg.top_p, cfg.temperature, rng);
        output_tokens.push_back(next_id);
        if (cfg.eos_id >= 0 && next_id == cfg.eos_id) break;
    }
    return output_tokens;
}

// 12 params per block: ln1(gamma,beta), mha(Wq,Wk,Wv,Wo), ln2(gamma,beta), ff(W1,b1,W2,b2)
static void sync_ad_to_inference(
    const std::vector<std::shared_ptr<ADTensor>>& params,
    int embed_param_idx,       // index of embedding weight param
    int block_start_idx,       // index where first block's params start
    int lm_bias_idx,           // index of b_lm param
    int num_layers,
    Embedding& embed,
    Transformer& transformer,
    Tensor& out_W,
    Tensor& out_b) {
    embed.weights = params[embed_param_idx]->val;
    int params_per_block = 12;
    for (int layer = 0; layer < num_layers; ++layer) {
        int base = block_start_idx + layer * params_per_block;
        auto& block = transformer.blocks[layer];
        block.ln1.gamma = params[base+0]->val;
        block.ln1.beta  = params[base+1]->val;
        block.mha.W_q = params[base+2]->val;
        block.mha.W_k = params[base+3]->val;
        block.mha.W_v = params[base+4]->val;
        block.mha.W_o = params[base+5]->val;
        block.ln2.gamma = params[base+6]->val;
        block.ln2.beta  = params[base+7]->val;
        block.ff.fc1.weights = params[base+8]->val;
        block.ff.fc1.bias    = params[base+9]->val;
        block.ff.fc2.weights = params[base+10]->val;
        block.ff.fc2.bias    = params[base+11]->val;
    }
    out_W = params[embed_param_idx]->val.transpose();
    out_b = params[lm_bias_idx]->val;
}

static std::vector<int> generate_tokens_cached(
    const std::vector<int>& prompt_tokens,
    Embedding& embed_layer,
    PositionalEncoding& posenc,
    Transformer& transformer,
    const Tensor& out_W,
    const Tensor& out_b,
    int vocab_size,
    const GenerateConfig& cfg,
    std::mt19937& rng) {
    transformer.clear_cache();
    std::vector<int> output_tokens = prompt_tokens;

    // prefill
    {
        Tensor x = embed_layer.forward(prompt_tokens);
        Tensor pos = posenc.forward((int)prompt_tokens.size());
        for (size_t i = 0; i < x.data.size(); ++i) x.data[i] += pos.data[i];
        Tensor h = transformer.forward(x, false, true);
        int last_idx = (int)prompt_tokens.size() - 1;
        Tensor h_last(h.rows, 1);
        for (int r = 0; r < h.rows; ++r) h_last.data[r] = h(r, last_idx);
        Tensor logits = out_W.matmul(h_last);
        for (int i = 0; i < vocab_size; ++i) logits.data[i] += out_b.data[i];
        std::vector<float> logit_v(vocab_size);
        for (int i = 0; i < vocab_size; ++i) logit_v[i] = logits.data[i];
        int next_id = sample_next_token(logit_v, cfg.top_k, cfg.top_p, cfg.temperature, rng);
        output_tokens.push_back(next_id);
        if (cfg.eos_id >= 0 && next_id == cfg.eos_id)
            return output_tokens;
    }

    // decode one token at a time
    for (int step = 1; step < cfg.max_new_tokens; ++step) {
        int token = output_tokens.back();
        Tensor x = embed_layer.forward({token});
        int pos_idx = (int)output_tokens.size() - 1;
        Tensor pos = posenc.forward(pos_idx + 1);
        for (int r = 0; r < x.rows; ++r)
            x.data[r] += pos(r, pos_idx);
        Tensor h = transformer.forward(x, false, true);
        Tensor logits = out_W.matmul(h);
        for (int i = 0; i < vocab_size; ++i) logits.data[i] += out_b.data[i];
        std::vector<float> logit_v(vocab_size);
        for (int i = 0; i < vocab_size; ++i) logit_v[i] = logits.data[i];
        int next_id = sample_next_token(logit_v, cfg.top_k, cfg.top_p, cfg.temperature, rng);
        output_tokens.push_back(next_id);
        if (cfg.eos_id >= 0 && next_id == cfg.eos_id) break;
    }
    return output_tokens;
}

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
    long pool_size_mb = 0;
    bool qat_enabled = false;
    int qat_bits = 8;
    std::string ptq_out;
    std::string generate_file;
    int max_new_tokens = 32;
    int top_k = 0;
    float top_p = 0.0f;
    float temperature = 1.0f;
    bool use_moe = false;
    int moe_num_experts = 4;
    int moe_top_k_experts = 2;
    float moe_aux_weight = 0.01f;
    // New feature flags
    bool use_rmsnorm = false;
    bool use_swiglu = false;
    bool use_rope = false;
    int warmup_steps = 0;
    std::string lr_schedule = "constant";
    int grad_accum_steps = 1;
    int beam_width = 0;

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
            mode = "cli";
        } else if (arg == "--max_new_tokens" && i + 1 < argc) {
            max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--top_k" && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        } else if (arg == "--top_p" && i + 1 < argc) {
            top_p = std::stof(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        } else if (arg == "--bpe-codes" && i + 1 < argc) {
            bpe_codes_file = argv[++i];
        } else if (arg == "--qat") {
            qat_enabled = true;
        } else if (arg == "--qat-bits" && i + 1 < argc) {
            qat_bits = std::stoi(argv[++i]);
        } else if (arg == "--ptq-out" && i + 1 < argc) {
            ptq_out = argv[++i];
        } else if (arg == "--pool_size_mb" && i + 1 < argc) {
            pool_size_mb = std::stol(argv[++i]);
        } else if (arg == "--timer") {
            Timer::enabled = true;
        } else if (arg == "--moe") {
            use_moe = true;
        } else if (arg == "--num_experts" && i + 1 < argc) {
            moe_num_experts = std::stoi(argv[++i]);
        } else if (arg == "--moe_top_k" && i + 1 < argc) {
            moe_top_k_experts = std::stoi(argv[++i]);
        } else if (arg == "--moe_aux_weight" && i + 1 < argc) {
            moe_aux_weight = std::stof(argv[++i]);
        } else if (arg == "--rmsnorm") {
            use_rmsnorm = true;
        } else if (arg == "--swiglu") {
            use_swiglu = true;
        } else if (arg == "--rope") {
            use_rope = true;
        } else if (arg == "--warmup_steps" && i + 1 < argc) {
            warmup_steps = std::stoi(argv[++i]);
        } else if (arg == "--lr_schedule" && i + 1 < argc) {
            lr_schedule = argv[++i];
        } else if (arg == "--grad_accum" && i + 1 < argc) {
            grad_accum_steps = std::stoi(argv[++i]);
        } else if (arg == "--beam_width" && i + 1 < argc) {
            beam_width = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: deepseek_ai [--train data.txt] [--generate prompt.txt] [options]\n"
                      << "Modes:\n"
                      << "  --train PATH         train model on text data\n"
                      << "  --generate PATH      generate from prompt file (one-shot)\n"
                      << "  --cli                interactive generation mode\n"
                      << "\nModel architecture:\n"
                      << "  --embed_dim N        embedding dimension (default: 64)\n"
                      << "  --hidden_dim N       hidden dimension (default: 64)\n"
                      << "  --n_heads N          number of attention heads (default: 4)\n"
                      << "  --num_layers N       number of transformer layers (default: 3)\n"
                      << "  --max_len N          maximum sequence length (default: 128)\n"
                      << "  --rmsnorm            use RMSNorm instead of LayerNorm (LLaMA-style)\n"
                      << "  --swiglu             use SwiGLU activation instead of GELU (LLaMA-style)\n"
                      << "  --rope               use Rotary Position Embeddings (LLaMA-style)\n"
                      << "\nTraining:\n"
                      << "  --vocab PATH         vocabulary file (default: input_files/vocab.txt)\n"
                      << "  --bpe-codes PATH     BPE merges file for true BPE (optional)\n"
                      << "  --seq_len N          training sequence length (default: 32)\n"
                      << "  --batch_size N       mini-batch size (default: 16)\n"
                      << "  --epochs N           number of training epochs (default: 5)\n"
                      << "  --lr FLOAT           learning rate (default: 1e-3)\n"
                      << "  --lr_schedule TYPE   LR schedule: constant|cosine (default: constant)\n"
                      << "  --warmup_steps N     linear warmup steps (default: 0)\n"
                      << "  --grad_accum N       gradient accumulation steps (default: 1)\n"
                      << "  --resume PATH        checkpoint file to load (default: none)\n"
                      << "  --save PATH          checkpoint file to save (default: checkpoint.bin)\n"
                      << "  --valid PATH         validation data file (default: none)\n"
                      << "  --patience N         early stopping patience (default: 2 epochs)\n"
                      << "\nGeneration:\n"
                      << "  --max_new_tokens N   maximum tokens to generate (default: 32)\n"
                      << "  --top_k N            top-k sampling (0=greedy)\n"
                      << "  --top_p FLOAT        top-p (nucleus) sampling (0=greedy)\n"
                      << "  --temperature FLOAT  sampling temperature (default: 1.0)\n"
                      << "  --beam_width N       beam search width (0=disabled, default: 0)\n"
                      << "\nQuantization:\n"
                      << "  --qat                enable quantization-aware training (fake quant)\n"
                      << "  --qat-bits N         bits for quantization (default: 8)\n"
                      << "  --ptq-out PATH       output path for post-training quantized model\n"
                      << "\nMixture of Experts:\n"
                      << "  --moe                enable Mixture of Experts\n"
                      << "  --num_experts N      number of MoE experts (default: 4)\n"
                      << "  --moe_top_k N        experts per token (default: 2)\n"
                      << "  --moe_aux_weight F   aux loss weight (default: 0.01)\n"
                      << "\nMisc:\n"
                      << "  --pool_size_mb N     memory pool size in MB (default: 0=disabled)\n"
                      << "  --timer              enable performance timers\n";
            return 0;
        } else {
            std::cerr << "Unknown option or missing argument: " << arg << "\n";
            return 1;
        }
    }

    // Build transformer config
    TransformerConfig tcfg;
    tcfg.embed_dim = embed_dim;
    tcfg.hidden_dim = hidden_dim;
    tcfg.n_heads = n_heads;
    tcfg.num_layers = num_layers;
    tcfg.use_moe = use_moe;
    tcfg.num_experts = moe_num_experts;
    tcfg.moe_top_k = moe_top_k_experts;
    tcfg.use_rmsnorm = use_rmsnorm;
    tcfg.use_swiglu = use_swiglu;
    tcfg.use_rope = use_rope;

    quant::g_qat_enabled = qat_enabled;
    quant::g_qat_bits = qat_bits;
    if (quant::g_qat_enabled) {
        std::cout << "Quantization-aware training enabled (" << qat_bits << " bits)\n";
    }
    if (pool_size_mb > 0) {
        if (pool_size_mb > 16384) {
            std::cerr << "Error: pool_size_mb too large (max 16384 MB)\n";
            return 1;
        }
        UnifiedMemoryManager::instance().init(static_cast<size_t>(pool_size_mb) * 1024 * 1024);
        std::cout << "Initialized on-chip memory pool of size " << pool_size_mb << " MB\n";
    }
    if (mode == "cli") {
        Tokenizer tokenizer(vocab_file, bpe_codes_file);
        int V = (int)tokenizer.vocab_size();
        ADEmbedding ad_embed(V, embed_dim);
        ADPositionalEncoding ad_posenc(embed_dim, max_len);
        ADTransformer ad_transformer(tcfg);
        auto W_embed = ad_embed.get_weights();
        Tensor tb_lm(V, 1); tb_lm.data.assign(V, 0.0f);
        auto b_lm = make_ad(tb_lm); register_parameter(b_lm);
        if (!resume_file.empty()) {
            if (!load_checkpoint(resume_file)) return 1;
            std::cout << "Loaded checkpoint from " << resume_file << "\n";
        } else {
            if (!load_checkpoint(save_file)) return 1;
            std::cout << "Loaded checkpoint from " << save_file << "\n";
        }
        Embedding inf_embed(V, embed_dim);
        PositionalEncoding inf_posenc(embed_dim, max_len);
        Transformer inf_transformer(num_layers, embed_dim, hidden_dim, n_heads);
        Tensor out_W(V, embed_dim), out_b(V, 1);
        auto& params = get_parameters();
        // param layout: embed(0), blocks start at 1, b_lm is last
        sync_ad_to_inference(params, 0, 1, (int)params.size()-1, num_layers,
                             inf_embed, inf_transformer, out_W, out_b);
        std::mt19937 gen(std::random_device{}());
        GenerateConfig cfg{max_new_tokens, seq_len, top_k, top_p, temperature,
                           tokenizer.to_id("</s>"), beam_width};
        std::string line;
        while (true) {
            std::cout << ">> " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (line.empty() || line == "exit") break;
            auto tokens = tokenizer.encode(line);
            std::vector<int> output_tokens;
            if (beam_width > 0) {
                output_tokens = beam_search_cached(tokens, inf_embed, inf_posenc,
                    inf_transformer, out_W, out_b, V, cfg);
            } else {
                output_tokens = generate_tokens_cached(tokens, inf_embed, inf_posenc,
                    inf_transformer, out_W, out_b, V, cfg, gen);
            }
            std::cout << tokenizer.decode(output_tokens) << std::endl;
        }
        return 0;
    }
    if (mode == "generate") {
        Tokenizer tokenizer(vocab_file, bpe_codes_file);
        int V = (int)tokenizer.vocab_size();
        std::string prompt;
        if (!generate_file.empty()) {
            std::ifstream gin(generate_file);
            if (!gin) { std::cerr << "Cannot open prompt file: " << generate_file << "\n"; return 1; }
            std::ostringstream gss; gss << gin.rdbuf(); prompt = gss.str();
        } else {
            std::cerr << "No prompt file provided for generation\n"; return 1;
        }
        auto tokens = tokenizer.encode(prompt);
        ADEmbedding ad_embed(V, embed_dim);
        ADPositionalEncoding ad_posenc(embed_dim, max_len);
        ADTransformer ad_transformer(tcfg);
        auto W_embed = ad_embed.get_weights();
        Tensor tb_lm(V, 1); tb_lm.data.assign(V, 0.0f);
        auto b_lm = make_ad(tb_lm); register_parameter(b_lm);
        if (!resume_file.empty()) {
            if (!load_checkpoint(resume_file)) return 1;
            std::cout << "Loaded checkpoint from " << resume_file << "\n";
        } else {
            if (!load_checkpoint(save_file)) return 1;
            std::cout << "Loaded checkpoint from " << save_file << "\n";
        }
        Embedding inf_embed(V, embed_dim);
        PositionalEncoding inf_posenc(embed_dim, max_len);
        Transformer inf_transformer(num_layers, embed_dim, hidden_dim, n_heads);
        Tensor out_W(V, embed_dim), out_b(V, 1);
        auto& params = get_parameters();
        sync_ad_to_inference(params, 0, 1, (int)params.size()-1, num_layers,
                             inf_embed, inf_transformer, out_W, out_b);
        std::mt19937 gen(std::random_device{}());
        GenerateConfig cfg{max_new_tokens, seq_len, top_k, top_p, temperature,
                           tokenizer.to_id("</s>"), beam_width};
        std::vector<int> output_tokens;
        if (beam_width > 0) {
            output_tokens = beam_search_cached(tokens, inf_embed, inf_posenc,
                inf_transformer, out_W, out_b, V, cfg);
        } else {
            output_tokens = generate_tokens_cached(tokens, inf_embed, inf_posenc,
                inf_transformer, out_W, out_b, V, cfg, gen);
        }
        std::cout << tokenizer.decode(output_tokens) << std::endl;
        return 0;
    }
    if (mode != "train" || data_file.empty()) {
        std::cerr << "Usage: deepseek_ai --train data.txt [--vocab vocab.txt] [--seq_len N] ...\n";
        return 1;
    }
    std::cout << "Training on: " << data_file << "\n"
              << "Vocab file: " << vocab_file << "\n"
              << "embed_dim=" << embed_dim << " hidden_dim=" << hidden_dim
              << " n_heads=" << n_heads << " num_layers=" << num_layers << "\n"
              << "max_len=" << max_len << " seq_len=" << seq_len
              << " batch_size=" << batch_size << " epochs=" << epochs
              << " lr=" << lr << "\n"
              << "Validation file: " << (valid_file.empty() ? std::string("none") : valid_file) << "\n"
              << "Early stopping patience: " << patience << "\n";
    if (use_moe) {
        std::cout << "MoE enabled: " << moe_num_experts << " experts, top-"
                  << moe_top_k_experts << ", aux_weight=" << moe_aux_weight << "\n";
    }
    if (use_rmsnorm) std::cout << "Using RMSNorm (LLaMA-style)\n";
    if (use_swiglu) std::cout << "Using SwiGLU activation (LLaMA-style)\n";
    if (use_rope) std::cout << "Using RoPE (Rotary Position Embeddings)\n";
    if (lr_schedule == "cosine") {
        std::cout << "LR schedule: cosine annealing, warmup=" << warmup_steps << " steps\n";
    }
    if (grad_accum_steps > 1) {
        std::cout << "Gradient accumulation: " << grad_accum_steps << " steps"
                  << " (effective batch size=" << batch_size * grad_accum_steps << ")\n";
    }

    Tokenizer tokenizer(vocab_file, bpe_codes_file);
    std::string text;
    try {
        std::ifstream in(data_file);
        if (!in) {
            std::cerr << "Cannot open data file: " << data_file << "\n";
            return 1;
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        if (in.bad()) {
            std::cerr << "I/O error reading data file: " << data_file << "\n";
            return 1;
        }
        text = ss.str();
    } catch (const std::exception& e) {
        std::cerr << "Error reading data file: " << e.what() << "\n";
        return 1;
    }
    auto data_tokens = tokenizer.encode(text);
    int N = data_tokens.size();
    if (N < seq_len + 1) {
        std::cerr << "Not enough tokens in data (need > " << seq_len + 1 << ")\n";
        return 1;
    }
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

    ADEmbedding ad_embed(tokenizer.vocab_size(), embed_dim);
    ADPositionalEncoding ad_posenc(embed_dim, max_len);
    ADTransformer ad_transformer(tcfg);
    auto W_embed = ad_embed.get_weights();
    int Vocab = (int)tokenizer.vocab_size();
    Tensor tb_lm(Vocab, 1);
    tb_lm.data.assign(Vocab, 0.0f);
    auto b_lm = make_ad(tb_lm);
    register_parameter(b_lm);
    AdamW optimizer(lr);
    if (!resume_file.empty()) {
        if (!load_checkpoint(resume_file)) return 1;
        std::cout << "Loaded checkpoint from " << resume_file << "\n";
    }

    std::vector<int> starts;
    for (int s = 0; s + seq_len < N; s += seq_len) {
        starts.push_back(s);
    }

    // Compute total training steps for LR scheduler
    int batches_per_epoch = ((int)starts.size() + batch_size - 1) / batch_size;
    int total_optimizer_steps = batches_per_epoch * epochs;
    if (grad_accum_steps > 1) {
        total_optimizer_steps = (batches_per_epoch / grad_accum_steps) * epochs;
    }
    LRScheduler lr_sched(lr, warmup_steps, total_optimizer_steps, lr * 0.1f);

    std::vector<float> loss_history;
    std::vector<float> val_history;
    std::mt19937 rng(1234);
    int no_improve = 0;
    float best_val_loss = std::numeric_limits<float>::infinity();
    int global_step = 0;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(starts.begin(), starts.end(), rng);
        float total_loss = 0.0f;
        int count = 0;
        int accum_count = 0;

        for (size_t batch_start = 0; batch_start < starts.size(); batch_start += batch_size) {
            if (accum_count == 0) {
                optimizer.zero_grad();
            }
            size_t batch_end = std::min(batch_start + batch_size, starts.size());
            for (size_t idx = batch_start; idx < batch_end; ++idx) {
                int start = starts[idx];
                std::vector<int> input_ids(data_tokens.begin() + start,
                                           data_tokens.begin() + start + seq_len);
                std::vector<int> target_ids(data_tokens.begin() + start + 1,
                                            data_tokens.begin() + start + seq_len + 1);
                auto embed_ad = ad_embed.forward(input_ids);
                auto pos_ad   = ad_posenc.forward(seq_len);
                auto x_ad     = add(embed_ad, pos_ad);
                std::shared_ptr<ADTensor> h_ad;
                std::shared_ptr<ADTensor> moe_aux_loss;
                {
                    Timer t("ADTransformer forward");
                    h_ad = ad_transformer.forward(x_ad, use_moe ? &moe_aux_loss : nullptr);
                }
                auto Wt = transpose(W_embed);
                auto logits_ad = matmul(Wt, h_ad);
                Tensor ones_bias_t(1, seq_len);
                ones_bias_t.data.assign(seq_len, 1.0f);
                auto ones_bias = make_ad(ones_bias_t);
                auto b_mat = matmul(b_lm, ones_bias);
                logits_ad = add(logits_ad, b_mat);
                int V = (int)tokenizer.vocab_size();
                Tensor target_tensor(V, seq_len);
                target_tensor.data.assign(V * seq_len, 0.0f);
                for (int t = 0; t < seq_len; ++t) {
                    int id = target_ids[t];
                    if (id >= 0 && id < V) {
                        target_tensor.data[id * seq_len + t] = 1.0f;
                    }
                }
                auto target_ad = make_ad(target_tensor);
                // cross-entropy via log-sum-exp
                auto prod_ad = mul(logits_ad, target_ad);
                auto sum1_ad = sum(prod_ad);
                Tensor max_per_col(1, seq_len);
                for (int col = 0; col < seq_len; ++col) {
                    float mx = logits_ad->val.data[0 * seq_len + col];
                    for (int row = 1; row < V; ++row) {
                        mx = std::max(mx, logits_ad->val.data[row * seq_len + col]);
                    }
                    max_per_col.data[col] = mx;
                }
                Tensor ones_col_v(V, 1);
                ones_col_v.data.assign(V, 1.0f);
                auto ones_col_ad = make_ad(ones_col_v);
                auto max_ad = make_ad(max_per_col);
                auto max_broadcast = matmul(ones_col_ad, max_ad);
                auto shifted_logits = sub(logits_ad, max_broadcast);
                Tensor ones_row_t(1, V);
                ones_row_t.data.assign(V, 1.0f);
                auto ones_row = make_ad(ones_row_t);
                auto exp_shifted = exp_ad(shifted_logits);
                auto denom_row = matmul(ones_row, exp_shifted);
                auto log_denoms = log_ad(denom_row);
                auto log_sum_exp = add(log_denoms, max_ad);
                auto sum2_ad = sum(log_sum_exp);
                auto loss_ad = sub(sum2_ad, sum1_ad);
                if (use_moe && moe_aux_loss) {
                    auto weighted_aux = scalar_mul(moe_aux_loss, moe_aux_weight);
                    loss_ad = add(loss_ad, weighted_aux);
                }
                loss_ad->backward();
                float loss = loss_ad->val.data[0];
                if (std::isnan(loss) || std::isinf(loss)) {
                    std::cerr << "Error: NaN/Inf detected in loss at batch " << batch_start
                              << ", halting training\n";
                    if (!save_file.empty()) save_checkpoint(save_file);
                    return 1;
                }
                total_loss += loss;
                ++count;
            }
            ++accum_count;

            // Step optimizer after accumulating enough gradients
            if (accum_count >= grad_accum_steps) {
                // Update LR if using schedule
                if (lr_schedule == "cosine") {
                    float current_lr = lr_sched.get_lr();
                    optimizer.lr = current_lr;
                    lr_sched.step();
                }
                optimizer.step();
                ++global_step;
                accum_count = 0;
            }
        }
        // Handle leftover accumulated gradients at epoch end
        if (accum_count > 0) {
            if (lr_schedule == "cosine") {
                float current_lr = lr_sched.get_lr();
                optimizer.lr = current_lr;
                lr_sched.step();
            }
            optimizer.step();
            ++global_step;
            accum_count = 0;
        }

        float avg_loss = total_loss / count;
        if (lr_schedule == "cosine") {
            std::cout << "Epoch " << epoch << ": Avg XEnt loss = " << avg_loss
                      << " (lr=" << optimizer.lr << ")\n";
        } else {
            std::cout << "Epoch " << epoch << ": Avg XEnt loss = " << avg_loss << "\n";
        }
        loss_history.push_back(avg_loss);
        if (!save_file.empty()) {
            if (!save_checkpoint(save_file))
                std::cerr << "Error: failed saving checkpoint to " << save_file << "\n";
            else
                std::cout << "Saved checkpoint to " << save_file << "\n";
        }
        if (!valid_file.empty()) {
            float val_loss = 0.0f;
            int val_count = 0;
            for (int vs : val_starts) {
                std::vector<int> inp(val_tokens.begin() + vs,
                                     val_tokens.begin() + vs + seq_len);
                std::vector<int> tgt(val_tokens.begin() + vs + 1,
                                     val_tokens.begin() + vs + seq_len + 1);
                auto embed_v = ad_embed.forward(inp);
                auto pos_v   = ad_posenc.forward(seq_len);
                auto x_v     = add(embed_v, pos_v);
                auto h_v     = ad_transformer.forward(x_v);
                auto Wt_v = transpose(W_embed);
                auto logits_v = matmul(Wt_v, h_v);
                Tensor ones_row_v_t(1, seq_len);
                ones_row_v_t.data.assign(seq_len, 1.0f);
                auto ones_row_v = make_ad(ones_row_v_t);
                auto b_mat_v = matmul(b_lm, ones_row_v);
                logits_v = add(logits_v, b_mat_v);
                int V = (int)tokenizer.vocab_size();
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
        std::cout << "Train trend: " << sparkline(loss_history) << "\n";
        if (!val_history.empty()) {
            std::cout << "Valid trend: " << sparkline(val_history) << "\n";
        }
    }
    std::cout << "Training complete.\n";
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
