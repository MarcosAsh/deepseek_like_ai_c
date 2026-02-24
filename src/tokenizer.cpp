#include "tokenizer.hpp"
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <climits>
#include <iostream>

Tokenizer::Tokenizer(const std::string& vocab_file, const std::string& bpe_codes_file) {
    load_vocab(vocab_file);
    if (!bpe_codes_file.empty()) {
        load_bpe_codes(bpe_codes_file);
    }
}

void Tokenizer::load_vocab(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open vocabulary file: " + vocab_file);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string token;
        // Read token
        if (!(iss >> token)) continue;
        int id;
        if (iss >> id) {
            // explicit ID provided
        } else {
            // assign next available ID
            id = static_cast<int>(vocab.size());
        }
        // Ensure vocab vector can hold at index id
        static constexpr int MAX_VOCAB_SIZE = 1000000;
        if (id >= MAX_VOCAB_SIZE) {
            throw std::runtime_error("Vocabulary ID " + std::to_string(id) +
                                     " exceeds maximum allowed size (" +
                                     std::to_string(MAX_VOCAB_SIZE) + ")");
        }
        if (id >= static_cast<int>(vocab.size())) {
            vocab.resize(id + 1);
        }
        vocab[id] = token;
        token_to_id[token] = id;
    }
}

void Tokenizer::load_bpe_codes(const std::string& codes_file) {
    std::ifstream in(codes_file);
    if (!in) {
        throw std::runtime_error("Could not open BPE codes file: " + codes_file);
    }
    std::string line;
    int rank = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string a, b;
        if (!(iss >> a >> b)) continue;
        bpe_ranks[{a, b}] = rank++;
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    std::istringstream iss(text);
    std::string word;
    const int unk_id = to_id("<unk>");
    while (iss >> word) {
        auto pieces = bpe_split(word);
        for (const auto& piece : pieces) {
            int id = to_id(piece);
            if (id >= 0) {
                tokens.push_back(id);
            } else if (unk_id >= 0) {
                tokens.push_back(unk_id);
            } else {
                std::cerr << "Warning: dropping unknown token '" << piece
                          << "' (no <unk> in vocabulary)\n";
            }
        }
    }
    return tokens;
}   

std::vector<std::string> Tokenizer::bpe_split(const std::string& word) const {
    if (bpe_ranks.empty()) {
        // fallback: treat whole word as single token
        return {word};
    }
    // Initialize symbols as characters + end-of-word marker
    std::vector<std::string> symbols;
    for (char c : word) {
        symbols.emplace_back(1, c);
    }
    if (!symbols.empty()) {
        symbols.back() += "</w>";
    }
    // BPE merge loop
    while (true) {
        int best_rank = INT_MAX;
        int best_i = -1;
        for (int i = 0; i + 1 < (int)symbols.size(); ++i) {
            auto pr = std::make_pair(symbols[i], symbols[i+1]);
            auto it = bpe_ranks.find(pr);
            if (it != bpe_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        // merge symbols[best_i] and symbols[best_i+1]
        symbols[best_i] = symbols[best_i] + symbols[best_i+1];
        symbols.erase(symbols.begin() + best_i + 1);
    }
    // remove end-of-word marker
    if (!symbols.empty()) {
        std::string &last = symbols.back();
        const std::string marker = "</w>";
        if (last.size() >= marker.size() &&
            last.compare(last.size() - marker.size(), marker.size(), marker) == 0) {
            last.erase(last.size() - marker.size());
        }
    }
    return symbols;
}
// Decode a sequence of token IDs back to a string (space-separated tokens)
std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::ostringstream oss;
    bool first = true;
    for (int id : tokens) {
        if (!first) oss << ' ';
        first = false;
        if (id >= 0 && id < static_cast<int>(vocab.size())) {
            oss << vocab[id];
        } else {
            oss << "<unk>";
        }
    }
    return oss.str();
}
// Return the ID of a token string, or -1 if not present
int Tokenizer::to_id(const std::string& token) const {
    auto it = token_to_id.find(token);
    if (it != token_to_id.end()) return it->second;
    return -1;
}