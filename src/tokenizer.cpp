#include "../include/tokenizer.hpp"
#include <algorithm>
#include <sstream>
#include <stdexcept>

Tokenizer::Tokenizer(const std::string& vocab_file) {
    load_vocab(vocab_file);
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
        if (id >= static_cast<int>(vocab.size())) {
            vocab.resize(id + 1);
        }
        vocab[id] = token;
        token_to_id[token] = id;
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        auto bpe_tokens = bpe_split(word);
        for (const auto& token : bpe_tokens) {
            auto it = token_to_id.find(token);
            if (it != token_to_id.end()) {
                tokens.push_back(it->second);
            } else {
                throw std::runtime_error("Token not found in vocabulary: " + token);
            }
        }
    }
    return tokens;
}   

std::vector<std::string> Tokenizer::bpe_split(const std::string& word) const {
    // Simplified BPE - in reality you'd need the merge rules
    std::vector<std::string> tokens;
    size_t start = 0;
    
    while (start < word.size()) {
        size_t end = word.size();
        std::string subword;
        
        // Find longest matching subword
        while (end > start) {
            subword = word.substr(start, end - start);
            if (token_to_id.count(subword)) {
                tokens.push_back(subword);
                start = end;
                break;
            }
            end--;
        }
        
        if (end == start) { // No match found
            tokens.push_back("<unk>");
            start++;
        }
    }
    
    return tokens;
}