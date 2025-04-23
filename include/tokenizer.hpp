#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <map>
#include <utility>

class Tokenizer {
public:
    // vocab_file: vocabulary file mapping tokens to IDs
    // bpe_codes_file: optional file with BPE merge rules (one per line: token1 token2)
    Tokenizer(const std::string& vocab_file, const std::string& bpe_codes_file = "");

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    size_t vocab_size() const { return vocab.size(); }
    // Get token ID for a given token string, or -1 if not found
    int to_id(const std::string& token) const;

private:
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> token_to_id;
    // BPE merge ranks: mapping symbol pairs to merge order
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    void load_vocab(const std::string& vocab_file);
    void load_bpe_codes(const std::string& codes_file);
    // Split a single word into subword tokens using BPE merges
    std::vector<std::string> bpe_split(const std::string& word) const;
};