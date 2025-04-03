#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

class Tokenizer {
public:
    Tokenizer(const std::string& vocab_file);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    size_t vocab_size() const { return vocab.size(); }

private:
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> token_to_id;

    void load_vocab(const std::string& vocab_file);
    std::vector<std::string> bpe_split(const std::string& word) const;
};