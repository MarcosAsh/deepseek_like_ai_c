// Simple BPE trainer (subword-nmt style) translated from Python
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>

// Hash for vector<string>
struct VectorHash {
    size_t operator()(std::vector<std::string> const& v) const noexcept {
        size_t h = 0;
        for (auto const& s : v) {
            h ^= std::hash<std::string>()(s) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
        }
        return h;
    }
};
// Hash for pair<string,string>
struct PairHash {
    size_t operator()(std::pair<std::string,std::string> const& p) const noexcept {
        return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
    }
};

using VocabMap = std::unordered_map<std::vector<std::string>, long long, VectorHash>;
using PairCount = std::unordered_map<std::pair<std::string,std::string>, long long, PairHash>;

// Compute frequency of adjacent symbol pairs
PairCount get_stats(VocabMap const& vocab) {
    PairCount pairs;
    for (auto const& item : vocab) {
        auto const& symbols = item.first;
        long long freq = item.second;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto key = std::make_pair(symbols[i], symbols[i+1]);
            pairs[key] += freq;
        }
    }
    return pairs;
}

// Apply a single merge to the vocabulary
VocabMap merge_vocab(std::pair<std::string,std::string> const& merge_pair, VocabMap const& vocab) {
    VocabMap out;
    auto const& a = merge_pair.first;
    auto const& b = merge_pair.second;
    std::string merged = a + b;
    for (auto const& item : vocab) {
        auto const& symbols = item.first;
        long long freq = item.second;
        std::vector<std::string> new_symbols;
        for (size_t i = 0; i < symbols.size(); ) {
            if (i + 1 < symbols.size() && symbols[i] == a && symbols[i+1] == b) {
                new_symbols.push_back(merged);
                i += 2;
            } else {
                new_symbols.push_back(symbols[i]);
                i += 1;
            }
        }
        out.emplace(std::move(new_symbols), freq);
    }
    return out;
}

// Main training function
void train_bpe(std::string const& corpus_file,
               std::string const& merges_file,
               std::string const& vocab_file,
               int num_merges) {
    std::cout << "Reading corpus from " << corpus_file << " ..." << std::endl;
    // Count word frequencies
    std::unordered_map<std::string, long long> word_counts;
    {
        std::ifstream in(corpus_file);
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream iss(line);
            std::string word;
            while (iss >> word) {
                if (!word.empty()) word_counts[word]++;
            }
        }
    }
    // Initialize vocab: split words into chars with </w> on last symbol
    VocabMap vocab;
    for (auto const& wc : word_counts) {
        auto const& word = wc.first;
        long long freq = wc.second;
        std::vector<std::string> symbols;
        // split on bytes (assumes UTF-8 / ASCII)
        for (char c : word) symbols.emplace_back(1, c);
        if (!symbols.empty()) {
            symbols.back() += "</w>";
        }
        vocab.emplace(std::move(symbols), freq);
    }
    // Learn merges
    std::vector<std::pair<std::string,std::string>> merges;
    merges.reserve(num_merges);
    for (int i = 0; i < num_merges; ++i) {
        auto pairs = get_stats(vocab);
        if (pairs.empty()) break;
        // find best
        auto best = std::max_element(pairs.begin(), pairs.end(),
            [](auto const& a, auto const& b){ return a.second < b.second; });
        merges.push_back(best->first);
        vocab = merge_vocab(best->first, vocab);
        if ((i+1) % 1000 == 0)
            std::cout << (i+1) << " merges..." << std::endl;
    }
    // Write merges
    {
        std::ofstream out(merges_file);
        for (auto const& m : merges) {
            out << m.first << ' ' << m.second << '\n';
        }
    }
    std::cout << "Written " << merges.size() << " merges to " << merges_file << std::endl;
    // Build final vocab tokens
    std::set<std::string> tokens;
    for (auto const& item : vocab) {
        for (auto const& sym : item.first) {
            if (sym.size() > 4 && sym.substr(sym.size()-4) == "</w>") {
                tokens.insert(sym.substr(0, sym.size()-4));
            } else {
                tokens.insert(sym);
            }
        }
    }
    // Write vocab
    {
        std::ofstream out(vocab_file);
        for (auto const& tok : tokens) {
            out << tok << '\n';
        }
    }
    std::cout << "Written " << tokens.size() << " tokens to " << vocab_file << std::endl;
}

// Simple argument parser
int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " --input CORPUS --merges MERGES --vocab VOCAB [--merges_count N]" << std::endl;
        return 1;
    }
    std::string input_file, merges_file, vocab_file;
    int merges_count = 10000;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--input" || arg == "-i") && i+1 < argc) {
            input_file = argv[++i];
        } else if ((arg == "--merges" || arg == "-m") && i+1 < argc) {
            merges_file = argv[++i];
        } else if ((arg == "--vocab" || arg == "-v") && i+1 < argc) {
            vocab_file = argv[++i];
        } else if ((arg == "--merges_count" || arg == "-n") && i+1 < argc) {
            merges_count = std::stoi(argv[++i]);
        }
    }
    if (input_file.empty() || merges_file.empty() || vocab_file.empty()) {
        std::cerr << "Missing required arguments" << std::endl;
        return 1;
    }
    train_bpe(input_file, merges_file, vocab_file, merges_count);
    return 0;
}