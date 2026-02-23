#include "tokenizer.hpp"
#include <fstream>
#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <cstdio>

int main() {
    // simple vocab, no BPE
    const std::string vocab_path = "tokenizer_vocab_test.txt";
    {
        std::ofstream vf(vocab_path);
        assert(vf);
        vf << "hello 0\n" << "world 1\n" << "<unk> 2\n";
    }
    {
        Tokenizer t(vocab_path);
        auto tok = t.encode("hello unknown world");
        assert(tok.size() == 3);
        assert(tok[0] == 0 && tok[1] == 2 && tok[2] == 1);
        assert(t.decode(tok) == "hello <unk> world");
    }

    // BPE merges: "abc" -> ["ab", "c"]
    const std::string vocab_bpe = "tokenizer_vocab_bpe_test.txt";
    const std::string bpe_path = "tokenizer_bpe_test.txt";
    {
        std::ofstream vf(vocab_bpe);
        assert(vf);
        vf << "a 0\n" << "b 1\n" << "ab 2\n" << "c 3\n" << "<unk> 4\n";
    }
    {
        std::ofstream bf(bpe_path);
        assert(bf);
        bf << "a b\n" << "b c</w>\n";
    }
    {
        Tokenizer t(vocab_bpe, bpe_path);
        auto tok = t.encode("abc");
        assert(tok.size() == 2 && tok[0] == 2 && tok[1] == 3);
        assert(t.decode(tok) == "ab c");
    }

    // single character
    const std::string vocab_single = "tokenizer_vocab_single_test.txt";
    {
        std::ofstream vf(vocab_single);
        assert(vf);
        vf << "a 0\n" << "<unk> 1\n";
    }
    {
        Tokenizer t(vocab_single);
        auto tok = t.encode("a");
        assert(tok.size() == 1 && tok[0] == 0);
        assert(t.decode(tok) == "a");
    }

    // all unknown
    {
        Tokenizer t(vocab_single);
        auto tok = t.encode("xyz abc");
        for (auto& id : tok) assert(id == 1);
    }

    // vocab_size and to_id
    {
        Tokenizer t(vocab_path);
        assert(t.vocab_size() == 3);
        assert(t.to_id("hello") == 0);
        assert(t.to_id("world") == 1);
        assert(t.to_id("<unk>") == 2);
        assert(t.to_id("nope") == -1);
    }

    std::remove(vocab_path.c_str());
    std::remove(vocab_bpe.c_str());
    std::remove(bpe_path.c_str());
    std::remove(vocab_single.c_str());

    std::cout << "All Tokenizer tests passed." << std::endl;
    return 0;
}
