#include "tokenizer.hpp"
#include <fstream>
#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <cstdio>

int main() {
    // Test 1: Simple vocab without BPE
    const std::string vocab_path = "tokenizer_vocab_test.txt";
    {
        std::ofstream vf(vocab_path);
        assert(vf);
        vf << "hello 0\n";
        vf << "world 1\n";
        vf << "<unk> 2\n";
    }
    {
        Tokenizer t(vocab_path);
        auto tok = t.encode("hello unknown world");
        assert(tok.size() == 3);
        assert(tok[0] == 0);
        assert(tok[1] == 2);
        assert(tok[2] == 1);
        std::string dec = t.decode(tok);
        assert(dec == "hello <unk> world");
    }
    
    // Test 2: Vocab with simple BPE merges
    const std::string vocab_bpe = "tokenizer_vocab_bpe_test.txt";
    const std::string bpe_path = "tokenizer_bpe_test.txt";
    {
        std::ofstream vf(vocab_bpe);
        assert(vf);
        vf << "a 0\n";
        vf << "b 1\n";
        vf << "ab 2\n";
        vf << "c 3\n";
        vf << "<unk> 4\n";
    }
    {
        std::ofstream bf(bpe_path);
        assert(bf);
        bf << "a b\n";
        bf << "b c</w>\n";  // merge 'b' and 'c</w>'
    }
    {
        Tokenizer t2(vocab_bpe, bpe_path);
        auto tok = t2.encode("abc");
        // 'abc' -> ['a','b','c</w>'] -> merge 'a'+'b' -> ['ab','c</w>'] -> ['ab','c']
        assert(tok.size() == 2);
        assert(tok[0] == 2);
        assert(tok[1] == 3);
        std::string dec = t2.decode(tok);
        assert(dec == "ab c");
    }
    
    // Cleanup temporary files
    std::remove(vocab_path.c_str());
    std::remove(vocab_bpe.c_str());
    std::remove(bpe_path.c_str());

    std::cout << "All Tokenizer tests passed." << std::endl;
    return 0;
}