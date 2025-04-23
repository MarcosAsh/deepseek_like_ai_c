#!/usr/bin/env python3
"""
Simple BPE trainer (subword-nmt style).
Usage: ./train_bpe.py --input corpus.txt --merges merges.txt --vocab vocab.txt --merges_count 10000
"""
import argparse
from collections import Counter, defaultdict


def get_stats(vocab):
    """Compute frequency of adjacent symbol pairs."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs


def merge_vocab(pair, vocab):
    """Apply a single merge to the vocabulary."""
    v_out = {}
    a, b = pair
    merge_token = a + b
    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i+1] == b:
                new_word.append(merge_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        v_out[tuple(new_word)] = freq
    return v_out


def train_bpe(corpus_file, merges_file, vocab_file, num_merges):
    # Read corpus and count word frequencies
    print(f"Reading corpus from {corpus_file} ...", flush=True)
    word_counts = Counter()
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            for word in line.strip().split():
                if word:
                    word_counts[word] += 1
    # Initialize vocabulary: split words into chars + </w>
    vocab = {}
    for word, freq in word_counts.items():
        symbols = list(word)
        if symbols:
            symbols[-1] = symbols[-1] + '</w>'
        vocab[tuple(symbols)] = freq
    # Train merges
    merges = []
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        vocab = merge_vocab(best, vocab)
        if (i+1) % 1000 == 0:
            print(f"{i+1} merges...", flush=True)
    # Write merges
    with open(merges_file, 'w', encoding='utf-8') as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")
    print(f"Written {len(merges)} merges to {merges_file}", flush=True)
    # Build final token set (remove end-of-word marker)
    tokens = set()
    for word in vocab:
        for sym in word:
            if sym.endswith('</w>'):
                tokens.add(sym[:-4])
            else:
                tokens.add(sym)
    # Write vocab
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for tok in sorted(tokens):
            f.write(tok + '\n')
    print(f"Written {len(tokens)} tokens to {vocab_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train BPE merges and vocab from corpus.")
    parser.add_argument('--input', '-i', required=True, help='Input corpus file')
    parser.add_argument('--merges', '-m', required=True, help='Output merges file')
    parser.add_argument('--vocab', '-v', required=True, help='Output vocab file')
    parser.add_argument('--merges_count', '-n', type=int, default=10000, help='Number of merges to learn')
    args = parser.parse_args()
    train_bpe(args.input, args.merges, args.vocab, args.merges_count)


if __name__ == '__main__':
    main()