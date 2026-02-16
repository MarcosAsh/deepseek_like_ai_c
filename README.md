# DeepSeek-Like LLM in C++

A from-scratch large language model implementation in C++17 with no external ML dependencies. Includes custom automatic differentiation, BPE tokenizer, transformer architecture with ALiBi attention, Mixture of Experts, KV-cached generation, and quantization-aware training.

## Features

- **Custom Autodiff Engine** - Dynamic computation graph with reverse-mode backpropagation
- **Transformer Architecture** - Pre-norm transformer blocks with residual connections
- **ALiBi Attention** - Attention with Linear Biases for positional encoding (no learned position embeddings needed)
- **Causal Masking** - Autoregressive masking during training to prevent future token leakage
- **Mixture of Experts (MoE)** - Top-k expert routing with load-balancing auxiliary loss
- **KV Cache** - Cached key/value tensors for fast autoregressive generation
- **BPE Tokenizer** - Byte Pair Encoding with configurable merge rules
- **AdamW Optimizer** - Adam with weight decay and gradient clipping
- **Quantization** - Quantization-aware training (QAT) and post-training quantization (PTQ)
- **Temperature Sampling** - Top-k, top-p (nucleus), and greedy decoding with temperature control
- **Tied Embeddings** - Weight sharing between input embeddings and output projection
- **Unified Memory Pool** - Optional on-chip memory allocator for reduced allocation overhead

## Building

Requires a C++17 compiler (GCC 7+, Clang 5+).

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Or without CMake:

```bash
g++ -std=c++17 -O3 -march=native -I include src/*.cpp src/layers/*.cpp -o deepseek_ai
```

## Usage

### Training

```bash
./deepseek_ai --train data.txt --vocab vocab.txt \
  --embed_dim 128 --hidden_dim 256 --n_heads 4 --num_layers 6 \
  --seq_len 64 --batch_size 16 --epochs 10 --lr 1e-3 \
  --save checkpoint.bin
```

### Training with MoE

```bash
./deepseek_ai --train data.txt --vocab vocab.txt \
  --moe --num_experts 8 --moe_top_k 2 --moe_aux_weight 0.01 \
  --embed_dim 128 --hidden_dim 256 --n_heads 4 --num_layers 6 \
  --seq_len 64 --batch_size 16 --epochs 10 --save checkpoint.bin
```

### Generation

```bash
./deepseek_ai --generate prompt.txt --vocab vocab.txt \
  --resume checkpoint.bin --max_new_tokens 128 \
  --temperature 0.8 --top_k 40
```

### Interactive Mode

```bash
./deepseek_ai --cli --vocab vocab.txt --resume checkpoint.bin \
  --temperature 0.7 --top_p 0.9
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--train PATH` | Train on text data | - |
| `--generate PATH` | Generate from prompt file | - |
| `--cli` | Interactive REPL mode | - |
| `--vocab PATH` | Vocabulary file | `input_files/vocab.txt` |
| `--bpe-codes PATH` | BPE merge rules file | - |
| `--embed_dim N` | Embedding dimension | 64 |
| `--hidden_dim N` | Feed-forward hidden dimension | 64 |
| `--n_heads N` | Number of attention heads | 4 |
| `--num_layers N` | Number of transformer layers | 3 |
| `--seq_len N` | Training sequence length | 32 |
| `--batch_size N` | Mini-batch size | 16 |
| `--epochs N` | Training epochs | 5 |
| `--lr FLOAT` | Learning rate | 1e-3 |
| `--temperature FLOAT` | Sampling temperature | 1.0 |
| `--top_k N` | Top-k sampling (0=greedy) | 0 |
| `--top_p FLOAT` | Nucleus sampling (0=greedy) | 0.0 |
| `--max_new_tokens N` | Max tokens to generate | 32 |
| `--moe` | Enable Mixture of Experts | off |
| `--num_experts N` | Number of MoE experts | 4 |
| `--moe_top_k N` | Experts activated per token | 2 |
| `--qat` | Enable quantization-aware training | off |
| `--qat-bits N` | Quantization bit width | 8 |
| `--resume PATH` | Load checkpoint | - |
| `--save PATH` | Save checkpoint | `checkpoint.bin` |
| `--timer` | Enable performance timers | off |

## Architecture

```
Input Tokens
    |
Embedding (tied weights)
    |
+ Positional Encoding
    |
[Transformer Block] x N
    |-- LayerNorm -> Multi-Head Attention (ALiBi + Causal Mask) -> Residual
    |-- LayerNorm -> FeedForward / MoE -> Residual
    |
Output Projection (tied with Embedding)
    |
Logits -> Sampling (greedy / top-k / top-p / temperature)
```

## Tests

```bash
# Run all tests
./tensor_test && ./layers_test && ./autodiff_test && \
./tokenizer_test && ./loss_test && ./ad_attention_test && \
./feed_forward_test && ./quantization_test
```

## Papers Implemented

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [ALiBi: Train Short, Test Long](https://arxiv.org/abs/2108.12409) - Attention Linear Biases
- [Switch Transformers](https://arxiv.org/abs/2101.03961) - Mixture of Experts routing
- [GELU](https://arxiv.org/abs/1606.08415) - Gaussian Error Linear Units activation
- [AdamW](https://arxiv.org/abs/1711.05101) - Decoupled weight decay regularization
