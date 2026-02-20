// Module documentation as HTML strings
// In a future iteration, this can be migrated to MDX files
export const MODULE_DOCS: Record<string, string> = {
  ADEmbedding: `
    <h2>Embedding Layer</h2>
    <p>The embedding layer converts discrete token IDs into continuous vector representations. Each token in the vocabulary is assigned a learnable vector of dimension <code>embed_dim</code>.</p>
    <h3>How it works</h3>
    <ol>
      <li>Maintains a weight matrix <em>W</em> of shape <code>[vocab_size x embed_dim]</code></li>
      <li>For each input token ID <em>i</em>, looks up the <em>i</em>-th row of <em>W</em></li>
      <li>Output shape: <code>[embed_dim x seq_len]</code> (column-major: each column is a token embedding)</li>
    </ol>
    <h3>Parameters</h3>
    <ul>
      <li><strong>vocab_size</strong>: Number of tokens in the vocabulary</li>
      <li><strong>embed_dim</strong>: Dimension of each embedding vector</li>
    </ul>
    <h3>Key Insight</h3>
    <p>Embeddings are the bridge between the discrete world of text and the continuous world of neural networks. They capture semantic relationships: similar words end up with similar vectors.</p>
  `,

  ADMultiHeadAttention: `
    <h2>Multi-Head Self-Attention</h2>
    <p>The attention mechanism allows each token to "attend to" every other token in the sequence, computing relevance scores that determine how much each token should influence the representation of every other token.</p>
    <h3>Architecture</h3>
    <ol>
      <li>Project input into <strong>Query (Q)</strong>, <strong>Key (K)</strong>, and <strong>Value (V)</strong> matrices</li>
      <li>Split into multiple heads (each head operates on <code>embed_dim / num_heads</code> dimensions)</li>
      <li>Compute attention scores: <code>softmax(Q * K^T / sqrt(d_k) + ALiBi_bias + causal_mask)</code></li>
      <li>Apply scores to values: <code>attention_weights * V</code></li>
      <li>Concatenate heads and project through output linear layer</li>
    </ol>
    <h3>Special Features</h3>
    <ul>
      <li><strong>ALiBi (Attention with Linear Biases)</strong>: Replaces positional encodings with linear distance-based biases in the attention scores, enabling length generalization</li>
      <li><strong>Causal Masking</strong>: Prevents tokens from attending to future positions (essential for autoregressive generation)</li>
    </ul>
    <h3>Parameters</h3>
    <ul>
      <li><strong>embed_dim</strong>: Total embedding dimension</li>
      <li><strong>num_heads</strong>: Number of attention heads (must divide embed_dim evenly)</li>
    </ul>
  `,

  ADFeedForward: `
    <h2>Feed-Forward Network</h2>
    <p>A position-wise feed-forward network applied independently to each token position. It consists of two linear transformations with a GELU activation in between.</p>
    <h3>Architecture</h3>
    <p><code>FFN(x) = W2 * GELU(W1 * x + b1) + b2</code></p>
    <ol>
      <li>Up-project: <code>[embed_dim] -> [hidden_dim]</code> (typically 4x expansion)</li>
      <li>GELU activation (smooth approximation of ReLU)</li>
      <li>Down-project: <code>[hidden_dim] -> [embed_dim]</code></li>
    </ol>
    <h3>Parameters</h3>
    <ul>
      <li><strong>embed_dim</strong>: Input/output dimension</li>
      <li><strong>hidden_dim</strong>: Intermediate dimension (usually 4x embed_dim)</li>
    </ul>
    <h3>Key Insight</h3>
    <p>The FFN provides the transformer with its non-linear processing capacity. While attention captures relationships between positions, the FFN processes each position's representation independently.</p>
  `,

  ADLayerNorm: `
    <h2>Layer Normalization</h2>
    <p>Normalizes the activations across the feature dimension for each token independently. This stabilizes training by ensuring consistent activation scales.</p>
    <h3>Formula</h3>
    <p><code>LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta</code></p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>dim</strong>: Feature dimension to normalize over</li>
      <li><strong>eps</strong>: Small constant for numerical stability (default: 1e-5)</li>
    </ul>
  `,

  ADMoE: `
    <h2>Mixture of Experts (MoE)</h2>
    <p>The MoE layer replaces the standard feed-forward network with multiple "expert" FFN networks and a learned router that selects which experts process each token.</p>
    <h3>Architecture</h3>
    <ol>
      <li><strong>Router</strong>: Linear layer that produces expert selection scores for each token</li>
      <li><strong>Top-K Selection</strong>: Only the top-k experts are activated per token (sparse activation)</li>
      <li><strong>Expert Networks</strong>: Independent feed-forward networks</li>
      <li><strong>Weighted Combination</strong>: Outputs are combined using softmax-normalized router scores</li>
    </ol>
    <h3>Parameters</h3>
    <ul>
      <li><strong>embed_dim</strong>: Input/output dimension</li>
      <li><strong>hidden_dim</strong>: Hidden dimension per expert</li>
      <li><strong>num_experts</strong>: Total number of expert networks</li>
      <li><strong>top_k</strong>: Number of experts activated per token</li>
    </ul>
    <h3>Auxiliary Loss</h3>
    <p>An optional load-balancing loss encourages uniform expert utilization, preventing "expert collapse" where only a few experts are ever selected.</p>
    <h3>Key Insight</h3>
    <p>MoE allows scaling model capacity (parameters) without proportionally increasing computation. DeepSeek-V2/V3 use this architecture extensively.</p>
  `,

  ADTransformerBlock: `
    <h2>Transformer Block</h2>
    <p>A complete transformer block combining layer normalization, multi-head attention, feed-forward processing, and residual connections.</p>
    <h3>Architecture</h3>
    <p><code>x = x + Attention(LayerNorm(x))</code></p>
    <p><code>x = x + FFN(LayerNorm(x))</code></p>
    <p>(Pre-norm architecture with residual connections)</p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>embed_dim</strong>: Model dimension</li>
      <li><strong>hidden_dim</strong>: FFN hidden dimension</li>
      <li><strong>n_heads</strong>: Number of attention heads</li>
      <li><strong>use_moe</strong>: Replace FFN with MoE layer</li>
      <li><strong>num_experts</strong>: Number of MoE experts (if use_moe=true)</li>
      <li><strong>moe_top_k</strong>: Top-k expert selection (if use_moe=true)</li>
    </ul>
  `,

  ADLinear: `
    <h2>Linear Layer</h2>
    <p>A fully-connected linear transformation: <code>y = W * x + b</code></p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>input_dim</strong>: Input feature dimension</li>
      <li><strong>output_dim</strong>: Output feature dimension</li>
    </ul>
  `,

  ADPositionalEncoding: `
    <h2>Positional Encoding</h2>
    <p>Learned positional embeddings that provide the model with information about token positions in the sequence.</p>
    <h3>How it works</h3>
    <ol>
      <li>Maintains a learnable position matrix of shape <code>[embed_dim x max_len]</code></li>
      <li>For a sequence of length <em>n</em>, returns the first <em>n</em> columns</li>
      <li>Added to token embeddings to inject position information</li>
    </ol>
    <h3>Parameters</h3>
    <ul>
      <li><strong>embed_dim</strong>: Embedding dimension (must match token embeddings)</li>
      <li><strong>max_len</strong>: Maximum supported sequence length</li>
    </ul>
  `,

  Tokenizer: `
    <h2>BPE Tokenizer</h2>
    <p>Converts raw text into a sequence of token IDs using Byte Pair Encoding (BPE).</p>
    <h3>How BPE Works</h3>
    <ol>
      <li>Start with individual characters as tokens</li>
      <li>Iteratively merge the most frequent pair of adjacent tokens</li>
      <li>Build a vocabulary of subword units</li>
    </ol>
    <h3>Parameters</h3>
    <ul>
      <li><strong>vocab_file</strong>: Path to vocabulary file</li>
      <li><strong>bpe_codes</strong>: Path to BPE merge rules (optional)</li>
    </ul>
  `,

  TextInput: `
    <h2>Text Input</h2>
    <p>A source module that provides a text string as input to the pipeline. Use this as the starting point for text processing graphs.</p>
    <h3>Configuration</h3>
    <ul>
      <li><strong>text</strong>: The input text string</li>
    </ul>
  `,

  TokenIDsInput: `
    <h2>Token IDs Input</h2>
    <p>A source module that provides a sequence of pre-tokenized token IDs. Useful when you want to skip tokenization and work directly with token IDs.</p>
    <h3>Configuration</h3>
    <ul>
      <li><strong>tokens</strong>: Array of integer token IDs</li>
    </ul>
  `,

  IntInput: `
    <h2>Integer Input</h2>
    <p>A source module that provides a single integer value. Used for parameters like sequence length.</p>
    <h3>Configuration</h3>
    <ul>
      <li><strong>value</strong>: The integer value</li>
    </ul>
  `,

  SeqLenExtractor: `
    <h2>Sequence Length Extractor</h2>
    <p>Extracts the length of a token sequence as an integer. Useful for feeding sequence length to positional encoding modules.</p>
  `,

  Add: `
    <h2>Element-wise Addition</h2>
    <p>Computes element-wise addition of two AD tensors: <code>output = a + b</code></p>
    <p>Supports automatic differentiation. Gradients flow through both inputs during backpropagation.</p>
  `,

  MatMul: `
    <h2>Matrix Multiplication</h2>
    <p>Computes matrix multiplication of two AD tensors: <code>output = a @ b</code></p>
    <p>Supports automatic differentiation with proper gradient computation for both operands.</p>
  `,

  Transpose: `
    <h2>Matrix Transpose</h2>
    <p>Transposes a 2D AD tensor: swaps rows and columns.</p>
    <p>Supports automatic differentiation. The transpose of the gradient flows backward.</p>
  `,

  CrossEntropy: `
    <h2>Cross-Entropy Loss</h2>
    <p>Computes the cross-entropy loss between model logits and target token IDs. Uses the log-sum-exp trick for numerical stability.</p>
    <h3>Formula</h3>
    <p><code>L = -sum(log(softmax(logits)[target]))</code></p>
    <h3>Inputs</h3>
    <ul>
      <li><strong>logits</strong>: Model output of shape <code>[vocab_size x seq_len]</code></li>
      <li><strong>targets</strong>: Target token IDs of length <code>seq_len</code></li>
    </ul>
  `,

  Backward: `
    <h2>Backward Pass</h2>
    <p>Triggers reverse-mode automatic differentiation (backpropagation) starting from the loss tensor.</p>
    <h3>How it works</h3>
    <ol>
      <li>Sets the gradient of the loss to 1.0</li>
      <li>Traverses the computation graph in reverse topological order</li>
      <li>Accumulates gradients for all parameters in the graph</li>
    </ol>
    <p>After execution, all AD tensors in the graph will have their <code>.grad</code> field populated.</p>
  `,
};
