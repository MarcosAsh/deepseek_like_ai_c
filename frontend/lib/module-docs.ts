// Module documentation as HTML strings
// In a future iteration, this can be migrated to MDX files
export const MODULE_DOCS: Record<string, string> = {
  ADEmbedding: `
    <h2>Embedding Layer</h2>
    <p>Looks up a dense vector for each token ID. The weight matrix <em>W</em> has shape <code>[vocab_size x embed_dim]</code> and is initialized with Xavier uniform. For an input sequence of token IDs, each ID selects the corresponding row from <em>W</em>, and the output is assembled into a matrix of shape <code>[embed_dim x seq_len]</code> (column-major, each column = one token's vector).</p>
    <p>During backprop, only the rows that were looked up receive gradient updates. This makes embedding layers sparse in practice -most of the weight matrix doesn't get touched on any given step.</p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>vocab_size</strong>  - number of distinct tokens the layer can handle</li>
      <li><strong>embed_dim</strong>  - length of each token's vector</li>
    </ul>
    <h3>Outputs</h3>
    <ul>
      <li><strong>output</strong>  - the embedded sequence <code>[embed_dim x seq_len]</code></li>
      <li><strong>weights</strong> (optional)  - the raw weight matrix, useful for weight tying with the output projection</li>
    </ul>
    <p>If you're building a full LM pipeline, you'll typically feed this into positional encoding (either learned or RoPE) and then into a transformer block.</p>
  `,

  ADMultiHeadAttention: `
    <h2>Multi-Head Self-Attention</h2>
    <p>Each token computes a weighted sum over all other tokens (or only previous ones, with causal masking). The weighting is learned through query/key dot products.</p>
    <h3>Forward pass, step by step</h3>
    <ol>
      <li>Multiply the input <code>[embed_dim x seq_len]</code> by three weight matrices <code>W_q, W_k, W_v</code> (all <code>[embed_dim x embed_dim]</code>) to get Q, K, V</li>
      <li>Split Q, K, V into <code>num_heads</code> slices of <code>head_dim = embed_dim / num_heads</code> rows each</li>
      <li>For each head: compute <code>scores = Q_h^T @ K_h / sqrt(head_dim)</code>, add an ALiBi bias (linear penalty proportional to distance between positions), mask out future positions if causal</li>
      <li>Softmax each row of scores, then compute <code>V_h @ attn^T</code> to get that head's output</li>
      <li>Concatenate all heads back into <code>[embed_dim x seq_len]</code>, multiply by <code>W_o</code></li>
    </ol>
    <h3>ALiBi</h3>
    <p>Instead of adding positional encodings to the input, ALiBi adds a bias of <code>-slope * |i - j|</code> directly to the attention scores. Each head gets a different slope (powers of 2, spaced geometrically). This means the model can generalize to longer sequences than it was trained on without any changes.</p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>embed_dim</strong>  - must be divisible by num_heads</li>
      <li><strong>num_heads</strong>  - more heads = finer-grained attention patterns, but each head works in a smaller subspace</li>
    </ul>
  `,

  ADFeedForward: `
    <h2>Feed-Forward Network (GELU)</h2>
    <p>Two linear layers with a GELU activation between them: <code>FFN(x) = W2 * GELU(W1 * x + b1) + b2</code>. Applied independently to each token position -there's no interaction between positions here, that's attention's job.</p>
    <p>The hidden dimension is usually 4x the embed dimension. This expansion/compression pattern gives the network room to do nonlinear processing in a higher-dimensional space before projecting back down.</p>
    <p>GELU (Gaussian Error Linear Unit) is a smooth activation that looks like ReLU but doesn't have a hard zero cutoff -it allows small negative gradients through, which tends to help with training stability.</p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>embed_dim</strong>  - input and output size</li>
      <li><strong>hidden_dim</strong>  - intermediate size (try 4x embed_dim as a starting point)</li>
    </ul>
    <p>If you want the gated variant (SwiGLU), use the ADSwiGLU module instead. It uses three weight matrices instead of two and tends to train better, but costs ~50% more parameters at the same hidden_dim.</p>
  `,

  ADLayerNorm: `
    <h2>Layer Normalization</h2>
    <p>Normalizes each token's feature vector to have zero mean and unit variance, then applies learned scale (<code>gamma</code>) and shift (<code>beta</code>):</p>
    <p><code>out = gamma * (x - mean) / sqrt(var + eps) + beta</code></p>
    <p>The normalization is computed per-column (per token position), across the feature dimension. This keeps activation magnitudes stable as they pass through many layers -without it, values tend to either explode or vanish after a few transformer blocks.</p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>dim</strong>  - feature dimension (should match embed_dim)</li>
      <li><strong>eps</strong>  - added inside the sqrt to prevent division by zero. Default 1e-5. You probably don't need to change this.</li>
    </ul>
    <p>For the variant without mean centering (just divides by RMS), see ADRMSNorm. In this codebase, the transformer block's <code>use_rmsnorm</code> flag controls which one gets used.</p>
  `,

  ADMoE: `
    <h2>Mixture of Experts</h2>
    <p>Replaces the single FFN in a transformer block with <code>num_experts</code> independent FFN networks and a learned router. For each token, only the top-k scoring experts actually run -the rest are skipped entirely.</p>
    <h3>How routing works</h3>
    <p>A linear layer (the "gate") maps each token's representation to a score vector of size <code>num_experts</code>. The top-k scores are selected, softmax-normalized to sum to 1, and used as weights to combine the corresponding expert outputs. Tokens that don't activate an expert don't pay for its computation.</p>
    <h3>Load balancing</h3>
    <p>Without regularization, the router tends to collapse -it learns to send everything to one or two experts and ignores the rest. The auxiliary loss penalizes uneven expert utilization by comparing the fraction of tokens routed to each expert against a uniform target. This loss gets added to the main training loss (usually with a small coefficient like 0.01).</p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>embed_dim</strong>  - input/output dimension</li>
      <li><strong>hidden_dim</strong>  - hidden size inside each expert FFN</li>
      <li><strong>num_experts</strong>  - total expert count (8 and 16 are common choices)</li>
      <li><strong>top_k</strong>  - how many experts each token activates (usually 2)</li>
    </ul>
    <p>The total parameter count scales with num_experts, but the FLOPs per token only scale with top_k. So you can have a huge model that's cheap to run per-token.</p>
  `,

  ADTransformerBlock: `
    <h2>Transformer Block</h2>
    <p>One layer of the transformer stack. Uses pre-norm residual connections:</p>
    <pre><code>x = x + Attention(Norm(x))
x = x + FFN(Norm(x))</code></pre>
    <p>The normalization (LayerNorm or RMSNorm) goes <em>before</em> the sublayer, not after. This is the "pre-norm" architecture -it's more stable to train than post-norm, especially at depth.</p>
    <p>The FFN slot can be filled by one of three variants depending on the config: standard GELU FFN, SwiGLU, or MoE. The block handles this internally -from the outside, it's just input tensor in, output tensor out.</p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>embed_dim</strong>, <strong>hidden_dim</strong>, <strong>n_heads</strong>  - the core dimensions</li>
      <li><strong>use_moe</strong>  - swap the FFN for a MoE layer</li>
      <li><strong>num_experts</strong>, <strong>moe_top_k</strong>  - MoE settings (ignored if use_moe is false)</li>
    </ul>
    <p>Stack multiple of these in an ADTransformer to build a full model. Depth (number of layers) is where most of a transformer's representational power comes from.</p>
  `,

  ADLinear: `
    <h2>Linear Layer</h2>
    <p>Standard affine transformation: <code>y = W * x + b</code>. Weights initialized with Xavier uniform, bias initialized to zero. The bias is broadcast across the sequence dimension.</p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>input_dim</strong>  - number of input features (rows of x)</li>
      <li><strong>output_dim</strong>  - number of output features (rows of y)</li>
    </ul>
  `,

  ADPositionalEncoding: `
    <h2>Learned Positional Encoding</h2>
    <p>A trainable matrix of shape <code>[embed_dim x max_len]</code> where each column is a learned position vector. For a sequence of length <em>n</em>, slice out the first <em>n</em> columns and add them to the token embeddings.</p>
    <p>This is the simplest form of position encoding -it works, but it can't generalize to sequence lengths longer than <code>max_len</code>. For length-agnostic alternatives, see RoPE (which encodes position through rotation) or ALiBi (which adds position bias directly to attention scores).</p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>embed_dim</strong>  - must match the token embedding dimension</li>
      <li><strong>max_len</strong>  - the longest sequence you'll ever feed in (default 512)</li>
    </ul>
  `,

  Tokenizer: `
    <h2>BPE Tokenizer</h2>
    <p>Splits raw text into subword tokens using Byte Pair Encoding. BPE starts with individual characters, then iteratively merges the most frequent adjacent pair into a single token. After training, common words get their own token while rare words are split into pieces.</p>
    <p>The merge rules are loaded from a file (<code>bpe_codes</code>). If no BPE codes are provided, the tokenizer falls back to character-level tokenization.</p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>vocab_file</strong>  - maps characters/tokens to integer IDs</li>
      <li><strong>bpe_codes</strong>  - merge rules file (optional, leave empty for character-level)</li>
    </ul>
  `,

  TextInput: `
    <h2>Text Input</h2>
    <p>Source node that provides a text string. This is your starting point for any graph that processes raw text. Connect it to a Tokenizer to convert the text into token IDs.</p>
    <h3>Config</h3>
    <ul>
      <li><strong>text</strong>  - the input string</li>
    </ul>
  `,

  TokenIDsInput: `
    <h2>Token IDs Input</h2>
    <p>Source node that provides a pre-tokenized sequence of integer IDs. Use this when you want to skip tokenization and work with token IDs directly.</p>
    <h3>Config</h3>
    <ul>
      <li><strong>tokens</strong>  - array of integer token IDs, e.g. <code>[1, 2, 3, 4]</code></li>
    </ul>
  `,

  IntInput: `
    <h2>Integer Input</h2>
    <p>Source node for a single integer value. Used to provide things like sequence length to positional encoding modules.</p>
    <h3>Config</h3>
    <ul>
      <li><strong>value</strong>  - the integer</li>
    </ul>
  `,

  SeqLenExtractor: `
    <h2>Sequence Length Extractor</h2>
    <p>Takes a token ID list and outputs its length as an integer. Mostly a convenience node -connect it between a tokenizer output and a positional encoding's <code>seq_len</code> input.</p>
  `,

  Add: `
    <h2>Element-wise Addition</h2>
    <p><code>output = a + b</code>. Both inputs must have the same shape. Gradient flows through to both inputs (d_out/d_a = 1, d_out/d_b = 1). This is how residual connections work in transformer blocks.</p>
  `,

  MatMul: `
    <h2>Matrix Multiplication</h2>
    <p><code>output = a @ b</code> where <code>a</code> is <code>[M x K]</code> and <code>b</code> is <code>[K x N]</code>, producing <code>[M x N]</code>. Backward: <code>grad_a = grad_out @ b^T</code>, <code>grad_b = a^T @ grad_out</code>.</p>
  `,

  Transpose: `
    <h2>Matrix Transpose</h2>
    <p>Swaps rows and columns. <code>[M x N]</code> becomes <code>[N x M]</code>. Backward just transposes the gradient.</p>
  `,

  CrossEntropy: `
    <h2>Cross-Entropy Loss</h2>
    <p>Measures how well the model's predicted logits match the target tokens. Internally: softmax the logits column-wise, take the log probability of the correct token at each position, negate and average.</p>
    <p>Uses log-sum-exp for numerical stability -without it, the exp() in softmax overflows on large logit values. The implementation subtracts the per-column max before exponentiating.</p>
    <h3>Inputs</h3>
    <ul>
      <li><strong>logits</strong>  - raw model output, shape <code>[vocab_size x seq_len]</code></li>
      <li><strong>targets</strong>  - ground truth token IDs, one per position</li>
    </ul>
    <h3>Output</h3>
    <p>A scalar loss value. Connect this to a Backward node to trigger gradient computation.</p>
  `,

  Backward: `
    <h2>Backward Pass</h2>
    <p>Kicks off reverse-mode autodiff from the loss tensor. Sets the loss gradient to 1.0, then walks the computation graph backward (reverse topological order), calling each node's gradient function to accumulate <code>.grad</code> on every tensor in the graph.</p>
    <p>After this runs, you can inspect any parameter's <code>.grad</code> field. In a training loop, you'd pass those gradients to an optimizer (SGD or AdamW) to update the weights.</p>
  `,

  ADRMSNorm: `
    <h2>RMS Normalization</h2>
    <p>A simpler alternative to LayerNorm. Instead of subtracting the mean and dividing by standard deviation, RMSNorm just divides by the root mean square:</p>
    <p><code>out = gamma * x / sqrt(mean(x^2) + eps)</code></p>
    <p>There's no mean centering and no beta (shift) parameter -just a learnable scale <code>gamma</code> (initialized to 1). This makes it cheaper to compute and has one fewer parameter per dimension. In practice, the missing mean centering doesn't seem to hurt quality.</p>
    <p>The backward pass needs the chain rule through the sqrt and the element-wise division, but since there's no variance computation (just RMS), the gradient is simpler than LayerNorm's.</p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>dim</strong>  - feature dimension</li>
      <li><strong>eps</strong>  - stability constant, default 1e-6 (note: smaller default than LayerNorm's 1e-5)</li>
    </ul>
    <h3>When to use</h3>
    <p>If you're building something LLaMA-style. Toggle it with <code>use_rmsnorm: true</code> in the TransformerConfig. The main reason to prefer it over LayerNorm is speed and simplicity.</p>
  `,

  ADSwiGLU: `
    <h2>SwiGLU Feed-Forward</h2>
    <p>A gated FFN variant that uses three weight matrices instead of two:</p>
    <p><code>SwiGLU(x) = W_down @ (swish(W_gate @ x) * W_up @ x)</code></p>
    <p>The gate path runs <code>x</code> through <code>W_gate</code> and applies the Swish activation (<code>x * sigmoid(x)</code>). This gets element-wise multiplied with a separate linear projection through <code>W_up</code>. The product then goes through <code>W_down</code> to project back to the original dimension.</p>
    <p>The gating means the network can learn to selectively zero out dimensions -if the gate output is near zero for some feature, that feature gets suppressed regardless of what the up-projection computes. This added control tends to produce better loss curves than a plain GELU FFN.</p>
    <h3>Parameter cost</h3>
    <p>Three matrices (<code>W_gate</code>, <code>W_up</code>, <code>W_down</code>) vs two (<code>W1</code>, <code>W2</code>) in the standard FFN. At the same <code>hidden_dim</code>, SwiGLU has ~50% more parameters. Some architectures compensate by using <code>hidden_dim = 2/3 * 4 * embed_dim</code> instead of <code>4 * embed_dim</code>.</p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>embed_dim</strong>  - input and output dimension</li>
      <li><strong>hidden_dim</strong>  - intermediate dimension for all three projections</li>
    </ul>
  `,

  RoPE: `
    <h2>Rotary Position Embedding (RoPE)</h2>
    <p>Encodes position by physically rotating the embedding vector. Pairs up adjacent dimensions (0&1, 2&3, etc.) and rotates each pair by an angle that depends on the token's position and the pair's index.</p>
    <h3>The rotation</h3>
    <p>For dimension pair <em>i</em> at position <em>t</em>, the rotation angle is <code>theta_i = t / (base ^ (2i / head_dim))</code> where <code>base</code> is typically 10000. Low-index pairs rotate slowly (so they encode coarse/long-range position), while high-index pairs rotate fast (fine/local position). The cos/sin tables are precomputed at construction time.</p>
    <h3>Why rotation works</h3>
    <p>When you take the dot product of two rotated vectors (as happens in attention's Q @ K^T), the rotations partially cancel out and what's left depends on the <em>difference</em> in positions, not the absolute positions. So the model naturally gets relative position information without explicitly encoding it.</p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>head_dim</strong>  - dimension of each attention head (must be even, since we rotate pairs)</li>
      <li><strong>max_len</strong>  - precompute cos/sin tables up to this length (default 4096, but can extrapolate beyond)</li>
    </ul>
    <p>This implementation supports both raw tensor application (<code>apply()</code>) and autodiff-tracked application (<code>apply_ad()</code>) with a backward pass that applies the inverse rotation (negated sin).</p>
  `,

  ADGQA: `
    <h2>Grouped Query Attention (GQA)</h2>
    <p>Standard multi-head attention uses separate K and V projections for each head. GQA shares K/V heads across groups of Q heads. If you have 8 query heads and 2 KV heads, each KV head serves a group of 4 query heads.</p>
    <h3>Concrete shapes</h3>
    <p>With <code>embed_dim=64, num_heads=8, num_kv_heads=2, head_dim=8</code>:</p>
    <ul>
      <li>W_q: <code>[64 x 64]</code>  - full size, produces all 8 Q heads</li>
      <li>W_k: <code>[16 x 64]</code>  - smaller, produces only 2 KV heads</li>
      <li>W_v: <code>[16 x 64]</code>  - same as K</li>
      <li>W_o: <code>[64 x 64]</code>  - full size output projection</li>
    </ul>
    <p>During the attention loop, Q heads 0-3 all use KV head 0, and Q heads 4-7 all use KV head 1. Each Q head still computes its own attention pattern, but they're looking at the same key/value representations.</p>
    <h3>Why bother</h3>
    <p>The KV cache during inference is proportional to the number of KV heads. Going from 8 KV heads to 2 cuts KV cache memory by 4x. The quality tradeoff is small -GQA with 2 KV heads is much closer in quality to full MHA than you'd expect.</p>
    <h3>Special cases</h3>
    <ul>
      <li><code>num_kv_heads = 1</code>  - Multi-Query Attention (MQA), all heads share one KV</li>
      <li><code>num_kv_heads = num_heads</code>  - standard Multi-Head Attention</li>
    </ul>
    <h3>Parameters</h3>
    <ul>
      <li><strong>embed_dim</strong>  - total model dimension</li>
      <li><strong>num_heads</strong>  - number of query heads</li>
      <li><strong>num_kv_heads</strong>  - number of key/value heads (must divide num_heads evenly)</li>
    </ul>
  `,

  ADLoRA: `
    <h2>LoRA (Low-Rank Adaptation)</h2>
    <p>Instead of fine-tuning all the weights in a linear layer, LoRA freezes the original weight matrix <code>W</code> and trains two small matrices <code>A</code> and <code>B</code> that represent the update as a low-rank factorization:</p>
    <p><code>y = (W + (alpha/rank) * B @ A) @ x + bias</code></p>
    <h3>How the matrices are set up</h3>
    <ul>
      <li><code>W</code>  - <code>[out x in]</code>, frozen (not in the optimizer's parameter list)</li>
      <li><code>A</code>  - <code>[rank x in]</code>, trainable, initialized with Xavier uniform</li>
      <li><code>B</code>  - <code>[out x rank]</code>, trainable, initialized to <strong>zero</strong></li>
    </ul>
    <p>Because B starts at zero, the initial output is exactly <code>W @ x + bias</code> -the LoRA contribution is zero at the start. As training progresses, B and A learn a low-rank correction to the frozen weights.</p>
    <h3>Parameter savings</h3>
    <p>A full <code>[64 x 64]</code> weight matrix has 4096 parameters. With rank=8, LoRA adds <code>8*64 + 64*8 = 1024</code> trainable params (the A and B matrices). That's 4x fewer parameters, and with rank=4 it's 8x fewer. In practice, rank 4-16 works well for most tasks.</p>
    <h3>The alpha/rank scaling</h3>
    <p>The LoRA output is scaled by <code>alpha/rank</code>. When <code>alpha = rank</code> (the default), the scale is 1. Setting alpha higher makes the LoRA contribution larger relative to the base weights. This is a hyperparameter you can tune -higher alpha means the adaptation has more influence.</p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>input_dim</strong>, <strong>output_dim</strong>  - dimensions of the linear layer being adapted</li>
      <li><strong>rank</strong>  - rank of the low-rank matrices (default 8)</li>
      <li><strong>alpha</strong>  - scaling factor (default 8.0, same as rank)</li>
    </ul>
  `,

  ADFlashAttention: `
    <h2>Flash Attention</h2>
    <p>Computes the same result as standard multi-head attention, but processes the attention matrix in tiles instead of materializing the entire <code>[seq_len x seq_len]</code> matrix at once.</p>
    <h3>Why tiling matters</h3>
    <p>Standard attention computes <code>softmax(Q^T @ K) @ V^T</code> for each head. The intermediate attention matrix is <code>[seq_len x seq_len]</code>, which gets large fast (1024 tokens = 4MB per head in float32). Flash attention avoids allocating this by processing blocks of queries against blocks of keys, maintaining running softmax statistics (max and sum) as it goes.</p>
    <h3>The online softmax trick</h3>
    <p>You can't just softmax each tile independently -the denominators would be wrong. Instead, for each query position, the implementation tracks a running max and a running sum-of-exponentials across all key tiles. When a new tile has a larger max, it corrects the accumulated values with <code>exp(old_max - new_max)</code>. This produces numerically identical results to computing the full softmax.</p>
    <h3>When it falls back to standard attention</h3>
    <p>If the sequence length is less than or equal to <code>tile_size</code>, there's nothing to tile -the implementation just does regular attention. The tiling only kicks in for longer sequences.</p>
    <h3>Parameters</h3>
    <ul>
      <li><strong>embed_dim</strong>, <strong>num_heads</strong>  - same as standard attention</li>
      <li><strong>tile_size</strong>  - controls the tile dimensions. Smaller tiles use less memory but have more loop overhead. Default 32 is reasonable; try smaller values for very long sequences.</li>
    </ul>
    <p>Note: this is a CPU implementation. The real wins from flash attention come on GPUs where the tiling aligns with SRAM/HBM boundaries. On CPU, the main benefit is reduced memory allocation for very long sequences.</p>
  `,

  ADWeightTying: `
    <h2>Weight Tying</h2>
    <p>Uses the embedding weight matrix as the output projection layer. Instead of learning a separate <code>[vocab_size x embed_dim]</code> matrix for the final logit projection, it reuses the one from the embedding layer.</p>
    <h3>What this actually does</h3>
    <p>The embedding matrix <code>W</code> has shape <code>[vocab_size x embed_dim]</code>. To get logits from hidden states <code>[embed_dim x seq_len]</code>, we just compute <code>W @ hidden</code>, which gives <code>[vocab_size x seq_len]</code> -a logit score for each vocab token at each position.</p>
    <p>Since it's the same tensor object as the embedding, gradients from the output loss flow back into the embedding weights. The embedding gets trained from two directions: the embedding lookup forward pass and the logit projection backward pass.</p>
    <h3>Inputs</h3>
    <ul>
      <li><strong>input</strong>  - hidden states, shape <code>[embed_dim x seq_len]</code></li>
      <li><strong>weights</strong>  - the embedding weight matrix (connect to the <code>weights</code> output of an ADEmbedding node)</li>
    </ul>
    <h3>When to use</h3>
    <p>Almost always, for language models. It cuts the parameter count (the embedding matrix is often the largest single parameter in a small model) and forces the input/output representations to be consistent.</p>
  `,

  ADRepetitionPenalty: `
    <h2>Repetition Penalty</h2>
    <p>Modifies logits to discourage the model from repeating tokens it has already generated. Applied right before sampling/argmax.</p>
    <h3>How the penalty works</h3>
    <p>For each token that appears in the <code>generated_ids</code> list:</p>
    <ul>
      <li>If its logit is positive, divide by <code>penalty</code> (makes it less likely)</li>
      <li>If its logit is negative, multiply by <code>penalty</code> (makes it even more negative, so even less likely)</li>
    </ul>
    <p>Tokens not in the list are left unchanged. This asymmetric treatment (dividing positives, multiplying negatives) comes from the CTRL paper and handles the sign correctly -both operations reduce the token's probability after softmax.</p>
    <h3>Choosing the penalty value</h3>
    <ul>
      <li><code>1.0</code>  - no effect</li>
      <li><code>1.1-1.3</code>  - mild, discourages exact repetition without changing the output much</li>
      <li><code>1.5-2.0</code>  - aggressive, noticeably changes the token distribution</li>
      <li><code>&gt;2.0</code>  - very strong, can make the output incoherent by suppressing common tokens too hard</li>
    </ul>
    <h3>Inputs</h3>
    <ul>
      <li><strong>logits</strong>  - model output logits <code>[vocab_size x seq_len]</code></li>
      <li><strong>generated_ids</strong>  - list of token IDs the model has produced so far</li>
    </ul>
  `,
};
