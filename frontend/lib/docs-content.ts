export interface DocPage {
  slug: string;
  title: string;
  description: string;
  category: "concept" | "tutorial";
  order: number;
  content: string; // HTML content
}

export const CONCEPTS: DocPage[] = [
  {
    slug: "tensors-and-matrices",
    title: "Tensors and Matrices",
    description: "The fundamental data structure of neural networks",
    category: "concept",
    order: 1,
    content: `
<h1>Tensors and Matrices</h1>
<p>Tensors are the fundamental data structure in neural networks. In our implementation, we use 2D tensors (matrices) represented as <code>[rows x cols]</code> arrays of floating-point numbers.</p>

<h2>What is a Tensor?</h2>
<p>A tensor is a multi-dimensional array of numbers. In mathematics:</p>
<ul>
  <li>A <strong>scalar</strong> is a 0-dimensional tensor (a single number)</li>
  <li>A <strong>vector</strong> is a 1-dimensional tensor (a list of numbers)</li>
  <li>A <strong>matrix</strong> is a 2-dimensional tensor (a grid of numbers)</li>
  <li>Higher-dimensional tensors generalize to 3D, 4D, etc.</li>
</ul>

<h2>Our Implementation</h2>
<p>In this project, the <code>Tensor</code> class stores data in a flat <code>std::vector&lt;float&gt;</code> with row-major layout. Access is via <code>tensor(row, col)</code>.</p>
<pre><code>class Tensor {
    int rows, cols;
    std::vector&lt;float&gt; data;

    float&amp; operator()(int r, int c) {
        return data[r * cols + c];
    }
};</code></pre>

<h2>Key Operations</h2>
<ul>
  <li><strong>Matrix Multiplication (MatMul)</strong>: <code>C = A @ B</code> where <code>C[i,j] = sum(A[i,k] * B[k,j])</code></li>
  <li><strong>Element-wise Addition</strong>: <code>C[i,j] = A[i,j] + B[i,j]</code></li>
  <li><strong>Transpose</strong>: Swap rows and columns</li>
</ul>

<div class="callout info">
  <p><strong>Try it:</strong> Go to the <a href="/modules/MatMul">MatMul module</a> to see matrix multiplication in action.</p>
</div>
`,
  },
  {
    slug: "automatic-differentiation",
    title: "Automatic Differentiation",
    description: "How gradients are computed automatically through the computation graph",
    category: "concept",
    order: 2,
    content: `
<h1>Automatic Differentiation</h1>
<p>Automatic differentiation (autodiff) is the technique that makes training neural networks possible. It computes exact gradients efficiently by tracking operations and applying the chain rule.</p>

<h2>Forward vs. Reverse Mode</h2>
<p>Our implementation uses <strong>reverse-mode autodiff</strong> (backpropagation), which is efficient when you have many inputs and few outputs (typical for loss functions).</p>

<h2>The AD Tensor</h2>
<p>Each <code>ADTensor</code> stores:</p>
<ul>
  <li><code>val</code>: The forward-pass value (a <code>Tensor</code>)</li>
  <li><code>grad</code>: The accumulated gradient (same shape as <code>val</code>)</li>
  <li><code>deps</code>: Parent tensors and their backpropagation functions</li>
</ul>

<h2>How Backpropagation Works</h2>
<ol>
  <li>Compute the forward pass, building a computation graph</li>
  <li>Set the loss gradient to 1.0</li>
  <li>Traverse the graph in reverse topological order</li>
  <li>At each node, call the backprop function to propagate gradients to parents</li>
</ol>

<h2>Chain Rule</h2>
<p>If <code>z = f(y)</code> and <code>y = g(x)</code>, then <code>dz/dx = dz/dy * dy/dx</code>.</p>
<p>Autodiff applies this rule automatically for every operation in the graph.</p>

<div class="callout info">
  <p><strong>Try it:</strong> Build a graph with a <a href="/graph">Backward module</a> to see gradients propagate through the computation graph.</p>
</div>
`,
  },
  {
    slug: "gradient-descent",
    title: "Gradient Descent",
    description: "The optimization algorithm that trains neural networks",
    category: "concept",
    order: 3,
    content: `
<h1>Gradient Descent</h1>
<p>Gradient descent is the core optimization algorithm for training neural networks. It iteratively adjusts parameters to minimize the loss function.</p>

<h2>The Update Rule</h2>
<p><code>theta = theta - learning_rate * gradient</code></p>
<p>Each parameter is moved in the direction that decreases the loss.</p>

<h2>Why It Works</h2>
<p>The gradient points in the direction of steepest increase. By moving in the opposite direction, we decrease the loss. With a small enough learning rate, this converges to a local minimum.</p>

<h2>In Practice</h2>
<ol>
  <li><strong>Forward pass</strong>: Compute predictions and loss</li>
  <li><strong>Backward pass</strong>: Compute gradients via autodiff</li>
  <li><strong>Update</strong>: Adjust each parameter using its gradient</li>
  <li>Repeat</li>
</ol>
`,
  },
  {
    slug: "softmax-function",
    title: "Softmax Function",
    description: "Converting raw scores into probabilities",
    category: "concept",
    order: 4,
    content: `
<h1>Softmax Function</h1>
<p>The softmax function converts a vector of arbitrary real numbers into a probability distribution. It's used extensively in attention mechanisms and output layers.</p>

<h2>Formula</h2>
<p><code>softmax(x_i) = exp(x_i) / sum(exp(x_j))</code></p>

<h2>Properties</h2>
<ul>
  <li>All output values are in [0, 1]</li>
  <li>Output values sum to 1 (valid probability distribution)</li>
  <li>Larger input values get exponentially larger probabilities</li>
  <li>Invariant to constant shifts: <code>softmax(x + c) = softmax(x)</code></li>
</ul>

<h2>Numerical Stability</h2>
<p>Our implementation uses the log-sum-exp trick: subtract the maximum value before exponentiating to prevent overflow.</p>
<pre><code>max_val = max(x)
softmax(x_i) = exp(x_i - max_val) / sum(exp(x_j - max_val))</code></pre>
`,
  },
  {
    slug: "cross-entropy-loss",
    title: "Cross-Entropy Loss",
    description: "Measuring the difference between predicted and actual distributions",
    category: "concept",
    order: 5,
    content: `
<h1>Cross-Entropy Loss</h1>
<p>Cross-entropy measures how different two probability distributions are. In language modeling, it measures how well the model's predicted distribution matches the actual next token.</p>

<h2>Formula</h2>
<p><code>L = -sum(y_true * log(y_pred))</code></p>
<p>For classification with one-hot targets, this simplifies to:</p>
<p><code>L = -log(p(correct_class))</code></p>

<h2>In Language Modeling</h2>
<p>The model outputs logits of shape <code>[vocab_size x seq_len]</code>. Cross-entropy:</p>
<ol>
  <li>Applies softmax to convert logits to probabilities</li>
  <li>Takes the log probability of the correct token at each position</li>
  <li>Averages the negative log probabilities</li>
</ol>

<div class="callout info">
  <p><strong>Try it:</strong> Use the <a href="/modules/CrossEntropy">CrossEntropy module</a> to compute loss between logits and targets.</p>
</div>
`,
  },
  {
    slug: "alibi-attention",
    title: "ALiBi Attention",
    description: "Attention with Linear Biases for position encoding",
    category: "concept",
    order: 6,
    content: `
<h1>ALiBi: Attention with Linear Biases</h1>
<p>ALiBi is a position encoding method that adds linear distance-based biases directly to attention scores, replacing traditional positional embeddings.</p>

<h2>How It Works</h2>
<p>Instead of adding positional embeddings to token embeddings, ALiBi subtracts a penalty proportional to the distance between the query and key positions:</p>
<p><code>attention_score[i,j] = q_i · k_j - m * |i - j|</code></p>
<p>where <code>m</code> is a head-specific slope that decreases geometrically across heads.</p>

<h2>Benefits</h2>
<ul>
  <li><strong>Length generalization</strong>: Can extrapolate to longer sequences than seen during training</li>
  <li><strong>Simplicity</strong>: No learned positional parameters</li>
  <li><strong>Efficiency</strong>: Biases can be precomputed</li>
</ul>

<h2>In Our Implementation</h2>
<p>The <code>ADMultiHeadAttention</code> module computes ALiBi biases per head with slopes <code>m = 1 / 2^(8*h/H)</code> where <code>h</code> is the head index and <code>H</code> is the total number of heads.</p>
`,
  },
  {
    slug: "residual-connections",
    title: "Residual Connections",
    description: "Skip connections that enable deep networks to train",
    category: "concept",
    order: 7,
    content: `
<h1>Residual Connections</h1>
<p>Residual connections (skip connections) add the input of a layer directly to its output: <code>output = layer(x) + x</code>. This simple idea enables training of very deep networks.</p>

<h2>Why They Work</h2>
<ul>
  <li><strong>Gradient flow</strong>: Gradients can flow directly through the skip connection, preventing vanishing gradients</li>
  <li><strong>Identity shortcut</strong>: The network can easily learn the identity function by setting layer weights to zero</li>
  <li><strong>Incremental learning</strong>: Each layer learns a small "residual" adjustment to its input</li>
</ul>

<h2>In Transformers</h2>
<p>Every transformer block uses two residual connections:</p>
<pre><code>x = x + Attention(LayerNorm(x))  // Attention sub-layer
x = x + FFN(LayerNorm(x))        // Feed-forward sub-layer</code></pre>

<div class="callout info">
  <p><strong>Try it:</strong> The <a href="/modules/ADTransformerBlock">TransformerBlock module</a> implements both residual connections.</p>
</div>
`,
  },
];

export const TUTORIALS: DocPage[] = [
  {
    slug: "understanding-tokens",
    title: "Understanding Tokens",
    description: "How text is converted into numbers for processing",
    category: "tutorial",
    order: 1,
    content: `
<h1>Tutorial 1: Understanding Tokens</h1>
<p>Language models don't process raw text. They work with numbers. Tokenization is the first step, converting text into a sequence of integer IDs.</p>

<h2>Step 1: Vocabulary</h2>
<p>A vocabulary is a fixed mapping from text pieces to integers. For example:</p>
<pre><code>"hello" -> 42
"world" -> 17
"the" -> 3</code></pre>

<h2>Step 2: BPE Tokenization</h2>
<p>Byte Pair Encoding (BPE) creates a vocabulary of subword units by iteratively merging common pairs:</p>
<ol>
  <li>Start with individual characters: <code>h e l l o</code></li>
  <li>Merge the most frequent pair: <code>h e ll o</code> (merged "l" + "l")</li>
  <li>Continue merging: <code>he ll o</code>, <code>hello</code></li>
</ol>

<h2>Try It</h2>
<ol>
  <li>Go to the <a href="/modules/TextInput">TextInput module</a> and enter some text</li>
  <li>Connect it to the <a href="/modules/Tokenizer">Tokenizer module</a></li>
  <li>See the resulting token IDs</li>
</ol>

<div class="callout tip">
  <p><strong>Tip:</strong> Use the <a href="/graph">Graph Editor</a> to chain TextInput -> Tokenizer and see the full pipeline.</p>
</div>
`,
  },
  {
    slug: "embedding-vectors",
    title: "Embedding Vectors",
    description: "Turning token IDs into rich vector representations",
    category: "tutorial",
    order: 2,
    content: `
<h1>Tutorial 2: Embedding Vectors</h1>
<p>After tokenization, each token ID is converted into a dense vector representation. This is the embedding step.</p>

<h2>Why Embeddings?</h2>
<p>Token IDs are arbitrary integers. The embedding layer gives each token a learnable vector that captures its meaning. Similar tokens end up with similar vectors.</p>

<h2>The Embedding Lookup</h2>
<p>The embedding layer is simply a matrix <code>W</code> of shape <code>[vocab_size x embed_dim]</code>. For token ID <code>i</code>, the embedding is row <code>i</code> of <code>W</code>.</p>

<h2>Try It</h2>
<ol>
  <li>Go to the <a href="/modules/ADEmbedding">ADEmbedding module</a></li>
  <li>Enter token IDs like <code>[1, 2, 3, 4, 5]</code></li>
  <li>Configure <code>vocab_size: 256</code> and <code>embed_dim: 64</code></li>
  <li>Run and observe the output tensor shape: <code>[64 x 5]</code> (each column is one token's embedding)</li>
</ol>
`,
  },
  {
    slug: "attention-mechanism",
    title: "The Attention Mechanism",
    description: "How tokens learn to attend to each other",
    category: "tutorial",
    order: 3,
    content: `
<h1>Tutorial 3: The Attention Mechanism</h1>
<p>Attention is the core innovation of the transformer architecture. It allows each token to "look at" every other token and determine which are most relevant.</p>

<h2>Query, Key, Value</h2>
<p>Each token's embedding is projected into three vectors:</p>
<ul>
  <li><strong>Query (Q)</strong>: "What am I looking for?"</li>
  <li><strong>Key (K)</strong>: "What do I contain?"</li>
  <li><strong>Value (V)</strong>: "What information do I provide?"</li>
</ul>

<h2>Attention Scores</h2>
<p><code>scores = softmax(Q * K^T / sqrt(d_k))</code></p>
<p>The dot product between query and key measures relevance. Softmax converts scores to weights that sum to 1.</p>

<h2>Multi-Head Attention</h2>
<p>Instead of one attention computation, we use multiple "heads" that attend to different aspects. Each head operates on a smaller dimension, and results are concatenated.</p>

<h2>Try It</h2>
<ol>
  <li>Go to the <a href="/modules/ADMultiHeadAttention">ADMultiHeadAttention module</a></li>
  <li>The output tensor shows how each token's representation is updated by attending to all other tokens</li>
</ol>
`,
  },
  {
    slug: "transformer-block",
    title: "The Transformer Block",
    description: "Combining attention and feed-forward layers",
    category: "tutorial",
    order: 4,
    content: `
<h1>Tutorial 4: The Transformer Block</h1>
<p>A transformer block is the fundamental building unit. Modern LLMs stack dozens or hundreds of these blocks.</p>

<h2>Architecture</h2>
<pre><code>x = x + Attention(LayerNorm(x))   // Sub-layer 1
x = x + FFN(LayerNorm(x))          // Sub-layer 2</code></pre>
<p>Each sub-layer has:</p>
<ol>
  <li><strong>Layer normalization</strong>: Stabilizes activations</li>
  <li><strong>Core operation</strong>: Attention or feed-forward</li>
  <li><strong>Residual connection</strong>: Adds input back to output</li>
</ol>

<h2>Try It</h2>
<p>Use the <a href="/graph">Graph Editor</a> and load the "Full Transformer Block" preset to see all components wired together.</p>
`,
  },
  {
    slug: "training-pipeline",
    title: "The Training Pipeline",
    description: "End-to-end training: forward pass, loss, and backpropagation",
    category: "tutorial",
    order: 5,
    content: `
<h1>Tutorial 5: The Training Pipeline</h1>
<p>Training a language model involves repeatedly: making predictions, measuring error, computing gradients, and updating parameters.</p>

<h2>The Full Pipeline</h2>
<ol>
  <li><strong>Tokenize</strong>: Convert text to token IDs</li>
  <li><strong>Embed</strong>: Convert token IDs to vectors</li>
  <li><strong>Transform</strong>: Process through transformer blocks</li>
  <li><strong>Predict</strong>: Project to vocabulary logits</li>
  <li><strong>Loss</strong>: Compute cross-entropy with targets</li>
  <li><strong>Backward</strong>: Compute gradients for all parameters</li>
  <li><strong>Update</strong>: Adjust parameters using gradient descent</li>
</ol>

<h2>Try It</h2>
<p>Load the "Full Training Pipeline" preset in the <a href="/graph">Graph Editor</a> to see the complete pipeline with tokenization, embedding, transformer, loss, and backward pass all connected.</p>
`,
  },
  {
    slug: "mixture-of-experts",
    title: "Mixture of Experts",
    description: "Scaling models with conditional computation",
    category: "tutorial",
    order: 6,
    content: `
<h1>Tutorial 6: Mixture of Experts</h1>
<p>Mixture of Experts (MoE) is a technique that scales model capacity without proportionally scaling computation. It's a key architecture in DeepSeek models.</p>

<h2>The Idea</h2>
<p>Instead of one large feed-forward network, use many smaller "expert" networks. A learned router selects which experts process each token.</p>

<h2>How Routing Works</h2>
<ol>
  <li>A linear layer computes scores for each expert: <code>scores = W_router * x</code></li>
  <li>Select the top-k experts with highest scores</li>
  <li>Normalize selected scores with softmax</li>
  <li>Combine expert outputs weighted by scores</li>
</ol>

<h2>Load Balancing</h2>
<p>Without regularization, the router might always select the same experts. An auxiliary loss encourages uniform utilization across all experts.</p>

<h2>Try It</h2>
<ol>
  <li>Go to the <a href="/modules/ADMoE">ADMoE module</a></li>
  <li>Configure 4 experts with top-2 routing</li>
  <li>Observe the output and optional auxiliary loss</li>
</ol>
`,
  },
];

export function getConceptBySlug(slug: string): DocPage | undefined {
  return CONCEPTS.find((c) => c.slug === slug);
}

export function getTutorialBySlug(slug: string): DocPage | undefined {
  return TUTORIALS.find((t) => t.slug === slug);
}
