// Maps module type to its C++ source files (relative to project root)
export const MODULE_SOURCE_FILES: Record<string, { src: string[]; include: string[] }> = {
  ADEmbedding: {
    src: ["src/layers/ad_embedding.cpp"],
    include: ["include/layers/ad_embedding.hpp"],
  },
  ADMultiHeadAttention: {
    src: ["src/layers/ad_multi_head_attention.cpp"],
    include: ["include/layers/ad_multi_head_attention.hpp"],
  },
  ADFeedForward: {
    src: ["src/layers/ad_feed_forward.cpp"],
    include: ["include/layers/ad_feed_forward.hpp"],
  },
  ADLayerNorm: {
    src: ["src/layers/ad_layer_norm.cpp"],
    include: ["include/layers/ad_layer_norm.hpp"],
  },
  ADMoE: {
    src: ["src/layers/ad_moe.cpp"],
    include: ["include/layers/ad_moe.hpp"],
  },
  ADTransformerBlock: {
    src: ["src/layers/ad_transformer.cpp"],
    include: ["include/layers/ad_transformer.hpp"],
  },
  ADLinear: {
    src: ["src/layers/ad_linear.cpp"],
    include: ["include/layers/ad_linear.hpp"],
  },
  ADPositionalEncoding: {
    src: ["src/layers/ad_positional_encoding.cpp"],
    include: ["include/layers/ad_positional_encoding.hpp"],
  },
  Tokenizer: {
    src: ["src/tokenizer.cpp"],
    include: ["include/tokenizer.hpp"],
  },
  Add: {
    src: ["src/autodiff.cpp"],
    include: ["include/autodiff.hpp"],
  },
  MatMul: {
    src: ["src/autodiff.cpp"],
    include: ["include/autodiff.hpp"],
  },
  Transpose: {
    src: ["src/autodiff.cpp"],
    include: ["include/autodiff.hpp"],
  },
  CrossEntropy: {
    src: ["src/loss.cpp"],
    include: ["include/loss.hpp"],
  },
  Backward: {
    src: ["src/autodiff.cpp"],
    include: ["include/autodiff.hpp"],
  },
  TextInput: {
    src: ["src/server/module_wrappers.cpp"],
    include: ["include/server/module_wrapper.hpp"],
  },
  TokenIDsInput: {
    src: ["src/server/module_wrappers.cpp"],
    include: ["include/server/module_wrapper.hpp"],
  },
  IntInput: {
    src: ["src/server/module_wrappers.cpp"],
    include: ["include/server/module_wrapper.hpp"],
  },
  SeqLenExtractor: {
    src: ["src/server/module_wrappers.cpp"],
    include: ["include/server/module_wrapper.hpp"],
  },
};

// Get the public URL path for a source file
export function getSourceUrl(projectPath: string): string {
  // Convert "src/layers/ad_embedding.cpp" -> "/source/src/layers/ad_embedding.cpp"
  return `/source/${projectPath}`;
}
