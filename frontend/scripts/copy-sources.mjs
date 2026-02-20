#!/usr/bin/env node
/**
 * Prebuild script: copies C++ source files into public/source/
 * so they're available as static assets on Vercel.
 */
import { mkdirSync, copyFileSync, existsSync } from "fs";
import { dirname, join } from "path";

const PROJECT_ROOT = join(import.meta.dirname, "..", "..");
const OUTPUT_DIR = join(import.meta.dirname, "..", "public", "source");

// All source files referenced by the frontend
const SOURCE_FILES = [
  "src/layers/ad_embedding.cpp",
  "src/layers/ad_multi_head_attention.cpp",
  "src/layers/ad_feed_forward.cpp",
  "src/layers/ad_layer_norm.cpp",
  "src/layers/ad_moe.cpp",
  "src/layers/ad_transformer.cpp",
  "src/layers/ad_linear.cpp",
  "src/layers/ad_positional_encoding.cpp",
  "src/tokenizer.cpp",
  "src/autodiff.cpp",
  "src/loss.cpp",
  "src/server/module_wrappers.cpp",
  "include/layers/ad_embedding.hpp",
  "include/layers/ad_multi_head_attention.hpp",
  "include/layers/ad_feed_forward.hpp",
  "include/layers/ad_layer_norm.hpp",
  "include/layers/ad_moe.hpp",
  "include/layers/ad_transformer.hpp",
  "include/layers/ad_linear.hpp",
  "include/layers/ad_positional_encoding.hpp",
  "include/tokenizer.hpp",
  "include/autodiff.hpp",
  "include/loss.hpp",
  "include/server/module_wrapper.hpp",
];

let copied = 0;
let skipped = 0;

for (const relPath of SOURCE_FILES) {
  const src = join(PROJECT_ROOT, relPath);
  const dest = join(OUTPUT_DIR, relPath);

  if (!existsSync(src)) {
    console.warn(`  SKIP: ${relPath} (not found)`);
    skipped++;
    continue;
  }

  mkdirSync(dirname(dest), { recursive: true });
  copyFileSync(src, dest);
  copied++;
}

console.log(`Copied ${copied} source files to public/source/ (${skipped} skipped)`);
