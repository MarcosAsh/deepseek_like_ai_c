import type { PortType, ModuleCategory } from "./types";

// Port type -> color for handles, badges, edges
export const PORT_TYPE_COLORS: Record<PortType, string> = {
  TEXT: "#3b82f6",       // blue-500
  TOKEN_IDS: "#22c55e",  // green-500
  TENSOR: "#f97316",     // orange-500
  AD_TENSOR: "#a855f7",  // purple-500
  SCALAR: "#eab308",     // yellow-500
  INT: "#6b7280",        // gray-500
};

// Convenience alias + fallback
export function getPortColor(type: PortType): string {
  return PORT_TYPE_COLORS[type] ?? "#6b7280";
}

// Tailwind class variants for port type badges
export const PORT_TYPE_BADGE_CLASSES: Record<PortType, string> = {
  TEXT: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  TOKEN_IDS: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  TENSOR: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200",
  AD_TENSOR: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
  SCALAR: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
  INT: "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200",
};

// Module category -> color for category badges
export const CATEGORY_COLORS: Record<string, string> = {
  input: "bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-200",
  preprocessing: "bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-200",
  embedding: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  normalization: "bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-200",
  attention: "bg-violet-100 text-violet-800 dark:bg-violet-900 dark:text-violet-200",
  feedforward: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200",
  linear: "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200",
  transformer: "bg-rose-100 text-rose-800 dark:bg-rose-900 dark:text-rose-200",
  math: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200",
  loss: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
  training: "bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-200",
  vision: "bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200",
  activation: "bg-lime-100 text-lime-800 dark:bg-lime-900 dark:text-lime-200",
  regularization: "bg-fuchsia-100 text-fuchsia-800 dark:bg-fuchsia-900 dark:text-fuchsia-200",
  pretrained: "bg-sky-100 text-sky-800 dark:bg-sky-900 dark:text-sky-200",
  utility: "bg-stone-100 text-stone-800 dark:bg-stone-800 dark:text-stone-200",
  generation: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
  moe: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
};

// Category display names
export const CATEGORY_LABELS: Record<string, string> = {
  input: "Input",
  preprocessing: "Preprocessing",
  embedding: "Embedding",
  normalization: "Normalization",
  attention: "Attention",
  feedforward: "Feed-Forward",
  linear: "Linear",
  transformer: "Transformer",
  math: "Math Operations",
  loss: "Loss",
  training: "Training",
  vision: "Vision",
  activation: "Activation",
  regularization: "Regularization",
  pretrained: "Pretrained",
  utility: "Utility",
  generation: "Generation",
  moe: "Mixture of Experts",
};

// Hex colors for minimap node coloring
export const CATEGORY_COLORS_HEX: Record<string, string> = {
  input: "#64748b",        // slate-500
  preprocessing: "#06b6d4", // cyan-500
  embedding: "#3b82f6",    // blue-500
  normalization: "#14b8a6", // teal-500
  attention: "#8b5cf6",    // violet-500
  feedforward: "#f97316",  // orange-500
  linear: "#f59e0b",       // amber-500
  transformer: "#f43f5e",  // rose-500
  math: "#10b981",         // emerald-500
  loss: "#ef4444",         // red-500
  training: "#ec4899",     // pink-500
  vision: "#6366f1",       // indigo-500
  activation: "#84cc16",   // lime-500
  regularization: "#d946ef", // fuchsia-500
  pretrained: "#0ea5e9",   // sky-500
  utility: "#78716c",      // stone-500
  generation: "#eab308",   // yellow-500
  moe: "#a855f7",          // purple-500
};

// All categories in display order
export const ALL_CATEGORIES: ModuleCategory[] = [
  "input",
  "preprocessing",
  "embedding",
  "normalization",
  "attention",
  "feedforward",
  "linear",
  "transformer",
  "vision",
  "activation",
  "regularization",
  "math",
  "loss",
  "training",
  "pretrained",
  "utility",
  "generation",
  "moe",
];
