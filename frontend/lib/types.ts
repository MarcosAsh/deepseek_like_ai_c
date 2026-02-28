// Port types matching the backend PortType enum exactly
export type PortType = "TEXT" | "TOKEN_IDS" | "TENSOR" | "AD_TENSOR" | "SCALAR" | "INT";

export interface PortDescriptor {
  name: string;
  type: PortType;
  optional: boolean;
}

export interface ModuleCatalogEntry {
  type: string;
  category: string;
  description: string;
  default_config: Record<string, unknown>;
  inputs: PortDescriptor[];
  outputs: PortDescriptor[];
}

export interface ModulesResponse {
  modules: ModuleCatalogEntry[];
}

// Tensor data as serialized by the backend
export interface TensorData {
  type: PortType;
  shape?: number[];
  data?: number[];
  value?: unknown;
  stats?: TensorStats;
  truncated?: boolean;
  grad?: {
    shape: number[];
    data: number[];
    stats: TensorStats;
    truncated: boolean;
  };
}

export interface TensorStats {
  min: number;
  max: number;
  mean: number;
  std: number;
}

// Graph definitions matching backend exactly
export interface NodeDef {
  id: string;
  type: string;
  config: Record<string, unknown>;
}

export interface EdgeDef {
  source_node: string;
  source_port: string;
  target_node: string;
  target_port: string;
}

export interface GraphDef {
  nodes: NodeDef[];
  edges: EdgeDef[];
}

// Execution results - matches actual backend response format
export interface NodeResult {
  node_id: string;
  node_type: string;
  execution_time_ms: number;
  outputs: Record<string, TensorData>;
  error: string;
}

// Raw backend response: nodes keyed by id
interface BackendNodeResult {
  type: string;
  execution_time_ms: number;
  outputs: Record<string, TensorData> | null;
  error?: string;
}

export interface GraphResultRaw {
  nodes: Record<string, BackendNodeResult>;
  execution_order: string[];
  total_time_ms: number;
  error?: string;
}

// Normalized format used by frontend
export interface GraphResult {
  node_results: NodeResult[];
  execution_order: string[];
  total_time_ms: number;
  error: string;
}

// Convert raw backend response to normalized format
export function normalizeGraphResult(raw: GraphResultRaw): GraphResult {
  const node_results: NodeResult[] = [];
  if (raw.nodes) {
    for (const [nodeId, data] of Object.entries(raw.nodes)) {
      node_results.push({
        node_id: nodeId,
        node_type: data.type,
        execution_time_ms: data.execution_time_ms,
        outputs: data.outputs ?? {},
        error: data.error ?? "",
      });
    }
  }
  return {
    node_results,
    execution_order: raw.execution_order ?? [],
    total_time_ms: raw.total_time_ms ?? 0,
    error: raw.error ?? "",
  };
}

// Single node execution
export interface ExecuteNodeRequest {
  type: string;
  config: Record<string, unknown>;
  inputs: Record<string, { type: PortType; value: unknown }>;
}

export interface ExecuteNodeResponse {
  [portName: string]: TensorData;
}

// Presets
export interface PresetGraph {
  name: string;
  description: string;
  nodes: NodeDef[];
  edges: EdgeDef[];
}

export interface PresetsResponse {
  presets: PresetGraph[];
}

// Health
export interface HealthResponse {
  status: string;
  version: string;
}

// Module categories
export type ModuleCategory =
  | "input"
  | "preprocessing"
  | "embedding"
  | "normalization"
  | "attention"
  | "feedforward"
  | "linear"
  | "transformer"
  | "vision"
  | "activation"
  | "regularization"
  | "math"
  | "loss"
  | "training"
  | "pretrained"
  | "utility"
  | "generation"
  | "moe";
