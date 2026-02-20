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
  shape?: [number, number];
  data?: number[];
  value?: unknown;
  stats?: TensorStats;
  truncated?: boolean;
  grad?: {
    shape: [number, number];
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

// Execution results
export interface NodeResult {
  node_id: string;
  node_type: string;
  execution_time_ms: number;
  outputs: Record<string, TensorData>;
  error: string;
}

export interface GraphResult {
  node_results: NodeResult[];
  execution_order: string[];
  total_time_ms: number;
  error: string;
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
  | "math"
  | "loss"
  | "training";
