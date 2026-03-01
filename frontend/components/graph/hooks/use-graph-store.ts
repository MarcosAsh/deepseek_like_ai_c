import { create } from "zustand";
import type { Node, Edge } from "@xyflow/react";
import type { CustomNodeData } from "@/lib/graph-utils";
import type { NodeResult, GraphResult } from "@/lib/types";
import type { ValidationResult } from "./use-graph-validation";

interface HistoryEntry {
  nodes: Node<CustomNodeData>[];
  edges: Edge[];
}

interface ExecutionSnapshot {
  id: string;
  timestamp: number;
  nodeResults: Map<string, NodeResult>;
  totalTimeMs: number;
  executionOrder: string[];
}

interface GraphState {
  nodes: Node<CustomNodeData>[];
  edges: Edge[];
  selectedNodeId: string | null;
  results: Map<string, NodeResult>;
  isExecuting: boolean;
  executingNodeId: string | null;

  // Animation state
  flowingEdgeIds: Set<string>;
  completedNodeIds: Set<string>;

  // Undo/redo
  history: HistoryEntry[];
  future: HistoryEntry[];

  // Validation
  validationResult: ValidationResult | null;

  // Execution history
  executionHistory: ExecutionSnapshot[];

  setNodes: (nodes: Node<CustomNodeData>[]) => void;
  setEdges: (edges: Edge[]) => void;
  onNodesChange: (changes: Node<CustomNodeData>[]) => void;
  onEdgesChange: (edges: Edge[]) => void;
  selectNode: (id: string | null) => void;
  addNode: (node: Node<CustomNodeData>) => void;
  removeNode: (id: string) => void;
  duplicateNode: (id: string) => void;
  updateNodeConfig: (
    id: string,
    config: Record<string, unknown>
  ) => void;
  setResults: (results: Map<string, NodeResult>) => void;
  setIsExecuting: (v: boolean) => void;
  setExecutingNodeId: (id: string | null) => void;
  setFlowingEdgeIds: (ids: Set<string>) => void;
  setCompletedNodeIds: (ids: Set<string>) => void;
  setValidationResult: (r: ValidationResult | null) => void;
  clearResults: () => void;
  clearAll: () => void;

  // Undo/redo
  pushHistory: () => void;
  undo: () => void;
  redo: () => void;

  // Execution history
  addExecutionSnapshot: (result: GraphResult) => void;
  clearExecutionHistory: () => void;
}

export const useGraphStore = create<GraphState>((set) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,
  results: new Map(),
  isExecuting: false,
  executingNodeId: null,
  flowingEdgeIds: new Set(),
  completedNodeIds: new Set(),
  validationResult: null,
  history: [],
  future: [],
  executionHistory: [],

  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),
  onNodesChange: (nodes) => set({ nodes }),
  onEdgesChange: (edges) => set({ edges }),
  selectNode: (id) => set({ selectedNodeId: id }),

  addNode: (node) =>
    set((state) => ({ nodes: [...state.nodes, node] })),

  removeNode: (id) =>
    set((state) => ({
      nodes: state.nodes.filter((n) => n.id !== id),
      edges: state.edges.filter(
        (e) => e.source !== id && e.target !== id
      ),
      selectedNodeId:
        state.selectedNodeId === id ? null : state.selectedNodeId,
    })),

  duplicateNode: (id) =>
    set((state) => {
      const node = state.nodes.find((n) => n.id === id);
      if (!node) return state;
      const data = node.data as CustomNodeData;
      const newId = `${data.moduleType.toLowerCase()}_${Date.now()}`;
      const newNode: Node<CustomNodeData> = {
        ...node,
        id: newId,
        position: {
          x: node.position.x + 50,
          y: node.position.y + 50,
        },
        data: { ...data, label: newId },
        selected: false,
      };
      return { nodes: [...state.nodes, newNode] };
    }),

  updateNodeConfig: (id, config) =>
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, config } } : n
      ),
    })),

  setResults: (results) => set({ results }),
  setIsExecuting: (v) => set({ isExecuting: v }),
  setExecutingNodeId: (id) => set({ executingNodeId: id }),
  setFlowingEdgeIds: (ids) => set({ flowingEdgeIds: ids }),
  setCompletedNodeIds: (ids) => set({ completedNodeIds: ids }),
  setValidationResult: (r) => set({ validationResult: r }),
  clearResults: () => set({ results: new Map(), executingNodeId: null, flowingEdgeIds: new Set(), completedNodeIds: new Set() }),
  clearAll: () =>
    set({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      results: new Map(),
      history: [],
      future: [],
      executingNodeId: null,
      flowingEdgeIds: new Set(),
      completedNodeIds: new Set(),
      validationResult: null,
    }),

  // Undo/redo
  pushHistory: () =>
    set((state) => ({
      history: [
        ...state.history.slice(-30),
        { nodes: state.nodes, edges: state.edges },
      ],
      future: [],
    })),

  undo: () =>
    set((state) => {
      if (state.history.length === 0) return state;
      const prev = state.history[state.history.length - 1];
      return {
        history: state.history.slice(0, -1),
        future: [
          { nodes: state.nodes, edges: state.edges },
          ...state.future,
        ],
        nodes: prev.nodes,
        edges: prev.edges,
      };
    }),

  redo: () =>
    set((state) => {
      if (state.future.length === 0) return state;
      const next = state.future[0];
      return {
        future: state.future.slice(1),
        history: [
          ...state.history,
          { nodes: state.nodes, edges: state.edges },
        ],
        nodes: next.nodes,
        edges: next.edges,
      };
    }),

  addExecutionSnapshot: (result: GraphResult) =>
    set((state) => {
      const resultsMap = new Map<string, NodeResult>();
      for (const nr of result.node_results ?? []) {
        resultsMap.set(nr.node_id, nr);
      }
      const snapshot: ExecutionSnapshot = {
        id: `exec_${Date.now()}`,
        timestamp: Date.now(),
        nodeResults: resultsMap,
        totalTimeMs: result.total_time_ms ?? 0,
        executionOrder: result.execution_order ?? [],
      };
      return {
        executionHistory: [
          snapshot,
          ...state.executionHistory.slice(0, 19),
        ],
      };
    }),

  clearExecutionHistory: () => set({ executionHistory: [] }),
}));
