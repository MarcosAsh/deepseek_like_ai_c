import type { Node, Edge } from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import type { GraphDef, NodeDef, EdgeDef, ModuleCatalogEntry } from "./types";

export interface CustomNodeData {
  moduleType: string;
  category: string;
  label: string;
  config: Record<string, unknown>;
  inputs: ModuleCatalogEntry["inputs"];
  outputs: ModuleCatalogEntry["outputs"];
  executionTime?: number;
  error?: string;
  outputData?: Record<string, unknown>;
  [key: string]: unknown;
}

// Convert React Flow state to backend GraphDef
export function toGraphDef(
  nodes: Node<CustomNodeData>[],
  edges: Edge[]
): GraphDef {
  const nodeDefs: NodeDef[] = nodes.map((n) => ({
    id: n.id,
    type: n.data.moduleType,
    config: n.data.config,
  }));

  const edgeDefs: EdgeDef[] = edges.map((e) => ({
    source_node: e.source,
    source_port: e.sourceHandle ?? "",
    target_node: e.target,
    target_port: e.targetHandle ?? "",
  }));

  return { nodes: nodeDefs, edges: edgeDefs };
}

// Convert backend GraphDef to React Flow state
export function fromGraphDef(
  graphDef: GraphDef,
  catalog: ModuleCatalogEntry[]
): { nodes: Node<CustomNodeData>[]; edges: Edge[] } {
  const catalogMap = new Map(catalog.map((m) => [m.type, m]));

  const nodes: Node<CustomNodeData>[] = graphDef.nodes.map((nd, i) => {
    const mod = catalogMap.get(nd.type);
    return {
      id: nd.id,
      type: "custom",
      position: { x: 0, y: 0 },
      data: {
        moduleType: nd.type,
        category: mod?.category ?? "unknown",
        label: nd.id,
        config: { ...mod?.default_config, ...nd.config },
        inputs: mod?.inputs ?? [],
        outputs: mod?.outputs ?? [],
      },
    };
  });

  const edges: Edge[] = graphDef.edges.map((ed, i) => {
    // Find the source port type for edge coloring
    const sourceNode = nodes.find((n) => n.id === ed.source_node);
    const sourceData = sourceNode?.data as CustomNodeData | undefined;
    const sourcePort = sourceData?.outputs?.find((p) => p.name === ed.source_port);
    return {
      id: `e-${ed.source_node}-${ed.source_port}-${ed.target_node}-${ed.target_port}`,
      source: ed.source_node,
      sourceHandle: ed.source_port,
      target: ed.target_node,
      targetHandle: ed.target_port,
      type: "custom",
      data: { portType: sourcePort?.type },
    };
  });

  return { nodes: applyDagreLayout(nodes, edges), edges };
}

// Auto-layout with dagre
export function applyDagreLayout(
  nodes: Node<CustomNodeData>[],
  edges: Edge[]
): Node<CustomNodeData>[] {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "LR", nodesep: 60, ranksep: 120 });

  for (const node of nodes) {
    g.setNode(node.id, { width: 240, height: 150 });
  }
  for (const edge of edges) {
    g.setEdge(edge.source, edge.target);
  }

  dagre.layout(g);

  return nodes.map((node) => {
    const pos = g.node(node.id);
    return {
      ...node,
      position: {
        x: pos.x - 120,
        y: pos.y - 75,
      },
    };
  });
}
