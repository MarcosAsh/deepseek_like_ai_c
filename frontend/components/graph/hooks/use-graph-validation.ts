import { useCallback } from "react";
import { useGraphStore } from "./use-graph-store";
import type { CustomNodeData } from "@/lib/graph-utils";
import type { Edge, Node } from "@xyflow/react";

export interface ValidationIssue {
  type: "error" | "warning";
  message: string;
  nodeId?: string;
  edgeId?: string;
}

export interface ValidationResult {
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
  valid: boolean;
}

function checkRequiredPorts(
  nodes: Node<CustomNodeData>[],
  edges: Edge[]
): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  for (const node of nodes) {
    const data = node.data as CustomNodeData;
    for (const input of data.inputs) {
      if (input.optional) continue;
      const hasEdge = edges.some(
        (e) => e.target === node.id && e.targetHandle === input.name
      );
      if (!hasEdge) {
        issues.push({
          type: "error",
          message: `"${data.moduleType}" is missing required input "${input.name}" (${input.type})`,
          nodeId: node.id,
        });
      }
    }
  }
  return issues;
}

function checkTypeMismatches(
  nodes: Node<CustomNodeData>[],
  edges: Edge[]
): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));

  for (const edge of edges) {
    const sourceNode = nodeMap.get(edge.source);
    const targetNode = nodeMap.get(edge.target);
    if (!sourceNode || !targetNode) continue;

    const sourceData = sourceNode.data as CustomNodeData;
    const targetData = targetNode.data as CustomNodeData;

    const sourcePort = sourceData.outputs.find((p) => p.name === edge.sourceHandle);
    const targetPort = targetData.inputs.find((p) => p.name === edge.targetHandle);

    if (sourcePort && targetPort && sourcePort.type !== targetPort.type) {
      issues.push({
        type: "error",
        message: `Type mismatch: ${sourceData.moduleType}.${sourcePort.name} (${sourcePort.type}) -> ${targetData.moduleType}.${targetPort.name} (${targetPort.type})`,
        edgeId: edge.id,
        nodeId: edge.target,
      });
    }
  }
  return issues;
}

function checkCycles(
  nodes: Node<CustomNodeData>[],
  edges: Edge[]
): ValidationIssue[] {
  // Kahn's algorithm - if we can't process all nodes, there's a cycle
  const inDegree = new Map<string, number>();
  const adj = new Map<string, string[]>();

  for (const node of nodes) {
    inDegree.set(node.id, 0);
    adj.set(node.id, []);
  }

  for (const edge of edges) {
    adj.get(edge.source)?.push(edge.target);
    inDegree.set(edge.target, (inDegree.get(edge.target) ?? 0) + 1);
  }

  const queue: string[] = [];
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id);
  }

  let processed = 0;
  while (queue.length > 0) {
    const nodeId = queue.shift()!;
    processed++;
    for (const neighbor of adj.get(nodeId) ?? []) {
      const newDeg = (inDegree.get(neighbor) ?? 1) - 1;
      inDegree.set(neighbor, newDeg);
      if (newDeg === 0) queue.push(neighbor);
    }
  }

  if (processed < nodes.length) {
    return [
      {
        type: "error",
        message: `Graph contains a cycle (${nodes.length - processed} nodes involved)`,
      },
    ];
  }
  return [];
}

function checkOrphanNodes(
  nodes: Node<CustomNodeData>[],
  edges: Edge[]
): ValidationIssue[] {
  if (nodes.length <= 1) return [];
  const issues: ValidationIssue[] = [];
  const connected = new Set<string>();
  for (const edge of edges) {
    connected.add(edge.source);
    connected.add(edge.target);
  }
  for (const node of nodes) {
    if (!connected.has(node.id)) {
      const data = node.data as CustomNodeData;
      issues.push({
        type: "warning",
        message: `"${data.moduleType}" has no connections`,
        nodeId: node.id,
      });
    }
  }
  return issues;
}

export function useGraphValidation() {
  const { nodes, edges } = useGraphStore();

  const validate = useCallback((): ValidationResult => {
    const errors: ValidationIssue[] = [];
    const warnings: ValidationIssue[] = [];

    const requiredPortIssues = checkRequiredPorts(nodes, edges);
    const typeMismatchIssues = checkTypeMismatches(nodes, edges);
    const cycleIssues = checkCycles(nodes, edges);
    const orphanIssues = checkOrphanNodes(nodes, edges);

    for (const issue of [...requiredPortIssues, ...typeMismatchIssues, ...cycleIssues, ...orphanIssues]) {
      if (issue.type === "error") errors.push(issue);
      else warnings.push(issue);
    }

    return { errors, warnings, valid: errors.length === 0 };
  }, [nodes, edges]);

  return { validate };
}
