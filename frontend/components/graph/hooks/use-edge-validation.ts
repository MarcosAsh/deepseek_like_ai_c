import { useCallback } from "react";
import type { Connection, Edge } from "@xyflow/react";
import { useGraphStore } from "./use-graph-store";
import type { CustomNodeData } from "@/lib/graph-utils";

export function useEdgeValidation() {
  const { nodes, edges } = useGraphStore();

  const isValidConnection = useCallback(
    (connection: Edge | Connection): boolean => {
      // Must have all fields
      if (
        !connection.source ||
        !connection.target ||
        !connection.sourceHandle ||
        !connection.targetHandle
      )
        return false;

      // No self-connections
      if (connection.source === connection.target) return false;

      // Find source and target nodes
      const sourceNode = nodes.find((n) => n.id === connection.source);
      const targetNode = nodes.find((n) => n.id === connection.target);
      if (!sourceNode || !targetNode) return false;

      const sourceData = sourceNode.data as CustomNodeData;
      const targetData = targetNode.data as CustomNodeData;

      // Find the port types
      const sourcePort = sourceData.outputs.find(
        (p) => p.name === connection.sourceHandle
      );
      const targetPort = targetData.inputs.find(
        (p) => p.name === connection.targetHandle
      );
      if (!sourcePort || !targetPort) return false;

      // Port types must match
      if (sourcePort.type !== targetPort.type) return false;

      // No fan-in: target port already has an incoming edge
      const hasExistingEdge = edges.some(
        (e) =>
          e.target === connection.target &&
          e.targetHandle === connection.targetHandle
      );
      if (hasExistingEdge) return false;

      return true;
    },
    [nodes, edges]
  );

  return { isValidConnection };
}
