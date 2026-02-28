"use client";

import { memo } from "react";
import { BaseEdge, getSmoothStepPath } from "@xyflow/react";
import type { EdgeProps } from "@xyflow/react";
import { getPortColor } from "@/lib/constants";
import type { PortType } from "@/lib/types";
import { useGraphStore } from "./hooks/use-graph-store";

function CustomEdgeComponent({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
  data,
}: EdgeProps) {
  const flowingEdgeIds = useGraphStore((s) => s.flowingEdgeIds);
  const completedNodeIds = useGraphStore((s) => s.completedNodeIds);
  const portType = (data as { portType?: PortType } | undefined)?.portType;
  const sourceNodeId = (data as { sourceNodeId?: string } | undefined)?.sourceNodeId;
  const strokeColor = portType ? getPortColor(portType) : "var(--border)";

  const isFlowing = flowingEdgeIds.has(id);
  const sourceCompleted = sourceNodeId ? completedNodeIds.has(sourceNodeId) : false;

  const [edgePath] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: 8,
  });

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          strokeWidth: isFlowing ? 3 : 2,
          stroke: strokeColor,
          opacity: sourceCompleted ? 1 : 0.7,
          filter: isFlowing ? `drop-shadow(0 0 4px ${strokeColor})` : undefined,
          transition: "stroke-width 0.3s, opacity 0.3s",
          ...style,
        }}
      />
      {isFlowing && (
        <>
          <circle r="4" fill={strokeColor} opacity="0.9">
            <animateMotion dur="0.6s" repeatCount="indefinite" path={edgePath} />
          </circle>
          <circle r="2" fill="white" opacity="0.8">
            <animateMotion dur="0.6s" repeatCount="indefinite" path={edgePath} />
          </circle>
        </>
      )}
    </>
  );
}

export const CustomEdge = memo(CustomEdgeComponent);
