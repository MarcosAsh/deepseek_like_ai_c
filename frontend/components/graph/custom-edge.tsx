"use client";

import { memo } from "react";
import { BaseEdge, getSmoothStepPath } from "@xyflow/react";
import type { EdgeProps } from "@xyflow/react";
import { getPortColor } from "@/lib/port-colors";
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
  const isExecuting = useGraphStore((s) => s.isExecuting);
  const portType = (data as { portType?: PortType } | undefined)?.portType;
  const strokeColor = portType ? getPortColor(portType) : "var(--border)";

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
          strokeWidth: 2,
          stroke: strokeColor,
          ...style,
        }}
      />
      {isExecuting && (
        <circle r="3" fill={strokeColor}>
          <animateMotion dur="1s" repeatCount="indefinite" path={edgePath} />
        </circle>
      )}
    </>
  );
}

export const CustomEdge = memo(CustomEdgeComponent);
