"use client";

import { memo } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { Badge } from "@/components/ui/badge";
import { CATEGORY_COLORS } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { getPortColor } from "@/lib/port-colors";
import { NodeOutputPreview } from "./node-output-preview";
import { useGraphStore } from "./hooks/use-graph-store";
import type { CustomNodeData } from "@/lib/graph-utils";
import type { PortType } from "@/lib/types";

function CustomNodeComponent({ id, data, selected }: NodeProps) {
  const nodeData = data as unknown as CustomNodeData;
  const results = useGraphStore((s) => s.results);
  const executingNodeId = useGraphStore((s) => s.executingNodeId);
  const result = results.get(id);
  const isExecuting = executingNodeId === id;

  return (
    <div
      className={cn(
        "bg-card border rounded-lg shadow-sm min-w-[240px] relative transition-shadow",
        selected && "ring-2 ring-primary",
        result?.error && "border-destructive",
        isExecuting && "ring-2 ring-blue-500 animate-pulse shadow-lg shadow-blue-500/20"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between gap-2 px-4 py-2.5 border-b bg-muted/50 rounded-t-lg min-w-0">
        <span className="text-sm font-semibold truncate min-w-0">
          {nodeData.moduleType}
        </span>
        <Badge
          variant="secondary"
          className={cn("text-[10px] px-1.5 py-0 shrink-0", CATEGORY_COLORS[nodeData.category])}
        >
          {nodeData.category}
        </Badge>
      </div>

      {/* Port labels - each port row has its handle inline */}
      <div className="px-4 py-3 flex justify-between gap-6">
        {/* Input ports */}
        <div className="space-y-2">
          {nodeData.inputs.map((port) => (
            <div key={port.name} className="flex items-center gap-1.5 relative h-5">
              <Handle
                type="target"
                position={Position.Left}
                id={port.name}
                className="!absolute !-left-[22px] !top-1/2 !-translate-y-1/2"
                style={{
                  width: 10,
                  height: 10,
                  backgroundColor: getPortColor(port.type as PortType),
                  border: "2px solid var(--background)",
                }}
                title={`${port.name} (${port.type})`}
              />
              <span className="text-[11px] font-mono text-muted-foreground">
                {port.name}
              </span>
            </div>
          ))}
        </div>

        {/* Output ports */}
        <div className="space-y-2 text-right">
          {nodeData.outputs.map((port) => (
            <div
              key={port.name}
              className="flex items-center gap-1.5 justify-end relative h-5"
            >
              <span className="text-[11px] font-mono text-muted-foreground">
                {port.name}
              </span>
              <Handle
                type="source"
                position={Position.Right}
                id={port.name}
                className="!absolute !-right-[22px] !top-1/2 !-translate-y-1/2"
                style={{
                  width: 10,
                  height: 10,
                  backgroundColor: getPortColor(port.type as PortType),
                  border: "2px solid var(--background)",
                }}
                title={`${port.name} (${port.type})`}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Rich output previews */}
      {result && !result.error && result.outputs && (
        <div className="px-3 py-1.5 border-t bg-muted/30">
          <NodeOutputPreview outputs={result.outputs} />
          <div className="text-[10px] text-muted-foreground font-mono mt-1">
            {result.execution_time_ms.toFixed(1)}ms
          </div>
        </div>
      )}

      {/* Error */}
      {result?.error && (
        <div className="px-3 py-1.5 border-t text-[10px] text-destructive font-mono truncate">
          {result.error}
        </div>
      )}
    </div>
  );
}

export const CustomNode = memo(CustomNodeComponent);
