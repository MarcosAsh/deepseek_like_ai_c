"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { GraphEditor } from "@/components/graph/graph-editor";
import { useGraphStore } from "@/components/graph/hooks/use-graph-store";
import { fromGraphDef } from "@/lib/graph-utils";
import { fetchModules } from "@/lib/api";
import { toast } from "sonner";

function GraphLoader() {
  const searchParams = useSearchParams();
  const { setNodes, setEdges } = useGraphStore();

  useEffect(() => {
    const graphParam = searchParams.get("g");
    if (!graphParam) return;

    async function loadShared() {
      try {
        const decoded = JSON.parse(atob(graphParam!));
        const modulesResp = await fetchModules();
        const { nodes, edges } = fromGraphDef(
          {
            nodes: decoded.nodes.map((n: { id: string; type: string; config: Record<string, unknown> }) => ({
              id: n.id,
              type: n.type,
              config: n.config || {},
            })),
            edges: decoded.edges.map((e: { source: string; sourceHandle: string; target: string; targetHandle: string }) => ({
              source_node: e.source,
              source_port: e.sourceHandle,
              target_node: e.target,
              target_port: e.targetHandle,
            })),
          },
          modulesResp.modules
        );
        setNodes(nodes);
        setEdges(edges);
      } catch {
        toast.error("Failed to load shared graph. The link may be invalid.");
      }
    }
    loadShared();
  }, [searchParams, setNodes, setEdges]);

  return null;
}

export default function GraphPage() {
  return (
    <>
      <Suspense fallback={null}>
        <GraphLoader />
      </Suspense>

      <div className="h-full">
        <GraphEditor />
      </div>
    </>
  );
}
