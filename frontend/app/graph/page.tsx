"use client";

import { Suspense, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { GraphEditor } from "@/components/graph/graph-editor";
import { useGraphStore } from "@/components/graph/hooks/use-graph-store";
import { fromGraphDef } from "@/lib/graph-utils";
import { fetchModules } from "@/lib/api";
import { Monitor } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";

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
        // Invalid shared URL, ignore
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

      {/* Desktop: show graph editor */}
      <div className="hidden md:block h-full">
        <GraphEditor />
      </div>

      {/* Mobile: show message */}
      <div className="md:hidden flex flex-col items-center justify-center h-full p-8 text-center gap-4">
        <Monitor className="h-16 w-16 text-muted-foreground" />
        <h2 className="text-xl font-bold">Desktop Required</h2>
        <p className="text-muted-foreground max-w-sm">
          The Graph Editor requires a larger screen for the best experience.
          Please open this page on a desktop or tablet.
        </p>
        <div className="flex gap-3">
          <Button asChild variant="outline">
            <Link href="/modules">Browse Modules</Link>
          </Button>
          <Button asChild variant="outline">
            <Link href="/docs">Read Docs</Link>
          </Button>
        </div>
      </div>
    </>
  );
}
