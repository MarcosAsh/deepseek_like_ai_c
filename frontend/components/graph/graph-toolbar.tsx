"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { useGraphStore } from "./hooks/use-graph-store";
import { useGraphExecution } from "./hooks/use-graph-execution";
import {
  Play,
  Trash2,
  Loader2,
  Download,
  Undo2,
  Redo2,
  History,
  Share2,
} from "lucide-react";
import { ExecutionHistory } from "./execution-history";
import { toast } from "sonner";

interface GraphToolbarProps {
  onLoadPreset: () => void;
}

export function GraphToolbar({ onLoadPreset }: GraphToolbarProps) {
  const {
    isExecuting,
    clearAll,
    clearResults,
    nodes,
    edges,
    undo,
    redo,
    history,
    future,
  } = useGraphStore();
  const { execute } = useGraphExecution();
  const [historyOpen, setHistoryOpen] = useState(false);

  function handleExport() {
    const data = JSON.stringify(
      {
        nodes: nodes.map((n) => ({
          id: n.id,
          type: n.data.moduleType,
          config: n.data.config,
          position: n.position,
        })),
        edges: edges.map((e) => ({
          source: e.source,
          sourceHandle: e.sourceHandle,
          target: e.target,
          targetHandle: e.targetHandle,
        })),
      },
      null,
      2
    );
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "graph.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleShare() {
    const graphData = {
      nodes: nodes.map((n) => ({
        id: n.id,
        type: n.data.moduleType,
        config: n.data.config,
        position: n.position,
      })),
      edges: edges.map((e) => ({
        source: e.source,
        sourceHandle: e.sourceHandle,
        target: e.target,
        targetHandle: e.targetHandle,
      })),
    };
    const encoded = btoa(JSON.stringify(graphData));
    const url = `${window.location.origin}/graph?g=${encoded}`;
    navigator.clipboard.writeText(url).then(() => {
      toast.success("Share link copied to clipboard");
    });
  }

  return (
    <div className="flex items-center gap-2 px-3 py-2 border-b bg-muted/30 flex-wrap">
      <Button size="sm" onClick={execute} disabled={isExecuting}>
        {isExecuting ? (
          <Loader2 className="h-4 w-4 mr-1 animate-spin" />
        ) : (
          <Play className="h-4 w-4 mr-1" />
        )}
        Run
      </Button>
      <Button size="sm" variant="outline" onClick={onLoadPreset}>
        Load Preset
      </Button>
      <Button size="sm" variant="outline" onClick={handleExport}>
        <Download className="h-3 w-3 mr-1" />
        Export
      </Button>
      <Button size="sm" variant="outline" onClick={handleShare} disabled={nodes.length === 0}>
        <Share2 className="h-3 w-3 mr-1" />
        Share
      </Button>

      <div className="w-px h-5 bg-border mx-1" />

      <Button
        size="sm"
        variant="ghost"
        onClick={undo}
        disabled={history.length === 0}
        title="Undo (Ctrl+Z)"
      >
        <Undo2 className="h-3.5 w-3.5" />
      </Button>
      <Button
        size="sm"
        variant="ghost"
        onClick={redo}
        disabled={future.length === 0}
        title="Redo (Ctrl+Shift+Z)"
      >
        <Redo2 className="h-3.5 w-3.5" />
      </Button>
      <Button
        size="sm"
        variant="ghost"
        onClick={() => setHistoryOpen(true)}
        title="Execution History"
      >
        <History className="h-3.5 w-3.5" />
      </Button>

      <div className="flex-1" />
      <Button size="sm" variant="ghost" onClick={clearResults}>
        Clear Results
      </Button>
      <Button
        size="sm"
        variant="ghost"
        className="text-destructive"
        onClick={clearAll}
      >
        <Trash2 className="h-3 w-3 mr-1" />
        Clear All
      </Button>

      <ExecutionHistory open={historyOpen} onOpenChange={setHistoryOpen} />
    </div>
  );
}
