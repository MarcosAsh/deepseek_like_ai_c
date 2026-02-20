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
  Bug,
  FileText,
} from "lucide-react";
import { ExecutionHistory } from "./execution-history";
import { ExecutionDebugger } from "./execution-debugger";
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
    results,
    executionHistory,
  } = useGraphStore();
  const { execute } = useGraphExecution();
  const [historyOpen, setHistoryOpen] = useState(false);
  const [debuggerOpen, setDebuggerOpen] = useState(false);

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

  function handleExportLogs() {
    const lines: string[] = [];
    lines.push("=== LLMs Unlocked - Execution Log ===");
    lines.push(`Date: ${new Date().toISOString()}`);
    lines.push(`Nodes: ${nodes.length}, Edges: ${edges.length}`);
    lines.push("");

    // Current results
    if (results.size > 0) {
      lines.push("--- Current Execution Results ---");
      for (const [nodeId, result] of results.entries()) {
        const node = nodes.find((n) => n.id === nodeId);
        const moduleType = node?.data?.moduleType ?? "unknown";
        lines.push("");
        lines.push(`[${nodeId}] ${moduleType}`);
        lines.push(`  Time: ${result.execution_time_ms.toFixed(3)}ms`);
        if (result.error) {
          lines.push(`  ERROR: ${result.error}`);
        } else {
          for (const [portName, data] of Object.entries(result.outputs)) {
            const d = data as { type?: string; shape?: number[]; value?: unknown };
            if (d.shape) {
              lines.push(`  ${portName}: ${d.type} [${d.shape.join("x")}]`);
            } else if (d.value !== undefined) {
              const val = String(d.value);
              lines.push(`  ${portName}: ${d.type} = ${val.length > 100 ? val.slice(0, 100) + "..." : val}`);
            } else {
              lines.push(`  ${portName}: ${d.type}`);
            }
          }
        }
      }
      lines.push("");
    }

    // Execution history
    if (executionHistory.length > 0) {
      lines.push("--- Execution History ---");
      for (const snapshot of executionHistory) {
        const date = new Date(snapshot.timestamp);
        lines.push("");
        lines.push(`Run at ${date.toLocaleString()} - ${snapshot.totalTimeMs.toFixed(1)}ms total`);
        lines.push(`  Order: ${snapshot.executionOrder.join(" -> ")}`);
        for (const [nid, nr] of snapshot.nodeResults.entries()) {
          if (nr.error) {
            lines.push(`  [${nid}] ${nr.node_type}: ERROR - ${nr.error}`);
          } else {
            lines.push(`  [${nid}] ${nr.node_type}: ${nr.execution_time_ms.toFixed(2)}ms`);
          }
        }
      }
    }

    // Graph definition
    lines.push("");
    lines.push("--- Graph Definition ---");
    for (const n of nodes) {
      lines.push(`Node: ${n.id} (${n.data.moduleType}) config=${JSON.stringify(n.data.config)}`);
    }
    for (const e of edges) {
      lines.push(`Edge: ${e.source}:${e.sourceHandle} -> ${e.target}:${e.targetHandle}`);
    }

    const text = lines.join("\n");
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `execution-log-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Execution log downloaded");
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

      <Button
        size="sm"
        variant="outline"
        onClick={handleExportLogs}
        disabled={results.size === 0}
        title="Export execution logs as text file"
      >
        <FileText className="h-3 w-3 mr-1" />
        Logs
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
        onClick={() => setDebuggerOpen(true)}
        title="Step-by-step Debugger"
      >
        <Bug className="h-3.5 w-3.5" />
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
      <ExecutionDebugger open={debuggerOpen} onOpenChange={setDebuggerOpen} />
    </div>
  );
}
