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
  ShieldCheck,
} from "lucide-react";
import { ExecutionHistory } from "./execution-history";
import { ExecutionDebugger } from "./execution-debugger";
import { ValidationPanel } from "./validation-panel";
import { useGraphValidation } from "./hooks/use-graph-validation";
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
    validationResult,
    setValidationResult,
  } = useGraphStore();
  const { execute } = useGraphExecution();
  const { validate } = useGraphValidation();
  const [historyOpen, setHistoryOpen] = useState(false);
  const [debuggerOpen, setDebuggerOpen] = useState(false);

  function handleValidate() {
    const result = validate();
    setValidationResult(result);
    if (result.valid && result.warnings.length === 0) {
      toast.success("Graph is valid");
    } else if (result.valid) {
      toast.warning(`Graph has ${result.warnings.length} warning(s)`);
    } else {
      toast.error(`Graph has ${result.errors.length} error(s)`);
    }
  }

  function handleExecuteWithValidation() {
    const result = validate();
    setValidationResult(result);
    if (!result.valid) {
      toast.error(`Cannot execute: ${result.errors.length} validation error(s)`);
      return;
    }
    execute();
  }

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

  function formatTensorData(shape: number[], data: number[]): string {
    if (shape.length === 2) {
      const [rows, cols] = shape;
      const lines: string[] = [];
      for (let r = 0; r < rows; r++) {
        const row = data.slice(r * cols, (r + 1) * cols);
        lines.push("    [" + row.map((v) => v.toFixed(4).padStart(9)).join(", ") + "]");
      }
      return "  [\n" + lines.join(",\n") + "\n  ]";
    }
    return "  [" + data.map((v) => v.toFixed(4)).join(", ") + "]";
  }

  function formatPortOutput(portName: string, data: Record<string, unknown>): string[] {
    const lines: string[] = [];
    const d = data as {
      type?: string;
      shape?: number[];
      data?: number[];
      value?: unknown;
      stats?: { min: number; max: number; mean: number; std: number };
      truncated?: boolean;
      grad?: { shape: number[]; data: number[]; stats: { min: number; max: number; mean: number; std: number } };
    };
    const type = d.type ?? "unknown";

    if (type === "TENSOR" || type === "AD_TENSOR") {
      const shapeStr = d.shape ? `[${d.shape.join("x")}]` : "[]";
      lines.push(`  ${portName}: ${type} ${shapeStr}`);
      if (d.stats) {
        lines.push(`    stats: min=${d.stats.min.toFixed(4)}, max=${d.stats.max.toFixed(4)}, mean=${d.stats.mean.toFixed(4)}, std=${d.stats.std.toFixed(4)}`);
      }
      if (d.data && d.shape) {
        lines.push(`    data:`);
        lines.push(formatTensorData(d.shape, d.data));
      }
      if (d.truncated) {
        lines.push(`    (data truncated)`);
      }
      if (d.grad && d.grad.data) {
        lines.push(`    gradient: [${d.grad.shape.join("x")}]`);
        if (d.grad.stats) {
          lines.push(`    grad stats: min=${d.grad.stats.min.toFixed(4)}, max=${d.grad.stats.max.toFixed(4)}, mean=${d.grad.stats.mean.toFixed(4)}, std=${d.grad.stats.std.toFixed(4)}`);
        }
        lines.push(`    grad data:`);
        lines.push(formatTensorData(d.grad.shape, d.grad.data));
      }
    } else if (type === "TOKEN_IDS") {
      const tokens = Array.isArray(d.value) ? d.value : [];
      lines.push(`  ${portName}: ${type} (${tokens.length} tokens)`);
      lines.push(`    [${tokens.join(", ")}]`);
    } else if (type === "TEXT") {
      lines.push(`  ${portName}: ${type}`);
      lines.push(`    "${String(d.value ?? "")}"`);
    } else if (type === "SCALAR") {
      lines.push(`  ${portName}: ${type} = ${Number(d.value).toFixed(6)}`);
    } else if (type === "INT") {
      lines.push(`  ${portName}: ${type} = ${d.value}`);
    } else {
      lines.push(`  ${portName}: ${type} = ${JSON.stringify(d.value)}`);
    }
    return lines;
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
            lines.push(...formatPortOutput(portName, data as unknown as Record<string, unknown>));
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
            for (const [portName, data] of Object.entries(nr.outputs)) {
              lines.push(...formatPortOutput(portName, data as unknown as Record<string, unknown>));
            }
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
      <Button size="sm" onClick={handleExecuteWithValidation} disabled={isExecuting}>
        {isExecuting ? (
          <Loader2 className="h-4 w-4 mr-1 animate-spin" />
        ) : (
          <Play className="h-4 w-4 mr-1" />
        )}
        Run
      </Button>
      <Button size="sm" variant="outline" onClick={handleValidate} title="Validate graph">
        <ShieldCheck className="h-3.5 w-3.5 mr-1" />
        Validate
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

      <ValidationPanel result={validationResult} />

      <ExecutionHistory open={historyOpen} onOpenChange={setHistoryOpen} />
      <ExecutionDebugger open={debuggerOpen} onOpenChange={setDebuggerOpen} />
    </div>
  );
}
