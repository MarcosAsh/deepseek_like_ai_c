"use client";

import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useGraphStore } from "./hooks/use-graph-store";
import type { ValidationResult, ValidationIssue } from "./hooks/use-graph-validation";
import { AlertTriangle, XCircle, CheckCircle2 } from "lucide-react";

interface ValidationPanelProps {
  result: ValidationResult | null;
}

function IssueItem({ issue, onSelect }: { issue: ValidationIssue; onSelect: (id: string) => void }) {
  const Icon = issue.type === "error" ? XCircle : AlertTriangle;
  const colorClass = issue.type === "error" ? "text-red-500" : "text-yellow-500";

  return (
    <button
      className="flex items-start gap-2 text-left w-full px-2 py-1.5 rounded hover:bg-muted/50 text-xs"
      onClick={() => issue.nodeId && onSelect(issue.nodeId)}
      disabled={!issue.nodeId}
    >
      <Icon className={`h-3.5 w-3.5 shrink-0 mt-0.5 ${colorClass}`} />
      <span className="text-muted-foreground">{issue.message}</span>
    </button>
  );
}

export function ValidationPanel({ result }: ValidationPanelProps) {
  const { selectNode } = useGraphStore();

  if (!result) return null;

  const totalErrors = result.errors.length;
  const totalWarnings = result.warnings.length;

  if (totalErrors === 0 && totalWarnings === 0) {
    return (
      <div className="flex items-center gap-1.5 text-xs text-emerald-600">
        <CheckCircle2 className="h-3.5 w-3.5" />
        Valid
      </div>
    );
  }

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          size="sm"
          variant="ghost"
          className={totalErrors > 0 ? "text-red-500" : "text-yellow-500"}
        >
          {totalErrors > 0 ? (
            <XCircle className="h-3.5 w-3.5 mr-1" />
          ) : (
            <AlertTriangle className="h-3.5 w-3.5 mr-1" />
          )}
          {totalErrors > 0 && `${totalErrors} error${totalErrors > 1 ? "s" : ""}`}
          {totalErrors > 0 && totalWarnings > 0 && ", "}
          {totalWarnings > 0 && `${totalWarnings} warning${totalWarnings > 1 ? "s" : ""}`}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80 p-2" align="start">
        <div className="space-y-1 max-h-64 overflow-y-auto">
          {result.errors.map((issue, i) => (
            <IssueItem key={`err-${i}`} issue={issue} onSelect={selectNode} />
          ))}
          {result.warnings.map((issue, i) => (
            <IssueItem key={`warn-${i}`} issue={issue} onSelect={selectNode} />
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}
