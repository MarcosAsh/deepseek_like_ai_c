"use client";

import { useState, useCallback, useMemo } from "react";
import { useGraphStore } from "./hooks/use-graph-store";
import { PortValueRenderer } from "@/components/visualization/port-value-renderer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  ChevronLeft,
  ChevronRight,
  SkipBack,
  SkipForward,
} from "lucide-react";
import type { TensorData, NodeResult } from "@/lib/types";
import type { CustomNodeData } from "@/lib/graph-utils";

interface ExecutionDebuggerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ExecutionDebugger({
  open,
  onOpenChange,
}: ExecutionDebuggerProps) {
  const { executionHistory, nodes, setExecutingNodeId } = useGraphStore();
  const latestExecution = executionHistory[0];
  const [currentStep, setCurrentStep] = useState(0);

  const executionOrder = latestExecution?.executionOrder ?? [];
  const totalSteps = executionOrder.length;

  const currentNodeId = executionOrder[currentStep];
  const currentResult = latestExecution?.nodeResults.get(currentNodeId ?? "");
  const currentNode = nodes.find((n) => n.id === currentNodeId);
  const currentData = currentNode?.data as CustomNodeData | undefined;

  // Build cumulative results up to current step
  const cumulativeResults = useMemo(() => {
    if (!latestExecution) return [];
    return executionOrder.slice(0, currentStep + 1).map((nodeId) => {
      const result = latestExecution.nodeResults.get(nodeId);
      const node = nodes.find((n) => n.id === nodeId);
      const data = node?.data as CustomNodeData | undefined;
      return { nodeId, result, data };
    });
  }, [latestExecution, executionOrder, currentStep, nodes]);

  const goToStep = useCallback(
    (step: number) => {
      const clamped = Math.max(0, Math.min(step, totalSteps - 1));
      setCurrentStep(clamped);
      const nodeId = executionOrder[clamped];
      if (nodeId) setExecutingNodeId(nodeId);
    },
    [totalSteps, executionOrder, setExecutingNodeId]
  );

  const handleClose = useCallback(() => {
    setExecutingNodeId(null);
    onOpenChange(false);
  }, [onOpenChange, setExecutingNodeId]);

  if (!latestExecution || totalSteps === 0) {
    return (
      <Dialog open={open} onOpenChange={handleClose}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Execution Debugger</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground py-8 text-center">
            No execution data available. Run a graph first.
          </p>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="w-[95vw] max-w-3xl max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            Execution Debugger
            <Badge variant="secondary" className="text-xs">
              Step {currentStep + 1} / {totalSteps}
            </Badge>
            <Badge variant="outline" className="text-xs">
              {latestExecution.totalTimeMs.toFixed(1)}ms total
            </Badge>
          </DialogTitle>
        </DialogHeader>

        {/* Step navigation */}
        <div className="flex items-center gap-2 py-2 border-b">
          <Button
            size="sm"
            variant="outline"
            onClick={() => goToStep(0)}
            disabled={currentStep === 0}
          >
            <SkipBack className="h-3 w-3" />
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => goToStep(currentStep - 1)}
            disabled={currentStep === 0}
          >
            <ChevronLeft className="h-3 w-3" />
          </Button>

          {/* Step dots */}
          <div className="flex items-center gap-1 flex-1 overflow-x-auto px-2 min-w-0">
            {executionOrder.map((nodeId, i) => {
              const node = nodes.find((n) => n.id === nodeId);
              const data = node?.data as CustomNodeData | undefined;
              const result = latestExecution.nodeResults.get(nodeId);
              const hasError = !!result?.error;
              return (
                <button
                  key={nodeId}
                  onClick={() => goToStep(i)}
                  className={`shrink-0 px-2 py-1 rounded text-[10px] font-mono transition-colors ${
                    i === currentStep
                      ? "bg-primary text-primary-foreground"
                      : i <= currentStep
                      ? hasError
                        ? "bg-destructive/20 text-destructive"
                        : "bg-accent text-accent-foreground"
                      : "bg-muted text-muted-foreground"
                  }`}
                  title={`${data?.moduleType ?? nodeId} (${result?.execution_time_ms.toFixed(1)}ms)`}
                >
                  {data?.moduleType ?? nodeId}
                </button>
              );
            })}
          </div>

          <Button
            size="sm"
            variant="outline"
            onClick={() => goToStep(currentStep + 1)}
            disabled={currentStep >= totalSteps - 1}
          >
            <ChevronRight className="h-3 w-3" />
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => goToStep(totalSteps - 1)}
            disabled={currentStep >= totalSteps - 1}
          >
            <SkipForward className="h-3 w-3" />
          </Button>
        </div>

        {/* Current step detail */}
        <ScrollArea className="flex-1 min-h-0">
          <div className="space-y-4 pr-4">
            {/* Current node info */}
            {currentData && currentResult && (
              <Card>
                <CardHeader className="py-2 px-4">
                  <CardTitle className="text-sm flex items-center gap-2">
                    {currentData.moduleType}
                    <Badge variant="secondary" className="text-[10px]">
                      {currentData.category}
                    </Badge>
                    <Badge variant="outline" className="text-[10px]">
                      {currentResult.execution_time_ms.toFixed(1)}ms
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="px-4 pb-4 overflow-hidden">
                  {currentResult.error ? (
                    <p className="text-sm text-destructive font-mono break-words">
                      {currentResult.error}
                    </p>
                  ) : (
                    <div className="space-y-3 overflow-x-auto">
                      <p className="text-xs text-muted-foreground font-medium">
                        Outputs:
                      </p>
                      {Object.entries(currentResult.outputs).map(
                        ([name, tensorData]) => (
                          <div key={name} className="space-y-1 min-w-0">
                            <p className="text-xs font-mono font-medium">
                              {name}
                            </p>
                            <div className="overflow-x-auto max-w-full">
                              <PortValueRenderer
                                data={tensorData as TensorData}
                              />
                            </div>
                          </div>
                        )
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Data flow timeline */}
            <div>
              <h4 className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">
                Data Flow (Steps 1-{currentStep + 1})
              </h4>
              <div className="space-y-2">
                {cumulativeResults.map(({ nodeId, result, data }, i) => (
                  <button
                    key={nodeId}
                    onClick={() => goToStep(i)}
                    className={`w-full text-left p-2 rounded border transition-colors ${
                      i === currentStep
                        ? "border-primary bg-primary/5"
                        : "border-transparent hover:bg-accent/50"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-muted-foreground w-4">
                        {i + 1}
                      </span>
                      <span className="text-xs font-medium">
                        {data?.moduleType ?? nodeId}
                      </span>
                      {result && !result.error && (
                        <div className="flex items-center gap-1 ml-auto">
                          {Object.entries(result.outputs).map(
                            ([name, td]) => {
                              const t = td as TensorData;
                              return (
                                <Badge
                                  key={name}
                                  variant="outline"
                                  className="text-[9px] px-1"
                                >
                                  {t.shape
                                    ? `[${t.shape.join("x")}]`
                                    : t.type}
                                </Badge>
                              );
                            }
                          )}
                          <span className="text-[10px] text-muted-foreground">
                            {result.execution_time_ms.toFixed(1)}ms
                          </span>
                        </div>
                      )}
                      {result?.error && (
                        <Badge
                          variant="destructive"
                          className="text-[9px] ml-auto"
                        >
                          error
                        </Badge>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
