"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useGraphStore } from "./hooks/use-graph-store";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Trash2 } from "lucide-react";

interface ExecutionHistoryProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ExecutionHistory({ open, onOpenChange }: ExecutionHistoryProps) {
  const { executionHistory, clearExecutionHistory, setResults } =
    useGraphStore();

  function handleRestore(index: number) {
    const snapshot = executionHistory[index];
    if (snapshot) {
      setResults(snapshot.nodeResults);
      onOpenChange(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-between">
            Execution History
            {executionHistory.length > 0 && (
              <Button
                size="sm"
                variant="ghost"
                className="text-destructive"
                onClick={clearExecutionHistory}
              >
                <Trash2 className="h-3 w-3 mr-1" />
                Clear
              </Button>
            )}
          </DialogTitle>
        </DialogHeader>
        <ScrollArea className="max-h-[400px]">
          {executionHistory.length === 0 ? (
            <p className="text-sm text-muted-foreground py-8 text-center">
              No executions yet. Run a graph to see history.
            </p>
          ) : (
            <div className="space-y-2">
              {executionHistory.map((snapshot, i) => {
                const errorCount = Array.from(
                  snapshot.nodeResults.values()
                ).filter((r) => r.error).length;
                const date = new Date(snapshot.timestamp);
                return (
                  <button
                    key={snapshot.id}
                    onClick={() => handleRestore(i)}
                    className="w-full text-left p-3 rounded-lg border hover:bg-accent/50 transition-colors"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium">
                        Run #{executionHistory.length - i}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {date.toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary" className="text-[10px]">
                        {snapshot.nodeResults.size} nodes
                      </Badge>
                      <Badge variant="secondary" className="text-[10px]">
                        {snapshot.totalTimeMs.toFixed(1)}ms
                      </Badge>
                      {errorCount > 0 && (
                        <Badge
                          variant="destructive"
                          className="text-[10px]"
                        >
                          {errorCount} errors
                        </Badge>
                      )}
                    </div>
                    <div className="text-[10px] text-muted-foreground mt-1 font-mono">
                      Order: {snapshot.executionOrder.join(" -> ")}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
