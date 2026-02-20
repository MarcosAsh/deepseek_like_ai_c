"use client";

import { Button } from "@/components/ui/button";
import { RotateCcw, Play, Loader2 } from "lucide-react";

interface CodeEditorToolbarProps {
  onReset: () => void;
  onRun: () => void;
  isRunning: boolean;
  isModified: boolean;
  canRun: boolean;
}

export function CodeEditorToolbar({
  onReset,
  onRun,
  isRunning,
  isModified,
  canRun,
}: CodeEditorToolbarProps) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 border-b bg-muted/30">
      <Button
        size="sm"
        variant="outline"
        onClick={onReset}
        disabled={!isModified}
      >
        <RotateCcw className="h-3 w-3 mr-1" />
        Reset to Original
      </Button>
      <Button size="sm" onClick={onRun} disabled={isRunning || !canRun}>
        {isRunning ? (
          <Loader2 className="h-3 w-3 mr-1 animate-spin" />
        ) : (
          <Play className="h-3 w-3 mr-1" />
        )}
        Compile & Run
      </Button>
      {!canRun && (
        <span className="text-xs text-muted-foreground ml-2">
          Backend compile endpoint not available yet
        </span>
      )}
      {isModified && (
        <span className="text-xs text-yellow-600 dark:text-yellow-400 ml-auto">
          Modified
        </span>
      )}
    </div>
  );
}
