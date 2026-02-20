"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface CompilationOutputProps {
  compilationOutput?: string;
  executionOutput?: string;
  error?: string | null;
}

export function CompilationOutput({
  compilationOutput,
  executionOutput,
  error,
}: CompilationOutputProps) {
  if (!compilationOutput && !executionOutput && !error) return null;

  return (
    <Card className={cn(error && "border-destructive")}>
      <CardHeader className="py-2 px-4">
        <CardTitle className="text-sm">
          {error ? "Error" : "Output"}
        </CardTitle>
      </CardHeader>
      <CardContent className="px-4 pb-3 space-y-2">
        {error && (
          <pre className="text-xs font-mono text-destructive whitespace-pre-wrap">
            {error}
          </pre>
        )}
        {compilationOutput && (
          <div>
            <p className="text-xs text-muted-foreground font-medium mb-1">
              Compilation:
            </p>
            <pre className="text-xs font-mono bg-muted p-2 rounded whitespace-pre-wrap max-h-40 overflow-y-auto">
              {compilationOutput}
            </pre>
          </div>
        )}
        {executionOutput && (
          <div>
            <p className="text-xs text-muted-foreground font-medium mb-1">
              Execution:
            </p>
            <pre className="text-xs font-mono bg-muted p-2 rounded whitespace-pre-wrap max-h-40 overflow-y-auto">
              {executionOutput}
            </pre>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
