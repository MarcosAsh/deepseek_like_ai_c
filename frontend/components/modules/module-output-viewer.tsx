"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PORT_TYPE_BADGE_CLASSES } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { TensorData } from "@/lib/types";
import { PortValueRenderer } from "@/components/visualization/port-value-renderer";

export function ModuleOutputViewer({
  outputs,
}: {
  outputs: Record<string, TensorData>;
}) {
  const entries = Object.entries(outputs);

  if (entries.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No outputs to display.</p>
    );
  }

  return (
    <div className="space-y-3">
      {entries.map(([name, data]) => (
        <Card key={name}>
          <CardHeader className="py-3 px-4">
            <div className="flex items-center gap-2">
              <CardTitle className="text-sm font-mono">{name}</CardTitle>
              <Badge
                variant="secondary"
                className={cn(
                  "text-xs",
                  PORT_TYPE_BADGE_CLASSES[data.type]
                )}
              >
                {data.type}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="px-4 pb-3 pt-0">
            <PortValueRenderer data={data} />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
