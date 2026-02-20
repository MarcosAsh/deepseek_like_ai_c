"use client";

import { Card, CardContent } from "@/components/ui/card";
import type { TensorStats } from "@/lib/types";

export function TensorStatsCard({
  stats,
  shape,
}: {
  stats: TensorStats;
  shape: [number, number];
}) {
  return (
    <Card>
      <CardContent className="py-3 px-4">
        <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm">
          <div className="font-mono">
            <span className="text-muted-foreground text-xs">Shape:</span>{" "}
            <span className="font-semibold">
              [{shape[0]} x {shape[1]}]
            </span>
          </div>
          <div className="font-mono">
            <span className="text-muted-foreground text-xs">Elements:</span>{" "}
            <span>{shape[0] * shape[1]}</span>
          </div>
          <div className="font-mono">
            <span className="text-muted-foreground text-xs">Min:</span>{" "}
            <span>{stats.min.toFixed(6)}</span>
          </div>
          <div className="font-mono">
            <span className="text-muted-foreground text-xs">Max:</span>{" "}
            <span>{stats.max.toFixed(6)}</span>
          </div>
          <div className="font-mono">
            <span className="text-muted-foreground text-xs">Mean:</span>{" "}
            <span>{stats.mean.toFixed(6)}</span>
          </div>
          <div className="font-mono">
            <span className="text-muted-foreground text-xs">Std:</span>{" "}
            <span>{stats.std.toFixed(6)}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
