"use client";

import { TensorHeatmap } from "./tensor-heatmap";
import { TensorStatsCard } from "./tensor-stats-card";
import type { TensorStats } from "@/lib/types";

interface GradientOverlayProps {
  data: number[];
  shape: [number, number];
  stats?: TensorStats;
}

export function GradientOverlay({
  data,
  shape,
  stats,
}: GradientOverlayProps) {
  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium text-muted-foreground">
        Gradient Magnitude
      </h4>
      <TensorHeatmap data={data} shape={shape} />
      {stats && <TensorStatsCard stats={stats} shape={shape} />}
    </div>
  );
}
