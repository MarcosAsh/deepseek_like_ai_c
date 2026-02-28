"use client";

import { TensorHeatmap } from "./tensor-heatmap";
import { TensorStatsCard } from "./tensor-stats-card";
import type { TensorStats } from "@/lib/types";

interface AttentionPatternProps {
  data: number[];
  shape: number[];
  stats?: TensorStats;
  headIndex?: number;
}

export function AttentionPattern({
  data,
  shape,
  stats,
  headIndex,
}: AttentionPatternProps) {
  return (
    <div className="space-y-3">
      {headIndex !== undefined && (
        <h4 className="text-sm font-medium">
          Attention Head {headIndex}
        </h4>
      )}
      <TensorHeatmap
        data={data}
        shape={shape}
        width={Math.min(400, shape[1] * 30)}
        height={Math.min(400, shape[0] * 30)}
      />
      {stats && <TensorStatsCard stats={stats} shape={shape} />}
    </div>
  );
}
