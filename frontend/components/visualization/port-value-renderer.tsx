"use client";

import type { TensorData } from "@/lib/types";
import { TensorHeatmap } from "./tensor-heatmap";
import { TensorStatsCard } from "./tensor-stats-card";
import { TensorDataTable } from "./tensor-data-table";
import { TokenDisplay } from "./token-display";
import { GradientOverlay } from "./gradient-overlay";
import { TensorShapeBadge } from "./tensor-shape-badge";
import { AttentionPattern } from "./attention-pattern";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export function PortValueRenderer({ data }: { data: TensorData }) {
  // TEXT type
  if (data.type === "TEXT") {
    return (
      <pre className="text-sm font-mono bg-muted p-3 rounded whitespace-pre-wrap">
        {String(data.value)}
      </pre>
    );
  }

  // TOKEN_IDS type
  if (data.type === "TOKEN_IDS") {
    const tokens = (data.value ?? data.data) as number[];
    return <TokenDisplay tokens={Array.isArray(tokens) ? tokens : []} />;
  }

  // SCALAR type
  if (data.type === "SCALAR") {
    return (
      <span className="text-lg font-mono font-semibold">
        {Number(data.value).toFixed(6)}
      </span>
    );
  }

  // INT type
  if (data.type === "INT") {
    return (
      <span className="text-lg font-mono font-semibold">
        {String(data.value)}
      </span>
    );
  }

  // TENSOR and AD_TENSOR types
  if (
    (data.type === "TENSOR" || data.type === "AD_TENSOR") &&
    data.shape &&
    data.data
  ) {
    const isSquare = data.shape[0] === data.shape[1];

    return (
      <div className="space-y-3 min-w-0 max-w-full">
        <div className="flex items-center gap-2 flex-wrap">
          <TensorShapeBadge shape={data.shape} />
          {data.truncated && (
            <span className="text-xs text-muted-foreground">
              (truncated to {data.data.length} values)
            </span>
          )}
        </div>

        {data.stats && (
          <TensorStatsCard stats={data.stats} shape={data.shape} />
        )}

        <Tabs defaultValue="heatmap">
          <TabsList>
            <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
            <TabsTrigger value="table">Table</TabsTrigger>
            {isSquare && (
              <TabsTrigger value="attention">Attention</TabsTrigger>
            )}
          </TabsList>
          <TabsContent value="heatmap" className="mt-2 overflow-x-auto">
            <TensorHeatmap
              data={data.data}
              shape={data.shape}
              width={Math.min(600, data.shape[1] * 20)}
              height={Math.min(400, data.shape[0] * 20)}
            />
          </TabsContent>
          <TabsContent value="table" className="mt-2 overflow-x-auto">
            <TensorDataTable data={data.data} shape={data.shape} />
          </TabsContent>
          {isSquare && (
            <TabsContent value="attention" className="mt-2 overflow-x-auto">
              <AttentionPattern
                data={data.data}
                shape={data.shape}
                stats={data.stats}
              />
            </TabsContent>
          )}
        </Tabs>

        {/* Gradient overlay for AD_TENSOR */}
        {data.type === "AD_TENSOR" && data.grad && data.grad.data && (
          <div className="border-t pt-3 mt-3 overflow-x-auto">
            <GradientOverlay
              data={data.grad.data}
              shape={data.grad.shape}
              stats={data.grad.stats}
            />
          </div>
        )}
      </div>
    );
  }

  // Fallback
  return (
    <pre className="text-xs font-mono bg-muted p-2 rounded overflow-x-auto max-h-[300px] overflow-y-auto">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}
