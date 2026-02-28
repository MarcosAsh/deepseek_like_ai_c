"use client";

import { useState } from "react";
import { TensorHeatmap } from "./tensor-heatmap";
import { TensorDataTable } from "./tensor-data-table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface NdimTensorViewerProps {
  data: number[];
  shape: number[];
}

export function NdimTensorViewer({ data, shape }: NdimTensorViewerProps) {
  // For N-dim tensors, we let the user slice through the leading dimensions
  // and display the last 2 dims as a 2D matrix
  const ndim = shape.length;

  // If 1D or 2D, just display directly
  if (ndim <= 2) {
    const rows = ndim === 1 ? 1 : shape[0];
    const cols = ndim === 1 ? shape[0] : shape[1];
    return (
      <Tabs defaultValue="heatmap">
        <TabsList>
          <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
          <TabsTrigger value="table">Table</TabsTrigger>
        </TabsList>
        <TabsContent value="heatmap" className="mt-2 overflow-x-auto">
          <TensorHeatmap
            data={data}
            shape={shape}
            width={Math.min(600, cols * 20)}
            height={Math.min(400, rows * 20)}
          />
        </TabsContent>
        <TabsContent value="table" className="mt-2 overflow-x-auto">
          <TensorDataTable data={data} shape={shape} />
        </TabsContent>
      </Tabs>
    );
  }

  // For >2D: leading dims are selectable, last 2 dims are the matrix
  const leadingDims = shape.slice(0, ndim - 2);
  const matRows = shape[ndim - 2];
  const matCols = shape[ndim - 1];
  const matSize = matRows * matCols;

  // Total number of 2D slices
  const totalSlices = leadingDims.reduce((a, b) => a * b, 1);

  const [sliceIndex, setSliceIndex] = useState(0);

  // Extract 2D slice from flat data
  const sliceOffset = sliceIndex * matSize;
  const sliceData = data.slice(sliceOffset, sliceOffset + matSize);
  const sliceShape: number[] = [matRows, matCols];

  // Convert flat slice index to multi-dim index for display
  function sliceLabel(idx: number): string {
    const indices: number[] = [];
    let remaining = idx;
    for (let i = leadingDims.length - 1; i >= 0; --i) {
      indices.unshift(remaining % leadingDims[i]);
      remaining = Math.floor(remaining / leadingDims[i]);
    }
    return indices.map((v, i) => `dim${i}=${v}`).join(", ");
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground">
        <Button
          variant="ghost"
          size="sm"
          className="h-6 w-6 p-0"
          disabled={sliceIndex === 0}
          onClick={() => setSliceIndex((i) => Math.max(0, i - 1))}
        >
          <ChevronLeft className="h-3 w-3" />
        </Button>
        <span>
          Slice {sliceIndex + 1}/{totalSlices} ({sliceLabel(sliceIndex)})
        </span>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 w-6 p-0"
          disabled={sliceIndex >= totalSlices - 1}
          onClick={() => setSliceIndex((i) => Math.min(totalSlices - 1, i + 1))}
        >
          <ChevronRight className="h-3 w-3" />
        </Button>
      </div>

      <Tabs defaultValue="heatmap">
        <TabsList>
          <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
          <TabsTrigger value="table">Table</TabsTrigger>
        </TabsList>
        <TabsContent value="heatmap" className="mt-2 overflow-x-auto">
          <TensorHeatmap
            data={sliceData}
            shape={sliceShape}
            width={Math.min(600, matCols * 20)}
            height={Math.min(400, matRows * 20)}
          />
        </TabsContent>
        <TabsContent value="table" className="mt-2 overflow-x-auto">
          <TensorDataTable data={sliceData} shape={sliceShape} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
