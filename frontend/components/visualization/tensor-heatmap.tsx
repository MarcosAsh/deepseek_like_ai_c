"use client";

import { useRef, useEffect } from "react";
import * as d3 from "d3";

interface TensorHeatmapProps {
  data: number[];
  shape: [number, number];
  width?: number;
  height?: number;
}

export function TensorHeatmap({
  data,
  shape,
  width = 500,
  height = 300,
}: TensorHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const [rows, cols] = shape;
    const cellW = width / cols;
    const cellH = height / rows;

    // Choose color scale based on data range
    const min = d3.min(data) ?? 0;
    const max = d3.max(data) ?? 1;
    const hasNegative = min < 0;

    const colorScale = hasNegative
      ? d3.scaleDiverging(d3.interpolateRdBu).domain([min, 0, max])
      : d3.scaleSequential(d3.interpolateViridis).domain([min, max]);

    // Draw heatmap
    ctx.clearRect(0, 0, width, height);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = data[r * cols + c];
        if (val === undefined) continue;
        ctx.fillStyle = colorScale(val) as string;
        ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
      }
    }

    // Draw cell values if small enough
    if (rows * cols <= 64 && cellW > 30 && cellH > 15) {
      ctx.fillStyle = "currentColor";
      ctx.font = `${Math.min(10, cellH * 0.6)}px monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const val = data[r * cols + c];
          if (val === undefined) continue;
          // Choose text color based on luminance
          const color = d3.color(colorScale(val) as string);
          const luminance = color
            ? 0.299 * (color as d3.RGBColor).r +
              0.587 * (color as d3.RGBColor).g +
              0.114 * (color as d3.RGBColor).b
            : 128;
          ctx.fillStyle = luminance > 128 ? "#000" : "#fff";
          ctx.fillText(
            val.toFixed(2),
            c * cellW + cellW / 2,
            r * cellH + cellH / 2
          );
        }
      }
    }
  }, [data, shape, width, height]);

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    const tooltip = tooltipRef.current;
    if (!canvas || !tooltip) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const [rows, cols] = shape;
    const col = Math.floor((x / width) * cols);
    const row = Math.floor((y / height) * rows);

    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      const val = data[row * cols + col];
      tooltip.style.display = "block";
      tooltip.style.left = `${x + 10}px`;
      tooltip.style.top = `${y - 25}px`;
      tooltip.textContent = `[${row},${col}] = ${val?.toFixed(6) ?? "N/A"}`;
    } else {
      tooltip.style.display = "none";
    }
  }

  return (
    <div className="relative inline-block">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="rounded border"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => {
          if (tooltipRef.current) tooltipRef.current.style.display = "none";
        }}
      />
      <div
        ref={tooltipRef}
        className="absolute hidden bg-popover text-popover-foreground px-2 py-1 rounded text-xs font-mono shadow-md border pointer-events-none"
      />
    </div>
  );
}
