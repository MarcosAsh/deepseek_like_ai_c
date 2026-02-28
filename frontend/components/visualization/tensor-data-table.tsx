"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronUp } from "lucide-react";

interface TensorDataTableProps {
  data: number[];
  shape: number[];
  maxRows?: number;
}

export function TensorDataTable({
  data,
  shape,
  maxRows = 10,
}: TensorDataTableProps) {
  const [expanded, setExpanded] = useState(false);
  const cols = shape[shape.length - 1] ?? 1;
  const rows = shape.length <= 1 ? data.length : data.length / cols;
  const displayRows = expanded ? rows : Math.min(rows, maxRows);

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto border rounded">
        <table className="text-xs font-mono">
          <thead>
            <tr className="bg-muted/50">
              <th className="px-2 py-1 text-left text-muted-foreground">#</th>
              {Array.from({ length: Math.min(cols, 20) }).map((_, c) => (
                <th
                  key={c}
                  className="px-2 py-1 text-right text-muted-foreground"
                >
                  {c}
                </th>
              ))}
              {cols > 20 && (
                <th className="px-2 py-1 text-muted-foreground">...</th>
              )}
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: displayRows }).map((_, r) => (
              <tr key={r} className="border-t">
                <td className="px-2 py-0.5 text-muted-foreground">{r}</td>
                {Array.from({ length: Math.min(cols, 20) }).map((_, c) => (
                  <td key={c} className="px-2 py-0.5 text-right tabular-nums">
                    {data[r * cols + c]?.toFixed(4) ?? "-"}
                  </td>
                ))}
                {cols > 20 && (
                  <td className="px-2 py-0.5 text-muted-foreground">...</td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows > maxRows && (
        <Button
          variant="ghost"
          size="sm"
          className="text-xs"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? (
            <>
              <ChevronUp className="h-3 w-3 mr-1" />
              Show less
            </>
          ) : (
            <>
              <ChevronDown className="h-3 w-3 mr-1" />
              Show all {rows} rows
            </>
          )}
        </Button>
      )}
    </div>
  );
}
