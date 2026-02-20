"use client";

import type { TensorData } from "@/lib/types";
import { Badge } from "@/components/ui/badge";

interface NodeOutputPreviewProps {
  outputs: Record<string, TensorData>;
}

function OutputPreviewItem({
  name,
  data,
}: {
  name: string;
  data: TensorData;
}) {
  let preview: React.ReactNode;

  switch (data.type) {
    case "TENSOR":
    case "AD_TENSOR":
      preview = data.shape ? (
        <Badge variant="outline" className="text-[9px] font-mono px-1 py-0">
          [{data.shape.join(" x ")}]
        </Badge>
      ) : (
        <span className="text-[9px] text-muted-foreground">tensor</span>
      );
      break;
    case "TOKEN_IDS": {
      const tokens = (data.value ?? data.data) as number[] | undefined;
      preview = (
        <span className="text-[9px] text-green-500 font-mono">
          {tokens ? `${tokens.length} tokens` : "tokens"}
        </span>
      );
      break;
    }
    case "TEXT":
      preview = (
        <span className="text-[9px] text-blue-400 font-mono truncate max-w-[100px] inline-block">
          {String(data.value).slice(0, 20)}
          {String(data.value).length > 20 ? "..." : ""}
        </span>
      );
      break;
    case "SCALAR":
      preview = (
        <span className="text-[9px] text-yellow-500 font-mono">
          {Number(data.value).toFixed(4)}
        </span>
      );
      break;
    case "INT":
      preview = (
        <span className="text-[9px] text-gray-400 font-mono">
          {String(data.value)}
        </span>
      );
      break;
    default:
      preview = (
        <span className="text-[9px] text-muted-foreground">output</span>
      );
  }

  return (
    <div className="flex items-center justify-between gap-1 px-1">
      <span className="text-[9px] text-muted-foreground truncate">
        {name}
      </span>
      {preview}
    </div>
  );
}

export function NodeOutputPreview({ outputs }: NodeOutputPreviewProps) {
  const entries = Object.entries(outputs);
  if (entries.length === 0) return null;

  return (
    <div className="space-y-0.5">
      {entries.map(([name, data]) => (
        <OutputPreviewItem key={name} name={name} data={data} />
      ))}
    </div>
  );
}
