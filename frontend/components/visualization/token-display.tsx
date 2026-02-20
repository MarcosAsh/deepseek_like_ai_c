"use client";

import { Badge } from "@/components/ui/badge";

export function TokenDisplay({ tokens }: { tokens: number[] }) {
  return (
    <div className="flex flex-wrap gap-1">
      {tokens.map((token, i) => (
        <Badge
          key={i}
          variant="secondary"
          className="font-mono text-xs bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
        >
          {token}
        </Badge>
      ))}
    </div>
  );
}
