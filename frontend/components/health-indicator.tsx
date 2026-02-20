"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "@/lib/api";
import { cn } from "@/lib/utils";

export function HealthIndicator() {
  const { data, isError, isLoading } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 30000,
    retry: 1,
  });

  const isHealthy = data?.status === "ok";

  return (
    <div className="flex items-center gap-2 text-sm text-muted-foreground">
      <span
        className={cn(
          "h-2 w-2 rounded-full",
          isLoading && "bg-yellow-500 animate-pulse",
          isHealthy && "bg-green-500",
          isError && "bg-red-500"
        )}
      />
      <span>
        {isLoading
          ? "Connecting..."
          : isHealthy
          ? `Backend v${data.version}`
          : "Backend offline"}
      </span>
    </div>
  );
}
