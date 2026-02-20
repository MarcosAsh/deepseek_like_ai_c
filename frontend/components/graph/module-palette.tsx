"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchModules } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { CATEGORY_COLORS, CATEGORY_LABELS } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";
import { GripVertical } from "lucide-react";
import type { ModuleCatalogEntry } from "@/lib/types";

interface ModulePaletteProps {
  onAddModule: (module: ModuleCatalogEntry) => void;
}

export function ModulePalette({ onAddModule }: ModulePaletteProps) {
  const { data, isLoading } = useQuery({
    queryKey: ["modules"],
    queryFn: fetchModules,
  });

  const modules = data?.modules ?? [];

  // Group by category
  const grouped = modules.reduce<Record<string, ModuleCatalogEntry[]>>(
    (acc, m) => {
      (acc[m.category] ??= []).push(m);
      return acc;
    },
    {}
  );

  function handleDragStart(
    e: React.DragEvent<HTMLButtonElement>,
    mod: ModuleCatalogEntry
  ) {
    e.dataTransfer.setData(
      "application/reactflow-module",
      JSON.stringify(mod)
    );
    e.dataTransfer.effectAllowed = "move";
  }

  return (
    <div className="w-56 shrink-0 border-r bg-muted/30 flex flex-col h-full overflow-hidden">
      <div className="px-3 py-2 border-b shrink-0">
        <h3 className="text-sm font-semibold">Modules</h3>
        <p className="text-[10px] text-muted-foreground">
          Drag or click to add
        </p>
      </div>
      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="p-2 space-y-3">
          {isLoading ? (
            Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-8 w-full" />
            ))
          ) : (
            Object.entries(grouped).map(([category, mods]) => (
              <div key={category}>
                <Badge
                  variant="secondary"
                  className={cn(
                    "text-[10px] mb-1",
                    CATEGORY_COLORS[category]
                  )}
                >
                  {CATEGORY_LABELS[category] || category}
                </Badge>
                <div className="space-y-0.5">
                  {mods.map((mod) => (
                    <button
                      key={mod.type}
                      onClick={() => onAddModule(mod)}
                      draggable
                      onDragStart={(e) => handleDragStart(e, mod)}
                      className="w-full text-left px-2 py-1.5 rounded text-xs hover:bg-accent transition-colors truncate flex items-center gap-1 cursor-grab active:cursor-grabbing"
                    >
                      <GripVertical className="h-3 w-3 text-muted-foreground shrink-0" />
                      {mod.type}
                    </button>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
