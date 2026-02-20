"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchPresets, fetchModules } from "@/lib/api";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useGraphStore } from "./hooks/use-graph-store";
import { fromGraphDef } from "@/lib/graph-utils";
import { Skeleton } from "@/components/ui/skeleton";

interface PresetLoaderProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function PresetLoader({ open, onOpenChange }: PresetLoaderProps) {
  const { data: presetsData, isLoading: presetsLoading } = useQuery({
    queryKey: ["presets"],
    queryFn: fetchPresets,
    enabled: open,
  });
  const { data: modulesData } = useQuery({
    queryKey: ["modules"],
    queryFn: fetchModules,
  });

  const { setNodes, setEdges, clearResults } = useGraphStore();

  function handleSelect(index: number) {
    const preset = presetsData?.presets[index];
    const catalog = modulesData?.modules;
    if (!preset || !catalog) return;

    const graphDef = { nodes: preset.nodes, edges: preset.edges };
    const { nodes, edges } = fromGraphDef(graphDef, catalog);
    setNodes(nodes);
    setEdges(edges);
    clearResults();
    onOpenChange(false);
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Load Preset Graph</DialogTitle>
        </DialogHeader>
        <div className="space-y-2 mt-2">
          {presetsLoading ? (
            Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))
          ) : (
            presetsData?.presets.map((preset, i) => (
              <Button
                key={i}
                variant="outline"
                className="w-full h-auto py-3 flex flex-col items-start text-left"
                onClick={() => handleSelect(i)}
              >
                <span className="font-semibold">{preset.name}</span>
                <span className="text-xs text-muted-foreground font-normal">
                  {preset.description}
                </span>
              </Button>
            ))
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
