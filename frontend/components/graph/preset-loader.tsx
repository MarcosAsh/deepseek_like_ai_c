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
import { Badge } from "@/components/ui/badge";
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
      <DialogContent className="max-w-2xl max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>Load Preset Graph</DialogTitle>
        </DialogHeader>
        <div className="space-y-3 mt-2 overflow-y-auto pr-2 flex-1 min-h-0">
          {presetsLoading ? (
            Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-20 w-full" />
            ))
          ) : (
            presetsData?.presets.map((preset, i) => (
              <Button
                key={i}
                variant="outline"
                className="w-full h-auto py-4 px-5 flex flex-col items-start text-left gap-1.5 shrink-0 whitespace-normal overflow-hidden"
                onClick={() => handleSelect(i)}
              >
                <div className="flex items-center gap-2 w-full min-w-0">
                  <span className="font-semibold text-base truncate">{preset.name}</span>
                  <Badge variant="secondary" className="text-[10px] shrink-0">
                    {preset.nodes.length} nodes
                  </Badge>
                </div>
                <span className="text-sm text-muted-foreground font-normal leading-relaxed whitespace-normal break-words w-full">
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
