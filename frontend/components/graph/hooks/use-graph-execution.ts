import { useCallback } from "react";
import { useGraphStore } from "./use-graph-store";
import { executeGraph } from "@/lib/api";
import { toGraphDef } from "@/lib/graph-utils";
import type { NodeResult } from "@/lib/types";
import { toast } from "sonner";

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function useGraphExecution() {
  const {
    nodes,
    edges,
    setResults,
    setIsExecuting,
    setExecutingNodeId,
    addExecutionSnapshot,
  } = useGraphStore();

  const execute = useCallback(async () => {
    if (nodes.length === 0) {
      toast.error("No nodes in the graph");
      return;
    }

    setIsExecuting(true);
    setExecutingNodeId(null);

    try {
      const graphDef = toGraphDef(nodes, edges);
      const result = await executeGraph(graphDef);

      if (result.error) {
        toast.error(`Graph error: ${result.error}`);
        return;
      }

      // Animate through execution order step by step
      const resultsMap = new Map<string, NodeResult>();
      for (const nodeId of result.execution_order) {
        setExecutingNodeId(nodeId);
        const nr = result.node_results.find((r) => r.node_id === nodeId);
        if (nr) {
          resultsMap.set(nr.node_id, nr);
          setResults(new Map(resultsMap));
        }
        await delay(200); // Brief pause for visual feedback
      }
      setExecutingNodeId(null);

      // Save to execution history
      addExecutionSnapshot(result);

      // Rich toast with stats
      const nodeCount = result.node_results.length;
      const slowest = result.node_results.reduce(
        (max, nr) =>
          nr.execution_time_ms > max.execution_time_ms ? nr : max,
        result.node_results[0]
      );
      const errorCount = result.node_results.filter((nr) => nr.error).length;

      toast.success(`Executed ${nodeCount} nodes in ${result.total_time_ms.toFixed(1)}ms`, {
        description: errorCount > 0
          ? `${errorCount} node(s) had errors`
          : `Slowest: ${slowest.node_type} (${slowest.execution_time_ms.toFixed(1)}ms)`,
      });
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Execution failed"
      );
    } finally {
      setIsExecuting(false);
      setExecutingNodeId(null);
    }
  }, [nodes, edges, setResults, setIsExecuting, setExecutingNodeId, addExecutionSnapshot]);

  return { execute };
}
