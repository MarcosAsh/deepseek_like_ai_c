import { describe, it, expect } from "vitest";
import { normalizeGraphResult } from "@/lib/types";
import type { GraphResultRaw } from "@/lib/types";

describe("normalizeGraphResult", () => {
  it("handles empty nodes", () => {
    const raw: GraphResultRaw = {
      nodes: {},
      execution_order: [],
      total_time_ms: 0,
    };
    const result = normalizeGraphResult(raw);
    expect(result.node_results).toEqual([]);
    expect(result.execution_order).toEqual([]);
    expect(result.total_time_ms).toBe(0);
    expect(result.error).toBe("");
  });

  it("normalizes a single node", () => {
    const raw: GraphResultRaw = {
      nodes: {
        node_1: {
          type: "ADLinear",
          execution_time_ms: 5,
          outputs: {
            output: {
              type: "TENSOR",
              shape: [4, 2],
              data: [1, 2, 3, 4, 5, 6, 7, 8],
            },
          },
        },
      },
      execution_order: ["node_1"],
      total_time_ms: 5,
    };
    const result = normalizeGraphResult(raw);
    expect(result.node_results).toHaveLength(1);
    expect(result.node_results[0].node_id).toBe("node_1");
    expect(result.node_results[0].node_type).toBe("ADLinear");
    expect(result.node_results[0].execution_time_ms).toBe(5);
    expect(result.node_results[0].outputs.output.type).toBe("TENSOR");
    expect(result.node_results[0].error).toBe("");
  });

  it("normalizes multiple nodes", () => {
    const raw: GraphResultRaw = {
      nodes: {
        n1: {
          type: "TextInput",
          execution_time_ms: 1,
          outputs: { text: { type: "TEXT", value: "hello" } },
        },
        n2: {
          type: "Tokenizer",
          execution_time_ms: 3,
          outputs: { token_ids: { type: "TOKEN_IDS", data: [1, 2, 3] } },
        },
      },
      execution_order: ["n1", "n2"],
      total_time_ms: 4,
    };
    const result = normalizeGraphResult(raw);
    expect(result.node_results).toHaveLength(2);
    expect(result.execution_order).toEqual(["n1", "n2"]);
    expect(result.total_time_ms).toBe(4);
  });

  it("handles missing fields gracefully", () => {
    const raw: GraphResultRaw = {
      nodes: {
        n1: {
          type: "Unknown",
          execution_time_ms: 0,
          outputs: null,
        },
      },
      execution_order: [],
      total_time_ms: 0,
    };
    const result = normalizeGraphResult(raw);
    expect(result.node_results[0].outputs).toEqual({});
    expect(result.node_results[0].error).toBe("");
  });

  it("preserves error field", () => {
    const raw: GraphResultRaw = {
      nodes: {
        n1: {
          type: "ADLinear",
          execution_time_ms: 0,
          outputs: null,
          error: "dimension mismatch",
        },
      },
      execution_order: ["n1"],
      total_time_ms: 1,
      error: "graph execution failed",
    };
    const result = normalizeGraphResult(raw);
    expect(result.node_results[0].error).toBe("dimension mismatch");
    expect(result.error).toBe("graph execution failed");
  });
});
