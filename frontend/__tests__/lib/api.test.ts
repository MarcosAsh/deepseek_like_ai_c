import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  fetchHealth,
  fetchModules,
  executeNode,
  executeGraph,
  fetchPresets,
} from "@/lib/api";

const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

beforeEach(() => {
  mockFetch.mockReset();
});

describe("fetchHealth", () => {
  it("returns health response on success", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: "ok", version: "1.0.0" }),
    });
    const result = await fetchHealth();
    expect(result).toEqual({ status: "ok", version: "1.0.0" });
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/v1/health"),
      expect.any(Object)
    );
  });

  it("throws on error response", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: async () => "Internal Server Error",
    });
    await expect(fetchHealth()).rejects.toThrow("API error 500");
  });
});

describe("fetchModules", () => {
  it("returns modules response", async () => {
    const modules = {
      modules: [
        {
          type: "TextInput",
          category: "input",
          description: "Text input",
          default_config: {},
          inputs: [],
          outputs: [{ name: "text", type: "TEXT", optional: false }],
        },
      ],
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => modules,
    });
    const result = await fetchModules();
    expect(result.modules).toHaveLength(1);
    expect(result.modules[0].type).toBe("TextInput");
  });

  it("throws on error", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 404,
      text: async () => "Not Found",
    });
    await expect(fetchModules()).rejects.toThrow("API error 404");
  });
});

describe("executeNode", () => {
  it("returns execution result", async () => {
    const response = {
      output: { type: "TENSOR", shape: [2, 2], data: [1, 2, 3, 4] },
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => response,
    });
    const result = await executeNode({
      type: "ADLinear",
      config: { input_dim: 2, output_dim: 2 },
      inputs: {},
    });
    expect(result.output.type).toBe("TENSOR");
  });

  it("sends POST with correct body", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({}),
    });
    const req = {
      type: "ADLinear",
      config: { input_dim: 3 },
      inputs: {},
    };
    await executeNode(req);
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/v1/execute_node"),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify(req),
      })
    );
  });
});

describe("executeGraph", () => {
  it("returns normalized graph result", async () => {
    const rawResponse = {
      nodes: {
        n1: {
          type: "TextInput",
          execution_time_ms: 1,
          outputs: { text: { type: "TEXT", value: "hello" } },
        },
      },
      execution_order: ["n1"],
      total_time_ms: 1,
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => rawResponse,
    });
    const result = await executeGraph({ nodes: [], edges: [] });
    expect(result.node_results).toHaveLength(1);
    expect(result.node_results[0].node_id).toBe("n1");
    expect(result.total_time_ms).toBe(1);
    expect(result.error).toBe("");
  });

  it("throws on error", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      text: async () => "Bad Request",
    });
    await expect(executeGraph({ nodes: [], edges: [] })).rejects.toThrow(
      "API error 400"
    );
  });
});

describe("fetchPresets", () => {
  it("returns presets", async () => {
    const presets = {
      presets: [
        {
          name: "Basic Pipeline",
          description: "A basic pipeline",
          nodes: [],
          edges: [],
        },
      ],
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => presets,
    });
    const result = await fetchPresets();
    expect(result.presets).toHaveLength(1);
    expect(result.presets[0].name).toBe("Basic Pipeline");
  });

  it("throws on error", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 503,
      text: async () => "Service Unavailable",
    });
    await expect(fetchPresets()).rejects.toThrow("API error 503");
  });
});
