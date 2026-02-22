import { describe, it, expect } from "vitest";
import { toGraphDef, fromGraphDef } from "@/lib/graph-utils";
import type { CustomNodeData } from "@/lib/graph-utils";
import type { Node, Edge } from "@xyflow/react";
import type { ModuleCatalogEntry, GraphDef } from "@/lib/types";

const mockCatalog: ModuleCatalogEntry[] = [
  {
    type: "TextInput",
    category: "input",
    description: "Text input",
    default_config: { text: "hello" },
    inputs: [],
    outputs: [{ name: "text", type: "TEXT", optional: false }],
  },
  {
    type: "Tokenizer",
    category: "preprocessing",
    description: "Tokenizer",
    default_config: {},
    inputs: [{ name: "text", type: "TEXT", optional: false }],
    outputs: [{ name: "token_ids", type: "TOKEN_IDS", optional: false }],
  },
  {
    type: "ADEmbedding",
    category: "embedding",
    description: "Embedding",
    default_config: { vocab_size: 100, embed_dim: 32 },
    inputs: [{ name: "token_ids", type: "TOKEN_IDS", optional: false }],
    outputs: [{ name: "output", type: "AD_TENSOR", optional: false }],
  },
];

function makeNode(
  id: string,
  moduleType: string,
  config: Record<string, unknown> = {}
): Node<CustomNodeData> {
  const mod = mockCatalog.find((m) => m.type === moduleType)!;
  return {
    id,
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      moduleType,
      category: mod.category,
      label: id,
      config: { ...mod.default_config, ...config },
      inputs: mod.inputs,
      outputs: mod.outputs,
    },
  };
}

describe("toGraphDef", () => {
  it("converts React Flow nodes and edges to GraphDef", () => {
    const nodes = [
      makeNode("text_1", "TextInput", { text: "test" }),
      makeNode("tok_1", "Tokenizer"),
    ];
    const edges: Edge[] = [
      {
        id: "e1",
        source: "text_1",
        sourceHandle: "text",
        target: "tok_1",
        targetHandle: "text",
      },
    ];

    const graphDef = toGraphDef(nodes, edges);

    expect(graphDef.nodes).toHaveLength(2);
    expect(graphDef.nodes[0].id).toBe("text_1");
    expect(graphDef.nodes[0].type).toBe("TextInput");
    expect(graphDef.nodes[0].config).toEqual({ text: "test" });

    expect(graphDef.edges).toHaveLength(1);
    expect(graphDef.edges[0].source_node).toBe("text_1");
    expect(graphDef.edges[0].source_port).toBe("text");
    expect(graphDef.edges[0].target_node).toBe("tok_1");
    expect(graphDef.edges[0].target_port).toBe("text");
  });

  it("handles empty graph", () => {
    const graphDef = toGraphDef([], []);
    expect(graphDef.nodes).toEqual([]);
    expect(graphDef.edges).toEqual([]);
  });

  it("handles edges without handles", () => {
    const nodes = [makeNode("a", "TextInput")];
    const edges: Edge[] = [
      { id: "e1", source: "a", target: "b" },
    ];
    const graphDef = toGraphDef(nodes, edges);
    expect(graphDef.edges[0].source_port).toBe("");
    expect(graphDef.edges[0].target_port).toBe("");
  });
});

describe("fromGraphDef", () => {
  it("converts GraphDef to React Flow nodes with catalog lookup", () => {
    const graphDef: GraphDef = {
      nodes: [
        { id: "text_1", type: "TextInput", config: { text: "hello" } },
        { id: "tok_1", type: "Tokenizer", config: {} },
      ],
      edges: [
        {
          source_node: "text_1",
          source_port: "text",
          target_node: "tok_1",
          target_port: "text",
        },
      ],
    };

    const { nodes, edges } = fromGraphDef(graphDef, mockCatalog);

    expect(nodes).toHaveLength(2);
    expect(nodes[0].id).toBe("text_1");
    expect((nodes[0].data as CustomNodeData).moduleType).toBe("TextInput");
    expect((nodes[0].data as CustomNodeData).category).toBe("input");
    expect((nodes[0].data as CustomNodeData).outputs).toHaveLength(1);

    expect(edges).toHaveLength(1);
    expect(edges[0].source).toBe("text_1");
    expect(edges[0].target).toBe("tok_1");
  });

  it("handles unknown module type gracefully", () => {
    const graphDef: GraphDef = {
      nodes: [{ id: "x", type: "NonExistent", config: {} }],
      edges: [],
    };
    const { nodes } = fromGraphDef(graphDef, mockCatalog);
    expect(nodes).toHaveLength(1);
    expect((nodes[0].data as CustomNodeData).category).toBe("unknown");
  });

  it("round-trip preserves graph structure", () => {
    const originalDef: GraphDef = {
      nodes: [
        { id: "text_1", type: "TextInput", config: { text: "world" } },
        { id: "emb_1", type: "ADEmbedding", config: { vocab_size: 50, embed_dim: 16 } },
      ],
      edges: [
        {
          source_node: "text_1",
          source_port: "text",
          target_node: "emb_1",
          target_port: "token_ids",
        },
      ],
    };

    const { nodes, edges } = fromGraphDef(originalDef, mockCatalog);
    const roundTripped = toGraphDef(nodes, edges);

    expect(roundTripped.nodes).toHaveLength(2);
    expect(roundTripped.nodes[0].id).toBe("text_1");
    expect(roundTripped.nodes[0].type).toBe("TextInput");
    expect(roundTripped.nodes[1].id).toBe("emb_1");
    expect(roundTripped.nodes[1].type).toBe("ADEmbedding");

    expect(roundTripped.edges).toHaveLength(1);
    expect(roundTripped.edges[0].source_node).toBe("text_1");
    expect(roundTripped.edges[0].target_node).toBe("emb_1");
  });
});
