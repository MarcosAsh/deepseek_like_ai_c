import { describe, it, expect, beforeEach } from "vitest";
import { useGraphStore } from "@/components/graph/hooks/use-graph-store";
import type { CustomNodeData } from "@/lib/graph-utils";
import type { Node, Edge } from "@xyflow/react";
import type { GraphResult } from "@/lib/types";

function makeTestNode(
  id: string,
  moduleType: string = "TextInput"
): Node<CustomNodeData> {
  return {
    id,
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      moduleType,
      category: "input",
      label: id,
      config: {},
      inputs: [],
      outputs: [],
    },
  };
}

function makeTestEdge(
  source: string,
  target: string
): Edge {
  return {
    id: `e-${source}-${target}`,
    source,
    target,
    sourceHandle: "output",
    targetHandle: "input",
  };
}

describe("useGraphStore", () => {
  beforeEach(() => {
    // Reset store to initial state
    useGraphStore.setState({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      results: new Map(),
      isExecuting: false,
      executingNodeId: null,
      history: [],
      future: [],
      executionHistory: [],
    });
  });

  describe("initial state", () => {
    it("has empty nodes and edges", () => {
      const state = useGraphStore.getState();
      expect(state.nodes).toEqual([]);
      expect(state.edges).toEqual([]);
      expect(state.selectedNodeId).toBeNull();
      expect(state.isExecuting).toBe(false);
      expect(state.executingNodeId).toBeNull();
      expect(state.history).toEqual([]);
      expect(state.future).toEqual([]);
      expect(state.executionHistory).toEqual([]);
    });
  });

  describe("addNode", () => {
    it("adds a node to the store", () => {
      const node = makeTestNode("n1");
      useGraphStore.getState().addNode(node);
      expect(useGraphStore.getState().nodes).toHaveLength(1);
      expect(useGraphStore.getState().nodes[0].id).toBe("n1");
    });

    it("adds multiple nodes", () => {
      useGraphStore.getState().addNode(makeTestNode("n1"));
      useGraphStore.getState().addNode(makeTestNode("n2"));
      expect(useGraphStore.getState().nodes).toHaveLength(2);
    });
  });

  describe("removeNode", () => {
    it("removes a node and its connected edges", () => {
      const n1 = makeTestNode("n1");
      const n2 = makeTestNode("n2");
      const edge = makeTestEdge("n1", "n2");

      useGraphStore.setState({ nodes: [n1, n2], edges: [edge] });
      useGraphStore.getState().removeNode("n1");

      expect(useGraphStore.getState().nodes).toHaveLength(1);
      expect(useGraphStore.getState().nodes[0].id).toBe("n2");
      expect(useGraphStore.getState().edges).toHaveLength(0);
    });

    it("clears selectedNodeId if removed node was selected", () => {
      const n1 = makeTestNode("n1");
      useGraphStore.setState({ nodes: [n1], selectedNodeId: "n1" });
      useGraphStore.getState().removeNode("n1");
      expect(useGraphStore.getState().selectedNodeId).toBeNull();
    });

    it("preserves selectedNodeId if different node removed", () => {
      const n1 = makeTestNode("n1");
      const n2 = makeTestNode("n2");
      useGraphStore.setState({ nodes: [n1, n2], selectedNodeId: "n1" });
      useGraphStore.getState().removeNode("n2");
      expect(useGraphStore.getState().selectedNodeId).toBe("n1");
    });
  });

  describe("duplicateNode", () => {
    it("duplicates a node with offset position", () => {
      const n1 = makeTestNode("n1");
      n1.position = { x: 100, y: 200 };
      useGraphStore.setState({ nodes: [n1] });
      useGraphStore.getState().duplicateNode("n1");

      const nodes = useGraphStore.getState().nodes;
      expect(nodes).toHaveLength(2);
      expect(nodes[1].position.x).toBe(150);
      expect(nodes[1].position.y).toBe(250);
      expect(nodes[1].id).not.toBe("n1");
    });

    it("does nothing for non-existent node", () => {
      useGraphStore.getState().duplicateNode("nonexistent");
      expect(useGraphStore.getState().nodes).toHaveLength(0);
    });
  });

  describe("undo/redo", () => {
    it("undo restores previous state", () => {
      const n1 = makeTestNode("n1");
      useGraphStore.setState({ nodes: [n1] });

      // Push current state to history
      useGraphStore.getState().pushHistory();

      // Add another node
      useGraphStore.getState().addNode(makeTestNode("n2"));
      expect(useGraphStore.getState().nodes).toHaveLength(2);

      // Undo
      useGraphStore.getState().undo();
      expect(useGraphStore.getState().nodes).toHaveLength(1);
      expect(useGraphStore.getState().nodes[0].id).toBe("n1");
    });

    it("redo restores undone state", () => {
      const n1 = makeTestNode("n1");
      useGraphStore.setState({ nodes: [n1] });
      useGraphStore.getState().pushHistory();

      useGraphStore.getState().addNode(makeTestNode("n2"));
      useGraphStore.getState().pushHistory();

      // Add a third node
      useGraphStore.getState().addNode(makeTestNode("n3"));
      expect(useGraphStore.getState().nodes).toHaveLength(3);

      // Undo twice
      useGraphStore.getState().undo();
      useGraphStore.getState().undo();
      expect(useGraphStore.getState().nodes).toHaveLength(1);

      // Redo once
      useGraphStore.getState().redo();
      expect(useGraphStore.getState().nodes).toHaveLength(2);
    });

    it("undo on empty history does nothing", () => {
      useGraphStore.getState().undo();
      expect(useGraphStore.getState().nodes).toEqual([]);
    });

    it("redo on empty future does nothing", () => {
      useGraphStore.getState().redo();
      expect(useGraphStore.getState().nodes).toEqual([]);
    });

    it("pushHistory clears future", () => {
      useGraphStore.setState({ nodes: [makeTestNode("n1")] });
      useGraphStore.getState().pushHistory();
      useGraphStore.getState().addNode(makeTestNode("n2"));

      // Undo creates future
      useGraphStore.getState().undo();
      expect(useGraphStore.getState().future).toHaveLength(1);

      // Push clears future
      useGraphStore.getState().pushHistory();
      expect(useGraphStore.getState().future).toHaveLength(0);
    });
  });

  describe("execution history", () => {
    it("adds execution snapshots", () => {
      const result: GraphResult = {
        node_results: [
          {
            node_id: "n1",
            node_type: "TextInput",
            execution_time_ms: 5,
            outputs: {},
            error: "",
          },
        ],
        execution_order: ["n1"],
        total_time_ms: 5,
        error: "",
      };

      useGraphStore.getState().addExecutionSnapshot(result);
      expect(useGraphStore.getState().executionHistory).toHaveLength(1);
      expect(
        useGraphStore.getState().executionHistory[0].totalTimeMs
      ).toBe(5);
    });

    it("limits to 20 snapshots", () => {
      for (let i = 0; i < 25; i++) {
        useGraphStore.getState().addExecutionSnapshot({
          node_results: [],
          execution_order: [],
          total_time_ms: i,
          error: "",
        });
      }
      expect(useGraphStore.getState().executionHistory.length).toBeLessThanOrEqual(20);
    });

    it("clearExecutionHistory empties the list", () => {
      useGraphStore.getState().addExecutionSnapshot({
        node_results: [],
        execution_order: [],
        total_time_ms: 1,
        error: "",
      });
      useGraphStore.getState().clearExecutionHistory();
      expect(useGraphStore.getState().executionHistory).toHaveLength(0);
    });
  });

  describe("clearAll", () => {
    it("resets all state except executionHistory", () => {
      useGraphStore.setState({
        nodes: [makeTestNode("n1")],
        edges: [makeTestEdge("n1", "n2")],
        selectedNodeId: "n1",
        results: new Map([
          ["n1", { node_id: "n1", node_type: "T", execution_time_ms: 1, outputs: {}, error: "" }],
        ]),
        history: [{ nodes: [], edges: [] }],
        future: [{ nodes: [], edges: [] }],
      });

      useGraphStore.getState().clearAll();
      const state = useGraphStore.getState();
      expect(state.nodes).toEqual([]);
      expect(state.edges).toEqual([]);
      expect(state.selectedNodeId).toBeNull();
      expect(state.results.size).toBe(0);
      expect(state.history).toEqual([]);
      expect(state.future).toEqual([]);
    });
  });

  describe("updateNodeConfig", () => {
    it("updates config for the specified node", () => {
      const n1 = makeTestNode("n1");
      useGraphStore.setState({ nodes: [n1] });
      useGraphStore.getState().updateNodeConfig("n1", { text: "updated" });
      const node = useGraphStore.getState().nodes[0];
      expect((node.data as CustomNodeData).config).toEqual({ text: "updated" });
    });

    it("does not affect other nodes", () => {
      const n1 = makeTestNode("n1");
      const n2 = makeTestNode("n2");
      useGraphStore.setState({ nodes: [n1, n2] });
      useGraphStore.getState().updateNodeConfig("n1", { text: "changed" });
      const node2 = useGraphStore.getState().nodes[1];
      expect((node2.data as CustomNodeData).config).toEqual({});
    });
  });

  describe("selectNode", () => {
    it("sets selectedNodeId", () => {
      useGraphStore.getState().selectNode("n1");
      expect(useGraphStore.getState().selectedNodeId).toBe("n1");
    });

    it("clears with null", () => {
      useGraphStore.getState().selectNode("n1");
      useGraphStore.getState().selectNode(null);
      expect(useGraphStore.getState().selectedNodeId).toBeNull();
    });
  });
});
