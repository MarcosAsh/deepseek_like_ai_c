"use client";

import { useCallback, useEffect, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  useReactFlow,
  type NodeChange,
  type EdgeChange,
  type Connection,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { CustomNode } from "./custom-node";
import { CustomEdge } from "./custom-edge";
import { ModulePalette } from "./module-palette";
import { GraphToolbar } from "./graph-toolbar";
import { PresetLoader } from "./preset-loader";
import { NodeConfigPanel } from "./node-config-panel";
import { NodeContextMenu } from "./node-context-menu";
import { useGraphStore } from "./hooks/use-graph-store";
import { useEdgeValidation } from "./hooks/use-edge-validation";
import type { ModuleCatalogEntry } from "@/lib/types";
import type { CustomNodeData } from "@/lib/graph-utils";
import { CATEGORY_COLORS_HEX } from "@/lib/constants";

const nodeTypes = { custom: CustomNode };
const edgeTypes = { custom: CustomEdge };

let nodeCounter = 0;

function GraphEditorInner() {
  const {
    nodes,
    edges,
    setNodes,
    setEdges,
    selectNode,
    selectedNodeId,
    pushHistory,
    undo,
    redo,
  } = useGraphStore();

  const { isValidConnection } = useEdgeValidation();
  const { screenToFlowPosition } = useReactFlow();
  const [presetOpen, setPresetOpen] = useState(false);
  const [contextMenu, setContextMenu] = useState<{
    nodeId: string;
    x: number;
    y: number;
  } | null>(null);

  // Keyboard shortcuts for undo/redo
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "z") {
        e.preventDefault();
        if (e.shiftKey) {
          redo();
        } else {
          undo();
        }
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [undo, redo]);

  const onNodesChange = useCallback(
    (changes: NodeChange<Node<CustomNodeData>>[]) => {
      const hasStructural = changes.some(
        (c) => c.type === "remove" || c.type === "add"
      );
      if (hasStructural) pushHistory();
      setNodes(applyNodeChanges(changes, nodes) as Node<CustomNodeData>[]);
    },
    [nodes, setNodes, pushHistory]
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      const hasStructural = changes.some(
        (c) => c.type === "remove" || c.type === "add"
      );
      if (hasStructural) pushHistory();
      setEdges(applyEdgeChanges(changes, edges));
    },
    [edges, setEdges, pushHistory]
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      // Find port type for edge coloring
      const sourceNode = nodes.find((n) => n.id === connection.source);
      const sourceData = sourceNode?.data as CustomNodeData | undefined;
      const sourcePort = sourceData?.outputs?.find(
        (p) => p.name === connection.sourceHandle
      );
      const portType = sourcePort?.type;

      pushHistory();
      setEdges(
        addEdge(
          {
            ...connection,
            type: "custom",
            data: { portType, sourceNodeId: connection.source },
          },
          edges
        )
      );
    },
    [edges, setEdges, nodes, pushHistory]
  );

  const handleAddModule = useCallback(
    (module: ModuleCatalogEntry, position?: { x: number; y: number }) => {
      const id = `${module.type.toLowerCase()}_${++nodeCounter}`;
      const newNode: Node<CustomNodeData> = {
        id,
        type: "custom",
        position: position ?? {
          x: 100 + Math.random() * 200,
          y: 100 + Math.random() * 200,
        },
        data: {
          moduleType: module.type,
          category: module.category,
          label: id,
          config: { ...module.default_config },
          inputs: module.inputs,
          outputs: module.outputs,
        },
      };
      pushHistory();
      setNodes([...nodes, newNode]);
    },
    [nodes, setNodes, pushHistory]
  );

  // Drag-and-drop from palette
  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const moduleJson = e.dataTransfer.getData(
        "application/reactflow-module"
      );
      if (!moduleJson) return;

      try {
        const mod: ModuleCatalogEntry = JSON.parse(moduleJson);
        const position = screenToFlowPosition({
          x: e.clientX,
          y: e.clientY,
        });
        handleAddModule(mod, position);
      } catch {
        // invalid JSON, ignore
      }
    },
    [screenToFlowPosition, handleAddModule]
  );

  // Context menu
  const onNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: Node) => {
      event.preventDefault();
      setContextMenu({
        nodeId: node.id,
        x: event.clientX,
        y: event.clientY,
      });
    },
    []
  );

  const closeContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

  // Minimap node color by category
  const minimapNodeColor = useCallback(
    (node: Node) => {
      const data = node.data as CustomNodeData;
      return CATEGORY_COLORS_HEX[data?.category] ?? "#6b7280";
    },
    []
  );

  return (
    <div className="flex h-full overflow-hidden">
      <ModulePalette onAddModule={handleAddModule} />
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <GraphToolbar onLoadPreset={() => setPresetOpen(true)} />
        <div className="flex-1 relative min-h-0">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onDragOver={onDragOver}
            onDrop={onDrop}
            isValidConnection={isValidConnection}
            onNodeClick={(_, node) => {
              selectNode(node.id);
              closeContextMenu();
            }}
            onPaneClick={() => {
              selectNode(null);
              closeContextMenu();
            }}
            onNodeContextMenu={onNodeContextMenu}
            fitView
            proOptions={{ hideAttribution: true }}
            className="bg-background"
          >
            <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
            <Controls />
            <MiniMap
              nodeColor={minimapNodeColor}
              maskColor="rgba(0,0,0,0.1)"
              className="!bg-muted/50"
              pannable
              zoomable
            />
          </ReactFlow>
          {contextMenu && (
            <NodeContextMenu
              nodeId={contextMenu.nodeId}
              x={contextMenu.x}
              y={contextMenu.y}
              onClose={closeContextMenu}
            />
          )}
        </div>
      </div>
      {selectedNodeId && <NodeConfigPanel />}
      <PresetLoader open={presetOpen} onOpenChange={setPresetOpen} />
    </div>
  );
}

export function GraphEditor() {
  return (
    <ReactFlowProvider>
      <GraphEditorInner />
    </ReactFlowProvider>
  );
}
