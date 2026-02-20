"use client";

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { Copy, Trash2, Eye, ExternalLink } from "lucide-react";
import { useGraphStore } from "./hooks/use-graph-store";
import type { CustomNodeData } from "@/lib/graph-utils";

interface NodeContextMenuProps {
  nodeId: string;
  x: number;
  y: number;
  onClose: () => void;
}

export function NodeContextMenu({
  nodeId,
  x,
  y,
  onClose,
}: NodeContextMenuProps) {
  const ref = useRef<HTMLDivElement>(null);
  const router = useRouter();
  const { nodes, removeNode, duplicateNode, selectNode, pushHistory } =
    useGraphStore();
  const node = nodes.find((n) => n.id === nodeId);
  const data = node?.data as CustomNodeData | undefined;

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as HTMLElement)) {
        onClose();
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [onClose]);

  if (!node || !data) return null;

  const items = [
    {
      label: "View Details",
      icon: Eye,
      action: () => {
        selectNode(nodeId);
        onClose();
      },
    },
    {
      label: "Duplicate",
      icon: Copy,
      action: () => {
        pushHistory();
        duplicateNode(nodeId);
        onClose();
      },
    },
    {
      label: "Go to Module Page",
      icon: ExternalLink,
      action: () => {
        router.push(`/modules/${data.moduleType}`);
        onClose();
      },
    },
    {
      label: "Delete",
      icon: Trash2,
      action: () => {
        pushHistory();
        removeNode(nodeId);
        onClose();
      },
      destructive: true,
    },
  ];

  return (
    <div
      ref={ref}
      className="fixed z-50 bg-popover border rounded-md shadow-lg py-1 min-w-[160px]"
      style={{ left: x, top: y }}
    >
      {items.map((item) => (
        <button
          key={item.label}
          onClick={item.action}
          className={`w-full flex items-center gap-2 px-3 py-1.5 text-sm hover:bg-accent transition-colors ${
            item.destructive ? "text-destructive" : ""
          }`}
        >
          <item.icon className="h-3.5 w-3.5" />
          {item.label}
        </button>
      ))}
    </div>
  );
}
