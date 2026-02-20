import type { PortType } from "./types";

export const PORT_COLORS: Record<PortType, string> = {
  TEXT: "#3b82f6",
  TOKEN_IDS: "#22c55e",
  TENSOR: "#f97316",
  AD_TENSOR: "#a855f7",
  SCALAR: "#eab308",
  INT: "#6b7280",
};

export function getPortColor(type: PortType): string {
  return PORT_COLORS[type] ?? "#6b7280";
}
