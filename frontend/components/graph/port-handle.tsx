import { Handle, Position } from "@xyflow/react";
import { getPortColor } from "@/lib/constants";
import type { PortType } from "@/lib/types";

interface PortHandleProps {
  type: "source" | "target";
  portType: PortType;
  portName: string;
  position: Position;
  style?: React.CSSProperties;
}

export function PortHandle({
  type,
  portType,
  portName,
  position,
  style,
}: PortHandleProps) {
  const color = getPortColor(portType);

  return (
    <Handle
      type={type}
      position={position}
      id={portName}
      style={{
        width: 10,
        height: 10,
        backgroundColor: color,
        border: "2px solid var(--background)",
        ...style,
      }}
      title={`${portName} (${portType})`}
    />
  );
}
