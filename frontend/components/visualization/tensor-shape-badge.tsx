import { Badge } from "@/components/ui/badge";

export function TensorShapeBadge({ shape }: { shape: [number, number] }) {
  return (
    <Badge variant="outline" className="font-mono text-xs">
      [{shape[0]} x {shape[1]}]
    </Badge>
  );
}
