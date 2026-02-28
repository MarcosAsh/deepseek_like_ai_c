import { Badge } from "@/components/ui/badge";

export function TensorShapeBadge({ shape }: { shape: number[] }) {
  return (
    <Badge variant="outline" className="font-mono text-xs">
      [{shape.join(" x ")}]
    </Badge>
  );
}
