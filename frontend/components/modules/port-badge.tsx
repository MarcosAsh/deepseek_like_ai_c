import { Badge } from "@/components/ui/badge";
import { PORT_TYPE_BADGE_CLASSES } from "@/lib/constants";
import type { PortType } from "@/lib/types";
import { cn } from "@/lib/utils";

export function PortBadge({
  type,
  name,
  optional,
}: {
  type: PortType;
  name: string;
  optional?: boolean;
}) {
  return (
    <Badge
      variant="secondary"
      className={cn("text-xs font-mono", PORT_TYPE_BADGE_CLASSES[type])}
    >
      {name}
      {optional && "?"}
    </Badge>
  );
}
