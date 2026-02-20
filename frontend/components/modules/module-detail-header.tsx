import { Badge } from "@/components/ui/badge";
import { PortBadge } from "./port-badge";
import { CATEGORY_COLORS } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { ModuleCatalogEntry } from "@/lib/types";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";

export function ModuleDetailHeader({ module }: { module: ModuleCatalogEntry }) {
  return (
    <div className="mb-6">
      <Link
        href="/modules"
        className="text-sm text-muted-foreground hover:text-foreground inline-flex items-center gap-1 mb-4"
      >
        <ArrowLeft className="h-3 w-3" />
        Back to catalog
      </Link>
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-3xl font-bold mb-1">{module.type}</h1>
          <p className="text-muted-foreground">{module.description}</p>
        </div>
        <Badge
          variant="secondary"
          className={cn("text-sm", CATEGORY_COLORS[module.category])}
        >
          {module.category}
        </Badge>
      </div>
      <div className="flex flex-wrap gap-6 mt-4">
        {module.inputs.length > 0 && (
          <div>
            <span className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
              Inputs
            </span>
            <div className="flex flex-wrap gap-1 mt-1">
              {module.inputs.map((port) => (
                <PortBadge
                  key={port.name}
                  type={port.type}
                  name={port.name}
                  optional={port.optional}
                />
              ))}
            </div>
          </div>
        )}
        {module.outputs.length > 0 && (
          <div>
            <span className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
              Outputs
            </span>
            <div className="flex flex-wrap gap-1 mt-1">
              {module.outputs.map((port) => (
                <PortBadge
                  key={port.name}
                  type={port.type}
                  name={port.name}
                  optional={port.optional}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
