import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PortBadge } from "./port-badge";
import { CATEGORY_COLORS } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { ModuleCatalogEntry } from "@/lib/types";
import { ArrowRight } from "lucide-react";

export function ModuleCard({ module }: { module: ModuleCatalogEntry }) {
  return (
    <Link href={`/modules/${module.type}`}>
      <Card className="h-full hover:border-primary/50 transition-colors group cursor-pointer">
        <CardHeader className="p-6 pb-4">
          <div className="flex items-start justify-between gap-2">
            <CardTitle className="text-lg font-semibold">
              {module.type}
            </CardTitle>
            <Badge
              variant="secondary"
              className={cn("text-xs shrink-0", CATEGORY_COLORS[module.category])}
            >
              {module.category}
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground line-clamp-2 leading-relaxed mt-1">
            {module.description}
          </p>
        </CardHeader>
        <CardContent className="px-6 pb-6 pt-0">
          {module.inputs.length > 0 && (
            <div className="mb-3">
              <span className="text-xs text-muted-foreground font-medium">
                Inputs:
              </span>
              <div className="flex flex-wrap gap-1.5 mt-1.5">
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
            <div className="mb-3">
              <span className="text-xs text-muted-foreground font-medium">
                Outputs:
              </span>
              <div className="flex flex-wrap gap-1.5 mt-1.5">
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
          <div className="flex items-center gap-1 text-sm text-muted-foreground group-hover:text-primary transition-colors mt-4">
            <span>Explore</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
