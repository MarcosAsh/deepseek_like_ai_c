import { cn } from "@/lib/utils";
import { Info, AlertTriangle, Lightbulb } from "lucide-react";

const variants = {
  info: {
    icon: Info,
    className: "border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950",
  },
  warning: {
    icon: AlertTriangle,
    className: "border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-950",
  },
  tip: {
    icon: Lightbulb,
    className: "border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950",
  },
};

export function Callout({
  variant = "info",
  children,
}: {
  variant?: keyof typeof variants;
  children: React.ReactNode;
}) {
  const v = variants[variant];
  return (
    <div className={cn("border rounded-lg p-4 flex gap-3 my-4", v.className)}>
      <v.icon className="h-5 w-5 shrink-0 mt-0.5" />
      <div className="text-sm">{children}</div>
    </div>
  );
}
