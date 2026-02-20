"use client";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export function ModuleConfigEditor({
  config,
  onChange,
}: {
  config: Record<string, unknown>;
  onChange: (config: Record<string, unknown>) => void;
}) {
  const entries = Object.entries(config);

  if (entries.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No configuration options.</p>
    );
  }

  function handleChange(key: string, rawValue: string) {
    const original = config[key];
    let parsed: unknown = rawValue;

    if (typeof original === "number") {
      const num = Number(rawValue);
      if (!isNaN(num)) parsed = num;
    } else if (typeof original === "boolean") {
      parsed = rawValue === "true";
    } else if (Array.isArray(original)) {
      try {
        parsed = JSON.parse(rawValue);
      } catch {
        parsed = rawValue;
      }
    }

    onChange({ ...config, [key]: parsed });
  }

  return (
    <div className="space-y-3">
      {entries.map(([key, value]) => (
        <div key={key}>
          <Label htmlFor={`config-${key}`} className="text-xs font-mono mb-1">
            {key}
          </Label>
          <Input
            id={`config-${key}`}
            value={
              typeof value === "object" ? JSON.stringify(value) : String(value)
            }
            onChange={(e) => handleChange(key, e.target.value)}
            className="font-mono text-sm h-8"
          />
        </div>
      ))}
    </div>
  );
}
