"use client";

import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { PortBadge } from "./port-badge";
import type { PortDescriptor, PortType } from "@/lib/types";

interface ModuleInputEditorProps {
  inputs: PortDescriptor[];
  values: Record<string, { type: PortType; value: unknown }>;
  onChange: (
    values: Record<string, { type: PortType; value: unknown }>
  ) => void;
}

export function ModuleInputEditor({
  inputs,
  values,
  onChange,
}: ModuleInputEditorProps) {
  if (inputs.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        This module has no inputs (it&apos;s an input source).
      </p>
    );
  }

  function handleChange(port: PortDescriptor, rawValue: string) {
    let parsed: unknown = rawValue;

    switch (port.type) {
      case "TEXT":
        parsed = rawValue;
        break;
      case "TOKEN_IDS":
        try {
          parsed = JSON.parse(rawValue);
        } catch {
          parsed = rawValue
            .split(",")
            .map((s) => parseInt(s.trim()))
            .filter((n) => !isNaN(n));
        }
        break;
      case "INT":
        parsed = parseInt(rawValue) || 0;
        break;
      case "SCALAR":
        parsed = parseFloat(rawValue) || 0;
        break;
      case "TENSOR":
      case "AD_TENSOR":
        try {
          parsed = JSON.parse(rawValue);
        } catch {
          parsed = rawValue;
        }
        break;
    }

    onChange({
      ...values,
      [port.name]: { type: port.type, value: parsed },
    });
  }

  function getDefaultValue(port: PortDescriptor): string {
    const existing = values[port.name];
    if (existing) {
      return typeof existing.value === "object"
        ? JSON.stringify(existing.value)
        : String(existing.value);
    }

    switch (port.type) {
      case "TEXT":
        return "Hello world";
      case "TOKEN_IDS":
        return "[1, 2, 3, 4, 5]";
      case "INT":
        return "8";
      case "SCALAR":
        return "1.0";
      default:
        return "";
    }
  }

  return (
    <div className="space-y-3">
      {inputs.map((port) => (
        <div key={port.name}>
          <div className="flex items-center gap-2 mb-1">
            <Label htmlFor={`input-${port.name}`} className="text-xs">
              {port.name}
              {port.optional && (
                <span className="text-muted-foreground ml-1">(optional)</span>
              )}
            </Label>
            <PortBadge type={port.type} name={port.type} />
          </div>
          {port.type === "TEXT" ? (
            <Textarea
              id={`input-${port.name}`}
              defaultValue={getDefaultValue(port)}
              onChange={(e) => handleChange(port, e.target.value)}
              className="font-mono text-sm min-h-[60px]"
              rows={2}
            />
          ) : (
            <Input
              id={`input-${port.name}`}
              defaultValue={getDefaultValue(port)}
              onChange={(e) => handleChange(port, e.target.value)}
              className="font-mono text-sm h-8"
            />
          )}
        </div>
      ))}
    </div>
  );
}
