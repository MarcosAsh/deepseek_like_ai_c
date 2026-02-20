"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { ModuleConfigEditor } from "./module-config-editor";
import { ModuleInputEditor } from "./module-input-editor";
import { ModuleOutputViewer } from "./module-output-viewer";
import { executeNode } from "@/lib/api";
import type {
  ModuleCatalogEntry,
  PortType,
  TensorData,
} from "@/lib/types";
import { Play, Loader2 } from "lucide-react";

export function ModuleExecutor({ module }: { module: ModuleCatalogEntry }) {
  const [config, setConfig] = useState<Record<string, unknown>>(
    module.default_config
  );
  const [inputs, setInputs] = useState<
    Record<string, { type: PortType; value: unknown }>
  >(() => {
    const init: Record<string, { type: PortType; value: unknown }> = {};
    for (const port of module.inputs) {
      let defaultValue: unknown = "";
      switch (port.type) {
        case "TEXT":
          defaultValue = "Hello world";
          break;
        case "TOKEN_IDS":
          defaultValue = [1, 2, 3, 4, 5];
          break;
        case "INT":
          defaultValue = 8;
          break;
        case "SCALAR":
          defaultValue = 1.0;
          break;
      }
      init[port.name] = { type: port.type, value: defaultValue };
    }
    return init;
  });

  const [outputs, setOutputs] = useState<Record<string, TensorData> | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [execTime, setExecTime] = useState<number | null>(null);

  async function handleRun() {
    setLoading(true);
    setError(null);
    setOutputs(null);
    setExecTime(null);

    const start = performance.now();
    try {
      const result = await executeNode({
        type: module.type,
        config,
        inputs,
      });
      setExecTime(performance.now() - start);
      setOutputs(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <ModuleConfigEditor config={config} onChange={setConfig} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Inputs</CardTitle>
        </CardHeader>
        <CardContent>
          <ModuleInputEditor
            inputs={module.inputs}
            values={inputs}
            onChange={setInputs}
          />
        </CardContent>
      </Card>

      <Button
        onClick={handleRun}
        disabled={loading}
        className="w-full"
        size="lg"
      >
        {loading ? (
          <>
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            Running...
          </>
        ) : (
          <>
            <Play className="h-4 w-4 mr-2" />
            Run Module
          </>
        )}
      </Button>

      {error && (
        <Card className="border-destructive">
          <CardContent className="py-3">
            <p className="text-sm text-destructive font-mono">{error}</p>
          </CardContent>
        </Card>
      )}

      {outputs && (
        <div>
          <Separator className="my-4" />
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium">Output</h3>
            {execTime !== null && (
              <span className="text-xs text-muted-foreground font-mono">
                {execTime.toFixed(1)}ms (round-trip)
              </span>
            )}
          </div>
          <ModuleOutputViewer outputs={outputs} />
        </div>
      )}
    </div>
  );
}
