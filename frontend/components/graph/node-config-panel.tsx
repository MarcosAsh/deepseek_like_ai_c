"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ModuleConfigEditor } from "@/components/modules/module-config-editor";
import { PortValueRenderer } from "@/components/visualization/port-value-renderer";
import { CodeViewer } from "@/components/code/code-viewer";
import { useGraphStore } from "./hooks/use-graph-store";
import { MODULE_DOCS } from "@/lib/module-docs";
import { MODULE_SOURCE_FILES, getSourceUrl } from "@/lib/source-code";
import type { CustomNodeData } from "@/lib/graph-utils";
import type { TensorData } from "@/lib/types";
import { X, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export function NodeConfigPanel() {
  const { nodes, selectedNodeId, selectNode, updateNodeConfig, results } =
    useGraphStore();

  const node = nodes.find((n) => n.id === selectedNodeId);
  if (!node) return null;

  const data = node.data as CustomNodeData;
  const result = results.get(node.id);

  return (
    <div className="w-80 border-l bg-muted/30 flex flex-col h-full overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <h3 className="text-sm font-semibold truncate">{data.moduleType}</h3>
          <Link href={`/modules/${data.moduleType}`}>
            <ExternalLink className="h-3 w-3 text-muted-foreground hover:text-foreground" />
          </Link>
        </div>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6 shrink-0"
          onClick={() => selectNode(null)}
        >
          <X className="h-3 w-3" />
        </Button>
      </div>

      <Tabs defaultValue="config" className="flex-1 flex flex-col min-h-0">
        <TabsList className="mx-3 mt-2 shrink-0">
          <TabsTrigger value="config" className="text-xs">Config</TabsTrigger>
          <TabsTrigger value="output" className="text-xs">Output</TabsTrigger>
          <TabsTrigger value="code" className="text-xs">Code</TabsTrigger>
          <TabsTrigger value="docs" className="text-xs">Docs</TabsTrigger>
        </TabsList>

        <ScrollArea className="flex-1 min-h-0">
          <TabsContent value="config" className="p-3 mt-0 space-y-3">
            <Card>
              <CardHeader className="py-2 px-3">
                <CardTitle className="text-xs">Configuration</CardTitle>
              </CardHeader>
              <CardContent className="px-3 pb-3">
                <ModuleConfigEditor
                  config={data.config}
                  onChange={(newConfig) => updateNodeConfig(node.id, newConfig)}
                />
              </CardContent>
            </Card>

            {result && !result.error && (
              <Card>
                <CardContent className="px-3 py-2">
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="text-[10px]">
                      {result.execution_time_ms.toFixed(1)}ms
                    </Badge>
                    <span className="text-[10px] text-muted-foreground">
                      {Object.keys(result.outputs).length} outputs
                    </span>
                  </div>
                </CardContent>
              </Card>
            )}

            {result?.error && (
              <Card className="border-destructive">
                <CardContent className="py-2 px-3">
                  <p className="text-xs text-destructive font-mono">
                    {result.error}
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="output" className="p-3 mt-0">
            <OutputTab nodeId={node.id} />
          </TabsContent>

          <TabsContent value="code" className="p-3 mt-0">
            <CodeTab moduleType={data.moduleType} />
          </TabsContent>

          <TabsContent value="docs" className="p-3 mt-0">
            <DocsTab moduleType={data.moduleType} />
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
}

function OutputTab({ nodeId }: { nodeId: string }) {
  const results = useGraphStore((s) => s.results);
  const result = results.get(nodeId);

  if (!result) {
    return (
      <p className="text-sm text-muted-foreground py-4 text-center">
        Run the graph to see outputs
      </p>
    );
  }

  if (result.error) {
    return (
      <Card className="border-destructive">
        <CardContent className="py-3 px-3">
          <p className="text-xs text-destructive font-mono">{result.error}</p>
        </CardContent>
      </Card>
    );
  }

  const outputs = Object.entries(result.outputs);

  return (
    <div className="space-y-3">
      {outputs.map(([name, tensorData]) => (
        <Card key={name}>
          <CardHeader className="py-2 px-3">
            <CardTitle className="text-xs font-mono">{name}</CardTitle>
          </CardHeader>
          <CardContent className="px-3 pb-3">
            <PortValueRenderer data={tensorData as TensorData} />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function CodeTab({ moduleType }: { moduleType: string }) {
  const sourceFiles = MODULE_SOURCE_FILES[moduleType];
  const allFiles = sourceFiles
    ? [...sourceFiles.src, ...sourceFiles.include]
    : [];
  const [selectedFile, setSelectedFile] = useState(allFiles[0] || "");
  const [sourceCode, setSourceCode] = useState<string>("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedFile) return;
    setLoading(true);
    fetch(getSourceUrl(selectedFile))
      .then((res) => (res.ok ? res.text() : "// Source file not available"))
      .then(setSourceCode)
      .catch(() => setSourceCode("// Failed to load source"))
      .finally(() => setLoading(false));
  }, [selectedFile]);

  if (allFiles.length === 0) {
    return (
      <p className="text-sm text-muted-foreground py-4 text-center">
        No source files available for this module
      </p>
    );
  }

  return (
    <div className="space-y-2">
      {allFiles.length > 1 && (
        <Select value={selectedFile} onValueChange={setSelectedFile}>
          <SelectTrigger className="w-full font-mono text-xs h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {allFiles.map((f) => (
              <SelectItem key={f} value={f} className="font-mono text-xs">
                {f.split("/").pop()}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
      {allFiles.length === 1 && (
        <p className="text-[10px] font-mono text-muted-foreground">
          {allFiles[0]}
        </p>
      )}
      {loading ? (
        <div className="h-[300px] animate-pulse bg-muted rounded" />
      ) : (
        <div className="border rounded overflow-hidden">
          <CodeViewer code={sourceCode} language="cpp" height="350px" />
        </div>
      )}
    </div>
  );
}

function DocsTab({ moduleType }: { moduleType: string }) {
  const docs = MODULE_DOCS[moduleType];

  if (!docs) {
    return (
      <p className="text-sm text-muted-foreground py-4 text-center">
        No documentation available yet.
      </p>
    );
  }

  return (
    <div
      className="prose dark:prose-invert prose-sm max-w-none"
      dangerouslySetInnerHTML={{ __html: docs }}
    />
  );
}
