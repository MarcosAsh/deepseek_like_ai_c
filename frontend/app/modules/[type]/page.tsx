"use client";

import { useQuery } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { fetchModules } from "@/lib/api";
import { MODULE_SOURCE_FILES, getSourceUrl } from "@/lib/source-code";
import { MODULE_DOCS } from "@/lib/module-docs";
import { ModuleDetailHeader } from "@/components/modules/module-detail-header";
import { ModuleExecutor } from "@/components/modules/module-executor";
import { CodeViewer } from "@/components/code/code-viewer";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export default function ModuleDetailPage() {
  const params = useParams<{ type: string }>();
  const moduleType = params.type;

  const { data, isLoading } = useQuery({
    queryKey: ["modules"],
    queryFn: fetchModules,
  });

  const module = data?.modules.find((m) => m.type === moduleType);

  // Source code loading
  const sourceFiles = MODULE_SOURCE_FILES[moduleType];
  const allFiles = sourceFiles
    ? [...sourceFiles.src, ...sourceFiles.include]
    : [];
  const [selectedFile, setSelectedFile] = useState(allFiles[0] || "");
  const [sourceCode, setSourceCode] = useState<string>("");
  const [sourceLoading, setSourceLoading] = useState(false);

  useEffect(() => {
    if (!selectedFile) return;
    setSourceLoading(true);
    fetch(getSourceUrl(selectedFile))
      .then((res) => (res.ok ? res.text() : "// Source file not available"))
      .then(setSourceCode)
      .catch(() => setSourceCode("// Failed to load source"))
      .finally(() => setSourceLoading(false));
  }, [selectedFile]);

  // Set initial file selection when module loads
  useEffect(() => {
    if (allFiles.length > 0 && !selectedFile) {
      setSelectedFile(allFiles[0]);
    }
  }, [moduleType]); // eslint-disable-line react-hooks/exhaustive-deps

  if (isLoading) {
    return (
      <div className="container mx-auto max-w-7xl px-4 py-8">
        <Skeleton className="h-8 w-64 mb-4" />
        <Skeleton className="h-4 w-96 mb-8" />
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  if (!module) {
    return (
      <div className="container mx-auto max-w-7xl px-4 py-8">
        <h1 className="text-2xl font-bold">Module not found: {moduleType}</h1>
        <p className="text-muted-foreground mt-2">
          Make sure the backend is running and the module type is correct.
        </p>
      </div>
    );
  }

  const docs = MODULE_DOCS[moduleType];

  return (
    <div className="container mx-auto max-w-7xl px-4 py-8">
      <ModuleDetailHeader module={module} />

      <Tabs defaultValue="interactive" className="mt-6">
        <TabsList>
          <TabsTrigger value="interactive">Interactive</TabsTrigger>
          <TabsTrigger value="code">Source Code</TabsTrigger>
          <TabsTrigger value="docs">Documentation</TabsTrigger>
        </TabsList>

        <TabsContent value="interactive" className="mt-4">
          <div className="max-w-2xl">
            <ModuleExecutor module={module} />
          </div>
        </TabsContent>

        <TabsContent value="code" className="mt-4">
          <div className="space-y-3">
            {allFiles.length > 1 && (
              <Select value={selectedFile} onValueChange={setSelectedFile}>
                <SelectTrigger className="w-80 font-mono text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {allFiles.map((f) => (
                    <SelectItem key={f} value={f} className="font-mono text-sm">
                      {f}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            {allFiles.length === 1 && (
              <p className="text-sm font-mono text-muted-foreground">
                {allFiles[0]}
              </p>
            )}
            {sourceLoading ? (
              <Skeleton className="h-[500px] w-full" />
            ) : (
              <div className="border rounded-lg overflow-hidden">
                <CodeViewer
                  code={sourceCode}
                  language={selectedFile.endsWith(".hpp") ? "cpp" : "cpp"}
                  height="600px"
                />
              </div>
            )}
          </div>
        </TabsContent>

        <TabsContent value="docs" className="mt-4">
          <Card>
            <CardContent className="prose dark:prose-invert max-w-none py-6">
              {docs ? (
                <div dangerouslySetInnerHTML={{ __html: docs }} />
              ) : (
                <p className="text-muted-foreground">
                  Documentation for {module.type} is coming soon.
                </p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
