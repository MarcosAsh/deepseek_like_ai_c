"use client";

import { useTheme } from "next-themes";
import { DiffEditor } from "@monaco-editor/react";
import { Skeleton } from "@/components/ui/skeleton";

interface CodeDiffViewerProps {
  original: string;
  modified: string;
  language?: string;
  height?: string;
}

export function CodeDiffViewer({
  original,
  modified,
  language = "cpp",
  height = "500px",
}: CodeDiffViewerProps) {
  const { resolvedTheme } = useTheme();

  return (
    <DiffEditor
      height={height}
      language={language}
      original={original}
      modified={modified}
      theme={resolvedTheme === "dark" ? "vs-dark" : "light"}
      options={{
        readOnly: true,
        minimap: { enabled: false },
        fontSize: 13,
        scrollBeyondLastLine: false,
        renderSideBySide: true,
      }}
      loading={<Skeleton className="h-full w-full" />}
    />
  );
}
