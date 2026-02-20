"use client";

import { useTheme } from "next-themes";
import Editor from "@monaco-editor/react";
import { Skeleton } from "@/components/ui/skeleton";

interface CodeViewerProps {
  code: string;
  language?: string;
  height?: string;
}

export function CodeViewer({
  code,
  language = "cpp",
  height = "500px",
}: CodeViewerProps) {
  const { resolvedTheme } = useTheme();

  return (
    <Editor
      height={height}
      language={language}
      value={code}
      theme={resolvedTheme === "dark" ? "vs-dark" : "light"}
      options={{
        readOnly: true,
        minimap: { enabled: false },
        fontSize: 13,
        lineNumbers: "on",
        scrollBeyondLastLine: false,
        wordWrap: "on",
        padding: { top: 8 },
      }}
      loading={<Skeleton className="h-full w-full" />}
    />
  );
}
