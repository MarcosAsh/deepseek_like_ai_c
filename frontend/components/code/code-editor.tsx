"use client";

import { useTheme } from "next-themes";
import Editor from "@monaco-editor/react";
import { Skeleton } from "@/components/ui/skeleton";
import type { editor } from "monaco-editor";
import { useRef } from "react";

interface CodeEditorProps {
  code: string;
  onChange: (value: string) => void;
  language?: string;
  height?: string;
  readOnly?: boolean;
  markers?: editor.IMarkerData[];
}

export function CodeEditor({
  code,
  onChange,
  language = "cpp",
  height = "500px",
  readOnly = false,
  markers,
}: CodeEditorProps) {
  const { resolvedTheme } = useTheme();
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);

  function handleMount(editor: editor.IStandaloneCodeEditor) {
    editorRef.current = editor;
    if (markers) {
      const model = editor.getModel();
      if (model) {
        const monaco = (window as unknown as { monaco: typeof import("monaco-editor") }).monaco;
        if (monaco) {
          monaco.editor.setModelMarkers(model, "compilation", markers);
        }
      }
    }
  }

  return (
    <Editor
      height={height}
      language={language}
      value={code}
      onChange={(value) => onChange(value ?? "")}
      theme={resolvedTheme === "dark" ? "vs-dark" : "light"}
      onMount={handleMount}
      options={{
        readOnly,
        minimap: { enabled: true },
        fontSize: 13,
        lineNumbers: "on",
        scrollBeyondLastLine: false,
        wordWrap: "off",
        padding: { top: 8 },
        automaticLayout: true,
      }}
      loading={<Skeleton className="h-full w-full" />}
    />
  );
}
