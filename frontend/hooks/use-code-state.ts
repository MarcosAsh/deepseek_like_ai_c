"use client";

import { useState, useCallback } from "react";

export function useCodeState(originalCode: string) {
  const [code, setCode] = useState(originalCode);
  const [compilationOutput, setCompilationOutput] = useState<string>("");
  const [executionOutput, setExecutionOutput] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const isModified = code !== originalCode;

  const reset = useCallback(() => {
    setCode(originalCode);
    setCompilationOutput("");
    setExecutionOutput("");
    setError(null);
  }, [originalCode]);

  const run = useCallback(async () => {
    setIsRunning(true);
    setError(null);
    setCompilationOutput("");
    setExecutionOutput("");

    try {
      const API_URL =
        process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
      const res = await fetch(`${API_URL}/api/v1/compile_and_run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_code: code }),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      }

      const data = await res.json();
      setCompilationOutput(data.compilation_output || "");
      setExecutionOutput(data.execution_output || "");
      if (data.error) setError(data.error);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to compile and run"
      );
    } finally {
      setIsRunning(false);
    }
  }, [code]);

  return {
    code,
    setCode,
    isModified,
    reset,
    run,
    isRunning,
    compilationOutput,
    executionOutput,
    error,
  };
}
