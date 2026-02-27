import type {
  ModulesResponse,
  ExecuteNodeRequest,
  ExecuteNodeResponse,
  GraphDef,
  GraphResult,
  GraphResultRaw,
  PresetsResponse,
  HealthResponse,
} from "./types";
import { normalizeGraphResult } from "./types";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";

async function fetchAPI<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const text = await res.text();
    const message = text.length > 200 ? text.slice(0, 200) + "..." : text;
    throw new Error(`Request failed (${res.status}): ${message}`);
  }
  return res.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  return fetchAPI<HealthResponse>("/api/v1/health");
}

export async function fetchModules(): Promise<ModulesResponse> {
  return fetchAPI<ModulesResponse>("/api/v1/modules");
}

export async function executeNode(
  req: ExecuteNodeRequest
): Promise<ExecuteNodeResponse> {
  return fetchAPI<ExecuteNodeResponse>("/api/v1/execute_node", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function executeGraph(
  graph: GraphDef
): Promise<GraphResult> {
  const raw = await fetchAPI<GraphResultRaw>("/api/v1/execute", {
    method: "POST",
    body: JSON.stringify(graph),
  });
  return normalizeGraphResult(raw);
}

export async function fetchPresets(): Promise<PresetsResponse> {
  return fetchAPI<PresetsResponse>("/api/v1/presets");
}
