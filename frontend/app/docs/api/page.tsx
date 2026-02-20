import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const endpoints = [
  {
    method: "GET",
    path: "/api/v1/health",
    description: "Server health check",
    response: `{
  "status": "ok",
  "version": "1.0.0"
}`,
  },
  {
    method: "GET",
    path: "/api/v1/modules",
    description: "List all registered modules with their metadata, ports, and default configuration",
    response: `{
  "modules": [
    {
      "type": "ADEmbedding",
      "category": "embedding",
      "description": "Token embedding lookup",
      "default_config": { "vocab_size": 256, "embed_dim": 64 },
      "inputs": [{ "name": "tokens", "type": "TOKEN_IDS", "optional": false }],
      "outputs": [{ "name": "output", "type": "AD_TENSOR", "optional": false }]
    },
    ...
  ]
}`,
  },
  {
    method: "POST",
    path: "/api/v1/execute_node",
    description: "Execute a single module with provided inputs and configuration",
    request: `{
  "type": "ADEmbedding",
  "config": { "vocab_size": 256, "embed_dim": 64 },
  "inputs": {
    "tokens": { "type": "TOKEN_IDS", "value": [1, 2, 3, 4, 5] }
  }
}`,
    response: `{
  "output": {
    "type": "AD_TENSOR",
    "shape": [64, 5],
    "data": [0.1, 0.2, ...],
    "stats": { "min": -0.5, "max": 0.5, "mean": 0.0, "std": 0.2 },
    "truncated": false
  }
}`,
  },
  {
    method: "POST",
    path: "/api/v1/execute",
    description: "Execute a computation graph with multiple connected nodes",
    request: `{
  "nodes": [
    { "id": "text_in", "type": "TextInput", "config": { "text": "Hello" } },
    { "id": "tok", "type": "Tokenizer", "config": {} }
  ],
  "edges": [
    {
      "source_node": "text_in",
      "source_port": "text",
      "target_node": "tok",
      "target_port": "text"
    }
  ]
}`,
    response: `{
  "node_results": [
    {
      "node_id": "text_in",
      "node_type": "TextInput",
      "execution_time_ms": 0.01,
      "outputs": { "text": { "type": "TEXT", "value": "Hello" } },
      "error": ""
    },
    ...
  ],
  "execution_order": ["text_in", "tok"],
  "total_time_ms": 1.23,
  "error": ""
}`,
  },
  {
    method: "GET",
    path: "/api/v1/presets",
    description: "Get preset graph templates for common pipelines",
    response: `{
  "presets": [
    {
      "name": "Embedding + Positional Encoding",
      "description": "TextInput -> Tokenizer -> Embedding -> PosEnc -> Add",
      "graph": { "nodes": [...], "edges": [...] }
    },
    ...
  ]
}`,
  },
];

export default function ApiReferencePage() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8">
      <h1 className="text-3xl font-bold mb-2">API Reference</h1>
      <p className="text-muted-foreground mb-8">
        Complete reference for the LLMs Unlocked REST API. All endpoints
        are prefixed with the server URL (default:{" "}
        <code className="text-sm">http://localhost:8080</code>).
      </p>

      <div className="space-y-6">
        {endpoints.map((ep) => (
          <Card key={ep.path} id={ep.path.replace(/\//g, "-")}>
            <CardHeader className="pb-3">
              <div className="flex items-center gap-3">
                <Badge
                  variant={ep.method === "GET" ? "secondary" : "default"}
                  className="font-mono"
                >
                  {ep.method}
                </Badge>
                <CardTitle className="text-base font-mono">
                  {ep.path}
                </CardTitle>
              </div>
              <p className="text-sm text-muted-foreground">
                {ep.description}
              </p>
            </CardHeader>
            <CardContent className="space-y-3">
              {"request" in ep && ep.request && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-1">
                    Request Body
                  </p>
                  <pre className="text-xs font-mono bg-muted p-3 rounded overflow-x-auto">
                    {ep.request}
                  </pre>
                </div>
              )}
              <div>
                <p className="text-xs font-medium text-muted-foreground mb-1">
                  Response
                </p>
                <pre className="text-xs font-mono bg-muted p-3 rounded overflow-x-auto">
                  {ep.response}
                </pre>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <section className="mt-12">
        <h2 className="text-xl font-semibold mb-4">Port Types</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {[
            { type: "TEXT", desc: "String value", color: "bg-blue-500" },
            { type: "TOKEN_IDS", desc: "Array of integers", color: "bg-green-500" },
            { type: "TENSOR", desc: "Raw 2D tensor", color: "bg-orange-500" },
            { type: "AD_TENSOR", desc: "Autodiff tensor with gradient", color: "bg-purple-500" },
            { type: "SCALAR", desc: "Single float", color: "bg-yellow-500" },
            { type: "INT", desc: "Single integer", color: "bg-gray-500" },
          ].map((pt) => (
            <Card key={pt.type}>
              <CardContent className="py-3 px-4 flex items-center gap-3">
                <span className={`h-3 w-3 rounded-full ${pt.color}`} />
                <div>
                  <p className="font-mono text-sm font-medium">{pt.type}</p>
                  <p className="text-xs text-muted-foreground">{pt.desc}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
