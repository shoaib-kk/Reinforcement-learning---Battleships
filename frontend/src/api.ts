// REST client for the FastAPI control plane (server/main.py).

import type {
  ArchLayer,
  EmbeddingData,
  EpisodeRecord,
  LayerHistograms,
  StepPayload,
  TrainStatus,
  Viz3DStatic,
} from "./types";

async function post<T>(url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${url}: ${res.status} ${await res.text()}`);
  return res.json();
}

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

export const api = {
  trainStart: (cfg: Record<string, unknown>) => post<TrainStatus>("/api/train/start", cfg),
  trainStop: () => post<TrainStatus>("/api/train/stop"),
  trainStatus: () => get<TrainStatus>("/api/train/status"),
  streamConfig: (cfg: { viz_every?: number; viz3d?: boolean; viz3d_every?: number }) =>
    post("/api/stream/config", cfg),
  viz3dStatic: (model: "train" | "demo") =>
    get<Viz3DStatic>(`/api/viz3d/static?model=${model}`),
  metrics: (after = 0) =>
    get<{ total: number; records: EpisodeRecord[] }>(`/api/metrics?after=${after}`),
  checkpoints: () =>
    get<{ checkpoints: { file: string; mtime: number; bytes: number }[] }>("/api/checkpoints"),
  architecture: () => get<{ layers: ArchLayer[] }>("/api/model/architecture"),
  weights: () => get<{ layers: LayerHistograms[] }>("/api/introspection/weights"),
  embeddings: (n = 400) => get<EmbeddingData>(`/api/introspection/embeddings?n=${n}`),
  demoReset: (source: string, opponent: string) =>
    post<{ board: StepPayload["board"]; source: string }>("/api/demo/reset", {
      source,
      opponent,
    }),
  demoStep: () => post<StepPayload>("/api/demo/step"),
};
