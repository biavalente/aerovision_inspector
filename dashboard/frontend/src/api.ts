import type {
  BoundingBox,
  Dashboard,
  HumanDecision,
  Inspection,
  TrainingQueue,
} from "./types";

type TrainingTrigger = {
  version_name: string;
  candidate_count: number;
  model_type: string;
  training_target: string;
};

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: { ...(options?.headers || {}) },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Erro ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),
  dashboard: () => request<Dashboard>("/api/dashboard"),
  inspections: () => request<{ items: Inspection[] }>("/api/inspections?limit=500"),
  seed: () => request<{ created: number; available: number; message: string }>("/api/inspections/seed", { method: "POST" }),
  analyse: (id: number) => request<Inspection>(`/api/inspections/${id}/analyse`, { method: "POST" }),
  review: (id: number, payload: { decision: HumanDecision; reason: string; actor: string; boxes: BoundingBox[] }) =>
    request<{ inspection: Inspection; training_trigger?: TrainingTrigger | null; training_triggers?: TrainingTrigger[] }>(
      `/api/inspections/${id}/review`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) },
    ),
  batchDecision: (payload: { inspection_ids: number[]; decision: HumanDecision; reason: string; actor: string }) =>
    request<{ updated_ids: number[]; missing_ids: number[]; message: string; training_triggers?: TrainingTrigger[] }>("/api/inspections/batch-decision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  history: (id: number) =>
    request<{ events: import("./types").AuditEvent[]; annotation_revisions: import("./types").AnnotationRevision[] }>(
      `/api/inspections/${id}/history`,
    ),
  trainingQueue: () => request<TrainingQueue>("/api/training/queue"),
  triggerTraining: () => request<{ triggered: boolean; queue: TrainingQueue }>("/api/training/trigger", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }),
  attributions: () => request<{ markdown: string }>("/api/attributions"),
  upload: async (file: File) => {
    const body = new FormData();
    body.append("image", file);
    return request<Inspection>("/api/inspections/upload", { method: "POST", body });
  },
};
