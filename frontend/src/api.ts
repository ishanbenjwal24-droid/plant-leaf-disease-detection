const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export type ClassScore = {
  class_name: string;
  display_name: string;
  score: number;
};

export type Prediction = {
  model: "ml" | "cnn";
  predicted_class: string;
  display_name: string;
  class_score: number;
  top_classes: ClassScore[];
  inference_ms: number;
  model_version: string | null;
};

export type ModelMetrics = {
  available: boolean;
  metrics: {
    accuracy: number;
    macro_precision: number;
    macro_recall: number;
    macro_f1: number;
    average_inference_ms_per_image: number;
    model_size_bytes: number;
    classification_report: Record<string, Record<string, number>>;
    confusion_matrix: number[][];
    class_names: string[];
  } | null;
};

export type MetricsResponse = {
  models: { ml: ModelMetrics; cnn: ModelMetrics };
  comparison: {
    accuracy_difference_cnn_minus_ml: number;
    macro_f1_difference_cnn_minus_ml: number;
  } | null;
};

export type HistoryItem = {
  id: number;
  created_at: string;
  mode: string;
  predictions: Prediction[];
  latency_ms: number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      // Keep the status-based message when the response is not JSON.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export function compareModels(file: File): Promise<{ ml: Prediction; cnn: Prediction; total_inference_ms: number }> {
  const form = new FormData();
  form.append("file", file);
  return request("/api/compare", { method: "POST", body: form });
}

export function getMetrics(): Promise<MetricsResponse> {
  return request("/api/metrics");
}

export function getHistory(): Promise<HistoryItem[]> {
  return request("/api/history?limit=100");
}

export function clearHistory(): Promise<{ deleted: number }> {
  return request("/api/history", { method: "DELETE" });
}
