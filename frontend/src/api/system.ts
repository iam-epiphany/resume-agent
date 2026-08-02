import { apiFetch } from "./client";
import type { RagHealthResponse } from "../types/api";

export function getRagHealth(): Promise<RagHealthResponse> {
  return apiFetch<RagHealthResponse>("/api/health/rag");
}
