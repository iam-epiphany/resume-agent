import { apiFetch } from "./client";
import type {
  QATaskCreateResponse,
  QATaskRequest,
  QATaskStatusResponse,
} from "../types/api";

/** 访客问答访问闸与预算状态（/api/qa/access/status）。 */
export interface QAAccessStatus {
  access_enabled: boolean;
  granted: boolean;
  daily_used: number;
  daily_remaining: number;
  daily_limit: number;
  budget_warning: boolean;
}

/** 访客访问码登录（校验通过后由服务端签发 httpOnly cookie，前端无需存储 token）。 */
export function qaAccessSubmit(code: string): Promise<{ granted: boolean; expires_at?: string }> {
  return apiFetch<{ granted: boolean; expires_at?: string }>("/api/qa/access", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code }),
  });
}

/** 查询访问闸与预算状态：决定显示访问码门还是问答界面，并轮询预算提醒。 */
export function qaAccessStatus(): Promise<QAAccessStatus> {
  return apiFetch<QAAccessStatus>("/api/qa/access/status");
}

export function createQuestionTask(
  question: string,
  clientRequestId: string,
  includeDebug = false,
  sessionId: string | null = null,
): Promise<QATaskCreateResponse> {
  const body: QATaskRequest = {
    question,
    client_request_id: clientRequestId,
    options: [],
    include_debug: includeDebug,
    session_id: sessionId,
  };
  return apiFetch<QATaskCreateResponse>("/api/qa/tasks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getQuestionTask(taskId: string): Promise<QATaskStatusResponse> {
  return apiFetch<QATaskStatusResponse>(`/api/qa/tasks/${taskId}`);
}

export function streamQuestionTask(
  taskId: string,
  onTask: (task: QATaskStatusResponse) => void,
  onConnectionChange: (connected: boolean) => void,
): () => void {
  if (typeof EventSource === "undefined") {
    onConnectionChange(false);
    return () => undefined;
  }
  const source = new EventSource(`/api/qa/tasks/${taskId}/stream`);
  source.onopen = () => onConnectionChange(true);
  source.addEventListener("task", (event) => {
    try {
      onTask(JSON.parse((event as MessageEvent<string>).data) as QATaskStatusResponse);
    } catch {
      onConnectionChange(false);
    }
  });
  source.onerror = () => onConnectionChange(false);
  return () => source.close();
}

export function cancelQuestionTask(taskId: string): Promise<QATaskStatusResponse> {
  return apiFetch<QATaskStatusResponse>(`/api/qa/tasks/${taskId}/cancel`, {
    method: "POST",
  });
}

export function listQuestionTasks(limit = 5): Promise<QATaskStatusResponse[]> {
  return apiFetch<QATaskStatusResponse[]>(`/api/qa/tasks?limit=${limit}`);
}
