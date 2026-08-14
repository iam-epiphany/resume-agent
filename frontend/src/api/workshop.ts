import { apiFetch, getAuthToken } from "./client";

/** 人物工坊 / 人物档案 API 客户端（管理员）。 */

export interface PersonaPublic {
  persona_id: string;
  name: string;
  display_name: string;
  status: string;
  profile_summary: string;
  is_active: boolean;
}

export interface WorkshopJobView {
  job_id: string;
  persona_id: string;
  status: string;
  stage: string;
  raw_filenames: string[];
  generated_document_ids: string[];
  generated_fact_count: number;
  llm_call_count: number;
  error: string | null;
  created_at: string | null;
  completed_at: string | null;
}

export function getActivePersona(): Promise<PersonaPublic> {
  return apiFetch<PersonaPublic>("/api/personas/active");
}

export function listPersonas(): Promise<PersonaPublic[]> {
  return apiFetch<PersonaPublic[]>("/api/personas");
}

export function activatePersona(personaId: string): Promise<PersonaPublic> {
  return apiFetch<PersonaPublic>(`/api/personas/${personaId}/activate`, {
    method: "POST",
  });
}

export function confirmPersona(
  personaId: string,
  profile: Record<string, unknown>,
): Promise<PersonaPublic> {
  return apiFetch<PersonaPublic>(`/api/personas/${personaId}`, {
    method: "PATCH",
    body: JSON.stringify({ profile, confirm: true }),
  });
}

export function listWorkshopJobs(): Promise<WorkshopJobView[]> {
  return apiFetch<WorkshopJobView[]>("/api/workshop/jobs");
}

export function rollbackWorkshopJob(jobId: string): Promise<{ job_id: string; status: string }> {
  return apiFetch<{ job_id: string; status: string }>(`/api/workshop/jobs/${jobId}/rollback`, {
    method: "POST",
  });
}

/** XHR 上传（multipart 需要手动注入 token，与 documents 上传一致）。 */
export function transformMaterials(files: File[]): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    for (const file of files) {
      form.append("files", file);
    }
    const request = new XMLHttpRequest();
    request.open("POST", "/api/workshop/transform");
    const token = getAuthToken();
    if (token) {
      request.setRequestHeader("Authorization", `Bearer ${token}`);
    }
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) {
        try {
          resolve(JSON.parse(request.responseText));
        } catch {
          resolve({});
        }
      } else {
        reject(new Error(`转换失败（HTTP ${request.status}）：${request.responseText}`));
      }
    };
    request.onerror = () => reject(new Error("网络错误，转换失败"));
    request.send(form);
  });
}
