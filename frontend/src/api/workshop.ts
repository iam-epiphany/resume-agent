import {
  AUTH_EXPIRED_EVENT,
  ApiError,
  apiFetch,
  clearAuthToken,
  getAuthToken,
} from "./client";
import type { ApiErrorBody } from "../types/api";

/** 人物工坊 / 人物档案 API 客户端（管理员）。 */

export interface SkillPackageInfo {
  /** 人物 Skill 包元信息（不含内容；完整包内容走管理员下载端点）。 */
  file_count: number;
  skill_version: string | null;
  generated_at: string | null;
}

export interface PersonaPublic {
  persona_id: string;
  name: string;
  display_name: string;
  status: string;
  profile_summary: string;
  is_active: boolean;
  skill_package: SkillPackageInfo | null;
}

export interface WorkshopJobView {
  job_id: string;
  persona_id: string;
  status: string;
  stage: string;
  skill_version: string | null;
  raw_filenames: string[];
  generated_document_ids: string[];
  generated_fact_count: number;
  llm_call_count: number;
  /** Reduce 归并冲突清单（同名异内容/事实多值冲突/重复合并），空数组 = 无冲突 */
  conflicts: string[];
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

/** 下载人物 Skill 包（zip，管理员）。返回下载文件名（取自 Content-Disposition）。 */
export async function downloadPersonaSkillPackage(personaId: string): Promise<string> {
  const token = getAuthToken();
  const response = await fetch(`/api/personas/${personaId}/skill-package`, {
    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
  });
  if (!response.ok) {
    let body: ApiErrorBody | null = null;
    try {
      body = (await response.json()) as ApiErrorBody;
    } catch {
      // 非 JSON 错误体（网关等）忽略
    }
    if (response.status === 401) {
      clearAuthToken();
      window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT));
    }
    throw new ApiError(response.status, body);
  }
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const match = disposition.match(/filename\*=UTF-8''([^;]+)/);
  const filename = match ? decodeURIComponent(match[1]) : "persona-skill.zip";
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  } finally {
    URL.revokeObjectURL(url);
  }
  return filename;
}
