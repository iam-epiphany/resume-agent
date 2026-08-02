import type { ApiErrorBody } from "../types/api";

export class ApiError extends Error {
  readonly status: number;
  readonly body: ApiErrorBody | null;

  constructor(status: number, body: ApiErrorBody | null) {
    const message = body?.error?.message ?? body?.message ?? body?.detail ?? `HTTP ${status}`;
    super(message);
    this.status = status;
    this.body = body;
  }
}

/** 管理员登录 token 的 localStorage 键名。 */
export const AUTH_TOKEN_KEY = "resumemind.auth.token";

/** 登录失效事件：任何 API 401 时广播，AuthContext 监听后清除登录态。 */
export const AUTH_EXPIRED_EVENT = "auth:expired";

export function getAuthToken(): string | null {
  try {
    return localStorage.getItem(AUTH_TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setAuthToken(token: string): void {
  try {
    localStorage.setItem(AUTH_TOKEN_KEY, token);
  } catch {
    // localStorage 不可用（隐私模式等）时静默失败，本次会话内仍可用
  }
}

export function clearAuthToken(): void {
  try {
    localStorage.removeItem(AUTH_TOKEN_KEY);
  } catch {
    // ignore
  }
}

export async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const headers = new Headers(options?.headers);
  const token = getAuthToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(path, { ...options, headers });
  const body = await readJson(response);

  if (!response.ok) {
    // 登录过期：清除本地 token 并广播，让界面回到匿名态
    if (response.status === 401 && !path.startsWith("/api/auth/")) {
      clearAuthToken();
      window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT));
    }
    throw new ApiError(response.status, body as ApiErrorBody | null);
  }

  return body as T;
}

async function readJson(response: Response): Promise<unknown | null> {
  const text = await response.text();
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text) as unknown;
  } catch {
    return { detail: text };
  }
}
