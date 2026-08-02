import { apiFetch } from "./client";

export interface LoginResponse {
  token: string;
  token_type: string;
  expires_at: string;
}

export interface MeResponse {
  authenticated: boolean;
  role: string;
}

export function login(password: string): Promise<LoginResponse> {
  return apiFetch<LoginResponse>("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password }),
  });
}

export function getMe(): Promise<MeResponse> {
  return apiFetch<MeResponse>("/api/auth/me");
}
