import { apiFetch } from "./client";
import type {
  AuditArchiveDeleteResponse,
  AuditArchiveDetailResponse,
  AuditArchiveListResponse,
  AuditLogListResponse,
} from "../types/api";

export function listAuditLogs(): Promise<AuditLogListResponse> {
  return apiFetch<AuditLogListResponse>("/api/audit/logs");
}

export function listAuditArchives(): Promise<AuditArchiveListResponse> {
  return apiFetch<AuditArchiveListResponse>("/api/audit/archives");
}

export function getAuditArchive(date: string): Promise<AuditArchiveDetailResponse> {
  return apiFetch<AuditArchiveDetailResponse>(`/api/audit/archives/${date}`);
}

export function deleteAuditArchive(date: string): Promise<AuditArchiveDeleteResponse> {
  return apiFetch<AuditArchiveDeleteResponse>(`/api/audit/archives/${date}`, {
    method: "DELETE",
  });
}
