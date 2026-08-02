import { apiFetch, getAuthToken } from "./client";

/**
 * XMLHttpRequest 不经过 apiFetch，需单独注入管理员 token。
 * 漏掉此头时上传接口会返回 401。
 */
function injectAuthHeader(request: XMLHttpRequest): void {
  const token = getAuthToken();
  if (token) {
    request.setRequestHeader("Authorization", `Bearer ${token}`);
  }
}
import type {
  DocumentBatchUploadResponse,
  DocumentBulkDeleteResponse,
  DocumentDeleteResponse,
  DocumentDetailResponse,
  DocumentListResponse,
  DocumentMetadataPatch,
  DocumentMetadataUpdateResponse,
  DocumentProcessingResponse,
  DocumentUploadPreflightRequestItem,
  DocumentUploadPreflightResponse,
  DocumentUploadResponse,
} from "../types/api";

export function listDocuments(): Promise<DocumentListResponse> {
  return apiFetch<DocumentListResponse>("/api/documents");
}

export function getDocument(documentId: string, chunkOffset = 0, chunkLimit = 50): Promise<DocumentDetailResponse> {
  const params = new URLSearchParams({
    chunk_offset: String(chunkOffset),
    chunk_limit: String(chunkLimit),
  });
  return apiFetch<DocumentDetailResponse>(`/api/documents/${documentId}?${params.toString()}`);
}

export function rebuildDocumentIndex(documentId: string): Promise<DocumentDetailResponse> {
  return apiFetch<DocumentDetailResponse>(`/api/documents/${documentId}/index`, {
    method: "POST",
  });
}

export function updateDocumentMetadata(
  documentId: string,
  metadata: DocumentMetadataPatch,
): Promise<DocumentMetadataUpdateResponse> {
  return apiFetch<DocumentMetadataUpdateResponse>(`/api/documents/${documentId}/metadata`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(metadata),
  });
}

export function confirmDocumentMetadata(documentId: string): Promise<DocumentMetadataUpdateResponse> {
  return apiFetch<DocumentMetadataUpdateResponse>(`/api/documents/${documentId}/metadata/confirm`, {
    method: "POST",
  });
}

export function uploadDocument(
  file: File,
  options: {
    idempotencyKey: string;
    filenameOverride?: string;
    overwriteDocumentId?: string;
    metadata?: Record<string, unknown>;
    onProgress?: (loaded: number, total: number) => void;
  },
): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (options.filenameOverride) {
    formData.append("filename_override", options.filenameOverride);
  }
  if (options.overwriteDocumentId) {
    formData.append("overwrite_document_id", options.overwriteDocumentId);
  }
  if (options.metadata && Object.keys(options.metadata).length > 0) {
    formData.append("metadata_json", JSON.stringify(options.metadata));
  }
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/documents/upload");
    request.setRequestHeader("Idempotency-Key", options.idempotencyKey);
    injectAuthHeader(request);
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        options.onProgress?.(event.loaded, event.total);
      }
    });
    request.addEventListener("load", () => {
      let body: unknown = null;
      try {
        body = JSON.parse(request.responseText);
      } catch {
        // The shared API client uses the same user-facing fallback below.
      }
      if (request.status >= 200 && request.status < 300) {
        resolve(body as DocumentUploadResponse);
        return;
      }
      const errorBody = body as { detail?: string; error?: { message?: string } } | null;
      reject(new Error(errorBody?.error?.message ?? errorBody?.detail ?? `HTTP ${request.status}`));
    });
    request.addEventListener("error", () => reject(new Error("上传连接中断，请使用同一文件重试。")));
    request.send(formData);
  });
}

export function preflightDocumentUploads(
  items: DocumentUploadPreflightRequestItem[],
): Promise<DocumentUploadPreflightResponse> {
  return apiFetch<DocumentUploadPreflightResponse>("/api/documents/upload-preflight", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
  });
}

export function uploadDocumentsBatch(
  files: File[],
  options: { idempotencyKey: string; onProgress?: (loaded: number, total: number) => void },
): Promise<DocumentBatchUploadResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/documents/batch-upload");
    request.setRequestHeader("Idempotency-Key", options.idempotencyKey);
    injectAuthHeader(request);
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        options.onProgress?.(event.loaded, event.total);
      }
    });
    request.addEventListener("load", () => {
      let body: unknown = null;
      try {
        body = JSON.parse(request.responseText);
      } catch {
        // The shared API client uses the same user-facing fallback below.
      }
      if (request.status >= 200 && request.status < 300) {
        resolve(body as DocumentBatchUploadResponse);
        return;
      }
      const errorBody = body as { detail?: string; error?: { message?: string } } | null;
      reject(new Error(errorBody?.error?.message ?? errorBody?.detail ?? `HTTP ${request.status}`));
    });
    request.addEventListener("error", () => reject(new Error("批量上传连接中断，请使用同一批文件重试。")));
    request.send(formData);
  });
}

export function getDocumentProcessing(documentId: string): Promise<DocumentProcessingResponse> {
  return apiFetch<DocumentProcessingResponse>(`/api/documents/${documentId}/processing`);
}

export function deleteDocument(documentId: string, signal?: AbortSignal): Promise<DocumentDeleteResponse> {
  return apiFetch<DocumentDeleteResponse>(`/api/documents/${documentId}`, {
    method: "DELETE",
    signal,
  });
}

export function deleteDocumentsBulk(documentIds: string[]): Promise<DocumentBulkDeleteResponse> {
  return apiFetch<DocumentBulkDeleteResponse>("/api/documents/bulk-delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_ids: documentIds }),
  });
}
