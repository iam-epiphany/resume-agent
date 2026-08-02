import type { ReactNode } from "react";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { listDocuments } from "../api/documents";
import { getRagHealth } from "../api/system";
import { useAuth } from "./authContext";
import type { DocumentSummary, RagHealthResponse } from "../types/api";

interface SystemStatusSnapshot {
  ragHealth: RagHealthResponse | null;
  documentCount: number;
  indexedDocumentCount: number;
  processingDocumentCount: number;
  isLoading: boolean;
  error: string | null;
  refreshedAt: string | null;
}

interface SystemStatusContextValue extends SystemStatusSnapshot {
  refresh: () => Promise<void>;
}

const SystemStatusContext = createContext<SystemStatusContextValue | null>(null);

export function SystemStatusProvider({ children }: { children: ReactNode }) {
  const { isAuthenticated } = useAuth();
  const [snapshot, setSnapshot] = useState<SystemStatusSnapshot>({
    ragHealth: null,
    documentCount: 0,
    indexedDocumentCount: 0,
    processingDocumentCount: 0,
    isLoading: true,
    error: null,
    refreshedAt: null,
  });

  const refresh = useCallback(async () => {
    setSnapshot((current) => ({ ...current, isLoading: true }));
    try {
      const [ragHealth, documentResult] = await Promise.all([getRagHealth(), listDocuments()]);
      setSnapshot({
        ragHealth,
        documentCount: documentResult.documents.length,
        indexedDocumentCount: countIndexedDocuments(documentResult.documents),
        processingDocumentCount: countProcessingDocuments(documentResult.documents),
        isLoading: false,
        error: null,
        refreshedAt: new Date().toISOString(),
      });
    } catch (error) {
      setSnapshot((current) => ({
        ...current,
        isLoading: false,
        error: error instanceof Error ? error.message : "系统状态暂时无法读取。",
        refreshedAt: new Date().toISOString(),
      }));
    }
  }, []);

  // 系统状态属后台信息（/health/rag 与文档列表均已收进管理员后台）：
  // 仅登录后轮询，匿名时不发起请求（避免匿名 401；前台用公开 /api/qa/status）
  useEffect(() => {
    if (!isAuthenticated) return;
    void refresh();
    const timer = window.setInterval(() => void refresh(), 30_000);
    return () => window.clearInterval(timer);
  }, [isAuthenticated, refresh]);

  const value = useMemo<SystemStatusContextValue>(() => ({ ...snapshot, refresh }), [refresh, snapshot]);
  return <SystemStatusContext.Provider value={value}>{children}</SystemStatusContext.Provider>;
}

export function useSystemStatus(): SystemStatusContextValue {
  const context = useContext(SystemStatusContext);
  if (!context) {
    throw new Error("useSystemStatus must be used inside SystemStatusProvider");
  }
  return context;
}

export function isKnowledgeBaseReady(snapshot: Pick<SystemStatusSnapshot, "ragHealth" | "indexedDocumentCount">): boolean {
  return Boolean(snapshot.ragHealth?.ready) && snapshot.indexedDocumentCount > 0;
}

export function isKnowledgeBaseAvailable(snapshot: Pick<SystemStatusSnapshot, "ragHealth">): boolean {
  const health = snapshot.ragHealth;
  return Boolean(health?.sqlite_ready && health.qdrant_ready);
}

function countIndexedDocuments(documents: DocumentSummary[]): number {
  return documents.filter((document) => document.status === "indexed").length;
}

function countProcessingDocuments(documents: DocumentSummary[]): number {
  return documents.filter((document) => ["uploaded", "index_queued", "indexing", "deleting"].includes(document.status)).length;
}
