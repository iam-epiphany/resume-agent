import type { ReactNode } from "react";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { apiFetch } from "../api/client";

export type LoadLevel = "green" | "yellow" | "red";

export interface LoadSignals {
  cpu_ratio: number;
  mem_ratio: number;
  running: number;
  queued: number;
  in_flight: number;
  cores: number;
}

export interface LoadStatus {
  level: LoadLevel;
  signals?: LoadSignals | null;
}

export interface PublicQaStatus {
  ready: boolean;
  message: string;
  load?: LoadStatus;
}

interface ReadinessSnapshot extends PublicQaStatus {
  isLoading: boolean;
  error: string | null;
  refreshedAt: string | null;
}

interface ReadinessContextValue extends ReadinessSnapshot {
  /** 系统负载分级：green=正常 / yellow=较繁忙 / red=繁忙；未取到时为 null */
  loadLevel: LoadLevel | null;
  refresh: () => Promise<void>;
}

const ReadinessContext = createContext<ReadinessContextValue | null>(null);

/** 前台（匿名）就绪状态：轮询公开轻量接口 /api/qa/status，不暴露内部细节。 */
export function ReadinessProvider({ children }: { children: ReactNode }) {
  const [snapshot, setSnapshot] = useState<ReadinessSnapshot>({
    ready: false,
    message: "正在检查系统状态…",
    isLoading: true,
    error: null,
    refreshedAt: null,
  });

  const refresh = useCallback(async () => {
    setSnapshot((current) => ({ ...current, isLoading: true }));
    try {
      const status = await apiFetch<PublicQaStatus>("/api/qa/status");
      setSnapshot({
        ready: status.ready,
        message: status.message,
        load: status.load,
        isLoading: false,
        error: null,
        refreshedAt: new Date().toISOString(),
      });
    } catch (error) {
      setSnapshot((current) => ({
        ...current,
        isLoading: false,
        error: error instanceof Error ? error.message : "系统状态暂时无法读取。",
      }));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 30_000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const value = useMemo<ReadinessContextValue>(
    () => ({ ...snapshot, loadLevel: snapshot.load?.level ?? null, refresh }),
    [refresh, snapshot],
  );
  return <ReadinessContext.Provider value={value}>{children}</ReadinessContext.Provider>;
}

export function useReadiness(): ReadinessContextValue {
  const context = useContext(ReadinessContext);
  if (!context) {
    throw new Error("useReadiness must be used inside ReadinessProvider");
  }
  return context;
}
