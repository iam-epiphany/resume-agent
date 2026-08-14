import type { ReactNode } from "react";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  answerMode?: string;
  status: "streaming" | "done" | "error" | "cancelled";
}

export interface ChatThread {
  id: string;
  title: string;
  sessionId: string;
  messages: ChatMessage[];
  updatedAt: number;
}

type ThreadPatch = Partial<ChatThread> | ((thread: ChatThread) => Partial<ChatThread>);

interface ChatHistoryValue {
  threads: ChatThread[];
  activeThreadId: string | null;
  activeThread: ChatThread | null;
  createThread: () => string;
  switchThread: (threadId: string) => void;
  updateActiveThread: (patch: ThreadPatch) => void;
  updateThread: (threadId: string, patch: ThreadPatch) => void;
  deleteThread: (threadId: string) => void;
}

const STORAGE_KEY = "resumemind.qa.threads.v1";
const MAX_THREADS = 30;
const MAX_MESSAGES_PER_THREAD = 80;

const ChatHistoryContext = createContext<ChatHistoryValue | null>(null);

function createId(): string {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID().replaceAll("-", "");
  return `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function newThread(): ChatThread {
  return {
    id: createId(),
    title: "新对话",
    sessionId: createId(),
    messages: [],
    updatedAt: Date.now(),
  };
}

interface StoredState {
  threads: ChatThread[];
  activeThreadId: string | null;
}

function loadState(): StoredState {
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return { threads: [], activeThreadId: null };
    const parsed = JSON.parse(raw) as Partial<StoredState>;
    const threads = Array.isArray(parsed.threads)
      ? parsed.threads
          .filter(
            (item) =>
              item &&
              typeof item.id === "string" &&
              typeof item.sessionId === "string" &&
              Array.isArray(item.messages),
          )
          .slice(-MAX_THREADS)
      : [];
    const activeThreadId =
      typeof parsed.activeThreadId === "string" &&
      threads.some((thread) => thread.id === parsed.activeThreadId)
        ? parsed.activeThreadId
        : threads.length > 0
          ? threads[threads.length - 1].id
          : null;
    return { threads, activeThreadId };
  } catch {
    return { threads: [], activeThreadId: null };
  }
}

function createInitialState(): StoredState {
  const loaded = loadState();
  if (loaded.threads.length > 0) return loaded;
  const thread = newThread();
  return { threads: [thread], activeThreadId: thread.id };
}

export function ChatHistoryProvider({ children }: { children: ReactNode }) {
  const [initial] = useState<StoredState>(createInitialState);
  const [threads, setThreads] = useState<ChatThread[]>(initial.threads);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(initial.activeThreadId);

  // 会话级持久化（sessionStorage）：刷新保留，关闭页面即清除
  useEffect(() => {
    try {
      window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ threads, activeThreadId } satisfies StoredState));
    } catch {
      // 存储不可用时仅失去页面级恢复能力
    }
  }, [threads, activeThreadId]);

  const createThread = useCallback((): string => {
    const thread = newThread();
    setThreads((current) => [...current.slice(-(MAX_THREADS - 1)), thread]);
    setActiveThreadId(thread.id);
    return thread.id;
  }, []);

  const switchThread = useCallback((threadId: string) => {
    setActiveThreadId(threadId);
  }, []);

  const applyThreadPatch = useCallback(
    (threadId: string, patch: ThreadPatch) => {
      setThreads((current) =>
        current.map((thread) => {
          if (thread.id !== threadId) return thread;
          const resolved = typeof patch === "function" ? patch(thread) : patch;
          return {
            ...thread,
            ...resolved,
            updatedAt: Date.now(),
            messages: resolved.messages
              ? resolved.messages.slice(-MAX_MESSAGES_PER_THREAD)
              : thread.messages,
          };
        }),
      );
    },
    [],
  );

  const updateActiveThread = useCallback(
    (patch: ThreadPatch) => {
      if (activeThreadId === null) return;
      applyThreadPatch(activeThreadId, patch);
    },
    [activeThreadId, applyThreadPatch],
  );

  const updateThread = useCallback(
    (threadId: string, patch: ThreadPatch) => {
      applyThreadPatch(threadId, patch);
    },
    [applyThreadPatch],
  );

  const deleteThread = useCallback((threadId: string) => {
    setThreads((current) => {
      const next = current.filter((thread) => thread.id !== threadId);
      if (threadId === activeThreadId) {
        // 删除当前线程后：无剩余线程则自动新建，保证始终有可用对话
        if (next.length === 0) {
          const thread = newThread();
          setActiveThreadId(thread.id);
          return [thread];
        }
        setActiveThreadId(next[next.length - 1].id);
      }
      return next;
    });
  }, [activeThreadId]);

  const activeThread = useMemo(
    () => threads.find((thread) => thread.id === activeThreadId) ?? null,
    [threads, activeThreadId],
  );

  const value = useMemo<ChatHistoryValue>(
    () => ({
      threads,
      activeThreadId,
      activeThread,
      createThread,
      switchThread,
      updateActiveThread,
      updateThread,
      deleteThread,
    }),
    [activeThread, activeThreadId, createThread, deleteThread, switchThread, threads, updateActiveThread, updateThread],
  );

  return <ChatHistoryContext.Provider value={value}>{children}</ChatHistoryContext.Provider>;
}

export function useChatHistory() {
  const context = useContext(ChatHistoryContext);
  if (!context) throw new Error("useChatHistory must be used inside ChatHistoryProvider");
  return context;
}
