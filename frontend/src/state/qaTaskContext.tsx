import type { ReactNode } from "react";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

import { cancelQuestionTask, createQuestionTask, getQuestionTask, streamQuestionTask } from "../api/qa";
import type { QAAnswerPreview, QAResponse, QATaskStatusResponse, RagProgressEvent } from "../types/api";

const QA_PENDING_RECEIPT_KEY = "resumemind.qa.pending-receipt";
const QA_SESSION_ID_KEY = "resumemind.qa.session-id";
const QA_TASK_POLL_INTERVAL_MS = 1000;
const QA_RETRY_DELAYS_MS = [1000, 2000, 4000, 8000, 10000];

interface QATaskSnapshot {
  taskId: string | null;
  clientRequestId: string | null;
  currentQuestion: string;
  options: string[];
  includeDebug: boolean;
  taskStatus: string;
  answerPreview: QAAnswerPreview | null;
  answer: QAResponse | null;
  message: string;
  progressEvents: RagProgressEvent[];
  updatedAt: string | null;
  createdAt: string | null;
  completedAt: string | null;
}

interface PendingReceipt {
  clientRequestId: string;
  question: string;
  includeDebug: boolean;
}

interface QATaskContextValue extends QATaskSnapshot {
  draftQuestion: string;
  sessionId: string;
  setSessionId: (value: string) => void;
  isSubmitting: boolean;
  isCancelling: boolean;
  setDraftQuestion: (value: string) => void;
  runQuestion: (value?: string, includeDebug?: boolean) => Promise<void>;
  cancelQuestion: () => Promise<void>;
  reloadTask: () => Promise<void>;
  restoreTask: (taskId: string) => Promise<void>;
  clearMessage: () => void;
}

const QATaskContext = createContext<QATaskContextValue | null>(null);

const emptyTaskSnapshot: QATaskSnapshot = {
  taskId: null,
  clientRequestId: null,
  currentQuestion: "",
  options: [],
  includeDebug: false,
  taskStatus: "",
  answerPreview: null,
  answer: null,
  message: "",
  progressEvents: [],
  updatedAt: null,
  createdAt: null,
  completedAt: null,
};

export function QATaskProvider({ children }: { children: ReactNode }) {
  const [snapshot, setSnapshot] = useState<QATaskSnapshot>(emptyTaskSnapshot);
  const [draftQuestion, setDraftQuestion] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [streamConnected, setStreamConnected] = useState(false);
  const [sessionId, setSessionId] = useState(getOrCreateSessionId);
  const activeTaskId = useRef<string | null>(null);
  const cancelRequested = useRef(false);
  const reconciliationStarted = useRef(false);

  useEffect(() => {
    activeTaskId.current = snapshot.taskId;
  }, [snapshot.taskId]);

  const applyTaskSnapshot = useCallback((task: QATaskStatusResponse | null | undefined) => {
    if (!task) {
      setSnapshot((current) => ({ ...current, message: "任务状态返回为空，正在重新读取。" }));
      return;
    }
    activeTaskId.current = task.task_id;
    setSnapshot((current) => ({
      ...current,
      taskId: task.task_id,
      clientRequestId: task.client_request_id ?? current.clientRequestId,
      currentQuestion: task.question,
      options: task.options,
      includeDebug: task.include_debug,
      taskStatus: task.status,
      answerPreview: task.answer_preview ?? null,
      answer: task.answer,
      message: task.error?.message ?? (task.status === "failed" ? current.message : ""),
      progressEvents: task.progress_events,
      updatedAt: task.updated_at,
      createdAt: task.created_at,
      completedAt: task.completed_at,
    }));
    const taskActive = isActiveTaskStatus(task.status);
    setIsSubmitting(taskActive);
    if (!taskActive) {
      cancelRequested.current = false;
      setIsCancelling(false);
    }
  }, []);

  // A full page refresh deliberately starts with an empty work area. The only
  // browser-persisted state is an uncertain POST receipt, reconciled silently so
  // an accepted task is neither lost nor duplicated in server-side history.
  useEffect(() => {
    if (reconciliationStarted.current) return;
    reconciliationStarted.current = true;
    const receipt = loadPendingReceipt();
    if (!receipt) return;
    const pendingReceipt = receipt;

    let stopped = false;
    let timer: number | undefined;
    let retryIndex = 0;

    async function reconcilePendingReceipt() {
      try {
        await createQuestionTask(
          pendingReceipt.question,
          pendingReceipt.clientRequestId,
          pendingReceipt.includeDebug,
          sessionId,
        );
        if (!stopped) clearPendingReceipt(pendingReceipt.clientRequestId);
      } catch {
        if (stopped) return;
        const delay = QA_RETRY_DELAYS_MS[Math.min(retryIndex, QA_RETRY_DELAYS_MS.length - 1)];
        retryIndex += 1;
        timer = window.setTimeout(reconcilePendingReceipt, delay);
      }
    }

    void reconcilePendingReceipt();
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [sessionId]);

  useEffect(() => {
    if (snapshot.taskId || !snapshot.clientRequestId || snapshot.taskStatus !== "queued") return;

    let stopped = false;
    let timer: number | undefined;
    let retryIndex = 0;

    async function submitPendingTask() {
      try {
        const created = await createQuestionTask(
          snapshot.currentQuestion,
          snapshot.clientRequestId as string,
          snapshot.includeDebug,
          sessionId,
        );
        if (stopped) return;
        clearPendingReceipt(created.client_request_id);
        activeTaskId.current = created.task_id;
        if (cancelRequested.current) {
          const cancelled = await cancelQuestionTask(created.task_id);
          if (!stopped) applyTaskSnapshot(cancelled);
          return;
        }
        setSnapshot((current) => ({
          ...current,
          taskId: created.task_id,
          clientRequestId: created.client_request_id,
          taskStatus: created.status,
          message: "",
          updatedAt: new Date().toISOString(),
        }));
      } catch {
        if (stopped) return;
        setSnapshot((current) => ({
          ...current,
          message: cancelRequested.current
            ? "正在确认任务并停止生成。"
            : "提交状态暂未确认，系统正在按同一请求编号重试。",
        }));
        const delay = QA_RETRY_DELAYS_MS[Math.min(retryIndex, QA_RETRY_DELAYS_MS.length - 1)];
        retryIndex += 1;
        timer = window.setTimeout(submitPendingTask, delay);
      }
    }

    void submitPendingTask();
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [applyTaskSnapshot, sessionId, snapshot.clientRequestId, snapshot.currentQuestion, snapshot.includeDebug, snapshot.taskId, snapshot.taskStatus]);

  const reloadTask = useCallback(async () => {
    if (!activeTaskId.current) return;
    try {
      const task = await getQuestionTask(activeTaskId.current);
      applyTaskSnapshot(task);
    } catch (error) {
      setSnapshot((current) => ({
        ...current,
        message: error instanceof Error ? error.message : "重新读取任务状态失败。",
      }));
    }
  }, [applyTaskSnapshot]);

  const restoreTask = useCallback(async (taskId: string) => {
    setIsSubmitting(true);
    try {
      const task = await getQuestionTask(taskId);
      applyTaskSnapshot(task);
    } catch (error) {
      setIsSubmitting(false);
      setSnapshot((current) => ({
        ...current,
        message: error instanceof Error ? error.message : "历史任务暂时无法恢复。",
      }));
    }
  }, [applyTaskSnapshot]);

  useEffect(() => {
    if (!snapshot.taskId || !isActiveTaskStatus(snapshot.taskStatus)) {
      setStreamConnected(false);
      return;
    }
    setStreamConnected(false);
    return streamQuestionTask(
      snapshot.taskId,
      applyTaskSnapshot,
      setStreamConnected,
    );
  }, [applyTaskSnapshot, snapshot.taskId, snapshot.taskStatus]);

  useEffect(() => {
    if (!snapshot.taskId || streamConnected) return;

    let stopped = false;
    let timer: number | undefined;
    let retryIndex = 0;

    async function pollTask() {
      try {
        const task = await getQuestionTask(snapshot.taskId as string);
        if (stopped) return;
        applyTaskSnapshot(task);
        retryIndex = 0;
        if (isActiveTaskStatus(task.status)) {
          timer = window.setTimeout(pollTask, QA_TASK_POLL_INTERVAL_MS);
        }
      } catch {
        if (stopped) return;
        setSnapshot((current) => ({ ...current, message: "连接暂时中断，正在恢复任务状态。" }));
        setIsSubmitting(true);
        const delay = QA_RETRY_DELAYS_MS[Math.min(retryIndex, QA_RETRY_DELAYS_MS.length - 1)];
        retryIndex += 1;
        timer = window.setTimeout(pollTask, delay);
      }
    }

    void pollTask();
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [applyTaskSnapshot, snapshot.taskId, streamConnected]);

  const runQuestion = useCallback(async (value?: string, includeDebug = false) => {
    if (isActiveTaskStatus(snapshot.taskStatus) || isSubmitting) return;
    const trimmedQuestion = (value ?? draftQuestion).trim();
    if (!trimmedQuestion) {
      setSnapshot((current) => ({ ...current, message: "请输入问题。" }));
      return;
    }

    const clientRequestId = createClientRequestId();
    const pendingReceipt: PendingReceipt = {
      clientRequestId,
      question: trimmedQuestion,
      includeDebug,
    };
    savePendingReceipt(pendingReceipt);
    cancelRequested.current = false;
    setIsCancelling(false);
    setIsSubmitting(true);
    setDraftQuestion("");
    activeTaskId.current = null;
    setSnapshot({
      taskId: null,
      clientRequestId,
      currentQuestion: trimmedQuestion,
      options: [],
      includeDebug,
      taskStatus: "queued",
      answerPreview: null,
      answer: null,
      message: "",
      progressEvents: [],
      updatedAt: new Date().toISOString(),
      createdAt: new Date().toISOString(),
      completedAt: null,
    });
  }, [draftQuestion, isSubmitting, snapshot.taskStatus]);

  const cancelQuestion = useCallback(async () => {
    if (!isActiveTaskStatus(snapshot.taskStatus) || isCancelling) return;
    cancelRequested.current = true;
    setIsCancelling(true);
    setSnapshot((current) => ({ ...current, message: "正在停止生成。" }));

    const taskId = activeTaskId.current ?? snapshot.taskId;
    if (!taskId) return;
    try {
      const task = await cancelQuestionTask(taskId);
      applyTaskSnapshot(task);
    } catch (error) {
      cancelRequested.current = false;
      setIsCancelling(false);
      setSnapshot((current) => ({
        ...current,
        message: error instanceof Error ? error.message : "停止生成失败，请重试。",
      }));
    }
  }, [applyTaskSnapshot, isCancelling, snapshot.taskId, snapshot.taskStatus]);

  const clearMessage = useCallback(() => {
    setSnapshot((current) => ({ ...current, message: "" }));
  }, []);

  const value = useMemo<QATaskContextValue>(
    () => ({
      ...snapshot,
      draftQuestion,
      sessionId,
      setSessionId,
      isSubmitting,
      isCancelling,
      setDraftQuestion,
      runQuestion,
      cancelQuestion,
      reloadTask,
      restoreTask,
      clearMessage,
    }),
    [cancelQuestion, clearMessage, draftQuestion, isCancelling, isSubmitting, reloadTask, restoreTask, runQuestion, sessionId, snapshot],
  );

  return <QATaskContext.Provider value={value}>{children}</QATaskContext.Provider>;
}

export function useQATask() {
  const context = useContext(QATaskContext);
  if (!context) throw new Error("useQATask must be used inside QATaskProvider");
  return context;
}

export function isActiveTaskStatus(status: string): boolean {
  return status === "queued" || status === "running";
}

function loadPendingReceipt(): PendingReceipt | null {
  try {
    const raw = window.localStorage.getItem(QA_PENDING_RECEIPT_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<PendingReceipt>;
    if (typeof parsed.clientRequestId !== "string" || typeof parsed.question !== "string") return null;
    return {
      clientRequestId: parsed.clientRequestId,
      question: parsed.question,
      includeDebug: parsed.includeDebug === true,
    };
  } catch {
    return null;
  }
}

function savePendingReceipt(receipt: PendingReceipt): void {
  try {
    window.localStorage.setItem(QA_PENDING_RECEIPT_KEY, JSON.stringify(receipt));
  } catch {
    // The task flow still works when browser storage is unavailable.
  }
}

function clearPendingReceipt(clientRequestId: string): void {
  try {
    const receipt = loadPendingReceipt();
    if (receipt?.clientRequestId === clientRequestId) {
      window.localStorage.removeItem(QA_PENDING_RECEIPT_KEY);
    }
  } catch {
    // Nothing else is required after a successful idempotent submission.
  }
}

function createClientRequestId(): string {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID().replaceAll("-", "");
  return `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 14)}`;
}

/** 首次加载生成并持久化一个会话编号，后续所有问答任务共用同一会话。 */
function getOrCreateSessionId(): string {
  try {
    const existing = window.localStorage.getItem(QA_SESSION_ID_KEY);
    if (existing && /^[A-Za-z0-9_-]{8,128}$/.test(existing)) return existing;
    const sessionId = createClientRequestId();
    window.localStorage.setItem(QA_SESSION_ID_KEY, sessionId);
    return sessionId;
  } catch {
    return createClientRequestId();
  }
}
