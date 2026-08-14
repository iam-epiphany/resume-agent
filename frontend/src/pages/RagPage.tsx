import { AlertTriangle, KeyRound, MessageSquareText } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { qaAccessStatus, qaAccessSubmit, type QAAccessStatus } from "../api/qa";
import { QuestionComposer } from "../components/qa/QuestionComposer";
import { QuestionHistoryDrawer } from "../components/qa/QuestionHistoryDrawer";
import { useAuth } from "../state/authContext";
import { type ChatMessage, useChatHistory } from "../state/chatHistoryContext";
import { isActiveTaskStatus, useQATask } from "../state/qaTaskContext";
import { useReadiness, type LoadLevel } from "../state/readinessContext";

/** 常见问题：点击后填入输入框 */
const SUGGESTED_QUESTIONS = [
  "请介绍一下你自己",
  "你的项目经历有哪些",
  "你的技术栈是什么",
  "你的优缺点是什么",
  "你的职业规划是什么",
];

function updateLastAssistant(
  messages: ChatMessage[],
  patch: Partial<ChatMessage>,
): ChatMessage[] {
  if (messages.length === 0) return messages;
  const next = [...messages];
  const index = next.length - 1;
  if (next[index].role !== "assistant") return messages;
  next[index] = { ...next[index], ...patch };
  return next;
}

function threadTitle(question: string): string {
  const trimmed = question.trim().replace(/\s+/g, " ");
  return trimmed.length > 18 ? `${trimmed.slice(0, 18)}…` : trimmed;
}

/** 负载分级文案与状态点类名（状态点在 styles.css 中 ok=绿 / 默认黄 / error=红） */
const LOAD_LABELS: Record<LoadLevel, string> = {
  green: "系统负载正常",
  yellow: "当前访问人数较多，回复可能较慢",
  red: "系统繁忙，请稍后再试",
};

function loadDotClass(level: LoadLevel): string {
  if (level === "green") return "ok";
  if (level === "red") return "error";
  return "";
}

export function RagPage() {
  const { isAuthenticated } = useAuth();
  const qa = useQATask();
  const readiness = useReadiness();
  const chat = useChatHistory();
  const [historyOpen, setHistoryOpen] = useState(false);
  // 访客访问闸状态：access_enabled && !granted → 显示访问码门
  const [accessStatus, setAccessStatus] = useState<QAAccessStatus | null>(null);
  const [accessCode, setAccessCode] = useState("");
  const [accessError, setAccessError] = useState("");
  const [accessSubmitting, setAccessSubmitting] = useState(false);
  // 预算提醒：每次会话只弹一次（可手动关闭）
  const [budgetBannerOpen, setBudgetBannerOpen] = useState(false);
  const budgetNotifiedRef = useRef(false);
  // 负载提醒（绿/黄/红）：黄色及以上显示横幅；可关闭；负载回绿后重置关闭态，
  // 再次转黄/红时重新提醒（新访客挂载轮询即看到，无需会话内只弹一次）
  const [loadBannerDismissed, setLoadBannerDismissed] = useState(false);
  const loadLevel = readiness.loadLevel;
  useEffect(() => {
    if (loadLevel === "green") {
      setLoadBannerDismissed(false);
    }
  }, [loadLevel]);
  const loadBannerVisible =
    (loadLevel === "yellow" || loadLevel === "red") && !loadBannerDismissed;
  const threadRef = useRef<HTMLDivElement | null>(null);
  // 当前任务所属线程：任务完成/流式更新时写回原线程（用户可能已切换线程）
  const taskThreadIdRef = useRef<string | null>(null);
  const active = isActiveTaskStatus(qa.taskStatus);
  const messages = useMemo(
    () => chat.activeThread?.messages ?? [],
    [chat.activeThread?.messages],
  );

  // 访问闸状态：挂载时拉取 + 每 60s 轮询（预算提醒与访问码过期都靠它刷新）
  useEffect(() => {
    let disposed = false;
    const refresh = () => {
      qaAccessStatus()
        .then((status) => {
          if (disposed) return;
          setAccessStatus(status);
          if (status.budget_warning && !budgetNotifiedRef.current) {
            budgetNotifiedRef.current = true;
            setBudgetBannerOpen(true);
          }
        })
        .catch(() => {
          // 状态接口失败不阻塞页面：保持当前界面，下次轮询再试
        });
    };
    refresh();
    const timer = window.setInterval(refresh, 60_000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, []);

  function submitAccessCode() {
    const code = accessCode.trim();
    if (!code || accessSubmitting) return;
    setAccessSubmitting(true);
    setAccessError("");
    qaAccessSubmit(code)
      .then(() => qaAccessStatus())
      .then((status) => {
        setAccessStatus(status);
        if (!status.granted) {
          setAccessError("访问码未生效，请重试。");
        }
      })
      .catch((error: unknown) => {
        setAccessError(error instanceof Error ? error.message : "访问码不正确，请重试。");
      })
      .finally(() => setAccessSubmitting(false));
  }

  // 切换线程 / 首次加载：把后端会话绑定到当前线程
  useEffect(() => {
    if (chat.activeThread) {
      qa.setSessionId(chat.activeThread.sessionId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chat.activeThreadId]);

  // 自动滚动到底部
  useEffect(() => {
    const el = threadRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, active]);

  // 流式预览：更新任务所属线程的最后一条助手消息
  useEffect(() => {
    if (!active || !qa.answerPreview) return;
    const threadId = taskThreadIdRef.current;
    if (threadId === null) return;
    chat.updateThread(threadId, (thread) => ({
      messages: updateLastAssistant(thread.messages, { content: qa.answerPreview?.answer ?? "" }),
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [qa.answerPreview, active]);

  // 任务完成：写入最终答案与回答模式
  useEffect(() => {
    const answer = qa.answer;
    if (!answer) return;
    const threadId = taskThreadIdRef.current;
    if (threadId === null) return;
    chat.updateThread(threadId, (thread) => ({
      messages: updateLastAssistant(thread.messages, {
        content: answer.answer ?? "",
        answerMode: answer.answer_mode,
        citations: answer.citations ?? undefined,
        status: "done",
      }),
    }));
    taskThreadIdRef.current = null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [qa.answer]);

  // 失败 / 取消
  useEffect(() => {
    const threadId = taskThreadIdRef.current;
    if (threadId === null) return;
    if (qa.taskStatus === "failed" && !qa.answer) {
      chat.updateThread(threadId, (thread) => ({
        messages: updateLastAssistant(thread.messages, { status: "error", content: qa.message || "本次问答未完成，请重试。" }),
      }));
      taskThreadIdRef.current = null;
    } else if (qa.taskStatus === "cancelled") {
      chat.updateThread(threadId, (thread) => ({
        messages: updateLastAssistant(thread.messages, { status: "cancelled", content: "已停止生成。" }),
      }));
      taskThreadIdRef.current = null;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [qa.taskStatus, qa.message, qa.answer]);

  function submitQuestion() {
    const question = qa.draftQuestion.trim();
    if (!question || active || chat.activeThreadId === null) return;
    taskThreadIdRef.current = chat.activeThreadId;
    chat.updateActiveThread((thread) => ({
      title: thread.messages.length === 0 ? threadTitle(question) : thread.title,
      messages: [
        ...thread.messages,
        { id: createMessageId(), role: "user", content: question, status: "done" },
        { id: createMessageId(), role: "assistant", content: "", status: "streaming" },
      ],
    }));
    void qa.runQuestion();
  }

  // 访客访问码门：简历中附带的访问码输入正确后（服务端签发 cookie）才显示问答界面
  if (accessStatus?.access_enabled && !accessStatus.granted) {
    return (
      <main className="page qa-page chat-page">
        <header className="product-header">
          <div>
            <p className="eyebrow">ResumeMind 智能问答</p>
            <div className="product-title-lockup">
              <h1>关于我的一切，都可以问</h1>
            </div>
          </div>
        </header>
        <div className="access-gate">
          <div className="access-gate__card">
            <KeyRound size={28} aria-hidden="true" />
            <h2>请输入访问码</h2>
            <p>
              本问答服务面向查看简历的面试官开放。访问码已随简历/面试邀请一并提供，
              输入后 24 小时内无需重复输入。
            </p>
            <form
              className="access-gate__form"
              onSubmit={(event) => {
                event.preventDefault();
                submitAccessCode();
              }}
            >
              <input
                type="password"
                value={accessCode}
                onChange={(event) => setAccessCode(event.target.value)}
                placeholder="访问码"
                autoFocus
                aria-label="访问码"
              />
              <button className="primary-button" type="submit" disabled={accessSubmitting || !accessCode.trim()}>
                {accessSubmitting ? "验证中…" : "进入问答"}
              </button>
            </form>
            {accessError ? <p className="access-gate__error">{accessError}</p> : null}
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="page qa-page chat-page">
      <header className="product-header">
        <div>
          <p className="eyebrow">ResumeMind 智能问答</p>
          <div className="product-title-lockup">
            <h1>关于我的一切，都可以问</h1>
          </div>
        </div>
        <div className="product-header-actions">
          {loadLevel ? (
            <span className={`load-indicator load-${loadLevel}`} title={LOAD_LABELS[loadLevel]}>
              <span className={`status-dot ${loadDotClass(loadLevel)}`} aria-hidden="true" />
              <span className="load-indicator-text">{LOAD_LABELS[loadLevel]}</span>
            </span>
          ) : null}
          {isAuthenticated ? (
            <button className="secondary-button" type="button" onClick={() => setHistoryOpen(true)}>
              <MessageSquareText size={17} />问答历史
            </button>
          ) : null}
        </div>
      </header>

      {budgetBannerOpen ? (
        <div className="budget-banner" role="alert">
          <AlertTriangle size={16} aria-hidden="true" />
          <span>今日问答预算即将超限，请合理使用。</span>
          <button type="button" aria-label="关闭提醒" onClick={() => setBudgetBannerOpen(false)}>
            ×
          </button>
        </div>
      ) : null}

      {loadBannerVisible ? (
        <div className={`load-banner load-banner-${loadLevel === "red" ? "red" : "yellow"}`} role="alert">
          <AlertTriangle size={16} aria-hidden="true" />
          <span>
            {loadLevel === "red"
              ? "当前系统访问人数较多，可能排队较久，请耐心等待。"
              : "当前访问人数较多，系统负载较高，回复可能较慢，请耐心等待。"}
          </span>
          <button
            type="button"
            aria-label="关闭提醒"
            onClick={() => setLoadBannerDismissed(true)}
          >
            ×
          </button>
        </div>
      ) : null}

      <div className="chat-thread" ref={threadRef} aria-label="对话记录">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <h2>您想了解我什么？</h2>
            <p>可以问我任何关于我的问题，比如项目经历、技术栈或求职意向。</p>
          </div>
        ) : (
          messages.map((message) => <ChatBubble key={message.id} message={message} />)
        )}
      </div>

      <QuestionComposer
        value={qa.draftQuestion}
        onChange={qa.setDraftQuestion}
        onSubmit={submitQuestion}
        active={active}
        cancelling={qa.isCancelling}
        onCancel={() => void qa.cancelQuestion()}
        ready={readiness.ready}
        scopeLabel={scopeLabel(readiness.isLoading, readiness.error, readiness.message)}
        message={qa.message || undefined}
      />

      {/* 常见问题：放在输入框下方 */}
      <div className="chat-suggestions">
        <span className="chat-suggestions__label">提出较多的问题：</span>
        {SUGGESTED_QUESTIONS.map((question) => (
          <button
            key={question}
            className="chat-suggestion"
            type="button"
            disabled={active}
            onClick={() => qa.setDraftQuestion(question)}
          >
            {question}
          </button>
        ))}
      </div>

      <QuestionHistoryDrawer open={historyOpen} onClose={() => setHistoryOpen(false)} onRestore={qa.restoreTask} />
    </main>
  );
}

function ChatBubble({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="chat-message chat-message--user">
        <div className="chat-bubble chat-bubble--user">{message.content}</div>
      </div>
    );
  }

  const thinking = message.status === "streaming" && !message.content;
  const badge =
    message.answerMode === "hedged"
      ? { text: "基于知识库推测", tone: "hedged" }
      : message.answerMode === "redirected"
        ? { text: "已引导至简历话题", tone: "redirected" }
        : null;
  const citations = message.citations ?? [];

  return (
    <div className="chat-message chat-message--assistant">
      {thinking ? (
        // 思考中：三个跳动点直接展示，不用聊天框包裹
        <span className="chat-thinking" role="status" aria-label="思考中">
          <span className="thinking-dots" aria-hidden="true"><i /><i /><i /></span>
        </span>
      ) : (
        <div className="chat-bubble-stack">
          {badge ? <span className={`answer-mode-badge answer-mode-badge--${badge.tone}`}>{badge.text}</span> : null}
          <div className="chat-bubble chat-bubble--assistant">{message.content}</div>
          {citations.length > 0 ? (
            <ul className="answer-citations" aria-label="回答依据">
              {citations.map((citation, index) => (
                <li key={`${citation.source_doc}-${citation.section_title ?? ""}-${index}`} className="answer-citations__item">
                  <span className="answer-citations__source">
                    {citation.source_doc}
                    {citation.section_title ? ` · ${citation.section_title}` : ""}
                    {citation.fact_status === "confirmed" ? " · 已核实" : ""}
                  </span>
                  {citation.excerpt ? (
                    <span className="answer-citations__excerpt">“{citation.excerpt}”</span>
                  ) : null}
                </li>
              ))}
            </ul>
          ) : null}
          {message.status === "error" ? (
            <span className="chat-message__status">回答未完成</span>
          ) : message.status === "cancelled" ? (
            <span className="chat-message__status">已停止</span>
          ) : null}
        </div>
      )}
    </div>
  );
}

function scopeLabel(isLoading: boolean, error: string | null, message: string): string {
  if (isLoading) return "正在检查系统状态";
  if (error) return "系统状态不可用";
  return message;
}

function createMessageId(): string {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  return `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}
