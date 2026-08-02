import { MessageSquareText } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { QuestionComposer } from "../components/qa/QuestionComposer";
import { QuestionHistoryDrawer } from "../components/qa/QuestionHistoryDrawer";
import { useAuth } from "../state/authContext";
import { type ChatMessage, useChatHistory } from "../state/chatHistoryContext";
import { isActiveTaskStatus, useQATask } from "../state/qaTaskContext";
import { useReadiness } from "../state/readinessContext";

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

export function RagPage() {
  const { isAuthenticated } = useAuth();
  const qa = useQATask();
  const readiness = useReadiness();
  const chat = useChatHistory();
  const [historyOpen, setHistoryOpen] = useState(false);
  const threadRef = useRef<HTMLDivElement | null>(null);
  // 当前任务所属线程：任务完成/流式更新时写回原线程（用户可能已切换线程）
  const taskThreadIdRef = useRef<string | null>(null);
  const active = isActiveTaskStatus(qa.taskStatus);
  const messages = chat.activeThread?.messages ?? [];

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

  return (
    <main className="page qa-page chat-page">
      <header className="product-header">
        <div>
          <p className="eyebrow">ResumeMind 智能问答</p>
          <div className="product-title-lockup">
            <h1>关于我的一切，都可以问</h1>
          </div>
        </div>
        {isAuthenticated ? (
          <button className="secondary-button" type="button" onClick={() => setHistoryOpen(true)}>
            <MessageSquareText size={17} />问答历史
          </button>
        ) : null}
      </header>

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
