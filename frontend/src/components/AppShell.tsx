import { Activity, BookOpenText, ClipboardList, Wand2, LogIn, LogOut, MessageSquarePlus, MessageSquareText, X } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";

import { useChatHistory } from "../state/chatHistoryContext";
import { isKnowledgeBaseReady, useSystemStatus } from "../state/systemStatusContext";
import { useAuth } from "../state/authContext";
import { useReadiness } from "../state/readinessContext";
import { BrandMark } from "./BrandMark";
import { SystemStatusPanel } from "./SystemStatusPanel";

interface AppShellProps {
  path: string;
  onNavigate: (path: string) => void;
  children: ReactNode;
}

export function AppShell({ path, onNavigate, children }: AppShellProps) {
  const { isAuthenticated, logout } = useAuth();
  const system = useSystemStatus();
  const readiness = useReadiness();
  const chat = useChatHistory();
  const qaActive = path === "/" || path === "/qa";
  const [statusOpen, setStatusOpen] = useState(false);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);

  // 状态圆点：登录后看后台系统状态（含文档数），匿名时看公开就绪状态
  const ready = isAuthenticated ? isKnowledgeBaseReady(system) : readiness.ready;
  const hasError = isAuthenticated ? Boolean(system.error) : Boolean(readiness.error);
  const isLoading = isAuthenticated ? system.isLoading : readiness.isLoading;

  useEffect(() => {
    if (!statusOpen) return;
    closeButtonRef.current?.focus();
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") setStatusOpen(false);
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [statusOpen]);

  function handleNewThread() {
    chat.createThread();
    onNavigate("/");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" type="button" onClick={() => onNavigate("/")} aria-label="进入 ResumeMind 智能问答">
          <span className="brand-mark"><BrandMark size={46} /></span>
          <span>
            <strong>ResumeMind</strong>
            <small>个人简历智能问答</small>
          </span>
        </button>
        {/* 前台（匿名）：单功能场景不显示导航；后台（管理员）：保留三模块导航 */}
        {isAuthenticated ? (
          <nav aria-label="主导航">
            <button type="button" aria-current={qaActive ? "page" : undefined} className={qaActive ? "nav-item active" : "nav-item"} onClick={() => onNavigate("/")}>
              <MessageSquareText size={18} />
              智能问答
            </button>
            <button type="button" aria-current={path === "/documents" ? "page" : undefined} className={path === "/documents" ? "nav-item active" : "nav-item"} onClick={() => onNavigate("/documents")}>
              <BookOpenText size={18} />
              知识库
            </button>
            <button type="button" aria-current={path === "/workshop" ? "page" : undefined} className={path === "/workshop" ? "nav-item active" : "nav-item"} onClick={() => onNavigate("/workshop")}>
              <Wand2 size={18} />
              人物工坊
            </button>
            <button type="button" aria-current={path === "/audit" ? "page" : undefined} className={path === "/audit" ? "nav-item active" : "nav-item"} onClick={() => onNavigate("/audit")}>
              <ClipboardList size={18} />
              操作日志
            </button>
          </nav>
        ) : null}

        {/* 历史对话（页面级，关闭页面即清除） */}
        {qaActive ? (
          <div className="sidebar-threads">
            <button className="sidebar-threads__new" type="button" onClick={handleNewThread}>
              <MessageSquarePlus size={16} />
              新对话
            </button>
            <div className="sidebar-threads__list" aria-label="历史对话">
              {chat.threads.map((thread) => (
                <div key={thread.id} className={`sidebar-thread${thread.id === chat.activeThreadId ? " active" : ""}`}>
                  <button
                    className="sidebar-thread__title"
                    type="button"
                    title={thread.title}
                    onClick={() => {
                      chat.switchThread(thread.id);
                      onNavigate("/");
                    }}
                  >
                    {thread.title}
                  </button>
                  <button
                    className="sidebar-thread__delete"
                    type="button"
                    onClick={() => chat.deleteThread(thread.id)}
                    aria-label={`删除对话：${thread.title}`}
                    title="删除对话"
                  >
                    <X size={13} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        <div className="sidebar-footer">
          {isAuthenticated ? (
            <>
              <button
                className="mobile-status-button"
                type="button"
                onClick={() => setStatusOpen(true)}
                aria-label="查看系统状态"
                title="查看系统状态"
              >
                <Activity size={18} />
                <span className={`status-dot ${hasError ? "error" : ready ? "ok" : isLoading ? "loading" : "warning"}`} />
              </button>
              <button
                className="sidebar-logout-button"
                type="button"
                onClick={() => { logout(); onNavigate("/"); }}
                title="退出管理员登录"
              >
                <LogOut size={16} />
                退出登录
              </button>
              <div className="sidebar-system-status"><SystemStatusPanel /></div>
            </>
          ) : (
            <button
              className="sidebar-login-button"
              type="button"
              onClick={() => onNavigate("/login")}
              title="管理员登录"
            >
              <LogIn size={16} />
              管理员登录
            </button>
          )}
        </div>
      </aside>
      <section className="content">{children}</section>
      {statusOpen ? (
        <div className="status-modal-backdrop" role="presentation" onMouseDown={() => setStatusOpen(false)}>
          <section className="status-modal" role="dialog" aria-modal="true" aria-label="系统状态" onMouseDown={(event) => event.stopPropagation()}>
            <div className="status-modal__head">
              <span>运行状态与基础设施</span>
              <button ref={closeButtonRef} className="icon-only-button" type="button" onClick={() => setStatusOpen(false)} aria-label="关闭系统状态">
                <X size={18} />
              </button>
            </div>
            <SystemStatusPanel />
          </section>
        </div>
      ) : null}
    </div>
  );
}
