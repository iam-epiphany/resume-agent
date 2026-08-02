import { Clock3, History, Loader2, X } from "lucide-react";
import { useEffect, useState } from "react";

import { listQuestionTasks } from "../../api/qa";
import type { QATaskStatusResponse } from "../../types/api";

interface QuestionHistoryDrawerProps {
  open: boolean;
  onClose: () => void;
  onRestore: (taskId: string) => Promise<void>;
}

export function QuestionHistoryDrawer({ open, onClose, onRestore }: QuestionHistoryDrawerProps) {
  const [tasks, setTasks] = useState<QATaskStatusResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    let stopped = false;
    setLoading(true);
    setError(null);
    void listQuestionTasks(12)
      .then((result) => { if (!stopped) setTasks(result); })
      .catch((reason) => { if (!stopped) setError(reason instanceof Error ? reason.message : "历史任务暂时无法读取。"); })
      .finally(() => { if (!stopped) setLoading(false); });
    return () => { stopped = true; };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose, open]);

  if (!open) return null;
  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <aside className="drawer" role="dialog" aria-modal="true" aria-labelledby="question-history-title">
        <div className="drawer__head">
          <div><span className="section-kicker">最近任务</span><h2 id="question-history-title">问答历史</h2></div>
          <button className="icon-only-button" type="button" onClick={onClose} aria-label="关闭问答历史"><X size={19} /></button>
        </div>
        {loading ? <div className="drawer-loading"><Loader2 size={19} className="spinning" />正在读取历史任务</div> : null}
        {error ? <p className="inline-notice error">{error}</p> : null}
        {!loading && !error && !tasks.length ? <p className="empty-copy">暂无历史问答。完成一次提问后可在这里恢复结果。</p> : null}
        <ol className="task-history-list">
          {tasks.map((task) => (
            <li key={task.task_id}>
              <button type="button" onClick={() => void onRestore(task.task_id).then(onClose)}>
                <span className="task-history-list__icon"><History size={17} /></span>
                <span className="task-history-list__body">
                  <strong>{task.question}</strong>
                  <small><Clock3 size={13} />{formatDateTime(task.updated_at)} · {taskStatusLabel(task.status)}</small>
                </span>
              </button>
            </li>
          ))}
        </ol>
      </aside>
    </div>
  );
}

function taskStatusLabel(status: string): string {
  return ({ queued: "已提交", running: "处理中", completed: "已回答", failed: "处理失败", cancelled: "已停止" } as Record<string, string>)[status] ?? status;
}

function formatDateTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false, month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}
