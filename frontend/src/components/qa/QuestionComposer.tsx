import { Loader2, SendHorizontal } from "lucide-react";
import type { FormEvent, KeyboardEvent } from "react";
import { useEffect, useRef } from "react";

interface QuestionComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  active: boolean;
  cancelling: boolean;
  onCancel: () => void;
  ready: boolean;
  scopeLabel: string;
  processingLabel?: string | null;
  message?: string;
}

export function QuestionComposer({
  value,
  onChange,
  onSubmit,
  active,
  cancelling,
  onCancel,
  ready,
  scopeLabel,
  processingLabel,
  message,
}: QuestionComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => resizeTextarea(textareaRef.current), [value]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (active) return;
    onSubmit();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!active && ready && value.trim()) {
        onSubmit();
      }
    }
  }

  return (
    <section className={`question-composer question-composer--docked${active ? " question-composer--active" : ""}`} aria-labelledby="question-composer-title">
      <div className="question-composer__scope">
        <span className={`scope-indicator ${ready ? "ready" : "warning"}`} />
        <span>{scopeLabel}</span>
        {processingLabel ? <span className="scope-processing">{processingLabel}</span> : null}
      </div>
      <form onSubmit={submit}>
        <label id="question-composer-title" className="field-label" htmlFor="resumemind-question">
          输入你想了解的简历相关问题
        </label>
        <textarea
          id="resumemind-question"
          ref={textareaRef}
          className="query-input"
          value={value}
          disabled={active}
          onKeyDown={handleKeyDown}
          onChange={(event) => {
            onChange(event.target.value);
            resizeTextarea(event.target);
          }}
          placeholder={active ? "当前问题正在处理，停止或完成后可继续提问。" : "例如：介绍一下你的项目经历、你拿过哪些奖、你的专业技能……"}
          aria-describedby="question-composer-help"
        />
        <div className="question-composer__footer">
          <p id="question-composer-help">{active ? "当前输入已锁定；任务完成或停止后可继续提问。" : "按 Enter 发送，Shift + Enter 换行。回答基于当前知识库中的简历与项目材料。"}</p>
          <div className="question-composer__actions">
            {active ? (
              <button
                className="composer-action-button composer-action-button--stop"
                type="button"
                disabled={cancelling}
                onClick={onCancel}
                aria-label={cancelling ? "正在停止" : "停止生成"}
                title={cancelling ? "正在停止" : "停止生成"}
              >
                {cancelling ? <Loader2 size={20} className="spinning" /> : <span className="stop-mark" aria-hidden="true" />}
              </button>
            ) : (
              <button
                className="composer-action-button composer-action-button--submit"
                type="submit"
                disabled={!ready || !value.trim()}
                aria-label="开始问答"
                title="开始问答（Enter）"
              >
                <SendHorizontal size={21} strokeWidth={2.2} aria-hidden="true" />
              </button>
            )}
          </div>
        </div>
        {!ready ? <p className="inline-notice warning">当前知识库尚未就绪，请先检查文档和系统状态。</p> : null}
        {message ? <p className="inline-notice error" role="alert">{message}</p> : null}
      </form>
    </section>
  );
}

function resizeTextarea(textarea: HTMLTextAreaElement | null) {
  if (!textarea) return;
  const minHeight = 52;
  const maxHeight = 240;
  textarea.style.height = "0px";
  const nextHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight);
  textarea.style.height = `${nextHeight}px`;
  textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
}
