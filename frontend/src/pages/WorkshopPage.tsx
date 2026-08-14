import { useCallback, useEffect, useRef, useState } from "react";
import {
  type PersonaPublic,
  type WorkshopJobView,
  activatePersona,
  confirmPersona,
  getActivePersona,
  listPersonas,
  listWorkshopJobs,
  rollbackWorkshopJob,
  transformMaterials,
} from "../api/workshop";

/**
 * 人物工坊（admin）：当前人物档案 + 任意简历材料 → LLM 加工 → 自动入库。
 */
export function WorkshopPage() {
  const [personas, setPersonas] = useState<PersonaPublic[]>([]);
  const [active, setActive] = useState<PersonaPublic | null>(null);
  const [jobs, setJobs] = useState<WorkshopJobView[]>([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [profileDraft, setProfileDraft] = useState<Record<string, unknown>>({});
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [personaList, personaActive, jobList] = await Promise.all([
        listPersonas(),
        getActivePersona(),
        listWorkshopJobs(),
      ]);
      setPersonas(personaList);
      setActive(personaActive);
      setJobs(jobList);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载人物信息失败");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function handleTransform(files: FileList | null) {
    if (!files || files.length === 0) return;
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const result = (await transformMaterials(Array.from(files))) as {
        job_id: string;
        status: string;
        generated_document_ids: string[];
        generated_fact_count: number;
        error?: string | null;
      };
      if (result.error) {
        setError(`转换失败：${result.error}`);
      } else {
        setMessage(
          `转换完成：生成 ${result.generated_document_ids.length} 篇文档、` +
            `${result.generated_fact_count} 条事实。人物档案为草稿，请确认后生效。`,
        );
      }
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "转换失败");
    } finally {
      setBusy(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleActivate(personaId: string) {
    try {
      await activatePersona(personaId);
      setMessage("已切换当前人物（知识库问答将按该人物隔离）。");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "切换失败");
    }
  }

  async function handleConfirmProfile() {
    if (!active) return;
    try {
      await confirmPersona(active.persona_id, profileDraft);
      setMessage("人物档案已确认，提示词个性化生效。");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "确认失败");
    }
  }

  async function handleRollback(jobId: string) {
    if (!window.confirm("回滚将删除该任务生成的全部文档与事实，确认继续？")) return;
    try {
      await rollbackWorkshopJob(jobId);
      setMessage("已回滚。");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "回滚失败");
    }
  }

  return (
    <main className="page workshop-page">
      <header className="page-header">
        <h1>人物工坊</h1>
        <p>把任意人的简历材料交给大模型加工成检索友好的知识库，自动入库后即可切换为该人物问答。</p>
      </header>

      {error ? <div className="alert alert--danger">{error}</div> : null}
      {message ? <div className="alert alert--success">{message}</div> : null}

      <section className="panel">
        <h2>当前人物</h2>
        {active ? (
          <div className="persona-card">
            <div className="persona-card__row">
              <strong>{active.display_name}</strong>
              <span className={`persona-status persona-status--${active.status}`}>
                {active.status === "confirmed" ? "已确认" : "草稿待确认"}
              </span>
            </div>
            <div className="persona-card__row persona-card__dim">{active.profile_summary || "（无摘要）"}</div>
            <div className="persona-card__actions">
              <label className="button button--primary">
                上传简历材料（LLM 加工）
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf,.doc,.docx,.md,.txt,.html,.htm,.jsonl"
                  hidden
                  disabled={busy}
                  onChange={(event) => void handleTransform(event.target.files)}
                />
              </label>
            </div>
          </div>
        ) : (
          <p>正在加载人物信息…</p>
        )}
      </section>

      <section className="panel">
        <h2>人物档案确认</h2>
        <p className="panel__hint">
          工坊转换时会自动提取人物档案（草稿状态，不参与提示词个性化）。确认后姓名与摘要将驱动问答提示词。
        </p>
        <label className="field">
          <span>档案 JSON（可编辑，保存并确认）</span>
          <textarea
            rows={6}
            value={JSON.stringify(profileDraft, null, 2)}
            onChange={(event) => {
              try {
                setProfileDraft(JSON.parse(event.target.value));
              } catch {
                // 编辑中不强制校验
              }
            }}
          />
        </label>
        <button className="button button--primary" onClick={() => void handleConfirmProfile()}>
          保存并确认档案
        </button>
      </section>

      <section className="panel">
        <h2>人物列表（切换当前人物）</h2>
        <ul className="persona-list">
          {personas.map((persona) => (
            <li key={persona.persona_id} className="persona-list__item">
              <span>
                {persona.display_name}
                {persona.is_active ? "（当前）" : ""}
              </span>
              <button
                className="button"
                disabled={persona.is_active}
                onClick={() => void handleActivate(persona.persona_id)}
              >
                切换
              </button>
            </li>
          ))}
        </ul>
      </section>

      <section className="panel">
        <h2>转换任务</h2>
        {jobs.length === 0 ? (
          <p className="panel__hint">暂无转换任务。上传一份简历材料即可开始。</p>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>任务</th>
                <th>状态</th>
                <th>生成文档</th>
                <th>事实</th>
                <th>LLM 调用</th>
                <th>加工版本</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <tr key={job.job_id}>
                  <td>
                    {job.raw_filenames.join("、")}
                    <div className="cell-dim">{job.job_id}</div>
                  </td>
                  <td>
                    {job.status}
                    {job.error ? <div className="cell-dim">{job.error}</div> : null}
                  </td>
                  <td>{job.generated_document_ids.length} 篇</td>
                  <td>{job.generated_fact_count} 条</td>
                  <td>{job.llm_call_count} 次</td>
                  <td>{job.skill_version ?? "—"}</td>
                  <td>
                    {job.status === "completed" ? (
                      <button className="button button--danger" onClick={() => void handleRollback(job.job_id)}>
                        回滚
                      </button>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </main>
  );
}
