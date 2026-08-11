import { AlertTriangle, CheckCircle2, FileText, Loader2, MoreVertical, RefreshCw, X } from "lucide-react";
import { useEffect, useState } from "react";

import { deleteAuditArchive, getAuditArchive, listAuditArchives, listAuditLogs } from "../api/audit";
import type { AuditArchiveDetailResponse, AuditArchiveSummary, AuditLogItem, Citation } from "../types/api";
import { useAuth } from "../state/authContext";
import { formatAuditLog, parseAuditArchiveContent, type ParsedAuditArchiveEntry } from "../utils/audit";

type AuditEventKind = "all" | "qa" | "upload" | "parse" | "index" | "refusal" | "exception";
type AuditLogRecord = AuditLogItem | ParsedAuditArchiveEntry;


export function AuditPage() {
  const { isAuthenticated } = useAuth();
  const [logs, setLogs] = useState<AuditLogItem[]>([]);
  const [archives, setArchives] = useState<AuditArchiveSummary[]>([]);
  const [selectedArchive, setSelectedArchive] = useState<AuditArchiveDetailResponse | null>(null);
  const [message, setMessage] = useState("仅显示当天审计日志，过期日志会自动归档。");
  const [severityFilter, setSeverityFilter] = useState<"all" | "info" | "warning" | "error">("all");
  const [eventKindFilter, setEventKindFilter] = useState<AuditEventKind>("all");
  const [selectedLog, setSelectedLog] = useState<AuditLogRecord | null>(null);
  const [openArchiveMenu, setOpenArchiveMenu] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const parsedArchive = selectedArchive ? parseAuditArchiveContent(selectedArchive.content) : null;
  const selectedArchiveSummary = selectedArchive ? archives.find((archive) => archive.date === selectedArchive.date) : null;
  const visibleLogs = logs.filter((log) => {
    const matchesSeverity = severityFilter === "all" || log.severity === severityFilter;
    const matchesKind = eventKindFilter === "all" || auditEventKind(log) === eventKindFilter;
    return matchesSeverity && matchesKind;
  });
  const errorCount = logs.filter((log) => log.severity === "error").length;
  const warningCount = logs.filter((log) => log.severity === "warning").length;

  useEffect(() => {
    void loadAuditData();
    // 登录态变化时切换数据源（匿名为空视图，登录后读全量+归档）
    // eslint-disable-next-line react-hooks/exhaustive-deps -- loadAuditData 每次渲染重建，仅需登录态变化时重载
  }, [isAuthenticated]);

  useEffect(() => {
    if (!selectedLog && !openArchiveMenu) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setSelectedLog(null);
      setOpenArchiveMenu(null);
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [openArchiveMenu, selectedLog]);

  async function loadAuditData() {
    setIsLoading(true);
    try {
      if (isAuthenticated) {
        const [logResult, archiveResult] = await Promise.all([listAuditLogs(), listAuditArchives()]);
        setLogs(logResult.logs);
        setArchives(archiveResult.archives);
        setMessage("仅显示当天审计日志，过期日志会自动归档。");
      } else {
        // 前台（匿名）：问答记录属隐私（访客之间互不可见），一律仅管理员可见
        setLogs([]);
        setArchives([]);
        setMessage("");
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "读取审计日志失败。");
    } finally {
      setIsLoading(false);
    }
  }

  async function showArchive(date: string) {
    try {
      const archive = await getAuditArchive(date);
      setSelectedArchive(archive);
      setMessage(`正在查看 ${date} 的日志归档。`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "读取日志归档失败。");
      await loadAuditData();
    }
  }

  async function removeArchive(date: string) {
    if (!window.confirm(`确认删除 ${date} 的日志归档？此操作不可恢复。`)) {
      return;
    }
    try {
      await deleteAuditArchive(date);
      if (selectedArchive?.date === date) {
        setSelectedArchive(null);
      }
      setMessage(`已删除 ${date} 的日志归档。`);
      await loadAuditData();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "删除日志归档失败。");
      await loadAuditData();
    }
  }

  return (
    <main className="page">
      <header className="product-header">
        <div>
          <p className="eyebrow">系统操作日志</p>
          <div className="product-title-lockup">
            <h1>问答与知识库运行记录</h1>
            <span className={errorCount ? "severity-badge error" : warningCount ? "severity-badge warning" : "severity-badge"}>
              {errorCount ? `${errorCount} 条严重` : warningCount ? `${warningCount} 条警告` : "状态正常"}
            </span>
          </div>
          <p className="page-lead">
            {isAuthenticated
              ? "记录文档上传、解析、索引、智能问答、拒答决策和系统异常。"
              : "问答与系统记录仅管理员可见，请登录后查看。"}
          </p>
        </div>
        <button className="icon-button" type="button" onClick={() => void loadAuditData()}>
          {isLoading ? <Loader2 size={17} className="spinning" /> : <RefreshCw size={17} />}
          {isLoading ? "读取中" : "刷新"}
        </button>
      </header>

      <section className="panel">
        <div className="document-list-toolbar">
          <div>
            <h2>今日审计记录</h2>
            <p className="toolbar-summary">{message}</p>
          </div>
          <div className="document-filters">
            <label>
              <span className="sr-only">事件筛选</span>
              <select aria-label="事件筛选" value={eventKindFilter} onChange={(event) => setEventKindFilter(event.target.value as AuditEventKind)}>
                <option value="all">全部事件（{logs.length}）</option>
                <option value="qa">问答</option>
                {isAuthenticated ? (
                  <>
                    <option value="upload">上传</option>
                    <option value="parse">解析</option>
                    <option value="index">索引</option>
                  </>
                ) : null}
                <option value="refusal">拒答</option>
                <option value="exception">异常</option>
              </select>
            </label>
            <label>
              <span className="sr-only">级别筛选</span>
              <select aria-label="级别筛选" value={severityFilter} onChange={(event) => setSeverityFilter(event.target.value as typeof severityFilter)}>
                <option value="all">全部级别</option>
                <option value="info">普通</option>
                <option value="warning">警告</option>
                <option value="error">严重</option>
              </select>
            </label>
          </div>
        </div>
        {isLoading ? (
          <div className="task-placeholder">
            <Loader2 size={20} className="spinning" />
            <div>
              <strong>正在读取审计日志</strong>
              <p className="muted">正在加载今日记录和历史归档。</p>
            </div>
          </div>
        ) : visibleLogs.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>时间</th>
                  <th>操作类型</th>
                  <th>操作对象</th>
                  <th>执行结果</th>
                  <th>耗时/次数</th>
                  <th>详情</th>
                </tr>
              </thead>
              <tbody>
                {visibleLogs.map((log) => {
                  const display = formatAuditLog(log);
                  return (
                    <tr key={log.id}>
                      <td data-label="时间">{formatDateTime(log.last_seen_at || log.created_at)}</td>
                      <td data-label="操作类型">{display.action}</td>
                      <td data-label="操作对象">{display.target}</td>
                      <td data-label="执行结果"><SeverityBadge severity={log.severity} /></td>
                      <td data-label="耗时/次数">{auditDurationOrCount(log)}</td>
                      <td data-label="说明" className="audit-detail">
                        <span>{auditTableSummary(log, display.detail)}</span>
                        <button className="text-button" type="button" onClick={() => setSelectedLog(log)}>详情</button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : !isAuthenticated ? (
          <div className="empty-state">
            <CheckCircle2 size={24} />
            <h2>仅管理员可见</h2>
            <p>问答记录与系统日志属隐私内容，登录管理员账号后查看。</p>
          </div>
        ) : (
          <div className="empty-state">
            {message.includes("失败") ? <AlertTriangle size={24} /> : <CheckCircle2 size={24} />}
            <h2>当前筛选条件下暂无日志</h2>
            <p>{message.includes("失败") ? message : "上传文档、建立索引或提交问答后会生成可追溯记录。"}</p>
          </div>
        )}
      </section>

      {isAuthenticated ? (
      <section className="panel">
        <div className="panel-title">
          <FileText size={20} />
          <h2>历史日志归档</h2>
        </div>
        {archives.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>日期</th>
                  <th>文件</th>
                  <th>大小</th>
                  <th>更新时间</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {archives.map((archive) => (
                  <tr key={archive.date}>
                    <td data-label="日期">{archive.date}</td>
                    <td data-label="文件">{archive.filename}</td>
                    <td data-label="大小">{formatFileSize(archive.size)}</td>
                    <td data-label="更新时间">{formatDateTime(archive.updated_at)}</td>
                    <td data-label="操作">
                      <div className="row-actions">
                        <button
                          className="icon-button--plain row-actions__trigger"
                          type="button"
                          aria-label={`${archive.date} 归档更多操作`}
                          aria-expanded={openArchiveMenu === archive.date}
                          onClick={() => setOpenArchiveMenu((current) => current === archive.date ? null : archive.date)}
                        >
                          <MoreVertical size={17} />
                        </button>
                        {openArchiveMenu === archive.date ? (
                          <div className="row-actions__menu" role="menu">
                            <button type="button" role="menuitem" onClick={() => { setOpenArchiveMenu(null); void showArchive(archive.date); }}>
                              查看内容
                            </button>
                            <button className="row-actions__danger" type="button" role="menuitem" onClick={() => { setOpenArchiveMenu(null); void removeArchive(archive.date); }}>
                              删除
                            </button>
                          </div>
                        ) : null}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">暂无历史归档。</p>
        )}
      </section>
      ) : null}

      {selectedArchive ? (
        <section className="panel">
          <div className="panel-title">
            <h2>{selectedArchive.date} 历史日志</h2>
            <button className="secondary-button" type="button" onClick={() => setSelectedArchive(null)}>
              收起
            </button>
          </div>
          <div className="archive-summary" aria-label="归档摘要">
            <div>
              <span className="archive-summary__label">日志条数</span>
              <strong>{parsedArchive?.entries.length ?? 0}</strong>
            </div>
            <div>
              <span className="archive-summary__label">归档文件</span>
              <strong>{selectedArchive.filename}</strong>
            </div>
            <div>
              <span className="archive-summary__label">文件大小</span>
              <strong>{selectedArchiveSummary ? formatFileSize(selectedArchiveSummary.size) : "未知"}</strong>
            </div>
            <div>
              <span className="archive-summary__label">最近归档</span>
              <strong>{formatArchiveTime(parsedArchive?.archived_at.at(-1))}</strong>
            </div>
          </div>

          {parsedArchive && parsedArchive.entries.length > 0 ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>时间</th>
                    <th>动作</th>
                    <th>对象</th>
                    <th>详情</th>
                  </tr>
                </thead>
                <tbody>
                  {parsedArchive.entries.map((entry) => {
                    const display = formatAuditLog(entry);
                    return (
                      <tr key={`${entry.created_at}-${entry.id}`}>
                        <td data-label="时间">{formatDateTime(entry.created_at)}</td>
                        <td data-label="动作">{display.action}</td>
                        <td data-label="对象">{display.target}</td>
                        <td data-label="详情" className="audit-detail">
                          <span>{auditTableSummary(entry, display.detail)}</span>
                          <button className="text-button" type="button" onClick={() => setSelectedLog(entry)}>详情</button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted">该归档暂时无法解析为日志条目，请刷新后重试。</p>
          )}
        </section>
      ) : null}

      {selectedLog ? (
        <div className="drawer-backdrop" role="presentation" onMouseDown={() => setSelectedLog(null)}>
          <aside className="drawer audit-detail-drawer" role="dialog" aria-modal="true" aria-label="日志详情" onMouseDown={(event) => event.stopPropagation()}>
            <header className="drawer__head">
              <div>
                <p className="eyebrow">日志详情</p>
                <h2 id="audit-detail-title">{formatAuditLog(selectedLog).action}</h2>
                <p className="muted">{formatDateTime(selectedLog.last_seen_at || selectedLog.created_at)}</p>
              </div>
              <button className="icon-button--plain" type="button" aria-label="关闭日志详情" onClick={() => setSelectedLog(null)}>
                <X size={20} />
              </button>
            </header>
            <AuditDetailDrawerBody log={selectedLog} />
          </aside>
        </div>
      ) : null}
    </main>
  );
}

function formatDateTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false });
}

function formatFileSize(value: number): string {
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function formatArchiveTime(value: string | undefined): string {
  return value ? formatDateTime(value) : "未知";
}

function SeverityBadge({ severity }: { severity: AuditLogItem["severity"] }) {
  const label = severity === "error" ? "严重" : severity === "warning" ? "警告" : "普通";
  const className = severity === "error" ? "severity-badge error" : severity === "warning" ? "severity-badge warning" : "severity-badge";
  return <span className={className}>{label}</span>;
}

function AuditDetailDrawerBody({ log }: { log: AuditLogRecord }) {
  const display = formatAuditLog(log);
  const detail = buildAuditDetailView(log);
  return (
    <div className="drawer-body audit-detail-drawer__body">
      <dl className="audit-detail-grid">
        <div><dt>事件类型</dt><dd>{display.action}</dd></div>
        <div><dt>对象</dt><dd>{display.target}</dd></div>
        <div><dt>结果</dt><dd><SeverityBadge severity={log.severity} /></dd></div>
        <div><dt>耗时/次数</dt><dd>{auditDurationOrCount(log)}</dd></div>
      </dl>

      {detail.question ? (
        <section className="audit-detail-section">
          <h3>完整问题</h3>
          <p>{detail.question}</p>
        </section>
      ) : null}
      {detail.answer !== null ? (
        <section className="audit-detail-section">
          <h3>完整回答</h3>
          <p>{detail.answer || "未生成回答"}</p>
        </section>
      ) : null}
      <section className="audit-detail-section">
        <h3>事件说明</h3>
        <pre className="audit-detail-pre">{detail.detailText}</pre>
      </section>
      {display.evidence.length > 0 ? (
        <section className="audit-detail-section">
          <h3>引用证据</h3>
          <AuditEvidenceList evidence={display.evidence} />
        </section>
      ) : null}
      {detail.parameters.length > 0 ? (
        <section className="audit-detail-section">
          <h3>运行参数</h3>
          <dl className="audit-detail-grid">
            {detail.parameters.map((item) => <div key={item.label}><dt>{item.label}</dt><dd>{item.value}</dd></div>)}
          </dl>
        </section>
      ) : null}
    </div>
  );
}

function auditEventKind(log: AuditLogRecord): AuditEventKind {
  if (log.severity === "error") return "exception";
  if (log.action === "qa_refused") return "refusal";
  if (log.target_type === "question" || log.action.startsWith("qa_")) return "qa";
  if (log.action.includes("upload")) return "upload";
  if (log.action.includes("parse")) return "parse";
  if (log.action.includes("index")) return "index";
  return "all";
}

function auditTableSummary(log: AuditLogRecord, formattedDetail: string): string {
  const parsed = parseAuditDetailJson(log);
  if (parsed.question) return truncateText(`问题：${parsed.question}`, 96);
  if (log.user_message) return truncateText(log.user_message, 96);
  return truncateText(formattedDetail.replace(/\s+/g, " "), 96);
}

function auditDurationOrCount(log: AuditLogRecord): string {
  const parsed = parseAuditDetailJson(log);
  const elapsed = valueByKeys(parsed.raw, ["elapsed_ms", "duration_ms", "latency_ms"]);
  if (typeof elapsed === "number") return `${(elapsed / 1000).toFixed(3)}s`;
  if (typeof elapsed === "string" && elapsed) return elapsed;
  return `${log.occurrence_count || 1} 次`;
}

function buildAuditDetailView(log: AuditLogRecord): {
  question: string;
  answer: string | null;
  detailText: string;
  parameters: Array<{ label: string; value: string }>;
} {
  const parsed = parseAuditDetailJson(log);
  return {
    question: parsed.question,
    answer: parsed.hasAnswer ? parsed.answer : null,
    detailText: auditDetailText(log, parsed.question || parsed.answer),
    parameters: [
      pair("事件码", log.action),
      pair("目标类型", log.target_type),
      pair("目标编号", log.target_id),
      pair("生成状态", parsed.generationStatus),
      pair("拒答原因", parsed.refusalReason),
      pair("引用数量", parsed.citationCount),
      pair("置信度", parsed.confidence),
      pair("事件键", log.event_key),
    ].filter((item): item is { label: string; value: string } => Boolean(item)),
  };
}

function auditDetailText(log: AuditLogRecord, hasStructuredQADetail: string): string {
  if (log.user_message) return log.user_message;
  if (log.summary) return log.summary;
  if (hasStructuredQADetail) return "问答事件已记录，完整问题、回答、引用证据和运行参数见本抽屉对应分区。";
  return log.detail || "无补充说明";
}

function parseAuditDetailJson(log: AuditLogRecord): {
  raw: Record<string, unknown>;
  question: string;
  answer: string;
  hasAnswer: boolean;
  generationStatus: string;
  refusalReason: string;
  citationCount: string;
  confidence: string;
} {
  const raw = parseRecord(log.details_json) || parseRecord(log.detail) || {};
  const answer = typeof raw.answer === "string" ? raw.answer : "";
  return {
    raw,
    question: typeof raw.question === "string" ? raw.question : "",
    answer,
    hasAnswer: Object.prototype.hasOwnProperty.call(raw, "answer"),
    generationStatus: stringValue(raw.generation_status),
    refusalReason: stringValue(raw.refusal_reason),
    citationCount: stringValue(raw.citation_count),
    confidence: typeof raw.confidence === "number" ? raw.confidence.toFixed(3) : stringValue(raw.confidence),
  };
}

function parseRecord(value: string | null): Record<string, unknown> | null {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed as Record<string, unknown> : null;
  } catch {
    return null;
  }
}

function valueByKeys(source: Record<string, unknown>, keys: string[]): unknown {
  for (const key of keys) {
    if (source[key] !== undefined && source[key] !== null) return source[key];
  }
  return null;
}

function truncateText(value: string, maxLength: number): string {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, Math.max(0, maxLength - 1))}…`;
}

function stringValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "";
  return String(value);
}

function AuditEvidenceList({ evidence }: { evidence: Citation[] }) {
  return (
    <ol className="audit-evidence-list">
      {evidence.map((item, index) => (
        <li key={item.metadata?.evidence_id ? String(item.metadata.evidence_id) : item.chunk_id}>
          <div className="audit-evidence-list__head">
            <strong>[{index + 1}] {item.filename}</strong>
            <span>{auditEvidenceLocation(item)}</span>
            <span className="evidence-role">{auditEvidenceRoleLabel(item.evidence_role)}</span>
          </div>
          <p>{item.excerpt || "未记录证据摘录。"}</p>
          <AuditEvidenceFacts citation={item} />
        </li>
      ))}
    </ol>
  );
}

function AuditEvidenceFacts({ citation }: { citation: Citation }) {
  const metadata = citation.metadata ?? {};
  const facts = [
    pair("颁发机构", metadata.issuing_authority),
    pair("证书编号", metadata.document_number),
    pair("颁发日期", metadata.publication_date),
    pair("材料主题", metadata.material_topic),
  ].filter((item): item is { label: string; value: string } => Boolean(item));
  if (!facts.length) return null;
  return (
    <dl className="evidence-facts">
      {facts.map((fact) => <div key={`${fact.label}-${fact.value}`}><dt>{fact.label}</dt><dd>{fact.value}</dd></div>)}
    </dl>
  );
}

function auditEvidenceLocation(citation: Citation): string {
  const parts = [];
  if (citation.section_title) parts.push(citation.section_title);
  if (citation.page_number) parts.push(`第 ${citation.page_number} 页`);
  return parts.length ? parts.join(" · ") : "未标注章节";
}

function auditEvidenceRoleLabel(role: string): string {
  const labels: Record<string, string> = {
    direct_evidence: "直接依据",
    related_context: "相关背景",
    expanded_context: "补充上下文",
  };
  return labels[role] ?? "引用依据";
}

function pair(label: string, value: unknown): { label: string; value: string } | null {
  if (value === null || value === undefined || value === "") return null;
  return { label, value: String(value) };
}
