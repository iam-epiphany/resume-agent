import type { AuditLogItem, Citation } from "../types/api";

interface AuditDisplay {
  action: string;
  target: string;
  detail: string;
  evidence: Citation[];
}

export interface ParsedAuditArchiveEntry {
  id: number;
  created_at: string;
  action: string;
  target_type: string;
  target_id: string | null;
  detail: string;
  severity: "info" | "warning" | "error";
  event_key: string | null;
  summary: string | null;
  user_message: string | null;
  details_json: string | null;
  first_seen_at: string | null;
  last_seen_at: string | null;
  occurrence_count: number;
  resolved: boolean;
}

export interface ParsedAuditArchive {
  archived_at: string[];
  entries: ParsedAuditArchiveEntry[];
}

interface QAAuditDetail {
  question?: string;
  answer?: string | null;
  refused?: boolean;
  confidence?: number;
  citation_count?: number;
  generation_status?: string;
  refusal_reason?: string | null;
  citations?: unknown;
}

const ACTION_LABELS: Record<string, string> = {
  document_uploaded: "文档上传",
  document_indexed: "文档已入库",
  document_index_failed: "文档入库失败",
  document_deleted: "文档删除",
  document_delete_failed: "文档删除失败",
  qa_answered: "提问",
  qa_refused: "提问未回答",
  qa_context_built: "提问",
  qa_cancelled: "停止生成",
};

const TARGET_LABELS: Record<string, string> = {
  document: "文档",
  question: "问答",
};

export function formatAuditLog(log: AuditLogItem): AuditDisplay {
  return {
    action: log.summary || ACTION_LABELS[log.action] || log.action,
    target: formatTarget(log),
    detail: isQAAuditAction(log.action) ? formatQADetail(log) : log.user_message || formatDetail(log),
    evidence: isQAAuditAction(log.action) ? extractQAEvidence(log) : [],
  };
}

export function parseAuditArchiveContent(content: string): ParsedAuditArchive {
  const archived_at = Array.from(content.matchAll(/^归档时间：(.+)$/gm), (match) => match[1].trim());
  const headings = Array.from(content.matchAll(/^## (.+?) · (.+)$/gm));
  const entries = headings.map((heading, index) => {
    const blockStart = heading.index ?? 0;
    const nextHeading = headings[index + 1];
    const blockEnd = nextHeading?.index ?? content.length;
    const block = content.slice(blockStart, blockEnd);
    const targetType = matchLine(block, "对象类型") || "unknown";
    const rawTargetId = matchLine(block, "对象编号");
    const detail = matchCodeBlock(block) || matchLine(block, "详情") || "";
    const detailsJson = matchLabeledCodeBlock(block, "结构化详情");
    const severity = cleanSeverity(matchLine(block, "级别"));

    return {
      id: index + 1,
      created_at: heading[1].trim(),
      action: heading[2].trim(),
      target_type: targetType,
      target_id: rawTargetId && rawTargetId !== "无" ? rawTargetId : null,
      detail,
      severity,
      event_key: null,
      summary: matchLine(block, "摘要"),
      user_message: null,
      details_json: detailsJson,
      first_seen_at: null,
      last_seen_at: null,
      occurrence_count: Number(matchLine(block, "出现次数") || 1),
      resolved: false,
    };
  });

  return { archived_at, entries };
}

function cleanSeverity(value: string | null): "info" | "warning" | "error" {
  if (value === "warning" || value === "警告") {
    return "warning";
  }
  if (value === "error" || value === "严重") {
    return "error";
  }
  return "info";
}

function formatTarget(log: AuditLogItem): string {
  const target = TARGET_LABELS[log.target_type] ?? log.target_type;
  return log.target_id ? `${target}：${log.target_id}` : target;
}

function formatDetail(log: AuditLogItem): string {
  if (log.action === "document_indexed") {
    return formatIndexedDetail(log.detail);
  }
  if (isQAAuditAction(log.action)) {
    return formatQADetail(log);
  }

  return log.detail || "无补充说明";
}

function formatIndexedDetail(detail: string): string {
  const match = /^chunks=(\d+)$/i.exec(detail.trim());
  if (!match) {
    return detail || "入库完成";
  }

  return `生成 ${match[1]} 个内容片段`;
}

function formatQADetail(log: AuditLogItem | ParsedAuditArchiveEntry): string {
  const parsed = parseQADetail(log.detail) || parseQADetail(log.details_json || "");
  if (!parsed?.question && !parsed?.answer) {
    return log.detail || log.user_message || "无补充说明";
  }
  const parts = [];
  if (parsed.question) {
    parts.push(`问题：${parsed.question}`);
  }
  parts.push(`回答：${parsed.answer || "未生成回答"}`);
  return parts.join("\n");
}

function isQAAuditAction(action: string): boolean {
  return action === "qa_answered" || action === "qa_refused" || action === "qa_context_built";
}

function extractQAEvidence(log: AuditLogItem | ParsedAuditArchiveEntry): Citation[] {
  const parsed = parseQADetail(log.details_json || "") || parseQADetail(log.detail);
  if (!Array.isArray(parsed?.citations)) {
    return [];
  }
  return parsed.citations.map(toCitation).filter((citation): citation is Citation => citation !== null);
}

function parseQADetail(detail: string): QAAuditDetail | null {
  try {
    const value = JSON.parse(detail) as QAAuditDetail;
    return value && typeof value === "object" ? value : null;
  } catch {
    return null;
  }
}

function toCitation(value: unknown): Citation | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const raw = value as Record<string, unknown>;
  if (!isNonEmptyString(raw.document_id) || !isNonEmptyString(raw.chunk_id) || !isNonEmptyString(raw.filename)) {
    return null;
  }
  return {
    document_id: raw.document_id,
    chunk_id: raw.chunk_id,
    filename: raw.filename,
    section_title: typeof raw.section_title === "string" ? raw.section_title : null,
    page_number: typeof raw.page_number === "number" ? raw.page_number : null,
    excerpt: typeof raw.excerpt === "string" ? raw.excerpt : "",
    score: typeof raw.score === "number" ? raw.score : null,
    rerank_score: typeof raw.rerank_score === "number" ? raw.rerank_score : null,
    chunk_type: typeof raw.chunk_type === "string" ? raw.chunk_type : "paragraph",
    evidence_role: typeof raw.evidence_role === "string" ? raw.evidence_role : "related_context",
    metadata: isRecord(raw.metadata) ? raw.metadata : {},
  };
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function matchLine(block: string, label: string): string | null {
  const match = new RegExp(`^- ${label}：(.+)$`, "m").exec(block);
  return match ? match[1].trim() : null;
}

function matchCodeBlock(block: string): string | null {
  const match = /```text\r?\n([\s\S]*?)\r?\n```/.exec(block);
  return match ? match[1].trim() : null;
}

function matchLabeledCodeBlock(block: string, label: string): string | null {
  const match = new RegExp(`^- ${label}：\\s*\\r?\\n\\s*\\r?\\n\`\`\`(?:json|text)\\r?\\n([\\s\\S]*?)\\r?\\n\`\`\``, "m").exec(block);
  return match ? match[1].trim() : null;
}
