import { describe, expect, it } from "vitest";

import type { AuditLogItem } from "../types/api";
import { formatAuditLog } from "./audit";

describe("formatAuditLog", () => {
  it("keeps QA audit summaries focused on question and answer while exposing evidence separately", () => {
    const details = {
      question: "你参与过哪些项目？",
      answer: "项目经历包括外卖平台、REV 与秒杀系统。[1]",
      refused: false,
      citation_count: 1,
      citations: [
        {
          document_id: "DOC-1",
          chunk_id: "DOC-1-CHUNK-1",
          filename: "rules.txt",
          section_title: "项目经历",
          page_number: null,
          excerpt: "项目经历包括外卖平台、REV 与秒杀系统。",
          score: 0.8,
          rerank_score: 0.9,
          chunk_type: "paragraph",
          evidence_role: "direct_evidence",
          metadata: {},
        },
      ],
    };
    const log: AuditLogItem = {
      id: 1,
      action: "qa_context_built",
      target_type: "question",
      target_id: null,
      detail: JSON.stringify({ ...details, citations: undefined }),
      severity: "info",
      event_key: null,
      summary: "问答完成",
      user_message: "已回答：你参与过哪些项目？",
      details_json: JSON.stringify(details),
      first_seen_at: null,
      last_seen_at: null,
      occurrence_count: 1,
      resolved: false,
      created_at: "2026-07-18T00:00:00Z",
    };

    const display = formatAuditLog(log);

    expect(display.detail).toBe("问题：你参与过哪些项目？\n回答：项目经历包括外卖平台、REV 与秒杀系统。[1]");
    expect(display.detail).not.toContain("引用数量");
    expect(display.detail).not.toContain("状态");
    expect(display.evidence).toHaveLength(1);
    expect(display.evidence[0].filename).toBe("rules.txt");
    expect(display.evidence[0].excerpt).toBe("项目经历包括外卖平台、REV 与秒杀系统。");
  });
});
