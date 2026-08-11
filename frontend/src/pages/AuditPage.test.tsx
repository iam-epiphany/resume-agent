import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { deleteAuditArchive, getAuditArchive, listAuditArchives, listAuditLogs } from "../api/audit";
import { AUTH_TOKEN_KEY } from "../api/client";
import { AuthProvider } from "../state/authContext";
import type { AuditArchiveListResponse, AuditLogListResponse } from "../types/api";
import { AuditPage } from "./AuditPage";

vi.mock("../api/audit", () => ({
  deleteAuditArchive: vi.fn(),
  getAuditArchive: vi.fn(),
  listAuditArchives: vi.fn(),
  listAuditLogs: vi.fn(),
}));
// 管理员登录态：getMe 校验成功
vi.mock("../api/auth", () => ({
  getMe: vi.fn().mockResolvedValue({ authenticated: true, role: "admin" }),
  login: vi.fn().mockResolvedValue({ token: "test-token", token_type: "bearer", expires_at: "2099-01-01T00:00:00Z" }),
}));

const listAuditLogsMock = vi.mocked(listAuditLogs);
const listAuditArchivesMock = vi.mocked(listAuditArchives);

function renderAuditPage() {
  return render(
    <AuthProvider>
      <AuditPage />
    </AuthProvider>,
  );
}

describe("AuditPage", () => {
  beforeEach(() => {
    // 默认管理员已登录（全量日志 + 归档可见）
    localStorage.setItem(AUTH_TOKEN_KEY, "test-token");
    vi.mocked(deleteAuditArchive).mockReset();
    vi.mocked(getAuditArchive).mockReset();
    listAuditLogsMock.mockReset();
    listAuditArchivesMock.mockReset();
    listAuditArchivesMock.mockResolvedValue({ archives: [] });
  });

  it("filters audit events and keeps full QA content in the detail drawer", async () => {
    const details = {
      question: "秒杀项目怎么防超卖？",
      answer: "秒杀系统通过 Redis 预扣库存防止超卖，与项目介绍文档一致。[1]",
      refused: false,
      confidence: 0.87,
      citation_count: 1,
      generation_status: "completed",
      elapsed_ms: 2740,
      citations: [
        {
          document_id: "DOC-1",
          chunk_id: "DOC-1-CHUNK-1",
          filename: "项目介绍_秒杀平台.md",
          section_title: "超卖防护",
          page_number: null,
          excerpt: "秒杀系统通过 Redis 预扣库存防止超卖。",
          score: 0.8,
          rerank_score: 0.9,
          chunk_type: "paragraph",
          evidence_role: "direct_evidence",
          metadata: { issuing_authority: "河南大学", material_topic: "项目经历" },
        },
      ],
    };
    listAuditLogsMock.mockResolvedValue({
      logs: [
        auditLog(1, "qa_answered", "question", JSON.stringify(details), JSON.stringify(details), "info"),
        auditLog(2, "document_uploaded", "document", "文件已接收", null, "info"),
        auditLog(3, "document_index_failed", "document", "解析失败", null, "error"),
      ],
      limit: 100,
      offset: 0,
      returned: 3,
    } as AuditLogListResponse);

    renderAuditPage();

    expect(await screen.findByText("问题：秒杀项目怎么防超卖？")).toBeInTheDocument();
    expect(screen.queryByText(details.answer)).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("事件筛选"), { target: { value: "exception" } });
    expect(screen.getByText("文档入库失败")).toBeInTheDocument();
    expect(screen.queryByText("问题：秒杀项目怎么防超卖？")).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("事件筛选"), { target: { value: "qa" } });
    fireEvent.click(screen.getByRole("button", { name: "详情" }));

    expect(await screen.findByRole("dialog", { name: "日志详情" })).toBeInTheDocument();
    expect(screen.getByText(details.answer)).toBeInTheDocument();
    expect(screen.getAllByText(/项目介绍_秒杀平台\.md/).length).toBeGreaterThan(0);
    expect(screen.getAllByText("2.740s").length).toBeGreaterThan(0);

    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "日志详情" })).not.toBeInTheDocument());
  });

  it("keeps archive row actions inside the more menu", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    listAuditLogsMock.mockResolvedValue({
      logs: [],
      limit: 100,
      offset: 0,
      returned: 0,
    } as AuditLogListResponse);
    listAuditArchivesMock.mockResolvedValue({
      archives: [{
        date: "2026-07-20",
        filename: "audit-2026-07-20.jsonl",
        size: 2048,
        updated_at: "2026-07-20T23:59:00Z",
      }],
    } as AuditArchiveListResponse);
    vi.mocked(getAuditArchive).mockResolvedValue({
      date: "2026-07-20",
      filename: "audit-2026-07-20.jsonl",
      content: "",
    });
    vi.mocked(deleteAuditArchive).mockResolvedValue({
      date: "2026-07-20",
      deleted: true,
    });

    renderAuditPage();

    expect(await screen.findByText("audit-2026-07-20.jsonl")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "查看内容" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "删除" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "2026-07-20 归档更多操作" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "查看内容" }));
    expect(getAuditArchive).toHaveBeenCalledWith("2026-07-20");

    fireEvent.click(screen.getByRole("button", { name: "2026-07-20 归档更多操作" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "删除" }));
    expect(confirmSpy).toHaveBeenCalledWith("确认删除 2026-07-20 的日志归档？此操作不可恢复。");
    await waitFor(() => expect(deleteAuditArchive).toHaveBeenCalledWith("2026-07-20"));

    confirmSpy.mockRestore();
  });

  it("anonymous visitors see a privacy notice and no logs or archive section", async () => {
    localStorage.removeItem(AUTH_TOKEN_KEY);

    render(
      <AuthProvider>
        <AuditPage />
      </AuthProvider>,
    );

    // 匿名：问答记录属隐私，仅显示"仅管理员可见"提示；不调任何日志接口
    expect(await screen.findByText("仅管理员可见")).toBeInTheDocument();
    expect(listAuditLogsMock).not.toHaveBeenCalled();
    expect(listAuditArchivesMock).not.toHaveBeenCalled();
    // 历史归档区不渲染；事件筛选不含知识库管理类型
    expect(screen.queryByText("历史日志归档")).not.toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "上传" })).not.toBeInTheDocument();
  });
});

function auditLog(
  id: number,
  action: string,
  targetType: string,
  detail: string,
  detailsJson: string | null,
  severity: "info" | "warning" | "error",
): AuditLogListResponse["logs"][number] {
  return {
    id,
    action,
    target_type: targetType,
    target_id: `TARGET-${id}`,
    detail,
    severity,
    event_key: null,
    summary: null,
    user_message: null,
    details_json: detailsJson,
    first_seen_at: null,
    last_seen_at: null,
    occurrence_count: 1,
    resolved: false,
    created_at: "2026-07-18T00:00:00Z",
  };
}
