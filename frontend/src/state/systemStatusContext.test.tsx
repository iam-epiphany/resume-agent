import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { listDocuments } from "../api/documents";
import { getRagHealth } from "../api/system";
import { AUTH_TOKEN_KEY } from "../api/client";
import { AuthProvider } from "../state/authContext";
import type { DocumentListResponse, RagHealthResponse } from "../types/api";
import { isKnowledgeBaseReady, SystemStatusProvider, useSystemStatus } from "./systemStatusContext";

vi.mock("../api/documents", () => ({ listDocuments: vi.fn() }));
vi.mock("../api/system", () => ({ getRagHealth: vi.fn() }));
// 管理员登录态：getMe 校验成功
vi.mock("../api/auth", () => ({
  getMe: vi.fn().mockResolvedValue({ authenticated: true, role: "admin" }),
  login: vi.fn().mockResolvedValue({ token: "test-token", token_type: "bearer", expires_at: "2099-01-01T00:00:00Z" }),
}));

const listDocumentsMock = vi.mocked(listDocuments);
const getRagHealthMock = vi.mocked(getRagHealth);

function Probe() {
  const status = useSystemStatus();
  return (
    <div>
      <span data-testid="total">{status.documentCount}</span>
      <span data-testid="indexed">{status.indexedDocumentCount}</span>
      <span data-testid="processing">{status.processingDocumentCount}</span>
      <span data-testid="ready">{String(isKnowledgeBaseReady(status))}</span>
    </div>
  );
}

describe("SystemStatusProvider", () => {
  beforeEach(() => {
    // 默认管理员已登录（系统状态属后台信息，仅登录后轮询）
    localStorage.setItem(AUTH_TOKEN_KEY, "test-token");
    listDocumentsMock.mockReset();
    getRagHealthMock.mockReset();
  });

  it("derives displayed counts from the live document response", async () => {
    getRagHealthMock.mockResolvedValue({ ready: true } as RagHealthResponse);
    listDocumentsMock.mockResolvedValue({
      documents: [
        { document_id: "doc-1", filename: "材料一.pdf", status: "indexed" },
        { document_id: "doc-2", filename: "材料二.docx", status: "indexed" },
        { document_id: "doc-3", filename: "报表.xlsx", status: "indexing" },
      ],
    } as DocumentListResponse);

    render(
      <AuthProvider>
        <SystemStatusProvider><Probe /></SystemStatusProvider>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("total")).toHaveTextContent("3"));
    expect(screen.getByTestId("indexed")).toHaveTextContent("2");
    expect(screen.getByTestId("processing")).toHaveTextContent("1");
    expect(screen.getByTestId("ready")).toHaveTextContent("true");
  });

  it("does not poll admin-only endpoints when anonymous", async () => {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    render(
      <AuthProvider>
        <SystemStatusProvider><Probe /></SystemStatusProvider>
      </AuthProvider>,
    );

    // 匿名不轮询后台接口（避免匿名 401；前台就绪走公开 /api/qa/status）
    expect(listDocumentsMock).not.toHaveBeenCalled();
    expect(getRagHealthMock).not.toHaveBeenCalled();
    expect(screen.getByTestId("ready")).toHaveTextContent("false");
  });
});
