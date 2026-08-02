import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  deleteDocument,
  deleteDocumentsBulk,
  getDocument,
  getDocumentProcessing,
  listDocuments,
  preflightDocumentUploads,
  rebuildDocumentIndex,
  uploadDocument,
  uploadDocumentsBatch,
} from "../api/documents";
import { getRagHealth } from "../api/system";
import type { DocumentListResponse, RagHealthResponse } from "../types/api";
import { DocumentsPage } from "./DocumentsPage";

vi.mock("../api/documents", () => ({
  deleteDocument: vi.fn(),
  deleteDocumentsBulk: vi.fn(),
  getDocument: vi.fn(),
  getDocumentProcessing: vi.fn(),
  listDocuments: vi.fn(),
  preflightDocumentUploads: vi.fn(),
  rebuildDocumentIndex: vi.fn(),
  uploadDocument: vi.fn(),
  uploadDocumentsBatch: vi.fn(),
}));

vi.mock("../api/system", () => ({ getRagHealth: vi.fn() }));

const listDocumentsMock = vi.mocked(listDocuments);
const deleteDocumentsBulkMock = vi.mocked(deleteDocumentsBulk);
const getRagHealthMock = vi.mocked(getRagHealth);
const getDocumentProcessingMock = vi.mocked(getDocumentProcessing);
const preflightDocumentUploadsMock = vi.mocked(preflightDocumentUploads);
const uploadDocumentMock = vi.mocked(uploadDocument);
const uploadDocumentsBatchMock = vi.mocked(uploadDocumentsBatch);

const health: RagHealthResponse = {
  build_id: "test",
  offline_mode: false,
  embedding_model_ready: true,
  reranker_model_ready: true,
  embedding_model_path: "bge-m3",
  reranker_model_path: "reranker",
  qdrant_ready: true,
  qdrant_collection: "resumemind_chunks",
  qdrant_collection_ready: true,
  sqlite_ready: true,
  libreoffice_ready: true,
  antiword_ready: true,
  libreoffice_version: "24.2",
  antiword_version: "0.37",
  index_tasks: {},
  qa_tasks: {},
  model_runtime: {},
  ready: true,
  model_device: {
    requested_device: "cpu",
    selected_device: "cpu",
    torch_version: "2.8",
    cuda_available: false,
    cuda_device_count: 0,
    cuda_device_name: null,
    fallback_reason: null,
  },
};

describe("DocumentsPage batch upload", () => {
  beforeEach(() => {
    vi.mocked(deleteDocument).mockReset();
    deleteDocumentsBulkMock.mockReset();
    vi.mocked(getDocument).mockReset();
    getDocumentProcessingMock.mockReset();
    listDocumentsMock.mockReset();
    preflightDocumentUploadsMock.mockReset();
    vi.mocked(rebuildDocumentIndex).mockReset();
    uploadDocumentMock.mockReset();
    uploadDocumentsBatchMock.mockReset();
    getRagHealthMock.mockReset();
    vi.stubGlobal("crypto", {
      randomUUID: () => "test-request-id",
      subtle: {
        digest: vi.fn(async () => new Uint8Array(32).buffer),
      },
    });

    listDocumentsMock.mockResolvedValue({ documents: [] } as DocumentListResponse);
    getRagHealthMock.mockResolvedValue(health);
    getDocumentProcessingMock.mockResolvedValue({
      document_id: "DOC-BATCH-1",
      task_id: "task-1",
      status: "completed",
      stage: "completed",
      completed_units: 1,
      total_units: 1,
      error_code: null,
      error_message: null,
      error: null,
      retry_count: 0,
      updated_at: "2026-07-18T00:00:00+00:00",
    });
  });

  it("submits multiple selected files and shows partial failure details", async () => {
    preflightDocumentUploadsMock.mockImplementation(async (items) => ({
      items: items.map((item) => ({
        client_file_id: item.client_file_id,
        filename: item.filename,
        status: "ready",
        existing_document: null,
        error_message: null,
      })),
    }));
    uploadDocumentsBatchMock.mockResolvedValue({
      batch_id: "batch-test",
      accepted_count: 1,
      failed_count: 1,
      items: [
        {
          filename: "rules.txt",
          status: "accepted",
          document_id: "DOC-BATCH-1",
          task_id: "task-1",
          stage: "queued",
          size: 12,
          error_message: null,
        },
        {
          filename: "empty.txt",
          status: "failed",
          document_id: null,
          task_id: null,
          stage: null,
          size: null,
          error_message: "上传文档不能为空",
        },
      ],
    });

    render(<DocumentsPage />);

    await screen.findByText("新增知识源");
    const fileInputs = document.querySelectorAll<HTMLInputElement>("input[type='file']");
    const batchInput = fileInputs[1];
    fireEvent.change(batchInput, {
      target: {
        files: [
          new File(["rules"], "rules.txt", { type: "text/plain" }),
          new File([""], "empty.txt", { type: "text/plain" }),
        ],
      },
    });

    await waitFor(() => expect(uploadDocumentsBatchMock).toHaveBeenCalledTimes(1));
    expect(uploadDocumentsBatchMock.mock.calls[0][0]).toHaveLength(2);
    expect(await screen.findByText("批量上传结果")).toBeInTheDocument();
    expect(screen.getByText("上传文档不能为空")).toBeInTheDocument();
  });

  it("shows an exact duplicate dialog before single upload", async () => {
    preflightDocumentUploadsMock.mockResolvedValue({
      items: [
        {
          client_file_id: "single",
          filename: "rules.txt",
          status: "exact_duplicate",
          existing_document: {
            document_id: "DOC-OLD",
            filename: "rules.txt",
            size: 5,
            file_sha256: "0".repeat(64),
            status: "indexed",
            uploaded_at: "2026-07-18T00:00:00+00:00",
            chunk_count: 1,
          },
          error_message: "文件已存在于知识库中。",
        },
      ],
    });

    render(<DocumentsPage />);

    await screen.findByText("新增知识源");
    const fileInputs = document.querySelectorAll<HTMLInputElement>("input[type='file']");
    fireEvent.change(fileInputs[0], {
      target: { files: [new File(["rules"], "rules.txt", { type: "text/plain" })] },
    });

    expect(await screen.findByText("文件已存在")).toBeInTheDocument();
    expect(screen.getAllByText("该文件内容已存在于知识库中，已跳过上传。").length).toBeGreaterThan(0);
    expect(uploadDocumentMock).not.toHaveBeenCalled();
  });

  it("preflights batch files and leaves conflicts for user action", async () => {
    preflightDocumentUploadsMock.mockImplementation(async (items) => ({
      items: items.map((item) => item.filename === "ready.txt" ? {
        client_file_id: item.client_file_id,
        filename: item.filename,
        status: "ready",
        existing_document: null,
        error_message: null,
      } : {
        client_file_id: item.client_file_id,
        filename: item.filename,
        status: "name_conflict",
        existing_document: {
          document_id: "DOC-OLD",
          filename: "rules.txt",
          size: 10,
          file_sha256: "1".repeat(64),
          status: "indexed",
          uploaded_at: "2026-07-18T00:00:00+00:00",
          chunk_count: 2,
        },
        error_message: "知识库中已存在同名文件。",
      }),
    }));
    uploadDocumentsBatchMock.mockResolvedValue({
      batch_id: "batch-ready",
      accepted_count: 1,
      failed_count: 0,
      items: [
        {
          filename: "ready.txt",
          status: "accepted",
          document_id: "DOC-BATCH-1",
          task_id: "task-1",
          stage: "queued",
          size: 12,
          error_message: null,
        },
      ],
    });

    render(<DocumentsPage />);

    await screen.findByText("新增知识源");
    const fileInputs = document.querySelectorAll<HTMLInputElement>("input[type='file']");
    fireEvent.change(fileInputs[1], {
      target: {
        files: [
          new File(["ready"], "ready.txt", { type: "text/plain" }),
          new File(["rules"], "rules.txt", { type: "text/plain" }),
        ],
      },
    });

    await waitFor(() => expect(uploadDocumentsBatchMock).toHaveBeenCalledTimes(1));
    expect(uploadDocumentsBatchMock.mock.calls[0][0]).toHaveLength(1);
    expect(await screen.findByText("批量上传待处理")).toBeInTheDocument();
    expect(screen.getByText("知识库中已存在同名文件，请覆盖旧文件或重命名新文件。")).toBeInTheDocument();
  });

  it("submits selected documents through the bulk delete action", async () => {
    listDocumentsMock
      .mockResolvedValueOnce({
        documents: [
          documentSummary("DOC-KEEP", "处理中.txt", "indexing"),
          documentSummary("DOC-DELETE-1", "材料A.txt", "indexed"),
          documentSummary("DOC-DELETE-2", "材料B.txt", "indexed"),
        ],
      } as DocumentListResponse)
      .mockResolvedValue({
        documents: [documentSummary("DOC-KEEP", "处理中.txt", "indexing")],
      } as DocumentListResponse);
    deleteDocumentsBulkMock.mockResolvedValue({
      requested_count: 2,
      deleted_count: 2,
      failed_count: 0,
      items: [
        { document_id: "DOC-DELETE-1", filename: "材料A.txt", status: "deleted", message: "删除成功" },
        { document_id: "DOC-DELETE-2", filename: "材料B.txt", status: "deleted", message: "删除成功" },
      ],
    });

    render(<DocumentsPage />);

    await screen.findByText("材料A.txt");
    expect(screen.queryByLabelText("选择 材料A.txt")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "选择" }));
    fireEvent.click(screen.getByRole("button", { name: "全选" }));
    fireEvent.click(screen.getByRole("button", { name: "批量删除" }));

    expect(await screen.findByText("确认批量删除文档")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "确认批量删除" }));

    await waitFor(() => expect(deleteDocumentsBulkMock).toHaveBeenCalledTimes(1));
    expect(deleteDocumentsBulkMock).toHaveBeenCalledWith(["DOC-DELETE-1", "DOC-DELETE-2"]);
    expect(await screen.findByText("已删除 2 份文档")).toBeInTheDocument();
  });

  it("keeps ledger checkboxes hidden until selection mode is enabled and supports cancelling selection", async () => {
    listDocumentsMock.mockResolvedValue({
      documents: [documentSummary("DOC-DELETE-1", "材料A.txt", "indexed")],
    } as DocumentListResponse);

    render(<DocumentsPage />);

    await screen.findByText("材料A.txt");
    expect(screen.queryByLabelText("选择 材料A.txt")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "选择" }));
    fireEvent.click(screen.getByLabelText("选择 材料A.txt"));
    expect(screen.getByText("已选 1 份")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "取消选择" }));
    expect(screen.queryByLabelText("选择 材料A.txt")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "选择" })).toBeInTheDocument();
  });
});

function documentSummary(documentId: string, filename: string, status: "indexed" | "indexing"): DocumentListResponse["documents"][number] {
  return {
    document_id: documentId,
    filename,
    file_type: "txt",
    size: 12,
    chunk_count: status === "indexed" ? 1 : 0,
    uploaded_at: "2026-07-18T00:00:00+00:00",
    status,
    index_version: "test",
    index_error: null,
    metadata: {},
  };
}
