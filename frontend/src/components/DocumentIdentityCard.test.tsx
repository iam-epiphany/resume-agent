import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { confirmDocumentMetadata, updateDocumentMetadata } from "../api/documents";
import type { DocumentDetailResponse, DocumentMetadata } from "../types/api";
import { DocumentIdentityCard } from "./DocumentIdentityCard";

vi.mock("../api/documents", () => ({
  confirmDocumentMetadata: vi.fn(),
  updateDocumentMetadata: vi.fn(),
}));

const updateMock = vi.mocked(updateDocumentMetadata);
const confirmMock = vi.mocked(confirmDocumentMetadata);

describe("DocumentIdentityCard", () => {
  beforeEach(() => {
    updateMock.mockReset();
    confirmMock.mockReset();
  });

  it("shows unknown fields without treating them as errors", () => {
    render(<DocumentIdentityCard document={detail({})} onMetadataChange={vi.fn()} />);

    expect(screen.getByText("材料信息卡")).toBeInTheDocument();
    expect(screen.getByText("待核对")).toBeInTheDocument();
    expect(screen.getAllByText("未知").length).toBeGreaterThan(3);
    expect(screen.getByText("材料信息用于检索与来源提示，系统按知识库材料作答，不对证书真伪作出鉴定。")).toBeInTheDocument();
  });

  it("saves only changed fields and then confirms the resulting snapshot", async () => {
    const onMetadataChange = vi.fn();
    updateMock.mockResolvedValue({
      document_id: "DOC-IDENTITY",
      metadata: {
        title: "人工核对标题",
        identity_review_status: "unreviewed",
      },
      metadata_refreshed: true,
      refresh_warning: null,
    });
    confirmMock.mockResolvedValue({
      document_id: "DOC-IDENTITY",
      metadata: {
        title: "人工核对标题",
        identity_review_status: "confirmed",
        identity_reviewed_at: "2026-07-22T08:00:00+00:00",
        identity_reviewed_snapshot_hash: "a".repeat(64),
      },
      metadata_refreshed: true,
      refresh_warning: null,
    });
    render(<DocumentIdentityCard document={detail({ title: "系统标题" })} onMetadataChange={onMetadataChange} />);

    fireEvent.click(screen.getByRole("button", { name: "编辑材料信息" }));
    fireEvent.change(screen.getByLabelText("材料标题"), { target: { value: "人工核对标题" } });
    fireEvent.click(screen.getByRole("button", { name: "保存并确认已核对" }));

    await waitFor(() => expect(updateMock).toHaveBeenCalledTimes(1));
    expect(updateMock).toHaveBeenCalledWith("DOC-IDENTITY", { title: "人工核对标题" });
    expect(confirmMock).toHaveBeenCalledWith("DOC-IDENTITY");
    expect(onMetadataChange).toHaveBeenCalledWith(
      expect.objectContaining({ identity_review_status: "confirmed" }),
      "材料信息已人工核对",
    );
  });

  it("allows confirmation while fields remain unknown", async () => {
    confirmMock.mockResolvedValue({
      document_id: "DOC-IDENTITY",
      metadata: {
        identity_review_status: "confirmed",
        identity_reviewed_snapshot_hash: "b".repeat(64),
      },
      metadata_refreshed: false,
      refresh_warning: null,
    });
    const onMetadataChange = vi.fn();
    render(<DocumentIdentityCard document={detail({})} onMetadataChange={onMetadataChange} />);

    fireEvent.click(screen.getByRole("button", { name: "编辑材料信息" }));
    fireEvent.click(screen.getByRole("button", { name: "保存并确认已核对" }));

    await waitFor(() => expect(confirmMock).toHaveBeenCalledTimes(1));
    expect(updateMock).not.toHaveBeenCalled();
    expect(onMetadataChange).toHaveBeenCalledWith(
      expect.objectContaining({ identity_review_status: "confirmed" }),
      "材料信息已人工核对",
    );
  });
});

function detail(metadata: DocumentMetadata): DocumentDetailResponse {
  return {
    document_id: "DOC-IDENTITY",
    filename: "证书说明.docx",
    file_type: "docx",
    size: 128,
    chunk_count: 0,
    uploaded_at: "2026-07-22T00:00:00+00:00",
    status: "indexed",
    index_version: "test",
    index_error: null,
    metadata: {
      source_filename: "证书说明.docx",
      file_sha256: "1".repeat(64),
      identity_review_status: "unreviewed",
      metadata_provenance: {},
      ...metadata,
    },
    chunks: [],
    chunk_total: 0,
    chunk_offset: 0,
    chunk_limit: 50,
  };
}
