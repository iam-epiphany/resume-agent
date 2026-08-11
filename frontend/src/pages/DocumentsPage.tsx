import { AlertCircle, AlertTriangle, CheckCircle2, FileUp, MoreVertical, RefreshCw, Search, Trash2, X } from "lucide-react";
import type { ChangeEvent } from "react";
import { useCallback, useEffect, useRef, useState } from "react";

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
import { StatusBadge } from "../components/StatusBadge";
import { DocumentIdentityCard } from "../components/DocumentIdentityCard";
import type {
  ChunkSummary,
  DocumentDetailResponse,
  DocumentMetadata,
  DocumentSummary,
  DocumentUploadPreflightItem,
  RagHealthResponse,
} from "../types/api";

interface UploadNotice {
  documentId: string;
  taskId: string;
  stage: string;
  completedUnits: number | null;
  totalUnits: number | null;
  completed: boolean;
}

interface BatchUploadFileNotice {
  filename: string;
  documentId: string | null;
  taskId: string | null;
  stage: string | null;
  completedUnits: number | null;
  totalUnits: number | null;
  completed: boolean;
  status: "accepted" | "failed" | "duplicate" | "conflict";
  errorMessage: string | null;
}

interface BatchUploadNotice {
  batchId: string;
  acceptedCount: number;
  failedCount: number;
  completed: boolean;
  items: BatchUploadFileNotice[];
}

interface ToastNotice {
  message: string;
  tone: "success" | "error";
}

interface UploadConflictNotice {
  file: File;
  fileSha256: string;
  result: DocumentUploadPreflightItem;
  renameValue: string;
}

interface BatchConflictIssue {
  id: string;
  file: File;
  fileSha256: string;
  result: DocumentUploadPreflightItem;
  renameValue: string;
  resolving: boolean;
}

const DOCUMENT_PAGE_SIZE = 20;
const CHUNK_PAGE_SIZE = 50;

export function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedDetail, setSelectedDetail] = useState<DocumentDetailResponse | null>(null);
  const [uploadNotice, setUploadNotice] = useState<UploadNotice | null>(null);
  const [batchUploadNotice, setBatchUploadNotice] = useState<BatchUploadNotice | null>(null);
  const [uploadConflict, setUploadConflict] = useState<UploadConflictNotice | null>(null);
  const [batchConflictIssues, setBatchConflictIssues] = useState<BatchConflictIssue[]>([]);
  const [toastNotice, setToastNotice] = useState<ToastNotice | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<DocumentSummary | null>(null);
  const [isSelectionMode, setIsSelectionMode] = useState(false);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false);
  const [openDocumentMenu, setOpenDocumentMenu] = useState<string | null>(null);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null);
  const [isBulkDeleting, setIsBulkDeleting] = useState(false);
  const [rebuildingDocumentId, setRebuildingDocumentId] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState("all");
  const [typeFilter, setTypeFilter] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [pageNumber, setPageNumber] = useState(1);
  const [ragHealth, setRagHealth] = useState<RagHealthResponse | null>(null);
  const [runtimeWarning, setRuntimeWarning] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isBatchUploading, setIsBatchUploading] = useState(false);
  const [isResolvingBatchIssues, setIsResolvingBatchIssues] = useState(false);
  const [uploadBytes, setUploadBytes] = useState<{ loaded: number; total: number } | null>(null);
  const [batchUploadBytes, setBatchUploadBytes] = useState<{ loaded: number; total: number; count: number } | null>(null);
  const deleteAbortController = useRef<AbortController | null>(null);
  const deleteDialogRef = useRef<HTMLDivElement | null>(null);
  const deleteCancelButtonRef = useRef<HTMLButtonElement | null>(null);

  const refreshKnowledgeBaseView = useCallback(async (background = false) => {
    if (!background) {
      setIsRefreshing(true);
    }
    try {
      const result = await listDocuments();
      setDocuments(result.documents);
      try {
        const health = await getRagHealth();
        setRagHealth(health);
        setRuntimeWarning(buildRuntimeWarning(health));
      } catch {
        setRagHealth(null);
        setRuntimeWarning("无法读取 RAG 运行状态，文档数量不能代表问答索引已经可用。");
      }
      setLoadError(null);
      return result.documents;
    } catch (error) {
      const message = error instanceof Error ? error.message : "知识库列表暂时无法加载。";
      setLoadError(message);
      return [];
    } finally {
      setIsLoading(false);
      if (!background) {
        setIsRefreshing(false);
      }
    }
  }, []);

  useEffect(() => {
    void refreshKnowledgeBaseView();
  }, [refreshKnowledgeBaseView]);

  useEffect(() => {
    const hasPendingIndex = documents.some(
      (document) => ["uploaded", "index_queued", "indexing", "deleting"].includes(document.status),
    );
    if (!hasPendingIndex) {
      return;
    }
    const timer = window.setTimeout(() => void refreshKnowledgeBaseView(), 3000);
    return () => window.clearTimeout(timer);
  }, [documents, refreshKnowledgeBaseView]);

  useEffect(() => {
    if (!uploadNotice || uploadNotice.completed) {
      return;
    }
    let stopped = false;
    let timer: number | undefined;
    const documentId = uploadNotice.documentId;

    async function pollProcessing() {
      try {
        const task = await getDocumentProcessing(documentId);
        if (stopped) {
          return;
        }
        setUploadNotice((current) => current && current.documentId === task.document_id ? {
          ...current,
          stage: task.stage,
          completedUnits: task.completed_units,
          totalUnits: task.total_units,
          completed: task.status === "completed" || task.status === "failed",
        } : current);
        if (task.status === "completed") {
          setToastNotice({ message: "文档已解析并可用于问答", tone: "success" });
          await refreshKnowledgeBaseView(true);
          return;
        }
        if (task.status === "failed") {
          setToastNotice({ message: task.error?.message ?? task.error_message ?? "文档处理失败", tone: "error" });
          await refreshKnowledgeBaseView(true);
          return;
        }
        timer = window.setTimeout(pollProcessing, 1200);
      } catch {
        if (!stopped) {
          timer = window.setTimeout(pollProcessing, 3000);
        }
      }
    }

    void pollProcessing();
    return () => {
      stopped = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [refreshKnowledgeBaseView, uploadNotice]);

  useEffect(() => {
    if (!batchUploadNotice || batchUploadNotice.completed) {
      return;
    }
    const pendingItems = batchUploadNotice.items.filter(
      (item) => item.status === "accepted" && item.documentId && !item.completed,
    );
    if (pendingItems.length === 0) {
      setBatchUploadNotice((current) => current ? { ...current, completed: true } : current);
      return;
    }

    let stopped = false;
    let timer: number | undefined;

    async function pollBatchProcessing() {
      const results = await Promise.allSettled(
        pendingItems.map((item) => getDocumentProcessing(item.documentId as string)),
      );
      if (stopped) {
        return;
      }

      const taskByDocumentId = new Map(
        results
          .filter((result) => result.status === "fulfilled")
          .map((result) => [result.value.document_id, result.value]),
      );
      let hasTerminalUpdate = false;
      let hasPending = results.some((result) => result.status === "rejected");
      let hasFailedTask = false;

      setBatchUploadNotice((current) => {
        if (!current) {
          return current;
        }
        const nextItems = current.items.map((item) => {
          if (!item.documentId) {
            return item;
          }
          const task = taskByDocumentId.get(item.documentId);
          if (!task) {
            return item;
          }
          const completed = task.status === "completed" || task.status === "failed";
          if (completed && !item.completed) {
            hasTerminalUpdate = true;
          }
          if (task.status === "failed") {
            hasFailedTask = true;
          }
          if (!completed) {
            hasPending = true;
          }
          return {
            ...item,
            stage: task.stage,
            completedUnits: task.completed_units,
            totalUnits: task.total_units,
            completed,
            errorMessage: task.error?.message ?? task.error_message ?? item.errorMessage,
          };
        });
        const allAcceptedDone = nextItems.every(
          (item) => item.status !== "accepted" || item.completed,
        );
        return { ...current, items: nextItems, completed: allAcceptedDone };
      });

      if (hasTerminalUpdate) {
        await refreshKnowledgeBaseView(true);
      }
      if (hasPending) {
        timer = window.setTimeout(pollBatchProcessing, 1500);
        return;
      }
      setToastNotice({
        message: hasFailedTask ? "批量上传中有文档处理失败" : "批量上传文档已处理完成",
        tone: hasFailedTask ? "error" : "success",
      });
    }

    void pollBatchProcessing();
    return () => {
      stopped = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [batchUploadNotice, refreshKnowledgeBaseView]);

  useEffect(() => {
    if (!toastNotice) {
      return;
    }
    const timer = window.setTimeout(() => setToastNotice(null), 2600);
    return () => window.clearTimeout(timer);
  }, [toastNotice]);

  useEffect(() => {
    if (!deleteTarget && !bulkDeleteOpen) {
      return;
    }
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    deleteCancelButtonRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setDeleteTarget(null);
        setBulkDeleteOpen(false);
        return;
      }
      if (event.key !== "Tab" || !deleteDialogRef.current) {
        return;
      }
      const focusable = Array.from(
        deleteDialogRef.current.querySelectorAll<HTMLElement>("button:not([disabled]), [href], [tabindex]:not([tabindex='-1'])"),
      );
      if (focusable.length === 0) {
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      previousFocus?.focus();
    };
  }, [bulkDeleteOpen, deleteTarget]);

  useEffect(() => {
    const availableIds = new Set(documents.map((document) => document.document_id));
    setSelectedDocumentIds((current) => current.filter((documentId) => availableIds.has(documentId)));
  }, [documents]);

  useEffect(() => {
    if (!selectedDetail) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !document.querySelector(".identity-editor")) setSelectedDetail(null);
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [selectedDetail]);

  useEffect(() => {
    if (!openDocumentMenu) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setOpenDocumentMenu(null);
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [openDocumentMenu]);

  async function submitSingleUpload(
    file: File,
    options: { filenameOverride?: string; overwriteDocumentId?: string; alreadyBusy?: boolean } = {},
  ) {
    if (!options.alreadyBusy) {
      setIsUploading(true);
    }
    setUploadBytes({ loaded: 0, total: file.size });
    try {
      const result = await uploadDocument(file, {
        idempotencyKey: createRequestId(),
        filenameOverride: options.filenameOverride,
        overwriteDocumentId: options.overwriteDocumentId,
        onProgress: (loaded, total) => setUploadBytes({ loaded, total }),
      });
      setUploadNotice({
        documentId: result.document_id,
        taskId: result.task_id,
        stage: result.stage,
        completedUnits: null,
        totalUnits: null,
        completed: false,
      });
      const latestDocuments = await refreshKnowledgeBaseView(true);
      const uploadedDocument = latestDocuments.find((document) => document.document_id === result.document_id);
      if (uploadedDocument?.status === "indexed") {
        setToastNotice({ message: "上传成功", tone: "success" });
        setUploadNotice({
          documentId: result.document_id,
          taskId: result.task_id,
          stage: "completed",
          completedUnits: uploadedDocument.chunk_count,
          totalUnits: uploadedDocument.chunk_count,
          completed: true,
        });
      }
      return result;
    } finally {
      setIsUploading(false);
      setUploadBytes(null);
    }
  }

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setUploadNotice(null);
    setUploadConflict(null);
    setToastNotice(null);
    setIsUploading(true);
    let submitted = false;
    try {
      const fileSha256 = await computeFileSha256(file);
      const preflight = await preflightDocumentUploads([
        { client_file_id: "single", filename: file.name, size: file.size, file_sha256: fileSha256 },
      ]);
      const result = preflight.items[0];
      if (result.status !== "ready") {
        setUploadConflict({
          file,
          fileSha256,
          result,
          renameValue: suggestedFilename(file.name),
        });
        setToastNotice({ message: uploadPreflightStatusLabel(result), tone: "error" });
        return;
      }
      submitted = true;
      await submitSingleUpload(file, { alreadyBusy: true });
    } catch (error) {
      setToastNotice({ message: error instanceof Error ? error.message : "上传失败", tone: "error" });
    } finally {
      if (!submitted) {
        setIsUploading(false);
      }
      event.target.value = "";
    }
  }

  async function handleBatchFileChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    if (files.length === 0) {
      return;
    }

    setUploadNotice(null);
    setBatchUploadNotice(null);
    setBatchConflictIssues([]);
    setToastNotice(null);
    setIsBatchUploading(true);
    try {
      const preparedFiles = await Promise.all(
        files.map(async (file, index) => ({
          id: `batch-${Date.now()}-${index}`,
          file,
          fileSha256: await computeFileSha256(file),
        })),
      );
      const preflight = await preflightDocumentUploads(
        preparedFiles.map((item) => ({
          client_file_id: item.id,
          filename: item.file.name,
          size: item.file.size,
          file_sha256: item.fileSha256,
        })),
      );
      const preflightById = new Map(preflight.items.map((item) => [item.client_file_id, item]));
      const readyFiles = preparedFiles.filter((item) => preflightById.get(item.id)?.status === "ready");
      const issues = preparedFiles
        .filter((item) => preflightById.get(item.id)?.status !== "ready")
        .map((item) => ({
          id: item.id,
          file: item.file,
          fileSha256: item.fileSha256,
          result: preflightById.get(item.id) as DocumentUploadPreflightItem,
          renameValue: suggestedFilename(item.file.name),
          resolving: false,
        }));
      setBatchConflictIssues(issues);

      if (readyFiles.length === 0) {
        setToastNotice({
          message: `批量预检完成，${issues.length} 个文件需要处理`,
          tone: issues.length > 0 ? "error" : "success",
        });
        return;
      }

      setBatchUploadBytes({
        loaded: 0,
        total: readyFiles.reduce((total, item) => total + item.file.size, 0),
        count: readyFiles.length,
      });
      const result = await uploadDocumentsBatch(readyFiles.map((item) => item.file), {
        idempotencyKey: createRequestId(),
        onProgress: (loaded, total) => setBatchUploadBytes({ loaded, total, count: readyFiles.length }),
      });
      const latestDocuments = await refreshKnowledgeBaseView(true);
      setBatchUploadNotice({
        batchId: result.batch_id,
        acceptedCount: result.accepted_count,
        failedCount: result.failed_count,
        completed: result.accepted_count === 0,
        items: result.items.map((item) => {
          const uploadedDocument = item.document_id
            ? latestDocuments.find((document) => document.document_id === item.document_id)
            : null;
          const completed = uploadedDocument?.status === "indexed";
          return {
            filename: item.filename,
            documentId: item.document_id,
            taskId: item.task_id,
            stage: completed ? "completed" : item.stage,
            completedUnits: completed ? uploadedDocument.chunk_count : null,
            totalUnits: completed ? uploadedDocument.chunk_count : null,
            completed,
            status: item.status,
            errorMessage: item.error_message,
          };
        }),
      });
      setToastNotice({
        message: issues.length > 0 || result.failed_count > 0
          ? `批量上传已接收 ${result.accepted_count} 份，${issues.length + result.failed_count} 份需要处理`
          : `批量上传已提交 ${result.accepted_count} 份文档`,
        tone: issues.length > 0 || result.failed_count > 0 ? "error" : "success",
      });
    } catch (error) {
      setToastNotice({ message: error instanceof Error ? error.message : "批量上传失败", tone: "error" });
    } finally {
      setIsBatchUploading(false);
      setBatchUploadBytes(null);
      event.target.value = "";
    }
  }

  async function confirmUploadConflict(action: "rename" | "overwrite") {
    if (!uploadConflict) {
      return;
    }
    setToastNotice(null);
    try {
      if (action === "rename") {
        const renameValue = uploadConflict.renameValue.trim();
        const preflight = await preflightDocumentUploads([
          {
            client_file_id: "single-rename",
            filename: renameValue,
            size: uploadConflict.file.size,
            file_sha256: uploadConflict.fileSha256,
          },
        ]);
        const result = preflight.items[0];
        if (result.status !== "ready") {
          setUploadConflict((current) => current ? { ...current, result, renameValue } : current);
          setToastNotice({ message: uploadPreflightStatusLabel(result), tone: "error" });
          return;
        }
        const file = uploadConflict.file;
        setUploadConflict(null);
        await submitSingleUpload(file, { filenameOverride: renameValue });
        return;
      }
      const existingDocumentId = uploadConflict.result.existing_document?.document_id;
      if (!existingDocumentId) {
        setToastNotice({ message: "没有可覆盖的已有文档。", tone: "error" });
        return;
      }
      const file = uploadConflict.file;
      setUploadConflict(null);
      await submitSingleUpload(file, { overwriteDocumentId: existingDocumentId });
    } catch (error) {
      setToastNotice({ message: error instanceof Error ? error.message : "冲突处理失败", tone: "error" });
    }
  }

  function updateBatchIssueRename(issueId: string, renameValue: string) {
    setBatchConflictIssues((current) => current.map((issue) => (
      issue.id === issueId ? { ...issue, renameValue } : issue
    )));
  }

  function skipBatchIssue(issueId: string) {
    setBatchConflictIssues((current) => current.filter((issue) => issue.id !== issueId));
  }

  async function resolveBatchIssue(issue: BatchConflictIssue, action: "rename" | "overwrite") {
    setIsResolvingBatchIssues(true);
    setBatchConflictIssues((current) => current.map((item) => (
      item.id === issue.id ? { ...item, resolving: true } : item
    )));
    try {
      if (action === "rename") {
        const renameValue = issue.renameValue.trim();
        const preflight = await preflightDocumentUploads([
          {
            client_file_id: `${issue.id}-rename`,
            filename: renameValue,
            size: issue.file.size,
            file_sha256: issue.fileSha256,
          },
        ]);
        const result = preflight.items[0];
        if (result.status !== "ready") {
          setBatchConflictIssues((current) => current.map((item) => (
            item.id === issue.id ? { ...item, result, renameValue, resolving: false } : item
          )));
          setToastNotice({ message: uploadPreflightStatusLabel(result), tone: "error" });
          return;
        }
        await submitResolvedBatchUpload(issue.file, { filenameOverride: renameValue });
        skipBatchIssue(issue.id);
        return;
      }
      const existingDocumentId = issue.result.existing_document?.document_id;
      if (!existingDocumentId) {
        setToastNotice({ message: "没有可覆盖的已有文档。", tone: "error" });
        return;
      }
      await submitResolvedBatchUpload(issue.file, { overwriteDocumentId: existingDocumentId });
      skipBatchIssue(issue.id);
    } catch (error) {
      setToastNotice({ message: error instanceof Error ? error.message : "问题文件处理失败", tone: "error" });
    } finally {
      setBatchConflictIssues((current) => current.map((item) => (
        item.id === issue.id ? { ...item, resolving: false } : item
      )));
      setIsResolvingBatchIssues(false);
    }
  }

  async function submitResolvedBatchUpload(
    file: File,
    options: { filenameOverride?: string; overwriteDocumentId?: string },
  ) {
    const result = await uploadDocument(file, {
      idempotencyKey: createRequestId(),
      filenameOverride: options.filenameOverride,
      overwriteDocumentId: options.overwriteDocumentId,
    });
    setBatchUploadNotice((current) => {
      const notice = current ?? {
        batchId: "resolved-batch-conflicts",
        acceptedCount: 0,
        failedCount: 0,
        completed: false,
        items: [],
      };
      return {
        ...notice,
        acceptedCount: notice.acceptedCount + 1,
        completed: false,
        items: [
          ...notice.items,
          {
            filename: result.filename,
            documentId: result.document_id,
            taskId: result.task_id,
            stage: result.stage,
            completedUnits: null,
            totalUnits: null,
            completed: false,
            status: "accepted",
            errorMessage: null,
          },
        ],
      };
    });
    await refreshKnowledgeBaseView(true);
    setToastNotice({ message: "问题文件已提交处理", tone: "success" });
  }

  async function resolveAllBatchIssuesWithRename() {
    const issues = batchConflictIssues.filter((issue) => issue.result.status !== "exact_duplicate");
    for (const [index, issue] of issues.entries()) {
      await resolveBatchIssue({ ...issue, renameValue: suggestedFilename(issue.file.name, index + 1) }, "rename");
    }
    setBatchConflictIssues((current) => current.filter((issue) => issue.result.status === "exact_duplicate"));
  }

  async function showDetail(documentId: string, chunkOffset = 0) {
    const detail = await getDocument(documentId, chunkOffset, CHUNK_PAGE_SIZE);
    setSelectedDetail(detail);
  }

  function handleIdentityMetadataChange(metadata: DocumentMetadata, notice?: string) {
    setSelectedDetail((current) => current ? { ...current, metadata } : current);
    setDocuments((current) => current.map((document) => (
      document.document_id === selectedDetail?.document_id ? { ...document, metadata } : document
    )));
    if (notice) {
      setToastNotice({ message: notice, tone: "success" });
    }
  }

  function toggleDocumentSelection(document: DocumentSummary, checked: boolean) {
    if (isDocumentDeletionBlocked(document)) {
      return;
    }
    setSelectedDocumentIds((current) => {
      if (checked) {
        return current.includes(document.document_id) ? current : [...current, document.document_id];
      }
      return current.filter((documentId) => documentId !== document.document_id);
    });
  }

  function cancelSelectionMode() {
    setIsSelectionMode(false);
    setSelectedDocumentIds([]);
    setBulkDeleteOpen(false);
  }

  function selectAllFilteredDocuments() {
    const filteredIds = filteredDeletableDocuments.map((document) => document.document_id);
    setSelectedDocumentIds((current) => Array.from(new Set([...current, ...filteredIds])));
  }

  function toggleVisibleSelection(checked: boolean) {
    const visibleIds = visibleDeletableDocuments.map((document) => document.document_id);
    setSelectedDocumentIds((current) => {
      if (checked) {
        return Array.from(new Set([...current, ...visibleIds]));
      }
      const visibleIdSet = new Set(visibleIds);
      return current.filter((documentId) => !visibleIdSet.has(documentId));
    });
  }

  async function confirmDelete() {
    if (!deleteTarget) {
      return;
    }

    const documentId = deleteTarget.document_id;
    const controller = new AbortController();
    deleteAbortController.current = controller;
    setDeleteTarget(null);
    setDeletingDocumentId(documentId);
    setUploadNotice(null);
    setToastNotice(null);
    try {
      const result = await deleteDocument(documentId, controller.signal);
      setToastNotice({
        message: result.vector_warning ? `删除成功，向量清理有警告：${result.vector_warning}` : "删除成功",
        tone: "success",
      });
      setSelectedDetail(null);
    } catch (error) {
      setToastNotice({
        message: isAbortError(error) ? "已取消等待删除结果，正在刷新文档列表。" : error instanceof Error ? error.message : "删除失败",
        tone: "error",
      });
    } finally {
      if (deleteAbortController.current === controller) {
        deleteAbortController.current = null;
      }
      setDeletingDocumentId(null);
      await refreshKnowledgeBaseView();
    }
  }

  async function confirmBulkDelete() {
    const documentIds = selectedDocumentIds.slice();
    if (documentIds.length === 0) {
      setBulkDeleteOpen(false);
      return;
    }

    setBulkDeleteOpen(false);
    setIsBulkDeleting(true);
    setUploadNotice(null);
    setToastNotice(null);
    try {
      const result = await deleteDocumentsBulk(documentIds);
      const deletedIds = new Set(
        result.items.filter((item) => item.status === "deleted").map((item) => item.document_id),
      );
      setSelectedDocumentIds([]);
      setIsSelectionMode(false);
      if (selectedDetail && deletedIds.has(selectedDetail.document_id)) {
        setSelectedDetail(null);
      }
      setToastNotice({
        message: result.failed_count > 0
          ? `已删除 ${result.deleted_count} 份，${result.failed_count} 份未删除`
          : `已删除 ${result.deleted_count} 份文档`,
        tone: result.failed_count > 0 ? "error" : "success",
      });
    } catch (error) {
      setToastNotice({ message: error instanceof Error ? error.message : "批量删除失败", tone: "error" });
    } finally {
      setIsBulkDeleting(false);
      await refreshKnowledgeBaseView();
    }
  }

  function cancelDeleteRequest() {
    deleteAbortController.current?.abort();
    deleteAbortController.current = null;
    setDeletingDocumentId(null);
  }

  async function rebuildIndex(documentId: string) {
    setRebuildingDocumentId(documentId);
    setToastNotice(null);
    try {
      await rebuildDocumentIndex(documentId);
      setToastNotice({ message: "已加入索引队列", tone: "success" });
      await refreshKnowledgeBaseView();
    } catch (error) {
      setToastNotice({ message: error instanceof Error ? error.message : "重建索引失败", tone: "error" });
    } finally {
      setRebuildingDocumentId(null);
    }
  }

  const normalizedSearch = searchTerm.trim().toLocaleLowerCase("zh-CN");
  const filteredDocuments = documents.filter((document) => {
    const matchesStatus = statusFilter === "all" || document.status === statusFilter;
    const matchesType = typeFilter === "all" || document.file_type.toLowerCase() === typeFilter;
    const matchesSearch = !normalizedSearch
      || document.filename.toLocaleLowerCase("zh-CN").includes(normalizedSearch)
      || document.document_id.toLocaleLowerCase("zh-CN").includes(normalizedSearch);
    return matchesStatus && matchesType && matchesSearch;
  });
  const pageCount = Math.max(1, Math.ceil(filteredDocuments.length / DOCUMENT_PAGE_SIZE));
  const activePage = Math.min(pageNumber, pageCount);
  const visibleDocuments = filteredDocuments.slice(
    (activePage - 1) * DOCUMENT_PAGE_SIZE,
    activePage * DOCUMENT_PAGE_SIZE,
  );
  const selectedDocumentIdSet = new Set(selectedDocumentIds);
  const filteredDeletableDocuments = filteredDocuments.filter((document) => !isDocumentDeletionBlocked(document));
  const visibleDeletableDocuments = visibleDocuments.filter((document) => !isDocumentDeletionBlocked(document));
  const visibleSelectedCount = visibleDeletableDocuments.filter((document) => selectedDocumentIdSet.has(document.document_id)).length;
  const allVisibleSelected = visibleDeletableDocuments.length > 0 && visibleSelectedCount === visibleDeletableDocuments.length;
  const filteredSelectedCount = filteredDeletableDocuments.filter((document) => selectedDocumentIdSet.has(document.document_id)).length;
  const allFilteredSelected = filteredDeletableDocuments.length > 0 && filteredSelectedCount === filteredDeletableDocuments.length;
  const selectedDocuments = selectedDocumentIds
    .map((documentId) => documents.find((document) => document.document_id === documentId))
    .filter((document): document is DocumentSummary => Boolean(document));
  const indexedCount = documents.filter((document) => document.status === "indexed").length;
  const processingCount = documents.filter((document) => ["uploaded", "index_queued", "indexing"].includes(document.status)).length;
  const failedCount = documents.filter((document) => ["index_failed", "source_missing", "delete_failed"].includes(document.status)).length;
  const availableFileTypes = Array.from(new Set(documents.map((document) => document.file_type.toLowerCase()).filter(Boolean))).sort();

  return (
    <main className="page">
      {toastNotice ? <Toast notice={toastNotice} /> : null}
      <header className="product-header">
        <div>
          <p className="eyebrow">知识库台账</p>
          <div className="product-title-lockup">
            <h1>个人材料与项目文档</h1>
            <StatusBadge tone={ragHealth?.ready && indexedCount > 0 ? "ok" : "warning"}>
              {ragHealth?.ready && indexedCount > 0 ? "可支撑问答" : "待完善"}
            </StatusBadge>
          </div>
          <p className="page-lead">上传简历、证书、荣誉和项目介绍文档；系统解析、分块并建立可追溯索引。</p>
        </div>
        <button className="icon-button" type="button" disabled={isRefreshing} onClick={() => void refreshKnowledgeBaseView()}>
          <RefreshCw size={17} className={isRefreshing ? "spinning" : undefined} />
          刷新
        </button>
      </header>

      {runtimeWarning ? (
        <div className="knowledge-warning" role="status">
          <AlertTriangle size={17} />
          <span>{runtimeWarning}</span>
        </div>
      ) : null}

      <section className="document-command-panel">
        <div className="panel-title">
          <FileUp size={20} />
          <h2>新增知识源</h2>
        </div>
        <div className="document-command-panel__body">
          <p>支持 PDF、Word、Excel、Markdown 等可提取文本的文件。</p>
          <div className="document-upload-actions">
            <label className={`file-input ${isUploading ? "is-disabled" : ""}`}>
              {isUploading ? "正在上传" : "选择文档"}
              <input
                type="file"
                accept=".txt,.md,.doc,.docx,.pdf,.xls,.xlsx,.csv,.jsonl,.html,.htm"
                onChange={(event) => void handleFileChange(event)}
                disabled={isUploading}
              />
            </label>
            <label className={`file-input ${isBatchUploading ? "is-disabled" : ""}`}>
              {isBatchUploading ? "批量上传中" : "批量上传"}
              <input
                type="file"
                accept=".txt,.md,.doc,.docx,.pdf,.xls,.xlsx,.csv,.jsonl,.html,.htm"
                multiple
                onChange={(event) => void handleBatchFileChange(event)}
                disabled={isBatchUploading}
              />
            </label>
          </div>
        </div>
        {uploadBytes ? (
          <div className="upload-progress" role="status" aria-live="polite">
            <span>正在上传文件</span>
            <progress value={uploadBytes.loaded} max={Math.max(uploadBytes.total, 1)} />
            <strong>{Math.round((uploadBytes.loaded / Math.max(uploadBytes.total, 1)) * 100)}%</strong>
          </div>
        ) : uploadNotice && !uploadNotice.completed ? (
          <div className="upload-progress" role="status" aria-live="polite">
            <span>{documentStageLabel(uploadNotice.stage)}</span>
            <progress
              value={uploadNotice.completedUnits ?? undefined}
              max={uploadNotice.totalUnits ?? undefined}
            />
            <strong>{formatProcessingUnits(uploadNotice.completedUnits, uploadNotice.totalUnits)}</strong>
          </div>
        ) : null}
        {batchUploadBytes ? (
          <div className="upload-progress" role="status" aria-live="polite">
            <span>正在上传 {batchUploadBytes.count} 个文件</span>
            <progress value={batchUploadBytes.loaded} max={Math.max(batchUploadBytes.total, 1)} />
            <strong>{Math.round((batchUploadBytes.loaded / Math.max(batchUploadBytes.total, 1)) * 100)}%</strong>
          </div>
        ) : null}
        {batchUploadNotice ? <BatchUploadSummary notice={batchUploadNotice} /> : null}
        {batchConflictIssues.length > 0 ? (
          <BatchConflictPanel
            issues={batchConflictIssues}
            disabled={isResolvingBatchIssues}
            onRenameChange={updateBatchIssueRename}
            onResolve={(issue, action) => void resolveBatchIssue(issue, action)}
            onSkip={skipBatchIssue}
            onRenameAll={() => void resolveAllBatchIssuesWithRename()}
            onSkipAll={() => setBatchConflictIssues([])}
          />
        ) : null}
        <p className="hint">可上传 doc、.docx、.pdf、.xls 、.xlsx 、.jsonl、.csv、.md、.txt文件</p>
      </section>

      <section className="panel">
        {loadError ? (
          <div className="page-load-error" role="alert">
            <AlertCircle size={18} />
            <span>{loadError}</span>
            <button className="secondary-button" type="button" onClick={() => void refreshKnowledgeBaseView()}>重试</button>
          </div>
        ) : null}
        <div className="document-list-toolbar">
          <div>
            <h2>文档台账</h2>
            <p className="toolbar-summary">
              共 {documents.length} 份，{indexedCount} 份可问答，{processingCount} 份处理中，{failedCount} 份需处理
            </p>
          </div>
          <div className="document-filters">
            {isSelectionMode ? (
              <div className="bulk-selection-actions" role="status" aria-live="polite">
                <span>已选 {selectedDocumentIds.length} 份</span>
                <button className="secondary-button" type="button" disabled={isBulkDeleting || allFilteredSelected || filteredDeletableDocuments.length === 0} onClick={selectAllFilteredDocuments}>
                  全选
                </button>
                <button className="secondary-button danger-button" type="button" disabled={isBulkDeleting || selectedDocumentIds.length === 0} onClick={() => setBulkDeleteOpen(true)}>
                  <Trash2 size={15} />
                  批量删除
                </button>
                <button className="secondary-button" type="button" disabled={isBulkDeleting} onClick={cancelSelectionMode}>
                  取消选择
                </button>
              </div>
            ) : (
              <button className="secondary-button" type="button" disabled={isBulkDeleting || filteredDeletableDocuments.length === 0} onClick={() => setIsSelectionMode(true)}>
                选择
              </button>
            )}
            <label className="document-search">
              <span className="sr-only">搜索文件名或文档编号</span>
              <Search size={16} />
              <input
                type="search"
                value={searchTerm}
                placeholder="搜索文件名或编号"
                onChange={(event) => {
                  setSearchTerm(event.target.value);
                  setPageNumber(1);
                }}
              />
            </label>
            <label>
              <span className="sr-only">类型筛选</span>
              <select
                aria-label="类型筛选"
                value={typeFilter}
                onChange={(event) => {
                  setTypeFilter(event.target.value);
                  setPageNumber(1);
                }}
              >
                <option value="all">全部类型</option>
                {availableFileTypes.map((type) => <option key={type} value={type}>{type.toUpperCase()}</option>)}
              </select>
            </label>
            <label>
              <span className="sr-only">状态筛选</span>
              <select
                aria-label="状态筛选"
                value={statusFilter}
                onChange={(event) => {
                  setStatusFilter(event.target.value);
                  setPageNumber(1);
                }}
              >
                <option value="all">全部状态（{documents.length}）</option>
                <option value="indexed">可问答</option>
                <option value="index_queued">排队中</option>
                <option value="indexing">处理中</option>
                <option value="index_failed">处理失败</option>
                <option value="source_missing">原文件缺失</option>
              </select>
            </label>
          </div>
        </div>
        {isLoading ? (
          <div className="empty-state"><RefreshCw size={24} className="spinning" /><p>正在读取知识库台账…</p></div>
        ) : filteredDocuments.length > 0 ? (
          <div className="table-wrap">
            <table className="documents-table">
              <thead>
                <tr>
                  {isSelectionMode ? (
                    <th className="selection-column">
                      <label className="selection-checkbox">
                        <input
                          type="checkbox"
                          aria-label="选择本页可删除文档"
                          checked={allVisibleSelected}
                          disabled={visibleDeletableDocuments.length === 0 || isBulkDeleting}
                          onChange={(event) => toggleVisibleSelection(event.target.checked)}
                        />
                        <span className="sr-only">选择本页</span>
                      </label>
                    </th>
                  ) : null}
                  <th>文件名称</th>
                  <th>文档分类</th>
                  <th>解析与索引状态</th>
                  <th><span className="document-count-heading">分块<br />数量</span></th>
                  <th>上传时间</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {visibleDocuments.map((document) => (
                  <tr key={document.document_id} className={isSelectionMode && selectedDocumentIdSet.has(document.document_id) ? "is-selected" : undefined}>
                    {isSelectionMode ? (
                      <td data-label="选择" className="selection-column">
                        <label className="selection-checkbox">
                          <input
                            type="checkbox"
                            aria-label={`选择 ${document.filename}`}
                            checked={selectedDocumentIdSet.has(document.document_id)}
                            disabled={isDocumentDeletionBlocked(document) || isBulkDeleting}
                            onChange={(event) => toggleDocumentSelection(document, event.target.checked)}
                          />
                        </label>
                      </td>
                    ) : null}
                    <td data-label="文件名称">
                      <button className="document-title-button" type="button" onClick={() => void showDetail(document.document_id)}>
                        {document.filename}
                      </button>
                      <span className="document-subtle">编号：{document.document_id}</span>
                      <span className={document.metadata.identity_review_status === "confirmed" ? "document-identity-state is-confirmed" : "document-identity-state"}>
                        {document.metadata.identity_review_status === "confirmed" ? "身份已核对" : "身份待核对"}
                      </span>
                    </td>
                    <td data-label="文档分类" className="document-meta-cell">{documentTypeLabel(document)}</td>
                    <td data-label="处理状态" className="document-status-cell" title={document.index_error ?? undefined}>
                      <StatusBadge tone={statusTone(document.status)}>{statusLabel(document.status)}</StatusBadge>
                      {document.index_error ? (
                        <details className="source-details">
                          <summary>{friendlyIndexError(document.index_error)}</summary>
                        </details>
                      ) : null}
                    </td>
                    <td data-label="分块数量" className="document-meta-cell document-count-cell">{document.chunk_count}</td>
                    <td data-label="上传时间" className="document-meta-cell">{formatDateTime(document.uploaded_at)}</td>
                    <td data-label="操作">
                      <div className="row-actions">
                        <button
                          className="icon-button--plain row-actions__trigger"
                          type="button"
                          aria-label={`${document.filename} 更多操作`}
                          aria-expanded={openDocumentMenu === document.document_id}
                          onClick={() => setOpenDocumentMenu((current) => current === document.document_id ? null : document.document_id)}
                        >
                          <MoreVertical size={17} />
                        </button>
                        {openDocumentMenu === document.document_id ? (
                          <div className="row-actions__menu" role="menu">
                            <button type="button" role="menuitem" onClick={() => { setOpenDocumentMenu(null); void showDetail(document.document_id); }}>
                              {document.chunk_count > CHUNK_PAGE_SIZE ? "分页查看内容" : "查看内容"}
                            </button>
                            {document.status === "index_failed" || document.status === "uploaded" ? (
                              <button
                                type="button"
                                role="menuitem"
                                disabled={rebuildingDocumentId === document.document_id}
                                onClick={() => { setOpenDocumentMenu(null); void rebuildIndex(document.document_id); }}
                              >
                                {rebuildingDocumentId === document.document_id ? "提交中" : "重建索引"}
                              </button>
                            ) : null}
                            {deletingDocumentId === document.document_id ? (
                              <button type="button" role="menuitem" onClick={() => { setOpenDocumentMenu(null); cancelDeleteRequest(); }}>
                                停止等待
                              </button>
                            ) : (
                              <button
                                className="row-actions__danger"
                                type="button"
                                role="menuitem"
                                disabled={isDocumentDeletionBlocked(document)}
                                onClick={() => { setOpenDocumentMenu(null); setDeleteTarget(document); }}
                              >
                                {document.status === "delete_failed" ? "重试删除" : "删除"}
                              </button>
                            )}
                          </div>
                        ) : null}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="pagination">
              <span>第 {activePage}/{pageCount} 页，共 {filteredDocuments.length} 个文档</span>
              <div>
                <button className="secondary-button" type="button" disabled={activePage <= 1} onClick={() => setPageNumber(activePage - 1)}>上一页</button>
                <button className="secondary-button" type="button" disabled={activePage >= pageCount} onClick={() => setPageNumber(activePage + 1)}>下一页</button>
              </div>
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <FileUp size={24} />
            <h2>当前筛选条件下暂无文档</h2>
            <p>上传文档或调整筛选条件后，系统会展示解析、分块和索引状态。</p>
          </div>
        )}
      </section>

      {selectedDetail ? (
        <div className="drawer-backdrop" role="presentation" onMouseDown={() => setSelectedDetail(null)}>
          <aside
            className="drawer document-detail-drawer"
            role="dialog"
            aria-modal="true"
            aria-labelledby="document-detail-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <header className="drawer__head">
              <div>
                <p className="eyebrow">知识源原文</p>
                <h2 id="document-detail-title">{selectedDetail.filename}</h2>
                <p className="muted">
                  第 {selectedDetail.chunk_offset + 1}–
                  {Math.min(selectedDetail.chunk_offset + selectedDetail.chunks.length, selectedDetail.chunk_total)} 条，
                  共 {selectedDetail.chunk_total} 条
                </p>
              </div>
              <button className="icon-button--plain" type="button" aria-label="关闭文档内容" onClick={() => setSelectedDetail(null)}>
                <X size={20} />
              </button>
            </header>
            <div className="drawer-body">
              <DocumentIdentityCard document={selectedDetail} onMetadataChange={handleIdentityMetadataChange} />
              <div className="document-content-divider">
                <span>解析内容与证据片段</span>
                <span>{selectedDetail.chunk_total} 条</span>
              </div>
              {selectedDetail.chunks.length > 0 ? (
                <ol className="evidence-list document-chunk-list">
                  {selectedDetail.chunks.map((chunk, index) => (
                    <li key={chunk.chunk_id} className="evidence-item">
                      <div className="evidence-head">
                        <span>{chunk.section_title ?? "未命名章节"}</span>
                        <span className="muted">第 {selectedDetail.chunk_offset + index + 1} 条</span>
                      </div>
                      <pre className={chunk.chunk_type === "table" ? "chunk-text chunk-text--table" : "chunk-text"}>
                        {displayChunkText(chunk)}
                      </pre>
                    </li>
                  ))}
                </ol>
              ) : <p className="muted">该文档暂无可预览片段。</p>}
            </div>
            <footer className="drawer-footer pagination">
              <span>
                第 {Math.floor(selectedDetail.chunk_offset / CHUNK_PAGE_SIZE) + 1}/
                {Math.max(1, Math.ceil(selectedDetail.chunk_total / CHUNK_PAGE_SIZE))} 页
              </span>
              <div>
                <button className="secondary-button" type="button" disabled={selectedDetail.chunk_offset <= 0} onClick={() => void showDetail(selectedDetail.document_id, Math.max(0, selectedDetail.chunk_offset - CHUNK_PAGE_SIZE))}>上一页</button>
                <button className="secondary-button" type="button" disabled={selectedDetail.chunk_offset + selectedDetail.chunk_limit >= selectedDetail.chunk_total} onClick={() => void showDetail(selectedDetail.document_id, selectedDetail.chunk_offset + CHUNK_PAGE_SIZE)}>下一页</button>
              </div>
            </footer>
          </aside>
        </div>
      ) : null}

      {uploadConflict ? (
        <UploadConflictDialog
          conflict={uploadConflict}
          isUploading={isUploading}
          onRenameChange={(renameValue) => setUploadConflict((current) => current ? { ...current, renameValue } : current)}
          onCancel={() => setUploadConflict(null)}
          onResolve={(action) => void confirmUploadConflict(action)}
        />
      ) : null}

      {deleteTarget ? (
        <div className="modal-backdrop" role="presentation">
          <div ref={deleteDialogRef} className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-dialog-title">
            <div className="confirm-dialog__icon">
              <AlertTriangle size={22} />
            </div>
            <div className="confirm-dialog__body">
              <h2 id="delete-dialog-title">确认删除文档</h2>
              <p>
                将删除原始文件、内容片段和检索索引。此操作完成后无法从界面恢复。
              </p>
              <dl className="confirm-dialog__facts">
                <div>
                  <dt>文档编号</dt>
                  <dd className="mono">{deleteTarget.document_id}</dd>
                </div>
                <div>
                  <dt>文件名</dt>
                  <dd>{deleteTarget.filename}</dd>
                </div>
              </dl>
              <div className="button-row">
                <button ref={deleteCancelButtonRef} className="secondary-button" type="button" onClick={() => setDeleteTarget(null)}>
                  取消
                </button>
                <button className="icon-button danger-solid-button" type="button" onClick={() => void confirmDelete()}>
                  <Trash2 size={16} />
                  确认删除
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {bulkDeleteOpen ? (
        <div className="modal-backdrop" role="presentation">
          <div ref={deleteDialogRef} className="confirm-dialog bulk-delete-dialog" role="dialog" aria-modal="true" aria-labelledby="bulk-delete-dialog-title">
            <div className="confirm-dialog__icon">
              <AlertTriangle size={22} />
            </div>
            <div className="confirm-dialog__body">
              <h2 id="bulk-delete-dialog-title">确认批量删除文档</h2>
              <p>
                将删除已选文档的原始文件、内容片段和检索索引。此操作完成后无法从界面恢复。
              </p>
              <dl className="confirm-dialog__facts">
                <div>
                  <dt>已选数量</dt>
                  <dd>{selectedDocuments.length} 份文档</dd>
                </div>
                <div>
                  <dt>文档清单</dt>
                  <dd>
                    <ul className="bulk-delete-list">
                      {selectedDocuments.slice(0, 8).map((document) => (
                        <li key={document.document_id}>
                          <span>{document.filename}</span>
                          <small className="mono">{document.document_id}</small>
                        </li>
                      ))}
                    </ul>
                    {selectedDocuments.length > 8 ? <span className="muted">另有 {selectedDocuments.length - 8} 份未展开显示</span> : null}
                  </dd>
                </div>
              </dl>
              <div className="button-row">
                <button ref={deleteCancelButtonRef} className="secondary-button" type="button" onClick={() => setBulkDeleteOpen(false)}>
                  取消
                </button>
                <button className="icon-button danger-solid-button" type="button" onClick={() => void confirmBulkDelete()}>
                  <Trash2 size={16} />
                  确认批量删除
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}

function Toast({ notice }: { notice: ToastNotice }) {
  const Icon = notice.tone === "success" ? CheckCircle2 : AlertCircle;
  return (
    <div className={`toast-notice toast-notice--${notice.tone}`} role="status" aria-live="polite">
      <Icon size={18} />
      <span>{notice.message}</span>
    </div>
  );
}

function BatchUploadSummary({ notice }: { notice: BatchUploadNotice }) {
  const completedCount = notice.items.filter((item) => item.status === "accepted" && item.completed).length;
  return (
    <div className="batch-upload-summary" role="status" aria-live="polite">
      <div className="batch-upload-summary__head">
        <strong>批量上传结果</strong>
        <span>
          已接收 {notice.acceptedCount} 份
          {notice.failedCount > 0 ? `，${notice.failedCount} 份失败` : ""}
          {notice.acceptedCount > 0 ? `，${completedCount}/${notice.acceptedCount} 份处理完成` : ""}
        </span>
      </div>
      <ul className="batch-upload-list">
        {notice.items.map((item, index) => (
          <li key={`${notice.batchId}-${index}-${item.filename}`}>
            <span className="batch-upload-list__name" title={item.filename}>{item.filename}</span>
            <span className={item.status !== "accepted" || (item.completed && item.errorMessage) ? "batch-upload-list__status is-error" : "batch-upload-list__status"}>
              {batchUploadItemLabel(item)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function UploadConflictDialog({
  conflict,
  isUploading,
  onRenameChange,
  onCancel,
  onResolve,
}: {
  conflict: UploadConflictNotice;
  isUploading: boolean;
  onRenameChange: (value: string) => void;
  onCancel: () => void;
  onResolve: (action: "rename" | "overwrite") => void;
}) {
  const existing = conflict.result.existing_document;
  const isExactDuplicate = conflict.result.status === "exact_duplicate";
  return (
    <div className="modal-backdrop" role="presentation">
      <div className="confirm-dialog upload-conflict-dialog" role="dialog" aria-modal="true" aria-labelledby="upload-conflict-title">
        <div className="confirm-dialog__icon">
          <AlertTriangle size={22} />
        </div>
        <div className="confirm-dialog__body">
          <h2 id="upload-conflict-title">{isExactDuplicate ? "文件已存在" : "文件名已存在"}</h2>
          <p>{uploadPreflightStatusLabel(conflict.result)}</p>
          <dl className="confirm-dialog__facts">
            <div>
              <dt>新文件</dt>
              <dd>{conflict.file.name}（{formatBytes(conflict.file.size)}）</dd>
            </div>
            {existing ? (
              <>
                <div>
                  <dt>已有文档</dt>
                  <dd>{existing.filename}</dd>
                </div>
                <div>
                  <dt>文档编号</dt>
                  <dd className="mono">{existing.document_id}</dd>
                </div>
              </>
            ) : null}
          </dl>
          {!isExactDuplicate ? (
            <label className="rename-field">
              <span>重命名新文件</span>
              <input
                type="text"
                value={conflict.renameValue}
                onChange={(event) => onRenameChange(event.target.value)}
              />
            </label>
          ) : null}
          <div className="button-row">
            <button className="secondary-button" type="button" onClick={onCancel} disabled={isUploading}>
              {isExactDuplicate ? "关闭" : "取消"}
            </button>
            {!isExactDuplicate ? (
              <>
                <button className="secondary-button" type="button" onClick={() => onResolve("rename")} disabled={isUploading || !conflict.renameValue.trim()}>
                  重命名上传
                </button>
                <button className="icon-button danger-solid-button" type="button" onClick={() => onResolve("overwrite")} disabled={isUploading || !existing}>
                  <Trash2 size={16} />
                  覆盖旧文件
                </button>
              </>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

function BatchConflictPanel({
  issues,
  disabled,
  onRenameChange,
  onResolve,
  onSkip,
  onRenameAll,
  onSkipAll,
}: {
  issues: BatchConflictIssue[];
  disabled: boolean;
  onRenameChange: (issueId: string, value: string) => void;
  onResolve: (issue: BatchConflictIssue, action: "rename" | "overwrite") => void;
  onSkip: (issueId: string) => void;
  onRenameAll: () => void;
  onSkipAll: () => void;
}) {
  const resolvableCount = issues.filter((issue) => issue.result.status !== "exact_duplicate").length;
  return (
    <div className="batch-conflict-panel" role="status" aria-live="polite">
      <div className="batch-upload-summary__head">
        <strong>批量上传待处理</strong>
        <span>{issues.length} 个文件需要确认</span>
      </div>
      <div className="button-row">
        <button className="secondary-button" type="button" disabled={disabled || resolvableCount === 0} onClick={onRenameAll}>
          全部自动重命名
        </button>
        <button className="secondary-button" type="button" disabled={disabled} onClick={onSkipAll}>
          全部跳过
        </button>
      </div>
      <ul className="batch-conflict-list">
        {issues.map((issue) => {
          const existing = issue.result.existing_document;
          const isExactDuplicate = issue.result.status === "exact_duplicate";
          return (
            <li key={issue.id}>
              <div className="batch-conflict-list__main">
                <strong title={issue.file.name}>{issue.file.name}</strong>
                <span>{uploadPreflightStatusLabel(issue.result)}</span>
                {existing ? <span className="muted">已有：{existing.filename}</span> : null}
              </div>
              {!isExactDuplicate ? (
                <label className="rename-field batch-conflict-list__rename">
                  <span>新文件名</span>
                  <input
                    type="text"
                    value={issue.renameValue}
                    disabled={disabled || issue.resolving}
                    onChange={(event) => onRenameChange(issue.id, event.target.value)}
                  />
                </label>
              ) : null}
              <div className="button-row">
                <button className="secondary-button" type="button" disabled={disabled || issue.resolving} onClick={() => onSkip(issue.id)}>
                  跳过
                </button>
                {!isExactDuplicate ? (
                  <button className="secondary-button" type="button" disabled={disabled || issue.resolving || !issue.renameValue.trim()} onClick={() => onResolve(issue, "rename")}>
                    重命名上传
                  </button>
                ) : null}
                {issue.result.status === "name_conflict" ? (
                  <button className="secondary-button danger-button" type="button" disabled={disabled || issue.resolving || !existing} onClick={() => onResolve(issue, "overwrite")}>
                    覆盖旧文件
                  </button>
                ) : null}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function formatDateTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false });
}

function statusTone(status: string): "neutral" | "ok" | "warning" | "error" {
  if (status === "indexed") {
    return "ok";
  }
  if (status === "uploaded") {
    return "neutral";
  }
  if (status === "index_queued") {
    return "warning";
  }
  if (status === "index_failed" || status === "source_missing") {
    return "error";
  }
  if (status === "delete_failed") {
    return "error";
  }
  if (status === "indexing" || status === "deleting") {
    return "warning";
  }
  return "neutral";
}

function statusLabel(status: string): string {
  if (status === "indexed") {
    return "可问答";
  }
  if (status === "uploaded") {
    return "待处理";
  }
  if (status === "indexing") {
    return "处理中";
  }
  if (status === "index_queued") {
    return "排队中";
  }
  if (status === "source_missing") {
    return "原文件缺失";
  }
  if (status === "index_failed") {
    return "处理失败";
  }
  if (status === "deleting") {
    return "删除中";
  }
  if (status === "delete_failed") {
    return "删除失败";
  }
  return status;
}

function documentTypeLabel(document: DocumentSummary): string {
  const metadataCategory = document.metadata?.category;
  if (typeof metadataCategory === "string" && metadataCategory.trim()) {
    return metadataCategory;
  }
  const type = document.file_type.toLowerCase();
  if (type === "pdf") {
    return "PDF 文档";
  }
  if (type === "doc" || type === "docx") {
    return "Word 文档";
  }
  if (type === "md" || type === "txt") {
    return "文本知识源";
  }
  return document.file_type || "未分类文档";
}

function friendlyIndexError(error: string): string {
  const normalized = error.toLowerCase();
  if (normalized.includes("document_marked_source_missing") || normalized.includes("source_missing")) {
    return "原始文件缺失，当前文档无法重新解析。";
  }
  if (normalized.includes("parse")) {
    return "解析失败，请检查文件是否可提取文本。";
  }
  if (normalized.includes("qdrant") || normalized.includes("vector")) {
    return "索引写入失败，请检查向量检索服务。";
  }
  if (normalized.includes("timeout")) {
    return "处理超时，可稍后重试。";
  }
  return "处理失败，可展开查看技术详情。";
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function isDocumentDeletionBlocked(document: DocumentSummary): boolean {
  return ["uploaded", "index_queued", "indexing", "deleting"].includes(document.status);
}

function buildRuntimeWarning(ragHealth: RagHealthResponse): string | null {
  if (!ragHealth.ready) {
    return "RAG 依赖未完全就绪。即使文档列表显示很多文件，问答仍可能因模型、SQLite、Qdrant 或解析器状态异常而拒答。";
  }
  return null;
}

function createRequestId(): string {
  return typeof crypto.randomUUID === "function"
    ? crypto.randomUUID().replaceAll("-", "")
    : `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 14)}`;
}

async function computeFileSha256(file: File): Promise<string> {
  if (!crypto.subtle) {
    throw new Error("当前浏览器不支持上传前重复检测，请更换现代浏览器后重试。");
  }
  const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return Array.from(new Uint8Array(digest))
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function suggestedFilename(filename: string, salt = 1): string {
  const dotIndex = filename.lastIndexOf(".");
  if (dotIndex <= 0) {
    return `${filename} (${salt})`;
  }
  return `${filename.slice(0, dotIndex)} (${salt})${filename.slice(dotIndex)}`;
}

function uploadPreflightStatusLabel(item: DocumentUploadPreflightItem): string {
  if (item.status === "exact_duplicate") {
    return "该文件内容已存在于知识库中，已跳过上传。";
  }
  if (item.status === "name_conflict") {
    return "知识库中已存在同名文件，请覆盖旧文件或重命名新文件。";
  }
  if (item.status === "selection_name_conflict") {
    return "本次选择中存在同名文件，请重命名或跳过其中一个。";
  }
  return item.error_message ?? "文件可以上传。";
}

function formatBytes(value: number): string {
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function documentStageLabel(stage: string): string {
  const labels: Record<string, string> = {
    queued: "文件已保存，等待解析",
    parsing: "正在解析文档内容",
    chunking: "正在按文档结构切分",
    metadata_indexing: "正在建立材料元数据索引",
    embedding: "正在生成检索向量",
    vector_upsert: "正在写入向量索引",
    verifying: "正在核对索引完整性",
  };
  return labels[stage] ?? "正在处理文档";
}

function formatProcessingUnits(completed: number | null, total: number | null): string {
  return completed !== null && total !== null ? `${completed}/${total}` : "处理中";
}

function batchUploadItemLabel(item: BatchUploadFileNotice): string {
  if (item.status === "failed") {
    return item.errorMessage ?? "上传失败";
  }
  if (item.status === "duplicate") {
    return item.errorMessage ?? "文件已存在";
  }
  if (item.status === "conflict") {
    return item.errorMessage ?? "文件名冲突";
  }
  if (item.completed) {
    return item.errorMessage ? `处理失败：${item.errorMessage}` : "处理完成";
  }
  return item.stage ? documentStageLabel(item.stage) : "等待处理";
}

function displayChunkText(chunk: ChunkSummary): string {
  const sectionTitle = chunk.section_title?.trim();
  const text = chunk.text.trim();
  if (!sectionTitle || !text.startsWith(sectionTitle)) {
    return chunk.text;
  }

  const withoutTitle = text.slice(sectionTitle.length).replace(/^\s+/, "");
  return withoutTitle || chunk.text;
}

