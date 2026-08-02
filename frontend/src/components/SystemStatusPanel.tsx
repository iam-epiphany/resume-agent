import { Cpu, Database, FileCheck2, Layers3, RefreshCw, ServerCog, ShieldCheck } from "lucide-react";
import type { ReactNode } from "react";

import { isKnowledgeBaseAvailable, isKnowledgeBaseReady, useSystemStatus } from "../state/systemStatusContext";
import type { RagHealthResponse } from "../types/api";

export function SystemStatusPanel() {
  const system = useSystemStatus();
  const health = system.ragHealth;
  const ready = isKnowledgeBaseReady(system);
  const knowledgeBaseAvailable = isKnowledgeBaseAvailable(system);
  const queue = queueSummary(health);
  const deviceFallback = isDeviceFallback(health);
  const modelLines = modelStatusLines(health);
  const readiness = readinessSummary(health, system.indexedDocumentCount);

  const rows: Array<{ icon: ReactNode; label: string; value: ReactNode; title: string; tone: "ok" | "warning" | "unknown" }> = [
    {
      icon: <ShieldCheck size={15} />,
      label: "问答就绪",
      value: readiness.label,
      title: readiness.detail,
      tone: readiness.tone,
    },
    {
      icon: <Cpu size={15} />,
      label: "计算设备",
      value: computeLabel(health),
      title: computeLabel(health),
      tone: deviceFallback ? "warning" : health ? "ok" : "unknown",
    },
    {
      icon: <Layers3 size={15} />,
      label: "语义模型",
      value: modelLines ? (
        <span className="system-status-model-lines">
          <span>{modelLines.embedding}</span>
          <span>{modelLines.reranker}</span>
        </span>
      ) : modelLabel(health),
      title: modelLabel(health),
      tone: health?.embedding_model_ready && health?.reranker_model_ready ? "ok" : health ? "warning" : "unknown",
    },
    {
      icon: <Database size={15} />,
      label: "向量检索",
      value: vectorLabel(health),
      title: vectorLabel(health),
      tone: health?.qdrant_ready ? "ok" : health ? "warning" : "unknown",
    },
    {
      icon: <FileCheck2 size={15} />,
      label: "知识库",
      value: `${system.indexedDocumentCount} 份可问答 · ${system.processingDocumentCount} 份处理中`,
      title: `${system.indexedDocumentCount} 份可问答 · ${system.processingDocumentCount} 份处理中`,
      tone: knowledgeBaseAvailable ? "ok" : health ? "warning" : "unknown",
    },
    {
      icon: <ServerCog size={15} />,
      label: "任务队列",
      value: queue.label,
      title: queue.label,
      tone: queue.tone,
    },
  ];

  return (
    <section className="system-status-panel" aria-labelledby="system-status-title">
      <header className="system-status-panel__head">
        <div>
          <span className={`status-dot ${system.error ? "error" : ready ? "ok" : system.isLoading ? "loading" : "warning"}`} />
          <div>
            <strong id="system-status-title">系统状态</strong>
            <small>{system.error ? "部分状态无法读取" : ready ? "智能问答服务已就绪" : "服务尚未完全就绪"}</small>
          </div>
        </div>
        <button className="status-refresh-button" type="button" disabled={system.isLoading} onClick={() => void system.refresh()} aria-label="刷新系统状态" title="刷新系统状态">
          <RefreshCw size={14} className={system.isLoading ? "spinning" : undefined} />
        </button>
      </header>

      <div className="system-status-list">
        {rows.map((row) => (
          <div className="system-status-row" key={row.label}>
            <span className="system-status-row__icon">{row.icon}</span>
            <div><span>{row.label}</span><strong title={row.title}>{row.value}</strong></div>
            <i className={`system-state-mark ${row.tone}`} aria-hidden="true" />
          </div>
        ))}
      </div>

      <details className="system-status-details">
        <summary>基础设施详情</summary>
        <dl>
          <div><dt>问答就绪</dt><dd>{readiness.detail}</dd></div>
          <div><dt>SQLite</dt><dd>{health ? (health.sqlite_ready ? "可用" : "不可用") : "未读取"}</dd></div>
          <div><dt>Office 解析</dt><dd>{parserLabel(health)}</dd></div>
          <div><dt>Collection</dt><dd>{health?.qdrant_collection || "未读取"}</dd></div>
          <div><dt>CUDA 状态</dt><dd>{cudaStatusLabel(health)}</dd></div>
          <div><dt>当前设备</dt><dd>{health?.model_device.selected_device || "未读取"}</dd></div>
          <div><dt>性能模式</dt><dd>{performanceModeLabel(health)}</dd></div>
          <div><dt>推理配置</dt><dd>{inferenceConfigLabel(health)}</dd></div>
          <div><dt>Build ID</dt><dd>{health?.build_id || "未读取"}</dd></div>
          <div><dt>最近刷新</dt><dd>{formatRefreshTime(system.refreshedAt)}</dd></div>
          {deviceFallback ? <div className="system-status-fallback"><dt>降级原因</dt><dd>{fallbackDetail(health)}</dd></div> : null}
          {health?.performance?.backend_fallback_reason ? (
            <div className="system-status-fallback"><dt>后端回退</dt><dd>{backendFallbackDetail(health)}</dd></div>
          ) : null}
          {system.error ? <div className="system-status-fallback"><dt>读取错误</dt><dd>{system.error}</dd></div> : null}
        </dl>
      </details>
    </section>
  );
}

function readinessSummary(health: RagHealthResponse | null, indexedDocumentCount: number): { label: string; detail: string; tone: "ok" | "warning" | "unknown" } {
  if (!health) return { label: "状态未读取", detail: "尚未读取后端健康状态。", tone: "unknown" };
  if (!health.sqlite_ready) return { label: "数据库不可用", detail: "本地元数据数据库暂不可用，文档和证据状态无法确认。", tone: "warning" };
  if (!health.qdrant_ready) return { label: "向量库不可用", detail: "Qdrant 暂不可用，无法执行语义检索。", tone: "warning" };
  if (!health.qdrant_collection_ready) return { label: "索引未就绪", detail: `Qdrant collection ${health.qdrant_collection} 尚未准备好。`, tone: "warning" };
  if (!health.embedding_model_ready || !health.reranker_model_ready) {
    return { label: "模型未就绪", detail: "BGE-Base-ZH 或 Reranker 模型尚未下载或加载。", tone: "warning" };
  }
  if (!health.ready) return { label: "依赖未就绪", detail: "智能问答依赖尚未全部通过健康检查。", tone: "warning" };
  if (indexedDocumentCount <= 0) return { label: "暂无可问答文档", detail: "系统依赖已就绪，但知识库里还没有完成索引的文档。", tone: "warning" };
  const warmup = health.performance?.warmup;
  if (health.performance?.warmup_policy === "background" && !warmup?.warmed) {
    return { label: "已就绪", detail: `系统依赖已就绪，当前有 ${indexedDocumentCount} 份文档可用于智能问答；后台预热未完成时，首次问答可能需要等待模型加载。`, tone: "ok" };
  }
  return { label: "已就绪", detail: `系统依赖已就绪，当前有 ${indexedDocumentCount} 份文档可用于智能问答。`, tone: "ok" };
}

function computeLabel(health: RagHealthResponse | null): string {
  if (!health) return "状态未读取";
  const device = health.model_device;
  const selected = device.selected_device.toLowerCase();
  if (selected.startsWith("cuda")) {
    const loaded = health.model_runtime.embedding?.loaded || health.model_runtime.reranker?.loaded;
    return loaded ? "CUDA 推理已就绪" : "CUDA 可用";
  }
  if (isDeviceFallback(health)) return "CUDA 异常 · 已回退至 CPU";
  if (health.performance?.selected_mode === "cpu_low_resource") return "CPU 低资源模式（实验）";
  return "CPU 平衡模式";
}

function isDeviceFallback(health: RagHealthResponse | null): boolean {
  if (!health) return false;
  const device = health.model_device;
  const selected = device.selected_device.toLowerCase();
  const explicitlyRequestedGpu = health.performance?.requested_mode === "gpu" || device.requested_device === "cuda";
  return !selected.startsWith("cuda") && explicitlyRequestedGpu && Boolean(device.fallback_reason);
}

function cudaStatusLabel(health: RagHealthResponse | null): string {
  if (!health) return "未读取";
  const device = health.model_device;
  if (!device.cuda_available) return "不可用";
  const name = device.cuda_device_name ? ` · ${device.cuda_device_name}` : "";
  return `可用${name}`;
}

function fallbackDetail(health: RagHealthResponse | null): string {
  if (!health) return "未读取";
  return health.model_device.fallback_reason || "请求了 CUDA，但运行时已回退到 CPU。";
}

function performanceModeLabel(health: RagHealthResponse | null): string {
  if (!health?.performance) return "未读取";
  const labels: Record<string, string> = {
    gpu: "GPU",
    cpu_balanced: "CPU 平衡",
    cpu_low_resource: "CPU 低资源（实验）",
  };
  return labels[health.performance.selected_mode] || health.performance.selected_mode;
}

function inferenceConfigLabel(health: RagHealthResponse | null): string {
  if (!health?.performance) return "未读取";
  const value = health.performance;
  const nativeThreads = value.omp_num_threads ? ` · OMP ${value.omp_num_threads}` : "";
  return `${value.backend} · Emb ${value.embedding_batch_size} · Rerank ${value.rerank_batch_size} · Torch ${value.torch_num_threads}${nativeThreads}`;
}

function backendFallbackDetail(health: RagHealthResponse | null): string {
  if (!health?.performance?.backend_fallback_reason) return "无";
  const requested = health.performance.requested_backend || "未知后端";
  return `${requested} 尚未通过质量门禁，当前实际使用 ${health.performance.backend}`;
}

function modelLabel(health: RagHealthResponse | null): string {
  if (!health) return "状态未读取";
  const lines = modelStatusLines(health);
  return lines ? `${lines.embedding} · ${lines.reranker}` : "状态未读取";
}

function modelStatusLines(health: RagHealthResponse | null): { embedding: string; reranker: string } | null {
  if (!health) return null;
  return {
    embedding: runtimeLabel("BGE-Base-ZH", health.embedding_model_ready, health.model_runtime.embedding),
    reranker: runtimeLabel("Reranker-Base", health.reranker_model_ready, health.model_runtime.reranker),
  };
}

function runtimeLabel(name: string, available: boolean, runtime?: { loaded?: boolean; warmed?: boolean }): string {
  if (!available) return `${name} 未下载`;
  if (runtime?.loaded || runtime?.warmed) return `${name} 已就绪`;
  return `${name} 已就绪`;
}

function vectorLabel(health: RagHealthResponse | null): string {
  if (!health) return "状态未读取";
  if (!health.qdrant_ready) return "Qdrant 不可用";
  return "Qdrant 已可用";
}

function queueSummary(health: RagHealthResponse | null): { label: string; tone: "ok" | "warning" | "unknown" } {
  if (!health) return { label: "状态未读取", tone: "unknown" };
  const running = countStatuses(health.index_tasks, ["running"]) + countStatuses(health.qa_tasks, ["running"]);
  const queued = Math.max(health.index_tasks.queued ?? 0, health.index_tasks.queue_depth ?? 0)
    + Math.max(health.qa_tasks.queued ?? 0, health.qa_tasks.queue_depth ?? 0);
  if (running || queued) return { label: `${running} 项运行 · ${queued} 项等待`, tone: "warning" };
  return { label: "当前空闲", tone: "ok" };
}

function countStatuses(source: Record<string, number>, statuses: string[]): number {
  return statuses.reduce((total, status) => total + (source[status] ?? 0), 0);
}

function parserLabel(health: RagHealthResponse | null): string {
  if (!health) return "未读取";
  const available = [health.libreoffice_ready ? "LibreOffice" : null, health.antiword_ready ? "Antiword" : null].filter(Boolean);
  return available.length ? available.join(" / ") : "不可用";
}

function formatRefreshTime(value: string | null): string {
  if (!value) return "尚未刷新";
  return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(new Date(value));
}
