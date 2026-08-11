import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { cancelQuestionTask, createQuestionTask, getQuestionTask, streamQuestionTask } from "../api/qa";
import type { QATaskStatusResponse } from "../types/api";
import { isActiveTaskStatus, QATaskProvider, useQATask } from "./qaTaskContext";

vi.mock("../api/qa", () => ({
  cancelQuestionTask: vi.fn(),
  createQuestionTask: vi.fn(),
  getQuestionTask: vi.fn(),
  streamQuestionTask: vi.fn(),
}));

const createMock = vi.mocked(createQuestionTask);
const getMock = vi.mocked(getQuestionTask);
const cancelMock = vi.mocked(cancelQuestionTask);
const streamMock = vi.mocked(streamQuestionTask);

function Probe() {
  const qa = useQATask();
  return (
    <div>
      <input aria-label="草稿" value={qa.draftQuestion} onChange={(event) => qa.setDraftQuestion(event.target.value)} />
      <button type="button" onClick={() => void qa.runQuestion(undefined, false)}>运行</button>
      <button type="button" disabled={!isActiveTaskStatus(qa.taskStatus)} onClick={() => void qa.cancelQuestion()}>停止生成</button>
      <span data-testid="draft">{qa.draftQuestion}</span>
      <span data-testid="question">{qa.currentQuestion}</span>
      <span data-testid="status">{qa.taskStatus}</span>
      <span data-testid="message">{qa.message}</span>
      <span data-testid="answer">{qa.answer?.answer ?? ""}</span>
      <span data-testid="preview">{qa.answerPreview?.answer ?? ""}</span>
      <span data-testid="revision">{qa.answerPreview?.revision ?? ""}</span>
    </div>
  );
}

function completedTask(clientRequestId: string, question = "你参与过哪些项目？", taskId = "task-0001"): QATaskStatusResponse {
  return {
    task_id: taskId,
    client_request_id: clientRequestId,
    question,
    options: [],
    include_debug: false,
    session_id: null,
    status: "completed",
    progress_events: [{
      stage: "retrieval",
      status: "completed",
      title: "检索完成",
      detail: "已获取相关材料",
    }],
    answer: {
      answer: "项目经历包括外卖平台、REV 与秒杀系统。",
      answer_mode: "answered",
      evidence_sufficiency: "sufficient",
      hedge_note: null,
      intent: "asset_validation",
      resolved_question: "你参与过哪些项目？",
      retrieval_fallback_level: 0,
      context_package: null,
      degraded: false,
      generation_status: "completed",
    },
    error: null,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    completed_at: new Date().toISOString(),
  };
}

function enterDraft(value = "你参与过哪些项目？") {
  fireEvent.change(screen.getByRole("textbox", { name: "草稿" }), { target: { value } });
}

describe("QATaskProvider", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
    createMock.mockReset();
    getMock.mockReset();
    cancelMock.mockReset();
    streamMock.mockReset();
    streamMock.mockImplementation((_taskId, _onTask, onConnectionChange) => {
      onConnectionChange(false);
      return () => undefined;
    });
    getMock.mockRejectedValue(new Error("getQuestionTask mock not configured"));
    cancelMock.mockRejectedValue(new Error("cancelQuestionTask mock not configured"));
  });

  it("clears the draft after submission and keeps the submitted question separate", async () => {
    createMock.mockImplementation(async (_question, clientRequestId) => ({
      task_id: "task-0001",
      client_request_id: clientRequestId,
      status: "queued",
    }));
    getMock.mockImplementation(async () => completedTask(createMock.mock.calls[0][1]));

    render(<QATaskProvider><Probe /></QATaskProvider>);
    enterDraft();
    fireEvent.click(screen.getByRole("button", { name: "运行" }));

    expect(screen.getByTestId("draft")).toBeEmptyDOMElement();
    expect(screen.getByTestId("question")).toHaveTextContent("你参与过哪些项目？");
    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("completed"));
    expect(createMock).toHaveBeenCalledTimes(1);
    expect(createMock.mock.calls[0][1]).toMatch(/^[A-Za-z0-9_]+$/);
    expect(createMock.mock.calls[0][2]).toBe(false);
    const sessionId = window.localStorage.getItem("resumemind.qa.session-id");
    expect(sessionId).not.toBeNull();
    expect(createMock.mock.calls[0][3]).toBe(sessionId);
    expect(window.localStorage.getItem("resumemind.qa.pending-receipt")).toBeNull();
    expect(window.localStorage.getItem("resumemind.qa.session")).toBeNull();
  });

  it("keeps the draft when the page child unmounts under the same provider", () => {
    function NavigationHarness() {
      const [show, setShow] = useState(true);
      return (
        <QATaskProvider>
          <button type="button" onClick={() => setShow((current) => !current)}>切换页面</button>
          {show ? <Probe /> : <p>其他页面</p>}
        </QATaskProvider>
      );
    }

    render(<NavigationHarness />);
    enterDraft("跨页面保留的草稿");
    fireEvent.click(screen.getByRole("button", { name: "切换页面" }));
    fireEvent.click(screen.getByRole("button", { name: "切换页面" }));
    expect(screen.getByRole("textbox", { name: "草稿" })).toHaveValue("跨页面保留的草稿");
  });

  it("replaces the visible result when a second question is submitted", async () => {
    const createdTasks = new Map<string, { requestId: string; question: string }>();
    createMock.mockImplementation(async (question, clientRequestId) => {
      const taskId = `task-${createdTasks.size + 1}`;
      createdTasks.set(taskId, { requestId: clientRequestId, question });
      return { task_id: taskId, client_request_id: clientRequestId, status: "queued" };
    });
    getMock.mockImplementation(async (taskId) => {
      const created = createdTasks.get(taskId);
      if (!created) throw new Error("unknown task");
      return completedTask(created.requestId, created.question, taskId);
    });

    render(<QATaskProvider><Probe /></QATaskProvider>);
    enterDraft("第一个独立问题");
    fireEvent.click(screen.getByRole("button", { name: "运行" }));
    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("completed"));
    enterDraft("第二个独立问题");
    fireEvent.click(screen.getByRole("button", { name: "运行" }));

    await waitFor(() => expect(screen.getByTestId("question")).toHaveTextContent("第二个独立问题"));
    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(2));
    expect(createMock.mock.calls.map((call) => call[0])).toEqual(["第一个独立问题", "第二个独立问题"]);
  });

  it("starts with an empty work area after a full provider remount", () => {
    window.localStorage.setItem("resumemind.qa.session", JSON.stringify({
      question: "旧版持久化问题",
      taskId: "old-task",
      taskStatus: "running",
    }));

    const first = render(<QATaskProvider><Probe /></QATaskProvider>);
    enterDraft("刷新前草稿");
    first.unmount();
    render(<QATaskProvider><Probe /></QATaskProvider>);

    expect(screen.getByRole("textbox", { name: "草稿" })).toHaveValue("");
    expect(screen.getByTestId("question")).toBeEmptyDOMElement();
    expect(screen.getByTestId("status")).toBeEmptyDOMElement();
    expect(getMock).not.toHaveBeenCalled();
  });

  it("reconciles a pending receipt with the same request id without restoring it to the work area", async () => {
    window.localStorage.setItem("resumemind.qa.pending-receipt", JSON.stringify({
      clientRequestId: "stable-request-id",
      question: "响应返回前刷新的问题",
      includeDebug: true,
    }));
    createMock.mockResolvedValue({
      task_id: "task-from-receipt",
      client_request_id: "stable-request-id",
      status: "queued",
    });

    render(<QATaskProvider><Probe /></QATaskProvider>);

    const sessionId = window.localStorage.getItem("resumemind.qa.session-id");
    await waitFor(() => expect(createMock).toHaveBeenCalledWith("响应返回前刷新的问题", "stable-request-id", true, sessionId));
    await waitFor(() => expect(window.localStorage.getItem("resumemind.qa.pending-receipt")).toBeNull());
    expect(screen.getByTestId("question")).toBeEmptyDOMElement();
    expect(screen.getByTestId("status")).toBeEmptyDOMElement();
    expect(getMock).not.toHaveBeenCalled();
  });

  it("keeps polling after a transient status failure", async () => {
    createMock.mockImplementation(async (_question, clientRequestId) => ({
      task_id: "task-0001",
      client_request_id: clientRequestId,
      status: "queued",
    }));
    getMock
      .mockRejectedValueOnce(new Error("temporary network failure"))
      .mockImplementation(async () => completedTask(createMock.mock.calls[0][1]));

    render(<QATaskProvider><Probe /></QATaskProvider>);
    enterDraft();
    fireEvent.click(screen.getByRole("button", { name: "运行" }));

    await waitFor(
      () => expect(screen.getByTestId("answer")).toHaveTextContent("项目经历包括外卖平台、REV 与秒杀系统"),
      { timeout: 3000 },
    );
    expect(getMock.mock.calls.length).toBeGreaterThanOrEqual(2);
  });

  it("uses the durable task stream as the primary verified-preview source", async () => {
    createMock.mockImplementation(async (_question, clientRequestId) => ({
      task_id: "task-stream-0001",
      client_request_id: clientRequestId,
      status: "queued",
    }));
    streamMock.mockImplementation((taskId, onTask, onConnectionChange) => {
      onConnectionChange(true);
      onTask({
        ...completedTask(createMock.mock.calls[0][1], "流式问题", taskId),
        status: "running",
        answer: null,
        answer_preview: {
          answer: "首条结论已通过依据核验。",
          revision: 3,
        },
        completed_at: null,
      });
      return () => undefined;
    });

    render(<QATaskProvider><Probe /></QATaskProvider>);
    enterDraft("流式问题");
    fireEvent.click(screen.getByRole("button", { name: "运行" }));

    await waitFor(() => expect(screen.getByTestId("preview")).toHaveTextContent("首条结论已通过依据核验"));
    expect(screen.getByTestId("revision")).toHaveTextContent("3");
    expect(streamMock).toHaveBeenCalledWith("task-stream-0001", expect.any(Function), expect.any(Function));
  });

  it("cancels the active durable task and keeps the cancelled state", async () => {
    createMock.mockImplementation(async (_question, clientRequestId) => ({
      task_id: "task-0001",
      client_request_id: clientRequestId,
      status: "queued",
    }));
    getMock.mockImplementation(async () => ({
      ...completedTask(createMock.mock.calls[0][1]),
      status: "running",
      answer: null,
      completed_at: null,
    }));
    cancelMock.mockImplementation(async () => ({
      ...completedTask(createMock.mock.calls[0][1]),
      status: "cancelled",
      answer: null,
    }));

    render(<QATaskProvider><Probe /></QATaskProvider>);
    enterDraft();
    fireEvent.click(screen.getByRole("button", { name: "运行" }));
    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("running"));
    fireEvent.click(screen.getByRole("button", { name: "停止生成" }));

    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("cancelled"));
    expect(cancelMock).toHaveBeenCalledWith("task-0001");
  });

  it("honors stop requests made before the create response returns", async () => {
    let resolveCreate: ((value: { task_id: string; client_request_id: string; status: "queued" }) => void) | undefined;
    createMock.mockImplementation(() => new Promise((resolve) => {
      resolveCreate = resolve;
    }));
    cancelMock.mockImplementation(async (taskId) => ({
      ...completedTask(createMock.mock.calls[0][1]),
      task_id: taskId,
      status: "cancelled",
      answer: null,
    }));

    render(<QATaskProvider><Probe /></QATaskProvider>);
    enterDraft();
    fireEvent.click(screen.getByRole("button", { name: "运行" }));
    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1));
    expect(window.localStorage.getItem("resumemind.qa.pending-receipt")).toContain(createMock.mock.calls[0][1]);
    fireEvent.click(screen.getByRole("button", { name: "停止生成" }));
    resolveCreate?.({
      task_id: "task-before-response",
      client_request_id: createMock.mock.calls[0][1],
      status: "queued",
    });

    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("cancelled"));
    expect(cancelMock).toHaveBeenCalledWith("task-before-response");
    expect(window.localStorage.getItem("resumemind.qa.pending-receipt")).toBeNull();
  });
});
