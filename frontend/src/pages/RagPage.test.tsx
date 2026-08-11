import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { qaAccessStatus } from "../api/qa";
import type { LoadLevel } from "../state/readinessContext";
import { RagPage } from "./RagPage";

/** 通过 hoisted 共享的可变 readiness 状态：用例间切换负载分级 */
const hoisted = vi.hoisted(() => ({
  readiness: {
    ready: true,
    message: "系统已就绪，可以开始提问。",
    isLoading: false,
    error: null,
    refreshedAt: "2026-08-12T00:00:00Z",
    refresh: () => Promise.resolve(),
    loadLevel: null as LoadLevel | null,
  },
}));

vi.mock("../api/qa", () => ({
  qaAccessStatus: vi.fn(),
  qaAccessSubmit: vi.fn(),
}));
vi.mock("../state/authContext", () => ({
  useAuth: () => ({ isAuthenticated: false }),
}));
vi.mock("../state/readinessContext", () => ({
  useReadiness: () => hoisted.readiness,
}));
vi.mock("../state/chatHistoryContext", () => ({
  useChatHistory: () => ({ activeThread: null }),
}));
vi.mock("../state/qaTaskContext", () => ({
  useQATask: () => ({
    draftQuestion: "",
    setDraftQuestion: () => undefined,
    taskStatus: "idle",
    isCancelling: false,
    cancelQuestion: () => Promise.resolve(),
    runQuestion: () => Promise.resolve(),
  }),
  isActiveTaskStatus: () => false,
}));

const qaAccessStatusMock = vi.mocked(qaAccessStatus);

function renderRagPage() {
  return render(<RagPage />);
}

describe("RagPage 负载指示灯与繁忙横幅", () => {
  beforeEach(() => {
    hoisted.readiness.loadLevel = null;
    qaAccessStatusMock.mockReset();
    qaAccessStatusMock.mockResolvedValue({
      access_enabled: false,
      granted: true,
      daily_used: 0,
      daily_remaining: 100,
      daily_limit: 100,
      budget_warning: false,
    });
  });

  it("负载未知时不渲染指示灯（避免首帧误报'正常'）", () => {
    renderRagPage();
    expect(screen.queryByText(/系统负载正常/)).toBeNull();
  });

  it("绿色：只显示指示灯，不显示繁忙横幅", () => {
    hoisted.readiness.loadLevel = "green";
    renderRagPage();
    expect(screen.getByText("系统负载正常")).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("黄色：显示黄色指示灯与'回复可能较慢'横幅", () => {
    hoisted.readiness.loadLevel = "yellow";
    renderRagPage();
    expect(screen.getByText("当前访问人数较多，回复可能较慢")).toBeTruthy();
    const banner = screen.getByRole("alert");
    expect(banner.textContent).toContain("当前访问人数较多，系统负载较高，回复可能较慢");
    expect(banner.className).toContain("load-banner-yellow");
  });

  it("红色：横幅文案提示排队较久", () => {
    hoisted.readiness.loadLevel = "red";
    renderRagPage();
    expect(screen.getByText("系统繁忙，请稍后再试")).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toContain("可能排队较久");
  });

  it("点击关闭后横幅消失；回绿再转黄时重新出现", () => {
    hoisted.readiness.loadLevel = "yellow";
    const { rerender } = renderRagPage();
    expect(screen.getByRole("alert")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "关闭提醒" }));
    expect(screen.queryByRole("alert")).toBeNull();
    // 指示灯不受关闭影响，仍在
    expect(screen.getByText("当前访问人数较多，回复可能较慢")).toBeTruthy();

    // 负载回绿 → 重置关闭态
    hoisted.readiness.loadLevel = "green";
    rerender(<RagPage />);
    expect(screen.queryByRole("alert")).toBeNull();

    // 再次转黄 → 横幅重新出现
    hoisted.readiness.loadLevel = "yellow";
    rerender(<RagPage />);
    expect(screen.getByRole("alert")).toBeTruthy();
  });

  it("黄色状态下仍保持绿色时横幅不出现（未转黄）", () => {
    hoisted.readiness.loadLevel = "green";
    const { rerender } = renderRagPage();
    expect(screen.queryByRole("alert")).toBeNull();
    hoisted.readiness.loadLevel = "yellow";
    rerender(<RagPage />);
    expect(screen.getByRole("alert")).toBeTruthy();
  });
});
