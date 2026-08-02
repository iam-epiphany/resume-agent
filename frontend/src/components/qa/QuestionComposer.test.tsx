import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { QuestionComposer } from "./QuestionComposer";

const baseProps = {
  value: "",
  onChange: vi.fn(),
  onSubmit: vi.fn(),
  active: false,
  cancelling: false,
  onCancel: vi.fn(),
  ready: true,
  scopeLabel: "当前知识库 · 3 份文档可问答",
};

describe("QuestionComposer", () => {
  it("grows from 52px to 240px and then scrolls internally", () => {
    const onChange = vi.fn();
    render(<QuestionComposer {...baseProps} value="多行问题" onChange={onChange} />);
    const textarea = screen.getByRole("textbox");
    Object.defineProperty(textarea, "scrollHeight", { configurable: true, value: 320 });

    fireEvent.change(textarea, { target: { value: "第一行\n第二行\n第三行" } });

    expect(onChange).toHaveBeenCalledWith("第一行\n第二行\n第三行");
    expect(textarea).toHaveStyle({ height: "240px", overflowY: "auto" });
  });

  it("submits with Enter when idle and keeps Shift+Enter for newlines", () => {
    const onSubmit = vi.fn();
    render(<QuestionComposer {...baseProps} value="测试问题" onSubmit={onSubmit} />);

    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" });
    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter", ctrlKey: true });

    expect(onSubmit).toHaveBeenCalledTimes(2);
    const submit = screen.getByRole("button", { name: "开始问答" });
    expect(submit).toHaveClass("composer-action-button", "composer-action-button--submit");
    expect(submit).toHaveTextContent("");
    expect(submit.querySelector("svg.lucide-send-horizontal")).toBeInTheDocument();
  });

  it("does not submit when Shift+Enter is pressed", () => {
    const onSubmit = vi.fn();
    render(<QuestionComposer {...baseProps} value="测试问题" onSubmit={onSubmit} />);

    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter", shiftKey: true });

    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("locks input and settings while exposing the stop action", () => {
    const onCancel = vi.fn();
    render(<QuestionComposer {...baseProps} active value="" onCancel={onCancel} />);

    expect(screen.getByRole("textbox")).toBeDisabled();
    expect(screen.queryByRole("button", { name: "开始问答" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "停止生成" }));
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "停止生成" })).toHaveTextContent("");
  });

  it("shows a disabled stopping state while cancellation is pending", () => {
    render(<QuestionComposer {...baseProps} active cancelling value="" />);
    expect(screen.getByRole("button", { name: "正在停止" })).toBeDisabled();
  });
});
