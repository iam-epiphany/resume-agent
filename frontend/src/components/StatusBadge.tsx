interface StatusBadgeProps {
  tone: "neutral" | "ok" | "warning" | "error";
  children: string;
}

export function StatusBadge({ tone, children }: StatusBadgeProps) {
  return <span className={`status-badge ${tone}`}>{children}</span>;
}
