/**
 * ResumeMind 品牌标识：渐变圆角方块 + 简历文档 + 对话气泡 + Agent 星点。
 * 寓意"简历智能问答 Agent"——文档代表简历、气泡代表智能问答、星点代表 AI Agent。
 */
export function BrandMark({ size = 40 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none" aria-hidden="true">
      <defs>
        <linearGradient id="resumemind-brand-bg" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
          <stop stopColor="#c8102e" />
          <stop offset="1" stopColor="#a30d26" />
        </linearGradient>
      </defs>
      {/* 背景圆角方块 */}
      <rect width="40" height="40" rx="11" fill="url(#resumemind-brand-bg)" />
      {/* Agent 星点：右上角四角星（智能感） */}
      <path
        d="M30.6 5.8l1.5 3.7 3.7 1.5-3.7 1.5-1.5 3.7-1.5-3.7-3.7-1.5 3.7-1.5z"
        fill="#fcd34d"
      />
      {/* 简历文档：左中白色文档 + 折角 + 三行文字 */}
      <path
        d="M10.5 9.5h9l5 5v15.2a2 2 0 0 1-2 2h-12a2 2 0 0 1-2-2V11.5a2 2 0 0 1 2-2z"
        stroke="white"
        strokeWidth="1.7"
        fill="rgba(255,255,255,0.14)"
        strokeLinejoin="round"
      />
      <path d="M19.5 9.5V14.5h5" stroke="white" strokeWidth="1.7" strokeLinejoin="round" />
      <path d="M13.5 18.5h8M13.5 22.5h8M13.5 26.5h5" stroke="white" strokeWidth="1.6" strokeLinecap="round" />
      {/* 对话气泡：右下角气泡 + 三点 */}
      <path
        d="M27.2 28.8c3 0 4.9-1.8 4.9-4.1s-1.9-4.1-4.9-4.1h-1.9"
        stroke="white"
        strokeWidth="1.6"
        strokeLinecap="round"
        fill="none"
      />
      <circle cx="26.4" cy="24.7" r="0.95" fill="white" />
      <circle cx="29.3" cy="24.7" r="0.95" fill="white" />
      <circle cx="32.2" cy="24.7" r="0.95" fill="white" />
    </svg>
  );
}
