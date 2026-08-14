/**
 * ResumeMind 品牌标识：以公开的 PNG 品牌图为源（frontend/public/brand-logo.png），
 * 用于侧栏品牌区、聊天助手头像与空状态图标。
 */
export function BrandMark({ size = 40 }: { size?: number }) {
  return (
    <img
      src="/brand-logo.png"
      width={size}
      height={size}
      alt=""
      aria-hidden="true"
      draggable={false}
      style={{ display: "block", objectFit: "contain" }}
    />
  );
}
