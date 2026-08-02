import { KeyRound, Loader2 } from "lucide-react";
import { FormEvent, useState } from "react";

import { ApiError } from "../api/client";
import { useAuth } from "../state/authContext";

interface LoginPageProps {
  /** 登录成功后跳转的路径（如 /documents）。 */
  redirectPath: string;
  onNavigate: (path: string) => void;
}

export function LoginPage({ redirectPath, onNavigate }: LoginPageProps) {
  const { login } = useAuth();
  const [password, setPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!password || isSubmitting) return;
    setIsSubmitting(true);
    setError(null);
    try {
      await login(password);
      onNavigate(redirectPath);
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 401) {
        setError("密码错误，请重试。");
      } else if (caught instanceof ApiError && caught.status === 429) {
        setError("尝试次数过多，请稍后再试。");
      } else {
        setError("登录失败，请检查网络后重试。");
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="page login-page">
      <header className="product-header">
        <div>
          <p className="eyebrow">管理员后台</p>
          <div className="product-title-lockup">
            <h1>登录</h1>
          </div>
          <p className="page-lead">输入管理员密码进入后台，管理知识库与系统状态。</p>
        </div>
      </header>

      <section className="panel login-card">
        <form onSubmit={(event) => void handleSubmit(event)}>
          <label className="login-label" htmlFor="admin-password">
            <KeyRound size={16} />
            管理员密码
          </label>
          <input
            id="admin-password"
            className="login-input"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder="请输入密码"
            autoFocus
            disabled={isSubmitting}
          />
          {error ? (
            <p className="login-error" role="alert">{error}</p>
          ) : null}
          <button className="primary-button" type="submit" disabled={!password || isSubmitting}>
            {isSubmitting ? <Loader2 size={17} className="spinning" /> : <KeyRound size={17} />}
            进入后台
          </button>
        </form>
      </section>
    </main>
  );
}
