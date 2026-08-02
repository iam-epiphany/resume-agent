import type { ReactNode } from "react";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { getMe, login as loginRequest } from "../api/auth";
import { AUTH_EXPIRED_EVENT, clearAuthToken, getAuthToken, setAuthToken } from "../api/client";

interface AuthContextValue {
  isAuthenticated: boolean;
  login: (password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  // 初始态：本地有 token 即认为已登录（随后用 /me 校验）
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(() => Boolean(getAuthToken()));

  useEffect(() => {
    // 启动时校验 token 有效性；网络错误时乐观保持登录态（离线场景不误登出）
    let cancelled = false;
    if (!getAuthToken()) {
      return;
    }
    void getMe()
      .then(() => {
        if (!cancelled) setIsAuthenticated(true);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        const status = (error as { status?: number } | null)?.status;
        if (status === 401) {
          clearAuthToken();
          setIsAuthenticated(false);
        } else {
          setIsAuthenticated(true); // 网络/服务错误：乐观保持
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const handleExpired = () => setIsAuthenticated(false);
    window.addEventListener(AUTH_EXPIRED_EVENT, handleExpired);
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, handleExpired);
  }, []);

  const login = useCallback(async (password: string) => {
    const response = await loginRequest(password);
    setAuthToken(response.token);
    setIsAuthenticated(true);
  }, []);

  const logout = useCallback(() => {
    clearAuthToken();
    setIsAuthenticated(false);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({ isAuthenticated, login, logout }),
    [isAuthenticated, login, logout],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return context;
}
