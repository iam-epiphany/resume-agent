import { useEffect, useMemo, useState } from "react";

import { AppShell } from "./components/AppShell";
import { AuditPage } from "./pages/AuditPage";
import { DocumentsPage } from "./pages/DocumentsPage";
import { LoginPage } from "./pages/LoginPage";
import { RagPage } from "./pages/RagPage";
import { AuthProvider, useAuth } from "./state/authContext";
import { ChatHistoryProvider } from "./state/chatHistoryContext";
import { QATaskProvider } from "./state/qaTaskContext";
import { ReadinessProvider } from "./state/readinessContext";
import { SystemStatusProvider } from "./state/systemStatusContext";

export function App() {
  return (
    <AuthProvider>
      <ReadinessProvider>
        {/* 系统状态（后台信息）仅登录后轮询；匿名前台用公开就绪状态 */}
        <SystemStatusProvider>
          <ChatHistoryProvider>
            <QATaskProvider>
              <RoutedApp />
            </QATaskProvider>
          </ChatHistoryProvider>
        </SystemStatusProvider>
      </ReadinessProvider>
    </AuthProvider>
  );
}

function RoutedApp() {
  const { isAuthenticated } = useAuth();
  const [path, setPath] = useState(() => window.location.pathname);

  function navigate(nextPath: string) {
    window.history.pushState({}, "", nextPath);
    setPath(nextPath);
  }

  useEffect(() => {
    const handlePopState = () => setPath(window.location.pathname);
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const route = useMemo(() => matchRoute(path), [path]);

  return (
    <AppShell path={path} onNavigate={navigate}>
      {renderRoute(route, isAuthenticated, navigate)}
    </AppShell>
  );
}

type Route = { name: "documents" } | { name: "qa" } | { name: "audit" } | { name: "login" };

function matchRoute(path: string): Route {
  const parts = path.split("/").filter(Boolean);
  if (parts[0] === "documents") {
    return { name: "documents" };
  }
  if (parts[0] === "qa") {
    return { name: "qa" };
  }
  if (parts[0] === "audit") {
    return { name: "audit" };
  }
  if (parts[0] === "login") {
    return { name: "login" };
  }
  return { name: "qa" };
}

function renderRoute(
  route: Route,
  isAuthenticated: boolean,
  navigate: (path: string) => void,
) {
  if (route.name === "documents") {
    // 知识库是后台功能：未登录先引导登录，登录后跳转回原路径
    return isAuthenticated ? <DocumentsPage /> : <LoginPage redirectPath="/documents" onNavigate={navigate} />;
  }
  if (route.name === "qa") {
    return <RagPage />;
  }
  if (route.name === "audit") {
    return <AuditPage />;
  }
  if (route.name === "login") {
    return <LoginPage redirectPath="/" onNavigate={navigate} />;
  }
  return <RagPage />;
}
