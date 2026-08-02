import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
  },
  server: {
    hmr: {
      host: "127.0.0.1",
      clientPort: Number(process.env.VITE_HMR_CLIENT_PORT ?? process.env.RESUME_FRONTEND_PORT ?? 5173),
    },
    proxy: {
      "/api": process.env.VITE_API_PROXY_TARGET ?? "http://127.0.0.1:8000",
    },
  },
});
