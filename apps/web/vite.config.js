import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 把 product 后端(FastAPI :8001)的路由代理过来，避免 CORS
const target = process.env.PROACTIVE_API || "http://localhost:8001";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5175,
    strictPort: true,
    proxy: {
      // ws:true —— /sessions/{id}/asr/stream 是 WebSocket(阿里云流式 ASR),必须转发 ws
      "/sessions": { target, ws: true, changeOrigin: true },
      "/prompt-decisions": target,
      "/prompts": target,
      "/memories": target,
      "/asr": target,
      "/health": target,
    },
  },
});
