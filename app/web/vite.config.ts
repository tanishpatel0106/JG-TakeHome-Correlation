/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// The API runs on :8000 by default (uvicorn app.api.main:app); the dev server proxies /api to it.
// Override with VITE_API_TARGET=http://localhost:8100 npm run dev.
export default defineConfig(({ mode }) => {
  // loadEnv also picks up VITE_* variables from the shell environment.
  const env = loadEnv(mode, ".", "VITE_");
  const target = env.VITE_API_TARGET || "http://localhost:8000";
  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173,
      proxy: {
        "/api": { target, changeOrigin: true },
      },
    },
    test: {
      environment: "node",
      include: ["src/**/*.test.ts"],
    },
    build: {
      chunkSizeWarningLimit: 1500, // echarts is one large, cacheable chunk
      rollupOptions: {
        output: {
          // Charting code in its own long-cacheable chunk (tree-shaken via echarts/core).
          manualChunks(id: string) {
            if (id.includes("node_modules/echarts") || id.includes("node_modules/zrender")) return "echarts";
            return undefined;
          },
        },
      },
    },
  };
});
