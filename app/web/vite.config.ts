import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The API runs on :8000 (uvicorn app.api.main:app); the dev server proxies /api to it.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  build: {
    chunkSizeWarningLimit: 1500, // echarts is one large, cacheable chunk
    rollupOptions: {
      output: {
        // Charting code in its own long-cacheable chunk (tree-shaken via echarts/core).
        manualChunks(id) {
          if (id.includes("node_modules/echarts") || id.includes("node_modules/zrender")) return "echarts";
          return undefined;
        },
      },
    },
  },
});
