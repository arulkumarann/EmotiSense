import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In development the UI runs on :5173 and talks to the local model server on :8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 900 },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
