import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
// From vitest/config, not vite: the same function widened to accept `test`.
import { defineConfig } from "vitest/config";

// Proxies the same paths nginx does, so dev and prod share one code path and the
// browser never makes a cross-origin request.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  test: {
    environment: "jsdom",
    include: ["tests/**/*.test.{ts,tsx}"],
  },
});
