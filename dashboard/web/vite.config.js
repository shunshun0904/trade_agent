import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// VITE_DATA_SOURCE=mock のときは 1 ファイルにまとめる（プレビューページ用。API には接続しない）。
// 開発時は VITE_API_BASE に API Gateway の URL を入れると /api を転送する。
export default defineConfig(({ mode }) => {
  const mock = process.env.VITE_DATA_SOURCE === "mock";
  return {
    plugins: [react(), ...(mock ? [viteSingleFile()] : [])],
    base: "./",
    build: { outDir: mock ? "dist-preview" : "dist", emptyOutDir: true },
    server: {
      proxy: process.env.VITE_API_BASE
        ? { "/api": { target: process.env.VITE_API_BASE, changeOrigin: true } }
        : undefined,
    },
  };
});
