import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// 常に 1 ファイル（dist/index.html）にまとめる。Lambda がこのファイルを画面として返す（build.sh が app/page.html に置く）。
// VITE_DATA_SOURCE=mock はプレビューページ用（ダミーデータ、API には接続しない）。
// 開発時は VITE_API_BASE に API Gateway の URL を入れると /api を転送する。
export default defineConfig(() => {
  const mock = process.env.VITE_DATA_SOURCE === "mock";
  return {
    plugins: [react(), viteSingleFile()],
    base: "./",
    build: { outDir: mock ? "dist-preview" : "dist", emptyOutDir: true },
    server: {
      proxy: process.env.VITE_API_BASE
        ? { "/api": { target: process.env.VITE_API_BASE, changeOrigin: true } }
        : undefined,
    },
  };
});
