import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    allowedHosts: ["enigma", "node2", "http://enigma:4200","enigma:4200", "enigma:5173"],
    host: true,
    watch: {
      usePolling: true,
      interval: 300, // 100–300 is typical
    },    
    proxy: {
      "/generate": "http://localhost:4200",
      "/superres": "http://localhost:4200",
      "/v1": "http://localhost:4200",
      "/storage": "http://localhost:4200",
      "/dreams": "http://localhost:4200",
    },
  },
});
