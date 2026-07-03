import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev-mode proxy so `npm run dev` (port 5173) talks to the FastAPI server
// (port 8000) without CORS friction. The production build is served by
// FastAPI itself from frontend/dist, same origin.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
