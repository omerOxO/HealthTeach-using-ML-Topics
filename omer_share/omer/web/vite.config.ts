import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

declare const process: { env: Record<string, string | undefined> } // no @types/node in this project

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // API_TARGET lets a second dev server point at a second backend (e.g. http://127.0.0.1:8010)
    proxy: { '/api': process.env.API_TARGET ?? 'http://127.0.0.1:8000' },
  },
})
