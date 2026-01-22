import { defineConfig } from 'vite'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  // Set base URL for GitHub Pages deployment
  // In dev: '/' (root), In production: '/gare-flags/' (or set via env var)
  base: mode === 'production' ? '/gare-flags/' : '/',
  build: {
    target: 'esnext',
  },
  optimizeDeps: {
    esbuildOptions: {
      target: 'esnext',
    },
  },
}))
