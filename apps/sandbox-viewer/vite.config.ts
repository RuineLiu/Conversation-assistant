import { defineConfig } from 'vite';

const apiTarget = process.env.SANDBOX_API_TARGET || 'http://127.0.0.1:8000';

export default defineConfig({
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
