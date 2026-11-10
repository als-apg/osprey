// `osprey app check` runs this with the application's directory as vitest's root.
import process from 'node:process';
import { fileURLToPath } from 'node:url';

// Detects a hang; never a speed budget. The same bound as the framework's own suite.
const HANG_BOUND_MS = 600_000;

export default {
  // Never under the root: an application's directory is build input.
  cacheDir: fileURLToPath(new URL('./.vite/', import.meta.url)),
  resolve: {
    alias: {
      // Panels import the design system by the absolute URL its server maps; the command
      // names the installed directory, this file having been copied beside the toolchain.
      '/design-system/js': process.env.OSPREY_DESIGN_SYSTEM_JS,
    },
  },
  test: {
    environment: 'happy-dom',
    testTimeout: HANG_BOUND_MS,
    hookTimeout: HANG_BOUND_MS,
    pool: 'vmThreads',
    setupFiles: [fileURLToPath(new URL('./vitest.setup.mjs', import.meta.url))],
    include: ['tests/**/*.test.{js,mjs}'],
  },
};
