import { fileURLToPath } from 'node:url';

// The package's own test config (D20). It does not inherit the root
// vitest.config.js: the engine tests need neither happy-dom nor the
// design-system alias, and they run in plain Node threads. Run it from the
// repo root with `npm run test:dock-engine`.

// One hang bound for every test and hook, the same as the root config's. It
// detects a hang; it is never a speed budget.
const HANG_BOUND_MS = 600_000;

export default {
  root: fileURLToPath(new URL('.', import.meta.url)),
  test: {
    environment: 'node',
    pool: 'threads',
    testTimeout: HANG_BOUND_MS,
    hookTimeout: HANG_BOUND_MS,
    include: ['ts/**/*.test.ts']
  }
};
