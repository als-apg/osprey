import { fileURLToPath } from 'node:url';

// Design-system modules are served at runtime under the absolute URL
// `/design-system/js/*` (each panel's HTTP server maps that prefix to this
// directory). Alias it here so the same absolute specifiers resolve under
// Node/Vitest. Vite's alias matcher requires an exact match or a match
// followed by "/", so this cannot accidentally match mid-path.
const designSystemJsDir = fileURLToPath(
  new URL('./src/osprey/interfaces/design_system/static/js', import.meta.url)
);

// One hang bound for every test and hook, sized like the Python lane's
// per-test cap (pytest --timeout=600). It detects a hang; it is never a speed
// budget. A JS time budget is not expressed through it: a test that needs one
// is timed alone in a serial step, or as a ratio against a baseline measured
// in the same run, the two shapes tests/README.md gives for Python. No test or
// invocation lowers it.
const HANG_BOUND_MS = 600_000;

export default {
  resolve: {
    alias: {
      '/design-system/js': designSystemJsDir
    }
  },
  test: {
    environment: 'happy-dom',
    testTimeout: HANG_BOUND_MS,
    hookTimeout: HANG_BOUND_MS,
    // Every test file gets a fresh VM context (its own happy-dom window and its
    // own evaluation of the modules it imports) inside a worker thread that is
    // reused across files. The default forks pool starts a new process per file,
    // and each start must answer within a fixed timeout, so a loaded host can
    // fail a run whose tests all pass. isolate: false is not a substitute:
    // suites here leave location, body classes, DOM nodes and fetch stubs
    // behind, and sharing one window across files makes other files fail.
    // Caveat: an error thrown by a Node built-in is not `instanceof Error`
    // inside the test context.
    pool: 'vmThreads',
    // Works around Node >= 26 shadowing happy-dom's localStorage with its own
    // undefined experimental global (see the comment in the setup file).
    setupFiles: ['./tests/vitest.setup.mjs'],
    include: ['tests/**/*.test.{js,mjs}', 'src/osprey/interfaces/**/*.test.{js,mjs}']
  }
};
