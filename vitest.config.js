import { fileURLToPath } from 'node:url';

// Design-system modules are served at runtime under the absolute URL
// `/design-system/js/*` (each panel's HTTP server maps that prefix to this
// directory). Alias it here so the same absolute specifiers resolve under
// Node/Vitest. Vite's alias matcher requires an exact match or a match
// followed by "/", so this cannot accidentally match mid-path.
const designSystemJsDir = fileURLToPath(
  new URL('./src/osprey/interfaces/design_system/static/js', import.meta.url)
);

export default {
  resolve: {
    alias: {
      '/design-system/js': designSystemJsDir
    }
  },
  test: {
    environment: 'happy-dom',
    // Works around Node >= 26 shadowing happy-dom's localStorage with its own
    // undefined experimental global (see the comment in the setup file).
    setupFiles: ['./tests/vitest.setup.mjs'],
    include: ['tests/**/*.test.{js,mjs}', 'src/osprey/interfaces/**/*.test.{js,mjs}']
  }
};
