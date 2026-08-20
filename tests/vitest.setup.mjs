// Vitest global setup (see vitest.config.js `test.setupFiles`).
//
// Node >= 26 ships an experimental `localStorage` accessor on globalThis that
// evaluates to `undefined` unless node is started with --localstorage-file.
// Vitest's happy-dom environment populates the test global by copying window
// keys that are NOT already present on globalThis — and in this environment
// `window === globalThis` — so Node's key wins, happy-dom's working
// localStorage is never installed, and every `window.localStorage.*` call in
// the suite throws. Replace any undefined web-storage global with a fresh
// happy-dom Storage so the suite passes on bare `npx vitest run` under
// Node 26 without contributors having to set NODE_OPTIONS. This is a no-op on
// Node <= 24 (CI), where happy-dom's storage is installed normally.
import { Storage } from 'happy-dom';

for (const key of ['localStorage', 'sessionStorage']) {
  if (globalThis[key] === undefined) {
    Object.defineProperty(globalThis, key, {
      value: new Storage(),
      configurable: true,
      writable: true
    });
  }
}
