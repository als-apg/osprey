// Node >= 26 ships an experimental `localStorage` accessor on globalThis that evaluates to
// `undefined` unless node is started with --localstorage-file. Vitest's happy-dom
// environment only copies window keys globalThis lacks, so Node's key wins and every
// `window.localStorage.*` throws. Install a happy-dom Storage in its place; a no-op where
// happy-dom's storage is installed normally.
import { Storage } from 'happy-dom';

for (const key of ['localStorage', 'sessionStorage']) {
  if (Reflect.get(globalThis, key) === undefined) {
    Object.defineProperty(globalThis, key, {
      value: new Storage(),
      configurable: true,
      writable: true,
    });
  }
}
