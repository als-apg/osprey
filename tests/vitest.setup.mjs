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
import { afterAll, afterEach } from 'vitest';
import { Storage } from 'happy-dom';

for (const key of ['localStorage', 'sessionStorage']) {
  if (Reflect.get(globalThis, key) === undefined) {
    Object.defineProperty(globalThis, key, {
      value: new Storage(),
      configurable: true,
      writable: true
    });
  }
}

// happy-dom navigates a child frame for real: an iframe with a src issues an
// HTTP request for that document to the environment's http://localhost:3000
// origin, where no server is listening. The request outlives the test that
// created the frame and its ECONNREFUSED surfaces after the last file, so the
// run reports every test passing and still exits non-zero. Suites here assert
// on an iframe's `src` and on messages posted to its window, never on a
// document it loaded, so the frame needs its URL and not a page: with child
// frame navigation disabled happy-dom sets the frame's location and skips the
// request, leaving `contentWindow` in place.
// happy-dom's environment object is not a key of `typeof globalThis`; read it
// once through Reflect.get, which is typed for a dynamic key, with the shape
// the two settings below rely on.
/**
 * @type {{ settings?: {
 *   navigation?: { disableChildFrameNavigation?: boolean },
 *   fetch?: { interceptor?: unknown }
 * } } | undefined}
 */
const happyDOM = Reflect.get(globalThis, 'happyDOM');

if (happyDOM?.settings?.navigation) {
  happyDOM.settings.navigation.disableChildFrameNavigation = true;
}

// Nothing serves the environment's origin (http://localhost:3000), so a request
// no test stubbed is a missing stub: the module under test reached for a
// dependency its test never declared. Answering it would hide that — a failed
// request is exactly what an unreachable origin gives, so the suite stays green
// while its subject talks to nothing. The request is refused instead, at the
// call site and by URL.
//
// A refusal on its own is not enough: most callers swallow a failed boot fetch
// on purpose, so the refusal would vanish into the code under test. Every
// refused URL is therefore recorded, and the record is drained after each test
// and after the file — whatever is left in it fails the test that made the
// request, and a request made while the module graph loads lands on the file's
// first test rather than disappearing.
//
// The record is published on globalThis under this key so a test may make an
// unstubbed request deliberately: it reads the record, clears it, and the drain
// then finds nothing (see tests/vitest-fetch-guard.test.mjs).
const UNSTUBBED_RECORD_KEY = '__OSPREY_UNSTUBBED_FETCHES__';

/**
 * Every request this environment refused, in the order it refused them.
 * @type {string[]}
 */
const unstubbed = [];

Object.defineProperty(globalThis, UNSTUBBED_RECORD_KEY, {
  value: unstubbed,
  configurable: true,
  writable: true
});

/**
 * Refuse one request and record it.
 * @param {string} url
 * @returns {never}
 */
function refuse(url) {
  unstubbed.push(url);
  throw new Error(`no server for ${url} — stub fetch in the test`);
}

/** Report every refusal nothing has accounted for, and forget them. */
function drainUnstubbed() {
  if (unstubbed.length === 0) return;
  const urls = unstubbed.splice(0, unstubbed.length);
  throw new Error(`no server for ${urls.join(', ')} — stub fetch in the test`);
}

afterEach(drainUnstubbed);
afterAll(drainUnstubbed);

if (happyDOM?.settings?.fetch) {
  happyDOM.settings.fetch.interceptor = {
    /** @param {{ request: { url: string } }} context */
    beforeAsyncRequest: async ({ request }) => refuse(request.url),
    /** @param {{ request: { url: string } }} context */
    beforeSyncRequest: ({ request }) => refuse(request.url)
  };
}
