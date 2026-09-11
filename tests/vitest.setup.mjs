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

// Nothing serves the environment's origin (http://localhost:3000), so a module
// whose fetch a test did not stub opens a real socket that is refused. The
// refusal lands after the test that started it has finished, where no test owns
// it: the run reports every test passing and still exits non-zero on the
// unhandled tail. Answer such a request in-process instead. The caller sees a
// failed request either way — an unreachable origin is what it would have got —
// and the failure is now deterministic and names the URL a test has yet to stub.
if (happyDOM?.settings?.fetch) {
  happyDOM.settings.fetch.interceptor = {
    /** @param {{ request: { url: string }, window: { Response: typeof Response } }} context */
    beforeAsyncRequest: async ({ request, window }) =>
      new window.Response(`no server for ${request.url} — stub fetch in the test`, {
        status: 503,
        statusText: 'Service Unavailable'
      }),
    /** @param {{ request: { url: string } }} context */
    beforeSyncRequest: ({ request }) => ({
      status: 503,
      statusText: 'Service Unavailable',
      ok: false,
      url: request.url,
      redirected: false,
      headers: {},
      body: null
    })
  };
}
