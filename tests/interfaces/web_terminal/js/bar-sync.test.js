/**
 * The bar layout sync layer — the client half of `/api/bar-items`, happy-dom
 * environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/js/bar-sync.test.js
 *
 * Three of these assertions are the reason this module is separate from the
 * host, and all three are named in the proposal:
 *
 *   - the boot GET is NEVER a first-paint dependency. A fetch that never
 *     settles must leave the server-rendered bars exactly as they were.
 *
 *   - a PUT only ever originates from an explicit user edit. Boot, a
 *     visibility re-GET and a normalization that repaired the document all
 *     issue zero PUTs; `saveLayout()` without `{edit: true}` issues zero too.
 *
 *   - FR5: a document this build had to drop content from is READ-ONLY. It is
 *     rendered, and it is never written back — zero PUTs, pinned on a fetch
 *     spy rather than inferred from a flag.
 *
 * The rest pins the conflict ladder (409 → adopt → re-apply → retry once →
 * "Layout not saved"), the visibility re-GET, and the two consequences of a
 * reconcile the driver owns: the overflow ladder re-runs, and the item options
 * the SSR never stamped arrive with the first GET.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const SYNC_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-sync.js';
const HOST_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js';
const OVERFLOW_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-overflow.js';

/** @typedef {import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-layout.js').BarLayout} BarLayout */

/** The freshly imported sync module under test. @type {any} */
let sync;
/** The freshly imported host module, for DOM assertions. @type {any} */
let host;
/** The freshly imported overflow module, for the ladder probe. @type {any} */
let overflow;
/** @type {any} */
let fetchSpy;
/** Undo callbacks for anything a test installed globally. @type {(() => void)[]} */
let cleanups = [];

const realFetch = globalThis.fetch;

/**
 * A layout document, as the server serves it.
 * @param {(string | {type: string, options: Record<string, unknown>})[]} header
 * @param {(string | {type: string, options: Record<string, unknown>})[]} status
 * @param {{rev?: number, version?: number, statusVisible?: boolean}} [extra]
 * @returns {Record<string, unknown>}
 */
function doc(header, status, extra = {}) {
  /** @param {string | {type: string, options: Record<string, unknown>}} entry */
  const item = (entry) => (typeof entry === 'string' ? { type: entry, options: {} } : entry);
  return {
    version: extra.version ?? 1,
    rev: extra.rev ?? 0,
    header: header.map(item),
    status: status.map(item),
    status_visible: extra.statusVisible ?? true,
  };
}

/** A `Response`-alike, which is all this module reads. */
function jsonResponse(/** @type {number} */ status, /** @type {unknown} */ body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

/**
 * A fake `/api/bar-items`. `get` is what every GET answers with (a value, or a
 * function called per request); `puts` are the PUT answers in order, the last
 * repeating. An `Error` anywhere is thrown, which is what a dropped network is.
 * `deletes` are the DELETE answers in the same shape; with none, a DELETE
 * answers with an empty deployment default.
 * @param {{get?: unknown, puts?: unknown[], deletes?: unknown[]}} config
 */
function endpoint({ get = doc([], []), puts = [], deletes = [] } = {}) {
  let index = 0;
  let deleted = 0;
  return vi.fn(async (/** @type {string} */ _url, /** @type {any} */ init = {}) => {
    const method = init.method ?? 'GET';
    if (method === 'GET') {
      const body = typeof get === 'function' ? get() : get;
      if (body instanceof Error) throw body;
      return jsonResponse(200, body);
    }
    if (method === 'DELETE') {
      const answer =
        deletes.length > 0
          ? deletes[Math.min(deleted, deletes.length - 1)]
          : jsonResponse(200, doc([], []));
      deleted += 1;
      if (answer instanceof Error) throw answer;
      return answer;
    }
    const answer = puts.length > 0 ? puts[Math.min(index, puts.length - 1)] : jsonResponse(200, {});
    index += 1;
    if (answer instanceof Error) throw answer;
    return answer;
  });
}

/** Every PUT the spy saw, as parsed bodies. @returns {any[]} */
function putBodies() {
  return fetchSpy.mock.calls
    .filter((/** @type {any[]} */ call) => (call[1]?.method ?? 'GET') === 'PUT')
    .map((/** @type {any[]} */ call) => JSON.parse(call[1].body));
}

/** How many GETs the spy saw. */
function getCount() {
  return fetchSpy.mock.calls.filter(
    (/** @type {any[]} */ call) => (call[1]?.method ?? 'GET') === 'GET'
  ).length;
}

/** Every DELETE the spy saw, as its `RequestInit`. @returns {any[]} */
function deleteCalls() {
  return fetchSpy.mock.calls
    .filter((/** @type {any[]} */ call) => call[1]?.method === 'DELETE')
    .map((/** @type {any[]} */ call) => call[1]);
}

/** Where `root()` stamps what this deployment offers. */
const CONTEXT_ATTR = 'data-bar-context';

/** A deployment that offers every item the catalog gates on a fact. */
const OFFERS_EVERYTHING = {
  identityAvailable: true,
  blueskyAvailable: true,
  statusBarIds: ['ariel-status'],
};

/**
 * The SSR DOM, then a fresh module graph on top of it. bar-host hydrates at
 * import time, so the body must be seeded BEFORE the import — which is also
 * what makes each test's module state its own.
 *
 * `context` is what the server stamped on `<html>`; `null` stamps nothing,
 * which is a page this build did not render.
 * @param {{fetch?: any, headerShells?: string, statusShells?: string,
 *          context?: Record<string, unknown> | null}} [options]
 */
async function boot({
  fetch = endpoint(),
  headerShells = '',
  statusShells = '',
  context = OFFERS_EVERYTHING,
} = {}) {
  vi.resetModules();
  if (context === null) document.documentElement.removeAttribute(CONTEXT_ATTR);
  else document.documentElement.setAttribute(CONTEXT_ATTR, JSON.stringify(context));
  document.body.innerHTML = `
    <header class="header">
      <div class="header-actions" data-bar-host="header">${headerShells}</div>
    </header>
    <footer class="status-bar" data-bar-host="status">${statusShells}</footer>
    <div id="bar-item-pool" hidden></div>
  `;
  fetchSpy = fetch;
  globalThis.fetch = fetchSpy;
  host = await import(HOST_PATH);
  overflow = await import(OVERFLOW_PATH);
  sync = await import(SYNC_PATH);
  return sync;
}

/** Let the boot GET and everything it queued settle. */
async function settle() {
  for (let turn = 0; turn < 6; turn += 1) await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

/** @returns {string[]} the `data-bar-item` types of a host, in DOM order */
function typesIn(/** @type {'header' | 'status'} */ which) {
  const container = host.hostElement(which, document);
  if (!container) return [];
  return Array.from(container.querySelectorAll('.bar-item')).map(
    (/** @type {any} */ el) => el.dataset.barItem ?? ''
  );
}

/** Drive `document.visibilityState` and fire the event the module listens for. */
function setVisibility(/** @type {'visible' | 'hidden'} */ state) {
  Object.defineProperty(document, 'visibilityState', {
    configurable: true,
    get: () => state,
  });
  document.dispatchEvent(new Event('visibilitychange'));
}

const LOGO_SHELL = '<div class="bar-item" data-bar-item="logo"><span>OSPREY</span></div>';

beforeEach(() => {
  cleanups = [];
});

afterEach(() => {
  sync?.stopBarSync?.();
  for (const undo of cleanups) undo();
  cleanups = [];
  document.documentElement.removeAttribute(CONTEXT_ATTR);
  document.body.innerHTML = '';
  globalThis.fetch = realFetch;
  vi.restoreAllMocks();
});

describe('the boot GET reconciles and is never a first-paint dependency', () => {
  test('a fetch that never settles leaves the server-rendered bars standing', async () => {
    const pending = vi.fn(() => new Promise(() => {}));
    await boot({ fetch: pending, headerShells: LOGO_SHELL });

    expect(typesIn('header')).toEqual(['logo']);
    expect(pending).toHaveBeenCalledTimes(1);
    expect(sync.currentLayout()).toBe(null);
  });

  test('the served document is reconciled into the bars', async () => {
    await boot({
      fetch: endpoint({ get: doc(['logo', 'search'], ['connection'], { rev: 3 }) }),
      headerShells: LOGO_SHELL,
    });
    await settle();

    expect(typesIn('header')).toEqual(['logo', 'search']);
    expect(typesIn('status')).toEqual(['connection']);
    expect(sync.currentLayout()?.rev).toBe(3);
  });

  test('the first GET restores the item options the SSR never stamped', async () => {
    // (1.14) `_bar_render_plan()` emits no `data-bar-options`, so a configured
    // UTC clock reaches first paint as a local one. This reconcile is the fix.
    await boot({
      fetch: endpoint({
        get: doc([], [{ type: 'clock', options: { zone: 'utc', seconds: true } }]),
      }),
    });
    await settle();

    const shell = document.querySelector('[data-bar-item="clock"]');
    expect(JSON.parse(/** @type {any} */ (shell).dataset.barOptions)).toEqual({
      zone: 'utc',
      seconds: true,
    });
  });

  test('the overflow ladder re-runs after the reconcile, not before it', async () => {
    /** @type {string[]} */
    const measured = [];
    // The GET is held open so the probe is in place before the reconcile runs;
    // the import itself flushes enough microtasks to settle an open fetch.
    /** @type {() => void} */
    let release = () => {};
    const open = new Promise((resolve) => {
      release = () => resolve(undefined);
    });
    const served = endpoint({ get: doc(['logo'], ['connection']) });
    let restore = () => {};
    await boot({
      fetch: vi.fn(async (/** @type {any} */ url, /** @type {any} */ init) => {
        await open;
        return served(url, init);
      }),
    });
    restore = overflow.mockCrowding((/** @type {any} */ container) => {
      measured.push(container.dataset.barHost);
      return { overflow: 0, width: 800 };
    });
    cleanups.push(() => {
      restore();
      overflow.resetOverflow();
    });

    expect(measured).toEqual([]);
    release();
    await settle();
    expect(measured).toEqual(['header', 'status']);
  });

  test('a failed GET is not fatal: the paint stands and nothing rejects', async () => {
    await boot({
      fetch: endpoint({ get: new Error('offline') }),
      headerShells: LOGO_SHELL,
    });
    await settle();

    expect(typesIn('header')).toEqual(['logo']);
    expect(sync.currentLayout()).toBe(null);
    expect(putBodies()).toEqual([]);
  });
});

describe('a PUT only ever originates from an explicit user edit', () => {
  test('boot and a visibility re-GET issue zero PUTs', async () => {
    await boot({ fetch: endpoint({ get: doc(['logo'], []) }) });
    await settle();
    setVisibility('visible');
    await settle();

    expect(getCount()).toBe(2);
    expect(putBodies()).toEqual([]);
  });

  test('a normalization that repaired the document still issues zero PUTs', async () => {
    // A served document missing `status_visible` is repaired on the way in.
    // Repair is not an edit, so it must not be written back.
    const served = doc(['logo'], []);
    delete served.status_visible;
    await boot({ fetch: endpoint({ get: served }) });
    await settle();

    expect(sync.currentLayout()?.status_visible).toBe(true);
    expect(putBodies()).toEqual([]);
  });

  test('saveLayout without the edit flag issues zero PUTs and rejects', async () => {
    await boot({ fetch: endpoint({ get: doc(['logo'], []) }) });
    await settle();

    await expect(sync.saveLayout(doc(['logo', 'search'], []))).rejects.toMatchObject({
      reason: 'not-an-edit',
    });
    expect(putBodies()).toEqual([]);
  });

  test('an explicit edit sends one PUT carrying the held revision', async () => {
    const saved = doc(['logo', 'search'], [], { rev: 8 });
    await boot({
      fetch: endpoint({ get: doc(['logo'], [], { rev: 7 }), puts: [jsonResponse(200, saved)] }),
    });
    await settle();

    const result = await sync.saveLayout(doc(['logo', 'search'], [], { rev: 7 }), { edit: true });

    const bodies = putBodies();
    expect(bodies).toHaveLength(1);
    expect(bodies[0].rev).toBe(7);
    expect(bodies[0].header.map((/** @type {any} */ i) => i.type)).toEqual(['logo', 'search']);
    expect(result.rev).toBe(8);
    expect(sync.currentLayout()?.rev).toBe(8);
    expect(typesIn('header')).toEqual(['logo', 'search']);
  });
});

describe('read-only documents are rendered and never written back (FR5)', () => {
  test('an unknown schema version leaves the paint standing and issues zero PUTs', async () => {
    await boot({
      fetch: endpoint({ get: doc(['logo'], [], { version: 99 }) }),
      headerShells: LOGO_SHELL,
    });
    await settle();
    setVisibility('visible');
    await settle();

    expect(sync.isLayoutReadonly()).toBe(true);
    // A version we cannot read must not blank the bars.
    expect(typesIn('header')).toEqual(['logo']);
    await expect(
      sync.saveLayout(doc(['logo', 'search'], []), { edit: true })
    ).rejects.toMatchObject({ reason: 'readonly' });
    expect(putBodies()).toEqual([]);
  });

  test('a normalized-away item makes the document read-only', async () => {
    await boot({ fetch: endpoint({ get: doc(['logo', 'not-a-real-item'], []) }) });
    await settle();

    expect(sync.isLayoutReadonly()).toBe(true);
    expect(typesIn('header')).toEqual(['logo']);
    await expect(sync.saveLayout(doc(['logo'], []), { edit: true })).rejects.toMatchObject({
      reason: 'readonly',
    });
    expect(putBodies()).toEqual([]);
  });
});

describe('a reset is a DELETE, and the way out of read-only', () => {
  test('resetLayout without the edit flag issues no DELETE and rejects', async () => {
    await boot({ fetch: endpoint({ get: doc(['logo'], []) }) });
    await settle();

    await expect(sync.resetLayout()).rejects.toMatchObject({ reason: 'not-an-edit' });
    expect(deleteCalls()).toEqual([]);
  });

  test('an explicit reset sends one bodyless DELETE and adopts what comes back', async () => {
    await boot({
      fetch: endpoint({
        get: doc(['logo', 'stopwatch'], [], { rev: 4 }),
        deletes: [jsonResponse(200, doc(['logo'], ['clock'], { rev: 0 }))],
      }),
    });
    await settle();
    expect(typesIn('header')).toEqual(['logo', 'stopwatch']);

    const restored = await sync.resetLayout({ edit: true });

    const calls = deleteCalls();
    expect(calls).toHaveLength(1);
    // No revision, and no document: a reset is unconditional by construction.
    expect(calls[0].body).toBe(undefined);
    expect(putBodies()).toEqual([]);
    expect(restored.rev).toBe(0);
    expect(typesIn('header')).toEqual(['logo']);
    expect(typesIn('status')).toEqual(['clock']);
    expect(sync.currentLayout()?.header.map((/** @type {any} */ i) => i.type)).toEqual(['logo']);
  });

  test('it lifts the read-only latch, so the next edit may be saved', async () => {
    await boot({
      fetch: endpoint({
        get: doc(['logo', 'not-a-real-item'], []),
        deletes: [jsonResponse(200, doc(['logo'], []))],
        puts: [jsonResponse(200, doc(['logo', 'search'], [], { rev: 1 }))],
      }),
    });
    await settle();
    expect(sync.isLayoutReadonly()).toBe(true);

    await sync.resetLayout({ edit: true });

    expect(sync.isLayoutReadonly()).toBe(false);
    await expect(sync.saveLayout(doc(['logo', 'search'], []), { edit: true })).resolves.toBeTruthy();
  });

  test('a deployment default this build cannot read latches again', async () => {
    await boot({
      fetch: endpoint({
        get: doc(['logo', 'not-a-real-item'], []),
        deletes: [jsonResponse(200, doc(['logo', 'still-not-a-real-item'], []))],
      }),
    });
    await settle();

    await sync.resetLayout({ edit: true });

    expect(sync.isLayoutReadonly()).toBe(true);
  });

  test('a refused reset says so in its own words and leaves the layout alone', async () => {
    /** @type {string[]} */
    const notices = [];
    await boot({
      fetch: endpoint({
        get: doc(['logo', 'stopwatch'], []),
        deletes: [jsonResponse(503, { detail: { error: 'store_write_failed' } })],
      }),
    });
    await settle();
    sync.onSyncNotice((/** @type {string} */ text) => notices.push(text));

    await expect(sync.resetLayout({ edit: true })).rejects.toMatchObject({
      reason: 'unavailable',
    });

    // "Layout not saved" would describe this backwards: the arrangement the
    // reset would have discarded is exactly what is still in place.
    expect(notices).toEqual(['Layout not reset']);
    expect(typesIn('header')).toEqual(['logo', 'stopwatch']);
  });

  test('an answer it cannot read is reported as a reset that DID happen', async () => {
    // The removal has already landed at this point; only the answer to it was
    // unreadable. Saying "not reset" would be backwards, and leaving the latch
    // down would let the next edit PUT the discarded arrangement back.
    let reads = 0;
    const unreadable = {
      ok: true,
      status: 200,
      json: async () => {
        throw new Error('not json');
      },
    };
    await boot({
      fetch: endpoint({
        get: () => (reads++ === 0 ? doc(['logo', 'stopwatch'], []) : new Error('offline')),
        deletes: [unreadable],
      }),
    });
    await settle();
    /** @type {string[]} */
    const notices = [];
    sync.onSyncNotice((/** @type {string} */ text) => notices.push(text));

    await expect(sync.resetLayout({ edit: true })).rejects.toMatchObject({ reason: 'invalid' });

    expect(notices).toEqual(['Layout reset. Reload to see it.']);
    expect(sync.isLayoutReadonly()).toBe(true);
    await expect(sync.saveLayout(doc(['logo'], []), { edit: true })).rejects.toMatchObject({
      reason: 'readonly',
    });
    expect(putBodies()).toEqual([]);
  });

  test('a dropped network is a refusal, not a silent reset', async () => {
    /** @type {string[]} */
    const notices = [];
    await boot({
      fetch: endpoint({ get: doc(['logo'], []), deletes: [new Error('offline')] }),
    });
    await settle();
    sync.onSyncNotice((/** @type {string} */ text) => notices.push(text));

    await expect(sync.resetLayout({ edit: true })).rejects.toMatchObject({ reason: 'network' });
    expect(notices).toEqual(['Layout not reset']);
    expect(typesIn('header')).toEqual(['logo']);
  });
});

describe('the deployment context is served, not inferred', () => {
  test('an offered item survives even though the SSR did not place it', async () => {
    // The inference this replaced read availability off the rendered shells,
    // so a deployment that OFFERS identity but does not PLACE it looked like
    // one that cannot render it — and the drop latched the document read-only
    // for the session. The stamp says what the deployment offers.
    await boot({
      fetch: endpoint({ get: doc(['logo', 'identity'], []) }),
      headerShells: LOGO_SHELL,
      context: { identityAvailable: true, blueskyAvailable: false, statusBarIds: [] },
    });
    await settle();

    expect(typesIn('header')).toEqual(['logo', 'identity']);
    expect(sync.isLayoutReadonly()).toBe(false);
  });

  test('an item the deployment does not offer is dropped and latches read-only', async () => {
    await boot({
      fetch: endpoint({ get: doc([], ['bluesky-queue', 'clock']) }),
      context: { identityAvailable: false, blueskyAvailable: false, statusBarIds: [] },
    });
    await settle();

    expect(typesIn('status')).toEqual(['clock']);
    expect(sync.isLayoutReadonly()).toBe(true);
    await expect(sync.saveLayout(doc([], ['clock']), { edit: true })).rejects.toMatchObject({
      reason: 'readonly',
    });
    expect(putBodies()).toEqual([]);
  });

  test('the same deployment stays writable while the document keeps quiet about it', async () => {
    await boot({
      fetch: endpoint({
        get: doc(['logo'], ['clock'], { rev: 2 }),
        puts: [jsonResponse(200, doc(['logo', 'search'], ['clock'], { rev: 3 }))],
      }),
      context: { identityAvailable: false, blueskyAvailable: false, statusBarIds: [] },
    });
    await settle();

    expect(sync.isLayoutReadonly()).toBe(false);
    await sync.saveLayout(doc(['logo', 'search'], ['clock']), { edit: true });
    expect(putBodies()).toHaveLength(1);
  });

  test('panel health follows the stamped dots, not the shipped panel catalog', async () => {
    // The inference read `PANELS`, which ships `ariel-status` on every build,
    // so the item read as available on a deployment serving no such panel.
    await boot({
      fetch: endpoint({ get: doc([], ['panel-health', 'clock']) }),
      context: { identityAvailable: false, blueskyAvailable: false, statusBarIds: [] },
    });
    await settle();

    expect(typesIn('status')).toEqual(['clock']);
    expect(sync.isLayoutReadonly()).toBe(true);
  });

  test('a stamped dot keeps the panel-health item', async () => {
    await boot({
      fetch: endpoint({ get: doc([], ['panel-health', 'clock']) }),
      context: {
        identityAvailable: false,
        blueskyAvailable: false,
        statusBarIds: ['ariel-status'],
      },
    });
    await settle();

    expect(typesIn('status')).toEqual(['panel-health', 'clock']);
    expect(sync.isLayoutReadonly()).toBe(false);
  });

  test('no stamp assumes nothing rather than guessing', async () => {
    // Only a page this build did not render reaches here. Guessing "available"
    // would let the client re-add an item the deployment cannot render; the
    // read-only latch says instead that we could not read the document fully.
    await boot({
      fetch: endpoint({ get: doc(['logo', 'identity'], []) }),
      headerShells: LOGO_SHELL,
      context: null,
    });
    await settle();

    expect(typesIn('header')).toEqual(['logo']);
    expect(sync.isLayoutReadonly()).toBe(true);
  });
});

describe('the conflict ladder', () => {
  /** The 409 body FastAPI serializes from the route's `HTTPException` detail. */
  function conflict(/** @type {Record<string, unknown>} */ layout) {
    return jsonResponse(409, {
      detail: { error: 'rev_conflict', message: 'saved elsewhere', layout },
    });
  }

  test('a 409 adopts the returned document, re-applies the edit and retries once', async () => {
    const theirs = doc(['logo', 'display'], [], { rev: 5 });
    const saved = doc(['logo', 'search'], [], { rev: 6 });
    await boot({
      fetch: endpoint({
        get: doc(['logo'], [], { rev: 4 }),
        puts: [conflict(theirs), jsonResponse(200, saved)],
      }),
    });
    await settle();

    const result = await sync.saveLayout(doc(['logo', 'search'], [], { rev: 4 }), { edit: true });

    const bodies = putBodies();
    expect(bodies).toHaveLength(2);
    expect(bodies[0].rev).toBe(4);
    expect(bodies[1].rev).toBe(5); // the retry carries the revision that won
    expect(bodies[1].header.map((/** @type {any} */ i) => i.type)).toEqual(['logo', 'search']);
    expect(result.rev).toBe(6);
    expect(typesIn('header')).toEqual(['logo', 'search']);
  });

  test('a second 409 stops at two PUTs and surfaces "Layout not saved"', async () => {
    /** @type {string[]} */
    const notices = [];
    await boot({
      fetch: endpoint({
        get: doc(['logo'], [], { rev: 4 }),
        puts: [conflict(doc(['logo'], [], { rev: 5 })), conflict(doc(['logo'], [], { rev: 6 }))],
      }),
    });
    await settle();
    cleanups.push(sync.onSyncNotice((/** @type {string} */ text) => notices.push(text)));

    await expect(
      sync.saveLayout(doc(['logo', 'search'], [], { rev: 4 }), { edit: true })
    ).rejects.toMatchObject({ reason: 'conflict' });

    expect(putBodies()).toHaveLength(2);
    expect(notices).toEqual(['Layout not saved']);
  });

  test('a 409 carrying a read-only document is not retried', async () => {
    await boot({
      fetch: endpoint({
        get: doc(['logo'], [], { rev: 4 }),
        puts: [conflict(doc(['logo'], [], { rev: 5, version: 99 }))],
      }),
    });
    await settle();

    await expect(
      sync.saveLayout(doc(['logo', 'search'], [], { rev: 4 }), { edit: true })
    ).rejects.toMatchObject({ reason: 'readonly' });
    expect(putBodies()).toHaveLength(1);
  });

  test('a 503 store failure surfaces "Layout not saved" without a retry', async () => {
    /** @type {string[]} */
    const notices = [];
    await boot({
      fetch: endpoint({
        get: doc(['logo'], []),
        puts: [jsonResponse(503, { detail: { error: 'store_unavailable' } })],
      }),
    });
    await settle();
    cleanups.push(sync.onSyncNotice((/** @type {string} */ text) => notices.push(text)));

    await expect(sync.saveLayout(doc(['logo'], []), { edit: true })).rejects.toMatchObject({
      reason: 'unavailable',
    });
    expect(putBodies()).toHaveLength(1);
    expect(notices).toEqual(['Layout not saved']);
  });

  test('a dropped network on the PUT surfaces "Layout not saved"', async () => {
    /** @type {string[]} */
    const notices = [];
    await boot({ fetch: endpoint({ get: doc(['logo'], []), puts: [new Error('offline')] }) });
    await settle();
    cleanups.push(sync.onSyncNotice((/** @type {string} */ text) => notices.push(text)));

    await expect(sync.saveLayout(doc(['logo'], []), { edit: true })).rejects.toMatchObject({
      reason: 'network',
    });
    expect(notices).toEqual(['Layout not saved']);
  });

  test('with no sink registered the refusal paints nothing at all', async () => {
    // This module owns no stylesheet, so it renders no notice of its own: the
    // customize sheet registers the sink and paints the message in its own
    // markup. Nothing is appended to the document here — the earlier built-in
    // pill was an inline-styled surface outside the design system.
    await boot({ fetch: endpoint({ get: doc(['logo'], []), puts: [new Error('offline')] }) });
    await settle();
    const before = document.body.childElementCount;

    await expect(sync.saveLayout(doc(['logo'], []), { edit: true })).rejects.toMatchObject({
      reason: 'network',
    });

    expect(document.body.childElementCount).toBe(before);
    expect(document.querySelector('[role="status"]')).toBeNull();
    expect(document.getElementById('bar-sync-notice')).toBeNull();
  });
});

describe('visibility', () => {
  test('becoming visible re-GETs and adopts what another tab saved', async () => {
    let served = doc(['logo'], [], { rev: 1 });
    await boot({ fetch: endpoint({ get: () => served }) });
    await settle();
    expect(typesIn('header')).toEqual(['logo']);

    served = doc(['logo', 'search'], [], { rev: 2 });
    setVisibility('visible');
    await settle();

    expect(typesIn('header')).toEqual(['logo', 'search']);
    expect(sync.currentLayout()?.rev).toBe(2);
  });

  test('going hidden does not GET', async () => {
    await boot({ fetch: endpoint({ get: doc(['logo'], []) }) });
    await settle();
    setVisibility('hidden');
    await settle();

    expect(getCount()).toBe(1);
  });

  test('stopBarSync stops the visibility re-GET', async () => {
    await boot({ fetch: endpoint({ get: doc(['logo'], []) }) });
    await settle();
    sync.stopBarSync();
    setVisibility('visible');
    await settle();

    expect(getCount()).toBe(1);
  });
});
