/**
 * Bar customize — edit mode and the item sheet, happy-dom environment:
 *   npx vitest run tests/interfaces/web_terminal/js/bar-customize.test.js
 *
 * The assertions that are the point of this module:
 *
 *   - edit mode is refused in Simple mode. The entry points land later, but a
 *     mode that hides the dock's rearrangement affordances must not offer the
 *     bars' either, whichever surface calls in.
 *
 *   - a refused edit is REFUSED, not silently dropped. `normalize()` would
 *     quietly discard an item over the per-host cap, one the deployment cannot
 *     render, or one the host may not hold. The sheet refuses those edits up
 *     front and names the reason on the tile, and issues zero PUTs.
 *
 *   - every write goes through `saveLayout(next, {edit: true})`. Entering and
 *     leaving edit mode, and rendering the sheet, PUT nothing at all.
 *
 *   - "Layout not saved" renders in the sheet. The module registers
 *     `onSyncNotice()`, which is what retires bar-sync.js's inline pill.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const CUSTOMIZE_PATH =
  '../../../../src/osprey/interfaces/web_terminal/static/js/bar-customize.js';
const SYNC_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-sync.js';
const HOST_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js';

/** The freshly imported module under test. @type {any} */
let customize;
/** The freshly imported sync module. @type {any} */
let sync;
/** @type {any} */
let fetchSpy;

const realFetch = globalThis.fetch;

/**
 * A layout document, as the server serves it.
 * @param {(string | {type: string, options: Record<string, unknown>})[]} header
 * @param {(string | {type: string, options: Record<string, unknown>})[]} status
 * @param {{rev?: number, version?: number, headerVisible?: boolean, statusVisible?: boolean}} [extra]
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
    header_visible: extra.headerVisible ?? true,
    status_visible: extra.statusVisible ?? true,
  };
}

/** A `Response`-alike, which is all the sync layer reads. */
function jsonResponse(/** @type {number} */ status, /** @type {unknown} */ body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

/**
 * A fake `/api/bar-items`. A PUT echoes the body back at the next revision
 * unless `puts` supplies an answer for it.
 * @param {{get?: unknown, puts?: unknown[]}} config
 */
function endpoint({ get = doc([], []), puts = [] } = {}) {
  let index = 0;
  return vi.fn(async (/** @type {string} */ _url, /** @type {any} */ init = {}) => {
    const method = init.method ?? 'GET';
    if (method === 'GET') return jsonResponse(200, get);
    const scripted = puts.length > 0 ? puts[Math.min(index, puts.length - 1)] : null;
    index += 1;
    if (scripted instanceof Error) throw scripted;
    if (scripted) return scripted;
    const sent = JSON.parse(init.body);
    return jsonResponse(200, { ...sent, rev: (sent.rev ?? 0) + 1 });
  });
}

/** Every PUT the spy saw, as parsed bodies. @returns {any[]} */
function putBodies() {
  return fetchSpy.mock.calls
    .filter((/** @type {any[]} */ call) => (call[1]?.method ?? 'GET') === 'PUT')
    .map((/** @type {any[]} */ call) => JSON.parse(call[1].body));
}

/**
 * The SSR DOM, then a fresh module graph on top of it. bar-host hydrates at
 * import time, so the body is seeded before the imports.
 *
 * `context` is what the server stamped on `<html>` as `data-bar-context`;
 * omitting it stamps nothing, which is a page this build did not render and
 * where every gated item reads as unavailable.
 * @param {{fetch?: any, uiMode?: string, context?: Record<string, unknown>}} [options]
 */
async function boot({ fetch = endpoint(), uiMode = 'expert', context } = {}) {
  vi.resetModules();
  document.documentElement.setAttribute('data-ui-mode', uiMode);
  if (context) document.documentElement.setAttribute('data-bar-context', JSON.stringify(context));
  document.body.innerHTML = `
    <header class="header">
      <div class="header-actions" data-bar-host="header"></div>
    </header>
    <footer class="status-bar" data-bar-host="status"></footer>
    <div id="bar-item-pool" hidden></div>
  `;
  fetchSpy = fetch;
  globalThis.fetch = fetchSpy;
  await import(HOST_PATH);
  sync = await import(SYNC_PATH);
  customize = await import(CUSTOMIZE_PATH);
  await settle();
  return customize;
}

/** Let the boot GET and everything it queued settle. */
async function settle() {
  for (let turn = 0; turn < 8; turn += 1) await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

/** The sheet element, or null when edit mode has never been entered. */
function sheet() {
  return document.querySelector('.bar-sheet');
}

/** The tile button for one item type. @param {string} type */
function tile(type) {
  return /** @type {any} */ (document.querySelector(`.bar-tile[data-bar-tile="${type}"]`));
}

/** The reason text a tile carries, or '' when it carries none. @param {string} type */
function tileReason(type) {
  return tile(type)?.querySelector('.bar-tile-reason')?.textContent ?? '';
}

/** A document filling the header to the per-host cap with separators. */
function fullHeader() {
  return doc(Array.from({ length: 20 }, () => 'separator'), []);
}

beforeEach(() => {
  document.documentElement.removeAttribute('data-ui-mode');
});

afterEach(() => {
  customize?.stopBarCustomize?.();
  sync?.stopBarSync?.();
  document.body.innerHTML = '';
  document.documentElement.removeAttribute('data-ui-mode');
  document.documentElement.removeAttribute('data-bar-context');
  globalThis.fetch = realFetch;
  vi.restoreAllMocks();
});

describe('what the sheet may offer is the SERVED deployment context', () => {
  test('the edit context is the stamp, parsed, and nothing else', async () => {
    await boot({
      context: { identityAvailable: true, blueskyAvailable: true, systemHealthAvailable: true },
    });

    expect(customize.editContext()).toEqual({
      identityAvailable: true,
      blueskyAvailable: true,
      systemHealthAvailable: true,
    });
  });

  test('an item the deployment OFFERS is offered, though no shell renders it', async () => {
    // The inference this replaced read availability off the rendered shells, so
    // a deployment that offers the plan queue but has not placed it looked like
    // one that cannot render it — and the sheet then refused a tile the
    // normalizer would have kept. Both must read the same served facts, or the
    // operator's save dies as `readonly` with nothing said.
    await boot({
      fetch: endpoint({ get: doc(['logo'], []) }),
      context: { identityAvailable: true, blueskyAvailable: false, systemHealthAvailable: false },
    });
    customize.enterEditMode();

    expect(document.querySelector('[data-bar-item="identity"]')).toBe(null);
    expect(customize.refusalFor('identity', 'header')).toBe(null);
    expect(tile('identity').disabled).toBe(false);
  });
});

describe('entering and leaving edit mode', () => {
  test('entering marks the page and opens the sheet', async () => {
    await boot();

    expect(customize.enterEditMode()).toBe(true);
    expect(customize.isEditing()).toBe(true);
    expect(document.body.classList.contains('bar-editing')).toBe(true);
    expect(sheet()?.classList.contains('is-open')).toBe(true);
  });

  test('Done leaves edit mode', async () => {
    await boot();
    customize.enterEditMode();

    /** @type {any} */ (document.querySelector('.bar-sheet-done')).click();

    expect(customize.isEditing()).toBe(false);
    expect(document.body.classList.contains('bar-editing')).toBe(false);
    expect(sheet()?.classList.contains('is-open')).toBe(false);
  });

  test('Escape leaves edit mode', async () => {
    await boot();
    customize.enterEditMode();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));

    expect(customize.isEditing()).toBe(false);
    expect(document.body.classList.contains('bar-editing')).toBe(false);
  });

  test('a key that is not Escape leaves edit mode alone', async () => {
    await boot();
    customize.enterEditMode();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));

    expect(customize.isEditing()).toBe(true);
  });

  test('listeners are told both ways', async () => {
    await boot();
    /** @type {boolean[]} */
    const seen = [];
    const off = customize.onEditModeChange((/** @type {boolean} */ on) => seen.push(on));

    customize.enterEditMode();
    customize.exitEditMode();
    off();
    customize.enterEditMode();

    expect(seen).toEqual([true, false]);
  });

  test('the sheet teaches the gesture that removes an item', async () => {
    // Dragging an item out of the bars removes it, with no confirmation. This
    // line is the only place that says so, which is why it is the design of
    // record's sentence and not a shorter one.
    await boot();
    customize.enterEditMode();

    expect(document.querySelector('.bar-sheet-hint')?.textContent).toBe(
      'Drag items into the header or the status bar. Drag an item out to remove it. ' +
        'Click an item for its options.'
    );
  });

  test('teardown takes the sheet with it', async () => {
    await boot();
    customize.enterEditMode();
    expect(sheet()).not.toBe(null);

    customize.stopBarCustomize();

    expect(sheet()).toBe(null);
  });

  test('every item can be removed, the wordmark included', async () => {
    await boot({ fetch: endpoint({ get: doc(['logo', 'clock'], []) }) });

    customize.enterEditMode();
    expect(await customize.removeAt('header', 0)).toBe(true);

    expect(putBodies()[0].header.map((/** @type {any} */ i) => i.type)).toEqual(['clock']);
  });
});

describe('Simple mode refuses edit mode', () => {
  test('entry is refused and nothing is opened', async () => {
    await boot({ uiMode: 'simple' });

    expect(customize.enterEditMode()).toBe(false);
    expect(customize.isEditing()).toBe(false);
    expect(document.body.classList.contains('bar-editing')).toBe(false);
    expect(sheet()).toBe(null);
  });

  test('Expert mode is not refused', async () => {
    await boot({ uiMode: 'expert' });

    expect(customize.enterEditMode()).toBe(true);
  });
});

describe('a tile names why it cannot be added', () => {
  test('any type may be added to either bar', async () => {
    await boot();

    expect(customize.refusalFor('logo', 'status')).toBe(null);
    expect(customize.refusalFor('logo', 'header')).toBe(null);
    expect(customize.refusalFor('search', 'status')).toBe(null);
  });

  test('a type the deployment does not render is refused', async () => {
    await boot();
    customize.enterEditMode();

    expect(customize.refusalFor('identity', 'header')).toBe('Not in this deployment');
    expect(tile('identity').disabled).toBe(true);
    expect(tileReason('identity')).toBe('Not in this deployment');
  });

  test('a renderable type carries no reason and is enabled', async () => {
    await boot();
    customize.enterEditMode();

    expect(tile('clock').disabled).toBe(false);
    expect(tileReason('clock')).toBe('');
  });

  test('a single-node type already in a bar is refused, and its tile dims', async () => {
    // Its body is one server-rendered node; a second shell could only be empty.
    await boot({ fetch: endpoint({ get: doc(['logo', 'docs'], ['clock']) }) });
    customize.enterEditMode();

    expect(customize.refusalFor('docs', 'status')).toBe('Already in the header');
    expect(tile('docs').classList.contains('is-in-bar')).toBe(true);
    expect(tile('docs').title).toBe('Already in the header');
    expect(tileReason('docs')).toBe('');
    tile('docs').click();
    await settle();
    expect(
      fetchSpy.mock.calls.filter((/** @type {any} */ c) => c[1]?.method === 'PUT')
    ).toHaveLength(0);

    // A type that may be placed twice is offered again.
    expect(customize.refusalFor('clock', 'header')).toBe(null);
    expect(tile('clock').classList.contains('is-in-bar')).toBe(false);
  });
});

describe('a tile shows the item', () => {
  test('a JS-built type previews through its own builder', async () => {
    await boot();
    customize.enterEditMode();

    expect(tile('clock').querySelector('.bar-tile-body .bar-clock')).not.toBe(null);
    expect(tile('clock').querySelector('.bar-tile-label')?.textContent).toBe('Clock');
  });

  test('an adopted type previews as a copy of its live node, ids stripped', async () => {
    await boot();
    document.querySelector('[data-bar-host="header"]')?.insertAdjacentHTML(
      'beforeend',
      '<div class="bar-item" data-bar-item="docs"><a class="status-item" id="docs-link" hidden>Docs</a></div>'
    );
    const host = await import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js');
    host.hydrate(document);
    customize.enterEditMode();

    const copy = tile('docs').querySelector('.bar-tile-body a');
    expect(copy?.textContent).toBe('Docs');
    expect(copy?.id).toBe('');
    expect(copy?.hasAttribute('hidden')).toBe(false);
    expect(document.querySelectorAll('#docs-link')).toHaveLength(1);
  });

  test('tiles sit under the catalog headings, in catalog order', async () => {
    await boot();
    customize.enterEditMode();

    const headings = Array.from(document.querySelectorAll('.bar-sheet-group-heading')).map(
      (h) => h.textContent
    );
    expect(headings).toEqual(['Identity', 'Machine', 'Panels', 'System', 'Tools', 'Layout']);
    expect(
      Array.from(document.querySelectorAll('.bar-tile')).map((t) => /** @type {any} */ (t).dataset.barTile)
    ).toHaveLength(13);
  });

  test('a full host refuses the tile by name', async () => {
    await boot({ fetch: endpoint({ get: fullHeader() }) });
    customize.enterEditMode();

    expect(customize.refusalFor('clock', 'header')).toBe('Header is full');
    expect(tile('clock').disabled).toBe(true);
    expect(tileReason('clock')).toBe('Header is full');
  });

  test('clicking a refused tile issues no PUT', async () => {
    await boot({ fetch: endpoint({ get: fullHeader() }) });
    customize.enterEditMode();

    tile('clock').click();
    await settle();

    expect(putBodies()).toEqual([]);
  });

  test('addItem over the cap resolves false and issues no PUT', async () => {
    await boot({ fetch: endpoint({ get: fullHeader() }) });

    await expect(customize.addItem('clock', 'header')).resolves.toBe(false);
    expect(putBodies()).toEqual([]);
  });

  test('a read-only document refuses every tile', async () => {
    // A document naming a type this build cannot render is read-only: it is
    // rendered and never written back.
    await boot({ fetch: endpoint({ get: doc(['logo', 'not-a-type'], []) }) });
    customize.enterEditMode();

    expect(sync.isLayoutReadonly()).toBe(true);
    expect(customize.refusalFor('clock', 'header')).toBe('Layout not editable');
    expect(tile('clock').disabled).toBe(true);

    tile('clock').click();
    await settle();
    expect(putBodies()).toEqual([]);
  });
});

describe('every edit goes through saveLayout', () => {
  test('entering and leaving edit mode PUTs nothing', async () => {
    await boot({ fetch: endpoint({ get: doc(['logo'], ['clock']) }) });

    customize.enterEditMode();
    customize.exitEditMode();
    await settle();

    expect(putBodies()).toEqual([]);
  });

  test('clicking a tile appends the item and PUTs once', async () => {
    await boot({ fetch: endpoint({ get: doc(['logo'], []) }) });
    customize.enterEditMode();

    tile('clock').click();
    await settle();

    const bodies = putBodies();
    expect(bodies).toHaveLength(1);
    expect(bodies[0].header.map((/** @type {any} */ i) => i.type)).toEqual(['logo', 'clock']);
  });

  test('the added item carries its option defaults', async () => {
    await boot({ fetch: endpoint({ get: doc([], []) }) });

    await customize.addItem('clock', 'header');

    expect(putBodies()[0].header[0].options).toEqual({
      zone: 'none',
      format: '24h',
      seconds: false,
    });
  });

  test('the client supplies the revision, never the caller', async () => {
    await boot({ fetch: endpoint({ get: doc(['logo'], [], { rev: 7 }) }) });

    await customize.addItem('clock', 'header');

    expect(putBodies()[0].rev).toBe(7);
  });

  test('the status-bar toggle round-trips through saveLayout', async () => {
    await boot({ fetch: endpoint({ get: doc(['logo'], ['clock']) }) });
    customize.enterEditMode();

    const check = /** @type {any} */ (document.querySelector('.bar-sheet-status-visible'));
    expect(check.checked).toBe(true);
    check.checked = false;
    check.dispatchEvent(new Event('change'));
    await settle();

    const bodies = putBodies();
    expect(bodies).toHaveLength(1);
    expect(bodies[0].status_visible).toBe(false);
    expect(sync.currentLayout()?.status_visible).toBe(false);
  });

  test('the toggle re-renders from the saved document', async () => {
    await boot({ fetch: endpoint({ get: doc([], [], { statusVisible: false }) }) });
    customize.enterEditMode();

    const check = /** @type {any} */ (document.querySelector('.bar-sheet-status-visible'));
    expect(check.checked).toBe(false);

    await customize.setBarVisible('status', true);
    expect(
      /** @type {any} */ (document.querySelector('.bar-sheet-status-visible')).checked
    ).toBe(true);
  });

  test('the header has a toggle of its own, and it round-trips the same way', async () => {
    await boot({ fetch: endpoint({ get: doc(['logo'], ['clock']) }) });
    customize.enterEditMode();

    const check = /** @type {any} */ (document.querySelector('.bar-sheet-header-visible'));
    expect(check.checked).toBe(true);
    check.checked = false;
    check.dispatchEvent(new Event('change'));
    await settle();

    expect(putBodies()[0].header_visible).toBe(false);
    expect(putBodies()[0].status_visible).toBe(true);
    expect(sync.currentLayout()?.header_visible).toBe(false);
    expect(document.documentElement.dataset.headerBar).toBe('hidden');
  });
});

describe('the sheet is the notice surface', () => {
  test('a refused save renders "Layout not saved" in the sheet', async () => {
    await boot({
      fetch: endpoint({
        get: doc(['logo'], []),
        puts: [jsonResponse(422, { error: 'malformed' })],
      }),
    });
    customize.enterEditMode();

    tile('clock').click();
    await settle();

    expect(document.querySelector('.bar-sheet-notice')?.textContent).toBe('Layout not saved');
  });

  test('the notice is a live region', async () => {
    await boot();
    customize.enterEditMode();

    expect(document.querySelector('.bar-sheet-notice')?.getAttribute('role')).toBe('status');
  });

  test('a successful save clears a standing notice', async () => {
    await boot({
      fetch: endpoint({
        get: doc(['logo'], []),
        puts: [jsonResponse(422, { error: 'malformed' }), null],
      }),
    });
    customize.enterEditMode();

    tile('clock').click();
    await settle();
    expect(document.querySelector('.bar-sheet-notice')?.textContent).toBe('Layout not saved');

    tile('docs').click();
    await settle();
    expect(document.querySelector('.bar-sheet-notice')?.textContent).toBe('');
  });
});
