/**
 * Shared scaffolding for the bar-customize suites.
 *
 * The three suites — edit mode, drag, options/presets — all need the same
 * thing: the SSR shell in `document.body`, a fake `/api/bar-items`, and a fresh
 * module graph on top of both. It lives here rather than in one of the suites
 * because a second copy of `boot()` is a second definition of what the server
 * renders, and the day the SSR gains an element the copies drift.
 *
 * happy-dom lays nothing out: every `getBoundingClientRect()` is zero. A test
 * that depends on WHERE something is says so with `withRect()`, which is also
 * the honest reading of these suites — they pin the state machine, and the
 * pixels are the browser lane's to check.
 */

import { vi } from 'vitest';

const CUSTOMIZE_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-customize.js';
const SYNC_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-sync.js';
const HOST_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js';

/** The fetch spy the current boot installed. @type {any} */
export let fetchSpy = null;

const realFetch = globalThis.fetch;

/**
 * A layout document, as the server serves it.
 * @param {(string | {type: string, options: Record<string, unknown>})[]} header
 * @param {(string | {type: string, options: Record<string, unknown>})[]} status
 * @param {{rev?: number, version?: number, statusVisible?: boolean}} [extra]
 * @returns {Record<string, unknown>}
 */
export function doc(header, status, extra = {}) {
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

/**
 * A `Response`-alike, which is all the sync layer reads.
 * @param {number} status
 * @param {unknown} body
 */
export function jsonResponse(status, body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

/**
 * A fake `/api/bar-items`. A PUT echoes the body back at the next revision
 * unless `puts` supplies an answer for it.
 * A DELETE answers with `reset`, which stands in for the deployment default the
 * route hands back after discarding the operator's arrangement.
 * @param {{get?: unknown, puts?: unknown[], reset?: unknown}} [config]
 */
export function endpoint({ get = doc([], []), puts = [], reset = doc([], []) } = {}) {
  let index = 0;
  return vi.fn(async (/** @type {string} */ _url, /** @type {any} */ init = {}) => {
    const method = init.method ?? 'GET';
    if (method === 'GET') return jsonResponse(200, get);
    if (method === 'DELETE') {
      if (reset instanceof Error) throw reset;
      const scripted = /** @type {any} */ (reset);
      return scripted && typeof scripted.ok === 'boolean' ? scripted : jsonResponse(200, reset);
    }
    const scripted = puts.length > 0 ? puts[Math.min(index, puts.length - 1)] : null;
    index += 1;
    if (scripted instanceof Error) throw scripted;
    if (scripted) return scripted;
    const sent = JSON.parse(init.body);
    return jsonResponse(200, { ...sent, rev: (sent.rev ?? 0) + 1 });
  });
}

/** How many DELETEs the spy saw. @returns {number} */
export function deleteCount() {
  return fetchSpy.mock.calls.filter((/** @type {any[]} */ call) => call[1]?.method === 'DELETE')
    .length;
}

/** Every PUT the spy saw, as parsed bodies. @returns {any[]} */
export function putBodies() {
  return fetchSpy.mock.calls
    .filter((/** @type {any[]} */ call) => (call[1]?.method ?? 'GET') === 'PUT')
    .map((/** @type {any[]} */ call) => JSON.parse(call[1].body));
}

/**
 * The SSR DOM, then a fresh module graph on top of it. bar-host hydrates at
 * import time, so the body is seeded before the imports.
 * `menu: true` adds the display menu as the component leaves it once connected:
 * the card, the View row that renders in both ui modes, and the action row the
 * component projects the page's own chrome into. That row is the projection
 * point the Customize entry mounts itself in.
 *
 * `menu: 'cold'` writes what the SERVER renders instead — the bare tag with its
 * projected children and no card at all, which is the DOM every real page has
 * until `osprey-display-menu.js` is evaluated. Use it to test what happens when
 * that module has not run yet.
 * @param {{fetch?: any, uiMode?: string, statusHidden?: boolean,
 *          menu?: boolean | 'cold'}} [options]
 * @returns {Promise<{customize: any, sync: any}>}
 */
export async function boot({
  fetch = endpoint(),
  uiMode = 'expert',
  statusHidden = false,
  menu = false,
} = {}) {
  vi.resetModules();
  document.documentElement.setAttribute('data-ui-mode', uiMode);
  if (statusHidden) document.documentElement.setAttribute('data-status-bar', 'hidden');
  const COLD_MENU = `<osprey-display-menu id="display-menu">
         <button class="display-menu-settings" id="display-menu-settings" type="button"
                 data-drawer-trigger="settings-drawer">Settings</button>
       </osprey-display-menu>`;
  const CONNECTED_MENU = `<osprey-display-menu id="display-menu">
         <div class="display-menu-card">
           <div class="display-menu-seg display-menu-view" role="group" aria-label="UI mode"></div>
           <div class="display-menu-actions">
             <button class="display-menu-settings" id="display-menu-settings" type="button">
               Settings
             </button>
           </div>
         </div>
       </osprey-display-menu>`;
  const displayMenu = menu === 'cold' ? COLD_MENU : menu ? CONNECTED_MENU : '';
  document.body.innerHTML = `
    <header class="header">
      <div class="header-actions" data-bar-host="header">${displayMenu}</div>
    </header>
    <footer class="status-bar" data-bar-host="status"></footer>
    <div id="bar-item-pool" hidden></div>
  `;
  fetchSpy = fetch;
  globalThis.fetch = fetchSpy;
  await import(HOST_PATH);
  const sync = await import(SYNC_PATH);
  const customize = await import(CUSTOMIZE_PATH);
  await settle();
  return { customize, sync };
}

/** Undo everything `boot()` installed. @param {{customize?: any, sync?: any}} modules */
export function teardown({ customize, sync } = {}) {
  customize?.stopBarCustomize?.();
  sync?.stopBarSync?.();
  document.body.innerHTML = '';
  document.documentElement.removeAttribute('data-ui-mode');
  document.documentElement.removeAttribute('data-status-bar');
  globalThis.fetch = realFetch;
  fetchSpy = null;
  vi.restoreAllMocks();
}

/** Let the boot GET and everything it queued settle. */
export async function settle() {
  for (let turn = 0; turn < 8; turn += 1) await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Give a node a box. happy-dom reports every rect as zero, so anything that
 * asks WHERE a node is gets the same answer for all of them until a test says
 * otherwise.
 * @param {any} node
 * @param {{left: number, right: number, top?: number, bottom?: number}} box
 */
export function withRect(node, { left, right, top = 0, bottom = 20 }) {
  node.getBoundingClientRect = () => ({
    left,
    right,
    top,
    bottom,
    width: right - left,
    height: bottom - top,
    x: left,
    y: top,
  });
  return node;
}

/**
 * Dispatch one pointer event.
 * @param {string} type
 * @param {any} target
 * @param {{x?: number, y?: number, button?: number, pointerId?: number}} [where]
 */
export function pointer(type, target, { x = 0, y = 0, button = 0, pointerId = 1 } = {}) {
  const event = new PointerEvent(type, {
    bubbles: true,
    cancelable: true,
    clientX: x,
    clientY: y,
    button,
    pointerId,
  });
  target.dispatchEvent(event);
  return event;
}

/** The live shell for a type (the first one). @param {string} type */
export function shell(type) {
  return /** @type {any} */ (
    document.querySelector(`[data-bar-host] [data-bar-item="${type}"]`)
  );
}

/** The item types a host currently renders, in order. @param {string} host */
export function rendered(host) {
  return Array.from(
    document.querySelectorAll(`[data-bar-host="${host}"] [data-bar-item]`)
  ).map((node) => /** @type {any} */ (node).dataset.barItem);
}

/** The sheet element, or null when edit mode has never been entered. */
export function sheet() {
  return document.querySelector('.bar-sheet');
}

/** The tile button for one item type. @param {string} type */
export function tile(type) {
  return /** @type {any} */ (document.querySelector(`.bar-tile[data-bar-tile="${type}"]`));
}

/** What the sheet's notice line currently says. */
export function noticeText() {
  return document.querySelector('.bar-sheet-notice')?.textContent ?? '';
}
