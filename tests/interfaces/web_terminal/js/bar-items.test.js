/**
 * Bar item builders and their attach/detach lifecycle, happy-dom environment
 * (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/js/bar-items.test.js
 *
 * The three assertions this file exists for are the connection item's whole
 * contract, and each one is a bug that shipped in the hardcoded status bar or
 * would have shipped the moment the dot became foldable:
 *
 *   - ATTACH SEEDS. The dot paints the CURRENT connection state the instant it
 *     is built, without waiting for a transition. A dot that only listens sits
 *     grey on a perfectly healthy session, because the socket opened long
 *     before the operator dragged the item into the bar and that event is
 *     never repeated.
 *
 *   - DETACH DISPOSES. Folding does not destroy an item — the host PARKS the
 *     node in `#bar-item-pool`, alive. A subscription that survives parking
 *     keeps writing into a body nobody can see for the rest of the page's
 *     life, and nothing anywhere would report it. So the test drives a real
 *     state change after the detach and pins that the parked body did not move.
 *
 *   - RE-ATTACH RE-SEEDS. Coming back out of the pool has to show what is true
 *     now, not the state the item froze at when it folded. This is the half
 *     that fails silently if `data-bar-built` is left stamped: the host would
 *     skip the rebuild, the body would look right, and it would be dead.
 *
 * These run against the REAL api.js and the REAL bar-host.js — no mock of the
 * subscription under test — because the thing being pinned is precisely that
 * three modules agree about a lifetime. `vi.resetModules()` plus a fresh
 * dynamic import per test gives each test untouched module state (bar-host's
 * shell index, bar-items' instance map and api.js's listener array are all
 * module-private and have no reset).
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

// `import.meta.dirname` is a plain string, so it sidesteps happy-dom's
// override of the global URL (which breaks fileURLToPath(new URL(...)) here).
const JS_DIR = join(
  import.meta.dirname,
  '../../../../src/osprey/interfaces/web_terminal/static/js'
);

/** @typedef {import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-layout.js').BarLayout} BarLayout */

const API_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/api.js';
const HOST_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js';
const ITEMS_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-items.js';
const TERMINAL_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/terminal.js';
const PANELS_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/panel-catalog.js';
const STATUS_BAR_PATH =
  '../../../../src/osprey/interfaces/web_terminal/static/js/panel-status-bar.js';

/** @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/api.js')} */
let api;
/** @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js')} */
let host;
/** @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-items.js')} */
let items;
/**
 * The LIVE panel catalog — the same module instance `bar-items.js` reads, so a
 * test can stand in for `panel-manager.js`, which filters this very array in
 * place against `/api/panels` at init.
 * @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/panel-catalog.js')}
 */
let panels;
/** @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/panel-status-bar.js')} */
let statusBar;
/** Every WebSocket api.js has constructed this test. @type {any[]} */
let sockets;
/**
 * What the mocked `terminal.js` reports as the fitted size. There is no way to
 * stand up a real xterm in happy-dom, so the ONE thing the terminal-size item
 * consumes is faked and the export it consumes is pinned separately, against
 * the real module, by the contract test at the foot of this file.
 * @type {{cols: number, rows: number} | null}
 */
let terminalDims;

beforeEach(async () => {
  sockets = [];
  terminalDims = null;
  vi.stubGlobal(
    'WebSocket',
    class {
      /** @param {string} url */
      constructor(url) {
        this.url = url;
        this.readyState = 0;
        sockets.push(this);
      }
      close() {
        this.readyState = 3;
      }
    }
  );
  vi.resetModules();
  vi.doMock(TERMINAL_PATH, () => ({ getTerminalDimensions: () => terminalDims }));
  api = await import(API_PATH);
  host = await import(HOST_PATH);
  panels = await import(PANELS_PATH);
  statusBar = await import(STATUS_BAR_PATH);
  items = await import(ITEMS_PATH);
});

afterEach(() => {
  items.disposeBarItems();
  vi.useRealTimers();
  vi.doUnmock(TERMINAL_PATH);
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

/**
 * The SSR DOM the host hydrates from: both hosts and the hidden pool. Shells
 * are passed as markup so a test can seed the connection item into either bar.
 * @param {string} [headerShells]
 * @param {string} [statusShells]
 */
function seedDom(headerShells = '', statusShells = '') {
  document.body.innerHTML = `
    <header class="header">
      <div class="header-actions" data-bar-host="header">${headerShells}</div>
    </header>
    <footer class="status-bar" data-bar-host="status">${statusShells}</footer>
    <div id="bar-item-pool" hidden></div>
  `;
}

const CONNECTION_SHELL = '<div class="bar-item" data-bar-item="connection"></div>';

/** @typedef {Record<string, string | number | boolean>} ItemOptions */
/** @typedef {string | {type: string, options: ItemOptions}} LayoutEntry */

/**
 * One server-rendered shell, optionally carrying the placed item's options in
 * `data-bar-options` — which is the production first-paint path for an item
 * whose body depends on them.
 * @param {string} type
 * @param {ItemOptions} [options]
 * @returns {string}
 */
function shellMarkup(type, options) {
  const stamped = options ? ` data-bar-options='${JSON.stringify(options)}'` : '';
  return `<div class="bar-item" data-bar-item="${type}"${stamped}></div>`;
}

/**
 * A layout document naming the given items per host. An entry is a bare type,
 * or a `{type, options}` pair where the options matter to the body.
 * @param {LayoutEntry[]} header
 * @param {LayoutEntry[]} status
 * @returns {BarLayout}
 */
function layoutOf(header, status) {
  /** @param {LayoutEntry} entry */
  const item = (entry) => (typeof entry === 'string' ? { type: entry, options: {} } : entry);
  return {
    version: 1,
    rev: 0,
    header: header.map(item),
    status: status.map(item),
    status_visible: true,
  };
}

/**
 * @param {string} selector
 * @returns {HTMLElement}
 */
function el(selector) {
  const node = document.querySelector(selector);
  if (!(node instanceof HTMLElement)) throw new Error(`no element matched ${selector}`);
  return node;
}

/** The connection item's shell, wherever it currently lives. */
function connectionShell() {
  return el('.bar-item[data-bar-item="connection"]');
}

/** The dot inside the connection item's CURRENT body. */
function connectionDot() {
  return el('.bar-item[data-bar-item="connection"] .status-dot');
}

/** Drive api.js's ws state to 'connecting' by opening a socket. */
function openSocket() {
  api.createWebSocket('ws://localhost:5000/ws/terminal');
  return sockets[sockets.length - 1];
}

/** Drive api.js's ws state all the way to 'connected'. */
function connect() {
  openSocket().onopen();
}

/**
 * Seed the status bar with a connection item and hydrate it, which is the
 * production first-paint path: bar-host indexes the server-rendered shell and
 * the registered builder fills it, synchronously, with no reconcile and no
 * network.
 */
function hydrateWithConnectionItem() {
  seedDom('', CONNECTION_SHELL);
  host.hydrate();
}

describe('connection item: attach seeds from the current state', () => {
  test('renders the live state immediately, without waiting for an event', () => {
    connect();
    expect(api.getConnectionState().ws).toBe('connected');

    // The only transition happened BEFORE the item existed. The dot can only
    // be green if the builder read getConnectionState() on attach.
    hydrateWithConnectionItem();

    expect(connectionDot().className).toBe('status-dot live');
  });

  test('renders a disconnected session as the error dot, not as an empty shell', () => {
    hydrateWithConnectionItem();
    expect(api.getConnectionState().ws).toBe('disconnected');
    expect(connectionDot().className).toBe('status-dot error');
  });

  test('a connecting session is the bare dot — neither healthy nor failed', () => {
    openSocket();
    hydrateWithConnectionItem();
    expect(connectionDot().className).toBe('status-dot');
  });

  test('the body is a passive readout, not a button, and carries the state for AT', () => {
    connect();
    hydrateWithConnectionItem();

    const body = el('.bar-item[data-bar-item="connection"] .status-item');
    expect(body.querySelector('button')).toBeNull();
    expect(body.getAttribute('role')).toBe('status');
    expect(body.getAttribute('aria-label')).toBe('WebSocket connected');
    expect(body.title).toBe('WebSocket: connected');
    expect(body.textContent).toBe('WS');
  });

  test('subsequent transitions reach the attached dot', () => {
    hydrateWithConnectionItem();
    expect(connectionDot().className).toBe('status-dot error');

    connect();
    expect(connectionDot().className).toBe('status-dot live');
  });

  test('the item renders in the header at header density too', () => {
    connect();
    seedDom(CONNECTION_SHELL, '');
    host.hydrate();

    expect(connectionShell().dataset.barDensity).toBe('comfortable');
    expect(connectionDot().className).toBe('status-dot live');
  });
});

describe('connection item: detach disposes the subscription', () => {
  test('a state change after detach does not touch the parked body', () => {
    connect();
    hydrateWithConnectionItem();
    const parkedDot = connectionDot();
    expect(parkedDot.className).toBe('status-dot live');

    host.reconcile(layoutOf([], []));
    items.syncBarItems();
    expect(host.isLive(connectionShell())).toBe(false);

    // A real transition, through the real api.js: the socket drops back to
    // 'connecting'. A leaked listener would repaint the pooled dot here.
    openSocket();
    expect(api.getConnectionState().ws).toBe('connecting');
    expect(parkedDot.className).toBe('status-dot live');
  });

  test('detach clears the build stamp so the next placement rebuilds', () => {
    connect();
    hydrateWithConnectionItem();
    expect(connectionShell().dataset.barBuilt).toBeTruthy();

    host.reconcile(layoutOf([], []));
    items.syncBarItems();

    expect(connectionShell().dataset.barBuilt).toBeUndefined();
  });

  test('syncBarItems is idempotent — a second pass disposes nothing twice', () => {
    connect();
    hydrateWithConnectionItem();
    host.reconcile(layoutOf([], []));

    items.syncBarItems();
    expect(() => items.syncBarItems()).not.toThrow();

    openSocket();
    expect(connectionDot().className).toBe('status-dot live');
  });

  test('parking disposes on its own, without a caller running the pass', async () => {
    connect();
    hydrateWithConnectionItem();
    const parkedDot = connectionDot();

    host.reconcile(layoutOf([], []));
    // No syncBarItems() call: the pool observer is the production path, for
    // the reconcile driver that has not been written yet.
    await Promise.resolve();
    await new Promise((resolve) => setTimeout(resolve, 0));

    openSocket();
    expect(parkedDot.className).toBe('status-dot live');
    expect(connectionShell().dataset.barBuilt).toBeUndefined();
  });

  test('a body removed from the document entirely is disposed as well', () => {
    connect();
    hydrateWithConnectionItem();
    const orphanDot = connectionDot();

    document.body.innerHTML = '';
    items.syncBarItems();

    openSocket();
    expect(orphanDot.className).toBe('status-dot live');
  });
});

describe('connection item: re-attach re-seeds', () => {
  test('coming back out of the pool shows the state that is true NOW', () => {
    connect();
    hydrateWithConnectionItem();
    const staleDot = connectionDot();

    host.reconcile(layoutOf([], []));
    items.syncBarItems();

    // The session changed while the item was folded away.
    const socket = openSocket();
    expect(api.getConnectionState().ws).toBe('connecting');

    host.reconcile(layoutOf([], ['connection']));

    const freshDot = connectionDot();
    expect(freshDot).not.toBe(staleDot);
    expect(freshDot.className).toBe('status-dot');
    expect(host.isLive(connectionShell())).toBe(true);

    // And the fresh subscription is a live one, not just a fresh paint.
    socket.onopen();
    expect(freshDot.className).toBe('status-dot live');
  });

  test('the disposed body stays frozen — the old listener is gone, not doubled', () => {
    connect();
    hydrateWithConnectionItem();
    const staleDot = connectionDot();

    host.reconcile(layoutOf([], []));
    items.syncBarItems();
    host.reconcile(layoutOf([], ['connection']));

    openSocket();
    expect(connectionDot().className).toBe('status-dot');
    expect(staleDot.className).toBe('status-dot live');
  });

  test('a header-to-status move rebuilds once — one live body, no doubled listener', () => {
    connect();
    seedDom(CONNECTION_SHELL, '');
    host.hydrate();
    const headerDot = connectionDot();

    // The move crosses densities, so the host rebuilds the body. A rebuild is
    // a detach followed by an attach: the new body must be seeded and live,
    // and the old one must be off the air rather than a second subscriber.
    host.reconcile(layoutOf([], ['connection']));
    items.syncBarItems();

    const statusDot = connectionDot();
    expect(connectionShell().parentElement?.classList.contains('status-bar')).toBe(true);
    expect(connectionShell().dataset.barDensity).toBe('compact');
    expect(statusDot).not.toBe(headerDot);
    expect(statusDot.className).toBe('status-dot live');

    openSocket();
    expect(statusDot.className).toBe('status-dot');
    expect(headerDot.className).toBe('status-dot live');
  });
});

describe('api.js: the disposer-returning subscription this item is built on', () => {
  test('onConnectionStateChange hands back its own unsubscribe', () => {
    const listener = vi.fn();
    const dispose = api.onConnectionStateChange(listener);
    expect(typeof dispose).toBe('function');

    openSocket();
    expect(listener).toHaveBeenCalledTimes(1);

    dispose();
    openSocket();
    expect(listener).toHaveBeenCalledTimes(1);
  });

  test('disposing twice is a no-op and does not evict another listener', () => {
    const first = vi.fn();
    const second = vi.fn();
    const disposeFirst = api.onConnectionStateChange(first);
    api.onConnectionStateChange(second);

    disposeFirst();
    disposeFirst();
    openSocket();

    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
  });

  test('a listener that disposes itself mid-notify does not skip the next one', () => {
    const after = vi.fn();
    /** @type {() => void} */
    let disposeSelf = () => {};
    disposeSelf = api.onConnectionStateChange(() => disposeSelf());
    api.onConnectionStateChange(after);

    openSocket();

    // Splicing the live array inside the `for` would have stepped over
    // `after` — which is exactly what a detaching bar item does.
    expect(after).toHaveBeenCalledTimes(1);
  });

  test('a caller that ignores the return value still receives notifications', () => {
    const listener = vi.fn();
    api.onConnectionStateChange(listener);

    connect();

    expect(listener).toHaveBeenCalledWith({ ws: 'connected', sse: 'disconnected' });
  });

  test('getConnectionState reports the current state without a transition', () => {
    expect(api.getConnectionState()).toEqual({ ws: 'disconnected', sse: 'disconnected' });
    connect();
    expect(api.getConnectionState()).toEqual({ ws: 'connected', sse: 'disconnected' });
  });
});

/* ============================================================================
 * The three interval-owning items. What the connection item proved for a
 * LISTENER, these prove for a TIMER, which is the leak that actually shipped:
 * `initStatusBar()` armed two bare `setInterval`s at boot, kept no handle to
 * either, and wrote into ids that the clean default layout does not even
 * render — so they ran forever against null for the life of every page.
 * ========================================================================= */

/** The instant every timed test is frozen at. 14:32:07 UTC, so every field differs. */
const AT = new Date('2026-09-01T14:32:07Z');

/**
 * Freeze the clock AND take over the timers. Called before `hydrate()` so the
 * item's interval is registered against the fake timer, which is what lets a
 * test assert on `vi.getTimerCount()` rather than on elapsed wall time.
 * @param {Date} at
 */
function freezeAt(at) {
  vi.useFakeTimers();
  vi.setSystemTime(at);
}

/**
 * What the local-time clock must read at `at` — computed from the same Date
 * the item sees, because the suite's zone is whatever the machine's is.
 * @param {Date} at
 * @param {boolean} seconds
 * @returns {string}
 */
function localTimeText(at, seconds) {
  const pad = (/** @type {number} */ n) => String(n).padStart(2, '0');
  const hm = `${pad(at.getHours())}:${pad(at.getMinutes())}`;
  return seconds ? `${hm}:${pad(at.getSeconds())}` : hm;
}

/**
 * @param {string} type
 * @returns {HTMLElement}
 */
function shellOf(type) {
  return el(`.bar-item[data-bar-item="${type}"]`);
}

/**
 * A node inside one item's CURRENT body.
 * @param {string} type
 * @param {string} selector
 * @returns {HTMLElement}
 */
function partOf(type, selector) {
  return el(`.bar-item[data-bar-item="${type}"] ${selector}`);
}

/** The clock's time field. */
function clockTime() {
  return partOf('clock', '.bar-clock-time');
}

/** The clock's zone suffix, or null when the density/option pair drops it. */
function clockZoneLabel() {
  return document.querySelector('.bar-item[data-bar-item="clock"] .bar-clock-zone');
}

/** The terminal-size value field. */
function termValue() {
  return partOf('terminal-size', '.bar-terminal-size-value');
}

/** The stopwatch's button. @returns {HTMLButtonElement} */
function stopwatchChip() {
  const chip = partOf('stopwatch', 'button.bar-stopwatch');
  return /** @type {HTMLButtonElement} */ (chip);
}

/** The stopwatch's elapsed reading. */
function stopwatchTime() {
  return partOf('stopwatch', '.bar-stopwatch-time').textContent;
}

describe('clock item: renders per the option spec', () => {
  test('the default is the local wall clock, to the minute', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock'));
    host.hydrate();

    expect(clockTime().textContent).toBe(localTimeText(AT, false));
  });

  test('the seconds option adds the seconds field', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock', { seconds: true }));
    host.hydrate();

    expect(clockTime().textContent).toBe(localTimeText(AT, true));
  });

  test('zone utc reads UTC rather than the browser zone', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock', { zone: 'utc' }));
    host.hydrate();

    expect(clockTime().textContent).toBe('14:32');
    expect(clockZoneLabel()?.textContent).toBe('UTC');
  });

  test('zone utc with seconds carries the whole UTC field', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock', { zone: 'utc', seconds: true }));
    host.hydrate();

    expect(clockTime().textContent).toBe('14:32:07');
  });

  test('zone both shows local beside UTC, and marks which half is which', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock', { zone: 'both' }));
    host.hydrate();

    expect(clockTime().textContent).toBe(`${localTimeText(AT, false)} · 14:32`);
    expect(clockZoneLabel()?.textContent).toBe('UTC');
  });

  test('an unknown zone falls back to local rather than rendering nothing', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock', { zone: 'mars' }));
    host.hydrate();

    expect(clockTime().textContent).toBe(localTimeText(AT, false));
  });

  test('the local zone suffix is a comfortable-density affordance', () => {
    freezeAt(AT);
    seedDom(shellMarkup('clock'), '');
    host.hydrate();

    // The header has room to say WHICH clock this is; the 20px status bar does
    // not, and an unlabelled local clock is simply "the time".
    expect(shellOf('clock').dataset.barDensity).toBe('comfortable');
    expect(clockZoneLabel()?.textContent).toBeTruthy();

    host.reconcile(layoutOf([], ['clock']));
    expect(shellOf('clock').dataset.barDensity).toBe('compact');
    expect(clockZoneLabel()).toBeNull();
  });

  test('a UTC clock keeps its label at compact density — dropping it would lie', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock', { zone: 'utc' }));
    host.hydrate();

    expect(shellOf('clock').dataset.barDensity).toBe('compact');
    expect(clockZoneLabel()?.textContent).toBe('UTC');
  });

  test('the readout is a timer region, and its value is its accessible name', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock', { zone: 'utc' }));
    host.hydrate();

    const body = partOf('clock', '.bar-clock');
    // role="timer" is implicitly aria-live="off". role="status" would make a
    // screen reader announce the time once a second, forever.
    expect(body.getAttribute('role')).toBe('timer');
    expect(body.getAttribute('aria-label')).toBeNull();
    expect(body.title).toBe('UTC');
    expect(body.textContent).toBe('14:32UTC');
  });

  test('it repaints itself as the minute turns', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock', { zone: 'utc', seconds: true }));
    host.hydrate();
    expect(clockTime().textContent).toBe('14:32:07');

    vi.advanceTimersByTime(60_000);

    expect(clockTime().textContent).toBe('14:33:07');
  });
});

describe('clock item: the interval is attach-scoped', () => {
  test('not one timer callback fires after the item is parked', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock', { zone: 'utc', seconds: true }));
    host.hydrate();
    const parked = clockTime();
    expect(vi.getTimerCount()).toBe(1);

    host.reconcile(layoutOf([], []));
    items.syncBarItems();

    // The whole bug in one assertion: a folded clock that kept its interval
    // would repaint a body in the hidden pool for the rest of the page's life.
    expect(vi.getTimerCount()).toBe(0);
    vi.advanceTimersByTime(60 * 60 * 1000);
    expect(parked.textContent).toBe('14:32:07');
  });

  test('coming back out of the pool ticks again, on a fresh body', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock', { zone: 'utc', seconds: true }));
    host.hydrate();
    const stale = clockTime();

    host.reconcile(layoutOf([], []));
    items.syncBarItems();
    vi.advanceTimersByTime(30 * 60 * 1000);
    host.reconcile(layoutOf([], [{ type: 'clock', options: { zone: 'utc', seconds: true } }]));

    const fresh = clockTime();
    expect(fresh).not.toBe(stale);
    expect(fresh.textContent).toBe('15:02:07');

    vi.advanceTimersByTime(1000);
    expect(fresh.textContent).toBe('15:02:08');
    expect(stale.textContent).toBe('14:32:07');
    expect(vi.getTimerCount()).toBe(1);
  });

  test('a rebuild leaves one ticker, not two', () => {
    freezeAt(AT);
    seedDom(shellMarkup('clock', { zone: 'utc', seconds: true }), '');
    host.hydrate();

    // A header-to-status move crosses densities, so the host rebuilds: the
    // previous instance must be disposed before the new one starts.
    host.reconcile(layoutOf([], [{ type: 'clock', options: { zone: 'utc', seconds: true } }]));
    items.syncBarItems();

    expect(vi.getTimerCount()).toBe(1);
  });
});

describe('terminal-size item', () => {
  test('renders the fitted columns by rows', () => {
    terminalDims = { cols: 120, rows: 40 };
    seedDom('', shellMarkup('terminal-size'));
    host.hydrate();

    expect(termValue().textContent).toBe('120×40');
    expect(partOf('terminal-size', '.bar-terminal-size').title).toBe(
      'Terminal size: 120 columns × 40 rows'
    );
  });

  test('an em dash before the terminal is up, never a blank body', () => {
    terminalDims = null;
    seedDom('', shellMarkup('terminal-size'));
    host.hydrate();

    expect(termValue().textContent).toBe('—');
    expect(partOf('terminal-size', '.bar-terminal-size').title).toBe('Terminal size: not ready');
  });

  test('it follows a re-fit without being told', () => {
    terminalDims = { cols: 80, rows: 24 };
    freezeAt(AT);
    seedDom('', shellMarkup('terminal-size'));
    host.hydrate();
    expect(termValue().textContent).toBe('80×24');

    terminalDims = { cols: 132, rows: 43 };
    vi.advanceTimersByTime(500);

    expect(termValue().textContent).toBe('132×43');
  });

  test('the poll stops on detach', () => {
    terminalDims = { cols: 80, rows: 24 };
    freezeAt(AT);
    seedDom('', shellMarkup('terminal-size'));
    host.hydrate();
    const parked = termValue();
    expect(vi.getTimerCount()).toBe(1);

    host.reconcile(layoutOf([], []));
    items.syncBarItems();

    expect(vi.getTimerCount()).toBe(0);
    terminalDims = { cols: 132, rows: 43 };
    vi.advanceTimersByTime(10 * 60 * 1000);
    expect(parked.textContent).toBe('80×24');
  });

  test('the standing label is a comfortable-density affordance', () => {
    terminalDims = { cols: 80, rows: 24 };
    seedDom(shellMarkup('terminal-size'), '');
    host.hydrate();
    expect(partOf('terminal-size', '.bar-terminal-size-label').textContent).toBe('Term');

    host.reconcile(layoutOf([], ['terminal-size']));

    expect(document.querySelector('.bar-terminal-size-label')).toBeNull();
    expect(termValue().textContent).toBe('80×24');
  });

  test('the readout is not a live region — a drag would announce every frame', () => {
    terminalDims = { cols: 80, rows: 24 };
    seedDom('', shellMarkup('terminal-size'));
    host.hydrate();

    const body = partOf('terminal-size', '.bar-terminal-size');
    expect(body.getAttribute('role')).toBeNull();
    expect(body.getAttribute('aria-live')).toBeNull();
  });

  test('the real terminal.js still exposes the getter this item consumes', async () => {
    /** @type {any} */
    const actual = await vi.importActual(TERMINAL_PATH);

    // The item is tested against a mock, so this is the one assertion holding
    // the seam itself: `getTerminalDimensions` exists, and it answers null
    // rather than throwing when no terminal has been initialised. Task 1.13
    // hangs a window seam off the same getter for contact_sheet.py.
    expect(typeof actual.getTerminalDimensions).toBe('function');
    expect(actual.getTerminalDimensions()).toBeNull();
  });
});

describe('stopwatch item', () => {
  test('starts stopped at zero, and runs no timer until it is started', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('stopwatch'));
    host.hydrate();

    expect(stopwatchTime()).toBe('00:00');
    expect(stopwatchChip().getAttribute('aria-pressed')).toBe('false');
    expect(stopwatchChip().title).toBe('Click to start');
    // A stopped stopwatch has nothing to repaint.
    expect(vi.getTimerCount()).toBe(0);
  });

  test('clicking starts it and the reading follows the wall clock', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('stopwatch'));
    host.hydrate();

    stopwatchChip().click();
    expect(stopwatchChip().getAttribute('aria-pressed')).toBe('true');
    expect(stopwatchChip().title).toBe('Running — click to stop');
    expect(vi.getTimerCount()).toBe(1);

    vi.advanceTimersByTime(65_000);
    expect(stopwatchTime()).toBe('01:05');
    expect(stopwatchChip().getAttribute('aria-label')).toBe('Stopwatch 01:05, running');
  });

  test('clicking again stops it, and the reading holds', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('stopwatch'));
    host.hydrate();

    stopwatchChip().click();
    vi.advanceTimersByTime(65_000);
    stopwatchChip().click();

    expect(vi.getTimerCount()).toBe(0);
    expect(stopwatchChip().title).toBe('Paused — click to resume, right-click to reset');

    vi.advanceTimersByTime(10 * 60 * 1000);
    expect(stopwatchTime()).toBe('01:05');
  });

  test('resuming adds to the reading rather than restarting it', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('stopwatch'));
    host.hydrate();

    stopwatchChip().click();
    vi.advanceTimersByTime(65_000);
    stopwatchChip().click();
    vi.advanceTimersByTime(60_000);
    stopwatchChip().click();
    vi.advanceTimersByTime(5_000);

    expect(stopwatchTime()).toBe('01:10');
  });

  test('hours appear only once there are hours', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('stopwatch'));
    host.hydrate();

    stopwatchChip().click();
    vi.advanceTimersByTime(3600_000 + 65_000);

    expect(stopwatchTime()).toBe('1:01:05');
  });

  test('right-click resets it to zero and stops the ticker', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('stopwatch'));
    host.hydrate();

    stopwatchChip().click();
    vi.advanceTimersByTime(65_000);
    stopwatchChip().dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));

    expect(stopwatchTime()).toBe('00:00');
    expect(stopwatchChip().getAttribute('aria-pressed')).toBe('false');
    expect(vi.getTimerCount()).toBe(0);
  });

  test('the elapsed time survives a pool round trip', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('stopwatch'));
    host.hydrate();
    stopwatchChip().click();
    vi.advanceTimersByTime(65_000);
    expect(stopwatchTime()).toBe('01:05');

    // Folded away: the body is parked in the pool and its ticker MUST stop...
    host.reconcile(layoutOf([], []));
    items.syncBarItems();
    expect(vi.getTimerCount()).toBe(0);

    // ...but the measurement is not the body's. Ten more seconds pass with the
    // item nowhere on screen, and they count, because elapsed is derived from
    // the wall clock rather than from the ticks that were not delivered.
    vi.advanceTimersByTime(10_000);
    host.reconcile(layoutOf([], ['stopwatch']));

    expect(stopwatchTime()).toBe('01:15');
    expect(stopwatchChip().getAttribute('aria-pressed')).toBe('true');
    expect(vi.getTimerCount()).toBe(1);
  });

  test('a stopped reading survives a pool round trip too', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('stopwatch'));
    host.hydrate();
    stopwatchChip().click();
    vi.advanceTimersByTime(65_000);
    stopwatchChip().click();

    host.reconcile(layoutOf([], []));
    items.syncBarItems();
    vi.advanceTimersByTime(10 * 60 * 1000);
    host.reconcile(layoutOf([], ['stopwatch']));

    expect(stopwatchTime()).toBe('01:05');
    expect(vi.getTimerCount()).toBe(0);
  });

  test('the reading survives a header-to-status move', () => {
    freezeAt(AT);
    seedDom(shellMarkup('stopwatch'), '');
    host.hydrate();
    stopwatchChip().click();
    vi.advanceTimersByTime(65_000);
    const headerChip = stopwatchChip();

    // The move crosses densities, so the body is rebuilt from scratch — and
    // the reading is keyed on the layout key, which the move does not change.
    host.reconcile(layoutOf([], ['stopwatch']));
    items.syncBarItems();

    expect(stopwatchChip()).not.toBe(headerChip);
    expect(shellOf('stopwatch').dataset.barDensity).toBe('compact');
    expect(stopwatchTime()).toBe('01:05');
    expect(vi.getTimerCount()).toBe(1);
  });
});

/* ---- panel health ---- */

/** Let the `hidden` MutationObserver callbacks bar-host queued actually run. */
function flush() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Every panel the LIVE catalog says owns a status-bar dot. Derived, never
 * retyped: retiring a panel's `statusBarId` must move this suite's expectation
 * with it rather than turning one of these tests red for the wrong reason.
 */
function declaringPanels() {
  return panels.PANELS.filter((panel) => panel.statusBarId);
}

/**
 * Stand in for `panel-manager.js`, which filters `PANELS` IN PLACE against
 * `/api/panels` before anything renders. A panel this deployment does not
 * serve is not flagged — it is gone from the array.
 * @param {string[]} panelIds - the ids to drop
 */
function disablePanels(panelIds) {
  const kept = panels.PANELS.filter((panel) => !panelIds.includes(panel.id));
  panels.PANELS.splice(0, panels.PANELS.length, ...kept);
}

/** Every panel-health shell in the document, in host order. */
function healthShells() {
  return /** @type {HTMLElement[]} */ (
    Array.from(document.querySelectorAll('.bar-item[data-bar-item="panel-health"]'))
  );
}

/**
 * The dots one shell renders, in document order.
 * @param {HTMLElement} [shell]
 * @returns {HTMLElement[]}
 */
function dotsIn(shell = healthShells()[0]) {
  return /** @type {HTMLElement[]} */ (Array.from(shell.querySelectorAll('.status-item')));
}

/** @param {HTMLElement} [shell] */
function dotIds(shell) {
  return dotsIn(shell).map((dot) => dot.id);
}

/**
 * One health poll settling, driven through the REAL `panel-status-bar.js` —
 * the module that owns this item's only writes, and the whole point of the
 * `hidden`-attribute convention the two of them share.
 * @param {string} panelId
 * @param {boolean} healthy
 */
function settleHealth(panelId, healthy) {
  const panel = panels.PANELS.find((entry) => entry.id === panelId);
  if (!panel) throw new Error(`no panel ${panelId} in the catalog`);
  statusBar.updateStatusBar(panel, { url: `/panel/${panelId}`, healthy });
}

/**
 * @param {string} id
 * @returns {HTMLElement}
 */
function dotById(id) {
  const node = document.getElementById(id);
  if (!node) throw new Error(`no dot #${id}`);
  return node;
}

/** @param {HTMLElement} item */
function dotClass(item) {
  const dot = item.querySelector('.status-dot');
  if (!dot) throw new Error(`no .status-dot inside #${item.id}`);
  return dot.className;
}

describe('panel-health item: a dot per enabled panel that declares one', () => {
  test('renders one dot per declaring panel, in catalog order, and none for the rest', () => {
    seedDom('', shellMarkup('panel-health'));
    host.hydrate();

    const declaring = declaringPanels();
    expect(declaring.length).toBeGreaterThan(0);
    expect(dotIds()).toEqual(declaring.map((panel) => panel.statusBarId));
    expect(dotsIn().map((dot) => dot.textContent)).toEqual(declaring.map((panel) => panel.label));
    // A panel that declares no id contributes nothing at all — not a dot with
    // no id, and not an unlabelled one.
    expect(dotsIn().length).toBe(declaring.length);
  });

  test('a panel this deployment does not serve has no dot at all, not a dark one', () => {
    const [dropped, ...kept] = declaringPanels();
    disablePanels([dropped.id]);

    seedDom('', shellMarkup('panel-health'));
    host.hydrate();

    expect(document.getElementById(/** @type {string} */ (dropped.statusBarId))).toBeNull();
    expect(dotIds()).toEqual(kept.map((panel) => panel.statusBarId));
  });

  test('a panel registered at runtime cannot mint a dot — the id set is closed', () => {
    const expected = declaringPanels().map((panel) => panel.statusBarId);
    panels.PANELS.push({
      id: 'facility',
      label: 'FACILITY',
      configEndpoint: '/api/facility-server',
      statusBarId: 'facility-status',
    });

    seedDom('', shellMarkup('panel-health'));
    host.hydrate();

    expect(document.getElementById('facility-status')).toBeNull();
    expect(dotIds()).toEqual(expected);
  });

  test('every dot ships hidden, and the shell collapses until one is revealed', async () => {
    seedDom('', shellMarkup('panel-health'));
    host.hydrate();
    await flush();

    expect(dotsIn().every((dot) => dot.hidden)).toBe(true);
    expect(dotsIn().map((dot) => dotClass(dot))).toEqual(dotsIn().map(() => 'status-dot'));
    // bar-host mirrors the body's `hidden` onto the shell: a bar with nothing
    // to report spends no gap on this item.
    expect(healthShells()[0].hidden).toBe(true);
  });

  test('a deployment with no declaring panel renders a hidden body, not an empty shell', async () => {
    disablePanels(declaringPanels().map((panel) => panel.id));

    seedDom('', shellMarkup('panel-health'));
    host.hydrate();
    await flush();

    expect(dotsIn()).toEqual([]);
    expect(healthShells()[0].hidden).toBe(true);
  });

  test('a second panel-health shell does not mint a second copy of the ids', async () => {
    seedDom(shellMarkup('panel-health'), shellMarkup('panel-health'));
    host.hydrate();
    await flush();

    const [first, second] = healthShells();
    for (const panel of declaringPanels()) {
      expect(document.querySelectorAll(`[id="${panel.statusBarId}"]`).length).toBe(1);
    }
    expect(dotIds(first)).toEqual(declaringPanels().map((panel) => panel.statusBarId));
    expect(dotIds(second)).toEqual([]);
    expect(second.hidden).toBe(true);
  });
});

describe('panel-health item: panel-status-bar.js drives the dots it renders', () => {
  test('a healthy poll reveals the dot and brings its shell back', async () => {
    seedDom('', shellMarkup('panel-health'));
    host.hydrate();
    const panel = declaringPanels()[0];

    settleHealth(panel.id, true);
    await flush();

    const item = dotById(/** @type {string} */ (panel.statusBarId));
    expect(item.hidden).toBe(false);
    expect(dotClass(item)).toBe('status-dot live');
    expect(healthShells()[0].hidden).toBe(false);
  });

  test('an unhealthy panel is the error dot, still shown', async () => {
    seedDom('', shellMarkup('panel-health'));
    host.hydrate();
    const panel = declaringPanels()[0];

    settleHealth(panel.id, false);
    await flush();

    expect(dotClass(dotById(/** @type {string} */ (panel.statusBarId)))).toBe('status-dot error');
    expect(healthShells()[0].hidden).toBe(false);
  });

  test('a host move does not blank a dot whose health has already settled', async () => {
    seedDom(shellMarkup('panel-health'), '');
    host.hydrate();
    const panel = declaringPanels()[0];
    const id = /** @type {string} */ (panel.statusBarId);
    settleHealth(panel.id, true);
    const before = dotById(id);

    // Header -> status crosses densities, so the body is rebuilt from scratch.
    // panel-health.js polls a healthy panel once every 10 s and exposes no
    // state to read back, so a dot that started over would sit dark for up to
    // ten seconds on a panel that never stopped being healthy.
    host.reconcile(layoutOf([], ['panel-health']));
    items.syncBarItems();
    await flush();

    const after = dotById(id);
    expect(after).not.toBe(before);
    expect(after.hidden).toBe(false);
    expect(dotClass(after)).toBe('status-dot live');
    expect(healthShells()[0].hidden).toBe(false);
  });

  test('a parked dot is still reachable by id — the pool never lets an id go dark', async () => {
    seedDom('', shellMarkup('panel-health'));
    host.hydrate();
    const panel = declaringPanels()[0];
    const id = /** @type {string} */ (panel.statusBarId);

    host.reconcile(layoutOf([], []));
    items.syncBarItems();
    expect(host.isLive(healthShells()[0])).toBe(false);

    // The health poll goes on running whether or not the operator kept the
    // item, and writing into the pooled node is exactly right: it is the same
    // node that comes back when the item is placed again.
    settleHealth(panel.id, true);
    await flush();
    expect(dotClass(dotById(id))).toBe('status-dot live');
  });

  test('the item starts nothing that has to be stopped', () => {
    vi.useFakeTimers();
    seedDom('', shellMarkup('panel-health'));
    host.hydrate();

    // No interval, no subscription: the dots are written from outside, by the
    // health poll the rail already runs.
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe('panel-health item: the retired status-bar ids', () => {
  test('name no dead id, and hard-code no live one', () => {
    for (const name of ['bar-items.js', 'panel-status-bar.js']) {
      const source = readFileSync(join(JS_DIR, name), 'utf8');
      // `#operator-status` / `#operator-dot` were dead markup with no panel
      // behind them; the SSR task deleted the markup half.
      expect(source).not.toContain('operator-status');
      expect(source).not.toContain('operator-dot');
      // And no dot id is written down here AT ALL — not the live ones, and not
      // `channel-finder-status`, whose declaration no template has ever
      // matched. The set comes from the catalogs, so retiring a declaration
      // retires its dot with no edit to either of these files.
      expect(source).not.toContain('channel-finder-status');
      expect(source).not.toContain('ariel-status');
    }
  });
});
