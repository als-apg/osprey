/**
 * Bar item builders and their attach/detach lifecycle, happy-dom environment
 * (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/js/bar-items.test.js
 *
 * The three assertions this file exists for are every live item's whole
 * contract, and each one is a bug that shipped in the hardcoded status bar or
 * would have shipped the moment its readouts became foldable:
 *
 *   - ATTACH SEEDS. A body paints what is true the instant it is built,
 *     without waiting for a tick or a transition. A clock that waited for
 *     its first interval would sit blank for a second on every placement.
 *
 *   - DETACH DISPOSES. Folding does not destroy an item — the host PARKS the
 *     node in `#bar-item-pool`, alive. A timer that survives parking keeps
 *     writing into a body nobody can see for the rest of the page's life,
 *     and nothing anywhere would report it. So the tests advance time after
 *     the detach and pin that the parked body did not move.
 *
 *   - RE-ATTACH RE-SEEDS. Coming back out of the pool has to show what is true
 *     now, not the state the item froze at when it folded. This is the half
 *     that fails silently if `data-bar-built` is left stamped: the host would
 *     skip the rebuild, the body would look right, and it would be dead.
 *
 * These run against the REAL bar-host.js — no mock of the lifecycle under
 * test — because the thing being pinned is precisely that the modules agree
 * about a lifetime. `vi.resetModules()` plus a fresh dynamic import per test
 * gives each test untouched module state (bar-host's shell index, bar-items'
 * instance map and api.js's listener array are all module-private and have
 * no reset).
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

/** @typedef {import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-layout.js').BarLayout} BarLayout */

const API_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/api.js';
const HOST_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js';
const ITEMS_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-items.js';
const QUEUE_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-item-queue.js';

/** @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/api.js')} */
let api;
/** @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js')} */
let host;
/** @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-items.js')} */
let items;
/** Every WebSocket api.js has constructed this test. @type {any[]} */
let sockets;
/** Every EventSource the plan-queue item has opened this test. @type {any[]} */
let sources;

beforeEach(async () => {
  sockets = [];
  sources = [];
  vi.stubGlobal(
    'EventSource',
    class {
      /** @param {string} url */
      constructor(url) {
        this.url = url;
        this.readyState = 0;
        /** @type {((e: {data: string}) => void) | null} */
        this.onmessage = null;
        /** @type {(() => void) | null} */
        this.onopen = null;
        /** @type {(() => void) | null} */
        this.onerror = null;
        sources.push(this);
      }
      close() {
        this.readyState = 2;
      }
    }
  );
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
  api = await import(API_PATH);
  host = await import(HOST_PATH);
  items = await import(ITEMS_PATH);
  await import(QUEUE_PATH);
});

afterEach(() => {
  items.disposeBarItems();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

/**
 * The SSR DOM the host hydrates from: both hosts and the hidden pool. Shells
 * are passed as markup so a test can seed an item into either bar.
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
    header_visible: true,
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

/** Drive api.js's ws state to 'connecting' by opening a socket. */
function openSocket() {
  api.createWebSocket('ws://localhost:5000/ws/terminal');
  return sockets[sockets.length - 1];
}

/** Drive api.js's ws state all the way to 'connected'. */
function connect() {
  openSocket().onopen();
}

describe('api.js: the disposer-returning subscription the house rule is built on', () => {
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
 * The interval-owning items. What api.js's subscription proves for a
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
 * The same, on the 12-hour cycle: `2:32 PM`, hour unpadded.
 * @param {Date} at
 * @returns {string}
 */
function localTime12Text(at) {
  const pad = (/** @type {number} */ n) => String(n).padStart(2, '0');
  const hours = at.getHours();
  return `${hours % 12 || 12}:${pad(at.getMinutes())} ${hours < 12 ? 'AM' : 'PM'}`;
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

  test('an unknown zone falls back to the plain clock rather than rendering nothing', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock', { zone: 'mars' }));
    host.hydrate();

    expect(clockTime().textContent).toBe(localTimeText(AT, false));
    expect(clockZoneLabel()).toBeNull();
  });

  test('the 12h format reads the meridiem, at either zone', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock', { format: '12h' }));
    host.hydrate();
    expect(clockTime().textContent).toBe(localTime12Text(AT));

    // AT is 14:32:07 UTC: the hour drops its leading zero and its 13–23 half.
    host.reconcile(layoutOf([], [{ type: 'clock', options: { zone: 'utc', format: '12h' } }]));
    expect(clockTime().textContent).toBe('2:32 PM');

    host.reconcile(
      layoutOf([], [{ type: 'clock', options: { zone: 'utc', format: '12h', seconds: true } }])
    );
    expect(clockTime().textContent).toBe('2:32:07 PM');

    host.reconcile(layoutOf([], [{ type: 'clock', options: { zone: 'both', format: '12h' } }]));
    expect(clockTime().textContent).toBe(`${localTime12Text(AT)} · 2:32 PM`);
  });

  test('the 12h format keeps midnight and noon on the clock face', () => {
    freezeAt(new Date('2026-03-14T00:05:00Z'));
    seedDom('', shellMarkup('clock', { zone: 'utc', format: '12h' }));
    host.hydrate();
    expect(clockTime().textContent).toBe('12:05 AM');

    // The item keeps running; moving the fake clock and letting one tick fire
    // is what repaints it at noon. (`freezeAt` would re-install the fake
    // timers and drop the interval with them.)
    vi.setSystemTime(new Date('2026-03-14T12:05:00Z'));
    vi.advanceTimersByTime(1000);
    expect(clockTime().textContent).toBe('12:05 PM');
  });

  test('the default clock is plain: no zone suffix at either density', () => {
    freezeAt(AT);
    seedDom(shellMarkup('clock'), '');
    host.hydrate();

    expect(shellOf('clock').dataset.barDensity).toBe('comfortable');
    expect(clockZoneLabel()).toBeNull();

    host.reconcile(layoutOf([], ['clock']));
    expect(shellOf('clock').dataset.barDensity).toBe('compact');
    expect(clockZoneLabel()).toBeNull();
  });

  test('the named local zone is a comfortable-density affordance', () => {
    freezeAt(AT);
    seedDom(shellMarkup('clock', { zone: 'local' }), '');
    host.hydrate();

    // The header has room to say WHICH clock this is; the 20px status bar does
    // not, and an unlabelled local clock is simply "the time".
    expect(shellOf('clock').dataset.barDensity).toBe('comfortable');
    expect(clockZoneLabel()?.textContent).toBeTruthy();

    host.reconcile(layoutOf([], [{ type: 'clock', options: { zone: 'local' } }]));
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

describe('clock item: the host lifecycle around the interval', () => {
  test('detach clears the build stamp so the next placement rebuilds', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock'));
    host.hydrate();
    expect(shellOf('clock').dataset.barBuilt).toBeTruthy();

    host.reconcile(layoutOf([], []));
    items.syncBarItems();

    expect(shellOf('clock').dataset.barBuilt).toBeUndefined();
    expect(host.isLive(shellOf('clock'))).toBe(false);
  });

  test('syncBarItems is idempotent — a second pass disposes nothing twice', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock'));
    host.hydrate();
    host.reconcile(layoutOf([], []));

    items.syncBarItems();
    expect(() => items.syncBarItems()).not.toThrow();
    expect(vi.getTimerCount()).toBe(0);
  });

  test('parking disposes on its own, without a caller running the pass', async () => {
    seedDom('', shellMarkup('clock'));
    host.hydrate();
    expect(shellOf('clock').dataset.barBuilt).toBeTruthy();

    host.reconcile(layoutOf([], []));
    // No syncBarItems() call: the pool observer is the production path.
    await Promise.resolve();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(shellOf('clock').dataset.barBuilt).toBeUndefined();
  });

  test('a body removed from the document entirely is disposed as well', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock', { zone: 'utc', seconds: true }));
    host.hydrate();
    const orphan = clockTime();
    expect(vi.getTimerCount()).toBe(1);

    document.body.innerHTML = '';
    items.syncBarItems();

    expect(vi.getTimerCount()).toBe(0);
    vi.advanceTimersByTime(60 * 60 * 1000);
    expect(orphan.textContent).toBe('14:32:07');
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

describe('feedback item: a second way to press the rail control', () => {
  test('a click forwards to the rail button, which owns the dialog', () => {
    seedDom(shellMarkup('feedback'));
    document.body.insertAdjacentHTML(
      'beforeend',
      '<button id="panel-feedback-btn" type="button">Feedback</button>'
    );
    const rail = el('#panel-feedback-btn');
    const pressed = vi.fn();
    rail.addEventListener('click', pressed);
    host.hydrate();

    partOf('feedback', 'button.bar-feedback').click();

    expect(pressed).toHaveBeenCalledTimes(1);
  });

  test('the header body carries the glyph and the label; the status bar the label', () => {
    seedDom(shellMarkup('feedback'), shellMarkup('feedback'));
    host.hydrate();

    const inHeader = el('[data-bar-host="header"] .bar-feedback');
    const inStatus = el('[data-bar-host="status"] .bar-feedback');
    expect(inHeader.querySelector('svg.bar-feedback-icon')).not.toBe(null);
    expect(inHeader.textContent).toBe('Feedback');
    expect(inStatus.querySelector('svg')).toBe(null);
    expect(inStatus.textContent).toBe('Feedback');
  });
});

describe('space item: edit-mode furniture', () => {
  test('a flexible space labels itself with the fill glyph and carries two grips', () => {
    seedDom(shellMarkup('space'));
    host.hydrate();

    expect(partOf('space', '.bar-space-label').textContent).toBe('⟷');
    expect(
      Array.from(shellOf('space').querySelectorAll('.bar-space-grip')).map(
        (grip) => /** @type {HTMLElement} */ (grip).dataset.edge
      )
    ).toEqual(['start', 'end']);
  });

  test('a fixed space names its width, and a rebuild follows the option', () => {
    seedDom(shellMarkup('space', { width: 120 }));
    host.hydrate();
    expect(partOf('space', '.bar-space-label').textContent).toBe('120 px');

    host.reconcile(layoutOf([{ type: 'space', options: { width: 48 } }], []));
    expect(partOf('space', '.bar-space-label').textContent).toBe('48 px');
    expect(shellOf('space').style.getPropertyValue('flex')).toBe('0 1 48px');
  });

  test('spaceLabel is the one spelling both the builder and the drag use', () => {
    expect(items.spaceLabel(0)).toBe('⟷');
    expect(items.spaceLabel(99.6)).toBe('100 px');
  });
});

/* ---- plan queue ---- */

/**
 * One bridge frame, in the shape `GET /queue/events` streams: a bounded
 * status summary, the pending items and the running item.
 * @param {Partial<{state: string, items: string[], running: Record<string, any> | null,
 *   available: boolean, stopPending: boolean}>} [frame]
 */
function queueFrame({
  state = 'idle',
  items = [],
  running = null,
  available = true,
  stopPending = false,
} = {}) {
  return {
    type: 'queue',
    status: {
      available,
      manager_state: available ? state : null,
      items_in_queue: items.length,
      queue_stop_pending: stopPending,
    },
    items: items.map((name, index) => ({ name, item_uid: `uid-${index}` })),
    running_item: running,
  };
}

/** Deliver one frame on the newest stream. @param {unknown} frame */
function pushFrame(frame) {
  const source = sources[sources.length - 1];
  source.readyState = 1;
  source.onopen?.();
  source.onmessage?.({ data: JSON.stringify(frame) });
}

/** The placed chip. */
const queueChip = () => el('[data-bar-item="bluesky-queue"] .bar-queue');
const queuePop = () => el('[data-bar-item="bluesky-queue"] .bar-queue-pop');
/** @param {string} label */
const popButton = (label) =>
  Array.from(queuePop().querySelectorAll('button')).find((b) => b.textContent === label) ?? null;

describe('plan-queue item: one shared stream through the Bluesky panel proxy', () => {
  test('attach opens the panel-proxied event stream and seeds a quiet chip', () => {
    seedDom('', shellMarkup('bluesky-queue'));
    host.hydrate();

    expect(sources.map((s) => s.url)).toEqual(['/panel/bluesky/queue/events']);
    const chip = queueChip();
    expect(chip.querySelector('.bar-queue-text')?.textContent).toBe('queue');
    expect(chip.querySelector('.bar-queue-dot')?.getAttribute('data-tone')).toBe('off');
    expect(chip.title).toContain('stream not connected');
  });

  test('a frame paints the state word, the running plan and the count', () => {
    seedDom('', shellMarkup('bluesky-queue'));
    host.hydrate();
    const chip = queueChip();
    const text = () => chip.querySelector('.bar-queue-text')?.textContent;
    const count = () => /** @type {HTMLElement | null} */ (chip.querySelector('.bar-queue-count'));
    const tone = () => chip.querySelector('.bar-queue-dot')?.getAttribute('data-tone');

    pushFrame(queueFrame({ items: ['rel_scan', 'count'] }));
    expect(text()).toBe('idle');
    expect(tone()).toBe('idle');
    expect(count()?.textContent).toBe('2 queued');
    expect(count()?.hidden).toBe(false);

    pushFrame(
      queueFrame({
        state: 'executing_queue',
        items: ['count'],
        running: { name: 'rel_scan', progress: { rows_seen: 3, expected_points: 10 } },
      })
    );
    expect(text()).toBe('rel_scan');
    expect(tone()).toBe('active');
    expect(count()?.textContent).toBe('3/10');

    pushFrame(queueFrame({ state: 'paused', running: { name: 'rel_scan' } }));
    expect(text()).toBe('rel_scan');
    expect(tone()).toBe('warn');
    expect(count()?.hidden).toBe(true);

    pushFrame(queueFrame({ available: false }));
    expect(text()).toBe('unavailable');
    expect(tone()).toBe('err');
  });

  test('the options decide what the chip says beside its dot', () => {
    seedDom('', shellMarkup('bluesky-queue', { progress: false, count: false, controls: 'none' }));
    host.hydrate();
    pushFrame(
      queueFrame({ state: 'executing_queue', items: ['count'], running: { name: 'rel_scan' } })
    );
    const chip = queueChip();
    expect(chip.querySelector('.bar-queue-text')?.textContent).toBe('running');
    const corner = /** @type {HTMLElement | null} */ (chip.querySelector('.bar-queue-count'));
    expect(corner?.hidden).toBe(true);
  });

  test('the stream is attach-scoped: parking closes it, placing back reopens it', () => {
    seedDom('', shellMarkup('bluesky-queue'));
    host.hydrate();
    const first = sources[0];
    expect(first.readyState).not.toBe(2);

    host.reconcile(layoutOf([], ['clock']));
    items.syncBarItems();
    expect(first.readyState).toBe(2);
    expect(sources).toHaveLength(1);

    host.reconcile(layoutOf([], ['bluesky-queue']));
    expect(sources).toHaveLength(2);
    expect(sources[1].url).toBe('/panel/bluesky/queue/events');
  });

  test("a preview beside a placed item shares the placed item's stream", () => {
    seedDom('', shellMarkup('bluesky-queue'));
    host.hydrate();

    const preview = items.previewBarItem('bluesky-queue', document, 'comfortable');
    if (!preview) throw new Error('no preview for the plan queue');
    expect(sources).toHaveLength(1);
    preview.dispose?.();
    expect(sources[0].readyState).not.toBe(2);
  });
});

describe('plan-queue item: the popover and its controls', () => {
  /** @type {any} */
  let fetchSpy;
  beforeEach(() => {
    fetchSpy = vi.fn(async () => ({ ok: true, status: 200, json: async () => ({}) }));
    vi.stubGlobal('fetch', fetchSpy);
  });

  test('the chip only opens; the card lists the queue and offers Open Bluesky alone', () => {
    seedDom('', shellMarkup('bluesky-queue'));
    host.hydrate();
    pushFrame(
      queueFrame({ state: 'executing_queue', items: ['count', 'grid'], running: { name: 'rel_scan' } })
    );
    const chip = queueChip();
    expect(queuePop().hidden).toBe(true);

    chip.click();

    expect(queuePop().hidden).toBe(false);
    expect(chip.getAttribute('aria-expanded')).toBe('true');
    const names = Array.from(queuePop().querySelectorAll('.bar-queue-row-name')).map(
      (n) => n.textContent
    );
    expect(names).toEqual(['rel_scan', 'count', 'grid']);
    expect(Array.from(queuePop().querySelectorAll('button')).map((b) => b.textContent)).toEqual([
      'Open Bluesky',
    ]);
    expect(fetchSpy).not.toHaveBeenCalled();

    chip.click();
    expect(queuePop().hidden).toBe(true);
    expect(chip.getAttribute('aria-expanded')).toBe('false');
  });

  test('`controls: stop` adds the plain stop, which fires on the first click', () => {
    seedDom('', shellMarkup('bluesky-queue', { controls: 'stop' }));
    host.hydrate();
    pushFrame(queueFrame({ state: 'executing_queue', running: { name: 'rel_scan' } }));
    queueChip().click();

    const stop = popButton('Stop after current item');
    if (!stop) throw new Error('no stop button');
    expect(popButton('Abort running plan')).toBe(null);
    stop.click();

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe('/panel/bluesky/queue/stop');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ cancel: false });
  });

  test('withdrawing a pending stop is two-step', () => {
    seedDom('', shellMarkup('bluesky-queue', { controls: 'stop' }));
    host.hydrate();
    pushFrame(
      queueFrame({ state: 'executing_queue', running: { name: 'rel_scan' }, stopPending: true })
    );
    queueChip().click();

    const withdraw = popButton('Withdraw stop');
    if (!withdraw) throw new Error('no withdraw button');
    withdraw.click();
    expect(fetchSpy).not.toHaveBeenCalled();
    const confirm = popButton('Confirm — the queue keeps draining');
    if (!confirm) throw new Error('no confirm button');
    confirm.click();

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(JSON.parse(fetchSpy.mock.calls[0][1].body)).toEqual({ cancel: true });
  });

  test('`controls: full` adds Start and the two-step abort', () => {
    seedDom('', shellMarkup('bluesky-queue', { controls: 'full' }));
    host.hydrate();
    pushFrame(queueFrame({ items: ['count'] }));
    queueChip().click();

    const start = popButton('Start');
    if (!start) throw new Error('no start button');
    expect(start.disabled).toBe(false);
    const abort = popButton('Abort running plan');
    if (!abort) throw new Error('no abort button');

    abort.click();
    expect(fetchSpy).not.toHaveBeenCalled();
    const confirm = popButton('Confirm abort');
    if (!confirm) throw new Error('abort did not arm');
    confirm.click();
    expect(fetchSpy.mock.calls[0][0]).toBe('/panel/bluesky/queue/abort');

    popButton('Start')?.click();
    expect(fetchSpy.mock.calls[1][0]).toBe('/panel/bluesky/queue/start');
  });

  test('Start is disabled while the queue is draining or empty', () => {
    seedDom('', shellMarkup('bluesky-queue', { controls: 'full' }));
    host.hydrate();
    pushFrame(queueFrame({ state: 'executing_queue', items: ['count'], running: { name: 'x' } }));
    queueChip().click();
    expect(popButton('Start')?.disabled).toBe(true);

    pushFrame(queueFrame({ items: [] }));
    expect(popButton('Start')?.disabled).toBe(true);
  });

  test("a refused write shows the bridge's own sentence", async () => {
    fetchSpy.mockResolvedValue({
      ok: false,
      status: 409,
      json: async () => ({ detail: { code: 'not_armed', detail: 'This deployment is not armed.' } }),
    });
    seedDom('', shellMarkup('bluesky-queue', { controls: 'stop' }));
    host.hydrate();
    pushFrame(queueFrame({ state: 'executing_queue', running: { name: 'rel_scan' } }));
    queueChip().click();
    popButton('Stop after current item')?.click();
    await vi.waitFor(() => {
      expect(queuePop().querySelector('.bar-queue-note')?.textContent).toBe(
        'This deployment is not armed.'
      );
    });
  });

  test('Escape and an outside click close the card', () => {
    seedDom('', shellMarkup('bluesky-queue'));
    host.hydrate();
    queueChip().click();
    expect(queuePop().hidden).toBe(false);

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(queuePop().hidden).toBe(true);

    queueChip().click();
    document.body.click();
    expect(queuePop().hidden).toBe(true);
  });
});

describe('previewBarItem: a body outside the bars', () => {
  test('builds through the registered factory without touching the pool', () => {
    seedDom();
    host.hydrate();

    const preview = items.previewBarItem('clock', document, 'comfortable');
    if (!preview) throw new Error('no preview for clock');
    const node = /** @type {HTMLElement} */ (preview.node);
    expect(node.classList.contains('bar-clock')).toBe(true);
    expect(el('#bar-item-pool').childElementCount).toBe(0);
    preview.dispose?.();
  });

  test('answers null for a type no factory renders', () => {
    expect(items.previewBarItem('docs', document, 'comfortable')).toBe(null);
  });
});
