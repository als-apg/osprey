/**
 * OSPREY Web Terminal — the status readouts, on the real boot path.
 *
 *   npx vitest run tests/interfaces/web_terminal/js/bar-status-readouts.test.js
 *
 * The bars became item hosts across several tasks: the template stopped
 * emitting `#ws-dot`, `#term-dims` and `#status-clock`, and `initStatusBar()`
 * — the one function that wrote all three — was deleted. Between those two
 * changes the status bar renders, every suite is green, and the three readouts
 * are DEAD: a bar of empty shells is indistinguishable from a bar of live ones
 * to any test that only asks whether the markup is there.
 *
 * Division of labour. `bar-items.test.js` proves each builder works when
 * it is CALLED, over a hand-built fixture with an explicit `host.hydrate()`;
 * `test_bar_items_ssr.py` proves the server emits the right shells; and the
 * real-browser smoke `test_load_smokes.py::test_web_terminal_status_readouts_are_live`
 * proves the SHIPPED page closes the chain end to end — including that
 * `app.js` still performs the bare `import './bar-items.js'`, which nothing
 * here reads. This file is the fast, unit-lane half of that: it runs the join
 * itself, without a browser. The shells are in the document BEFORE the module
 * graph loads, and the only thing that fills them is `bar-host.js` hydrating
 * from its own module body as a side effect of the bare import. So that is the
 * order these tests boot in — DOM first, then import — rather than calling
 * `hydrate()` by hand, which is the one step a broken boot would not perform.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { qs } from '../../_support/dom.mjs';

// `import.meta.dirname` is a plain string, so it sidesteps happy-dom's
// override of the global URL (which breaks fileURLToPath(new URL(...)) here).
const STATIC_DIR = join(
  import.meta.dirname,
  '../../../../src/osprey/interfaces/web_terminal/static'
);
const indexHtml = readFileSync(join(STATIC_DIR, 'index.html'), 'utf8');

const API_MODULE = '../../../../src/osprey/interfaces/web_terminal/static/js/api.js';
const BAR_ITEMS_MODULE = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-items.js';
const TERMINAL_MODULE = '../../../../src/osprey/interfaces/web_terminal/static/js/terminal.js';

/** The three readouts, and the id each one used to be hardcoded as. */
const RETIRED_IDS = { connection: 'ws-dot', 'terminal-size': 'term-dims', clock: 'status-clock' };

describe('status readouts are live after boot', () => {
  /** Every WebSocket `api.js` has constructed this test. @type {any[]} */
  let sockets = [];
  /** What the stubbed `terminal.js` reports as the fitted size. */
  let terminalDims = /** @type {{cols: number, rows: number} | null} */ (null);
  /** @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-items.js') | null} */
  let barItems = null;

  beforeEach(() => {
    document.body.innerHTML = '';
    sockets = [];
    terminalDims = { cols: 120, rows: 40 };
    barItems = null;
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
  });

  afterEach(() => {
    // Each item owns a subscription or an interval; leaving them armed would
    // outlive the test that started them.
    barItems?.disposeBarItems();
    barItems = null;
    vi.doUnmock(TERMINAL_MODULE);
    vi.resetModules();
    vi.unstubAllGlobals();
    document.body.innerHTML = '';
  });

  /**
   * The status bar as a bar host, carrying one empty shell per readout in the
   * shape the template emits (`data-bar-item`, no body), plus
   * the hidden pool the host parks folded items in.
   *
   * The footer's opening tag is read from the shipped index.html so the host
   * attributes are the real ones; the shells cannot be sliced out with it,
   * because the footer's body is a Jinja loop over the effective layout.
   *
   * @param {readonly string[]} types
   * @returns {string}
   */
  function readoutBarMarkup(types) {
    const openTag = /<footer class="status-bar"[^>]*>/.exec(indexHtml);
    expect(openTag, 'index.html has no .status-bar footer').not.toBeNull();
    expect(openTag?.[0], 'status-bar tag is not plain markup').not.toMatch(/\{[%{]/);
    const shells = types
      .map((type) => `<div class="bar-item" data-bar-item="${type}"></div>`)
      .join('');
    return `${openTag?.[0]}${shells}</footer><div id="bar-item-pool" hidden></div>`;
  }

  /**
   * Stand the server-rendered bar up and hand back `api.js`, WITHOUT loading
   * the items yet — so a test can drive the connection state to where it will
   * already be by the time the bundle parses on a real page.
   *
   * @param {readonly string[]} [types]
   * @returns {Promise<typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/api.js')>}
   */
  async function renderBar(types = ['connection', 'terminal-size', 'clock']) {
    document.body.innerHTML = readoutBarMarkup(types);
    vi.resetModules();
    // xterm cannot be stood up in happy-dom, so the ONE value the terminal-size
    // item consumes is stubbed. That the real module still exports this getter
    // is pinned separately, against the real file, in js/bar-items.test.js.
    vi.doMock(TERMINAL_MODULE, () => ({ getTerminalDimensions: () => terminalDims }));
    return await import(API_MODULE);
  }

  /** Load the bundle the way `app.js` does: a bare import, for its side effect. */
  async function loadBars() {
    barItems = await import(BAR_ITEMS_MODULE);
  }

  /** Drive `api.js`'s socket state all the way to connected. */
  /** @param {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/api.js')} api */
  function connect(api) {
    api.createWebSocket('ws://localhost:5000/ws/terminal');
    sockets[sockets.length - 1].onopen();
  }

  /** @param {string} type @param {string} selector @returns {HTMLElement} */
  function partOf(type, selector) {
    return qs(document, `.bar-item[data-bar-item="${type}"] ${selector}`);
  }

  /** The local wall clock as `buildClock()` formats it. @param {Date} at */
  function hhmm(at) {
    const pad = (/** @type {number} */ n) => String(n).padStart(2, '0');
    return `${pad(at.getHours())}:${pad(at.getMinutes())}`;
  }

  test('the connection dot paints the socket state the boot already reached', async () => {
    const api = await renderBar();
    // The transition happens BEFORE the bundle parses, exactly as it does on a
    // real page — the socket opens while the browser is still fetching modules.
    // A dot that only listens for transitions would sit grey on a healthy
    // session forever, and no markup assertion would notice.
    connect(api);
    expect(api.getConnectionState().ws).toBe('connected');

    await loadBars();

    expect(partOf('connection', '.status-dot').className).toBe('status-dot live');
  });

  test('the terminal size reads the fitted dimensions, not the not-ready dash', async () => {
    terminalDims = { cols: 96, rows: 30 };
    await renderBar();

    await loadBars();

    const value = partOf('terminal-size', '.bar-terminal-size-value');
    expect(value.textContent).toBe('96×30');
    expect(value.textContent).toMatch(/^\d+×\d+$/);
    expect(partOf('terminal-size', '.bar-terminal-size').title).toBe(
      'Terminal size: 96 columns × 30 rows'
    );
  });

  test('the clock reads the real wall clock', async () => {
    const before = new Date();
    await renderBar();

    await loadBars();

    // Bracketed rather than frozen: the assertion is that the readout tracks
    // real time, which a fixed system time cannot distinguish from a constant.
    // The bracket only has to survive a minute turning over mid-boot.
    const text = partOf('clock', '.bar-clock-time').textContent;
    expect(text).toMatch(/^\d{2}:\d{2}$/);
    expect([hhmm(before), hhmm(new Date())]).toContain(text);
  });

  test('all three are live at once, and none of the retired ids is back', async () => {
    const api = await renderBar();
    connect(api);

    await loadBars();

    // The gate: a bar of empty shells passes every other assertion in this
    // file. Each readout has to have a BODY with a value in it.
    expect(partOf('connection', '.status-dot').className).toBe('status-dot live');
    expect(partOf('terminal-size', '.bar-terminal-size-value').textContent).toBe('120×40');
    expect(partOf('clock', '.bar-clock-time').textContent).toMatch(/^\d{2}:\d{2}$/);
    for (const type of Object.keys(RETIRED_IDS)) {
      const shell = qs(document, `.bar-item[data-bar-item="${type}"]`);
      expect(shell.dataset.barBuilt, `${type} shell was never built`).toBeDefined();
      expect(shell.textContent?.trim(), `${type} rendered an empty body`).not.toBe('');
    }
    // The old ids are gone for good: an item can be moved, folded or left out,
    // so anything that found one again would be a hardcoded readout returning.
    for (const id of Object.values(RETIRED_IDS)) {
      expect(document.getElementById(id), `#${id} is back`).toBeNull();
    }
  });

  test('a shell the layout leaves out costs the others nothing', async () => {
    // A deployment that places only the clock must still get a live clock —
    // the readouts do not depend on each other, and the shipped default layout
    // places none of the other two.
    await renderBar(['clock']);

    await loadBars();

    expect(partOf('clock', '.bar-clock-time').textContent).toMatch(/^\d{2}:\d{2}$/);
    expect(document.querySelector('.bar-connection')).toBeNull();
    expect(document.querySelector('.bar-terminal-size')).toBeNull();
  });

  test('the shipped presets offer all three, so a layout can name them', async () => {
    const { PRESETS } = await import(
      '../../../../src/osprey/interfaces/web_terminal/static/js/bar-layout.js'
    );
    const full = PRESETS.find((preset) => preset.id === 'full');
    expect(full, 'no "full" preset').toBeDefined();

    // Binds the fixture above to shipped configuration: these three types are
    // placeable in the status bar, so the boot path they exercise is reachable.
    expect(full?.layout.status.map((item) => item.type)).toEqual(
      expect.arrayContaining(Object.keys(RETIRED_IDS))
    );
  });
});
