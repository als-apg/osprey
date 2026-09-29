/**
 * OSPREY Web Terminal — the status readouts, on the real boot path.
 *
 *   npx vitest run tests/interfaces/web_terminal/js/bar-status-readouts.test.js
 *
 * The bars became item hosts across several tasks: the template stopped
 * emitting `#ws-dot`, `#term-dims` and `#status-clock`, and `initStatusBar()`
 * — the one function that wrote all three — was deleted (the connection and
 * terminal-size readouts have since been retired outright). Between those two
 * changes the status bar renders, every suite is green, and the readout is
 * DEAD: a bar of empty shells is indistinguishable from a bar of live ones to
 * any test that only asks whether the markup is there.
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

const BAR_ITEMS_MODULE = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-items.js';

/** Every id the template used to emit — two of them have no item behind them any more. */
const RETIRED_IDS = ['ws-dot', 'term-dims', 'status-clock'];

describe('status readouts are live after boot', () => {
  /** @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-items.js') | null} */
  let barItems = null;

  beforeEach(() => {
    document.body.innerHTML = '';
    barItems = null;
  });

  afterEach(() => {
    // Each item owns a subscription or an interval; leaving them armed would
    // outlive the test that started them.
    barItems?.disposeBarItems();
    barItems = null;
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
   * Stand the server-rendered bar up WITHOUT loading the items yet — the
   * shells are in the document before the module graph is.
   * @param {readonly string[]} [types]
   */
  function renderBar(types = ['clock']) {
    document.body.innerHTML = readoutBarMarkup(types);
    vi.resetModules();
  }

  /** Load the bundle the way `app.js` does: a bare import, for its side effect. */
  async function loadBars() {
    barItems = await import(BAR_ITEMS_MODULE);
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

  test('the clock is live on the real wall clock, and none of the retired ids is back', async () => {
    const before = new Date();
    renderBar();

    await loadBars();

    // The gate: a bar of empty shells passes every other check. The readout
    // has to have a BODY with a value in it, and the value has to track real
    // time — bracketed rather than frozen, because a fixed system time cannot
    // tell a clock from a constant. The bracket only has to survive a minute
    // turning over mid-boot.
    const shell = qs(document, '.bar-item[data-bar-item="clock"]');
    expect(shell.dataset.barBuilt, 'clock shell was never built').toBeDefined();
    const text = partOf('clock', '.bar-clock-time').textContent;
    expect([hhmm(before), hhmm(new Date())]).toContain(text);
    // The old ids are gone from the template for good: an item can be moved,
    // folded or left out, so a template that emitted one again would be a
    // hardcoded readout returning.
    for (const id of RETIRED_IDS) {
      expect(indexHtml, `#${id} is back in index.html`).not.toMatch(new RegExp(`id="${id}"`));
    }
  });

});
