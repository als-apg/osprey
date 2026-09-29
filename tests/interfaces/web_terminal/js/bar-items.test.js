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
 * gives each test untouched module state (bar-host's shell index and
 * bar-items' instance map are module-private).
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

/** @typedef {import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-layout.js').BarLayout} BarLayout */

const HOST_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js';
const ITEMS_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-items.js';

/** @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js')} */
let host;
/** @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-items.js')} */
let items;

beforeEach(async () => {
  vi.resetModules();
  host = await import(HOST_PATH);
  items = await import(ITEMS_PATH);
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

/* ============================================================================
 * The interval-owning items. An interval that outlives its body keeps
 * writing into a node nobody can see for the life of the page, so every
 * timed item is pinned to stop the moment its shell leaves the bar.
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
  // Every row is seeded into the status bar, so the UTC rows also pin that a
  // UTC clock keeps its label at compact density: an unmarked UTC readout is
  // not terse, it is wrong.
  test.each([
    ['the default is the local wall clock, to the minute', {}, () => localTimeText(AT, false), null],
    ['the seconds option adds the seconds field', { seconds: true }, () => localTimeText(AT, true), null],
    ['zone utc reads UTC rather than the browser zone', { zone: 'utc' }, () => '14:32', 'UTC'],
    ['zone utc with seconds carries the whole UTC field', { zone: 'utc', seconds: true }, () => '14:32:07', 'UTC'],
    [
      'zone both shows local beside UTC, and marks which half is which',
      { zone: 'both' },
      () => `${localTimeText(AT, false)} · 14:32`,
      'UTC',
    ],
    [
      'an unknown zone falls back to the plain clock rather than rendering nothing',
      { zone: 'mars' },
      () => localTimeText(AT, false),
      null,
    ],
  ])('%s', (_name, options, expected, label) => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock', options));
    host.hydrate();

    expect(shellOf('clock').dataset.barDensity).toBe('compact');
    expect(clockTime().textContent).toBe(expected());
    expect(clockZoneLabel()?.textContent ?? null).toBe(label);
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
  test('coming back out of the pool ticks again, on a fresh body', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock', { zone: 'utc', seconds: true }));
    host.hydrate();
    const stale = clockTime();

    host.reconcile(layoutOf([], []));
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

    expect(vi.getTimerCount()).toBe(1);
  });
});

describe('clock item: the host lifecycle around the interval', () => {
  test('parking disposes on its own, in the same call as the move', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock'));
    host.hydrate();
    expect(vi.getTimerCount()).toBe(1);
    expect(shellOf('clock').dataset.barBuilt).toBeTruthy();

    // No await: the host's detach hook disposes the body before reconcile()
    // returns.
    host.reconcile(layoutOf([], []));

    expect(vi.getTimerCount()).toBe(0);
    expect(shellOf('clock').dataset.barBuilt).toBeUndefined();
  });

  test('a fold parks through the same hook — parkShell alone disposes', () => {
    freezeAt(AT);
    seedDom('', shellMarkup('clock', { zone: 'utc', seconds: true }));
    host.hydrate();
    const parked = clockTime();

    host.parkShell(shellOf('clock'));

    expect(vi.getTimerCount()).toBe(0);
    vi.advanceTimersByTime(60 * 60 * 1000);
    expect(parked.textContent).toBe('14:32:07');
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
