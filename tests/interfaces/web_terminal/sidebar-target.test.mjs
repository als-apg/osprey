// @ts-check
/**
 * The sidebar's control-system block (sidebar-target.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-target.test.mjs
 *
 * The block dresses the control-target chip mounted in `#sidebar-target` —
 * decoration only, all of it aria-hidden, with the chip's own words read from
 * `targetReading()` for the collapsed sidebar's tooltip — and adds a meta line
 * with the SYSTEM panel's health and the facility clock. Where no chip mounts
 * (the picker is off), the meta line stands alone in the block.
 *
 * Seams: control-target-chip.js is mocked (the chip element, its subscribers
 * and its reading are what this module reads; the chip's own suite pins how it
 * derives them), `fetch` is stubbed per test, and the facility zone is stamped
 * on <html> through tests/interfaces/_support/facility-zone.mjs. Module state
 * is isolated by vi.resetModules() + dynamic import per test and the module's
 * own teardown.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { byId, qs } from '../_support/dom.mjs';
import {
  FACILITY_ZONE,
  VIEWER_ZONE,
  stampFacilityZone,
  zoneName,
} from '../_support/facility-zone.mjs';

const MODULE = '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-target.js';

/** The chip as this module sees it: its element, its reading, its subscribers. */
const chip = vi.hoisted(() => ({
  /** @type {HTMLElement | null} */
  el: null,
  /** @type {{name: string, state: string, kind: string, word: string} | null} */
  reading: null,
  /** @type {Set<() => void>} */
  listeners: new Set(),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/control-target-chip.js', () => ({
  getChipElement: () => chip.el,
  targetReading: () => chip.reading,
  subscribe: (/** @type {() => void} */ fn) => {
    chip.listeners.add(fn);
    return () => chip.listeners.delete(fn);
  },
}));

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-target.js')} */
let mod;
/** @type {ReturnType<typeof vi.fn>} */
let fetchMock;

/** A fixed instant, so the clock and the tooltip are deterministic. */
const AT = Date.UTC(2026, 0, 15, 20, 4);

/** @param {unknown} body @param {number} [status] */
const reply = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });

/** Answer every health read with `answer` (a reply, or a function of the call count). */
function stubHealth(/** @type {any} */ answer) {
  fetchMock = vi.fn(async () => (typeof answer === 'function' ? answer() : answer));
  vi.stubGlobal('fetch', fetchMock);
}

/** The block with the chip mounted in it, in control-target-chip.js's shape. */
function mountWithChip() {
  document.body.innerHTML = `
    <nav class="sidebar" id="sidebar" aria-label="Panels">
      <div class="sidebar-target" id="sidebar-target">
        <div class="ctc-anchor">
          <button class="control-target-chip" type="button" aria-haspopup="true"
                  aria-expanded="false" aria-label="Control target: Simulator, writes on">
            <span class="ctc-dot"></span><span class="ctc-short">Simulator</span><span class="ctc-sep">·</span><span class="ctc-state">writes on</span><span class="ctc-caret">▾</span>
          </button>
        </div>
      </div>
    </nav>`;
  chip.el = qs(document, '.control-target-chip');
  chip.reading = { name: 'Simulator', state: 'writes on', kind: 'simulator', word: 'on' };
}

/** The block with no chip: the deployment has no picker. */
function mountWithoutChip() {
  document.body.innerHTML = `
    <nav class="sidebar" id="sidebar" aria-label="Panels">
      <div class="sidebar-target" id="sidebar-target"></div>
    </nav>`;
  chip.el = null;
  chip.reading = null;
}

/** Let the read chain (fetch, then json) land. */
async function settle() {
  for (let i = 0; i < 10; i++) await Promise.resolve();
}

const host = () => byId('sidebar-target');
const meta = () => document.getElementById('sidebar-target-meta');
const dot = () => qs(document, '.sidebar-health');
const clock = () => qs(document, '.sidebar-target-clock');
const chipEl = () => /** @type {HTMLElement} */ (chip.el);

/** The `HH:MM` the block shows in *zone*, computed with Intl directly. */
function wallClock(
  /** @type {string} */ zone,
  /** @type {Intl.DateTimeFormatOptions} */ extra = {}
) {
  return new Intl.DateTimeFormat(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
    timeZone: zone,
    ...extra,
  }).format(AT);
}

/** The city a zone id names, as the block spells it. @param {string} zone */
const cityOf = (zone) => (zone.split('/').pop() ?? zone).replaceAll('_', ' ');

beforeEach(async () => {
  vi.resetModules();
  chip.el = null;
  chip.reading = null;
  chip.listeners.clear();
  stampFacilityZone(FACILITY_ZONE);
  Reflect.deleteProperty(window, '__OSPREY_PREFIX__');
  stubHealth(reply({ results: [{ status: 'ok' }], interval_s: 30 }));
  mod = await import(MODULE);
});

afterEach(() => {
  mod.teardownSidebarTarget();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  stampFacilityZone(null);
  document.body.innerHTML = '';
});

/* ---- readHealth ----------------------------------------------------------- */

describe('readHealth', () => {
  test('checking before any answer, unreachable after a failed read', () => {
    expect(mod.readHealth(null)).toEqual({ tone: 'off', word: 'checking' });
    expect(mod.readHealth(undefined)).toEqual({ tone: 'err', word: 'unreachable' });
  });

  test('warming while the first suite runs', () => {
    expect(mod.readHealth({ warming: true, results: [{ status: 'error' }] })).toEqual({
      tone: 'off',
      word: 'warming',
    });
  });

  test('no checks when nothing ran', () => {
    for (const report of [{}, { results: [] }, { results: [{ status: 'skip' }] }, 'garbage']) {
      expect(mod.readHealth(report)).toEqual({ tone: 'off', word: 'no checks' });
    }
  });

  test('the worst outcome, counted', () => {
    const results = (/** @type {string[]} */ ...statuses) => ({
      results: statuses.map((status) => ({ status })),
    });
    expect(mod.readHealth(results('error', 'warning', 'ok'))).toEqual({
      tone: 'err',
      word: '1 error',
    });
    expect(mod.readHealth(results('error', 'error', 'warning'))).toEqual({
      tone: 'err',
      word: '2 errors',
    });
    expect(mod.readHealth(results('warning', 'ok', 'skip'))).toEqual({
      tone: 'warn',
      word: '1 warning',
    });
    expect(mod.readHealth(results('warning', 'warning'))).toEqual({
      tone: 'warn',
      word: '2 warnings',
    });
    expect(mod.readHealth(results('ok', 'skip', 'ok'))).toEqual({ tone: 'ok', word: 'ok' });
  });
});

/* ---- clockText ------------------------------------------------------------ */

describe('clockText', () => {
  test("shows the facility's hour and minute and its zone's city", () => {
    expect(mod.clockText(AT)).toBe(`${wallClock(FACILITY_ZONE)} ${cityOf(FACILITY_ZONE)}`);
  });

  test("spells a city's underscores as spaces", () => {
    stampFacilityZone('America/Los_Angeles');
    expect(mod.clockText(AT)).toBe(`${wallClock('America/Los_Angeles')} Los Angeles`);
  });

  test("falls back to the viewer's zone and names it", () => {
    stampFacilityZone(null);
    const text = mod.clockText(AT);
    expect(text).toBe(wallClock(VIEWER_ZONE, { timeZoneName: 'short' }));
    expect(text).toContain(zoneName(VIEWER_ZONE, AT));
  });
});

/* ---- the block with a chip ------------------------------------------------ */

describe('with the chip mounted', () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(AT);
    mountWithChip();
  });

  test('dresses the chip with decoration only', async () => {
    mod.initSidebarTarget();
    await settle();
    const button = chipEl();

    const [glyph, eyebrow, caret] = [...button.children];
    expect(glyph.classList.contains('sidebar-target-glyph')).toBe(true);
    expect(eyebrow.classList.contains('sidebar-target-eyebrow')).toBe(true);
    expect(eyebrow.textContent).toBe('Control system');
    expect(caret.classList.contains('sidebar-target-caret')).toBe(true);
    for (const el of [glyph, eyebrow, caret]) expect(el.getAttribute('aria-hidden')).toBe('true');

    // The chip's own words and name are the chip's.
    expect(qs(button, '.ctc-short').textContent).toBe('Simulator');
    expect(qs(button, '.ctc-state').textContent).toBe('writes on');
    expect(button.getAttribute('aria-label')).toBe('Control target: Simulator, writes on');
    expect(host().dataset.target).toBe('chip');
  });

  test('puts the meta line inside the chip and describes the chip with it', async () => {
    mod.initSidebarTarget();
    await settle();
    const line = /** @type {HTMLElement} */ (meta());

    expect(line.parentElement).toBe(chipEl());
    expect(chipEl().lastElementChild).toBe(line);
    expect(chipEl().getAttribute('aria-describedby')).toBe('sidebar-target-meta');
    expect(line.querySelector('.osprey-dot.sidebar-health')).not.toBeNull();
    expect(clock().textContent).toBe(mod.clockText(AT));
  });

  test("the tooltip repeats the chip's reading and the clock", async () => {
    mod.initSidebarTarget();
    await settle();
    expect(chipEl().dataset.tip).toBe(`Simulator · writes on · ${mod.clockText(AT)}`);

    // A chip render reaches the tooltip through the chip's subscribers.
    chip.reading = { name: 'Real machine', state: 'switching…', kind: 'real', word: 'on' };
    for (const fn of chip.listeners) fn();
    expect(chipEl().dataset.tip).toBe(`Real machine · switching… · ${mod.clockText(AT)}`);

    // Nothing read yet: no tooltip rather than a half one.
    chip.reading = null;
    for (const fn of chip.listeners) fn();
    expect(chipEl().dataset.tip).toBeUndefined();
  });

  test('the tooltip steps aside while the popover is open', async () => {
    mod.initSidebarTarget();
    await settle();
    expect(chipEl().dataset.tip).toBeDefined();

    chipEl().setAttribute('aria-expanded', 'true');
    await vi.waitFor(() => expect(chipEl().dataset.tip).toBeUndefined());

    chipEl().setAttribute('aria-expanded', 'false');
    await vi.waitFor(() =>
      expect(chipEl().dataset.tip).toBe(`Simulator · writes on · ${mod.clockText(AT)}`)
    );
  });

  test('paints the health dot from the report', async () => {
    stubHealth(reply({ results: [{ status: 'warning' }, { status: 'ok' }], interval_s: 30 }));
    mod.initSidebarTarget();
    await settle();

    expect(dot().hidden).toBe(false);
    expect(dot().classList.contains('warn')).toBe(true);
    expect(dot().classList.contains('ok')).toBe(false);
    expect(dot().getAttribute('role')).toBe('img');
    expect(dot().getAttribute('aria-label')).toBe('Service health: 1 warning');
  });

  test('reads the report through the per-user prefix, never from cache', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    mod.initSidebarTarget();
    await settle();
    expect(fetchMock).toHaveBeenCalledWith('/u/alice/panel/system-health/checks', {
      cache: 'no-store',
    });
    expect(mod.HEALTH_API).toBe('/panel/system-health/checks');
  });

  test('a chip outside the block is not dressed', async () => {
    const elsewhere = document.createElement('div');
    elsewhere.append(chipEl());
    document.body.append(elsewhere);

    mod.initSidebarTarget();
    await settle();

    expect(chipEl().querySelector('.sidebar-target-glyph')).toBeNull();
    expect(meta()?.parentElement).toBe(host());
    expect(host().dataset.target).toBe('none');
  });
});

/* ---- the block without a chip --------------------------------------------- */

describe('without a chip', () => {
  beforeEach(() => {
    mountWithoutChip();
  });

  test('the meta line stands alone in the block', async () => {
    mod.initSidebarTarget();
    await settle();

    expect(meta()?.parentElement).toBe(host());
    expect(host().dataset.target).toBe('none');
    expect(document.querySelector('[aria-describedby]')).toBeNull();
    expect(dot().classList.contains('ok')).toBe(true);
    expect(dot().getAttribute('aria-label')).toBe('Service health: ok');
    expect(clock().textContent).not.toBe('');
  });

  test('a second init adds nothing', async () => {
    mod.initSidebarTarget();
    mod.initSidebarTarget();
    await settle();
    expect(document.querySelectorAll('#sidebar-target-meta')).toHaveLength(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test('a page without the block is left alone', async () => {
    document.body.innerHTML = '<div id="elsewhere"></div>';
    mod.initSidebarTarget();
    await settle();
    expect(meta()).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  test('a failed read shows unreachable in the error tone', async () => {
    stubHealth(reply({ detail: 'boom' }, 502));
    mod.initSidebarTarget();
    await settle();
    expect(dot().hidden).toBe(false);
    expect(dot().classList.contains('err')).toBe(true);
    expect(dot().getAttribute('aria-label')).toBe('Service health: unreachable');
  });

  test('says checking before the first answer', () => {
    stubHealth(new Promise(() => {}));
    mod.initSidebarTarget();
    expect(dot().getAttribute('aria-label')).toBe('Service health: checking');
    expect(dot().classList.contains('ok')).toBe(false);
  });
});

/* ---- polling -------------------------------------------------------------- */

describe('polling', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(AT);
    mountWithoutChip();
  });

  test("reads again on the report's own cadence", async () => {
    mod.initSidebarTarget();
    await settle();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(29_000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1_000);
    await settle();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  test('reads every 3 s while the sidecar warms up', async () => {
    stubHealth(reply({ warming: true }));
    mod.initSidebarTarget();
    await settle();
    expect(dot().getAttribute('aria-label')).toBe('Service health: warming');

    await vi.advanceTimersByTimeAsync(3_000);
    await settle();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  test('a 404 hides the dot and nothing is read again', async () => {
    stubHealth(reply({ detail: 'Not Found' }, 404));
    mod.initSidebarTarget();
    await settle();

    expect(dot().hidden).toBe(true);
    // The clock stays: only the health half is missing.
    expect(clock().textContent).toBe(mod.clockText(AT));

    await vi.advanceTimersByTimeAsync(10 * 60_000);
    await settle();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test('the clock ticks at the top of the minute', async () => {
    mod.initSidebarTarget();
    await settle();
    const before = clock().textContent;

    await vi.advanceTimersByTimeAsync(60_000);
    expect(clock().textContent).toBe(mod.clockText(AT + 60_000));
    expect(clock().textContent).not.toBe(before);
  });
});

/* ---- teardown ------------------------------------------------------------- */

describe('teardownSidebarTarget', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(AT);
  });

  test('removes the meta line and stops the reads and the clock', async () => {
    mountWithoutChip();
    mod.initSidebarTarget();
    await settle();

    mod.teardownSidebarTarget();
    expect(meta()).toBeNull();

    await vi.advanceTimersByTimeAsync(10 * 60_000);
    await settle();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test('a read still in flight lands nowhere', async () => {
    mountWithoutChip();
    /** @type {(value: unknown) => void} */
    let release = () => {};
    stubHealth(new Promise((resolve) => (release = resolve)));
    mod.initSidebarTarget();

    mod.teardownSidebarTarget();
    release(reply({ results: [{ status: 'error' }] }));
    await settle();

    expect(meta()).toBeNull();
    await vi.advanceTimersByTimeAsync(10 * 60_000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test('drops the chip subscription, and a fresh init builds the block again', async () => {
    mountWithChip();
    mod.initSidebarTarget();
    await settle();
    expect(chip.listeners.size).toBe(1);

    mod.teardownSidebarTarget();
    expect(chip.listeners.size).toBe(0);

    mountWithoutChip();
    mod.initSidebarTarget();
    await settle();
    expect(meta()?.parentElement).toBe(host());
  });
});
