// @ts-check
/**
 * The sidebar's control-system block (sidebar-target.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-target.test.mjs
 *
 * The block dresses the control-target chip mounted in `#sidebar-target` as
 * one row — glyph, the chip's own name and writes pill, the SYSTEM panel's
 * health dot, caret — with the decoration aria-hidden and the chip's own words
 * read from `targetReading()` for its tooltip text. There is no caption, no
 * clock and no meta line. Where no chip mounts (the picker is off), the health
 * dot stands alone in the block.
 *
 * Seams: control-target-chip.js is mocked (the chip element, its subscribers
 * and its reading are what this module reads; the chip's own suite pins how it
 * derives them) and `fetch` is stubbed per test. The SYSTEM panel is served
 * unless a test says otherwise, through the real panel-availability.js. Module
 * state is isolated by vi.resetModules() + dynamic import per test and the
 * module's own teardown.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { byId, qs } from '../_support/dom.mjs';

const MODULE = '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-target.js';
const AVAILABILITY = '../../../src/osprey/interfaces/web_terminal/static/js/panel-availability.js';

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
/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/panel-availability.js')} */
let availability;
/** @type {ReturnType<typeof vi.fn>} */
let fetchMock;

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
          <button class="control-target-chip" type="button" aria-haspopup="dialog"
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
const dot = () => qs(document, '.sidebar-health');
const healthEl = () => document.getElementById('sidebar-target-health');
const chipEl = () => /** @type {HTMLElement} */ (chip.el);

beforeEach(async () => {
  vi.resetModules();
  chip.el = null;
  chip.reading = null;
  chip.listeners.clear();
  Reflect.deleteProperty(window, '__OSPREY_PREFIX__');
  stubHealth(reply({ results: [{ status: 'ok' }], interval_s: 30 }));
  mod = await import(MODULE);
  availability = await import(AVAILABILITY);
  availability.setPanelServed('system-health', true);
});

afterEach(() => {
  mod.teardownSidebarTarget();
  vi.useRealTimers();
  vi.unstubAllGlobals();
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

/* ---- the block with a chip ------------------------------------------------ */

describe('with the chip mounted', () => {
  beforeEach(() => {
    mountWithChip();
  });

  test('dresses the chip as one row: glyph first, health dot and caret last', async () => {
    mod.initSidebarTarget();
    await settle();
    const button = chipEl();
    const children = [...button.children];

    const glyph = children[0];
    const [health, caret] = children.slice(-2);
    expect(glyph.classList.contains('sidebar-target-glyph')).toBe(true);
    expect(health).toBe(dot());
    expect(caret.classList.contains('sidebar-target-caret')).toBe(true);
    for (const el of [glyph, caret]) expect(el.getAttribute('aria-hidden')).toBe('true');

    // No caption and no clock: the row is the glyph, the chip's words, the
    // health dot and the caret.
    expect(button.querySelector('.sidebar-target-eyebrow')).toBeNull();
    expect(button.querySelector('.sidebar-target-clock')).toBeNull();
    expect(button.textContent).not.toContain('Control system');

    // The chip's own words and name are the chip's.
    expect(qs(button, '.ctc-short').textContent).toBe('Simulator');
    expect(qs(button, '.ctc-state').textContent).toBe('writes on');
    expect(button.getAttribute('aria-label')).toBe('Control target: Simulator, writes on');
    expect(host().dataset.target).toBe('chip');
  });

  test('describes the chip with the health dot', async () => {
    mod.initSidebarTarget();
    await settle();

    expect(healthEl()?.parentElement).toBe(chipEl());
    expect(chipEl().getAttribute('aria-describedby')).toBe('sidebar-target-health');
  });

  test("the tooltip text repeats the chip's reading", async () => {
    mod.initSidebarTarget();
    await settle();
    expect(chipEl().dataset.tip).toBe('Simulator');
    expect(chipEl().dataset.tipDetail).toBe('Control system: Simulator, writes on');

    // A chip render reaches the tooltip through the chip's subscribers.
    chip.reading = { name: 'Live', state: 'switching…', kind: 'real', word: 'on' };
    for (const fn of chip.listeners) fn();
    expect(chipEl().dataset.tip).toBe('Live');
    expect(chipEl().dataset.tipDetail).toBe('Control system: Live, switching…');

    // Nothing read yet: no tooltip rather than a half one.
    chip.reading = null;
    for (const fn of chip.listeners) fn();
    expect(chipEl().dataset.tip).toBeUndefined();
    expect(chipEl().dataset.tipDetail).toBeUndefined();
  });

  test('the tooltip text stays while the popover is open: the tooltip layer steps aside', async () => {
    mod.initSidebarTarget();
    await settle();

    chipEl().setAttribute('aria-expanded', 'true');
    await settle();
    expect(chipEl().dataset.tip).toBe('Simulator');
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
    expect(healthEl()?.parentElement).toBe(host());
    expect(host().dataset.target).toBe('none');
  });
});

/* ---- the block without a chip --------------------------------------------- */

describe('without a chip', () => {
  beforeEach(() => {
    mountWithoutChip();
  });

  test('the health dot stands alone in the block', async () => {
    mod.initSidebarTarget();
    await settle();

    expect(healthEl()?.parentElement).toBe(host());
    expect(host().dataset.target).toBe('none');
    expect(document.querySelector('[aria-describedby]')).toBeNull();
    expect(dot().classList.contains('ok')).toBe(true);
    expect(dot().getAttribute('aria-label')).toBe('Service health: ok');
  });

  test('a second init adds nothing', async () => {
    mod.initSidebarTarget();
    mod.initSidebarTarget();
    await settle();
    expect(document.querySelectorAll('#sidebar-target-health')).toHaveLength(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test('a page without the block is left alone', async () => {
    document.body.innerHTML = '<div id="elsewhere"></div>';
    mod.initSidebarTarget();
    await settle();
    expect(healthEl()).toBeNull();
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

  test('a 404 from a served panel reads as unreachable and the reads go on', async () => {
    stubHealth(reply({ detail: 'Not Found' }, 404));
    mountWithoutChip();
    mod.initSidebarTarget();
    await settle();

    expect(dot().hidden).toBe(false);
    expect(dot().getAttribute('aria-label')).toBe('Service health: unreachable');

    await vi.advanceTimersByTimeAsync(60_000);
    await settle();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

/* ---- following the SYSTEM panel ------------------------------------------- */

describe('while the SYSTEM panel is served', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    availability.setPanelServed('system-health', false);
    mountWithoutChip();
  });

  test('absent: the dot is hidden and nothing is ever read', async () => {
    mod.initSidebarTarget();
    await settle();

    expect(dot().hidden).toBe(true);
    await vi.advanceTimersByTimeAsync(10 * 60_000);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  test('appearing shows the dot and starts the reads, without a reload', async () => {
    mod.initSidebarTarget();
    await settle();

    availability.setPanelServed('system-health', true);
    await settle();

    expect(dot().hidden).toBe(false);
    expect(dot().classList.contains('ok')).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(30_000);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  test('leaving hides the dot and stops the reads, a read in flight included', async () => {
    availability.setPanelServed('system-health', true);
    mod.initSidebarTarget();
    await settle();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    /** @type {(answer: unknown) => void} */
    let answer = () => {};
    stubHealth(() => new Promise((resolve) => (answer = resolve)));
    await vi.advanceTimersByTimeAsync(30_000);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    availability.setPanelServed('system-health', false);
    answer(reply({ results: [{ status: 'error' }], interval_s: 30 }));
    await settle();

    expect(dot().hidden).toBe(true);
    expect(dot().classList.contains('err')).toBe(false);
    await vi.advanceTimersByTimeAsync(10 * 60_000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test('coming back starts from checking, with one chain of reads', async () => {
    availability.setPanelServed('system-health', true);
    mod.initSidebarTarget();
    await settle();
    availability.setPanelServed('system-health', false);
    availability.setPanelServed('system-health', true);
    await settle();

    fetchMock.mockClear();
    await vi.advanceTimersByTimeAsync(30_000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test('after teardown a change of the panel reaches nothing', async () => {
    mod.initSidebarTarget();
    mod.teardownSidebarTarget();

    availability.setPanelServed('system-health', true);
    await settle();

    expect(fetchMock).not.toHaveBeenCalled();
    expect(healthEl()).toBeNull();
  });
});

/* ---- teardown ------------------------------------------------------------- */

describe('teardownSidebarTarget', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  test('removes the health dot and stops the reads', async () => {
    mountWithoutChip();
    mod.initSidebarTarget();
    await settle();

    mod.teardownSidebarTarget();
    expect(healthEl()).toBeNull();

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

    expect(healthEl()).toBeNull();
    await vi.advanceTimersByTimeAsync(10 * 60_000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test('takes the decoration off the chip and drops its subscription', async () => {
    mountWithChip();
    mod.initSidebarTarget();
    await settle();
    expect(chip.listeners.size).toBe(1);

    mod.teardownSidebarTarget();
    expect(chip.listeners.size).toBe(0);
    expect(chipEl().querySelector('.sidebar-target-glyph, .sidebar-target-caret')).toBeNull();
    expect(healthEl()).toBeNull();

    // A fresh init on the same chip dresses it once, not twice.
    mod.initSidebarTarget();
    await settle();
    expect(chipEl().querySelectorAll('.sidebar-target-glyph')).toHaveLength(1);
  });
});
