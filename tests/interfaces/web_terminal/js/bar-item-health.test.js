/**
 * The system-health bar item, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/js/bar-item-health.test.js
 *
 * The item reads the SYSTEM panel's `/checks` envelope through the terminal's
 * panel proxy and paints the worst outcome as one dot. What is pinned:
 *
 *   - ATTACH POLLS, ONCE PER PAGE. The first placed body starts the poller; a
 *     second body (the customize sheet's preview) joins it rather than
 *     opening a poller of its own.
 *   - DETACH STOPS. Parking the shell in the pool ends the polling: no
 *     further request, however long the page lives.
 *   - THE READING. Errors outrank warnings outrank ok; `skip` counts for
 *     nothing; warming and an unreachable sidecar each have their own word
 *     while a stale report keeps its reading (the card says it is stale);
 *     `text: status` puts the word on the chip, `detail: checks` lists every
 *     check instead of one row per category.
 *
 * These run against the REAL bar-host.js and bar-items.js, with `fetch`
 * stubbed to answer envelopes in the shape the sidecar documents.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const HOST_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js';
const ITEMS_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-items.js';
const HEALTH_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-item-health.js';

/** @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js')} */
let host;
/** @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-items.js')} */
let items;
/** Every URL `fetch` was asked for this test. @type {string[]} */
let requested;
/** What the next `fetch` answers: an envelope, or `null` to fail the request. */
/** @type {(() => unknown) | null} */
let answer;

/**
 * One envelope in the shape `GET /checks` serves: the report's own keys plus
 * the sidecar's `stale` / `warming` / `interval_s`.
 * @param {Array<{name: string, category: string, status: string, message?: string, value?: string}>} results
 * @param {Partial<{warming: boolean, stale: boolean, interval_s: number}>} [extra]
 */
function envelope(results, extra = {}) {
  const counted = results.filter((r) => r.status !== 'skip');
  const ok = counted.filter((r) => r.status === 'ok').length;
  return {
    summary: `${ok}/${counted.length} checks passed`,
    ok,
    warnings: counted.filter((r) => r.status === 'warning').length,
    errors: counted.filter((r) => r.status === 'error').length,
    skips: results.length - counted.length,
    total: results.length,
    results: results.map((r) => ({ message: '', ...r })),
    elapsed_ms: 12,
    deadline_hit: false,
    stale: false,
    warming: false,
    interval_s: 60,
    title: 'System Health',
    ...extra,
  };
}

const HEALTHY = () => envelope([{ name: 'a.one', category: 'a', status: 'ok' }]);

const MIXED = () =>
  envelope([
    { name: 'control_system.connect', category: 'control_system', status: 'ok' },
    { name: 'control_system.read', category: 'control_system', status: 'ok' },
    { name: 'llm.provider', category: 'llm', status: 'warning', message: 'provider cborg: 401' },
    { name: 'llm.model', category: 'llm', status: 'ok', value: 'claude' },
    { name: 'services.archiver', category: 'services', status: 'skip', message: 'no archiver' },
  ]);

beforeEach(async () => {
  requested = [];
  answer = HEALTHY;
  vi.stubGlobal(
    'fetch',
    vi.fn((/** @type {string} */ url) => {
      requested.push(url);
      const reply = answer;
      if (!reply) return Promise.reject(new TypeError('Failed to fetch'));
      return Promise.resolve({ ok: true, status: 200, json: async () => reply() });
    })
  );
  vi.resetModules();
  host = await import(HOST_PATH);
  items = await import(ITEMS_PATH);
  await import(HEALTH_PATH);
});

afterEach(() => {
  items.disposeBarItems();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

/**
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

/** @param {Record<string, unknown>} [options] */
function shellMarkup(options) {
  const stamped = options ? ` data-bar-options='${JSON.stringify(options)}'` : '';
  return `<div class="bar-item" data-bar-item="system-health"${stamped}></div>`;
}

/** @param {string[]} header @param {string[]} status */
function layoutOf(header, status) {
  const item = (/** @type {string} */ type) => ({ type, options: {} });
  return {
    version: 1,
    rev: 0,
    header: header.map(item),
    status: status.map(item),
    header_visible: true,
    status_visible: true,
  };
}

/** Let the pending fetch resolve and its listeners run. */
async function settle() {
  for (let i = 0; i < 8; i += 1) await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

/** @param {string} selector */
function el(selector) {
  const node = document.querySelector(selector);
  if (!(node instanceof HTMLElement)) throw new Error(`no element matched ${selector}`);
  return node;
}

const chip = () => el('[data-bar-item="system-health"] .bar-health');
const dotTone = () => chip().querySelector('.bar-health-dot')?.getAttribute('data-tone');
const chipText = () => /** @type {HTMLElement} */ (chip().querySelector('.bar-health-text'));
const pop = () => el('[data-bar-item="system-health"] .bar-health-pop');
const rows = () =>
  Array.from(pop().querySelectorAll('.bar-health-row')).map((row) => ({
    tone: row.querySelector('.bar-health-dot')?.getAttribute('data-tone'),
    name: row.querySelector('.bar-health-row-name')?.textContent,
    count: row.querySelector('.bar-health-row-count')?.textContent,
    aside: row.querySelector('.bar-health-row-aside')?.textContent,
  }));

describe('system-health item: one shared poll through the SYSTEM panel proxy', () => {
  test('attach asks the panel proxy for the report and seeds a quiet chip', async () => {
    seedDom('', shellMarkup());
    host.hydrate();

    expect(requested).toEqual(['/panel/system-health/checks']);
    expect(dotTone()).toBe('off');
    expect(chip().title).toBe('System health: checking');
    expect(chipText().hidden).toBe(true);

    await settle();
    expect(dotTone()).toBe('ok');
    expect(chip().title).toBe('System health: ok · 1/1 checks passed');
  });

  test('the dot is the worst outcome: errors over warnings over ok, skips uncounted', async () => {
    answer = MIXED;
    vi.useFakeTimers();
    seedDom('', shellMarkup({ text: 'status' }));
    host.hydrate();
    await vi.advanceTimersByTimeAsync(0);

    expect(dotTone()).toBe('warn');
    expect(chipText().hidden).toBe(false);
    expect(chipText().textContent).toBe('1 warning');

    answer = () =>
      envelope([
        { name: 'a.x', category: 'a', status: 'error', message: 'down' },
        { name: 'a.y', category: 'a', status: 'error', message: 'down' },
        { name: 'b.z', category: 'b', status: 'warning' },
      ]);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(dotTone()).toBe('err');
    expect(chipText().textContent).toBe('2 errors');
  });

  test('warming and unreachable read as their own word; stale keeps the report', async () => {
    answer = () => envelope([], { warming: true, stale: true });
    vi.useFakeTimers();
    seedDom('', shellMarkup({ text: 'status' }));
    host.hydrate();
    await vi.advanceTimersByTimeAsync(0);
    expect(dotTone()).toBe('off');
    expect(chipText().textContent).toBe('warming');

    answer = () => envelope([{ name: 'a.x', category: 'a', status: 'ok' }], { stale: true });
    // A warming sidecar is asked again after seconds, not after a full interval.
    await vi.advanceTimersByTimeAsync(3_000);
    expect(dotTone()).toBe('ok');
    expect(chipText().textContent).toBe('ok');
    chip().click();
    expect(pop().querySelector('.bar-health-note')?.textContent).toMatch(/^Data may be stale/);
    chip().click();

    answer = null;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(dotTone()).toBe('err');
    expect(chipText().textContent).toBe('unreachable');
  });

  test('the poll repeats at the cadence the envelope names, and stops on detach', async () => {
    answer = () => envelope([{ name: 'a.x', category: 'a', status: 'ok' }], { interval_s: 30 });
    vi.useFakeTimers();
    seedDom('', shellMarkup());
    host.hydrate();
    await vi.advanceTimersByTimeAsync(0);
    expect(requested).toHaveLength(1);

    await vi.advanceTimersByTimeAsync(30_000);
    expect(requested).toHaveLength(2);

    host.reconcile(layoutOf([], []));
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(120_000);
    expect(requested).toHaveLength(2);

    host.reconcile(layoutOf([], ['system-health']));
    await vi.advanceTimersByTimeAsync(0);
    expect(requested).toHaveLength(3);
  });

  test('a second body joins the poll rather than starting one of its own', async () => {
    seedDom(shellMarkup());
    host.hydrate();
    const preview = items.previewBarItem('system-health', document, 'comfortable');
    expect(preview).not.toBeNull();
    expect(requested).toHaveLength(1);
    preview?.dispose?.();
  });

  test('the header chip names the panel; the status bar chip is the dot alone', async () => {
    seedDom(shellMarkup(), '');
    host.hydrate();
    expect(chip().querySelector('.bar-health-name')?.textContent).toBe('System');

    host.reconcile(layoutOf([], ['system-health']));
    await settle();
    expect(chip().querySelector('.bar-health-name')).toBeNull();
  });
});

describe('system-health item: the card', () => {
  test('one row per category, worst dot, passed/counted and the loudest message', async () => {
    answer = MIXED;
    seedDom('', shellMarkup());
    host.hydrate();
    await settle();

    chip().click();
    expect(pop().hidden).toBe(false);
    expect(pop().querySelector('.bar-health-pop-title')?.textContent).toContain('1 warning');
    expect(pop().querySelector('.bar-health-pop-summary')?.textContent).toBe(
      '3/4 checks passed'
    );
    expect(rows()).toEqual([
      { tone: 'ok', name: 'control system', count: '2/2', aside: '' },
      { tone: 'warn', name: 'llm', count: '1/2', aside: 'provider cborg: 401' },
      { tone: 'off', name: 'services', count: 'skipped', aside: 'no archiver' },
    ]);
    expect(pop().querySelector('.bar-health-note')?.textContent).toMatch(/^Read \d\d:\d\d:\d\d$/);
    const open = Array.from(pop().querySelectorAll('button')).find(
      (b) => b.textContent === 'Open SYSTEM'
    );
    expect(open).toBeDefined();
  });

  test('`detail: checks` lists every check by its title, with its value or message', async () => {
    answer = MIXED;
    seedDom('', shellMarkup({ detail: 'checks' }));
    host.hydrate();
    await settle();

    chip().click();
    expect(rows().map((r) => [r.tone, r.name, r.aside])).toEqual([
      ['ok', 'Connect', ''],
      ['ok', 'Read', ''],
      ['warn', 'Provider', 'provider cborg: 401'],
      ['ok', 'Model', 'claude'],
      ['off', 'Archiver', 'no archiver'],
    ]);
  });

  test('the card closes on Escape, on a click elsewhere, and on the chip again', async () => {
    seedDom('', shellMarkup());
    host.hydrate();
    await settle();

    chip().click();
    expect(pop().hidden).toBe(false);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(pop().hidden).toBe(true);

    chip().click();
    document.body.click();
    expect(pop().hidden).toBe(true);

    chip().click();
    chip().click();
    expect(pop().hidden).toBe(true);
    expect(chip().getAttribute('aria-expanded')).toBe('false');
  });

  test('an unreachable sidecar says so on the card and lists nothing', async () => {
    answer = null;
    seedDom('', shellMarkup());
    host.hydrate();
    await settle();

    chip().click();
    expect(rows()).toEqual([]);
    expect(pop().querySelector('.bar-health-note')?.textContent).toBe(
      'The SYSTEM panel could not be reached.'
    );
  });
});
