// @ts-check
/**
 * Unit tests for the System Health panel chrome (chrome.js), and the
 * dashboard wired to it (dashboard.js on the shipped index.html):
 *   npx vitest run tests/interfaces/health/chrome.test.mjs
 */

import { readFileSync } from 'node:fs';
import { cwd } from 'node:process';
import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  FILTERS,
  MENU_ENTRY,
  MENU_ID,
  catCount,
  catLabel,
  cliCommands,
  countsText,
  initFilter,
  initScanButton,
  menuItems,
  renderCategories,
  scanTitle,
  summaryText,
} from '../../../src/osprey/interfaces/health/static/js/chrome.js';

const INDEX_HTML = `${cwd()}/src/osprey/interfaces/health/static/index.html`;
const DASHBOARD = '../../../src/osprey/interfaces/health/static/js/dashboard.js';

/** @param {string} name @param {string} category @param {string} status @param {object} [extra] */
const check = (name, category, status, extra = {}) => ({
  name,
  category,
  status,
  message: name + ' says ' + status,
  ...extra,
});

describe('filter', () => {
  test('builds All / OK / Warn / Err / Skip with counts, and marks the active one', () => {
    const seg = document.createElement('div');
    const picked = /** @type {(string|null)[]} */ ([]);
    const filter = initFilter(seg, (s) => picked.push(s));
    const buttons = Array.from(seg.querySelectorAll('button'));
    expect(buttons.map((b) => b.firstChild?.textContent)).toEqual(FILTERS.map((f) => f.label));
    expect(buttons[0].getAttribute('aria-pressed')).toBe('true');

    filter.update({ total: 9, ok: 5, warnings: 2, errors: 1, skips: 1 }, 'warning');
    expect(buttons.map((b) => b.querySelector('.health-seg-count')?.textContent)).toEqual([
      '9',
      '5',
      '2',
      '1',
      '1',
    ]);
    expect(buttons.map((b) => b.getAttribute('aria-pressed'))).toEqual([
      'false',
      'false',
      'true',
      'false',
      'false',
    ]);

    buttons[3].click();
    buttons[0].click();
    expect(picked).toEqual(['error', null]);
  });
});

describe('scan button', () => {
  test('is a refresh icon named Scan now whose tooltip carries the countdown', () => {
    const button = document.createElement('button');
    const onScan = vi.fn();
    const scan = initScanButton(button, onScan);
    expect(button.querySelector('svg')).not.toBeNull();
    expect(button.getAttribute('aria-label')).toBe('Scan now');
    expect(button.title).toBe('Scan now');

    scan.setCountdown(57);
    expect(button.title).toBe('Scan now (next automatic scan in 57 s)');
    expect(button.getAttribute('aria-label')).toBe('Scan now');

    button.click();
    expect(onScan).toHaveBeenCalledOnce();
  });

  test('the title never shows a negative countdown', () => {
    expect(scanTitle(-1)).toBe('Scan now (next automatic scan in 0 s)');
    expect(scanTitle(null)).toBe('Scan now');
  });
});

describe('categories', () => {
  test('labels and counts', () => {
    expect(catLabel('file_system')).toBe('File system');
    expect(
      catCount({
        cat: 'a',
        checks: [check('x', 'a', 'ok'), check('y', 'a', 'error')],
        onDemand: false,
      })
    ).toBe('1/2');
    expect(catCount({ cat: 'b', checks: [check('x', 'b', 'skip')], onDemand: true })).toBe(
      'on demand'
    );
  });

  test('one row per group with its worst-status dot; a press selects it', () => {
    const list = document.createElement('div');
    const onSelect = vi.fn();
    renderCategories(
      list,
      [
        {
          cat: 'configuration',
          checks: [check('a', 'configuration', 'ok'), check('b', 'configuration', 'warning')],
          onDemand: false,
        },
        { cat: 'model_chat', checks: [check('m', 'model_chat', 'skip')], onDemand: true },
      ],
      'model_chat',
      onSelect
    );
    const rows = Array.from(list.querySelectorAll('button.osprey-list-row'));
    expect(rows.map((r) => r.querySelector('.osprey-list-title')?.textContent)).toEqual([
      'Configuration',
      'Model chat',
    ]);
    expect(rows[0].querySelector('.dot')?.classList.contains('s-wn')).toBe(true);
    expect(rows.map((r) => r.getAttribute('aria-selected'))).toEqual(['false', 'true']);
    rows[0].dispatchEvent(new MouseEvent('click'));
    expect(onSelect).toHaveBeenCalledWith('configuration');
  });
});

describe('menu and clipboard texts', () => {
  test('the three entries, their disabled states and the collapse label', () => {
    const items = menuItems({ hasData: true, hasDetails: true, allOpen: false, hasCli: false });
    expect(items.map((i) => i.id)).toEqual([MENU_ENTRY.expand, MENU_ENTRY.summary, MENU_ENTRY.cli]);
    expect(items.map((i) => i.label)).toEqual([
      'Expand all details',
      'Copy summary',
      'Copy on-demand CLI commands',
    ]);
    expect(items[2].disabled).toBe(true);
    expect(
      menuItems({ hasData: true, hasDetails: true, allOpen: true, hasCli: true })[0].label
    ).toBe('Collapse all details');
    expect(
      menuItems({ hasData: false, hasDetails: false, allOpen: false, hasCli: false }).every(
        (i) => i.disabled
      )
    ).toBe(true);
  });

  test('summary lists the summary line, the notes, then errors before warnings', () => {
    const text = summaryText({
      title: 'System Health',
      summary: '1/3 checks passed (1 warning, 1 error)',
      total: 3,
      ok: 1,
      warnings: 1,
      errors: 1,
      skips: 0,
      stale: true,
      results: [
        check('providers.cborg', 'providers', 'warning'),
        check('file_system.disk', 'file_system', 'ok'),
        check('control_system.write', 'control_system', 'error'),
      ],
    });
    const lines = text.split('\n');
    expect(lines[0]).toBe('System Health: 1/3 checks passed (1 warning, 1 error)');
    expect(lines[1]).toBe('Data may be stale.');
    expect(lines[3].startsWith('Error: Control system')).toBe(true);
    expect(lines[4].startsWith('Warning: Providers')).toBe(true);
    expect(text).not.toContain('disk');
  });

  test('CLI commands cover only the on-demand groups', () => {
    expect(
      cliCommands([
        { cat: 'web', checks: [], onDemand: false },
        { cat: 'model_chat', checks: [], onDemand: true },
        { cat: 'claude_cli_pinned', checks: [], onDemand: true },
      ])
    ).toBe(
      'osprey health --full --category model_chat\nosprey health --full --category claude_cli_pinned'
    );
    expect(countsText({ total: 3, ok: 3, warnings: 0, errors: 0, skips: 0 })).toBe('');
    expect(countsText({ total: 9, ok: 2, warnings: 2, errors: 1, skips: 4 })).toBe(
      '2 warnings, 1 error, 4 skipped'
    );
  });
});

// -- the dashboard on the shipped markup ------------------------------------------

const RESULTS = [
  check('configuration.a', 'configuration', 'ok'),
  check('configuration.b', 'configuration', 'warning'),
  check('file_system.c', 'file_system', 'ok', { value: '12 GB' }),
  check('control_system.d', 'control_system', 'error', { details: 'IOC offline' }),
  {
    name: 'model_chat',
    category: 'model_chat',
    status: 'skip',
    message: 'on_demand category; run with --full',
  },
];

const ENVELOPE = {
  summary: '2/5 checks passed (1 warning, 1 error, 1 skipped)',
  ok: 2,
  warnings: 1,
  errors: 1,
  skips: 1,
  total: 5,
  elapsed_ms: 10,
  deadline_hit: false,
  results: RESULTS,
  stale: false,
  warming: false,
  interval_s: 60,
  title: 'System Health',
};

describe('dashboard', () => {
  /** @type {any[][]} */
  let contributions = [];
  /** @type {import('vitest').Mock} */
  let writeText;

  /** @param {string} id */
  const byId = (id) => /** @type {HTMLElement} */ (document.getElementById(id));
  const cards = () =>
    Array.from(byId('grid').querySelectorAll('.card')).map(
      (c) => /** @type {HTMLElement} */ (c).dataset.category
    );
  const flush = async () => {
    for (let i = 0; i < 20; i++) await Promise.resolve();
  };
  /** @param {string} value */
  const menuAction = (value) =>
    window.dispatchEvent(
      new MessageEvent('message', {
        data: { type: 'canopy-host-header-action', id: MENU_ID, value },
        origin: window.location.origin,
      })
    );
  /** @param {string} status */
  const seg = (status) =>
    /** @type {HTMLElement} */ (byId('filter').querySelector(`[data-status="${status}"]`));

  /** @param {object} envelope */
  async function boot(envelope = ENVELOPE) {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: true, status: 200, json: async () => envelope }))
    );
    await import(/* @vite-ignore */ DASHBOARD);
    await flush();
  }

  beforeEach(() => {
    vi.resetModules();
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'setTimeout', 'clearTimeout'] });
    contributions = [];
    const html = readFileSync(INDEX_HTML, 'utf-8');
    const body = html.match(/<body>([\s\S]*)<\/body>/);
    if (!body) throw new Error('could not extract <body> from index.html');
    document.body.innerHTML = body[1].replace(/<script[\s\S]*?<\/script>/g, '');
    document.body.classList.add('embedded');
    Object.defineProperty(window, 'parent', {
      configurable: true,
      value: {
        /** @param {any} message */
        postMessage(message) {
          if (message?.type === 'canopy-host-header-contribution')
            contributions.push(message.items);
        },
      },
    });
    writeText = vi.fn(async () => {});
    vi.stubGlobal('navigator', { ...navigator, clipboard: { writeText } });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    document.body.innerHTML = '';
    document.body.classList.remove('embedded');
  });

  test('the old header, footer countdown and orbiting beam are gone', () => {
    expect(document.querySelector('.hdr, .ftr, #cd, #beam, .gauge-orbit, #ttl')).toBeNull();
    expect(document.querySelector('.osprey-panel-toolbar #sidebar-toggle')).not.toBeNull();
  });

  test('shows the centred loading state until the first report arrives', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => new Promise(() => {}))
    );
    await import(/* @vite-ignore */ DASHBOARD);
    expect(byId('state').querySelector('.osprey-state-message')?.textContent).toBe('Scanning…');
    expect(byId('grid').hidden).toBe(true);
  });

  test('an unreachable engine shows the state with Retry, and Retry scans again', async () => {
    const fetchMock = vi.fn(async () => ({ ok: false, status: 503, json: async () => ({}) }));
    vi.stubGlobal('fetch', fetchMock);
    await import(/* @vite-ignore */ DASHBOARD);
    await flush();
    const state = byId('state');
    expect(state.querySelector('.osprey-state-message')?.textContent).toBe(
      'Cannot reach the health checks'
    );
    /** @type {HTMLElement} */ (state.querySelector('button')).click();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  test('the filter seg changes the shown cards and its counts follow the report', async () => {
    await boot();
    expect(cards()).toEqual(['configuration', 'file_system', 'control_system', 'model_chat']);
    expect(seg('all').querySelector('.health-seg-count')?.textContent).toBe('5');

    seg('error').click();
    expect(cards()).toEqual(['control_system']);
    expect(seg('error').getAttribute('aria-pressed')).toBe('true');
    expect(seg('all').getAttribute('aria-pressed')).toBe('false');

    seg('warning').click();
    expect(cards()).toEqual(['configuration']);
    // Only the matching rows show inside a card.
    expect(byId('grid').querySelectorAll('.ck').length).toBe(1);

    seg('all').click();
    expect(cards()).toHaveLength(4);
  });

  test('the sidebar lists the categories; a hidden one brings the filter back to All', async () => {
    await boot();
    const rows = Array.from(byId('category-list').querySelectorAll('.osprey-list-row'));
    expect(rows.map((r) => r.querySelector('.health-cat-count')?.textContent)).toEqual([
      '1/2',
      '1/1',
      '0/1',
      'on demand',
    ]);
    seg('error').click();
    /** @type {HTMLElement} */ (
      byId('category-list').querySelector('[data-category="file_system"]')
    ).click();
    expect(cards()).toContain('file_system');
    expect(seg('all').getAttribute('aria-pressed')).toBe('true');
    expect(
      byId('category-list')
        .querySelector('[data-category="file_system"]')
        ?.getAttribute('aria-selected')
    ).toBe('true');
  });

  test('contributes one v3 menu, and its entries fire', async () => {
    await boot();
    const last = contributions[contributions.length - 1];
    expect(last).toHaveLength(1);
    expect(last[0].kind).toBe('menu');
    expect(last[0].id).toBe(MENU_ID);
    expect(last[0].items.map((/** @type {any} */ i) => i.label)).toEqual([
      'Expand all details',
      'Copy summary',
      'Copy on-demand CLI commands',
    ]);

    menuAction(MENU_ENTRY.expand);
    const boxes = Array.from(byId('grid').querySelectorAll('.dt-box'));
    expect(boxes.length).toBeGreaterThan(0);
    expect(boxes.every((b) => b.classList.contains('show'))).toBe(true);
    const now = contributions[contributions.length - 1];
    expect(now[0].items[0].label).toBe('Collapse all details');

    menuAction(MENU_ENTRY.expand);
    expect(
      Array.from(byId('grid').querySelectorAll('.dt-box')).some((b) => b.classList.contains('show'))
    ).toBe(false);

    menuAction(MENU_ENTRY.summary);
    await flush();
    expect(writeText).toHaveBeenLastCalledWith(expect.stringContaining(ENVELOPE.summary));

    menuAction(MENU_ENTRY.cli);
    await flush();
    expect(writeText).toHaveBeenLastCalledWith('osprey health --full --category model_chat');
  });

  test('the Scan now tooltip counts down to the next automatic scan', async () => {
    await boot();
    const button = byId('scan-now');
    expect(button.title).toBe('Scan now (next automatic scan in 60 s)');
    vi.advanceTimersByTime(3000);
    expect(button.title).toBe('Scan now (next automatic scan in 57 s)');
  });

  test('warming shows the first-scan state instead of the report', async () => {
    await boot({ ...ENVELOPE, warming: true, results: [] });
    expect(byId('state').querySelector('.osprey-state-message')?.textContent).toBe(
      'First scan in progress…'
    );
    expect(byId('hero').hidden).toBe(true);
  });
});
