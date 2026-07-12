// @ts-check
/**
 * Unit tests for the Web Terminal's hook debug toggle + activity-log viewer
 * (hook-debug.js):
 *   npx vitest run tests/interfaces/web_terminal/hook-debug.test.mjs
 *
 * `initHookDebug()` builds a toggle bar into `#hook-debug-bar` and a
 * collapsible log viewer into `#hook-debug-log-section` (both live in the
 * Safety tab, `#tab-safety`). The log body (`#hook-debug-log-body`) is filled
 * lazily by a private `_loadLogEntries` that fetches
 * `/api/hooks/debug-log?limit=50` and builds a `table.hook-debug-log-table`
 * with one `<tr>` per entry; an empty/failed fetch renders a
 * `.hook-debug-log-empty` node instead. The load is triggered by expanding the
 * viewer -- clicking its header (`#hook-debug-log-toggle`) -- so every test
 * that exercises rendering clicks the header and then flushes the fetch chain.
 *
 * Every log cell is built via `document.createElement` + `.textContent` (never
 * `innerHTML`), so a hostile `detail` string is inert by construction; the
 * hostile-payload test pins that on the parsed DOM.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { initHookDebug } from '../../../src/osprey/interfaces/web_terminal/static/js/hook-debug.js';

/**
 * Build a Response-like object for the stubbed `fetch`. `fetchJSON` only
 * touches `.ok` and `.json()`.
 * @param {unknown} data
 * @param {boolean} [ok]
 * @returns {any}
 */
function jsonResponse(data, ok = true) {
  return {
    ok,
    status: ok ? 200 : 500,
    statusText: ok ? 'OK' : 'Internal Server Error',
    json: async () => data,
  };
}

/**
 * Route the stubbed `fetch` by URL: the debug-log endpoint returns
 * `logPayload`; anything else (e.g. the tab-activate debug-status probe)
 * returns a benign default so unrelated calls never reject.
 * @param {unknown} logPayload
 */
function stubFetch(logPayload) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url) => {
      if (typeof url === 'string' && url.includes('/api/hooks/debug-log')) {
        return jsonResponse(logPayload);
      }
      return jsonResponse({ enabled: false });
    })
  );
}

/** Flush the microtask queue so an awaited `_loadLogEntries` settles. */
function flush() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/** Mount the containers `initHookDebug` looks up by id. */
function mountFixture() {
  document.body.innerHTML = `
    <div id="hook-debug-bar"></div>
    <div id="hook-debug-log-section"></div>
    <div id="tab-safety"></div>
  `;
}

/** Expand the log viewer (fires the lazy `_loadLogEntries`) and flush. */
async function expandLog() {
  const header = /** @type {HTMLElement} */ (
    document.getElementById('hook-debug-log-toggle')
  );
  header.click();
  await flush();
}

function logBodyEl() {
  return /** @type {HTMLElement} */ (
    document.getElementById('hook-debug-log-body')
  );
}

beforeEach(() => {
  mountFixture();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('populated activity log', () => {
  /** @type {Array<Record<string, string>>} */
  const ENTRIES = [
    {
      ts: '2026-07-10T12:00:00.000Z',
      hook: 'PreToolUse',
      tool: 'Bash',
      status: 'allowed',
      detail: 'ran ls -la',
    },
    {
      ts: '2026-07-10T12:00:01.000Z',
      hook: 'PreToolUse',
      tool: 'Write',
      status: 'blocked',
      detail: 'write denied by policy',
    },
  ];

  test('renders one row per entry with status classes mapped from allowed/blocked', async () => {
    stubFetch({ entries: ENTRIES });
    initHookDebug();
    await expandLog();

    const body = logBodyEl();
    const table = body.querySelector('table.hook-debug-log-table');
    expect(table).not.toBeNull();

    const rows = body.querySelectorAll('tbody tr');
    expect(rows.length).toBe(ENTRIES.length);

    // Header columns are present.
    const headers = Array.from(body.querySelectorAll('thead th')).map(
      (th) => th.textContent
    );
    expect(headers).toEqual(['Time', 'Hook', 'Tool', 'Status', 'Detail']);

    // The hook event name surfaces in a rendered row.
    const hookCells = Array.from(rows).map(
      (tr) => tr.querySelectorAll('td')[1]?.textContent
    );
    expect(hookCells).toContain('PreToolUse');

    // status -> class mapping.
    const okCell = body.querySelector('td.status-ok');
    const blockedCell = body.querySelector('td.status-blocked');
    expect(okCell?.textContent).toBe('allowed');
    expect(blockedCell?.textContent).toBe('blocked');
    // Exactly one of each, matching the two seeded rows.
    expect(body.querySelectorAll('td.status-ok').length).toBe(1);
    expect(body.querySelectorAll('td.status-blocked').length).toBe(1);
  });
});

describe('empty activity log', () => {
  test('an empty entries array renders the "No entries" placeholder, not a table', async () => {
    stubFetch({ entries: [] });
    initHookDebug();
    await expandLog();

    const body = logBodyEl();
    expect(body.querySelector('table.hook-debug-log-table')).toBeNull();

    const empty = body.querySelector('.hook-debug-log-empty');
    expect(empty).not.toBeNull();
    expect(empty?.textContent).toBe('No entries');
  });
});

describe('hostile payload inertness', () => {
  const HOSTILE = '<img src=x onerror=alert(1)>';

  test('a malicious detail string is rendered as inert escaped text, never a live element', async () => {
    stubFetch({
      entries: [
        {
          ts: '2026-07-10T12:00:00.000Z',
          hook: 'PreToolUse',
          tool: 'Bash',
          status: 'allowed',
          detail: HOSTILE,
        },
      ],
    });
    initHookDebug();
    await expandLog();

    const body = logBodyEl();

    // Canonical inertness shape: no live <img> was parsed into the DOM.
    expect(body.querySelector('img[onerror]')).toBeNull();
    expect(document.querySelector('img[onerror]')).toBeNull();

    // The raw markup survives verbatim as the cell's text content.
    const detailCell = body.querySelector('td.log-detail');
    expect(detailCell).not.toBeNull();
    expect(detailCell?.textContent).toBe(HOSTILE);
    // ...and it is genuinely stored as escaped text, not parsed children.
    expect(detailCell?.children.length).toBe(0);
    expect(body.innerHTML).toContain('&lt;img');
  });
});

describe('failed log fetch', () => {
  test('a rejected debug-log fetch renders the "Failed to load log" placeholder', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (typeof url === 'string' && url.includes('/api/hooks/debug-log')) {
          return jsonResponse(null, false); // res.ok === false -> fetchJSON throws
        }
        return jsonResponse({ enabled: false });
      })
    );
    // Silence the module's console.error for the expected failure path.
    vi.spyOn(console, 'error').mockImplementation(() => {});

    initHookDebug();
    await expandLog();

    const body = logBodyEl();
    expect(body.querySelector('table.hook-debug-log-table')).toBeNull();
    const placeholder = body.querySelector('.hook-debug-log-empty');
    expect(placeholder?.textContent).toBe('Failed to load log');
  });
});
