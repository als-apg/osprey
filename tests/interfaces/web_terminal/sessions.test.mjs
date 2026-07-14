// @ts-check
/**
 * Unit tests for the Web Terminal session picker (sessions.js):
 *   npx vitest run tests/interfaces/web_terminal/sessions.test.mjs
 *
 * `initSessionSelector(containerId)` builds the picker button + dropdown
 * shell. The session list is rendered lazily: the picker button's click
 * handler `await`s a private `fetchSessions()` (which GETs `/api/sessions`
 * into module-private `sessionsData`), then `renderSessionList()`, then opens
 * the dropdown. These tests drive that real click path with a stubbed `fetch`
 * and assert on the parsed DOM the list actually produces -- one record per
 * `.session-item`, id round-trips through `data-session-id`, previews are
 * escaped inert, selection closes the dropdown, and an empty result renders
 * the placeholder.
 *
 * Module isolation: `sessionsData` is a module-private `let` that survives
 * across calls within one module instance. `vi.resetModules()` + a fresh
 * dynamic `import()` per test gives each test a never-before-touched instance,
 * so a session list loaded by one test cannot leak into the next.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

/** @typedef {{ session_id: string, last_modified: string, message_count: number, first_message?: string }} SessionRecord */

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/sessions.js')} */
let sessions;

/** A recent ISO timestamp so `relativeTime` stays deterministic-ish ("just now"). */
const NOW_ISO = new Date().toISOString();

/** @type {SessionRecord[]} */
const RECORDS = [
  {
    session_id: 'aaaaaaaa-1111-2222-3333-444444444444',
    last_modified: NOW_ISO,
    message_count: 7,
    first_message: 'Investigate the vacuum pressure spike',
  },
  {
    session_id: 'bbbbbbbb-5555-6666-7777-888888888888',
    last_modified: NOW_ISO,
    message_count: 2,
    first_message: 'Check corrector magnet limits',
  },
];

/** Resolve after the current microtask queue drains (the click handler is async). */
const flush = () => new Promise((r) => setTimeout(r, 0));

/**
 * Install a `fetch` stub whose `/api/sessions` response yields `sessions`.
 * @param {SessionRecord[]} records
 */
function stubSessions(records) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ sessions: records }),
    }))
  );
}

function mountFixture() {
  document.body.innerHTML = `<div id="session-selector"></div>`;
}

/** Open the picker: click `#session-picker-btn` and let the async handler settle. */
async function openPicker() {
  const btn = /** @type {HTMLElement} */ (document.getElementById('session-picker-btn'));
  btn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  await flush();
}

beforeEach(async () => {
  vi.resetModules();
  mountFixture();
  sessions = await import('../../../src/osprey/interfaces/web_terminal/static/js/sessions.js');
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('initSessionSelector: dropdown shell', () => {
  test('builds the picker button and dropdown structure inside the container', () => {
    sessions.initSessionSelector('session-selector');

    const container = /** @type {HTMLElement} */ (document.getElementById('session-selector'));
    expect(container.querySelector('#session-picker-btn')).not.toBeNull();
    const dropdown = container.querySelector('#session-dropdown');
    expect(dropdown).not.toBeNull();
    expect(container.querySelector('#session-dropdown-list')).not.toBeNull();
    // Closed until the picker is opened.
    expect(/** @type {Element} */ (dropdown).classList.contains('open')).toBe(false);
  });
});

describe('renderSessionList: one item per record', () => {
  test('renders exactly one .session-item per fetched record and round-trips each id through data-session-id', async () => {
    stubSessions(RECORDS);
    sessions.initSessionSelector('session-selector');
    await openPicker();

    const dropdown = /** @type {HTMLElement} */ (document.getElementById('session-dropdown'));
    expect(dropdown.classList.contains('open')).toBe(true);

    const items = document.querySelectorAll('.session-item[data-session-id]');
    expect(items.length).toBe(RECORDS.length);

    const ids = Array.from(items).map((el) => /** @type {HTMLElement} */ (el).dataset.sessionId);
    expect(ids).toEqual(RECORDS.map((r) => r.session_id));

    // Each item exposes the header/preview/meta sub-structure the CSS targets.
    const first = /** @type {HTMLElement} */ (items[0]);
    expect(first.querySelector('.session-item-id')?.textContent).toBe(
      RECORDS[0].session_id.slice(0, 8)
    );
    expect(first.querySelector('.session-item-time')).not.toBeNull();
    expect(first.querySelector('.session-item-meta')?.textContent).toBe('7 messages');
  });
});

describe('renderSessionList: preview escaping', () => {
  test('an HTML-bearing first_message renders inert -- no live element, raw markup survives as escaped text', async () => {
    const payload = '<img src=x onerror=alert(1)>';
    stubSessions([
      {
        session_id: 'cccccccc-9999-0000-1111-222222222222',
        last_modified: NOW_ISO,
        message_count: 1,
        first_message: payload,
      },
    ]);
    sessions.initSessionSelector('session-selector');
    await openPicker();

    const item = /** @type {HTMLElement} */ (document.querySelector('.session-item[data-session-id]'));
    const preview = /** @type {HTMLElement} */ (item.querySelector('.session-item-preview'));

    // No live <img> was parsed out of the injected markup.
    expect(preview.querySelector('img[onerror]')).toBeNull();
    expect(preview.querySelector('img')).toBeNull();
    // The markup survives verbatim as text, proving escapeHtml neutered it.
    expect(preview.textContent).toBe(payload);
  });
});

describe('session selection', () => {
  test('clicking a .session-item closes the dropdown', async () => {
    stubSessions(RECORDS);
    sessions.initSessionSelector('session-selector');
    await openPicker();

    const dropdown = /** @type {HTMLElement} */ (document.getElementById('session-dropdown'));
    expect(dropdown.classList.contains('open')).toBe(true);

    const item = /** @type {HTMLElement} */ (document.querySelector('.session-item[data-session-id]'));
    item.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));

    expect(dropdown.classList.contains('open')).toBe(false);
  });
});

describe('renderSessionList: empty state', () => {
  test('an empty sessions list renders the "No previous sessions" placeholder', async () => {
    stubSessions([]);
    sessions.initSessionSelector('session-selector');
    await openPicker();

    const empty = /** @type {HTMLElement} */ (document.querySelector('.session-item.empty'));
    expect(empty).not.toBeNull();
    expect(empty.textContent).toBe('No previous sessions');
    // No real session records rendered alongside the placeholder.
    expect(document.querySelectorAll('.session-item[data-session-id]').length).toBe(0);
  });
});
