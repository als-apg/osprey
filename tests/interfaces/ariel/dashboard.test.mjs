// @ts-check
/**
 * Unit tests for the ARIEL Status drawer module (dashboard.js): Status loads
 * when the store's `drawer` becomes 'status', auto-refreshes only while that
 * drawer is open, stops when it closes, and renders only the modules the
 * server reports (no hard-coded future row).
 *
 *   npx vitest run tests/interfaces/ariel/dashboard.test.mjs
 *
 * dashboard.js never opens or closes the drawer element itself (layout.js
 * does); these tests therefore drive the store's `drawer` key directly.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/api.js', async (importOriginal) => {
  const actual = /** @type {any} */ (await importOriginal());
  return {
    ...actual,
    statusApi: { ...actual.statusApi, get: vi.fn() },
  };
});

const API_PATH = '../../../src/osprey/interfaces/ariel/static/js/api.js';
const STORE_PATH = '../../../src/osprey/interfaces/ariel/static/js/store.js';
const DASHBOARD_PATH = '../../../src/osprey/interfaces/ariel/static/js/dashboard.js';

/** @type {typeof import('../../../src/osprey/interfaces/ariel/static/js/store.js')} */
let store;
/** @type {typeof import('../../../src/osprey/interfaces/ariel/static/js/dashboard.js')} */
let dashboard;
/** @type {any} */
let getStatus;
/** @type {{destroy: () => void} | null} */
let api = null;

const STATUS = {
  healthy: true,
  database_connected: true,
  database_uri: 'postgresql://ariel',
  entry_count: 1234,
  last_ingestion: null,
  active_embedding_model: 'nomic',
  enabled_search_modules: ['keyword'],
  enabled_enhancement_modules: [],
  embedding_tables: [],
  errors: [],
};

function mountFixture() {
  document.body.innerHTML = `
    <div id="status-drawer"><button id="refresh-status"></button><div id="status-content"></div></div>
  `;
}

/** Let the awaited statusApi.get() resolve and the render land. */
async function flush() {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

/** @param {{drawer?: 'status'|'settings'|null}} [opts] */
async function boot({ drawer = null } = {}) {
  mountFixture();
  vi.resetModules();
  store = await import(STORE_PATH);
  const apiMod = /** @type {any} */ (await import(API_PATH));
  getStatus = vi.mocked(apiMod.statusApi.get);
  getStatus.mockReset();
  getStatus.mockResolvedValue(STATUS);
  if (drawer !== null) store.setState({ drawer });
  dashboard = await import(DASHBOARD_PATH);
  api = dashboard.initDashboard();
}

const content = () => /** @type {HTMLElement} */ (document.getElementById('status-content'));

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  api?.destroy();
  api = null;
  dashboard?.stopAutoRefresh();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('status drawer: open and close drive loading and refresh', () => {
  test('nothing loads while the drawer is closed', async () => {
    await boot();
    await vi.advanceTimersByTimeAsync(120000);
    expect(getStatus).not.toHaveBeenCalled();
  });

  test('opening the drawer loads status once and renders it', async () => {
    await boot();
    store.setState({ drawer: 'status' });
    await flush();
    expect(getStatus).toHaveBeenCalledTimes(1);
    expect(content().textContent).toContain('Logbook is healthy');
  });

  test('while open, status re-fetches on every 30 s tick', async () => {
    await boot();
    store.setState({ drawer: 'status' });
    await vi.advanceTimersByTimeAsync(30000);
    expect(getStatus).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(30000);
    expect(getStatus).toHaveBeenCalledTimes(3);
  });

  test('closing the drawer stops the refresh', async () => {
    await boot();
    store.setState({ drawer: 'status' });
    await vi.advanceTimersByTimeAsync(30000);
    expect(getStatus).toHaveBeenCalledTimes(2);
    store.setState({ drawer: null });
    await vi.advanceTimersByTimeAsync(120000);
    expect(getStatus).toHaveBeenCalledTimes(2);
  });

  test('switching to the Settings drawer stops the refresh', async () => {
    await boot();
    store.setState({ drawer: 'status' });
    await flush();
    store.setState({ drawer: 'settings' });
    await vi.advanceTimersByTimeAsync(120000);
    expect(getStatus).toHaveBeenCalledTimes(1);
  });

  test('a drawer already open at init loads and refreshes', async () => {
    await boot({ drawer: 'status' });
    await flush();
    expect(getStatus).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(30000);
    expect(getStatus).toHaveBeenCalledTimes(2);
  });

  test('reopening restarts a single refresh, not a second one', async () => {
    await boot();
    store.setState({ drawer: 'status' });
    store.setState({ drawer: null });
    store.setState({ drawer: 'status' });
    await flush();
    expect(getStatus).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(30000);
    expect(getStatus).toHaveBeenCalledTimes(3);
  });

  test('destroy stops the refresh and ignores later drawer changes', async () => {
    await boot();
    store.setState({ drawer: 'status' });
    await flush();
    api?.destroy();
    api = null;
    await vi.advanceTimersByTimeAsync(120000);
    store.setState({ drawer: null });
    store.setState({ drawer: 'status' });
    await flush();
    expect(getStatus).toHaveBeenCalledTimes(1);
  });

  test('the refresh button loads status on demand', async () => {
    await boot();
    /** @type {HTMLElement} */ (document.getElementById('refresh-status')).click();
    await flush();
    expect(getStatus).toHaveBeenCalledTimes(1);
  });
});

describe('status drawer: module rows', () => {
  test('renders no Vision Search row and no (Future) label', async () => {
    await boot({ drawer: 'status' });
    await flush();
    const text = content().textContent || '';
    expect(text).toContain('Keyword Search');
    expect(text).not.toContain('Vision Search');
    expect(text).not.toContain('(Future)');
  });

  test('a module the server does not enable still renders Disabled', async () => {
    await boot({ drawer: 'status' });
    await flush();
    const text = content().textContent || '';
    expect(text).toContain('Semantic Search');
    expect(text).toContain('Disabled');
    expect(text).toContain('Enabled');
  });

  test('a failed load renders the error state', async () => {
    await boot();
    getStatus.mockRejectedValueOnce(new Error('boom'));
    vi.spyOn(console, 'error').mockImplementation(() => {});
    store.setState({ drawer: 'status' });
    await flush();
    expect(content().textContent).toContain('Failed to Load Status');
  });
});
