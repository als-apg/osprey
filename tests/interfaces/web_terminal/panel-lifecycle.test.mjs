// @vitest-environment jsdom
/**
 * Contract tests for panel-lifecycle.js's sidecar start seam: the start
 * outcome every follower hears (onPanelStart) as a start is followed, and the
 * restart of a running sidecar (restartPanelServer). The rest of the module is
 * pinned through panel-manager.test.mjs and panel-sse.test.mjs. Run:
 *   npx vitest run tests/interfaces/web_terminal/panel-lifecycle.test.mjs
 */
import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const m = vi.hoisted(() => ({
  fetchJSON: vi.fn(),
  apiRequest: vi.fn(),
  startHealthPolling: vi.fn(),
  stopHealthPolling: vi.fn(),
  setEntryEnabled: vi.fn(),
  setEntryStatus: vi.fn(),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/api.js', () => ({
  fetchJSON: m.fetchJSON,
  apiRequest: m.apiRequest,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-health.js', () => ({
  startHealthPolling: m.startHealthPolling,
  stopHealthPolling: m.stopHealthPolling,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js', () => ({
  setKnownServicePanels: vi.fn(),
  setPanelReachable: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-menu-policy.js', () => ({
  railOptions: vi.fn(() => ({})),
  RAIL_MENU_HINT: 'right-click for actions',
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-rail.js', () => ({
  clockTime: vi.fn(() => '12:00'),
  createRail: vi.fn(),
  addEntry: vi.fn(),
  placeEntry: vi.fn(),
  setActive: vi.fn(),
  setEntryEnabled: m.setEntryEnabled,
  setEntryPoppedOut: vi.fn(),
  setEntryReachable: vi.fn(),
  setEntryStatus: m.setEntryStatus,
}));

const lifecycle = await import(`${JS}/panel-lifecycle.js`);

const PANEL = {
  id: 'jupyter',
  label: 'Jupyter',
  configEndpoint: '/api/jupyter-server',
  healthEndpoint: '/panel/jupyter/api/status',
  startEndpoint: '/api/panels/jupyter/start',
};
const STARTING = { url: null, available: false, state: 'starting', message: 'JUPYTER is starting' };
const RUNNING = { url: '/panel/jupyter', available: true, state: 'running', message: null };
const FAILED = {
  url: null,
  available: false,
  state: 'failed',
  message: 'JUPYTER failed to start: boom',
};

/** @type {Record<string, any>} */
let panelState;
/** @type {{ panelId: string, start: any }[]} */
let heard;
/** @type {() => void} */
let unfollow;

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
  panelState = { jupyter: lifecycle.freshPanelState() };
  lifecycle.initPanelLifecycle({
    panelState,
    memberSet: new Set(['jupyter']),
    getRailEl: () => document.body,
    getActive: () => null,
    ensureActive: vi.fn(),
    activate: vi.fn(),
    navigatePending: vi.fn(),
  });
  heard = [];
  unfollow = lifecycle.onPanelStart((/** @type {string} */ panelId, /** @type {any} */ start) =>
    heard.push({ panelId, start })
  );
});

afterEach(() => {
  unfollow();
  vi.useRealTimers();
  vi.clearAllMocks();
});

const states = () => heard.map((h) => h.start.state);

describe('onPanelStart', () => {
  test('a start in flight at load is heard as starting, then running', async () => {
    m.fetchJSON.mockResolvedValueOnce(STARTING).mockResolvedValueOnce(RUNNING);

    await lifecycle.initPanel(PANEL);
    expect(heard).toEqual([{ panelId: 'jupyter', start: { state: 'starting', message: null } }]);

    await vi.advanceTimersByTimeAsync(1000);
    expect(states()).toEqual(['starting', 'running']);
    expect(panelState.jupyter.url).toBe('/panel/jupyter');
  });

  test("a failed config answer is heard as failed, with the server's sentence", async () => {
    m.fetchJSON.mockResolvedValueOnce(FAILED);

    await lifecycle.initPanel(PANEL);

    expect(heard).toEqual([
      { panelId: 'jupyter', start: { state: 'failed', message: 'JUPYTER failed to start: boom' } },
    ]);
  });

  test('a retry is heard as starting with the start answer, then its outcome', async () => {
    m.fetchJSON.mockResolvedValueOnce(FAILED);
    await lifecycle.initPanel(PANEL);
    m.apiRequest.mockResolvedValueOnce(STARTING);
    m.fetchJSON.mockResolvedValueOnce(FAILED);

    await lifecycle.retryPanelStart(PANEL);
    await vi.advanceTimersByTimeAsync(1000);

    expect(m.apiRequest).toHaveBeenCalledWith('/api/panels/jupyter/start', expect.anything());
    expect(heard.map((h) => h.start)).toEqual([
      { state: 'failed', message: 'JUPYTER failed to start: boom' },
      { state: 'starting', message: 'JUPYTER is starting' },
      { state: 'failed', message: 'JUPYTER failed to start: boom' },
    ]);
  });

  test('a follower that stopped hears nothing more', async () => {
    unfollow();
    m.fetchJSON.mockResolvedValueOnce(FAILED);

    await lifecycle.initPanel(PANEL);

    expect(heard).toEqual([]);
  });
});

describe('restartPanelServer', () => {
  async function running() {
    m.fetchJSON.mockResolvedValueOnce(RUNNING);
    await lifecycle.initPanel(PANEL);
    panelState.jupyter.healthy = true;
    heard.length = 0;
    vi.clearAllMocks();
  }

  test('POSTs the endpoint, stops polling the old server and follows the new start', async () => {
    await running();
    const iframe = document.createElement('iframe');
    panelState.jupyter.iframe = iframe;
    m.apiRequest.mockResolvedValueOnce(STARTING);
    m.fetchJSON.mockResolvedValueOnce(RUNNING);

    await lifecycle.restartPanelServer(PANEL, '/api/panels/jupyter/restart');

    expect(m.apiRequest).toHaveBeenCalledWith('/api/panels/jupyter/restart', expect.anything());
    expect(m.stopHealthPolling).toHaveBeenCalledWith(panelState.jupyter);
    expect(panelState.jupyter.healthy).toBe(false);
    expect(m.setEntryEnabled).toHaveBeenCalledWith(document.body, 'jupyter', false);
    expect(states()).toEqual(['starting']);

    await vi.advanceTimersByTimeAsync(1000);

    expect(states()).toEqual(['starting', 'running']);
    expect(m.startHealthPolling).toHaveBeenCalled();
    expect(iframe.getAttribute('src')).toContain('/panel/jupyter');
  });

  test('a refused restart throws and leaves the running panel as it was', async () => {
    await running();
    m.apiRequest.mockRejectedValueOnce(new Error('Could not restart the panel (HTTP 404)'));

    await expect(
      lifecycle.restartPanelServer(PANEL, '/api/panels/jupyter/restart')
    ).rejects.toThrow('HTTP 404');

    expect(panelState.jupyter.healthy).toBe(true);
    expect(m.stopHealthPolling).not.toHaveBeenCalled();
    expect(heard).toEqual([]);
  });
});
