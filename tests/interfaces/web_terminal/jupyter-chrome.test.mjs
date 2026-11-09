// @vitest-environment jsdom
/**
 * Contract tests for the Jupyter tile's hub chrome (jupyter-chrome.js): the
 * tile bar's Reload icon and its action, the ⋯ "This panel" rows (Restart
 * kernel server behind a confirm, Show server log in an info popover), and the
 * tile body's starting and failed states with Retry. Run:
 *   npx vitest run tests/interfaces/web_terminal/jupyter-chrome.test.mjs
 */
import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const m = vi.hoisted(() => ({
  PANELS: /** @type {any[]} */ ([]),
  contributeLocal: vi.fn(),
  setServiceMenuExtras: vi.fn(),
  setPanelBodyState: vi.fn(),
  onPanelStart: vi.fn(),
  restartPanelServer: vi.fn(() => Promise.resolve()),
  retryPanelStart: vi.fn(() => Promise.resolve()),
  reloadPanel: vi.fn(),
  fetchJSON: vi.fn(),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-catalog.js', () => ({
  PANELS: m.PANELS,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/tile-header-contrib.js', () => ({
  contributeLocal: m.contributeLocal,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-menu-policy.js', () => ({
  setServiceMenuExtras: m.setServiceMenuExtras,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js', () => ({
  reloadPanel: m.reloadPanel,
  setPanelBodyState: m.setPanelBodyState,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-lifecycle.js', () => ({
  onPanelStart: m.onPanelStart,
  restartPanelServer: m.restartPanelServer,
  retryPanelStart: m.retryPanelStart,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/api.js', () => ({
  fetchJSON: m.fetchJSON,
}));

const { initJupyterChrome, JUPYTER_PANEL_ID, LOG_ENDPOINT, RESTART_CONFIRM, RESTART_ENDPOINT } =
  await import(`${JS}/jupyter-chrome.js`);

const PANEL = {
  id: 'jupyter',
  label: 'Jupyter',
  configEndpoint: '/api/jupyter-server',
  startEndpoint: '/api/panels/jupyter/start',
};

/** The last contribution: [panelId, items, dispatch]. */
const lastBar = () => /** @type {any[]} */ (m.contributeLocal.mock.calls.at(-1));
/** The last registration: [panelId, rows]. */
const extrasCall = () => /** @type {any[]} */ (m.setServiceMenuExtras.mock.calls.at(-1));
/** The registered "This panel" rows. */
const rows = () => /** @type {any[]} */ (extrasCall()[1]());
/** @param {string} label */
const row = (label) => rows().find((r) => r.label === label);
/**
 * Tell the module a start outcome, as panel-lifecycle does.
 * @param {string} panelId
 * @param {string} state
 * @param {string | null} [message]
 */
const tell = (panelId, state, message = null) =>
  /** @type {any[]} */ (m.onPanelStart.mock.calls.at(-1))[0](panelId, { state, message });
/** Paint the last body state into a fresh host; null when the state was cleared. */
function paintedBody() {
  const [panelId, paint] = /** @type {any[]} */ (m.setPanelBodyState.mock.calls.at(-1));
  expect(panelId).toBe(JUPYTER_PANEL_ID);
  if (!paint) return null;
  const host = document.createElement('div');
  paint(host);
  document.body.append(host);
  return host;
}

/** A tile bar for the notebook panel, with its ⋯ button. */
function mountTile() {
  const tab = document.createElement('div');
  tab.className = 'tile-tab';
  tab.dataset.panelId = 'iframe:jupyter';
  const more = document.createElement('button');
  more.className = 'tile-tab-more osprey-icon-btn';
  tab.append(more);
  document.body.append(tab);
  return more;
}

beforeEach(() => {
  m.PANELS.splice(0, m.PANELS.length, PANEL);
  initJupyterChrome();
});

afterEach(() => {
  vi.clearAllMocks();
  vi.restoreAllMocks();
  document.body.replaceChildren();
});

describe('tile bar', () => {
  test('contributes one Reload icon and no other kind', () => {
    const [panelId, items] = lastBar();
    expect(panelId).toBe('jupyter');
    expect(items).toEqual([
      { kind: 'icon', id: 'reload', icon: 'refresh', title: 'Reload JupyterLab' },
    ]);
  });

  test('Reload reloads the notebook frame through the shared reloadPanel', () => {
    const [, , dispatch] = lastBar();
    dispatch('reload');
    expect(m.reloadPanel).toHaveBeenCalledWith('jupyter');
  });

  test('Reload is disabled while the server starts or after it failed, and back once it runs', () => {
    tell('jupyter', 'starting');
    expect(lastBar()[1][0].disabled).toBe(true);
    tell('jupyter', 'failed', 'Jupyter failed to start: boom');
    expect(lastBar()[1][0].disabled).toBe(true);
    tell('jupyter', 'running');
    expect(lastBar()[1][0].disabled).toBeUndefined();
  });

  test('a page without the notebook panel gets no chrome', () => {
    vi.clearAllMocks();
    m.PANELS.splice(0, m.PANELS.length);
    initJupyterChrome();
    expect(m.contributeLocal).not.toHaveBeenCalled();
    expect(m.setServiceMenuExtras).not.toHaveBeenCalled();
    expect(m.onPanelStart).not.toHaveBeenCalled();
  });
});

describe('⋯ This panel', () => {
  test('lists Restart kernel server, then Show server log', () => {
    expect(extrasCall()[0]).toBe('jupyter');
    expect(rows().map((r) => r.label)).toEqual(['Restart kernel server', 'Show server log']);
  });

  test('Show server log opens the tail in an info popover on the tile ⋯', async () => {
    const more = mountTile();
    m.fetchJSON.mockResolvedValue({ text: '[I] Serving notebooks\n[W] a warning' });

    row('Show server log').run();
    await vi.waitFor(() => expect(document.querySelector('.osprey-info-popover')).not.toBeNull());

    expect(m.fetchJSON).toHaveBeenCalledWith(LOG_ENDPOINT);
    const pop = /** @type {HTMLElement} */ (document.querySelector('.osprey-info-popover'));
    expect(pop.textContent).toContain('JupyterLab server log');
    expect(pop.textContent).toContain('[I] Serving notebooks\n[W] a warning');
    expect(more.getAttribute('aria-expanded')).toBe('true');
  });

  test('an empty log says so', async () => {
    mountTile();
    m.fetchJSON.mockResolvedValue({ text: '' });

    row('Show server log').run();
    await vi.waitFor(() => expect(document.querySelector('.osprey-info-popover')).not.toBeNull());

    expect(document.querySelector('.osprey-info-popover')?.textContent).toContain(
      'The server wrote no log yet.'
    );
  });

  test('without a tile the popover opens on the sidebar row', async () => {
    const rowEl = document.createElement('button');
    rowEl.className = 'panel-rail-button';
    rowEl.dataset.panelId = 'jupyter';
    document.body.append(rowEl);
    m.fetchJSON.mockRejectedValue(new Error('HTTP 404: Not Found'));

    row('Show server log').run();
    await vi.waitFor(() => expect(rowEl.getAttribute('aria-expanded')).toBe('true'));

    expect(document.querySelector('.osprey-info-popover')?.textContent).toContain(
      'Could not read the server log: HTTP 404: Not Found'
    );
  });

  test('Restart asks first and restarts only on yes', () => {
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);
    row('Restart kernel server').run();
    expect(confirm).toHaveBeenCalledWith(RESTART_CONFIRM);
    expect(RESTART_CONFIRM).toBe('Restart JupyterLab? Running kernels stop.');
    expect(m.restartPanelServer).not.toHaveBeenCalled();

    confirm.mockReturnValue(true);
    row('Restart kernel server').run();
    expect(m.restartPanelServer).toHaveBeenCalledWith(PANEL, RESTART_ENDPOINT);
  });

  test('Restart is disabled while a start is in flight', () => {
    tell('jupyter', 'starting');
    expect(row('Restart kernel server').disabled).toBe(true);
    tell('jupyter', 'failed', null);
    expect(row('Restart kernel server').disabled).toBe(false);
  });
});

describe('tile body states', () => {
  test('starting shows the one message, no action', () => {
    tell('jupyter', 'starting', 'Jupyter is starting');
    const host = /** @type {HTMLElement} */ (paintedBody());
    expect(host.querySelector('.osprey-state-message')?.textContent).toBe('Starting JupyterLab…');
    expect(host.querySelector('.osprey-state-detail')).toBeNull();
    expect(host.querySelector('button')).toBeNull();
  });

  test("failed shows the server's reason and Retry, which starts the panel again", () => {
    tell('jupyter', 'failed', 'Jupyter failed to start: exited with status 1');
    const host = /** @type {HTMLElement} */ (paintedBody());
    expect(host.querySelector('.osprey-state-message')?.textContent).toBe(
      'JupyterLab did not start'
    );
    expect(host.querySelector('.osprey-state-detail')?.textContent).toBe('exited with status 1');
    const retry = /** @type {HTMLButtonElement} */ (host.querySelector('button'));
    expect(retry.textContent).toBe('Retry');
    retry.click();
    expect(m.retryPanelStart).toHaveBeenCalledWith(PANEL);
  });

  test('a failure with no sentence has no second line', () => {
    tell('jupyter', 'failed', null);
    const host = /** @type {HTMLElement} */ (paintedBody());
    expect(host.querySelector('.osprey-state-detail')).toBeNull();
  });

  test('running takes the state away', () => {
    tell('jupyter', 'starting');
    tell('jupyter', 'running');
    expect(paintedBody()).toBeNull();
  });

  test("another panel's start outcome is not this tile's", () => {
    tell('canopy', 'failed', 'Canopy failed to start: boom');
    expect(m.setPanelBodyState).not.toHaveBeenCalled();
  });
});
