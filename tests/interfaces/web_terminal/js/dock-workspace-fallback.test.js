/**
 * Unit tests for the no-dockview fallback's tile bar (initFallbackTileBar in
 * dock-workspace.js), happy-dom environment:
 *   npx vitest run tests/interfaces/web_terminal/js/dock-workspace-fallback.test.js
 *
 * Embed contract v2 gives an embedded panel exactly one header, the hub's tile
 * bar, and panels hide their in-body copies of what they contribute. When the
 * dockview bundle does not load, the fallback host must still carry that bar,
 * or a panel's contributed controls (a session's Open / Take over) are
 * reachable nowhere. These tests boot initDockWorkspace with no dockview
 * global and drive the REAL hub receive side (tile-header-contrib.js): a
 * contribution renders into the fallback bar, a click round-trips to the
 * iframe, the bar follows the active panel, and it steps aside only when no
 * panel is active — in both views, as a docked tile's bar does.
 */

import { test, expect, describe, beforeAll, beforeEach, afterEach, vi } from 'vitest';

beforeAll(() => {
  if (typeof globalThis.ResizeObserver === 'undefined') {
    globalThis.ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  }
});

vi.mock('../../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  fitTerminal: () => {},
  getCurrentSessionId: () => null,
}));
vi.mock('../../../../src/osprey/interfaces/web_terminal/static/js/api.js', () => ({
  fetchJSON: async () => ({}),
}));

import {
  initDockWorkspace,
  isTerminalPanelOpen,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js';
import { initHeaderContrib } from '../../../../src/osprey/interfaces/web_terminal/static/js/tile-header-contrib.js';

/** Let MutationObserver callbacks run. */
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

/** The hub page's DOM as index.html ships it, minus dockview. */
function mountHub() {
  document.documentElement.setAttribute('data-ui-mode', 'expert');
  document.body.innerHTML = `
    <nav>
      <button class="panel-rail-button" data-panel-id="tuning" aria-label="Tuning"></button>
      <button class="panel-rail-button" data-panel-id="artifacts" aria-label="Workspace"></button>
    </nav>
    <div class="dock-root" id="dock-root"></div>
    <div class="dock-panel-sources" id="dock-panel-sources">
      <div class="files-panel" id="right-panel">
        <div class="panel-content" id="panel-content"></div>
      </div>
      <div class="terminal-panel"><div class="terminal-card">
        <div class="terminal-header"></div>
      </div></div>
    </div>`;
}

/**
 * A panel iframe in the fallback host, with a stand-in contentWindow the hub
 * matches contributions against and sends actions to.
 * @param {string} panelId
 */
function addIframe(panelId) {
  const iframe = document.createElement('iframe');
  iframe.className = 'panel-iframe';
  iframe.dataset.panelId = panelId;
  const contentWindow = { postMessage: vi.fn() };
  Object.defineProperty(iframe, 'contentWindow', { value: contentWindow });
  document.getElementById('panel-content')?.appendChild(iframe);
  return contentWindow;
}

/**
 * Deliver a contribution as if posted by *source*.
 * @param {object} source  the posting iframe's contentWindow stand-in
 * @param {object[]} items
 */
function contribute(source, items) {
  window.dispatchEvent(
    new MessageEvent('message', {
      origin: window.location.origin,
      source: /** @type {any} */ (source),
      data: { type: 'osprey-header-contribution', version: 1, items },
    })
  );
}

const SESSION_ITEMS = [
  { kind: 'button', id: 'session-open', label: 'Open session' },
  { kind: 'button', id: 'session-takeover', label: 'Take over', disabled: true },
];

/** @returns {HTMLElement} */
function bar() {
  const el = document.querySelector('#right-panel > .fallback-tile-bar');
  if (!(el instanceof HTMLElement)) throw new Error('no fallback tile bar');
  return el;
}

const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

describe('hub with no dockview', () => {
  beforeEach(() => {
    delete (/** @type {any} */ (window)['dockview-core']);
    mountHub();
    initHeaderContrib();
  });
  afterEach(() => {
    document.body.innerHTML = '';
    errorSpy.mockClear();
  });

  test('the fallback host gets one tile bar above the panel content', () => {
    initDockWorkspace();
    expect(errorSpy).toHaveBeenCalledWith(
      'dockview-core global unavailable; workspace shell not initialized'
    );
    const host = /** @type {HTMLElement} */ (document.getElementById('right-panel'));
    expect(host.querySelectorAll(':scope > .fallback-tile-bar')).toHaveLength(1);
    expect(bar().nextElementSibling?.id).toBe('panel-content');
    expect(host.style.display).not.toBe('none');
  });

  test("the active panel's session controls render in the bar and round-trip", async () => {
    const win = addIframe('tuning');
    initDockWorkspace();
    const host = /** @type {HTMLElement} */ (document.getElementById('right-panel'));
    host.dataset.activePanel = 'tuning';
    await flush();
    contribute(win, SESSION_ITEMS);

    expect(bar().style.display).toBe('');
    expect(bar().querySelector('.tile-tab-title')?.textContent).toBe('Tuning');
    const open = bar().querySelector('[data-contrib-key="button:session-open"]');
    expect(open).toBeInstanceOf(HTMLButtonElement);
    expect(open?.textContent).toBe('Open session');
    const takeover = bar().querySelector('[data-contrib-key="button:session-takeover"]');
    expect(/** @type {HTMLButtonElement} */ (takeover).disabled).toBe(true);

    /** @type {HTMLButtonElement} */ (open).click();
    expect(win.postMessage).toHaveBeenCalledWith(
      { type: 'osprey-header-action', id: 'session-open' },
      window.location.origin
    );
  });

  test('a contribution that arrives before activation renders once the panel is active', async () => {
    const win = addIframe('tuning');
    initDockWorkspace();
    contribute(win, SESSION_ITEMS);
    expect(bar().querySelector('[data-contrib-key="button:session-open"]')).toBeNull();

    /** @type {HTMLElement} */ (document.getElementById('right-panel')).dataset.activePanel =
      'tuning';
    await flush();
    expect(bar().querySelector('[data-contrib-key="button:session-open"]')).not.toBeNull();
  });

  test('the bar follows the active panel and shows only its contribution', async () => {
    const tuning = addIframe('tuning');
    const artifacts = addIframe('artifacts');
    initDockWorkspace();
    contribute(tuning, SESSION_ITEMS);
    contribute(artifacts, [{ kind: 'button', id: 'refresh', label: 'Refresh' }]);
    const host = /** @type {HTMLElement} */ (document.getElementById('right-panel'));

    host.dataset.activePanel = 'artifacts';
    await flush();
    expect(bar().querySelector('.tile-tab-title')?.textContent).toBe('Workspace');
    expect(bar().querySelectorAll('.tile-tab-contrib')).toHaveLength(1);
    expect(bar().querySelector('[data-contrib-key="button:refresh"]')).not.toBeNull();
    expect(bar().querySelector('[data-contrib-key="button:session-open"]')).toBeNull();

    host.dataset.activePanel = 'tuning';
    await flush();
    expect(bar().querySelectorAll('.tile-tab-contrib')).toHaveLength(1);
    expect(bar().querySelector('[data-contrib-key="button:refresh"]')).toBeNull();
    const open = bar().querySelector('[data-contrib-key="button:session-open"]');
    /** @type {HTMLButtonElement} */ (open).click();
    expect(tuning.postMessage).toHaveBeenCalledTimes(1);
    expect(artifacts.postMessage).not.toHaveBeenCalled();
  });

  test('the bar hides only with no active panel, and shows in both views', async () => {
    addIframe('tuning');
    initDockWorkspace();
    expect(bar().style.display).toBe('none');

    const host = /** @type {HTMLElement} */ (document.getElementById('right-panel'));
    host.dataset.activePanel = 'tuning';
    await flush();
    expect(bar().style.display).toBe('');

    document.documentElement.setAttribute('data-ui-mode', 'simple');
    await flush();
    expect(bar().style.display).toBe('');
    document.documentElement.setAttribute('data-ui-mode', 'expert');

    delete host.dataset.activePanel;
    await flush();
    expect(bar().style.display).toBe('none');
  });

  test('the Agent card is never in the dock: the pane counts as open and nothing is stamped', () => {
    initDockWorkspace();
    expect(isTerminalPanelOpen()).toBe(true);
    expect(document.documentElement.hasAttribute('data-agent')).toBe(false);
  });
});
