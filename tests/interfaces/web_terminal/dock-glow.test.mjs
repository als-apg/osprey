// @ts-check
/**
 * Unit tests for the dock iframe adapter's AGENT ATTRIBUTION GLOW
 * (glowPanel in dock-iframe.js):
 *
 *   npx vitest run tests/interfaces/web_terminal/dock-glow.test.mjs
 *
 * The rail entry's ~20px tab is too small to carry "the agent touched THIS
 * panel", so an agent action also flashes the affected tile's whole body.
 * These tests pin the contract the call sites (panel-manager.js) wire against:
 *
 *   - the glow is a dedicated `.tile-glow` element in the overlay layer, sized
 *     to the placeholder group's content rectangle with applyGeometry's math —
 *     never `.agent-flash` on the iframe, which would clip the embedded app to
 *     the flash's border-radius;
 *   - the rectangle is read in a requestAnimationFrame, because a glow usually
 *     follows the activation that created the tile and dockview's geometry only
 *     lands on settle;
 *   - it no-ops for anything not genuinely on screen (unmanaged, hidden, no
 *     placeholder, or sitting behind another tab) — glowing the wrong tile
 *     misattributes the action;
 *   - the element is reused across flashes, so repeated activity cannot litter
 *     the overlay;
 *   - with no dockview at all there are no tiles, so the fallback host
 *     (#panel-content) takes the flash directly.
 *
 * dockview and dock-workspace are stubbed at the module boundary exactly as in
 * dock-iframe.test.mjs; `flashElement` is the REAL design-system helper (via the
 * `/design-system/js` vitest alias) so the `.agent-flash` contract is exercised
 * rather than mocked away.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const ADAPTER = '../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js';

const { getDockApi, state, redock } = vi.hoisted(() => ({
  state: { api: /** @type {any} */ (null) },
  redock: { fn: /** @type {null | (() => void)} */ (null) },
  getDockApi: vi.fn(() => /** @type {any} */ (null)),
}));
getDockApi.mockImplementation(() => state.api);

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  getDockApi,
  defaultServiceWidth: () => 600,
  setServiceRedock: (/** @type {() => void} */ fn) => { redock.fn = fn; },
  onDragGesture: () => [],
}));

/** The adapter's live-follow observer; geometry itself is browser-suite turf. */
class FakeResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

/** Queued requestAnimationFrame callbacks — drained explicitly by flushFrame(). */
let frameCallbacks = /** @type {(() => void)[]} */ ([]);

function flushFrame() {
  const queued = frameCallbacks;
  frameCallbacks = [];
  for (const cb of queued) cb();
}

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  getDockApi.mockImplementation(() => state.api);
  state.api = null;
  redock.fn = null;
  frameCallbacks = [];
  vi.stubGlobal('ResizeObserver', FakeResizeObserver);
  vi.stubGlobal('requestAnimationFrame', (/** @type {() => void} */ cb) => {
    frameCallbacks.push(cb);
    return frameCallbacks.length;
  });
  document.body.innerHTML = '<div class="main-container"></div>';
});

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

/**
 * The same hand-built DockviewApi stand-in dock-iframe.test.mjs uses: a
 * 'within' add joins the reference group, anything else opens a new one, and
 * the added panel becomes active.
 * @returns {any}
 */
function makeApi() {
  let groupSeq = 0;
  /** @type {any} */
  const api = {
    activePanel: null,
    groups: /** @type {any[]} */ ([]),
    panels: /** @type {any[]} */ ([]),
    _activeCbs: /** @type {(() => void)[]} */ ([]),
    onDidLayoutChange: vi.fn(() => ({ dispose() {} })),
    onDidActivePanelChange: vi.fn((/** @type {() => void} */ cb) => {
      api._activeCbs.push(cb);
      return { dispose() {} };
    }),
    getPanel: (/** @type {string} */ id) => api.panels.find((/** @type {any} */ p) => p.id === id) ?? null,
    addPanel: (/** @type {any} */ opts) => {
      const group = opts.position?.referenceGroup && opts.position.direction === 'within'
        ? opts.position.referenceGroup
        : makeGroup();
      const panel = { id: opts.id, title: opts.title, group, api: { setActive: vi.fn() } };
      group.panels.push(panel);
      group.activePanel = panel;
      api.panels.push(panel);
      api.activePanel = panel;
      for (const cb of api._activeCbs) cb();
      return panel;
    },
    removePanel: (/** @type {any} */ panel) => {
      api.panels = api.panels.filter((/** @type {any} */ p) => p !== panel);
      const group = panel.group;
      group.panels = group.panels.filter((/** @type {any} */ p) => p !== panel);
      if (group.panels.length === 0) {
        api.groups = api.groups.filter((/** @type {any} */ g) => g !== group);
      } else if (group.activePanel === panel) {
        group.activePanel = group.panels[0];
      }
      if (api.activePanel === panel) api.activePanel = group.panels[0] ?? api.panels[0] ?? null;
    },
  };
  function makeGroup() {
    const element = document.createElement('div');
    const content = document.createElement('div');
    content.className = 'dv-content-container';
    element.appendChild(content);
    const group = { id: `group-${++groupSeq}`, panels: [], activePanel: null, element };
    api.groups.push(group);
    return group;
  }
  api._makeGroup = makeGroup;
  return api;
}

/** Seed the fake api with the native terminal card in its own group. @param {any} api */
function addTerminal(api) {
  const group = api._makeGroup();
  const terminal = { id: 'terminal', group, api: { setActive: vi.fn() } };
  group.panels.push(terminal);
  group.activePanel = terminal;
  api.panels.push(terminal);
  api.activePanel = terminal;
  return terminal;
}

function makeIframe() {
  return document.createElement('iframe');
}

/** @param {any} api */
async function freshAdapter(api) {
  state.api = api;
  const mod = await import(ADAPTER);
  mod.initDockIframeAdapter({ fallbackHost: null });
  return mod;
}

/**
 * happy-dom lays nothing out, so both rectangles the glow math consumes are
 * stubbed: the overlay's origin and the tile's content box.
 * @param {any} api @param {string} placeholderId
 * @param {{left: number, top: number, width: number, height: number}} rect
 */
function stubTileRect(api, placeholderId, rect) {
  const overlay = /** @type {HTMLElement} */ (document.querySelector('.dock-iframe-overlay'));
  overlay.getBoundingClientRect = () => /** @type {any} */ ({ left: 100, top: 40 });
  const content = api.getPanel(placeholderId).group.element.querySelector('.dv-content-container');
  content.getBoundingClientRect = () => /** @type {any} */ (rect);
}

/** @returns {HTMLElement[]} */
function glowEls() {
  return /** @type {HTMLElement[]} */ ([...document.querySelectorAll('.dock-iframe-overlay .tile-glow')]);
}

describe('glowPanel — the tile body carries the agent attribution', () => {
  test('a dedicated .tile-glow element takes the flash, sized to the tile rectangle', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    const frame = makeIframe();
    mod.adoptIframe('ariel', frame, { title: 'ARIEL' });
    stubTileRect(api, 'iframe:ariel', { left: 340, top: 90, width: 620, height: 480 });

    mod.glowPanel('ariel');
    flushFrame();

    const [glow] = glowEls();
    expect(glow).toBeTruthy();
    expect(glow.classList.contains('agent-flash')).toBe(true);
    // applyGeometry's math: the tile rect expressed in the overlay's own origin.
    expect(glow.style.left).toBe('240px');
    expect(glow.style.top).toBe('50px');
    expect(glow.style.width).toBe('620px');
    expect(glow.style.height).toBe('480px');
  });

  test('the iframe itself is never flashed — .agent-flash would clip the embedded app', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    const frame = makeIframe();
    mod.adoptIframe('ariel', frame, { title: 'ARIEL' });
    stubTileRect(api, 'iframe:ariel', { left: 340, top: 90, width: 620, height: 480 });

    mod.glowPanel('ariel');
    flushFrame();

    expect(frame.classList.contains('agent-flash')).toBe(false);
  });

  test('the rectangle is read on the next frame, not synchronously', async () => {
    // A glow usually follows the activation that created the tile; dockview's
    // geometry only lands once the layout settles.
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    mod.adoptIframe('ariel', makeIframe(), { title: 'ARIEL' });
    stubTileRect(api, 'iframe:ariel', { left: 340, top: 90, width: 620, height: 480 });

    mod.glowPanel('ariel');
    expect(glowEls()).toHaveLength(0); // nothing measured yet

    flushFrame();
    expect(glowEls()).toHaveLength(1);
  });

  test('repeated activity reuses the one element and re-fires the flash', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    mod.adoptIframe('ariel', makeIframe(), { title: 'ARIEL' });
    stubTileRect(api, 'iframe:ariel', { left: 340, top: 90, width: 620, height: 480 });

    mod.glowPanel('ariel');
    flushFrame();
    const [first] = glowEls();
    // The flash is self-cleaning on animationend; a second call must restart it.
    first.dispatchEvent(new Event('animationend'));
    expect(first.classList.contains('agent-flash')).toBe(false);

    mod.glowPanel('ariel');
    flushFrame();

    expect(glowEls()).toEqual([first]);
    expect(first.classList.contains('agent-flash')).toBe(true);
  });

  test('two panels glow independently — one element per tile', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    mod.adoptIframe('ariel', makeIframe(), { title: 'ARIEL' });
    api.addPanel({ id: 'iframe:artifacts', component: 'dock-iframe-placeholder', title: 'WORKSPACE' });
    mod.adoptIframe('artifacts', makeIframe(), { title: 'WORKSPACE' });
    stubTileRect(api, 'iframe:ariel', { left: 340, top: 90, width: 300, height: 480 });
    stubTileRect(api, 'iframe:artifacts', { left: 640, top: 90, width: 300, height: 480 });

    mod.glowPanel('ariel');
    mod.glowPanel('artifacts');
    flushFrame();

    expect(glowEls().map((/** @type {HTMLElement} */ el) => el.style.left)).toEqual(['240px', '540px']);
  });
});

describe('glowPanel — no-op unless the panel is genuinely on screen', () => {
  test('a panel the adapter never adopted glows nothing', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);

    mod.glowPanel('ghost');
    flushFrame();

    expect(glowEls()).toHaveLength(0);
  });

  test('a hidden panel glows nothing — its placeholder is gone', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    mod.adoptIframe('ariel', makeIframe(), { title: 'ARIEL' });
    mod.hidePanel('ariel');

    mod.glowPanel('ariel');
    flushFrame();

    expect(glowEls()).toHaveLength(0);
  });

  test('a panel closed between the call and the frame glows nothing', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    mod.adoptIframe('ariel', makeIframe(), { title: 'ARIEL' });
    stubTileRect(api, 'iframe:ariel', { left: 340, top: 90, width: 620, height: 480 });

    mod.glowPanel('ariel');
    mod.hidePanel('ariel'); // the tile went away during the frame
    flushFrame();

    expect(glowEls()).toHaveLength(0);
  });

  test('a panel sitting behind another tab glows nothing — that tile is not showing it', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    mod.adoptIframe('ariel', makeIframe(), { title: 'ARIEL' });
    stubTileRect(api, 'iframe:ariel', { left: 340, top: 90, width: 620, height: 480 });
    // Another panel took the foreground of the same group.
    const group = api.getPanel('iframe:ariel').group;
    group.activePanel = { id: 'iframe:other', group };

    mod.glowPanel('ariel');
    flushFrame();

    expect(glowEls()).toHaveLength(0);
  });
});

describe('glowPanel — fallback mode has no tiles', () => {
  test('the mounted host takes the flash directly', async () => {
    const host = document.createElement('div');
    host.id = 'panel-content';
    document.body.appendChild(host);
    state.api = null; // no dock shell at all
    const mod = await import(ADAPTER);
    mod.initDockIframeAdapter({ fallbackHost: host });
    mod.adoptIframe('ariel', makeIframe(), { title: 'ARIEL' });

    mod.glowPanel('ariel');

    expect(host.classList.contains('agent-flash')).toBe(true);
    expect(document.querySelector('.tile-glow')).toBeNull();
  });

  test('with no host either, glowing is inert rather than throwing', async () => {
    state.api = null;
    const mod = await import(ADAPTER);
    mod.initDockIframeAdapter({ fallbackHost: null });

    expect(() => mod.glowPanel('ariel')).not.toThrow();
  });
});
