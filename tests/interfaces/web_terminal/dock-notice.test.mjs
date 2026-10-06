// @ts-check
/**
 * Unit tests for the dock iframe adapter's NOT-ANSWERING NOTICE
 * (setPanelReachable in dock-iframe.js):
 *
 *   npx vitest run tests/interfaces/web_terminal/dock-notice.test.mjs
 *
 * A panel whose backend stops answering keeps its overlay iframe on the last
 * page it rendered, so the tile alone cannot tell a quiet panel from a dead
 * one. The adapter draws a notice over the tile body. These tests pin the
 * contract the caller (panel-lifecycle.js) wires against:
 *
 *   - the notice is one `.tile-notice` element per managed panel in the
 *     overlay layer, sized to the placeholder group's content rectangle with
 *     applyGeometry's math, and the iframe under it is never moved, blanked or
 *     reloaded;
 *   - it exists from adoption on, laid out but empty while the panel answers,
 *     so its polite live region is in the accessibility tree before its text;
 *   - the caller hands in an already formatted time; the adapter never formats
 *     one, and the text always carries it;
 *   - reachability is held per panel id, so a call before the tile exists or
 *     while it is closed takes effect when the tile is next on screen;
 *   - it is hidden wherever the iframe is hidden (behind another tab, closed,
 *     evicted) and follows every resize and regroup;
 *   - it takes no pointer events, and the drag pointer shield never hands it
 *     any;
 *   - with no dockview at all, the fallback host (#panel-content) carries one
 *     notice for whichever panel it is showing.
 *
 * dockview and dock-workspace are stubbed at the module boundary exactly as in
 * dock-glow.test.mjs, over the same shared fake api (_dock-fake.mjs).
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { makeDockApi as makeApi, addTerminal } from './_dock-fake.mjs';

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

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  getDockApi.mockImplementation(() => state.api);
  state.api = null;
  redock.fn = null;
  vi.stubGlobal('ResizeObserver', FakeResizeObserver);
  document.body.innerHTML = '<div class="main-container"></div>';
});

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

/** The opaque display time the caller hands in; the adapter never formats it. */
const SINCE = '2:02 PM';

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
 * happy-dom lays nothing out, so both rectangles the geometry math consumes are
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

/**
 * Fire the adapter's dockview layout handler. happy-dom lays nothing out, and
 * the first geometry pass ran before the rectangle was stubbed.
 * @param {any} api
 */
function relayout(api) {
  api.onDidLayoutChange.mock.calls[0][0]();
}

/** @returns {HTMLElement[]} */
function noticeEls() {
  return /** @type {HTMLElement[]} */ ([...document.querySelectorAll('.dock-iframe-overlay > .tile-notice')]);
}

/** @param {string} id @returns {HTMLElement} */
function noticeFor(id) {
  return /** @type {HTMLElement} */ (document.querySelector(`.tile-notice[data-panel="${id}"]`));
}

/** @param {HTMLElement} el @returns {string} */
function textOf(el) {
  return /** @type {HTMLElement} */ (el.querySelector('.tile-notice-text')).textContent ?? '';
}

/** Adopt ariel into its own tile, stub its rect and mark it not answering. */
async function arielDown() {
  const api = makeApi();
  addTerminal(api);
  const mod = await freshAdapter(api);
  const frame = makeIframe();
  frame.src = 'http://example.test/panel/ariel/';
  mod.adoptIframe('ariel', frame, { title: 'ARIEL' });
  stubTileRect(api, 'iframe:ariel', { left: 340, top: 90, width: 620, height: 480 });
  relayout(api);
  return { api, mod, frame };
}

describe('setPanelReachable — the tile body says its backend stopped answering', () => {
  test('the notice is one .tile-notice element sized to the tile rectangle', async () => {
    const { mod, frame } = await arielDown();
    const before = {
      src: frame.src,
      left: frame.style.left,
      top: frame.style.top,
      width: frame.style.width,
      height: frame.style.height,
    };

    mod.setPanelReachable('ariel', false, SINCE);

    const els = noticeEls();
    expect(els).toHaveLength(1);
    const [el] = els;
    expect(el.dataset.panel).toBe('ariel');
    expect(el.getAttribute('role')).toBe('status');
    expect(el.getAttribute('aria-live')).toBe('polite');
    expect(el.hasAttribute('data-unreachable')).toBe(true);
    expect(el.style.left).toBe('240px');
    expect(el.style.top).toBe('50px');
    expect(el.style.width).toBe('620px');
    expect(el.style.height).toBe('480px');
    expect(textOf(el)).toContain(SINCE);
    // The frame under the notice is never moved, blanked or reloaded.
    expect(frame.style.display).toBe('');
    expect({
      src: frame.src,
      left: frame.style.left,
      top: frame.style.top,
      width: frame.style.width,
      height: frame.style.height,
    }).toEqual(before);
  });

  test('the notice exists before it has anything to say', async () => {
    // A polite live region must be in the accessibility tree before its text
    // arrives, or the arrival is not announced.
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    mod.adoptIframe('ariel', makeIframe(), { title: 'ARIEL' });

    const el = noticeFor('ariel');
    expect(el).toBeTruthy();
    expect(el.getAttribute('role')).toBe('status');
    expect(el.hasAttribute('data-unreachable')).toBe(false);
    expect(textOf(el)).toBe('');
  });

  test('answering again clears the text and keeps the region', async () => {
    const { mod } = await arielDown();
    mod.setPanelReachable('ariel', false, SINCE);
    const el = noticeFor('ariel');

    mod.setPanelReachable('ariel', true);

    expect(noticeFor('ariel')).toBe(el);
    expect(el.isConnected).toBe(true);
    expect(el.hasAttribute('data-unreachable')).toBe(false);
    expect(textOf(el)).toBe('');
  });

  test('a repeated identical call changes nothing', async () => {
    const { mod } = await arielDown();
    mod.setPanelReachable('ariel', false, SINCE);
    mod.setPanelReachable('ariel', false, SINCE);
    expect(noticeEls()).toHaveLength(1);

    mod.setPanelReachable('ariel', false, '2:05 PM');
    expect(noticeEls()).toHaveLength(1);
    expect(textOf(noticeFor('ariel'))).toContain('2:05 PM');
    expect(textOf(noticeFor('ariel'))).not.toContain(SINCE);
  });

  test('a time recorded before the tile exists shows when the tile is adopted', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);

    mod.setPanelReachable('ariel', false, SINCE);
    expect(document.querySelector('.tile-notice')).toBeNull();

    mod.adoptIframe('ariel', makeIframe(), { title: 'ARIEL' });
    relayout(api);

    const el = noticeFor('ariel');
    expect(el.hasAttribute('data-unreachable')).toBe(true);
    expect(textOf(el)).toContain(SINCE);
  });

  test('two panels are independent', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    mod.adoptIframe('ariel', makeIframe(), { title: 'ARIEL' });
    api.addPanel({ id: 'iframe:artifacts', component: 'dock-iframe-placeholder', title: 'WORKSPACE' });
    mod.adoptIframe('artifacts', makeIframe(), { title: 'WORKSPACE' });
    stubTileRect(api, 'iframe:ariel', { left: 340, top: 90, width: 300, height: 480 });
    stubTileRect(api, 'iframe:artifacts', { left: 640, top: 90, width: 300, height: 480 });

    mod.setPanelReachable('ariel', false, SINCE);

    const down = noticeEls().filter((el) => el.hasAttribute('data-unreachable'));
    expect(down).toHaveLength(1);
    expect(down[0].dataset.panel).toBe('ariel');
  });
});

describe('setPanelReachable — the notice follows the tile, never a tile that is not showing it', () => {
  test('it follows a resize or regroup', async () => {
    const { api, mod } = await arielDown();
    mod.setPanelReachable('ariel', false, SINCE);

    stubTileRect(api, 'iframe:ariel', { left: 400, top: 120, width: 300, height: 200 });
    relayout(api);

    const el = noticeFor('ariel');
    expect(el.style.left).toBe('300px');
    expect(el.style.top).toBe('80px');
    expect(el.style.width).toBe('300px');
    expect(el.style.height).toBe('200px');
  });

  test('it hides behind another tab and comes back', async () => {
    const { api, mod } = await arielDown();
    mod.setPanelReachable('ariel', false, SINCE);
    const group = api.getPanel('iframe:ariel').group;
    const own = group.activePanel;

    group.activePanel = { id: 'iframe:other', group };
    relayout(api);
    expect(noticeFor('ariel').style.display).toBe('none');

    group.activePanel = own;
    relayout(api);
    expect(noticeFor('ariel').style.display).toBe('');
    expect(noticeFor('ariel').hasAttribute('data-unreachable')).toBe(true);
  });

  for (const close of /** @type {const} */ (['hidePanel', 'concealPanel'])) {
    test(`closing the tile hides it, reopening shows it again (${close})`, async () => {
      const { api, mod } = await arielDown();
      mod.setPanelReachable('ariel', false, SINCE);

      mod[close]('ariel');
      expect(noticeFor('ariel').style.display).toBe('none');

      mod.focusPanel('ariel');
      relayout(api);
      const el = noticeFor('ariel');
      expect(el.style.display).toBe('');
      expect(el.hasAttribute('data-unreachable')).toBe(true);
      expect(textOf(el)).toContain(SINCE);
    });
  }

  test("an evicted occupant's notice goes with it", async () => {
    const { mod } = await arielDown();
    mod.setPanelReachable('ariel', false, SINCE);

    mod.adoptIframe('artifacts', makeIframe(), { title: 'WORKSPACE' });

    expect(noticeFor('ariel').style.display).toBe('none');
  });

  test('the notice takes no pointer events', async () => {
    const { mod } = await arielDown();
    mod.setPanelReachable('ariel', false, SINCE);
    const el = noticeFor('ariel');
    expect(el.style.pointerEvents).toBe('none');

    mod.setIframePointerShield(true);
    mod.setIframePointerShield(false);

    expect(el.style.pointerEvents).toBe('none');
  });
});

describe('setPanelReachable — fallback mode has no tiles', () => {
  /** @returns {Promise<{ mod: any, host: HTMLElement }>} */
  async function fallback() {
    const host = document.createElement('div');
    host.id = 'panel-content';
    document.body.appendChild(host);
    state.api = null; // no dock shell at all
    const mod = await import(ADAPTER);
    mod.initDockIframeAdapter({ fallbackHost: host });
    mod.adoptIframe('ariel', makeIframe(), { title: 'ARIEL' });
    mod.adoptIframe('artifacts', makeIframe(), { title: 'WORKSPACE' });
    mod.focusPanel('ariel');
    mod.setPanelReachable('ariel', false, SINCE);
    return { mod, host };
  }

  test('the host carries the notice for the panel it is showing', async () => {
    const { mod, host } = await fallback();

    let el = /** @type {HTMLElement} */ (host.firstElementChild);
    expect(el.classList.contains('tile-notice')).toBe(true);
    expect(el.hasAttribute('data-unreachable')).toBe(true);
    expect(textOf(el)).toContain(SINCE);

    mod.focusPanel('artifacts');
    el = /** @type {HTMLElement} */ (host.querySelector('.tile-notice'));
    expect(el.hasAttribute('data-unreachable')).toBe(false);
    expect(textOf(el)).toBe('');

    mod.focusPanel('ariel');
    el = /** @type {HTMLElement} */ (host.querySelector('.tile-notice'));
    expect(el.hasAttribute('data-unreachable')).toBe(true);
    expect(textOf(el)).toContain(SINCE);
  });

  test('a host wiped by the empty state gets its notice back', async () => {
    const { mod, host } = await fallback();

    host.innerHTML = ''; // what panel-empty-state.js does
    mod.focusPanel('ariel');

    const el = /** @type {HTMLElement} */ (host.firstElementChild);
    expect(el.classList.contains('tile-notice')).toBe(true);
    expect(el.isConnected).toBe(true);
  });
});
