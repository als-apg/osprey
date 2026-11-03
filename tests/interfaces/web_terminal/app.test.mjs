// @ts-check
/**
 * Unit tests for the hub's focus-exit bridge (app.js's `moveFocusOutOfPanel`,
 * reached through the `initIframePasteBridge` message listener):
 *   npx vitest run tests/interfaces/web_terminal/app.test.mjs
 *
 * A frame cannot move focus in its parent, so a panel whose Tab order runs off
 * either end posts `{type:'canopy-host-focus-exit', direction}` to `window.parent`
 * and the hub moves focus for it: to the previous/next tabbable element in host
 * document order relative to where the panel sits in the dock. Where that is
 * — the panel's placeholder in dock mode, the iframe in fallback mount — is
 * dock-iframe.js's `panelFocusAnchor` (dock-iframe.test.mjs), stubbed here so
 * the walk itself is what is pinned: the anchor's own subtree is never a
 * landing, elements the anchor says to skip (other panels' overlay frames)
 * are stepped over, and so is anything Tab cannot reach.
 *
 * Only a managed panel frame may ask: a same-origin message from any other
 * window is ignored, and so is any message from another origin.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

const APP = '../../../src/osprey/interfaces/web_terminal/static/js/app.js';

const state = vi.hoisted(() => ({
  /** @type {{ anchor: Element, skip: (el: Element) => boolean } | null} */
  found: null,
  /** @type {unknown} */
  askedBy: null,
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  initTerminal: vi.fn(),
  startTerminal: vi.fn(),
  stopTerminal: vi.fn(),
  restartTerminal: vi.fn(),
  switchSession: vi.fn(),
  setSessionLabel: vi.fn(),
  notifySessionChange: vi.fn(),
  clearStoredSessionId: vi.fn(),
  fitTerminal: vi.fn(),
  getTerminalInstance: () => null,
  getCurrentSessionId: () => null,
  focusTerminal: vi.fn(),
  pasteToTerminal: vi.fn(),
  onSessionChange: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js', () => ({
  panelFocusAnchor: (/** @type {unknown} */ source) => {
    state.askedBy = source;
    return source === PANEL_WINDOW ? state.found : null;
  },
}));

/** The panel frame's window, as `event.source` carries it. */
const PANEL_WINDOW = /** @type {any} */ ({ panel: true });

/** Message listeners the bridge registered this test, removed in afterEach. */
/** @type {EventListenerOrEventListenerObject[]} */
let listeners = [];

/** @param {string} id */
function button(id) {
  const el = document.createElement('button');
  el.id = id;
  el.textContent = id;
  return el;
}

/**
 * The host page around one panel: `before` and `after` are tabbable; the
 * anchor holds its own button (never a landing) and an overlay frame sits
 * after everything, to be skipped.
 */
function page() {
  const before = button('before');
  const anchor = document.createElement('div');
  anchor.id = 'anchor';
  anchor.appendChild(button('inside'));
  const after = button('after');
  const overlay = document.createElement('div');
  overlay.id = 'overlay';
  const frame = document.createElement('iframe');
  frame.id = 'other-frame';
  overlay.appendChild(frame);
  document.body.append(before, anchor, after, overlay);
  state.found = { anchor, skip: (el) => overlay.contains(el) };
  return { before, anchor, after, overlay };
}

const activeId = () => document.activeElement?.id ?? '';

/** @param {any} data @param {unknown} source @param {string} [origin] */
function post(data, source, origin = window.location.origin) {
  const event = new MessageEvent('message', { data, origin, source: /** @type {any} */ (source) });
  for (const fn of listeners) {
    if (typeof fn === 'function') fn(event);
    else fn.handleEvent(event);
  }
}

beforeEach(() => {
  vi.resetModules();
  state.found = null;
  state.askedBy = null;
  document.body.innerHTML = '';
  const add = window.addEventListener.bind(window);
  vi.spyOn(window, 'addEventListener').mockImplementation((type, fn, opts) => {
    if (type === 'message') listeners.push(fn);
    add(type, fn, opts);
  });
});

afterEach(() => {
  for (const fn of listeners) window.removeEventListener('message', fn);
  listeners = [];
  vi.restoreAllMocks();
});

describe('canopy-host-focus-exit: the walk', () => {
  test('backward focuses the tabbable element before the panel', async () => {
    const app = await import(APP);
    page();
    expect(app.moveFocusOutOfPanel(PANEL_WINDOW, 'backward')).toBe(true);
    expect(activeId()).toBe('before');
  });

  test('forward focuses the element after the panel, stepping over the overlay', async () => {
    const app = await import(APP);
    page();
    expect(app.moveFocusOutOfPanel(PANEL_WINDOW, 'forward')).toBe(true);
    expect(activeId()).toBe('after');
  });

  test("never lands inside the panel's own anchor", async () => {
    const app = await import(APP);
    const { before } = page();
    before.remove();
    expect(app.moveFocusOutOfPanel(PANEL_WINDOW, 'backward')).toBe(false);
    expect(activeId()).toBe('');
  });

  test('skips elements that Tab cannot reach', async () => {
    const app = await import(APP);
    const { anchor } = page();
    const hidden = button('hidden');
    hidden.style.display = 'none';
    const disabled = button('disabled');
    disabled.disabled = true;
    const untabbable = button('untabbable');
    untabbable.tabIndex = -1;
    const inert = document.createElement('div');
    inert.hidden = true;
    inert.appendChild(button('under-hidden'));
    anchor.after(hidden, disabled, untabbable, inert);
    expect(app.moveFocusOutOfPanel(PANEL_WINDOW, 'forward')).toBe(true);
    expect(activeId()).toBe('after');
  });

  test('with nothing on that side, focus stays put', async () => {
    const app = await import(APP);
    const { before, after } = page();
    after.remove();
    document.getElementById('overlay')?.remove();
    before.focus();
    expect(app.moveFocusOutOfPanel(PANEL_WINDOW, 'forward')).toBe(false);
    expect(activeId()).toBe('before');
  });
});

describe('canopy-host-focus-exit: the message', () => {
  test('a same-origin message from the panel frame moves focus', async () => {
    const app = await import(APP);
    page();
    app.initIframePasteBridge();
    post({ type: 'canopy-host-focus-exit', direction: 'forward' }, PANEL_WINDOW);
    expect(activeId()).toBe('after');
    expect(state.askedBy).toBe(PANEL_WINDOW);
  });

  test("a viewer's old osprey-focus-exit name is read for one release", async () => {
    const app = await import(APP);
    page();
    app.initIframePasteBridge();
    post({ type: 'osprey-focus-exit', direction: 'forward' }, PANEL_WINDOW);
    expect(activeId()).toBe('after');
  });

  test('a window that is not a panel frame is ignored', async () => {
    const app = await import(APP);
    page();
    app.initIframePasteBridge();
    post({ type: 'canopy-host-focus-exit', direction: 'forward' }, { other: true });
    post({ type: 'canopy-host-focus-exit', direction: 'forward' }, null);
    expect(activeId()).toBe('');
  });

  test('a message from another origin is ignored, even from the panel frame', async () => {
    const app = await import(APP);
    page();
    app.initIframePasteBridge();
    post(
      { type: 'canopy-host-focus-exit', direction: 'forward' },
      PANEL_WINDOW,
      'https://evil.example'
    );
    expect(activeId()).toBe('');
    expect(state.askedBy).toBeNull();
  });

  test('an unknown direction is ignored', async () => {
    const app = await import(APP);
    page();
    app.initIframePasteBridge();
    post({ type: 'canopy-host-focus-exit', direction: 'sideways' }, PANEL_WINDOW);
    expect(activeId()).toBe('');
  });
});
