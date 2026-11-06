// @vitest-environment jsdom
/**
 * The tile header's Pop out (attached-popout.js): a service panel's overlay
 * iframe, or the Agent pane's live subtree, moves into an attached window;
 * the tile keeps its place and shows a stand-in; the content comes back when
 * the window closes. The layout is never asked to change — no tile closes,
 * none opens — except to give back a tile the operator closed meanwhile.
 *
 * The window itself is attached-window.js, run for real against a stand-in
 * `window.open`; the dock modules around it are doubles.
 *
 *   npx vitest run tests/interfaces/web_terminal/attached-popout.test.mjs
 */
import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const iframes = vi.hoisted(() => ({
  managedIframe: vi.fn(),
  setIframeStandIn: vi.fn(),
  isPanelDocked: vi.fn(() => true),
}));
const dock = vi.hoisted(() => ({
  agentPaneContent: vi.fn(() => /** @type {HTMLElement | null} */ (null)),
  isTerminalPanelOpen: vi.fn(() => true),
  openTerminalPanel: vi.fn(),
  closeTerminalPanel: vi.fn(),
  closeDockPanel: vi.fn(),
  resetDockLayout: vi.fn(),
}));
const term = vi.hoisted(() => ({ fitTerminal: vi.fn() }));
const notice = vi.hoisted(() => ({ showUndoNotice: vi.fn() }));
const popout = vi.hoisted(() => ({ redockPanel: vi.fn() }));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js', () => iframes);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => dock);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => term);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/undo-notice.js', () => notice);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/popout.js', () => popout);

const mod = await import(`${JS}/attached-popout.js`);
const win = await import(`${JS}/attached-window.js`);

/** A popup: its own document, `close()` firing `pagehide`. */
function fakePopup() {
  const target = new EventTarget();
  const popup = {
    document: document.implementation.createHTMLDocument(''),
    closed: false,
    focus: vi.fn(),
    close: vi.fn(() => {
      if (popup.closed) return;
      popup.closed = true;
      target.dispatchEvent(new Event('pagehide'));
    }),
    addEventListener: target.addEventListener.bind(target),
    removeEventListener: target.removeEventListener.bind(target),
    dispatchEvent: target.dispatchEvent.bind(target),
  };
  return popup;
}

/** @type {ReturnType<typeof fakePopup>[]} */
let popups;
/** @type {HTMLElement} */
let overlay;
/** @type {HTMLIFrameElement} */
let frame;
/** @type {HTMLElement} */
let host;
/** @type {HTMLElement} */
let agent;

beforeEach(() => {
  vi.clearAllMocks();
  document.body.replaceChildren();
  popups = [];
  vi.stubGlobal(
    'open',
    vi.fn(() => {
      const popup = fakePopup();
      popups.push(popup);
      return popup;
    })
  );
  overlay = document.createElement('div');
  overlay.className = 'dock-iframe-overlay';
  frame = document.createElement('iframe');
  frame.className = 'panel-iframe';
  frame.dataset.panelId = 'ariel';
  overlay.append(frame);
  host = document.createElement('div');
  host.className = 'dock-panel-host';
  agent = document.createElement('div');
  agent.className = 'terminal-panel';
  host.append(agent);
  document.body.append(overlay, host);
  iframes.managedIframe.mockImplementation((/** @type {string} */ id) =>
    id === 'ariel' ? { iframe: frame, title: 'ARIEL' } : null
  );
  iframes.isPanelDocked.mockReturnValue(true);
  dock.agentPaneContent.mockReturnValue(agent);
  dock.isTerminalPanelOpen.mockReturnValue(true);
});

afterEach(() => {
  window.dispatchEvent(new Event('pagehide'));
  vi.unstubAllGlobals();
});

/** Nothing in the layout moved: no tile closed, none opened. */
function expectLayoutUntouched() {
  expect(dock.closeTerminalPanel).not.toHaveBeenCalled();
  expect(dock.closeDockPanel).not.toHaveBeenCalled();
  expect(dock.openTerminalPanel).not.toHaveBeenCalled();
  expect(dock.resetDockLayout).not.toHaveBeenCalled();
  expect(popout.redockPanel).not.toHaveBeenCalled();
}

describe('a service panel', () => {
  test('its iframe moves into the window and the tile draws a stand-in in its place', () => {
    expect(mod.popOutAttached('ariel')).toBe(true);
    const popup = popups[0];
    expect(popup.document.title).toBe('Ariel · OSPREY');
    expect(popup.document.querySelector('iframe.panel-iframe')).toBe(frame);
    const standIn = /** @type {HTMLElement} */ (overlay.querySelector('.attached-standin'));
    expect(standIn.textContent).toContain('Ariel is in its own window.');
    expect(iframes.setIframeStandIn).toHaveBeenCalledExactlyOnceWith('ariel', standIn);
    expect(mod.isAttached('ariel')).toBe(true);
    expectLayoutUntouched();
  });

  test('closing the window brings the iframe back to its tile, layout untouched', () => {
    mod.popOutAttached('ariel');
    popups[0].close();
    expect(frame.parentElement).toBe(overlay);
    expect(overlay.querySelector('.attached-standin')).toBeNull();
    expect(iframes.setIframeStandIn).toHaveBeenLastCalledWith('ariel', null);
    expectLayoutUntouched();
  });

  test('a tile closed meanwhile is given back on return, by the open rule', () => {
    mod.popOutAttached('ariel');
    iframes.isPanelDocked.mockReturnValue(false);
    popups[0].close();
    expect(popout.redockPanel).toHaveBeenCalledExactlyOnceWith('ariel');
  });

  test('the hub unloading returns the iframe and opens no tile', () => {
    mod.popOutAttached('ariel');
    iframes.isPanelDocked.mockReturnValue(false);
    window.dispatchEvent(new Event('pagehide'));
    expect(frame.parentElement).toBe(overlay);
    expect(popups[0].closed).toBe(true);
    expect(popout.redockPanel).not.toHaveBeenCalled();
  });

  test('the stand-in brings the panel back, and shows its window', () => {
    mod.popOutAttached('ariel');
    const standIn = /** @type {HTMLElement} */ (overlay.querySelector('.attached-standin'));
    /** @type {HTMLButtonElement} */ (standIn.querySelector('.attached-standin-show')).click();
    expect(popups[0].focus).toHaveBeenCalledTimes(1);
    /** @type {HTMLButtonElement} */ (standIn.querySelector('.attached-standin-back')).click();
    expect(popups[0].closed).toBe(true);
    expect(frame.parentElement).toBe(overlay);
  });

  test('a second Pop out brings the window forward', () => {
    mod.popOutAttached('ariel');
    expect(mod.popOutAttached('ariel')).toBe(true);
    expect(popups).toHaveLength(1);
    expect(popups[0].focus).toHaveBeenCalledTimes(1);
  });

  test('an opaque-origin panel, or one without its iframe yet, cannot go', () => {
    frame.dataset.panelFrame = 'opaque';
    expect(mod.attachBlocker('ariel')).toBe(
      'This panel runs at an opaque origin and cannot be popped out.'
    );
    expect(mod.popOutAttached('ariel')).toBe(false);
    expect(mod.attachBlocker('notes')).toBe('This panel has not loaded yet.');
    expect(mod.canAttach('notes')).toBe(false);
    expect(popups).toHaveLength(0);
  });

  test('a refused window says so and leaves the tile as it was', () => {
    vi.stubGlobal(
      'open',
      vi.fn(() => null)
    );
    expect(mod.popOutAttached('ariel')).toBe(false);
    expect(notice.showUndoNotice).toHaveBeenCalledExactlyOnceWith({
      message: 'Pop-ups are blocked for this site.',
    });
    expect(frame.parentElement).toBe(overlay);
    expect(iframes.setIframeStandIn).not.toHaveBeenCalled();
  });
});

describe('the Agent pane', () => {
  test('its live subtree moves into the window, the tile keeps a stand-in, and the terminal refits', () => {
    const child = document.createElement('textarea');
    agent.append(child);
    expect(mod.popOutAttached('terminal')).toBe(true);
    const popup = popups[0];
    expect(popup.document.title).toBe('Agent · OSPREY');
    expect(popup.document.querySelector('.terminal-panel > textarea')).toBe(child);
    expect(host.querySelector('.attached-standin')?.textContent).toContain(
      'Agent is in its own window.'
    );
    expect(term.fitTerminal).toHaveBeenCalled();
    popup.dispatchEvent(new Event('resize'));
    expect(term.fitTerminal).toHaveBeenCalledTimes(2);
    expectLayoutUntouched();
  });

  test('closing the window puts it back in its tile', () => {
    mod.popOutAttached('terminal');
    popups[0].close();
    expect(agent.parentElement).toBe(host);
    expect(host.querySelector('.attached-standin')).toBeNull();
    expectLayoutUntouched();
  });

  test('an Agent tile closed meanwhile is reopened when the pane comes back', () => {
    mod.popOutAttached('terminal');
    dock.isTerminalPanelOpen.mockReturnValue(false);
    popups[0].close();
    expect(dock.openTerminalPanel).toHaveBeenCalledTimes(1);
  });

  test('cannot pop out before the dock has the pane', () => {
    dock.agentPaneContent.mockReturnValue(null);
    expect(mod.canAttach('terminal')).toBe(false);
    expect(mod.popOutAttached('terminal')).toBe(false);
  });

  test('Bring back returns it and closes the window', () => {
    mod.popOutAttached('terminal');
    mod.bringBackAttached('terminal');
    expect(popups[0].closed).toBe(true);
    expect(agent.parentElement).toBe(host);
    expect(win.isAttached('terminal')).toBe(false);
  });
});
