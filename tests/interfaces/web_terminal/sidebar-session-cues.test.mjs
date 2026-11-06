// @ts-check
/**
 * The session rows' cues (sidebar-session-cues.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-session-cues.test.mjs
 *
 * Pins busy from the chat bus (a redraw only on a change, cleared on abort),
 * the accent held on the session being switched to until the switch settles,
 * the peek's refresh, and "+" opening the pane before a new session.
 * The tree's suite drives the same cues through a real tree.
 */

import { test, expect, beforeEach, vi } from 'vitest';

const m = vi.hoisted(() => ({
  /** @type {((detail: any) => void)[]} */
  busy: [],
  off: vi.fn(),
  isTerminalPanelOpen: vi.fn(() => true),
  openTerminalPanel: vi.fn(),
  switchTo: vi.fn(async () => true),
  peek: { watch: vi.fn(), refresh: vi.fn() },
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/chat-events.js', () => ({
  onChat: (/** @type {string} */ name, /** @type {(d: any) => void} */ fn) => {
    if (name === 'busy') m.busy.push(fn);
    return m.off;
  },
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  isTerminalPanelOpen: m.isTerminalPanelOpen,
  openTerminalPanel: m.openTerminalPanel,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/session-model.js', () => ({
  switchTo: m.switchTo,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-session-peek.js', () => ({
  createSessionPeek: () => m.peek,
}));

const { createSessionCues, NEW_SESSION_ID } =
  await import('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-session-cues.js');

/** @type {import('vitest').Mock<() => void>} */
let render;
/** @type {AbortController} */
let abort;
/** @type {ReturnType<typeof createSessionCues>} */
let cues;

beforeEach(() => {
  vi.clearAllMocks();
  m.busy = [];
  m.isTerminalPanelOpen.mockReturnValue(true);
  document.body.innerHTML = `<button id="${NEW_SESSION_ID}"></button>`;
  render = vi.fn(() => {});
  abort = new AbortController();
  cues = createSessionCues({ render, peek: /** @type {any} */ ({}) });
  cues.watch(abort.signal);
});

test('busy follows the chat bus and redraws only when it changes; abort clears it', () => {
  expect(cues.busy()).toBe(false);
  m.busy[0]({ busy: true });
  m.busy[0]({ busy: true });
  expect(cues.busy()).toBe(true);
  expect(render).toHaveBeenCalledOnce();
  abort.abort();
  expect(m.off).toHaveBeenCalled();
  expect(cues.busy()).toBe(false);
});

test('select holds the accent on the session until the switch settles, and passes its answer', async () => {
  /** @type {(v: boolean) => void} */
  let finish = () => {};
  const run = () => new Promise((resolve) => (finish = resolve));
  const done = cues.select('t2', /** @type {any} */ (run));
  expect(cues.selecting()).toBe('t2');
  expect(render).toHaveBeenCalledOnce();
  finish(false);
  expect(await done).toBe(false);
  expect(cues.selecting()).toBeNull();
});

test('refresh redraws the peek, which watches the same signal', () => {
  cues.refresh();
  expect(m.peek.refresh).toHaveBeenCalledOnce();
  expect(m.peek.watch).toHaveBeenCalledWith(abort.signal);
});

test('"+" opens a closed Agent pane, then starts a new session', () => {
  m.isTerminalPanelOpen.mockReturnValue(false);
  /** @type {HTMLElement} */ (document.getElementById(NEW_SESSION_ID)).click();
  expect(m.openTerminalPanel).toHaveBeenCalledOnce();
  expect(m.switchTo).toHaveBeenCalledWith({ new: true });
  m.isTerminalPanelOpen.mockReturnValue(true);
  /** @type {HTMLElement} */ (document.getElementById(NEW_SESSION_ID)).click();
  expect(m.openTerminalPanel).toHaveBeenCalledOnce();
});
