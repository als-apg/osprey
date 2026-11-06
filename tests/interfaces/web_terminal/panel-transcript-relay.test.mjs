// @ts-check
/**
 * The hub's transcript-move fan-out (panel-transcript-relay.js):
 *   npx vitest run tests/interfaces/web_terminal/panel-transcript-relay.test.mjs
 *
 * A move on the key the hub shows tells every panel iframe and every pop-out
 * window that same key again; a move on any other key, or on no key, tells
 * nobody. The end-to-end wiring through panel-manager is pinned in
 * panel-manager.test.mjs.
 */

import { test, expect, beforeEach, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  sessionId: /** @type {string | null} */ ('sess-1'),
  popouts: /** @type {Array<{ closed: boolean, postMessage: any }>} */ ([]),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  getCurrentSessionId: () => mocks.sessionId,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/popout.js', () => ({
  broadcastToPopouts: (/** @type {(win: any) => void} */ send) => {
    for (const win of mocks.popouts) if (!win.closed) send(win);
  },
}));
vi.mock('/design-system/js/theme-manager.js', () => ({ getTheme: () => 'dark' }));

const { relayTranscriptMove } =
  await import('../../../src/osprey/interfaces/web_terminal/static/js/panel-transcript-relay.js');

/** A stand-in panel state whose iframe window records what it was sent. */
function fakePanel() {
  const postMessage = vi.fn();
  return {
    state: {
      iframe: /** @type {HTMLIFrameElement} */ (
        /** @type {unknown} */ ({ contentWindow: { postMessage } })
      ),
    },
    postMessage,
  };
}

const change = { type: 'osprey-session-change', session_id: 'sess-1' };

beforeEach(() => {
  mocks.sessionId = 'sess-1';
  mocks.popouts = [];
});

test('a move on the shown key re-sends that key to every panel and open pop-out', () => {
  const panels = [fakePanel(), fakePanel()];
  const popout = { closed: false, postMessage: vi.fn() };
  const closed = { closed: true, postMessage: vi.fn() };
  mocks.popouts = [popout, closed];

  relayTranscriptMove(
    'sess-1',
    panels.map((p) => p.state)
  );

  for (const { postMessage } of panels) {
    expect(postMessage).toHaveBeenCalledTimes(1);
    expect(postMessage).toHaveBeenCalledWith(change, window.location.origin);
  }
  expect(popout.postMessage).toHaveBeenCalledTimes(1);
  expect(popout.postMessage).toHaveBeenCalledWith(change, window.location.origin);
  expect(closed.postMessage).not.toHaveBeenCalled();
});

test('a move on another key, or on no key, sends nothing', () => {
  const panel = fakePanel();
  const popout = { closed: false, postMessage: vi.fn() };
  mocks.popouts = [popout];

  relayTranscriptMove('sess-2', [panel.state]);
  relayTranscriptMove('', [panel.state]);

  expect(panel.postMessage).not.toHaveBeenCalled();
  expect(popout.postMessage).not.toHaveBeenCalled();
});

test('with no shown session a move sends nothing', () => {
  mocks.sessionId = null;
  const panel = fakePanel();

  relayTranscriptMove('sess-1', [panel.state]);

  expect(panel.postMessage).not.toHaveBeenCalled();
});
