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
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  getCurrentSessionId: () => mocks.sessionId,
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
});

test('a move on the shown key re-sends that key to every panel', () => {
  const panels = [fakePanel(), fakePanel()];

  relayTranscriptMove(
    'sess-1',
    panels.map((p) => p.state)
  );

  for (const { postMessage } of panels) {
    expect(postMessage).toHaveBeenCalledTimes(1);
    expect(postMessage).toHaveBeenCalledWith(change, window.location.origin);
  }
});

test('a move on another key, or on no key, sends nothing', () => {
  const panel = fakePanel();

  relayTranscriptMove('sess-2', [panel.state]);
  relayTranscriptMove('', [panel.state]);

  expect(panel.postMessage).not.toHaveBeenCalled();
});

test('with no shown session a move sends nothing', () => {
  mocks.sessionId = null;
  const panel = fakePanel();

  relayTranscriptMove('sess-1', [panel.state]);

  expect(panel.postMessage).not.toHaveBeenCalled();
});
