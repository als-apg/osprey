// @ts-check
/**
 * The hub's single-iframe sends (panel-iframe-sync.js):
 *   npx vitest run tests/interfaces/web_terminal/panel-iframe-sync.test.mjs
 *
 * Every send funnels through one guarded postMessage: a frame with no window
 * yet (not loaded, detached) is skipped, a cross-origin throw is swallowed, and
 * every message but the theme is addressed to the hub's own origin, so an
 * opaque panel never receives the session id, the mode, a header action or an
 * artifact message. The embed src carries the load-time half of the same
 * state.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  sessionId: /** @type {string | null} */ ('sess-1'),
  theme: 'dark',
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  getCurrentSessionId: () => mocks.sessionId,
}));
vi.mock('/design-system/js/theme-manager.js', () => ({ getTheme: () => mocks.theme }));

const sync =
  await import('../../../src/osprey/interfaces/web_terminal/static/js/panel-iframe-sync.js');

/** A stand-in iframe whose window records what it was sent. */
function fakeFrame() {
  const postMessage = vi.fn();
  return {
    iframe: /** @type {HTMLIFrameElement} */ (
      /** @type {unknown} */ ({ contentWindow: { postMessage } })
    ),
    postMessage,
  };
}

beforeEach(() => {
  mocks.sessionId = 'sess-1';
  mocks.theme = 'dark';
  document.documentElement.setAttribute('data-ui-mode', 'simple');
});

afterEach(() => {
  document.documentElement.removeAttribute('data-ui-mode');
  vi.restoreAllMocks();
});

describe('the guarded send', () => {
  test('a frame with no window yet, or no frame at all, is skipped', () => {
    const noWindow = /** @type {HTMLIFrameElement} */ (
      /** @type {unknown} */ ({ contentWindow: null })
    );
    expect(() => sync.sendArtifactSelectToIframe(noWindow, 'a1')).not.toThrow();
    expect(() => sync.sendArtifactSelectToIframe(null, 'a1')).not.toThrow();
  });

  test('a throwing cross-origin window is swallowed', () => {
    const { iframe, postMessage } = fakeFrame();
    postMessage.mockImplementation(() => {
      throw new Error('SecurityError');
    });
    expect(() => sync.sendModeToIframe(iframe)).not.toThrow();
  });
});

describe('the state messages', () => {
  test('the theme goes to any origin, so an opaque panel follows it too', () => {
    const { iframe, postMessage } = fakeFrame();
    sync.sendThemeToIframe(iframe);
    expect(postMessage).toHaveBeenCalledWith({ type: 'osprey-theme-change', theme: 'dark' }, '*');
  });

  test('the session goes to the hub origin only, and only once there is one', () => {
    const { iframe, postMessage } = fakeFrame();
    sync.sendSessionToIframe(iframe);
    expect(postMessage).toHaveBeenCalledWith(
      { type: 'osprey-session-change', session_id: 'sess-1' },
      window.location.origin
    );
    mocks.sessionId = null;
    postMessage.mockClear();
    sync.sendSessionToIframe(iframe);
    expect(postMessage).not.toHaveBeenCalled();
  });

  test('the mode is read off <html>, expert when unstamped', () => {
    const { iframe, postMessage } = fakeFrame();
    sync.sendModeToIframe(iframe);
    expect(postMessage).toHaveBeenLastCalledWith(
      { type: 'osprey-mode-change', mode: 'simple' },
      window.location.origin
    );
    document.documentElement.removeAttribute('data-ui-mode');
    sync.sendModeToIframe(iframe);
    expect(postMessage).toHaveBeenLastCalledWith(
      { type: 'osprey-mode-change', mode: 'expert' },
      window.location.origin
    );
  });

  test('a header action carries its value only when it has one', () => {
    const { iframe, postMessage } = fakeFrame();
    sync.sendHeaderActionToIframe(iframe, 'refresh');
    sync.sendHeaderActionToIframe(iframe, 'browse-mode', 'activity');
    expect(postMessage.mock.calls.map((call) => call[0])).toEqual([
      { type: 'osprey-header-action', id: 'refresh' },
      { type: 'osprey-header-action', id: 'browse-mode', value: 'activity' },
    ]);
  });
});

describe('the artifacts viewer messages', () => {
  test('select names the artifact to show, or null for the empty viewer', () => {
    const { iframe, postMessage } = fakeFrame();
    sync.sendArtifactSelectToIframe(iframe, 'a1');
    sync.sendArtifactSelectToIframe(iframe, null);
    expect(postMessage.mock.calls).toEqual([
      [{ type: 'osprey-artifact-select', artifact_id: 'a1' }, window.location.origin],
      [{ type: 'osprey-artifact-select', artifact_id: null }, window.location.origin],
    ]);
  });

  test('an action names the artifact and the verb, for the hub origin only', () => {
    const { iframe, postMessage } = fakeFrame();
    sync.sendArtifactActionToIframe(iframe, 'a1', 'delete');
    expect(postMessage).toHaveBeenCalledWith(
      { type: 'osprey-artifact-action', artifact_id: 'a1', action: 'delete' },
      window.location.origin
    );
  });

  test('a change names the artifact and carries a copy only for an update', () => {
    const { iframe, postMessage } = fakeFrame();
    const copy = { id: 'a1', title: 'Tunes report', pinned: true };
    sync.sendArtifactChangedToIframe(iframe, 'a1', 'updated', copy);
    sync.sendArtifactChangedToIframe(iframe, 'a1', 'deleted');
    expect(postMessage.mock.calls).toEqual([
      [
        { type: 'osprey-artifact-changed', artifact_id: 'a1', change: 'updated', artifact: copy },
        window.location.origin,
      ],
      [
        { type: 'osprey-artifact-changed', artifact_id: 'a1', change: 'deleted' },
        window.location.origin,
      ],
    ]);
    expect(Object.keys(postMessage.mock.calls[1][0])).not.toContain('artifact');
  });
});

describe('buildEmbedSrc', () => {
  test('stamps embedded, theme and mode and keeps a prefixed path verbatim', () => {
    const src = new URL(sync.buildEmbedSrc('/u/alice/panel/artifacts/?x=1'));
    expect(src.pathname).toBe('/u/alice/panel/artifacts/');
    expect(src.searchParams.get('x')).toBe('1');
    expect(src.searchParams.get('embedded')).toBe('true');
    expect(src.searchParams.get('theme')).toBe('dark');
    expect(src.searchParams.get('mode')).toBe('simple');
  });
});
