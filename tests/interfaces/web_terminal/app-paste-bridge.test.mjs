// @ts-check
/**
 * Unit tests for the hub's iframe paste bridge (app.js's `initIframePasteBridge`):
 *   npx vitest run tests/interfaces/web_terminal/app-paste-bridge.test.mjs
 *
 * A panel rendered in an iframe cannot type for the operator, so it asks the
 * hub to: it posts `{type:'osprey-paste-to-terminal', text}` to `window.parent`
 * and the hub puts that text where the operator is composing. The contract this
 * file pins is that the text ARRIVES and nothing else happens — it is never
 * sent, it never replaces a half-written prompt in the Simple view, and a
 * message from another origin is not text at all.
 *
 * Both views are covered because the hub has two places to compose in and the
 * panel knows neither: expert types into the terminal, Simple into the chat
 * textarea, and routing between them is `insertPrompt`'s job (first-contact.js).
 *
 * Seams: terminal.js is mocked whole — it is the module app.js's import graph
 * reaches for the terminal side of that routing, and a real one would want
 * xterm.js and a socket. app.js is imported once and statically, exactly as in
 * app-logout.test.mjs: its DOMContentLoaded bootstrap has already passed by the
 * time a test module evaluates, and `initIframePasteBridge` is exported so it
 * can be driven directly instead.
 */

import { afterEach, beforeAll, beforeEach, describe, expect, test, vi } from 'vitest';

/** Mutable stand-in for terminal.js, reachable from the hoisted vi.mock factory. */
const term = vi.hoisted(() => ({
  paste: vi.fn(),
  focus: vi.fn(),
  /** @type {(() => void)[]} */
  sessionListeners: [],
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
  focusTerminal: term.focus,
  pasteToTerminal: term.paste,
  /** @param {() => void} fn */
  onSessionChange: (fn) => term.sessionListeners.push(fn),
}));

import { initIframePasteBridge } from '../../../src/osprey/interfaces/web_terminal/static/js/app.js';

/** The Simple view's input, in the shape the insert seam looks for. */
function mountSimpleInput() {
  document.body.innerHTML =
    '<div id="operator-container"><div class="op-input-area"><textarea></textarea></div></div>';
  return /** @type {HTMLTextAreaElement} */ (
    document.querySelector('#operator-container .op-input-area textarea')
  );
}

/** What the Channel Finder panel posts: addresses space-joined on one line. */
const ADDRESSES = 'SR:BPM1:X SR:BPM2:X SR:BPM3:X';

/**
 * Deliver one postMessage to the hub's listener.
 * @param {unknown} data
 * @param {string} [origin]
 */
function post(data, origin = window.location.origin) {
  window.dispatchEvent(new MessageEvent('message', { data, origin }));
}

// Once, as boot does it: the listener is bound to the shared `window` and the
// bridge has no teardown, so re-arming it per test would leave every earlier
// test's listener live and each message would be handled several times.
beforeAll(() => {
  initIframePasteBridge();
});

beforeEach(() => {
  document.body.innerHTML = '';
  document.documentElement.removeAttribute('data-ui-mode');
  vi.clearAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('initIframePasteBridge: expert view', () => {
  test('pastes the panel text into the terminal, once, verbatim', () => {
    document.documentElement.setAttribute('data-ui-mode', 'expert');

    post({ type: 'osprey-paste-to-terminal', text: ADDRESSES });

    expect(term.paste).toHaveBeenCalledTimes(1);
    expect(term.paste).toHaveBeenCalledWith(ADDRESSES);
    // One line: a newline would be an Enter the operator never pressed.
    expect(term.paste.mock.calls[0][0]).not.toContain('\n');
    expect(term.focus).toHaveBeenCalled();
  });
});

describe('initIframePasteBridge: Simple view', () => {
  test('appends the addresses under what the operator is already writing', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    const input = mountSimpleInput();
    input.value = 'set these to nominal:';

    post({ type: 'osprey-paste-to-terminal', text: ADDRESSES });

    expect(input.value).toBe(`set these to nominal:\n${ADDRESSES}`);
    expect(term.paste).not.toHaveBeenCalled();
  });

  test('a second panel message adds a second line, losing neither', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    const input = mountSimpleInput();

    post({ type: 'osprey-paste-to-terminal', text: 'SR:BPM1:X' });
    post({ type: 'osprey-paste-to-terminal', text: 'SR:BPM2:X' });

    expect(input.value).toBe('SR:BPM1:X\nSR:BPM2:X');
  });

  test('nothing is submitted: the operator still sends the turn', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    const input = mountSimpleInput();
    input.value = 'set these to nominal:';
    let submits = 0;
    document.addEventListener(
      'submit',
      () => {
        submits += 1;
      },
      true
    );

    post({ type: 'osprey-paste-to-terminal', text: ADDRESSES });

    expect(submits).toBe(0);
    expect(input.value.endsWith('\n')).toBe(false);
  });
});

describe('initIframePasteBridge: what it refuses', () => {
  test('ignores a message from another origin', () => {
    document.documentElement.setAttribute('data-ui-mode', 'expert');

    post({ type: 'osprey-paste-to-terminal', text: ADDRESSES }, 'https://evil.example');

    expect(term.paste).not.toHaveBeenCalled();
  });

  test('ignores an empty text, so a stray post cannot blank the input', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    const input = mountSimpleInput();
    input.value = 'half a question';

    post({ type: 'osprey-paste-to-terminal', text: '' });

    expect(input.value).toBe('half a question');
    expect(term.paste).not.toHaveBeenCalled();
  });

  test('ignores another message type entirely', () => {
    document.documentElement.setAttribute('data-ui-mode', 'expert');

    post({ type: 'osprey-something-else', text: ADDRESSES });

    expect(term.paste).not.toHaveBeenCalled();
  });
});
