// @ts-check
/**
 * Per-persona scoping of the settings drawer's warning acknowledgement:
 *   npx vitest run tests/interfaces/web_terminal/js/storage-scope-settings.test.js
 *
 * The drawer is gated behind a "these settings control agent behavior, safety
 * hooks, and security policies" dialog, acknowledged once per SERVER session and
 * remembered in localStorage. On a shared origin that memory was one slot for
 * everybody: whoever acknowledged first silently waved the warning away for
 * every other persona on the box. The ack is now keyed per persona.
 *
 * The gate is driven through `openDrawerTab()` (the exported seam that routes
 * through it) and observed through the dialog's own presence in the DOM, which
 * is exactly what the ack decides. `/health` is stubbed — the module's only
 * network call on this path — so nothing dials out.
 *
 * Lives apart from storage-scope-keys.test.js because mocking api.js is
 * file-wide, and the modules pinned there must run against the real one.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const health = vi.hoisted(() => ({ sessionId: /** @type {string|null} */ ('server-session-1') }));

// api.js is stubbed whole: settings.js takes fetchJSON/apiRequest from it, and
// terminal.js (imported by settings.js) takes the WebSocket helpers, so every
// name either module imports has to exist on the mock. The path is spelled out
// rather than held in a const — vi.mock is hoisted above every binding.
vi.mock('../../../../src/osprey/interfaces/web_terminal/static/js/api.js', () => ({
  fetchJSON: vi.fn(async (path) => {
    if (path === '/health') return { session_id: health.sessionId };
    return {};
  }),
  apiRequest: vi.fn(async () => ({})),
  createWebSocket: vi.fn(() => ({ send() {}, close() {} })),
  wsUrl: vi.fn((path) => `ws://localhost${path}`),
  withPrefix: vi.fn((path) => path),
}));

const { openDrawerTab } = await import(
  '../../../../src/osprey/interfaces/web_terminal/static/js/settings.js'
);

const SCOPE_ATTR = 'data-osprey-storage-scope';
const BASE = 'osprey-settings-warning-ack';

/** Serve this document as `user` would be served on a multi-user mount.
 * @param {string} user */
function serveAs(user) {
  document.documentElement.setAttribute(SCOPE_ATTR, user);
}

/** Let the /health race and the gate's awaits settle. */
async function settleGate() {
  for (let i = 0; i < 3; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

/** @returns {boolean} whether the warning dialog is currently mounted. */
function warningShown() {
  return document.querySelector('.settings-warning-dialog') !== null;
}

/** Dismiss whatever dialog is up, so the gate's pending flag always resets.
 * @param {'proceed'|'cancel'} how */
function dismiss(how) {
  const btn = document.querySelector(`.settings-warning-${how}`);
  if (!(btn instanceof HTMLElement)) throw new Error(`expected the ${how} button`);
  btn.click();
}

beforeEach(() => {
  localStorage.clear();
  document.body.innerHTML = '';
  document.documentElement.removeAttribute(SCOPE_ATTR);
  health.sessionId = 'server-session-1';
});

afterEach(() => {
  // The gate's pending flag only clears on a dismissal, so a test that fails
  // with a dialog up would otherwise wedge every later one. Settle it here.
  const cancel = document.querySelector('.settings-warning-cancel');
  if (cancel instanceof HTMLElement) cancel.click();
  document.documentElement.removeAttribute(SCOPE_ATTR);
});

describe('settings warning acknowledgement', () => {
  test("another persona's acknowledgement does not wave the warning away", async () => {
    // The shared slot: someone else already clicked Proceed on this server.
    localStorage.setItem(BASE, 'server-session-1');
    serveAs('bob');

    const gate = openDrawerTab('config');
    await settleGate();

    expect(warningShown()).toBe(true);
    dismiss('cancel');
    await gate;
  });

  test('this persona\'s own acknowledgement is honoured', async () => {
    localStorage.setItem(`${BASE}--bob`, 'server-session-1');
    serveAs('bob');

    const gate = openDrawerTab('config');
    await settleGate();

    expect(warningShown()).toBe(false);
    await gate;
  });

  test('Proceed records the acknowledgement under this persona\'s key', async () => {
    serveAs('bob');

    const gate = openDrawerTab('config');
    await settleGate();
    expect(warningShown()).toBe(true);

    dismiss('proceed');
    await gate;

    expect(localStorage.getItem(`${BASE}--bob`)).toBe('server-session-1');
    expect(localStorage.getItem(BASE)).toBe(null);
  });

  test('unscoped serving reads and writes the legacy key unchanged', async () => {
    const first = openDrawerTab('config');
    await settleGate();
    expect(warningShown()).toBe(true);
    dismiss('proceed');
    await first;

    expect(localStorage.getItem(BASE)).toBe('server-session-1');

    // And the stored ack now short-circuits the gate, exactly as before.
    document.body.innerHTML = '';
    const second = openDrawerTab('config');
    await settleGate();
    expect(warningShown()).toBe(false);
    await second;
  });
});
