// @ts-check
/**
 * Unit tests for the session half of a change of the Agent tile's surface
 * (app.js's `initSurfaceFollowUps`):
 *   ./node_modules/.bin/vitest run tests/interfaces/web_terminal/app-surface-handoff.test.mjs
 *
 * The chat and the terminal are windows onto one session key, and the server
 * hands that key to whichever surface asks for it — so the change is what moves
 * the live agent. Opening the terminal resumes the key over the terminal socket;
 * going back to chat drops that socket and lets the console take the key over.
 * What this file pins is that neither direction reloads the page, and that two
 * quick changes produce two hand-offs one after the other rather than two
 * acquires racing for the same key.
 *
 * Seams: terminal.js and chat.js are mocked whole — one wants xterm.js and a
 * socket, the other a server to hand the key over. The surface change itself
 * goes through the real agent-surface.js. sidebar-artifacts.js is mocked: app.js
 * only calls its init, which this file never reaches.
 */

import { afterEach, beforeAll, beforeEach, describe, expect, test, vi } from 'vitest';

// Nothing serves this environment, so no request is answered: any fetch is a
// dependency this file has not declared, and fails loudly. Stubbed from
// `vi.hoisted`, which runs before the static imports are evaluated.
vi.hoisted(() => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url) => {
      throw new Error(`unstubbed fetch: ${url}`);
    })
  );
});

/** Every hand-off step that ran, in the order it ran. */
const calls = vi.hoisted(() => /** @type {string[]} */ ([]));

/** Stand-ins for the two surfaces, reachable from the hoisted mock factories. */
const surfaces = vi.hoisted(() => ({
  /** Resolves the acquire by default; a test may hold it open. */
  startTerminalSurface: vi.fn(() => {
    calls.push('startTerminalSurface');
    return Promise.resolve();
  }),
  /** Resolves on the dropped socket's close by default; a test may hold it. */
  stopTerminal: vi.fn(() => {
    calls.push('stopTerminal');
    return Promise.resolve();
  }),
  /** Resolves the hand-off by default; a test may hold it open. */
  enterFromTerminal: vi.fn(() => {
    calls.push('enterFromTerminal');
    return Promise.resolve();
  }),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  initTerminal: vi.fn(),
  startTerminal: vi.fn(),
  startTerminalSurface: surfaces.startTerminalSurface,
  stopTerminal: surfaces.stopTerminal,
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

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/chat.js', () => ({
  initChat: vi.fn(),
  enterFromTerminal: surfaces.enterFromTerminal,
  transportNotice: vi.fn(() => ''),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifacts.js', () => ({
  initSidebarArtifacts: vi.fn(),
  teardownSidebarArtifacts: vi.fn(),
}));

import { initSurfaceFollowUps } from '../../../src/osprey/interfaces/web_terminal/static/js/app.js';
import {
  showChatSurface,
  showTerminalSurface,
} from '../../../src/osprey/interfaces/web_terminal/static/js/agent-surface.js';

const backToChat = showChatSurface;
const openTerminal = showTerminalSurface;

/** Let the hand-off chain's microtasks (and any queued task) run. */
function settle() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

// Once, as boot does it: a change has no teardown, so re-arming it per test
// would run every earlier test's copy too.
beforeAll(() => {
  initSurfaceFollowUps();
});

beforeEach(() => {
  calls.length = 0;
  vi.clearAllMocks();
  surfaces.startTerminalSurface.mockImplementation(() => {
    calls.push('startTerminalSurface');
    return Promise.resolve();
  });
  surfaces.stopTerminal.mockImplementation(() => {
    calls.push('stopTerminal');
    return Promise.resolve();
  });
  surfaces.enterFromTerminal.mockImplementation(() => {
    calls.push('enterFromTerminal');
    return Promise.resolve();
  });
  document.documentElement.setAttribute('data-agent-surface', 'terminal');
});

afterEach(async () => {
  // Leave no hand-off in flight: the chain is module state shared by the file.
  await settle();
  vi.restoreAllMocks();
});

describe('back to chat', () => {
  test('drops the terminal, then asks the console to take the key over', async () => {
    backToChat();
    await settle();

    expect(calls).toEqual(['stopTerminal', 'enterFromTerminal']);
    expect(surfaces.startTerminalSurface).not.toHaveBeenCalled();
    // No interrupt: the change never cuts a running turn.
    expect(surfaces.enterFromTerminal).toHaveBeenCalledTimes(1);
    expect(surfaces.enterFromTerminal).toHaveBeenCalledWith();
  });
});

describe('open terminal', () => {
  test('resumes the key over the terminal socket and touches nothing else', async () => {
    document.documentElement.removeAttribute('data-agent-surface');

    openTerminal();
    await settle();

    expect(calls).toEqual(['startTerminalSurface']);
    expect(surfaces.startTerminalSurface).toHaveBeenCalledWith();
    expect(surfaces.stopTerminal).not.toHaveBeenCalled();
    expect(surfaces.enterFromTerminal).not.toHaveBeenCalled();
  });
});

describe('the change itself', () => {
  test('reloads nothing in either direction', async () => {
    const reload = vi.spyOn(window.location, 'reload').mockImplementation(() => {});

    backToChat();
    await settle();
    openTerminal();
    await settle();

    expect(reload).not.toHaveBeenCalled();
  });

  test('asking for the surface already on screen hands nothing over', async () => {
    openTerminal();
    await settle();

    expect(calls).toEqual([]);
  });

  test('a second change waits for the first hand-off instead of racing it', async () => {
    /** @type {() => void} */
    let finishHandoff = () => {};
    surfaces.enterFromTerminal.mockImplementation(() => {
      calls.push('enterFromTerminal');
      return new Promise((resolve) => {
        finishHandoff = () => resolve(undefined);
      });
    });

    backToChat();
    await settle();
    expect(calls).toContain('enterFromTerminal');

    // The operator goes straight back while the console is still acquiring:
    // a socket opened now would be a second acquire on the same key, which
    // the server refuses outright.
    openTerminal();
    await settle();
    expect(surfaces.startTerminalSurface).not.toHaveBeenCalled();

    finishHandoff();
    await settle();
    expect(surfaces.startTerminalSurface).toHaveBeenCalledTimes(1);
    expect(calls[calls.length - 1]).toBe('startTerminalSurface');
  });

  test('the console asks for the key only once the terminal socket has closed', async () => {
    /** @type {() => void} */
    let finishClose = () => {};
    surfaces.stopTerminal.mockImplementation(() => {
      calls.push('stopTerminal');
      return new Promise((resolve) => {
        finishClose = () => resolve(undefined);
      });
    });

    backToChat();
    await settle();

    // The server reads the closed socket as the terminal letting go, so a
    // hand-off asked for ahead of it would find the key still held.
    expect(surfaces.stopTerminal).toHaveBeenCalledTimes(1);
    expect(surfaces.enterFromTerminal).not.toHaveBeenCalled();

    finishClose();
    await settle();
    expect(surfaces.enterFromTerminal).toHaveBeenCalledTimes(1);
    expect(calls.slice(-2)).toEqual(['stopTerminal', 'enterFromTerminal']);
  });

  test('going back to chat waits for the terminal acquire it queued behind', async () => {
    document.documentElement.removeAttribute('data-agent-surface');
    /** @type {() => void} */
    let finishAcquire = () => {};
    surfaces.startTerminalSurface.mockImplementation(() => {
      calls.push('startTerminalSurface');
      return new Promise((resolve) => {
        finishAcquire = () => resolve(undefined);
      });
    });

    openTerminal();
    await settle();
    expect(calls).toContain('startTerminalSurface');

    // Dropping the socket now would tear the terminal channel down before the
    // server had seen it open, and the console's acquire would then arrive to
    // find the key still held by a surface that has already gone.
    backToChat();
    await settle();
    expect(surfaces.stopTerminal).not.toHaveBeenCalled();
    expect(surfaces.enterFromTerminal).not.toHaveBeenCalled();

    finishAcquire();
    await settle();
    expect(surfaces.stopTerminal).toHaveBeenCalledTimes(1);
    expect(surfaces.enterFromTerminal).toHaveBeenCalledTimes(1);
    expect(calls.slice(-2)).toEqual(['stopTerminal', 'enterFromTerminal']);
  });

  test('a failed hand-off is reported and leaves the next change working', async () => {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    surfaces.enterFromTerminal.mockImplementation(() => {
      calls.push('enterFromTerminal');
      return Promise.reject(new Error('server gone'));
    });

    backToChat();
    await settle();
    expect(errSpy).toHaveBeenCalled();

    openTerminal();
    await settle();
    expect(surfaces.startTerminalSurface).toHaveBeenCalledTimes(1);
  });
});
