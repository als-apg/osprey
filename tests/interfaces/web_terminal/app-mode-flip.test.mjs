// @ts-check
/**
 * Unit tests for the session half of an Expert/Simple flip (app.js's
 * `initUiModeFollowUps`):
 *   ./node_modules/.bin/vitest run tests/interfaces/web_terminal/app-mode-flip.test.mjs
 *
 * Both views are windows onto one session key, and the server hands that key
 * to whichever surface asks for it — so the flip is what moves the live agent.
 * Flipping to Expert resumes the key over the terminal socket; flipping to
 * Simple drops that socket and lets the console take the key over. What this
 * file pins is the order (the view's layout lands before its transitional
 * state), that neither direction reloads the page, and that two quick flips
 * produce two hand-offs one after the other rather than two acquires racing
 * for the same key.
 *
 * Seams: terminal.js and chat.js are mocked whole — one wants xterm.js and a
 * socket, the other a server to hand the key over. panel-manager.js and
 * dock-workspace.js keep their real modules and lend only the flip's own
 * follow-up functions to spies, so the dock and panel halves stay under test
 * alongside the new one. The mode message itself goes through the real
 * frame-params.js, which is what decides a re-pick of the current mode is not
 * a flip at all.
 */

import { afterEach, beforeAll, beforeEach, describe, expect, test, vi } from 'vitest';

/** Every flip step that ran, in the order it ran. */
const calls = vi.hoisted(() => /** @type {string[]} */ ([]));

/** Stand-ins for the two surfaces, reachable from the hoisted mock factories. */
const surfaces = vi.hoisted(() => ({
  /** Resolves the acquire by default; a test may hold it open. */
  startExpert: vi.fn(() => {
    calls.push('startExpert');
    return Promise.resolve();
  }),
  /** Resolves on the dropped socket's close by default; a test may hold it. */
  stopTerminal: vi.fn(() => {
    calls.push('stopTerminal');
    return Promise.resolve();
  }),
  /** Resolves the hand-off by default; a test may hold it open. */
  enterFromExpert: vi.fn(() => {
    calls.push('enterFromExpert');
    return Promise.resolve();
  }),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  initTerminal: vi.fn(),
  startTerminal: vi.fn(),
  startExpert: surfaces.startExpert,
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
  enterFromExpert: surfaces.enterFromExpert,
  handoffRefusal: vi.fn(() => null),
  transportNotice: vi.fn(() => ''),
}));

/** The follow-ups the flip already had, spied on their real modules. */
const followUps = vi.hoisted(() => ({
  broadcastMode: vi.fn(() => {
    calls.push('broadcastMode');
  }),
  handleUiModeFlip: vi.fn(() => {
    calls.push('handleUiModeFlip');
  }),
  applyDockMode: vi.fn(() => {
    calls.push('applyDockMode');
  }),
}));

vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js',
  async (importOriginal) => ({
    .../** @type {object} */ (await importOriginal()),
    broadcastMode: followUps.broadcastMode,
    handleUiModeFlip: followUps.handleUiModeFlip,
  })
);

vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js',
  async (importOriginal) => ({
    .../** @type {object} */ (await importOriginal()),
    applyDockMode: followUps.applyDockMode,
  })
);

import { initUiModeFollowUps } from '../../../src/osprey/interfaces/web_terminal/static/js/app.js';

/**
 * Flip the view the way the header's View row does: the same-origin
 * `osprey-mode-change` message frame-params.js listens for.
 * @param {'expert'|'simple'} mode
 */
function flipTo(mode) {
  window.dispatchEvent(
    new MessageEvent('message', {
      data: { type: 'osprey-mode-change', mode },
      origin: window.location.origin,
    })
  );
}

/** Let the hand-off chain's microtasks (and any queued task) run. */
function settle() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

// Once, as boot does it: the listener is bound to the shared `window` and the
// flip has no teardown, so re-arming it per test would run every earlier
// test's copy too.
beforeAll(() => {
  initUiModeFollowUps();
});

beforeEach(() => {
  calls.length = 0;
  vi.clearAllMocks();
  surfaces.startExpert.mockImplementation(() => {
    calls.push('startExpert');
    return Promise.resolve();
  });
  surfaces.stopTerminal.mockImplementation(() => {
    calls.push('stopTerminal');
    return Promise.resolve();
  });
  surfaces.enterFromExpert.mockImplementation(() => {
    calls.push('enterFromExpert');
    return Promise.resolve();
  });
  document.documentElement.setAttribute('data-ui-mode', 'expert');
});

afterEach(async () => {
  // Leave no hand-off in flight: the chain is module state shared by the file.
  await settle();
  vi.restoreAllMocks();
});

describe('flip to Simple', () => {
  test('drops the terminal, then asks the console to take the key over', async () => {
    flipTo('simple');
    await settle();

    expect(calls).toEqual([
      'broadcastMode',
      'applyDockMode',
      'handleUiModeFlip',
      'stopTerminal',
      'enterFromExpert',
    ]);
    expect(surfaces.startExpert).not.toHaveBeenCalled();
  });

  test('asks for the key with no interrupt: the flip never cuts a running turn', async () => {
    flipTo('simple');
    await settle();

    expect(surfaces.enterFromExpert).toHaveBeenCalledTimes(1);
    expect(surfaces.enterFromExpert).toHaveBeenCalledWith();
  });
});

describe('flip to Expert', () => {
  test('resumes the key over the terminal socket and touches nothing else', async () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');

    flipTo('expert');
    await settle();

    expect(calls).toEqual(['broadcastMode', 'applyDockMode', 'handleUiModeFlip', 'startExpert']);
    expect(surfaces.startExpert).toHaveBeenCalledWith();
    expect(surfaces.stopTerminal).not.toHaveBeenCalled();
    expect(surfaces.enterFromExpert).not.toHaveBeenCalled();
  });
});

describe('the flip itself', () => {
  test('reloads nothing in either direction', async () => {
    const reload = vi.spyOn(window.location, 'reload').mockImplementation(() => {});

    flipTo('simple');
    await settle();
    flipTo('expert');
    await settle();

    expect(reload).not.toHaveBeenCalled();
  });

  test('a re-pick of the mode already on screen hands nothing over', async () => {
    flipTo('expert');
    await settle();

    expect(calls).toEqual([]);
  });

  test('a second flip waits for the first hand-off instead of racing it', async () => {
    /** @type {() => void} */
    let finishHandoff = () => {};
    surfaces.enterFromExpert.mockImplementation(() => {
      calls.push('enterFromExpert');
      return new Promise((resolve) => {
        finishHandoff = () => resolve(undefined);
      });
    });

    flipTo('simple');
    await settle();
    expect(calls).toContain('enterFromExpert');

    // The operator flips straight back while the console is still acquiring:
    // a socket opened now would be a second acquire on the same key, which
    // the server refuses outright.
    flipTo('expert');
    await settle();
    expect(surfaces.startExpert).not.toHaveBeenCalled();

    finishHandoff();
    await settle();
    expect(surfaces.startExpert).toHaveBeenCalledTimes(1);
    expect(calls[calls.length - 1]).toBe('startExpert');
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

    flipTo('simple');
    await settle();

    // The server reads the closed socket as the Expert view letting go, so a
    // hand-off asked for ahead of it would find the key still held.
    expect(surfaces.stopTerminal).toHaveBeenCalledTimes(1);
    expect(surfaces.enterFromExpert).not.toHaveBeenCalled();

    finishClose();
    await settle();
    expect(surfaces.enterFromExpert).toHaveBeenCalledTimes(1);
    expect(calls.slice(-2)).toEqual(['stopTerminal', 'enterFromExpert']);
  });

  test('a flip back to Simple waits for the Expert acquire it queued behind', async () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    /** @type {() => void} */
    let finishAcquire = () => {};
    surfaces.startExpert.mockImplementation(() => {
      calls.push('startExpert');
      return new Promise((resolve) => {
        finishAcquire = () => resolve(undefined);
      });
    });

    flipTo('expert');
    await settle();
    expect(calls).toContain('startExpert');

    // Dropping the socket now would tear the Expert channel down before the
    // server had seen it open, and the console's acquire would then arrive to
    // find the key still held by a view that has already gone.
    flipTo('simple');
    await settle();
    expect(surfaces.stopTerminal).not.toHaveBeenCalled();
    expect(surfaces.enterFromExpert).not.toHaveBeenCalled();

    finishAcquire();
    await settle();
    expect(surfaces.stopTerminal).toHaveBeenCalledTimes(1);
    expect(surfaces.enterFromExpert).toHaveBeenCalledTimes(1);
    expect(calls.slice(-2)).toEqual(['stopTerminal', 'enterFromExpert']);
  });

  test('a failed hand-off is reported and leaves the next flip working', async () => {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    surfaces.enterFromExpert.mockImplementation(() => {
      calls.push('enterFromExpert');
      return Promise.reject(new Error('server gone'));
    });

    flipTo('simple');
    await settle();
    expect(errSpy).toHaveBeenCalled();

    flipTo('expert');
    await settle();
    expect(surfaces.startExpert).toHaveBeenCalledTimes(1);
  });
});
