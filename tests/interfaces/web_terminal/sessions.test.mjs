// @ts-check
/**
 * Unit tests for the two session moves (sessions.js):
 *   npx vitest run tests/interfaces/web_terminal/sessions.test.mjs
 *
 * `resumeSession(key)` and `startNewSession()` are the low-level moves the
 * session model drives; operators pick sessions in the header switcher
 * (session-switcher.test.mjs) and the model guards the pick
 * (session-model.test.mjs). What is pinned here is what each move does on
 * each surface: in chat, a resume only moves the pointer — the console replays
 * the key and no terminal is started beside it; on the terminal, a resume
 * takes the warm socket's fast path, and falls back to a full stop/start when
 * no socket is open.
 *
 * terminal.js, agent-surface.js and session-pointer.js are stubbed at the
 * module boundary, so each move is asserted as the calls it makes.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  terminalSurface: false,
  stopTerminal: vi.fn(),
  startTerminal: vi.fn(),
  restartTerminal: vi.fn(async () => {}),
  notifySessionChange: vi.fn(),
  switchSession: vi.fn(() => false),
  setSessionLabel: vi.fn(),
  setPointer: vi.fn(),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  stopTerminal: mocks.stopTerminal,
  startTerminal: mocks.startTerminal,
  restartTerminal: mocks.restartTerminal,
  notifySessionChange: mocks.notifySessionChange,
  switchSession: mocks.switchSession,
  setSessionLabel: mocks.setSessionLabel,
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/agent-surface.js', () => ({
  isTerminalSurface: () => mocks.terminalSurface,
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/session-pointer.js', () => ({
  setPointer: mocks.setPointer,
}));

const { resumeSession, startNewSession } =
  await import('../../../src/osprey/interfaces/web_terminal/static/js/sessions.js');

const KEY = 'aaaaaaaa-1111-2222-3333-444444444444';

beforeEach(() => {
  mocks.terminalSurface = false;
  for (const fn of [
    mocks.stopTerminal,
    mocks.startTerminal,
    mocks.restartTerminal,
    mocks.notifySessionChange,
    mocks.switchSession,
    mocks.setSessionLabel,
    mocks.setPointer,
  ])
    fn.mockClear();
  mocks.switchSession.mockImplementation(() => false);
});

afterEach(() => {
  vi.useRealTimers();
});

describe('resumeSession in chat', () => {
  test('moves the pointer and starts no terminal', async () => {
    await resumeSession(KEY);
    expect(mocks.setPointer).toHaveBeenCalledWith(KEY);
    expect(mocks.switchSession).not.toHaveBeenCalled();
    expect(mocks.stopTerminal).not.toHaveBeenCalled();
    expect(mocks.startTerminal).not.toHaveBeenCalled();
  });
});

describe('resumeSession on the terminal', () => {
  beforeEach(() => {
    mocks.terminalSurface = true;
  });

  test('a warm socket takes the fast path: switch_session and nothing else', async () => {
    mocks.switchSession.mockImplementation(() => true);
    await resumeSession(KEY);
    expect(mocks.switchSession).toHaveBeenCalledWith(KEY);
    expect(mocks.setPointer).not.toHaveBeenCalled();
    expect(mocks.stopTerminal).not.toHaveBeenCalled();
    expect(mocks.startTerminal).not.toHaveBeenCalled();
  });

  test('no socket open: a full stop, label, start-in-resume, then the change notice', async () => {
    vi.useFakeTimers();
    const done = resumeSession(KEY);
    expect(mocks.stopTerminal).toHaveBeenCalledTimes(1);
    expect(mocks.setSessionLabel).toHaveBeenCalledWith(KEY);
    expect(mocks.startTerminal).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(100);
    await done;
    expect(mocks.startTerminal).toHaveBeenCalledWith(KEY, 'resume');
    expect(mocks.notifySessionChange).toHaveBeenCalledWith(KEY);
    expect(mocks.setPointer).not.toHaveBeenCalled();
  });
});

describe('startNewSession', () => {
  test('restarts, clears the label, then starts a fresh terminal', async () => {
    await startNewSession();
    expect(mocks.restartTerminal).toHaveBeenCalledTimes(1);
    expect(mocks.setSessionLabel).toHaveBeenCalledWith(null);
    expect(mocks.startTerminal).toHaveBeenCalledWith();
    const restartOrder = mocks.restartTerminal.mock.invocationCallOrder[0];
    expect(mocks.startTerminal.mock.invocationCallOrder[0]).toBeGreaterThan(restartOrder);
  });
});
