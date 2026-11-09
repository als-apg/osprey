// @ts-check
/**
 * Unit tests for a direct visit's handling in the hub (direct-visit.js):
 *   npx vitest run tests/interfaces/web_terminal/direct-visit.test.mjs
 *
 * `?open`/`?at` leave the address at once; the panel opens only after the session
 * layout settled and only once it answers, and the page it opens on is the one
 * a panel page or the bundle route put in `at`.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const mocks = vi.hoisted(() => {
  const state = {
    known: new Set(['ariel']),
    addressed: true,
    healthy: true,
    /** @type {string[]} */
    calls: [],
    /** @type {() => void} */
    settle: () => {},
    /** @type {Promise<void>} */
    settled: Promise.resolve(),
    notices: /** @type {string[]} */ ([]),
  };
  return state;
});

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js', () => ({
  isPanelKnown: (/** @type {string} */ id) => mocks.known.has(id),
  isPanelHealthy: () => mocks.healthy,
  getPanelStandaloneUrl: () => (mocks.addressed ? '/panel/ariel/' : null),
  navigatePanel: (/** @type {string} */ id, /** @type {string} */ url) =>
    mocks.calls.push(`navigate ${id} ${url}`),
  showPanel: (/** @type {string} */ id) => mocks.calls.push(`show ${id}`),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/session-layout.js', () => ({
  get settled() {
    return mocks.settled;
  },
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/undo-notice.js', () => ({
  showUndoNotice: (/** @type {{message: string}} */ { message }) => mocks.notices.push(message),
}));

const { ANSWER_TIMEOUT_MS, openDirectVisit, takeDirectVisit } =
  await import('../../../src/osprey/interfaces/web_terminal/static/js/direct-visit.js');

async function flush() {
  await vi.advanceTimersByTimeAsync(0);
}

beforeEach(() => {
  vi.useFakeTimers();
  mocks.known = new Set(['ariel']);
  mocks.addressed = true;
  mocks.healthy = true;
  mocks.calls = [];
  mocks.notices = [];
  mocks.settled = new Promise((resolve) => {
    mocks.settle = resolve;
  });
});

afterEach(() => {
  vi.useRealTimers();
  window.history.replaceState({}, '', '/');
});

describe('takeDirectVisit', () => {
  test('reads the visit, decoded, and takes it out of the address', () => {
    const at = '/u/alice/panel/ariel/#entry?id=X';
    window.history.replaceState(
      {},
      '',
      `/u/alice/?open=ariel&at=${encodeURIComponent(at)}&keep=1#frag`
    );

    expect(takeDirectVisit()).toEqual({ id: 'ariel', at });
    expect(window.location.pathname).toBe('/u/alice/');
    expect(window.location.search).toBe('?keep=1');
    expect(window.location.hash).toBe('#frag');
  });

  test('takes the address the bundle route answers with', () => {
    window.history.replaceState(
      {},
      '',
      '/u/alice/?open=mine&at=%2Fu%2Falice%2Fpanel-static%2Fmine%2F%3Fx%3D1'
    );
    expect(takeDirectVisit()).toEqual({ id: 'mine', at: '/u/alice/panel-static/mine/?x=1' });
  });

  test('an ordinary boot is not a visit and leaves the address alone', () => {
    window.history.replaceState({}, '', '/?keep=1');
    expect(takeDirectVisit()).toBeNull();
    expect(window.location.search).toBe('?keep=1');
  });

  test('an at with no panel is stripped and opens nothing', () => {
    window.history.replaceState({}, '', '/?at=%2Fpanel%2Fariel%2F');
    expect(takeDirectVisit()).toBeNull();
    expect(window.location.search).toBe('');
  });
});

describe('openDirectVisit', () => {
  const visit = { id: 'ariel', at: '/panel/ariel/#entry?id=X' };

  test('no visit does nothing', async () => {
    await openDirectVisit(null, Promise.resolve());
    expect(mocks.calls).toEqual([]);
  });

  test('waits for the panels and for settled, then navigates and shows', async () => {
    /** @type {() => void} */
    let panelsReady = () => {};
    const ready = new Promise((resolve) => {
      panelsReady = () => resolve(undefined);
    });
    const done = openDirectVisit(visit, ready);
    await flush();
    expect(mocks.calls).toEqual([]);

    panelsReady();
    await flush();
    expect(mocks.calls).toEqual([]);

    mocks.settle();
    await done;
    expect(mocks.calls).toEqual(['navigate ariel /panel/ariel/#entry?id=X', 'show ariel']);
    expect(mocks.notices).toEqual([]);
  });

  test('a panel healthy only after settle still opens, with its page stored first', async () => {
    mocks.healthy = false;
    mocks.settle();
    const done = openDirectVisit(visit, Promise.resolve());
    await flush();
    expect(mocks.calls).toEqual(['navigate ariel /panel/ariel/#entry?id=X']);

    mocks.healthy = true;
    await vi.advanceTimersByTimeAsync(1000);
    await done;
    expect(mocks.calls.at(-1)).toBe('show ariel');
  });

  test('at waits for the address the mount check needs', async () => {
    mocks.addressed = false;
    mocks.settle();
    const done = openDirectVisit(visit, Promise.resolve());
    await flush();
    expect(mocks.calls).toEqual([]);

    mocks.addressed = true;
    await vi.advanceTimersByTimeAsync(1000);
    await done;
    expect(mocks.calls[0]).toBe('navigate ariel /panel/ariel/#entry?id=X');
  });

  test('a visit with no at opens the panel where it starts', async () => {
    mocks.settle();
    await openDirectVisit({ id: 'ariel', at: null }, Promise.resolve());
    expect(mocks.calls).toEqual(['show ariel']);
  });

  test('a panel that never answers gets the notice and is not shown', async () => {
    mocks.healthy = false;
    mocks.settle();
    const done = openDirectVisit(visit, Promise.resolve());
    await vi.advanceTimersByTimeAsync(ANSWER_TIMEOUT_MS + 200);
    await done;
    expect(mocks.notices).toEqual(['ariel is not answering yet.']);
    expect(mocks.calls).not.toContain('show ariel');
  });

  test('an unknown id gets the notice', async () => {
    mocks.settle();
    await openDirectVisit({ id: 'nope', at: '/panel/nope/' }, Promise.resolve());
    expect(mocks.notices).toEqual(['nope is not a panel here.']);
    expect(mocks.calls).toEqual([]);
  });
});
