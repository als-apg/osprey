/**
 * Unit tests for the embedded viewer's channel to its host (viewer.js): the
 * hub's `osprey-artifact-select` / `osprey-artifact-action` messages and the
 * `osprey-artifact-ready` announcement.
 *
 * Pure logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/artifacts/viewer.test.mjs
 *
 * viewer.js takes every effect injected, so these tests hand it fakes: a held
 * list, a fetch that answers when the test says so, a `show` and one handler
 * per action. Most tests also give it a fake window whose `parent` is a
 * separate frame, because in this environment `window.parent === window`: the
 * real window can only show the unframed case, and the one test that uses it
 * checks exactly that case.
 */

import { test, expect, describe, vi } from 'vitest';

import {
  createViewer,
  VIEWER_ACTIONS,
} from '../../../src/osprey/interfaces/artifacts/static/js/viewer.js';

const ORIGIN = 'http://hub.test';

/**
 * A promise the test settles by hand, so a fetch answers when the test says.
 * @returns {{promise: Promise<any>, resolve: (value: any) => void}}
 */
function deferred() {
  /** @type {(value: any) => void} */
  let resolve = () => {};
  const promise = new Promise((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

/**
 * A window the viewer can listen on. A delivered message goes straight to its
 * listeners; `parent` is a separate frame unless `framed` is false.
 * @param {{framed?: boolean}} [options]
 */
function fakeWindow({ framed = true } = {}) {
  /** @type {((e: {origin: string, data: any}) => void)[]} */
  const listeners = [];
  const parent = { postMessage: vi.fn() };
  /** @type {any} */
  const win = {
    location: { origin: ORIGIN },
    addEventListener: vi.fn((/** @type {string} */ type, /** @type {any} */ listener) => {
      if (type === 'message') listeners.push(listener);
    }),
    postMessage: vi.fn(),
  };
  win.parent = framed ? parent : win;
  return {
    win,
    parent,
    /**
     * @param {any} data
     * @param {string} [origin]
     */
    deliver(data, origin = ORIGIN) {
      listeners.forEach((listener) => listener({ origin, data }));
    },
  };
}

/**
 * @param {string} id
 * @returns {any}
 */
function makeArtifact(id) {
  return { id, title: `Artifact ${id}` };
}

/** One spy per action the viewer routes. */
function makeActions() {
  return { pin: vi.fn(), unpin: vi.fn(), print: vi.fn(), logbook: vi.fn(), delete: vi.fn() };
}

/**
 * A viewer over a fake window, with every effect a spy.
 * @param {{held?: any[], fetchEntry?: (id: string) => Promise<any>, framed?: boolean}} [options]
 */
function setup({ held = [], fetchEntry = () => Promise.resolve(null), framed = true } = {}) {
  const host = fakeWindow({ framed });
  const deps = {
    getHeld: vi.fn(() => held),
    fetchEntry: vi.fn(fetchEntry),
    show: vi.fn(),
    actions: makeActions(),
  };
  createViewer({ ...deps, win: host.win });
  return { ...deps, ...host };
}

/** @param {any} id */
function select(id) {
  return { type: 'osprey-artifact-select', artifact_id: id };
}

/**
 * @param {any} id
 * @param {any} name
 */
function action(id, name) {
  return { type: 'osprey-artifact-action', artifact_id: id, action: name };
}

/** Let every pending promise callback run. */
function settle() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

describe('the ready announcement', () => {
  test('posts ready to the parent at the page origin, after the listener is bound', () => {
    const { win, parent } = setup();

    expect(parent.postMessage).toHaveBeenCalledTimes(1);
    expect(parent.postMessage).toHaveBeenCalledWith({ type: 'osprey-artifact-ready' }, ORIGIN);
    expect(win.addEventListener.mock.invocationCallOrder[0]).toBeLessThan(
      parent.postMessage.mock.invocationCallOrder[0]
    );
  });

  test('posts nothing when the page is not in a frame', () => {
    const { win } = setup({ framed: false });

    expect(win.postMessage).not.toHaveBeenCalled();
  });

  test('on the real window, which is not framed here, it listens and posts nothing', () => {
    const post = vi.spyOn(window, 'postMessage');
    const held = makeArtifact('a1');
    const show = vi.fn();
    const handlers = makeActions();

    createViewer({
      getHeld: () => [held],
      fetchEntry: () => Promise.resolve(null),
      show,
      actions: handlers,
    });
    window.dispatchEvent(
      new MessageEvent('message', { data: select('a1'), origin: window.location.origin })
    );

    expect(show).toHaveBeenCalledWith(held);
    expect(post).not.toHaveBeenCalled();
    post.mockRestore();
  });
});

describe('osprey-artifact-select', () => {
  test('shows a held artifact at once, with no request', () => {
    const held = makeArtifact('a1');
    const viewer = setup({ held: [held] });

    viewer.deliver(select('a1'));

    expect(viewer.show).toHaveBeenCalledTimes(1);
    expect(viewer.show).toHaveBeenCalledWith(held);
    expect(viewer.fetchEntry).not.toHaveBeenCalled();
  });

  test('fetches an artifact the gallery does not hold and shows the server copy', async () => {
    const fetched = makeArtifact('b2');
    const viewer = setup({ held: [makeArtifact('a1')], fetchEntry: async () => fetched });

    viewer.deliver(select('b2'));
    expect(viewer.fetchEntry).toHaveBeenCalledWith('b2');
    await settle();

    expect(viewer.show).toHaveBeenCalledTimes(1);
    expect(viewer.show).toHaveBeenCalledWith(fetched);
  });

  test('a select of null shows the empty state', () => {
    const viewer = setup({ held: [makeArtifact('a1')] });

    viewer.deliver(select(null));

    expect(viewer.show).toHaveBeenCalledWith(null);
    expect(viewer.fetchEntry).not.toHaveBeenCalled();
  });

  test.each([
    ['answers null', async () => null],
    [
      'fails',
      async () => {
        throw new TypeError('network down');
      },
    ],
  ])('shows the empty state when the fetch %s', async (_label, fetchEntry) => {
    const viewer = setup({ fetchEntry });

    viewer.deliver(select('gone'));
    await settle();

    expect(viewer.show).toHaveBeenCalledTimes(1);
    expect(viewer.show).toHaveBeenCalledWith(null);
  });

  test('drops a fetch that a newer select of a held artifact overtook', async () => {
    const slow = deferred();
    const held = makeArtifact('a1');
    const viewer = setup({ held: [held], fetchEntry: () => slow.promise });

    viewer.deliver(select('slow'));
    viewer.deliver(select('a1'));
    slow.resolve(makeArtifact('slow'));
    await settle();

    expect(viewer.show.mock.calls).toEqual([[held]]);
  });

  test('drops a fetch that a newer fetch overtook, whichever answers first', async () => {
    /** @type {Record<string, ReturnType<typeof deferred>>} */
    const pending = { first: deferred(), second: deferred() };
    const viewer = setup({ fetchEntry: (id) => pending[id].promise });
    const second = makeArtifact('second');

    viewer.deliver(select('first'));
    viewer.deliver(select('second'));
    pending.second.resolve(second);
    await settle();
    pending.first.resolve(makeArtifact('first'));
    await settle();

    expect(viewer.show.mock.calls).toEqual([[second]]);
  });

  test('drops a fetch that a newer select of null overtook', async () => {
    const slow = deferred();
    const viewer = setup({ fetchEntry: () => slow.promise });

    viewer.deliver(select('slow'));
    viewer.deliver(select(null));
    slow.resolve(makeArtifact('slow'));
    await settle();

    expect(viewer.show.mock.calls).toEqual([[null]]);
  });

  test('ignores a select from another origin', async () => {
    const viewer = setup({ held: [makeArtifact('a1')] });

    viewer.deliver(select('a1'), 'https://evil.example');
    viewer.deliver(select(null), 'null');
    await settle();

    expect(viewer.show).not.toHaveBeenCalled();
    expect(viewer.fetchEntry).not.toHaveBeenCalled();
  });

  test.each([
    ['no artifact_id', { type: 'osprey-artifact-select' }],
    ['a numeric artifact_id', select(42)],
    ['an empty artifact_id', select('')],
    ['an object artifact_id', select({ id: 'a1' })],
    ['another message type', { type: 'osprey-session-change', artifact_id: 'a1' }],
    ['a bare string', 'osprey-artifact-select'],
    ['no data', null],
  ])('ignores a message with %s', async (_label, data) => {
    const viewer = setup({ held: [makeArtifact('a1')] });

    viewer.deliver(data);
    await settle();

    expect(viewer.show).not.toHaveBeenCalled();
    expect(viewer.fetchEntry).not.toHaveBeenCalled();
  });
});

describe('osprey-artifact-action', () => {
  test.each(VIEWER_ACTIONS)('routes "%s" to its handler with the held entry', (name) => {
    const held = makeArtifact('a1');
    const viewer = setup({ held: [held] });

    viewer.deliver(action('a1', name));

    expect(viewer.actions[name]).toHaveBeenCalledTimes(1);
    expect(viewer.actions[name]).toHaveBeenCalledWith(held);
    for (const other of VIEWER_ACTIONS.filter((n) => n !== name)) {
      expect(viewer.actions[other]).not.toHaveBeenCalled();
    }
    expect(viewer.fetchEntry).not.toHaveBeenCalled();
  });

  test('resolves an artifact the gallery does not hold from the server first', async () => {
    const fetched = makeArtifact('b2');
    const viewer = setup({ fetchEntry: async () => fetched });

    viewer.deliver(action('b2', 'print'));
    await settle();

    expect(viewer.fetchEntry).toHaveBeenCalledWith('b2');
    expect(viewer.actions.print).toHaveBeenCalledWith(fetched);
  });

  test('does nothing when the server has no such artifact', async () => {
    const viewer = setup({ fetchEntry: async () => null });

    viewer.deliver(action('gone', 'delete'));
    await settle();

    expect(viewer.actions.delete).not.toHaveBeenCalled();
  });

  test('never changes what the viewer shows', () => {
    const viewer = setup({ held: [makeArtifact('a1')] });

    viewer.deliver(action('a1', 'pin'));

    expect(viewer.show).not.toHaveBeenCalled();
  });

  test.each(['toString', 'constructor', '__proto__', 'archive', '', undefined])(
    'ignores the action %j, which is outside the closed set',
    async (name) => {
      const viewer = setup({ held: [makeArtifact('a1')] });

      viewer.deliver(action('a1', name));
      await settle();

      for (const handler of Object.values(viewer.actions)) {
        expect(handler).not.toHaveBeenCalled();
      }
      expect(viewer.fetchEntry).not.toHaveBeenCalled();
    }
  );

  test.each([
    ['no artifact id', action(null, 'pin')],
    ['an empty artifact id', action('', 'pin')],
  ])('ignores an action with %s', async (_label, data) => {
    const viewer = setup({ held: [makeArtifact('a1')] });

    viewer.deliver(data);
    await settle();

    expect(viewer.actions.pin).not.toHaveBeenCalled();
    expect(viewer.fetchEntry).not.toHaveBeenCalled();
  });

  test('ignores an action from another origin', async () => {
    const viewer = setup({ held: [makeArtifact('a1')] });

    viewer.deliver(action('a1', 'delete'), 'https://evil.example');
    await settle();

    expect(viewer.actions.delete).not.toHaveBeenCalled();
  });
});
