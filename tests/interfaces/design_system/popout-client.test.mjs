// @ts-check
/**
 * The popup side of the pop-out protocol (popout-client.js):
 *   npx vitest run tests/interfaces/design_system/popout-client.test.mjs
 *
 * A page the hub opened under the protocol's window name reports to its
 * opener — hello on load, bye on pagehide, hello again on the hub's
 * roll-call — addressed to its own origin and only while the opener is
 * alive. A page with no such opener, or without the name, is a plain
 * standalone page and sends nothing. The roll-call is taken from the
 * BroadcastChannel only; a window message that looks like one, whoever posts
 * it, drives nothing.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const MOD = '../../../src/osprey/interfaces/design_system/static/js/popout-client.js';

/** Every channel constructed, with its listeners. @type {any[]} */
let channels = [];

class FakeBroadcastChannel {
  /** @param {string} name */
  constructor(name) {
    this.name = name;
    this.listeners = /** @type {((event: any) => void)[]} */ ([]);
    channels.push(this);
  }
  /** @param {string} type @param {(event: any) => void} listener */
  addEventListener(type, listener) {
    if (type === 'message') this.listeners.push(listener);
  }
  /** Deliver a message as another channel's postMessage would. @param {any} data */
  receive(data) {
    for (const listener of this.listeners) listener({ data });
  }
}

/** A stand-in for the opener's WindowProxy. */
function fakeHub() {
  return { closed: false, postMessage: vi.fn() };
}

/** @returns {Promise<typeof import('../../../src/osprey/interfaces/design_system/static/js/popout-client.js')>} */
async function load() {
  vi.resetModules();
  return import(MOD);
}

beforeEach(() => {
  channels = [];
  vi.stubGlobal('BroadcastChannel', FakeBroadcastChannel);
  window.name = 'osprey-popout-ariel';
});

afterEach(() => {
  vi.unstubAllGlobals();
  window.name = '';
  document.documentElement.removeAttribute('data-osprey-storage-scope');
});

describe('the panel id', () => {
  test('is read off the window name the hub opened the page under', async () => {
    const mod = await load();

    expect(mod.popoutPanelId()).toBe('ariel');
    window.name = 'osprey-popout-channel-finder';
    expect(mod.popoutPanelId()).toBe('channel-finder');
    for (const name of ['', 'osprey-popout-', 'something-else', 'popout-ariel']) {
      window.name = name;
      expect(mod.popoutPanelId()).toBeNull();
    }
  });
});

describe('reporting to the hub', () => {
  test('says hello on load, to the opener, addressed to its own origin', async () => {
    const hub = fakeHub();
    vi.stubGlobal('opener', hub);
    const mod = await load();

    expect(mod.initPopoutClient()).toBe(true);

    expect(hub.postMessage).toHaveBeenCalledExactlyOnceWith(
      { type: 'osprey-popout-hello', panelId: 'ariel' },
      window.location.origin
    );
  });

  test('says bye on pagehide', async () => {
    const hub = fakeHub();
    vi.stubGlobal('opener', hub);
    const mod = await load();
    mod.initPopoutClient();
    hub.postMessage.mockClear();

    window.dispatchEvent(new Event('pagehide'));

    // The window is shared across this file's module loads, so every earlier
    // load's pagehide listener fires too: each says bye, nothing else.
    expect(hub.postMessage).toHaveBeenCalled();
    for (const call of hub.postMessage.mock.calls) {
      expect(call).toEqual([
        { type: 'osprey-popout-bye', panelId: 'ariel' },
        window.location.origin,
      ]);
    }
  });

  test("answers the hub's roll-call on the osprey-popout channel with another hello", async () => {
    const hub = fakeHub();
    vi.stubGlobal('opener', hub);
    const mod = await load();
    mod.initPopoutClient();
    hub.postMessage.mockClear();
    expect(channels.map((c) => c.name)).toEqual(['osprey-popout']);

    channels[0].receive({ type: 'osprey-popout-roll-call' });
    channels[0].receive({ type: 'something-else' });
    channels[0].receive(null);

    expect(hub.postMessage).toHaveBeenCalledExactlyOnceWith(
      { type: 'osprey-popout-hello', panelId: 'ariel' },
      window.location.origin
    );
  });

  test('listens on the channel of its own render on a multi-user mount', async () => {
    document.documentElement.setAttribute('data-osprey-storage-scope', 'alice');
    vi.stubGlobal('opener', fakeHub());
    const mod = await load();

    mod.initPopoutClient();

    expect(channels.map((c) => c.name)).toEqual(['osprey-popout--alice']);
  });

  test('a roll-call posted as a window message, by the opener or anyone else, drives nothing', async () => {
    const hub = fakeHub();
    vi.stubGlobal('opener', hub);
    const mod = await load();
    mod.initPopoutClient();
    hub.postMessage.mockClear();

    for (const source of [hub, { postMessage() {} }, null]) {
      window.dispatchEvent(
        new MessageEvent('message', {
          data: { type: 'osprey-popout-roll-call' },
          origin: window.location.origin,
          source: /** @type {any} */ (source),
        })
      );
    }

    expect(hub.postMessage).not.toHaveBeenCalled();
  });

  test('an opener that has gone away gets no bye, and the page keeps running', async () => {
    const hub = fakeHub();
    vi.stubGlobal('opener', hub);
    const mod = await load();
    mod.initPopoutClient();
    hub.postMessage.mockClear();

    hub.closed = true;
    expect(() => window.dispatchEvent(new Event('pagehide'))).not.toThrow();
    expect(() => channels[0].receive({ type: 'osprey-popout-roll-call' })).not.toThrow();

    expect(hub.postMessage).not.toHaveBeenCalled();
  });
});

describe('a plain standalone page', () => {
  test.each([
    ['no opener', null, 'osprey-popout-ariel'],
    ['an opener that is closed', { closed: true, postMessage: vi.fn() }, 'osprey-popout-ariel'],
    ['a window not named by the hub', fakeHub(), 'my-own-window'],
  ])('%s: reports to nobody and opens no channel', async (_name, opener, windowName) => {
    vi.stubGlobal('opener', opener);
    window.name = windowName;
    const mod = await load();

    expect(mod.initPopoutClient()).toBe(false);

    if (opener) expect(opener.postMessage).not.toHaveBeenCalled();
    expect(channels).toEqual([]);
  });
});
