// @ts-check
/**
 * The hub side of the pop-out protocol (popout.js):
 *   npx vitest run tests/interfaces/web_terminal/popout.test.mjs
 *
 * A panel pops out into a named popup sized like its tile; the hub keeps the
 * handle, retires the tile and tells the subscribers. The popup's hello and
 * bye are taken only from this hub's own windows, by origin and source; the
 * roll-call goes out once on the BroadcastChannel; a closed window — or one
 * that said bye and never came back — is released and its tile re-docked.
 * The panel-manager verbs arrive as deps, so this file drives the module
 * alone, with stand-ins for the windows the browser would hand back.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const MOD = '../../../src/osprey/interfaces/web_terminal/static/js/popout.js';

const { showUndoNotice } = vi.hoisted(() => ({ showUndoNotice: vi.fn() }));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/undo-notice.js', () => ({
  showUndoNotice,
}));

/** Every channel constructed, with what was posted on it. @type {any[]} */
let channels = [];

class FakeBroadcastChannel {
  /** @param {string} name */
  constructor(name) {
    this.name = name;
    this.posted = /** @type {any[]} */ ([]);
    this.closed = false;
    channels.push(this);
  }
  /** @param {any} message */
  postMessage(message) {
    this.posted.push(message);
  }
  close() {
    this.closed = true;
  }
}

/**
 * A stand-in for the WindowProxy `window.open` returns, shaped like a window
 * this hub opened under the protocol's name.
 * @param {string} panelId
 */
function fakePopup(panelId) {
  return {
    closed: false,
    opener: window,
    name: `osprey-popout-${panelId}`,
    focus: vi.fn(),
    close: vi.fn(),
    postMessage: vi.fn(),
  };
}

/**
 * @param {object} data
 * @param {{ origin?: string, source?: any }} [init]
 */
function message(data, { origin = window.location.origin, source = null } = {}) {
  window.dispatchEvent(new MessageEvent('message', { data, origin, source }));
}

/** @returns {Promise<{ mod: typeof import('../../../src/osprey/interfaces/web_terminal/static/js/popout.js'), deps: any }>} */
async function boot() {
  vi.resetModules();
  const mod = await import(MOD);
  const deps = {
    popoutUrl: vi.fn((/** @type {string} */ id) =>
      id === 'opaque' || id === 'unresolved' ? null : `/panel/${id}/?popout=true&theme=dark`
    ),
    conceal: vi.fn(),
    redock: vi.fn(),
    sync: vi.fn(),
  };
  mod.initPopouts(deps);
  return { mod, deps };
}

/**
 * A tile iframe for `panelId` whose box measures `width` × `height`.
 * @param {string} panelId
 * @param {number} width
 * @param {number} height
 */
function tileIframe(panelId, width, height) {
  const iframe = document.createElement('iframe');
  iframe.className = 'panel-iframe';
  iframe.dataset.panelId = panelId;
  iframe.getBoundingClientRect = () =>
    /** @type {DOMRect} */ ({ width, height, left: 0, top: 0, right: width, bottom: height });
  document.body.appendChild(iframe);
  return iframe;
}

beforeEach(() => {
  channels = [];
  vi.stubGlobal('BroadcastChannel', FakeBroadcastChannel);
  vi.stubGlobal('screen', { availWidth: 1600, availHeight: 900, availLeft: 0, availTop: 0 });
  showUndoNotice.mockClear();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.useRealTimers();
  document.body.innerHTML = '';
  document.documentElement.removeAttribute('data-osprey-storage-scope');
});

describe('opening', () => {
  test('opens the pop-out URL as a popup named for the panel, sized like its tile, centred', async () => {
    const { mod, deps } = await boot();
    tileIframe('ariel', 800, 600);
    const popup = fakePopup('ariel');
    const open = vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (popup));

    expect(mod.popOutPanel('ariel')).toBe(true);

    expect(open).toHaveBeenCalledWith(
      '/panel/ariel/?popout=true&theme=dark',
      'osprey-popout-ariel',
      'popup=yes,width=800,height=600,left=400,top=150'
    );
    expect(mod.isPoppedOut('ariel')).toBe(true);
    expect(deps.conceal).toHaveBeenCalledWith('ariel');
    expect(deps.redock).not.toHaveBeenCalled();
  });

  test('the window is never smaller than 480×360 nor larger than the screen', async () => {
    const { mod } = await boot();
    const open = vi
      .spyOn(window, 'open')
      .mockImplementation(() => /** @type {any} */ (fakePopup('x')));

    tileIframe('small', 100, 50);
    mod.popOutPanel('small');
    expect(open.mock.calls[0][2]).toBe('popup=yes,width=480,height=360,left=560,top=270');

    tileIframe('huge', 4000, 3000);
    mod.popOutPanel('huge');
    expect(open.mock.calls[1][2]).toBe('popup=yes,width=1600,height=900,left=0,top=0');
  });

  test('a panel with no tile to measure opens at the default size', async () => {
    const { mod } = await boot();
    const open = vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (fakePopup('ariel')));

    mod.popOutPanel('ariel');

    expect(open.mock.calls[0][2]).toBe('popup=yes,width=960,height=720,left=320,top=90');
  });

  test('a panel with no pop-out page (unresolved config, opaque origin) does not open', async () => {
    const { mod, deps } = await boot();
    const open = vi.spyOn(window, 'open');

    expect(mod.canPopOut('opaque')).toBe(false);
    expect(mod.canPopOut('unresolved')).toBe(false);
    expect(mod.canPopOut('ariel')).toBe(true);
    expect(mod.popOutPanel('opaque')).toBe(false);

    expect(open).not.toHaveBeenCalled();
    expect(deps.conceal).not.toHaveBeenCalled();
  });

  test('a blocked popup leaves the tile where it is and says so', async () => {
    const { mod, deps } = await boot();
    vi.spyOn(window, 'open').mockReturnValue(null);

    expect(mod.popOutPanel('ariel')).toBe(false);

    expect(showUndoNotice).toHaveBeenCalledWith({ message: 'Pop-ups are blocked for this site.' });
    expect(mod.isPoppedOut('ariel')).toBe(false);
    expect(deps.conceal).not.toHaveBeenCalled();
  });

  test('popping out a panel that is already out brings its window to the front', async () => {
    const { mod } = await boot();
    const popup = fakePopup('ariel');
    const open = vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (popup));
    mod.popOutPanel('ariel');

    expect(mod.popOutPanel('ariel')).toBe(true);

    expect(open).toHaveBeenCalledTimes(1);
    expect(popup.focus).toHaveBeenCalledTimes(1);
  });

  test('subscribers hear every pop-out and every return', async () => {
    const { mod } = await boot();
    const popup = fakePopup('ariel');
    vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (popup));
    const changes = /** @type {[string, boolean][]} */ ([]);
    const unsubscribe = mod.onPopoutChange((id, out) => changes.push([id, out]));

    mod.popOutPanel('ariel');
    popup.closed = true;
    mod.focusPopout('ariel');
    unsubscribe();
    mod.popOutPanel('ariel');

    expect(changes).toEqual([
      ['ariel', true],
      ['ariel', false],
    ]);
  });
});

describe('the roll-call', () => {
  test('goes out once on the osprey-popout channel at init, which is then closed', async () => {
    await boot();

    expect(channels).toHaveLength(1);
    expect(channels[0].name).toBe('osprey-popout');
    expect(channels[0].posted).toEqual([{ type: 'osprey-popout-roll-call' }]);
    expect(channels[0].closed).toBe(true);
  });

  test('is scoped to this render on a multi-user mount', async () => {
    document.documentElement.setAttribute('data-osprey-storage-scope', 'alice');
    await boot();

    expect(channels[0].name).toBe('osprey-popout--alice');
  });
});

describe('hello and bye', () => {
  test('a hello from a window this hub opened is adopted: tile retired, state sent', async () => {
    const { mod, deps } = await boot();
    const popup = fakePopup('ariel');

    message({ type: 'osprey-popout-hello', panelId: 'ariel' }, { source: popup });

    expect(mod.isPoppedOut('ariel')).toBe(true);
    expect(deps.conceal).toHaveBeenCalledWith('ariel');
    expect(deps.sync).toHaveBeenCalledWith(popup);
  });

  test('a hello from the window already held re-sends the state and changes nothing else', async () => {
    const { mod, deps } = await boot();
    const popup = fakePopup('ariel');
    vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (popup));
    mod.popOutPanel('ariel');
    deps.conceal.mockClear();

    message({ type: 'osprey-popout-hello', panelId: 'ariel' }, { source: popup });

    expect(deps.sync).toHaveBeenCalledTimes(1);
    expect(deps.conceal).not.toHaveBeenCalled();
  });

  test('a hello from a foreign origin is ignored', async () => {
    const { mod, deps } = await boot();

    message(
      { type: 'osprey-popout-hello', panelId: 'ariel' },
      { origin: 'https://elsewhere.example', source: fakePopup('ariel') }
    );

    expect(mod.isPoppedOut('ariel')).toBe(false);
    expect(deps.conceal).not.toHaveBeenCalled();
    expect(deps.sync).not.toHaveBeenCalled();
  });

  test('a hello from a window this hub did not open, or named for another panel, is ignored', async () => {
    const { mod, deps } = await boot();

    message(
      { type: 'osprey-popout-hello', panelId: 'ariel' },
      { source: { ...fakePopup('ariel'), opener: null } }
    );
    message({ type: 'osprey-popout-hello', panelId: 'ariel' }, { source: fakePopup('okf') });
    message({ type: 'osprey-popout-hello', panelId: 'ariel' });

    expect(mod.isPoppedOut('ariel')).toBe(false);
    expect(deps.conceal).not.toHaveBeenCalled();
    expect(deps.sync).not.toHaveBeenCalled();
  });

  test('a bye from the held window releases it after the grace, re-docking the tile', async () => {
    vi.useFakeTimers();
    const { mod, deps } = await boot();
    const popup = fakePopup('ariel');
    vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (popup));
    mod.popOutPanel('ariel');

    message({ type: 'osprey-popout-bye', panelId: 'ariel' }, { source: popup });
    await vi.advanceTimersByTimeAsync(1000);
    expect(mod.isPoppedOut('ariel')).toBe(true);
    await vi.advanceTimersByTimeAsync(1000);

    expect(mod.isPoppedOut('ariel')).toBe(false);
    expect(deps.redock).toHaveBeenCalledWith('ariel');
  });

  test('a bye followed by a hello inside the grace is a reload: nothing is re-docked', async () => {
    vi.useFakeTimers();
    const { mod, deps } = await boot();
    const popup = fakePopup('ariel');
    vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (popup));
    mod.popOutPanel('ariel');

    message({ type: 'osprey-popout-bye', panelId: 'ariel' }, { source: popup });
    await vi.advanceTimersByTimeAsync(500);
    message({ type: 'osprey-popout-hello', panelId: 'ariel' }, { source: popup });
    await vi.advanceTimersByTimeAsync(5000);

    expect(mod.isPoppedOut('ariel')).toBe(true);
    expect(deps.redock).not.toHaveBeenCalled();
    expect(deps.sync).toHaveBeenCalledWith(popup);
  });

  test('a bye from any other window, or from a foreign origin, is ignored', async () => {
    vi.useFakeTimers();
    const { mod, deps } = await boot();
    const popup = fakePopup('ariel');
    vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (popup));
    mod.popOutPanel('ariel');

    message({ type: 'osprey-popout-bye', panelId: 'ariel' }, { source: fakePopup('ariel') });
    message({ type: 'osprey-popout-bye', panelId: 'ariel' }, { source: null });
    message(
      { type: 'osprey-popout-bye', panelId: 'ariel' },
      { origin: 'https://elsewhere.example', source: popup }
    );
    await vi.advanceTimersByTimeAsync(5000);

    expect(mod.isPoppedOut('ariel')).toBe(true);
    expect(deps.redock).not.toHaveBeenCalled();
  });
});

describe('closing', () => {
  test('a closed window is noticed by the 500 ms poll, which stops with the last window', async () => {
    vi.useFakeTimers();
    const { mod, deps } = await boot();
    const popup = fakePopup('ariel');
    vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (popup));
    mod.popOutPanel('ariel');
    expect(vi.getTimerCount()).toBe(1);

    popup.closed = true;
    await vi.advanceTimersByTimeAsync(499);
    expect(mod.isPoppedOut('ariel')).toBe(true);
    await vi.advanceTimersByTimeAsync(1);

    expect(mod.isPoppedOut('ariel')).toBe(false);
    expect(deps.redock).toHaveBeenCalledWith('ariel');
    expect(vi.getTimerCount()).toBe(0);
  });

  test('bringBackPanel closes the window and leaves the return to the poll', async () => {
    vi.useFakeTimers();
    const { mod, deps } = await boot();
    const popup = fakePopup('ariel');
    vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (popup));
    mod.popOutPanel('ariel');

    mod.bringBackPanel('ariel');
    expect(popup.close).toHaveBeenCalledTimes(1);
    expect(mod.isPoppedOut('ariel')).toBe(true);
    popup.closed = true;
    await vi.advanceTimersByTimeAsync(500);

    expect(deps.redock).toHaveBeenCalledWith('ariel');
    expect(() => mod.bringBackPanel('ariel')).not.toThrow();
  });

  test('focusPopout focuses an open window and releases a closed one', async () => {
    const { mod, deps } = await boot();
    const popup = fakePopup('ariel');
    vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (popup));
    mod.popOutPanel('ariel');

    expect(mod.focusPopout('ariel')).toBe(true);
    expect(popup.focus).toHaveBeenCalledTimes(1);
    expect(mod.focusPopout('okf')).toBe(false);

    popup.closed = true;
    expect(mod.focusPopout('ariel')).toBe(false);
    expect(mod.isPoppedOut('ariel')).toBe(false);
    expect(deps.redock).toHaveBeenCalledWith('ariel');
  });
});

describe('broadcastToPopouts', () => {
  test('sends to every open window and skips a closed one', async () => {
    const { mod } = await boot();
    const ariel = fakePopup('ariel');
    const okf = fakePopup('okf');
    vi.spyOn(window, 'open')
      .mockReturnValueOnce(/** @type {any} */ (ariel))
      .mockReturnValueOnce(/** @type {any} */ (okf));
    mod.popOutPanel('ariel');
    mod.popOutPanel('okf');
    okf.closed = true;
    const send = vi.fn();

    mod.broadcastToPopouts(send);

    expect(send.mock.calls).toEqual([[ariel]]);
  });
});
