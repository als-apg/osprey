// @ts-check
// @vitest-environment jsdom
/**
 * A pane in a window of its own (pane-window.js), over the real attached-window
 * and the real layout holder and placement rules; jsdom opens no windows, so
 * `window.open` returns a stand-in whose document is a second jsdom document,
 * and what the panel stack hands the module is faked.
 *
 *   npx vitest run tests/interfaces/web_terminal/pane-window.test.mjs
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';
const SHELL_MARKER = 'data-osprey-attached-shell';

const TWO_TILES = {
  split: 'row',
  items: [
    { tile: 't1', tabs: ['ariel'], showing: 'ariel', size: 0.5 },
    { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.5 },
  ],
};

/** @param {boolean} [loaded] */
function shellDocument(loaded = true) {
  const doc = document.implementation.createHTMLDocument('');
  if (loaded) doc.documentElement.setAttribute(SHELL_MARKER, '');
  return doc;
}

/** A popup as the hub sees it: `close()` fires `pagehide` the way a closing window does. */
function fakePopup(loaded = true) {
  const target = new EventTarget();
  const win = {
    document: shellDocument(loaded),
    closed: false,
    focus: vi.fn(),
    close: vi.fn(() => {
      if (win.closed) return;
      win.closed = true;
      target.dispatchEvent(new Event('pagehide'));
    }),
    addEventListener: target.addEventListener.bind(target),
    removeEventListener: target.removeEventListener.bind(target),
    dispatchEvent: target.dispatchEvent.bind(target),
  };
  return win;
}

/**
 * The window's bar as the module sees it; dock-tab.test.mjs pins the real one.
 * @param {string} id
 * @param {{mode: string, onReattach: () => void}} options
 */
function fakeTab(id, options) {
  const element = document.createElement('header');
  element.className = 'fake-tab';
  const button = document.createElement('button');
  button.className = 'fake-reattach';
  button.addEventListener('click', options.onReattach);
  element.append(button);
  return {
    id,
    mode: options.mode,
    element,
    init: vi.fn(),
    attached: vi.fn(() => tabDocuments.push(element.ownerDocument)),
    dispose: vi.fn(),
  };
}

/** @type {Document[]} */ let tabDocuments;
/** @type {any} */ let mod;
/** @type {any} */ let holder;
/** @type {any} */ let attached;
/** @type {any} */ let deps;
/** @type {ReturnType<typeof fakePopup>[]} */ let popups;
/** @type {boolean} */ let nextLoaded;
/** @type {HTMLIFrameElement} */ let iframe;
/** @type {HTMLElement} */ let overlay;
/** @type {HTMLElement} */ let host;
/** @type {HTMLElement} */ let agent;
/** @type {Map<string, string|null>} */ let pages;
/** @type {(() => void)[]} */ let themeListeners;

beforeEach(async () => {
  vi.resetModules();
  document.head.replaceChildren();
  document.body.replaceChildren();
  popups = [];
  nextLoaded = true;
  vi.stubGlobal(
    'open',
    vi.fn(() => {
      const win = fakePopup(nextLoaded);
      popups.push(win);
      return win;
    })
  );
  overlay = document.createElement('div');
  iframe = document.createElement('iframe');
  iframe.dataset.panelId = 'ariel';
  iframe.setAttribute('style', 'position: absolute; left: 7px;');
  overlay.append(iframe);
  host = document.createElement('div');
  agent = document.createElement('section');
  host.append(agent);
  document.body.append(overlay, host);

  const { initLayoutHolder } = await import(`${JS}/dock-apply.js`);
  const { detachPlacement } = await import(`${JS}/dock-open.js`);
  holder = initLayoutHolder({ dockPx: { width: 1200, height: 800 }, placement: detachPlacement });
  holder.restore(TWO_TILES);
  mod = await import(`${JS}/pane-window.js`);
  attached = await import(`${JS}/attached-window.js`);

  pages = new Map([['ariel', '/panel/ariel/#entry?id=X']]);
  themeListeners = [];
  tabDocuments = [];
  deps = {
    createTileTab: vi.fn(fakeTab),
    managedIframe: vi.fn((/** @type {string} */ id) =>
      id === 'ariel' ? { iframe, title: 'LOGBOOK' } : null
    ),
    concealPanel: vi.fn(),
    panelReturned: vi.fn(),
    getPanelPage: vi.fn((/** @type {string} */ id) => pages.get(id) ?? null),
    buildEmbedSrc: vi.fn((/** @type {string} */ page) => `http://hub${page}?embedded=true`),
    sendThemeToIframe: vi.fn(),
    subscribeTheme: vi.fn((/** @type {() => void} */ fn) => themeListeners.push(fn)),
    clearAccent: vi.fn(),
    showNotice: vi.fn(),
    agentPaneContent: () => agent,
    fitAgent: vi.fn(),
    withEchoSuppressed: vi.fn((/** @type {() => any} */ fn) => fn()),
  };
  mod.initPaneWindow(deps);
});

afterEach(() => {
  mod.closeAllDetached();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('the seam', () => {
  test('imports only its four modules, statically', () => {
    const here = path.dirname(fileURLToPath(import.meta.url));
    const source = readFileSync(path.join(here, `${JS}/pane-window.js`), 'utf8');
    const found = [...source.matchAll(/from '([^']+)'|import\(/g)].map((m) => m[1] ?? 'dynamic');
    expect(found.sort()).toEqual(
      [
        './attached-window.js',
        './dock-apply.js',
        './panel-catalog.js',
        '/design-system/js/storage-scope.js',
      ].sort()
    );
  });

  test('deps merge: a later part keeps what an earlier part handed in', () => {
    mod.initPaneWindow({ showNotice: vi.fn() });
    expect(mod.detachBlocker('ariel')).toBeNull();
    expect(mod.detachBlocker('terminal')).toBeNull();
  });

  test('a pane whose deps have not arrived is blocked, by name', async () => {
    vi.resetModules();
    const { initLayoutHolder } = await import(`${JS}/dock-apply.js`);
    initLayoutHolder({ dockPx: { width: 1200, height: 800 } }).restore(TWO_TILES);
    const fresh = await import(`${JS}/pane-window.js`);
    expect(fresh.detachBlocker('ariel')).toMatch(/not ready/);
    fresh.initPaneWindow({ withEchoSuppressed: deps.withEchoSuppressed });
    expect(fresh.detachBlocker('ariel')).toMatch(/has not loaded/);
    expect(fresh.detachBlocker('terminal')).toMatch(/Agent pane is not ready/);
  });

  test('an opaque panel cannot leave; one whose kind is still pending can', () => {
    iframe.dataset.panelFrame = 'opaque';
    expect(mod.detachBlocker('ariel')).toBe(
      'This panel runs at an opaque origin and cannot open in a new window.'
    );
    expect(mod.detach('ariel')).toBe(false);
    expect(popups).toHaveLength(0);
    iframe.dataset.panelFrame = 'pending';
    expect(mod.detachBlocker('ariel')).toBeNull();
    delete iframe.dataset.panelFrame;
  });

  test('the window name is scoped per render', async () => {
    document.documentElement.setAttribute('data-osprey-storage-scope', 'alice');
    mod.detach('ariel');
    document.documentElement.removeAttribute('data-osprey-storage-scope');
    const name = /** @type {any} */ (window.open).mock.calls[0][1];
    expect(name).toMatch(/^osprey-pane-ariel.+alice/);
  });
});

describe('detach', () => {
  test('a service pane: the tile leaves, its frame moves on its page, under echo suppression', () => {
    expect(mod.detach('ariel')).toBe(true);
    expect(deps.withEchoSuppressed).toHaveBeenCalled();
    expect(holder.isDetached('ariel')).toBe(true);
    expect(holder.detachedEntry('ariel').page).toBe('/panel/ariel/#entry?id=X');
    expect(holder.layout.tabs ?? holder.layout.items).toBeTruthy();
    expect(iframe.src).toBe('http://hub/panel/ariel/#entry?id=X?embedded=true');
    const mount = popups[0].document.querySelector('.attached-window-root');
    expect(iframe.parentElement).toBe(mount);
    expect(mount?.getAttribute('data-osprey-detached')).toBe('true');
    expect(overlay.querySelector('iframe')).toBeNull();
  });

  test('the accent is cleared through the seam that checks it pointed at the pane', () => {
    mod.detach('ariel');
    expect(deps.clearAccent).toHaveBeenCalledExactlyOnceWith('ariel');
  });

  test('the Agent pane moves without a src and refits', () => {
    mod.detach('terminal');
    expect(holder.isDetached('terminal')).toBe(true);
    expect(agent.ownerDocument).toBe(popups[0].document);
    expect(deps.fitAgent).toHaveBeenCalled();
    expect(deps.clearAccent).not.toHaveBeenCalled();
  });

  test('nothing is stored or moved until the shell has loaded', () => {
    vi.useFakeTimers();
    nextLoaded = false;
    mod.detach('ariel');
    expect(holder.isDetached('ariel')).toBe(false);
    expect(iframe.parentElement).toBe(overlay);
    popups[0].document.documentElement.setAttribute(SHELL_MARKER, '');
    vi.advanceTimersByTime(attached.READY_POLL_MS);
    expect(holder.isDetached('ariel')).toBe(true);
  });

  test('a tile that vanished during the marker wait aborts: window closed, nothing stored', () => {
    vi.useFakeTimers();
    nextLoaded = false;
    const onChange = vi.fn();
    mod.onDetachedChange(onChange);
    mod.detach('ariel');
    holder.restore({ tile: 't2', tabs: ['terminal'], showing: 'terminal' });
    popups[0].document.documentElement.setAttribute(SHELL_MARKER, '');
    vi.advanceTimersByTime(attached.READY_POLL_MS);
    expect(popups[0].closed).toBe(true);
    expect(holder.isDetached('ariel')).toBe(false);
    expect(iframe.parentElement).toBe(overlay);
    expect(deps.clearAccent).not.toHaveBeenCalled();
    expect(onChange).not.toHaveBeenCalled();
  });

  test('a frame that turns opaque during the marker wait does not move', () => {
    vi.useFakeTimers();
    nextLoaded = false;
    mod.detach('ariel');
    iframe.dataset.panelFrame = 'opaque';
    popups[0].document.documentElement.setAttribute(SHELL_MARKER, '');
    vi.advanceTimersByTime(attached.READY_POLL_MS);
    expect(popups[0].closed).toBe(true);
    expect(holder.isDetached('ariel')).toBe(false);
    expect(iframe.parentElement).toBe(overlay);
    expect(deps.showNotice).toHaveBeenCalledTimes(1);
    delete iframe.dataset.panelFrame;
  });

  test('a pending frame waits in the hub with its window open, then moves in', () => {
    vi.useFakeTimers();
    iframe.dataset.panelFrame = 'pending';
    expect(mod.detach('ariel')).toBe(true);
    expect(popups).toHaveLength(1);
    vi.advanceTimersByTime(attached.READY_POLL_MS * 4);
    expect(iframe.parentElement).toBe(overlay);
    expect(holder.isDetached('ariel')).toBe(false);

    iframe.dataset.panelFrame = 'first-party';
    vi.advanceTimersByTime(attached.READY_POLL_MS);
    expect(iframe.ownerDocument).toBe(popups[0].document);
    expect(holder.isDetached('ariel')).toBe(true);
    expect(deps.showNotice).not.toHaveBeenCalled();
    delete iframe.dataset.panelFrame;
  });

  test('a pending frame that resolves opaque closes its window and stays docked', () => {
    vi.useFakeTimers();
    iframe.dataset.panelFrame = 'pending';
    mod.detach('ariel');
    vi.advanceTimersByTime(attached.READY_POLL_MS);

    iframe.dataset.panelFrame = 'opaque';
    vi.advanceTimersByTime(attached.READY_POLL_MS);
    expect(popups[0].closed).toBe(true);
    expect(holder.isDetached('ariel')).toBe(false);
    expect(holder.holds('ariel')).toBe(true);
    expect(iframe.parentElement).toBe(overlay);
    expect(deps.showNotice).toHaveBeenCalledExactlyOnceWith(
      'This panel runs at an opaque origin and cannot open in a new window.'
    );
    delete iframe.dataset.panelFrame;
  });

  test('a frame that never resolves gives up, with a notice and the pane in place', () => {
    vi.useFakeTimers();
    iframe.dataset.panelFrame = 'pending';
    mod.detach('ariel');
    vi.advanceTimersByTime(attached.READY_TIMEOUT_MS + attached.READY_POLL_MS);
    expect(popups[0].closed).toBe(true);
    expect(holder.holds('ariel')).toBe(true);
    expect(deps.showNotice).toHaveBeenCalledExactlyOnceWith('This panel has not loaded yet.');
    delete iframe.dataset.panelFrame;
  });

  test('a refused window says so and changes nothing', () => {
    /** @type {any} */ (window.open).mockReturnValue(null);
    expect(mod.detach('ariel')).toBe(false);
    expect(deps.showNotice).toHaveBeenCalledExactlyOnceWith('Pop-ups are blocked for this site.');
    expect(holder.isDetached('ariel')).toBe(false);
  });

  test('a pane already out comes to the front', () => {
    mod.detach('ariel');
    expect(mod.detach('ariel')).toBe(true);
    expect(popups).toHaveLength(1);
    expect(popups[0].focus).toHaveBeenCalled();
    expect(mod.focusWindow('terminal')).toBe(false);
  });

  test('subscribers hear the pane leave once and come back once', () => {
    const onChange = vi.fn();
    mod.onDetachedChange(onChange);
    mod.detach('ariel');
    expect(onChange.mock.calls).toEqual([['ariel', true]]);
    mod.reattach('ariel');
    expect(onChange.mock.calls).toEqual([
      ['ariel', true],
      ['ariel', false],
    ]);
  });
});

describe('a pane with no tile', () => {
  const AGENT_ONLY = { tile: 't2', tabs: ['terminal'], showing: 'terminal' };

  test('a panel with a cached frame opens in a window, and Reattach docks it', () => {
    holder.restore(AGENT_ONLY);
    expect(mod.detachBlocker('ariel')).toBeNull();
    expect(mod.detach('ariel')).toBe(true);
    expect(holder.isDetached('ariel')).toBe(true);
    expect(holder.detachedEntry('ariel').record).toBeNull();
    expect(holder.detachedEntry('ariel').panelRecord).toEqual({ title: 'LOGBOOK' });
    expect(iframe.ownerDocument).toBe(popups[0].document);

    expect(mod.reattach('ariel')).toBe(true);
    expect(holder.holds('ariel')).toBe(true);
    expect(holder.holds('terminal')).toBe(true);
    expect(iframe.parentElement).toBe(overlay);
    expect(deps.panelReturned).toHaveBeenCalledExactlyOnceWith('ariel');
  });

  test('a panel that never loaded gets its frame, undocked, inside the gesture', () => {
    holder.restore(AGENT_ONLY);
    iframe.remove();
    let loaded = false;
    deps.managedIframe.mockImplementation((/** @type {string} */ id) =>
      id === 'ariel' && loaded ? { iframe, title: 'LOGBOOK' } : null
    );
    mod.initPaneWindow({
      canLoadFrame: () => true,
      loadFrame: vi.fn(() => {
        overlay.append(iframe);
        loaded = true;
      }),
    });
    expect(mod.detachBlocker('ariel')).toBeNull();
    expect(mod.detach('ariel')).toBe(true);
    expect(iframe.ownerDocument).toBe(popups[0].document);
  });

  test('a panel with no resolved URL stays blocked, with today’s reason', () => {
    holder.restore(AGENT_ONLY);
    deps.managedIframe.mockReturnValue(null);
    mod.initPaneWindow({ canLoadFrame: () => false, loadFrame: vi.fn() });
    expect(mod.detachBlocker('ariel')).toMatch(/has not loaded/);
    expect(mod.detach('ariel')).toBe(false);
    expect(popups).toHaveLength(0);
  });

  test('a first click on a never-loaded panel opens the window, then moves the frame in', () => {
    vi.useFakeTimers();
    holder.restore(AGENT_ONLY);
    let loaded = false;
    deps.managedIframe.mockImplementation((/** @type {string} */ id) =>
      id === 'ariel' && loaded ? { iframe, title: 'LOGBOOK' } : null
    );
    mod.initPaneWindow({
      canLoadFrame: () => true,
      loadFrame: () => {
        iframe.dataset.panelFrame = 'pending';
        loaded = true;
      },
    });
    expect(mod.detach('ariel')).toBe(true);
    expect(popups).toHaveLength(1);
    expect(deps.showNotice).not.toHaveBeenCalled();
    expect(holder.isDetached('ariel')).toBe(false);

    iframe.dataset.panelFrame = 'first-party';
    vi.advanceTimersByTime(attached.READY_POLL_MS);
    expect(holder.isDetached('ariel')).toBe(true);
    expect(iframe.ownerDocument).toBe(popups[0].document);
    delete iframe.dataset.panelFrame;
  });

  test('a panel with no tile at the gesture is not aborted by the marker wait', () => {
    vi.useFakeTimers();
    holder.restore(AGENT_ONLY);
    nextLoaded = false;
    mod.detach('ariel');
    popups[0].document.documentElement.setAttribute(SHELL_MARKER, '');
    vi.advanceTimersByTime(attached.READY_POLL_MS);
    expect(holder.isDetached('ariel')).toBe(true);
    expect(popups[0].closed).toBe(false);
  });

  test('the closed Agent pane opens in a window and reattaches', () => {
    holder.restore({ tile: 't1', tabs: ['ariel'], showing: 'ariel' });
    expect(mod.detachBlocker('terminal')).toBeNull();
    expect(mod.detach('terminal')).toBe(true);
    expect(agent.ownerDocument).toBe(popups[0].document);
    expect(mod.reattach('terminal')).toBe(true);
    expect(holder.holds('terminal')).toBe(true);
    expect(holder.holds('ariel')).toBe(true);
    expect(agent.parentElement).toBe(host);
  });

  test('closing its window closes it: back out of sight, no tile', () => {
    vi.useFakeTimers();
    holder.restore(AGENT_ONLY);
    mod.detach('ariel');
    popups[0].close();
    vi.advanceTimersByTime(attached.CLOSED_POLL_MS);
    expect(holder.isDetached('ariel')).toBe(false);
    expect(holder.holds('ariel')).toBe(false);
    expect(deps.concealPanel).toHaveBeenCalledExactlyOnceWith('ariel');
  });
});

describe('the stored page follows the frame', () => {
  test('in-window navigation: a frame load stores the page and is announced', () => {
    const changes = vi.fn();
    holder.onDetachedChange(changes);
    mod.detach('ariel');
    changes.mockClear();
    pages.set('ariel', '/panel/ariel/#entry?id=Y');
    iframe.dispatchEvent(new Event('load'));
    expect(holder.detachedEntry('ariel').page).toBe('/panel/ariel/#entry?id=Y');
    expect(changes).toHaveBeenCalledTimes(1);
    // The same page again is no change, and an unreadable one keeps the last.
    iframe.dispatchEvent(new Event('load'));
    pages.set('ariel', null);
    iframe.dispatchEvent(new Event('load'));
    expect(holder.detachedEntry('ariel').page).toBe('/panel/ariel/#entry?id=Y');
    expect(changes).toHaveBeenCalledTimes(1);
  });

  test('the persisted form carries the pane on the page it moved to', () => {
    mod.detach('ariel');
    pages.set('ariel', '/panel/ariel/#entry?id=Y');
    iframe.dispatchEvent(new Event('load'));
    expect(holder.persisted()).toBeTruthy();
    expect(holder.detachedEntry('ariel').page).toBe('/panel/ariel/#entry?id=Y');
  });

  test('a frame load after its src changed updates the stored page', () => {
    mod.detach('ariel');
    pages.set('ariel', '/panel/ariel/#entry?id=Z');
    iframe.src = 'http://hub/panel/ariel/#entry?id=Z?embedded=true';
    iframe.dispatchEvent(new Event('load'));
    expect(holder.detachedEntry('ariel').page).toBe('/panel/ariel/#entry?id=Z');
  });
});

describe('a hub navigation of a detached frame stands', () => {
  const TARGET = '/panel/ariel/#entry?id=Z';
  const TARGET_SRC = 'http://hub/panel/ariel/#entry?id=Z?embedded=true';

  /** @param {ReturnType<typeof fakePopup>} win */
  function finishReload(win) {
    win.document = shellDocument();
    vi.advanceTimersByTime(attached.READY_POLL_MS);
  }

  test('navigated between the pagehide and the re-dress, the new document loads it', () => {
    vi.useFakeTimers();
    mod.detach('ariel');
    popups[0].dispatchEvent(new Event('pagehide'));
    pages.set('ariel', null);
    mod.noteDetachedNavigation('ariel', TARGET);
    finishReload(popups[0]);
    expect(iframe.src).toBe(TARGET_SRC);
    expect(holder.detachedEntry('ariel').page).toBe(TARGET);
  });

  test('navigated and reloaded before the new page loaded, the old page does not come back', () => {
    vi.useFakeTimers();
    mod.detach('ariel');
    // The frame still shows the old page: its `load` has not fired.
    mod.noteDetachedNavigation('ariel', TARGET);
    popups[0].dispatchEvent(new Event('pagehide'));
    finishReload(popups[0]);
    expect(iframe.src).toBe(TARGET_SRC);
    expect(holder.detachedEntry('ariel').page).toBe(TARGET);
  });

  test('reattached before the new page loaded, it docks on the target', () => {
    mod.detach('ariel');
    mod.noteDetachedNavigation('ariel', TARGET);
    expect(mod.reattach('ariel')).toBe(true);
    expect(iframe.src).toBe(TARGET_SRC);
  });

  test('once the page has loaded, the frame says where it is again', () => {
    mod.detach('ariel');
    mod.noteDetachedNavigation('ariel', TARGET);
    pages.set('ariel', '/panel/ariel/#entry?id=W');
    iframe.dispatchEvent(new Event('load'));
    expect(holder.detachedEntry('ariel').page).toBe('/panel/ariel/#entry?id=W');
  });

  test('a pane that is not detached has nothing to note', () => {
    expect(() => mod.noteDetachedNavigation('ariel', TARGET)).not.toThrow();
    mod.detach('ariel');
    expect(holder.detachedEntry('ariel').page).toBe('/panel/ariel/#entry?id=X');
  });
});

describe('reattach', () => {
  test('docks beside the neighbour, sets src from the live page, moves back, closes the window', () => {
    mod.detach('ariel');
    pages.set('ariel', '/panel/ariel/#entry?id=Y');
    expect(mod.reattach('ariel')).toBe(true);
    expect(holder.isDetached('ariel')).toBe(false);
    expect(holder.holds('ariel')).toBe(true);
    expect(iframe.parentElement).toBe(overlay);
    expect(iframe.getAttribute('style')).toBe('position: absolute; left: 7px;');
    expect(iframe.src).toBe('http://hub/panel/ariel/#entry?id=Y?embedded=true');
    expect(popups[0].closed).toBe(true);
    expect(deps.panelReturned).toHaveBeenCalledExactlyOnceWith('ariel');
    expect(deps.concealPanel).not.toHaveBeenCalled();
    expect(overlay.querySelector('[data-pane-anchor]')).toBeNull();
  });

  test('no room: a notice names the pane, the entry and the window stay', () => {
    holder.resized(null, { width: 70, height: 200 });
    mod.detach('ariel');
    expect(mod.reattach('ariel')).toBe(false);
    expect(deps.showNotice).toHaveBeenCalledExactlyOnceWith(
      'No room to reattach Logbook.',
      undefined
    );
    expect(holder.isDetached('ariel')).toBe(true);
    expect(popups[0].closed).toBe(false);
    expect(iframe.ownerDocument).toBe(popups[0].document);
  });

  test('the bar’s Reattach is this path', () => {
    mod.detach('terminal');
    /** @type {HTMLElement} */ (popups[0].document.querySelector('.fake-reattach')).click();
    expect(holder.isDetached('terminal')).toBe(false);
    expect(agent.parentElement).toBe(host);
    expect(popups[0].closed).toBe(true);
  });

  test('a pane that is not detached does not reattach', () => {
    expect(mod.reattach('ariel')).toBe(false);
    expect(deps.showNotice).not.toHaveBeenCalled();
  });
});

describe('closing the window closes the pane (R4)', () => {
  test('a service pane: forgotten first, then concealed in the overlay', () => {
    vi.useFakeTimers();
    /** @type {boolean[]} */
    const detachedAtConceal = [];
    deps.concealPanel.mockImplementation(() => detachedAtConceal.push(holder.isDetached('ariel')));
    mod.detach('ariel');
    popups[0].close();
    vi.advanceTimersByTime(attached.CLOSED_POLL_MS);
    expect(holder.isDetached('ariel')).toBe(false);
    expect(holder.holds('ariel')).toBe(false);
    expect(iframe.parentElement).toBe(overlay);
    expect(deps.concealPanel).toHaveBeenCalledExactlyOnceWith('ariel');
    expect(detachedAtConceal).toEqual([false]);
    expect(deps.panelReturned).not.toHaveBeenCalled();
  });

  test('the Agent pane: back at its host anchor, not docked', () => {
    vi.useFakeTimers();
    mod.detach('terminal');
    popups[0].close();
    vi.advanceTimersByTime(attached.CLOSED_POLL_MS);
    expect(holder.isDetached('terminal')).toBe(false);
    expect(holder.holds('terminal')).toBe(false);
    expect(agent.parentElement).toBe(host);
  });

  test('a slow close is still R4: pagehide, then `closed` turns true on a later tick', () => {
    vi.useFakeTimers();
    mod.detach('ariel');
    pages.set('ariel', '/panel/ariel/#entry?id=Y');
    popups[0].dispatchEvent(new Event('pagehide'));
    expect(holder.isDetached('ariel')).toBe(true);
    popups[0].closed = true;
    vi.advanceTimersByTime(attached.CLOSED_POLL_MS);
    expect(holder.isDetached('ariel')).toBe(false);
    expect(holder.holds('ariel')).toBe(false);
    expect(deps.concealPanel).toHaveBeenCalledExactlyOnceWith('ariel');
    // The last page it showed is the one the hidden frame keeps.
    expect(iframe.src).toBe('http://hub/panel/ariel/#entry?id=Y?embedded=true');
  });

  test('closeDetached takes the same path and closes the window', () => {
    mod.detach('ariel');
    expect(mod.closeDetached('ariel')).toBe(true);
    expect(popups[0].closed).toBe(true);
    expect(deps.concealPanel).toHaveBeenCalledExactlyOnceWith('ariel');
    expect(mod.closeDetached('ariel')).toBe(false);
  });

  test('a removed anchor is no orphan: the frame returns to the overlay', () => {
    vi.useFakeTimers();
    mod.detach('ariel');
    overlay.querySelector('[data-pane-anchor]')?.remove();
    popups[0].close();
    vi.advanceTimersByTime(attached.CLOSED_POLL_MS);
    expect(iframe.parentElement).toBe(overlay);
  });
});

describe('a reload of the window keeps the pane detached (R10)', () => {
  /** Reload the shell: the old document's pagehide, then a new one with the marker. */
  function reload(/** @type {ReturnType<typeof fakePopup>} */ win) {
    win.dispatchEvent(new Event('pagehide'));
    win.document = shellDocument();
    vi.advanceTimersByTime(attached.READY_POLL_MS);
  }

  test('a service frame is dressed again, on the page it was on, with its bar', () => {
    vi.useFakeTimers();
    const onChange = vi.fn();
    mod.onDetachedChange(onChange);
    mod.detach('ariel');
    pages.set('ariel', '/panel/ariel/#entry?id=Y');
    reload(popups[0]);
    expect(holder.isDetached('ariel')).toBe(true);
    expect(holder.detachedEntry('ariel').page).toBe('/panel/ariel/#entry?id=Y');
    const doc = popups[0].document;
    expect(iframe.ownerDocument).toBe(doc);
    expect(iframe.src).toBe('http://hub/panel/ariel/#entry?id=Y?embedded=true');
    expect(doc.querySelector('.fake-tab')).not.toBeNull();
    expect(doc.querySelector('.attached-window-root')?.getAttribute('data-osprey-detached')).toBe(
      'true'
    );
    expect(onChange).toHaveBeenCalledTimes(1);
    // The new document's Reattach works, and only one return follows.
    /** @type {HTMLElement} */ (doc.querySelector('.fake-reattach')).click();
    expect(holder.isDetached('ariel')).toBe(false);
    expect(iframe.parentElement).toBe(overlay);
  });

  test('the Agent is refit in the new document', () => {
    vi.useFakeTimers();
    mod.detach('terminal');
    deps.fitAgent.mockClear();
    reload(popups[0]);
    expect(agent.ownerDocument).toBe(popups[0].document);
    expect(deps.fitAgent).toHaveBeenCalled();
    expect(holder.isDetached('terminal')).toBe(true);
  });

  test('a second reload works the same way, and a close after it is R4', () => {
    vi.useFakeTimers();
    mod.detach('ariel');
    reload(popups[0]);
    reload(popups[0]);
    expect(iframe.ownerDocument).toBe(popups[0].document);
    popups[0].close();
    vi.advanceTimersByTime(attached.CLOSED_POLL_MS);
    expect(holder.isDetached('ariel')).toBe(false);
    expect(deps.concealPanel).toHaveBeenCalledExactlyOnceWith('ariel');
  });

  test('a marker that never returns reattaches the pane', () => {
    vi.useFakeTimers();
    mod.detach('ariel');
    popups[0].dispatchEvent(new Event('pagehide'));
    vi.advanceTimersByTime(attached.READY_TIMEOUT_MS + attached.READY_POLL_MS);
    expect(holder.isDetached('ariel')).toBe(false);
    expect(holder.holds('ariel')).toBe(true);
    expect(iframe.parentElement).toBe(overlay);
    expect(popups[0].closed).toBe(true);
  });

  test('a marker that never returns with no room closes the pane', () => {
    vi.useFakeTimers();
    holder.resized(null, { width: 70, height: 200 });
    mod.detach('ariel');
    popups[0].dispatchEvent(new Event('pagehide'));
    vi.advanceTimersByTime(attached.READY_TIMEOUT_MS + attached.READY_POLL_MS);
    expect(holder.isDetached('ariel')).toBe(false);
    expect(iframe.parentElement).toBe(overlay);
    expect(deps.concealPanel).toHaveBeenCalledExactlyOnceWith('ariel');
  });
});

describe('the hub page goes away (R7)', () => {
  test('the windows close; the detached map, popouts and persisted form stay', () => {
    vi.useFakeTimers();
    mod.detach('ariel');
    const before = holder.persisted();
    window.dispatchEvent(new Event('pagehide'));
    expect(popups[0].closed).toBe(true);
    // The save a following visibilitychange runs still has the pane docked.
    document.dispatchEvent(new Event('visibilitychange'));
    vi.advanceTimersByTime(attached.CLOSED_POLL_MS * 3);
    expect(holder.isDetached('ariel')).toBe(true);
    expect(holder.popouts).toEqual(['ariel']);
    expect(holder.persisted()).toEqual(before);
    expect(deps.concealPanel).not.toHaveBeenCalled();
  });
});

describe('theme', () => {
  test('a theme change reaches the detached frame and no other', () => {
    expect(themeListeners).toHaveLength(1);
    themeListeners[0]();
    expect(deps.sendThemeToIframe).not.toHaveBeenCalled();
    mod.detach('ariel');
    mod.detach('terminal');
    themeListeners[0]();
    expect(deps.sendThemeToIframe).toHaveBeenCalledExactlyOnceWith(iframe);
  });

  test('a second init does not subscribe twice', () => {
    mod.initPaneWindow({ subscribeTheme: deps.subscribeTheme });
    expect(themeListeners).toHaveLength(1);
  });
});

describe('the window’s bar is the tile’s bar', () => {
  /** @returns {any} the tab built last */
  const lastTab = () => deps.createTileTab.mock.results.at(-1).value;

  test.each([
    ['ariel', 'Logbook'],
    ['terminal', 'Agent'],
  ])('%s: a detached tab, named, above the content, attached once in the window', (id, title) => {
    mod.detach(id);
    const tab = lastTab();
    expect(tab.id).toBe(id);
    expect(tab.mode).toBe('detached');
    expect(tab.init).toHaveBeenCalledExactlyOnceWith({ title });
    const mount = popups[0].document.querySelector('.attached-window-root');
    expect(mount?.firstElementChild).toBe(tab.element);
    expect(tabDocuments).toEqual([popups[0].document]);
  });

  test('the tab is disposed on reattach, and on a close', () => {
    mod.detach('ariel');
    const tab = lastTab();
    expect(mod.reattach('ariel')).toBe(true);
    expect(tab.dispose).toHaveBeenCalledTimes(1);

    mod.detach('terminal');
    const agentTab = lastTab();
    mod.closeDetached('terminal');
    expect(agentTab.dispose).toHaveBeenCalledTimes(1);
  });

  test('a window closed by the operator disposes its tab too', () => {
    vi.useFakeTimers();
    mod.detach('ariel');
    const tab = lastTab();
    popups[0].close();
    vi.advanceTimersByTime(attached.CLOSED_POLL_MS);
    expect(tab.dispose).toHaveBeenCalledTimes(1);
  });

  test('a reload builds a new tab on the new document and disposes the old (R10)', () => {
    vi.useFakeTimers();
    mod.detach('ariel');
    const first = lastTab();
    popups[0].dispatchEvent(new Event('pagehide'));
    popups[0].document = shellDocument(true);
    vi.advanceTimersByTime(attached.READY_POLL_MS);
    const second = lastTab();
    expect(second).not.toBe(first);
    expect(first.dispose).toHaveBeenCalledTimes(1);
    expect(tabDocuments).toEqual([expect.anything(), popups[0].document]);
    expect(popups[0].document.querySelectorAll('.fake-tab')).toHaveLength(1);
    expect(second.dispose).not.toHaveBeenCalled();
  });

  test('“No room” from the window’s own bar shows in that window', () => {
    mod.detach('ariel');
    holder.reattach = vi.fn(() => false);
    /** @type {HTMLElement} */ (popups[0].document.querySelector('.fake-reattach')).click();
    expect(deps.showNotice).toHaveBeenCalledExactlyOnceWith(
      'No room to reattach Logbook.',
      popups[0].document
    );
    expect(popups[0].closed).toBe(false);
  });
});
