// @vitest-environment jsdom
/**
 * The attached window (attached-window.js): a window the hub opens, its styles
 * and <html> attributes mirrored from the hub, a live node of the hub moved into
 * it, the node dressed in again after a reload of the shell, and the node back in
 * its place when the window closes and when the hub asks for it.
 *
 * jsdom opens no windows: `window.open` returns a stand-in window whose
 * document is a second jsdom document, which is what the same-origin shell
 * page is to the hub. It has loaded (it carries the shell's marker) unless a
 * test says otherwise.
 *
 *   npx vitest run tests/interfaces/web_terminal/attached-window.test.mjs
 */
import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const MOD = '../../../src/osprey/interfaces/web_terminal/static/js/attached-window.js';

/** The marker the shell page's <html> carries (static/attached-window.html). */
const SHELL_MARKER = 'data-osprey-attached-shell';

/** @param {boolean} loaded @returns {Document} */
function shellDocument(loaded) {
  const doc = document.implementation.createHTMLDocument('');
  if (loaded) doc.documentElement.setAttribute(SHELL_MARKER, '');
  return doc;
}

/**
 * A popup as the hub sees it: its own document, its own event target, and
 * `close()` firing `pagehide` the way a closing window does.
 * @param {boolean} [loaded]  whether the shell page has loaded yet
 */
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

/** @type {ReturnType<typeof fakePopup>[]} */
let popups;
/** @type {import('vitest').Mock} */
let open;
/** @type {HTMLElement} */
let parent;
/** @type {HTMLElement} */
let content;
/** @type {HTMLElement} */
let after;

/** @returns {HTMLElement} */
function anchorEl() {
  const el = document.createElement('div');
  el.className = 'anchor';
  return el;
}

/** Let the MutationObservers deliver. */
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

/** @type {any} */
let mod;

beforeEach(async () => {
  vi.resetModules();
  document.head.replaceChildren();
  document.body.replaceChildren();
  document.documentElement.setAttribute('data-theme', 'dark');
  popups = [];
  open = vi.fn(() => {
    const win = fakePopup();
    popups.push(win);
    return win;
  });
  vi.stubGlobal('open', open);
  parent = document.createElement('div');
  content = document.createElement('section');
  content.className = 'live';
  content.setAttribute('style', 'position: absolute; left: 4px;');
  after = document.createElement('span');
  parent.append(content, after);
  document.body.append(parent);
  mod = await import(MOD);
});

afterEach(() => {
  mod.closeAllAttached();
  vi.unstubAllGlobals();
  vi.useRealTimers();
  document.documentElement.removeAttribute('data-theme');
});

describe('open', () => {
  test('opens a named popup on the shell page and moves the live node into it, the stand-in in its place', () => {
    const marker = anchorEl();
    const win = mod.openAttached({
      key: 'ariel',
      title: 'Ariel',
      content,
      name: 'w-k',
      anchor: marker,
    });

    expect(win).toBe(popups[0]);
    const [url, name, features] = open.mock.calls[0];
    expect(url).toBe('/static/attached-window.html');
    expect(name).toBe('w-k');
    expect(features).toMatch(/^popup=yes,width=\d+,height=\d+/);
    expect(win.document.title).toBe('Ariel');
    expect(win.document.querySelector('.attached-window-root > .live')).toBe(content);
    expect([...parent.children]).toEqual([marker, after]);
    expect(mod.hasWindow('ariel')).toBe(true);
  });

  test('a shell page under a URL prefix is opened under it', () => {
    vi.stubGlobal('__OSPREY_PREFIX__', '/u/alice');
    mod.openAttached({ key: 'k', title: 't', content, name: 'w-k', anchor: anchorEl() });
    expect(open.mock.calls[0][0]).toBe('/u/alice/static/attached-window.html');
  });

  test('the node leaves at the pagehide, before the old document can strip it', () => {
    /** @type {boolean[]} */
    const order = [];
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
      onPageHide: () => order.push(content.isConnected),
    });
    win.dispatchEvent(new Event('pagehide'));
    // The hook read what it needed first; the node is out now, and in no document.
    expect(order).toEqual([true]);
    expect(content.isConnected).toBe(false);
    expect(mod.hasWindow('k')).toBe(true);
  });

  test('the window has no bar of its own: onDress hands the caller the mount', () => {
    const onDress = vi.fn();
    const win = mod.openAttached({
      key: 'k',
      title: 'Agent · OSPREY',
      content,
      name: 'w-k',
      anchor: anchorEl(),
      onDress,
    });
    const root = win.document.querySelector('.attached-window-root');
    expect([...root.children]).toEqual([content]);
    expect(onDress).toHaveBeenCalledExactlyOnceWith(root);
  });

  test('the node moves only once the shell has loaded', () => {
    vi.useFakeTimers();
    open.mockImplementation(() => {
      const win = fakePopup(false);
      popups.push(win);
      return win;
    });
    const onDress = vi.fn();
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
      onDress,
    });
    expect(win).toBe(popups[0]);

    vi.advanceTimersByTime(mod.READY_POLL_MS * 4);
    expect(content.parentElement).toBe(parent);
    expect(onDress).not.toHaveBeenCalled();
    // A second Open in new window meanwhile brings the loading window forward.
    expect(
      mod.openAttached({ key: 'k', title: 't', content, name: 'w-k', anchor: anchorEl() })
    ).toBe(win);
    expect(open).toHaveBeenCalledTimes(1);

    win.document = shellDocument(true);
    vi.advanceTimersByTime(mod.READY_POLL_MS);
    expect(win.document.querySelector('.attached-window-root > .live')).toBe(content);
    expect(mod.hasWindow('k')).toBe(true);
    expect(onDress).toHaveBeenCalledTimes(1);
  });

  test('a window closed, or never loaded, before its shell loaded moves nothing', () => {
    vi.useFakeTimers();
    open.mockImplementation(() => {
      const win = fakePopup(false);
      popups.push(win);
      return win;
    });
    const onReturn = vi.fn();
    const closed = mod.openAttached({
      key: 'a',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
      onReturn,
    });
    closed.closed = true;
    vi.advanceTimersByTime(mod.CLOSED_POLL_MS);
    expect(content.parentElement).toBe(parent);
    expect(onReturn).not.toHaveBeenCalled();

    const stuck = mod.openAttached({
      key: 'b',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
      onReturn,
    });
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.advanceTimersByTime(mod.READY_TIMEOUT_MS + mod.READY_POLL_MS);
    expect(stuck.close).toHaveBeenCalledTimes(1);
    expect(content.parentElement).toBe(parent);
    expect(onReturn).not.toHaveBeenCalled();
    // Gone: a new Open in new window opens a new window.
    mod.openAttached({ key: 'b', title: 't', content, name: 'w-k', anchor: anchorEl() });
    expect(open).toHaveBeenCalledTimes(3);
  });

  test('beforeMove is asked once the shell has loaded; a no closes the window and moves nothing', () => {
    const onDress = vi.fn();
    const beforeMove = vi.fn(() => false);
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
      beforeMove,
      onDress,
    });
    expect(beforeMove).toHaveBeenCalledTimes(1);
    expect(win.closed).toBe(true);
    expect(content.parentElement).toBe(parent);
    expect(onDress).not.toHaveBeenCalled();
    expect(mod.hasWindow('k')).toBe(false);
  });

  test('closing every window while a shell loads moves nothing', () => {
    open.mockImplementation(() => {
      const win = fakePopup(false);
      popups.push(win);
      return win;
    });
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
    });
    mod.closeAllAttached();
    expect(win.closed).toBe(true);
    expect(content.parentElement).toBe(parent);
  });

  test('a node that lost its place while the shell loaded stays with the hub', () => {
    vi.useFakeTimers();
    open.mockImplementation(() => {
      const win = fakePopup(false);
      popups.push(win);
      return win;
    });
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
    });
    content.remove();
    win.document = shellDocument(true);
    vi.advanceTimersByTime(mod.READY_POLL_MS);
    expect(win.close).toHaveBeenCalledTimes(1);
    expect(content.isConnected).toBe(false);
    expect(mod.hasWindow('k')).toBe(false);
  });

  test('a refused window moves nothing', () => {
    open.mockReturnValue(null);
    expect(
      mod.openAttached({ key: 'k', title: 't', content, name: 'w-k', anchor: anchorEl() })
    ).toBeNull();
    expect(content.parentElement).toBe(parent);
    expect(mod.hasWindow('k')).toBe(false);
  });

  test('a node with no place to return to is not moved', () => {
    content.remove();
    expect(
      mod.openAttached({ key: 'k', title: 't', content, name: 'w-k', anchor: anchorEl() })
    ).toBeNull();
    expect(open).not.toHaveBeenCalled();
  });

  test('a second open of the same key brings its window forward instead of opening another', () => {
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
    });
    expect(
      mod.openAttached({ key: 'k', title: 't', content, name: 'w-k', anchor: anchorEl() })
    ).toBe(win);
    expect(open).toHaveBeenCalledTimes(1);
    expect(win.focus).toHaveBeenCalledTimes(1);
    expect(mod.focusAttached('k')).toBe(true);
    expect(mod.focusAttached('other')).toBe(false);
  });
});

describe('styles and theme are mirrored', () => {
  test('the hub stylesheets are cloned, links by their resolved URL, and followed', async () => {
    const style = document.createElement('style');
    style.textContent = '.live { color: red; }';
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = '/static/css/terminal.css';
    document.head.append(style, link);

    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
    });
    const head = win.document.head;
    expect(head.querySelector('style')?.textContent).toBe('.live { color: red; }');
    expect(head.querySelector('link[rel="stylesheet"]')?.getAttribute('href')).toBe(link.href);
    expect(link.href).toMatch(/^http/);

    const late = document.createElement('style');
    late.textContent = '.late {}';
    document.head.append(late);
    await flush();
    expect([...head.querySelectorAll('style')].map((s) => s.textContent)).toEqual([
      '.live { color: red; }',
      '.late {}',
    ]);
  });

  test('the <html> attributes follow the hub: a theme change and a removed attribute', async () => {
    document.documentElement.setAttribute('data-agent', 'detached');
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
    });
    const root = win.document.documentElement;
    expect(root.getAttribute('data-theme')).toBe('dark');
    expect(root.getAttribute('data-agent')).toBe('detached');

    document.documentElement.setAttribute('data-theme', 'light');
    document.documentElement.removeAttribute('data-agent');
    await flush();
    expect(root.getAttribute('data-theme')).toBe('light');
    expect(root.hasAttribute('data-agent')).toBe(false);
  });
});

describe('return', () => {
  test('closing the window puts the node back in its place, with its style', () => {
    vi.useFakeTimers();
    const onReturn = vi.fn();
    const marker = anchorEl();
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: marker,
      onReturn,
    });
    expect(content.getAttribute('style')).not.toContain('left: 4px');

    win.close();
    vi.advanceTimersByTime(mod.CLOSED_POLL_MS);

    expect([...parent.children]).toEqual([content, after]);
    expect(content.getAttribute('style')).toBe('position: absolute; left: 4px;');
    expect(marker.isConnected).toBe(false);
    expect(onReturn).toHaveBeenCalledExactlyOnceWith('closed');
    expect(mod.hasWindow('k')).toBe(false);
  });

  test('a window that closed without a pagehide is caught by the poll', () => {
    vi.useFakeTimers();
    const onReturn = vi.fn();
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
      onReturn,
    });
    win.closed = true;
    vi.advanceTimersByTime(mod.CLOSED_POLL_MS);
    expect(content.parentElement).toBe(parent);
    expect(onReturn).toHaveBeenCalledExactlyOnceWith('closed');
  });

  test('Reattach from the hub returns the node and closes the window', () => {
    const onReturn = vi.fn();
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
      onReturn,
    });
    mod.returnAttached('k');
    expect(win.close).toHaveBeenCalledTimes(1);
    expect(content.parentElement).toBe(parent);
    expect(onReturn).toHaveBeenCalledExactlyOnceWith('reattach');
  });

  test('closeAttached is a close: the reason is `closed`', () => {
    const onReturn = vi.fn();
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
      onReturn,
    });
    mod.closeAttached('k');
    expect(win.closed).toBe(true);
    expect(content.parentElement).toBe(parent);
    expect(onReturn).toHaveBeenCalledExactlyOnceWith('closed');
  });

  test('closeAllAttached closes every window, moves nothing back and tells no one', () => {
    vi.useFakeTimers();
    const onReturn = vi.fn();
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
      onReturn,
    });
    mod.closeAllAttached();
    vi.advanceTimersByTime(mod.CLOSED_POLL_MS * 2);
    expect(win.closed).toBe(true);
    expect(onReturn).not.toHaveBeenCalled();
    expect(content.ownerDocument).toBe(win.document);
    expect(mod.hasWindow('k')).toBe(false);
  });

  test('an anchor taken off the page leaves the node in the parent it left', () => {
    vi.useFakeTimers();
    const marker = anchorEl();
    const win = mod.openAttached({ key: 'k', title: 't', content, name: 'w-k', anchor: marker });
    marker.remove();
    win.close();
    vi.advanceTimersByTime(mod.CLOSED_POLL_MS);
    expect(content.parentElement).toBe(parent);
    expect(content.ownerDocument).toBe(document);
  });
});

describe('a reload of the shell', () => {
  /** @param {any} win */
  const startReload = (win) => win.dispatchEvent(new Event('pagehide'));

  test('dresses the new document: the same node under a new bar, the listeners on the new page', () => {
    vi.useFakeTimers();
    const onDress = vi.fn();
    const onPageHide = vi.fn();
    const onResize = vi.fn();
    const onReturn = vi.fn();
    const win = mod.openAttached({
      key: 'k',
      title: 'Ariel',
      content,
      name: 'w-k',
      anchor: anchorEl(),
      onDress,
      onPageHide,
      onResize,
      onReturn,
    });
    const first = win.document;

    startReload(win);
    expect(onPageHide).toHaveBeenCalledTimes(1);
    // The old document is still there until the new one commits: not a new shell.
    vi.advanceTimersByTime(mod.READY_POLL_MS * 2);
    expect(onDress).toHaveBeenCalledTimes(1);

    win.document = shellDocument(true);
    vi.advanceTimersByTime(mod.READY_POLL_MS);
    expect(win.document).not.toBe(first);
    expect(win.document.title).toBe('Ariel');
    expect(win.document.querySelector('.attached-window-root')).not.toBeNull();
    expect(content.ownerDocument).toBe(win.document);
    expect(onDress).toHaveBeenCalledTimes(2);
    expect(onReturn).not.toHaveBeenCalled();
    expect(mod.hasWindow('k')).toBe(true);

    win.dispatchEvent(new Event('resize'));
    expect(onResize).toHaveBeenCalledTimes(1);
  });

  test('a pagehide with `closed` already true is not a close yet: the poll says so', () => {
    vi.useFakeTimers();
    const onReturn = vi.fn();
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
      onReturn,
    });
    win.closed = true;
    win.dispatchEvent(new Event('pagehide'));
    expect(onReturn).not.toHaveBeenCalled();
    vi.advanceTimersByTime(mod.CLOSED_POLL_MS);
    expect(onReturn).toHaveBeenCalledExactlyOnceWith('closed');
  });

  test('a window that closes during the wait is a close, on the poll', () => {
    vi.useFakeTimers();
    const onReturn = vi.fn();
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
      onReturn,
    });
    startReload(win);
    win.closed = true;
    vi.advanceTimersByTime(mod.CLOSED_POLL_MS);
    expect(onReturn).toHaveBeenCalledExactlyOnceWith('closed');
    expect(content.parentElement).toBe(parent);
    expect(content.getAttribute('style')).toBe('position: absolute; left: 4px;');
  });

  test('a shell that does not come back is the caller’s call; the node stays out until then', () => {
    vi.useFakeTimers();
    const onShellLost = vi.fn();
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
      onShellLost,
    });
    startReload(win);
    vi.advanceTimersByTime(mod.READY_TIMEOUT_MS + mod.READY_POLL_MS);
    expect(onShellLost).toHaveBeenCalledTimes(1);
    expect(mod.hasWindow('k')).toBe(true);
    expect(content.parentElement).not.toBe(parent);
  });

  test('a window on a page out of reach does not throw from the wait', () => {
    vi.useFakeTimers();
    const onShellLost = vi.fn();
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
      onShellLost,
    });
    startReload(win);
    Object.defineProperty(win, 'document', {
      get() {
        throw new DOMException('cross-origin');
      },
    });
    vi.advanceTimersByTime(mod.READY_TIMEOUT_MS + mod.READY_POLL_MS);
    expect(onShellLost).toHaveBeenCalledTimes(1);
  });
});

describe('the window is the hub’s, in its messages and lookups', () => {
  test('a frame message posted to the window reaches the hub’s listeners, as sent', () => {
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
    });
    /** @type {MessageEvent[]} */
    const heard = [];
    const listen = (/** @type {MessageEvent} */ e) => heard.push(e);
    window.addEventListener('message', listen);
    win.dispatchEvent(
      new MessageEvent('message', { data: { type: 'hello' }, origin: 'http://hub.test' })
    );
    window.removeEventListener('message', listen);
    expect(heard).toHaveLength(1);
    expect(heard[0].data).toEqual({ type: 'hello' });
    expect(heard[0].origin).toBe('http://hub.test');
  });

  test('queryAllDocuments finds the moved frames beside the hub’s own', () => {
    const frame = document.createElement('iframe');
    frame.className = 'panel-iframe';
    content.append(frame);
    const own = document.createElement('iframe');
    own.className = 'panel-iframe';
    document.body.append(own);
    expect(mod.queryAllDocuments('iframe.panel-iframe')).toEqual([frame, own]);
    mod.openAttached({ key: 'k', title: 't', content, name: 'w-k', anchor: anchorEl() });
    expect(mod.queryAllDocuments('iframe.panel-iframe')).toEqual([own, frame]);
  });

  test('the window resizing reaches the caller', () => {
    const onResize = vi.fn();
    const win = mod.openAttached({
      key: 'k',
      title: 't',
      content,
      name: 'w-k',
      anchor: anchorEl(),
      onResize,
    });
    win.dispatchEvent(new Event('resize'));
    expect(onResize).toHaveBeenCalledTimes(1);
  });
});
