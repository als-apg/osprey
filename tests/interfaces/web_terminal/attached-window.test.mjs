// @vitest-environment jsdom
/**
 * The attached pop-out window (attached-window.js): a blank window the hub
 * opens, its styles and <html> attributes mirrored from the hub, a live node
 * of the hub moved into it, and the node back in its place when the window
 * closes, when the hub asks for it, and when the hub page goes away.
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
function standIn() {
  const el = document.createElement('div');
  el.className = 'stand-in';
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
  // The hub's unload returns whatever a test left out.
  window.dispatchEvent(new Event('pagehide'));
  vi.unstubAllGlobals();
  vi.useRealTimers();
  document.documentElement.removeAttribute('data-theme');
});

describe('open', () => {
  test('opens a named popup on the shell page and moves the live node into it, the stand-in in its place', () => {
    const changes = /** @type {[string, boolean][]} */ ([]);
    mod.onAttachedChange((/** @type {string} */ key, /** @type {boolean} */ out) =>
      changes.push([key, out])
    );
    const marker = standIn();
    const win = mod.openAttached({ key: 'ariel', title: 'Ariel', content, standIn: marker });

    expect(win).toBe(popups[0]);
    const [url, name, features] = open.mock.calls[0];
    expect(url).toBe('/static/attached-window.html');
    expect(name).toBe('osprey-attached-ariel');
    expect(features).toMatch(/^popup=yes,width=\d+,height=\d+/);
    expect(win.document.title).toBe('Ariel');
    expect(win.document.querySelector('.attached-window-root > .live')).toBe(content);
    expect([...parent.children]).toEqual([marker, after]);
    expect(mod.isAttached('ariel')).toBe(true);
    expect(mod.attachedWindow('ariel')).toBe(win);
    expect(changes).toEqual([['ariel', true]]);
  });

  test('a shell page under a URL prefix is opened under it', () => {
    vi.stubGlobal('__OSPREY_PREFIX__', '/u/alice');
    mod.openAttached({ key: 'k', title: 't', content, standIn: standIn() });
    expect(open.mock.calls[0][0]).toBe('/u/alice/static/attached-window.html');
  });

  test('the window has a bar with the pane’s name and Reattach, above the node', () => {
    const onReturn = vi.fn();
    const win = mod.openAttached({
      key: 'k',
      title: 'Agent · OSPREY',
      label: 'Agent',
      content,
      standIn: standIn(),
      onReturn,
    });
    const root = win.document.querySelector('.attached-window-root');
    const bar = root.firstElementChild;
    expect(bar.className).toBe('attached-window-bar');
    expect(bar.querySelector('.attached-window-title').textContent).toBe('Agent');
    expect(root.lastElementChild).toBe(content);
    const button = bar.querySelector('button.attached-window-reattach');
    expect(button.textContent).toBe('Reattach');

    button.click();

    expect(content.parentElement).toBe(parent);
    expect(win.close).toHaveBeenCalledTimes(1);
    expect(onReturn).toHaveBeenCalledExactlyOnceWith('bring-back');
    expect(mod.isAttached('k')).toBe(false);
  });

  test('the bar names the title when no label is given', () => {
    const win = mod.openAttached({ key: 'k', title: 'Ariel', content, standIn: standIn() });
    expect(win.document.querySelector('.attached-window-title').textContent).toBe('Ariel');
  });

  test('the node moves only once the shell has loaded', () => {
    vi.useFakeTimers();
    open.mockImplementation(() => {
      const win = fakePopup(false);
      popups.push(win);
      return win;
    });
    const onOpen = vi.fn();
    const changes = /** @type {[string, boolean][]} */ ([]);
    mod.onAttachedChange((/** @type {string} */ key, /** @type {boolean} */ out) =>
      changes.push([key, out])
    );
    const win = mod.openAttached({ key: 'k', title: 't', content, standIn: standIn(), onOpen });
    expect(win).toBe(popups[0]);

    vi.advanceTimersByTime(mod.READY_POLL_MS * 4);
    expect(content.parentElement).toBe(parent);
    expect(mod.isAttached('k')).toBe(false);
    expect(mod.attachedWindow('k')).toBeNull();
    expect(onOpen).not.toHaveBeenCalled();
    // A second Pop out meanwhile brings the loading window forward.
    expect(mod.openAttached({ key: 'k', title: 't', content, standIn: standIn() })).toBe(win);
    expect(open).toHaveBeenCalledTimes(1);

    win.document = shellDocument(true);
    vi.advanceTimersByTime(mod.READY_POLL_MS);
    expect(win.document.querySelector('.attached-window-root > .live')).toBe(content);
    expect(mod.isAttached('k')).toBe(true);
    expect(onOpen).toHaveBeenCalledTimes(1);
    expect(changes).toEqual([['k', true]]);
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
      standIn: standIn(),
      onReturn,
    });
    closed.closed = true;
    vi.advanceTimersByTime(mod.CLOSED_POLL_MS);
    expect(content.parentElement).toBe(parent);
    expect(onReturn).not.toHaveBeenCalled();

    const stuck = mod.openAttached({ key: 'b', title: 't', content, standIn: standIn(), onReturn });
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.advanceTimersByTime(mod.READY_TIMEOUT_MS + mod.READY_POLL_MS);
    expect(stuck.close).toHaveBeenCalledTimes(1);
    expect(content.parentElement).toBe(parent);
    expect(onReturn).not.toHaveBeenCalled();
    // Gone: a new Pop out opens a new window.
    mod.openAttached({ key: 'b', title: 't', content, standIn: standIn() });
    expect(open).toHaveBeenCalledTimes(3);
  });

  test('the hub going away while a shell loads closes that window', () => {
    open.mockImplementation(() => {
      const win = fakePopup(false);
      popups.push(win);
      return win;
    });
    const win = mod.openAttached({ key: 'k', title: 't', content, standIn: standIn() });
    window.dispatchEvent(new Event('pagehide'));
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
    const win = mod.openAttached({ key: 'k', title: 't', content, standIn: standIn() });
    content.remove();
    win.document = shellDocument(true);
    vi.advanceTimersByTime(mod.READY_POLL_MS);
    expect(win.close).toHaveBeenCalledTimes(1);
    expect(content.isConnected).toBe(false);
    expect(mod.isAttached('k')).toBe(false);
  });

  test('a refused window moves nothing', () => {
    open.mockReturnValue(null);
    expect(mod.openAttached({ key: 'k', title: 't', content, standIn: standIn() })).toBeNull();
    expect(content.parentElement).toBe(parent);
    expect(mod.isAttached('k')).toBe(false);
  });

  test('a node with no place to return to is not moved', () => {
    content.remove();
    expect(mod.openAttached({ key: 'k', title: 't', content, standIn: standIn() })).toBeNull();
    expect(open).not.toHaveBeenCalled();
  });

  test('a second open of the same key brings its window forward instead of opening another', () => {
    const win = mod.openAttached({ key: 'k', title: 't', content, standIn: standIn() });
    expect(mod.openAttached({ key: 'k', title: 't', content, standIn: standIn() })).toBe(win);
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

    const win = mod.openAttached({ key: 'k', title: 't', content, standIn: standIn() });
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
    document.documentElement.setAttribute('data-agent', 'popped');
    const win = mod.openAttached({ key: 'k', title: 't', content, standIn: standIn() });
    const root = win.document.documentElement;
    expect(root.getAttribute('data-theme')).toBe('dark');
    expect(root.getAttribute('data-agent')).toBe('popped');

    document.documentElement.setAttribute('data-theme', 'light');
    document.documentElement.removeAttribute('data-agent');
    await flush();
    expect(root.getAttribute('data-theme')).toBe('light');
    expect(root.hasAttribute('data-agent')).toBe(false);
  });
});

describe('return', () => {
  test('closing the window puts the node back in its place, with its style', () => {
    const onReturn = vi.fn();
    const changes = /** @type {[string, boolean][]} */ ([]);
    mod.onAttachedChange((/** @type {string} */ key, /** @type {boolean} */ out) =>
      changes.push([key, out])
    );
    const marker = standIn();
    const win = mod.openAttached({ key: 'k', title: 't', content, standIn: marker, onReturn });
    expect(content.getAttribute('style')).not.toContain('left: 4px');

    win.close();

    expect([...parent.children]).toEqual([content, after]);
    expect(content.getAttribute('style')).toBe('position: absolute; left: 4px;');
    expect(marker.isConnected).toBe(false);
    expect(onReturn).toHaveBeenCalledExactlyOnceWith('closed');
    expect(mod.isAttached('k')).toBe(false);
    expect(changes).toEqual([
      ['k', true],
      ['k', false],
    ]);
  });

  test('a window that closed without a pagehide is caught by the poll', () => {
    vi.useFakeTimers();
    const onReturn = vi.fn();
    const win = mod.openAttached({ key: 'k', title: 't', content, standIn: standIn(), onReturn });
    win.closed = true;
    vi.advanceTimersByTime(mod.CLOSED_POLL_MS);
    expect(content.parentElement).toBe(parent);
    expect(onReturn).toHaveBeenCalledExactlyOnceWith('closed');
  });

  test('Reattach from the hub returns the node and closes the window', () => {
    const onReturn = vi.fn();
    const win = mod.openAttached({ key: 'k', title: 't', content, standIn: standIn(), onReturn });
    mod.returnAttached('k');
    expect(win.close).toHaveBeenCalledTimes(1);
    expect(content.parentElement).toBe(parent);
    expect(onReturn).toHaveBeenCalledExactlyOnceWith('bring-back');
  });

  test('the hub page going away returns every node and closes every window', () => {
    const other = document.createElement('div');
    document.body.append(other);
    const onReturn = vi.fn();
    const a = mod.openAttached({ key: 'a', title: 't', content, standIn: standIn(), onReturn });
    const b = mod.openAttached({ key: 'b', title: 't', content: other, standIn: standIn() });

    window.dispatchEvent(new Event('pagehide'));

    expect(content.parentElement).toBe(parent);
    expect(other.parentElement).toBe(document.body);
    expect(a.closed && b.closed).toBe(true);
    expect(onReturn).toHaveBeenCalledExactlyOnceWith('unload');
    expect(mod.isAttached('a') || mod.isAttached('b')).toBe(false);
  });

  test('a stand-in taken off the page leaves the node with the hub, unattached', () => {
    const marker = standIn();
    const win = mod.openAttached({ key: 'k', title: 't', content, standIn: marker });
    marker.remove();
    win.close();
    expect(content.isConnected).toBe(false);
    expect(content.ownerDocument).toBe(document);
  });
});

describe('the window is the hub’s, in its messages and lookups', () => {
  test('a frame message posted to the window reaches the hub’s listeners, as sent', () => {
    const win = mod.openAttached({ key: 'k', title: 't', content, standIn: standIn() });
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
    mod.openAttached({ key: 'k', title: 't', content, standIn: standIn() });
    expect(mod.queryAllDocuments('iframe.panel-iframe')).toEqual([own, frame]);
  });

  test('the window resizing reaches the caller', () => {
    const onResize = vi.fn();
    const win = mod.openAttached({ key: 'k', title: 't', content, standIn: standIn(), onResize });
    win.dispatchEvent(new Event('resize'));
    expect(onResize).toHaveBeenCalledTimes(1);
  });
});
