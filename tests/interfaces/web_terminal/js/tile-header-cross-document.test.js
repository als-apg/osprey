// @ts-check
// @vitest-environment jsdom
/**
 * The tile bar's contribution region and its menu popover in a second document,
 * the way a pane's window is to the hub. A same-origin iframe stands in for that
 * window: jsdom gives it a document, a `defaultView` and a realm of its own.
 *
 *   npx vitest run tests/interfaces/web_terminal/js/tile-header-cross-document.test.js
 */

import { afterEach, beforeAll, beforeEach, describe, expect, test, vi } from 'vitest';

beforeAll(() => {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

import {
  contributeLocal,
  registerContribHost,
  unregisterContribHost,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/tile-header-contrib.js';
import { closeMenu } from '../../../../src/osprey/interfaces/web_terminal/static/js/tile-header-items.js';

let seq = 0;
const freshId = () => `cross-panel-${(seq += 1)}`;

/** @returns {{win: any, doc: any}} a window of its own */
function otherWindow() {
  const frame = document.createElement('iframe');
  document.body.append(frame);
  const win = /** @type {any} */ (frame.contentWindow);
  return { win, doc: win.document };
}

const MENU = [{ kind: 'menu', id: 'more', items: [{ id: 'a', label: 'Row' }] }];

/** A host in the hub's realm, moved into `doc`, as a TileTab's region is. */
function hostIn(/** @type {any} */ doc) {
  const host = document.createElement('div');
  doc.body.append(host);
  return host;
}

beforeEach(() => {
  document.body.replaceChildren();
});

afterEach(() => {
  closeMenu();
});

describe('registerContribHost', () => {
  test('the observer comes from the host’s realm', () => {
    const { win, doc } = otherWindow();
    const observe = vi.fn();
    const made = vi.fn();
    win.ResizeObserver = class {
      constructor() {
        made();
      }
      observe = observe;
      disconnect() {}
    };
    const hubObserver = vi.spyOn(globalThis, 'ResizeObserver');
    const id = freshId();
    const host = hostIn(doc);
    registerContribHost(id, host);
    expect(made).toHaveBeenCalledTimes(1);
    expect(observe).toHaveBeenCalledExactlyOnceWith(host);
    expect(hubObserver).not.toHaveBeenCalled();
    unregisterContribHost(id, host);
    hubObserver.mockRestore();
  });

  test('a host in the hub keeps the hub’s observer', () => {
    const hubObserver = vi.spyOn(globalThis, 'ResizeObserver');
    const id = freshId();
    const host = hostIn(document);
    registerContribHost(id, host);
    expect(hubObserver).toHaveBeenCalledTimes(1);
    unregisterContribHost(id, host);
    hubObserver.mockRestore();
  });

  test('a re-register replaces the old host’s observer and listeners', () => {
    const { doc } = otherWindow();
    const id = freshId();
    /** @type {any[]} */
    const observers = [];
    const Real = globalThis.ResizeObserver;
    globalThis.ResizeObserver = class extends Real {
      constructor(/** @type {ResizeObserverCallback} */ cb) {
        super(cb);
        this.disconnect = vi.fn();
        observers.push(this);
      }
    };
    const oldHost = hostIn(document);
    registerContribHost(id, oldHost);
    const removed = vi.spyOn(AbortSignal.prototype, 'addEventListener');
    contributeLocal(id, [{ kind: 'search', id: 's', placeholder: 'Find' }], vi.fn());
    const oldWrap = /** @type {HTMLElement} */ (oldHost.querySelector('.contrib-search'));
    oldWrap.classList.add('contrib-search-collapsed');

    const newHost = hostIn(document);
    registerContribHost(id, newHost);
    expect(observers[0].disconnect).toHaveBeenCalledTimes(1);

    // The old host's delegated click no longer expands its search.
    oldWrap.click();
    expect(oldWrap.classList.contains('contrib-search-collapsed')).toBe(true);
    // The new one's does, and it was rendered from the cached contribution.
    const newWrap = /** @type {HTMLElement} */ (newHost.querySelector('.contrib-search'));
    newWrap.classList.add('contrib-search-collapsed');
    newWrap.click();
    expect(newWrap.classList.contains('contrib-search-collapsed')).toBe(false);

    // Unregistering the replaced host is a no-op; the new one stays live.
    unregisterContribHost(id, oldHost);
    expect(observers[1].disconnect).not.toHaveBeenCalled();
    unregisterContribHost(id, newHost);
    expect(observers[1].disconnect).toHaveBeenCalledTimes(1);
    removed.mockRestore();
    doc.body.replaceChildren();
    globalThis.ResizeObserver = Real;
  });

  test('a menu left open on the old host closes when the host is replaced', () => {
    const id = freshId();
    const oldHost = hostIn(document);
    registerContribHost(id, oldHost);
    contributeLocal(id, MENU, vi.fn());
    /** @type {HTMLElement} */ (oldHost.querySelector('.contrib-menu-btn')).click();
    expect(document.querySelector('.contrib-menu-popover')).not.toBeNull();
    const newHost = hostIn(document);
    registerContribHost(id, newHost);
    expect(document.querySelector('.contrib-menu-popover')).toBeNull();
    unregisterContribHost(id, newHost);
  });
});

describe('a contributed menu in another document', () => {
  function openIn() {
    const { win, doc } = otherWindow();
    win.ResizeObserver = globalThis.ResizeObserver;
    const id = freshId();
    const host = hostIn(doc);
    registerContribHost(id, host);
    contributeLocal(id, MENU, vi.fn());
    const btn = /** @type {HTMLElement} */ (host.querySelector('.contrib-menu-btn'));
    btn.click();
    return { win, doc, btn };
  }

  /** @param {any} doc */
  const popIn = (doc) => doc.querySelector('.contrib-menu-popover');

  test('mounts on that document’s body', () => {
    const { doc } = openIn();
    expect(popIn(doc)?.parentElement).toBe(doc.body);
    expect(popIn(document)).toBeNull();
  });

  test('clamps to that window', () => {
    const { win, doc, btn } = openIn();
    closeMenu();
    win.innerWidth = 300;
    window.innerWidth = 5000;
    btn.getBoundingClientRect = () =>
      /** @type {DOMRect} */ ({ right: 800, bottom: 30, top: 10, left: 780 });
    btn.click();
    // right edge - 0 width, clamped: left = min(800, 300 - 0 - 4)
    expect(popIn(doc).style.left).toBe('296px');
  });

  test('its window losing focus closes it (a click into the panel)', () => {
    const { win, doc } = openIn();
    window.dispatchEvent(new Event('blur'));
    expect(popIn(doc)).not.toBeNull();
    win.dispatchEvent(new win.Event('blur'));
    expect(popIn(doc)).toBeNull();
  });

  test('Escape in that document closes it', () => {
    const { win, doc } = openIn();
    doc.dispatchEvent(new win.KeyboardEvent('keydown', { key: 'Escape' }));
    expect(popIn(doc)).toBeNull();
  });

  test('a press on a node of that realm closes it, and one on its button does not', () => {
    const { win, doc, btn } = openIn();
    btn.dispatchEvent(new win.Event('pointerdown', { bubbles: true }));
    expect(popIn(doc)).not.toBeNull();
    const outside = doc.createElement('div');
    doc.body.append(outside);
    outside.dispatchEvent(new win.Event('pointerdown', { bubbles: true }));
    expect(popIn(doc)).toBeNull();
  });

  test('a scroll of a node of that realm that does not hold the button leaves it open', () => {
    const { win, doc } = openIn();
    const inner = doc.createElement('div');
    doc.body.append(inner);
    inner.dispatchEvent(new win.Event('scroll'));
    expect(popIn(doc)).not.toBeNull();
  });

  test('closing it removes its listeners from that document and window', () => {
    const { win, doc } = openIn();
    const docRemove = vi.spyOn(doc, 'removeEventListener');
    const winRemove = vi.spyOn(win, 'removeEventListener');
    closeMenu();
    expect(docRemove.mock.calls.map((c) => c[0]).sort()).toEqual(['keydown', 'pointerdown']);
    expect(winRemove.mock.calls.map((c) => c[0]).sort()).toEqual(['blur', 'resize', 'scroll']);
  });
});
