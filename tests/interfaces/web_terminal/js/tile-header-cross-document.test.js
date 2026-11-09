// @ts-check
// @vitest-environment jsdom
/**
 * The tile bar's contribution region in a second document, the way a pane's
 * window is to the hub. (A contributed menu has no popover of its own: its rows
 * are in the tile ⋯, whose document handling context-menu.js owns.) A same-origin iframe stands in for that
 * window: jsdom gives it a document, a `defaultView` and a realm of its own.
 *
 *   npx vitest run tests/interfaces/web_terminal/js/tile-header-cross-document.test.js
 */

import { beforeAll, beforeEach, describe, expect, test, vi } from 'vitest';

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

let seq = 0;
const freshId = () => `cross-panel-${(seq += 1)}`;

/** @returns {{win: any, doc: any}} a window of its own */
function otherWindow() {
  const frame = document.createElement('iframe');
  document.body.append(frame);
  const win = /** @type {any} */ (frame.contentWindow);
  return { win, doc: win.document };
}

/** A host in the hub's realm, moved into `doc`, as a TileTab's region is. */
function hostIn(/** @type {any} */ doc) {
  const host = document.createElement('div');
  doc.body.append(host);
  return host;
}

beforeEach(() => {
  document.body.replaceChildren();
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
});
