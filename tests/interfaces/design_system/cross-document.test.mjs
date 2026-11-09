// @ts-check
// @vitest-environment jsdom
/**
 * The design system's popover, context menu and tooltip in a second document, the
 * way a pane's window is to the web terminal's hub. A same-origin iframe stands in
 * for that window: jsdom gives it a document, a `defaultView` and a realm of its
 * own, so a node made there is not `instanceof` the hub's classes. jsdom, and not
 * the suite's usual happy-dom, because happy-dom does not adopt a node appended
 * into another document.
 *
 *   npx vitest run tests/interfaces/design_system/cross-document.test.mjs
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import { closePopover, openPopover } from '/design-system/js/popover.js';
import { openContextMenu } from '/design-system/js/context-menu.js';
import { hideTooltip, initTooltips } from '/design-system/js/tooltip.js';

/** @returns {{win: any, doc: any}} a window of its own */
function otherWindow() {
  const frame = document.createElement('iframe');
  document.body.append(frame);
  const win = /** @type {any} */ (frame.contentWindow);
  return { win, doc: win.document };
}

/**
 * What a target's add and remove leave registered.
 * @param {any} target
 * @returns {Set<string>}
 */
function countListeners(target) {
  const live = new Set();
  const add = target.addEventListener.bind(target);
  const remove = target.removeEventListener.bind(target);
  target.addEventListener = (/** @type {[any, any, any]} */ ...args) => {
    live.add(args[0]);
    add(...args);
  };
  target.removeEventListener = (/** @type {[any, any, any]} */ ...args) => {
    live.delete(args[0]);
    remove(...args);
  };
  return live;
}

/**
 * The window events these modules dismiss on; jsdom's frame window registers its
 * own listeners besides.
 * @param {string} type
 */
const isDismissal = (type) => type === 'blur' || type === 'resize';

beforeEach(() => {
  document.body.replaceChildren();
});

describe('a popover opened from another document', () => {
  function setUp() {
    const { win, doc } = otherWindow();
    const anchor = doc.createElement('button');
    doc.body.append(anchor);
    const pop = document.createElement('div');
    pop.className = 'osprey-popover';
    pop.hidden = true;
    document.body.append(pop);
    return { win, doc, anchor, pop };
  }

  test("moves into the opener's document and places itself in that viewport", () => {
    const { win, doc, anchor, pop } = setUp();
    win.innerWidth = 500;
    anchor.getBoundingClientRect = () => ({ top: 10, bottom: 30, left: 400, right: 480 });
    openPopover(pop, anchor);
    expect(pop.ownerDocument).toBe(doc);
    expect(pop.parentElement).toBe(doc.body);
    expect(pop.style.right).toBe('20px');
    closePopover(pop);
  });

  test('its window losing focus closes it, and the hub losing focus does not', () => {
    const { win, anchor, pop } = setUp();
    openPopover(pop, anchor);
    window.dispatchEvent(new Event('blur'));
    expect(pop.hidden).toBe(false);
    win.dispatchEvent(new win.Event('blur'));
    expect(pop.hidden).toBe(true);
    expect(anchor.getAttribute('aria-expanded')).toBe('false');
  });

  test('Escape in that document closes it', () => {
    const { win, doc, anchor, pop } = setUp();
    openPopover(pop, anchor);
    doc.dispatchEvent(new win.KeyboardEvent('keydown', { key: 'Escape' }));
    expect(pop.hidden).toBe(true);
  });

  test('a press outside it in that document closes it, on a node of that realm', () => {
    const { win, doc, anchor, pop } = setUp();
    const outside = doc.createElement('div');
    doc.body.append(outside);
    openPopover(pop, anchor);
    outside.dispatchEvent(new win.Event('pointerdown', { bubbles: true }));
    expect(pop.hidden).toBe(true);
  });

  test('focus comes back to the opener through that document', () => {
    const { doc, anchor, pop } = setUp();
    const row = document.createElement('button');
    row.className = 'osprey-popover-item';
    pop.append(row);
    openPopover(pop, anchor);
    expect(doc.activeElement).toBe(row);
    closePopover(pop);
    expect(doc.activeElement).toBe(anchor);
  });

  test('closing the last one removes every listener from that document and window', () => {
    const { win, doc, anchor, pop } = setUp();
    const onDoc = countListeners(doc);
    const onWin = countListeners(win);
    openPopover(pop, anchor);
    expect(onDoc.size).toBeGreaterThan(0);
    expect([...onWin].filter(isDismissal)).toEqual(['resize', 'blur']);
    closePopover(pop);
    expect(onDoc.size).toBe(0);
    expect([...onWin].filter(isDismissal)).toEqual([]);
  });

  describe('opened while another is open there, it keeps its dismissal', () => {
    function twoOpen() {
      const set = setUp();
      const second = document.createElement('div');
      second.className = 'osprey-popover';
      second.hidden = true;
      document.body.append(second);
      const secondAnchor = set.doc.createElement('button');
      set.doc.body.append(secondAnchor);
      const outside = set.doc.createElement('div');
      set.doc.body.append(outside);
      openPopover(set.pop, set.anchor);
      openPopover(second, secondAnchor);
      expect(set.pop.hidden).toBe(true);
      expect(second.hidden).toBe(false);
      return { ...set, second, outside };
    }

    test('Escape closes it', () => {
      const { win, doc, second } = twoOpen();
      doc.dispatchEvent(new win.KeyboardEvent('keydown', { key: 'Escape' }));
      expect(second.hidden).toBe(true);
    });

    test('an outside press closes it', () => {
      const { win, outside, second } = twoOpen();
      outside.dispatchEvent(new win.Event('pointerdown', { bubbles: true }));
      expect(second.hidden).toBe(true);
    });

    test('its window losing focus closes it', () => {
      const { win, second } = twoOpen();
      win.dispatchEvent(new win.Event('blur'));
      expect(second.hidden).toBe(true);
    });

    test('re-opening the same open popover keeps it too', () => {
      const { win, anchor, pop } = setUp();
      openPopover(pop, anchor);
      openPopover(pop, anchor);
      expect(pop.hidden).toBe(false);
      win.dispatchEvent(new win.Event('blur'));
      expect(pop.hidden).toBe(true);
    });
  });

  test('opens back in the hub when a hub opener asks', () => {
    const { anchor, pop } = setUp();
    openPopover(pop, anchor);
    closePopover(pop);
    const hubAnchor = document.createElement('button');
    document.body.append(hubAnchor);
    openPopover(pop, hubAnchor);
    expect(pop.ownerDocument).toBe(document);
    closePopover(pop);
  });
});

describe('a menu anchored in another document', () => {
  function setUp() {
    const { win, doc } = otherWindow();
    const anchorEl = doc.createElement('div');
    doc.body.append(anchorEl);
    return { win, doc, anchorEl };
  }

  /**
   * @param {any} anchorEl
   * @param {number} [at]
   */
  const open = (anchorEl, at = 10) =>
    openContextMenu({
      x: at,
      y: at,
      anchorEl,
      ariaLabel: 'Actions',
      items: [{ label: 'One', run: () => {} }],
    });

  /** @param {any} doc */
  const menuIn = (doc) => doc.querySelector('.osprey-context-menu');

  test('mounts in that document and takes focus there', () => {
    const { win, doc, anchorEl } = setUp();
    open(anchorEl);
    expect(menuIn(doc)?.parentElement).toBe(doc.body);
    expect(menuIn(document)).toBeNull();
    expect(doc.activeElement?.textContent).toContain('One');
    doc.dispatchEvent(new win.KeyboardEvent('keydown', { key: 'Escape' }));
  });

  test('clamps to that window, not the hub', () => {
    const { win, doc, anchorEl } = setUp();
    win.innerWidth = 100;
    win.innerHeight = 100;
    window.innerWidth = 2000;
    window.innerHeight = 2000;
    const rect = vi
      .spyOn(HTMLElement.prototype, 'getBoundingClientRect')
      .mockReturnValue(/** @type {DOMRect} */ ({ right: 150, bottom: 150, width: 60, height: 60 }));
    open(anchorEl, 90);
    expect(menuIn(doc).style.left).toBe('30px');
    expect(menuIn(doc).style.top).toBe('30px');
    rect.mockRestore();
    doc.dispatchEvent(new win.KeyboardEvent('keydown', { key: 'Escape' }));
  });

  test('its window losing focus closes it, and the hub losing focus does not', () => {
    const { win, doc, anchorEl } = setUp();
    open(anchorEl);
    window.dispatchEvent(new Event('blur'));
    expect(menuIn(doc)).not.toBeNull();
    win.dispatchEvent(new win.Event('blur'));
    expect(menuIn(doc)).toBeNull();
  });

  test('Escape in that document closes it', () => {
    const { win, doc, anchorEl } = setUp();
    open(anchorEl);
    doc.dispatchEvent(new win.KeyboardEvent('keydown', { key: 'Escape' }));
    expect(menuIn(doc)).toBeNull();
  });

  test('a press outside it closes it, on a node of that realm', () => {
    const { win, doc, anchorEl } = setUp();
    const outside = doc.createElement('div');
    doc.body.append(outside);
    open(anchorEl);
    outside.dispatchEvent(new win.Event('pointerdown', { bubbles: true }));
    expect(menuIn(doc)).toBeNull();
  });

  test('a scroll of that document closes it, and an inner scroller does not', () => {
    const { win, doc, anchorEl } = setUp();
    const inner = doc.createElement('div');
    doc.body.append(inner);
    open(anchorEl);
    inner.dispatchEvent(new win.Event('scroll'));
    expect(menuIn(doc)).not.toBeNull();
    doc.dispatchEvent(new win.Event('scroll'));
    expect(menuIn(doc)).toBeNull();
  });

  test('closing it removes every listener from that document and window', () => {
    const { win, doc, anchorEl } = setUp();
    const onDoc = countListeners(doc);
    const onWin = countListeners(win);
    open(anchorEl);
    expect(onDoc.size).toBe(3);
    expect([...onWin].filter(isDismissal)).toEqual(['blur', 'resize']);
    doc.dispatchEvent(new win.KeyboardEvent('keydown', { key: 'Escape' }));
    expect(onDoc.size).toBe(0);
    expect([...onWin].filter(isDismissal)).toEqual([]);
  });

  test('a menu in the hub replaces one in the window', () => {
    const { doc, anchorEl } = setUp();
    open(anchorEl);
    open(null);
    expect(menuIn(doc)).toBeNull();
    expect(menuIn(document)).not.toBeNull();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
  });
});

describe('a tip in another document', () => {
  const SHOW_DELAY = 450;

  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    hideTooltip();
    vi.useRealTimers();
  });

  function setUp() {
    const { win, doc } = otherWindow();
    const el = doc.createElement('button');
    el.setAttribute('data-tip', 'Reattach');
    doc.body.append(el);
    const stop = initTooltips(doc.body);
    const over = () => el.dispatchEvent(new win.Event('pointerover', { bubbles: true }));
    return { win, doc, el, stop, over };
  }

  /** @param {any} doc @returns {any} */
  const layerIn = (doc) => doc.querySelector('.osprey-tooltip');

  test('shows in a layer of that document, placed in that window', () => {
    const { win, doc, el, stop, over } = setUp();
    win.innerWidth = 200;
    win.innerHeight = 200;
    window.innerWidth = 4000;
    el.getBoundingClientRect = () => ({ top: 10, bottom: 30, left: 180, right: 195 });
    over();
    vi.advanceTimersByTime(SHOW_DELAY);
    const layer = layerIn(doc);
    expect(layer?.hidden).toBe(false);
    expect(layer?.textContent).toContain('Reattach');
    expect(layerIn(document)).toBeNull();
    expect(parseFloat(layer?.style.left)).toBeLessThan(200);
    stop();
  });

  test('Escape in that document hides it', () => {
    const { win, doc, stop, over } = setUp();
    over();
    vi.advanceTimersByTime(SHOW_DELAY);
    doc.dispatchEvent(new win.KeyboardEvent('keydown', { key: 'Escape' }));
    expect(layerIn(doc)?.hidden).toBe(true);
    stop();
  });

  test('a press in that document hides it', () => {
    const { win, doc, stop, over } = setUp();
    over();
    vi.advanceTimersByTime(SHOW_DELAY);
    doc.body.dispatchEvent(new win.Event('pointerdown', { bubbles: true }));
    expect(layerIn(doc)?.hidden).toBe(true);
    stop();
  });

  test('stopping the last watch removes the listeners from that document', () => {
    const { doc } = otherWindow();
    const live = countListeners(doc);
    const stopOne = initTooltips(doc.body);
    const stopTwo = initTooltips(doc.body);
    expect(live.size).toBe(2);
    stopOne();
    expect(live.size).toBe(2);
    stopTwo();
    expect(live.size).toBe(0);
  });
});
