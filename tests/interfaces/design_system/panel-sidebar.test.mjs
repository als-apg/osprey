// @ts-check
/**
 * Unit tests for the panel sidebar (design_system/static/js/panel-sidebar.js):
 *   npx vitest run tests/interfaces/design_system/panel-sidebar.test.mjs
 *
 * The rule under test is what gets stored: the operator's own gestures (the
 * toggle, the handle) are stored, and every automatic open or close (a
 * default that changes with the view, the narrow rule, calls from code) is
 * not. happy-dom does no layout, so the panel root's width comes from a
 * stubbed ResizeObserver (and `clientWidth` for the first measurement).
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { initPanelSidebar } from '/design-system/js/panel-sidebar.js';

const KEY = 'osprey-test-panel-sidebar';

/** @type {Set<{cb: ResizeObserverCallback, el: Element | null}>} */
const observers = new Set();

class FakeResizeObserver {
  /** @param {ResizeObserverCallback} cb */
  constructor(cb) {
    this.cb = cb;
    /** @type {Element | null} */
    this.el = null;
    observers.add(this);
  }

  /** @param {Element} el */
  observe(el) {
    this.el = el;
  }

  disconnect() {
    observers.delete(this);
  }
}

/**
 * Report a new content width for the panel root to every live observer.
 * @param {number} width
 */
function resizeRoot(width) {
  for (const observer of observers) {
    const entry = /** @type {ResizeObserverEntry} */ (
      /** @type {unknown} */ ({ target: observer.el, contentRect: { width } })
    );
    observer.cb([entry], /** @type {ResizeObserver} */ (/** @type {unknown} */ (observer)));
  }
}

/** @param {string} id @returns {HTMLElement} */
const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));

function render() {
  document.body.innerHTML = `
    <div id="root" data-panel-root>
      <div class="osprey-panel-toolbar">
        <button id="toggle" class="osprey-icon-btn" type="button"></button>
      </div>
      <div id="row">
        <aside id="side" class="osprey-panel-sidebar" data-open="false">
          <button id="inside" type="button">row</button>
        </aside>
        <div id="handle" class="osprey-splitter" tabindex="0"></div>
        <main id="content"><button id="outside" type="button">content</button></main>
      </div>
    </div>
  `;
}

/** @param {Partial<import('/design-system/js/panel-sidebar.js').PanelSidebarOptions>} [overrides] */
function setup(overrides = {}) {
  return initPanelSidebar({
    sidebar: $('side'),
    handle: $('handle'),
    toggle: $('toggle'),
    storageKey: KEY,
    ...overrides,
  });
}

/** The stored record, parsed, or null. */
function stored() {
  const raw = localStorage.getItem(KEY);
  return raw === null ? null : JSON.parse(raw);
}

/** @param {string} key @param {EventTarget} [target] */
function press(key, target = document.body) {
  target.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
}

/** @param {EventTarget} target */
function pointerDown(target) {
  target.dispatchEvent(new Event('pointerdown', { bubbles: true }));
}

/** @type {import('/design-system/js/panel-sidebar.js').PanelSidebarApi | null} */
let api = null;

beforeEach(() => {
  localStorage.clear();
  observers.clear();
  vi.stubGlobal('ResizeObserver', FakeResizeObserver);
  render();
});

afterEach(() => {
  api?.destroy();
  api = null;
  vi.unstubAllGlobals();
});

describe('initPanelSidebar', () => {
  test('opens at the default width and mirrors the state', () => {
    api = setup();
    expect(api.isOpen()).toBe(true);
    expect($('side').style.width).toBe('260px');
    expect($('side').getAttribute('data-open')).toBe('true');
    expect($('root').getAttribute('data-open')).toBe('true');
    expect($('toggle').getAttribute('aria-expanded')).toBe('true');
    expect($('toggle').getAttribute('aria-controls')).toBe('side');
    expect($('toggle').querySelector('svg.osprey-ic')).not.toBeNull();

    $('toggle').click();
    expect($('root').getAttribute('data-open')).toBe('false');
    expect($('toggle').getAttribute('aria-expanded')).toBe('false');
    expect($('toggle').title).not.toBe('');
  });

  test("the operator's toggle is stored and survives a reload", () => {
    api = setup();
    $('toggle').click();
    expect(stored()).toEqual({ collapsed: true });

    api.destroy();
    render();
    api = setup();
    expect(api.isOpen()).toBe(false);

    $('toggle').click();
    expect(api.isOpen()).toBe(true);
    expect(stored().collapsed).toBe(false);
  });

  test('a change of default opens and closes without storing', () => {
    let view = 'overview';
    api = setup({ defaultOpen: () => view === 'page' });
    expect(api.isOpen()).toBe(false);

    view = 'page';
    api.applyDefault();
    expect(api.isOpen()).toBe(true);

    api.setDefaultOpen(false);
    expect(api.isOpen()).toBe(false);
    expect(stored()).toBeNull();
  });

  test('open(), close() and toggle() from code are not stored', () => {
    api = setup();
    api.close();
    api.toggle();
    api.close();
    expect(api.isOpen()).toBe(false);
    expect(stored()).toBeNull();
  });

  test('a stored hide holds across a view change, and wins over an open default', () => {
    let view = 'page';
    api = setup({ defaultOpen: () => view === 'page' });
    $('toggle').click();
    expect(stored().collapsed).toBe(true);

    view = 'overview';
    api.applyDefault();
    view = 'page';
    api.applyDefault();
    expect(api.isOpen()).toBe(false);
  });

  test('a toggle where the default is closed holds only until the next automatic change', () => {
    let view = 'overview';
    api = setup({ defaultOpen: () => view === 'page' });
    $('toggle').click();
    expect(api.isOpen()).toBe(true);
    expect(stored()?.collapsed).toBeUndefined();

    view = 'page';
    api.applyDefault();
    view = 'overview';
    api.applyDefault();
    expect(api.isOpen()).toBe(false);
  });

  test('entering a narrow panel closes without storing; leaving it applies the rule again', () => {
    api = setup();
    resizeRoot(600);
    expect(api.isOpen()).toBe(false);
    expect($('side').hasAttribute('data-overlay')).toBe(true);
    expect(stored()).toBeNull();

    resizeRoot(900);
    expect(api.isOpen()).toBe(true);
    expect($('side').hasAttribute('data-overlay')).toBe(false);
  });

  test('starts closed on a narrow panel, whatever is stored', () => {
    localStorage.setItem(KEY, JSON.stringify({ size: 300, collapsed: false }));
    Object.defineProperty($('root'), 'clientWidth', { value: 600, configurable: true });
    api = setup();
    expect(api.isOpen()).toBe(false);
    expect($('side').hasAttribute('data-overlay')).toBe(true);
  });

  test('the narrow overlay closes on Escape and on a press outside, not on one inside', () => {
    api = setup();
    resizeRoot(600);
    const scrim = /** @type {HTMLElement} */ (
      document.querySelector('.osprey-panel-sidebar-scrim')
    );

    $('toggle').click();
    expect(api.isOpen()).toBe(true);
    expect(scrim.hidden).toBe(false);

    $('inside').focus();
    press('Escape', $('inside'));
    expect(api.isOpen()).toBe(false);
    expect(scrim.hidden).toBe(true);
    expect(document.activeElement).toBe($('toggle'));

    $('toggle').click();
    pointerDown($('inside'));
    expect(api.isOpen()).toBe(true);
    pointerDown($('outside'));
    expect(api.isOpen()).toBe(false);

    $('toggle').click();
    pointerDown(scrim);
    expect(api.isOpen()).toBe(false);
    expect(stored()).toBeNull();
  });

  test('the width clamps to min and max and the default width is an option', () => {
    localStorage.setItem(KEY, JSON.stringify({ size: 900 }));
    api = setup();
    expect($('side').style.width).toBe('480px');
    api.destroy();

    render();
    localStorage.setItem(KEY, JSON.stringify({ size: 40 }));
    api = setup();
    expect($('side').style.width).toBe('180px');
    api.destroy();

    render();
    localStorage.clear();
    api = setup({ width: 300, min: 200, max: 400 });
    expect($('side').style.width).toBe('300px');
  });

  test('a nudge on the handle stores the width; resetWidth forgets it', () => {
    api = setup();
    press('ArrowRight', $('handle'));
    expect($('side').style.width).toBe('276px');
    expect(stored()).toEqual({ size: 276 });

    api.resetWidth();
    expect($('side').style.width).toBe('260px');
    expect(stored()).toBeNull();
  });

  test('destroy removes the listeners, the observer and the scrim', () => {
    api = setup();
    resizeRoot(600);
    $('toggle').click();
    expect(api.isOpen()).toBe(true);

    api.destroy();
    expect(observers.size).toBe(0);
    expect(document.querySelector('.osprey-panel-sidebar-scrim')).toBeNull();

    press('Escape');
    pointerDown($('outside'));
    $('toggle').click();
    expect(api.isOpen()).toBe(true);
    expect(stored()).toBeNull();
    api = null;
  });

  test('a narrowQuery that is not "(max-width: <n>px)" fails at the call site', () => {
    expect(() => setup({ narrowQuery: '(min-width: 40em)' })).toThrow(/narrowQuery/);
  });
});
