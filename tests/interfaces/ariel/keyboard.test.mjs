// @ts-check
/**
 * Unit tests for the ARIEL keyboard shortcuts (keyboard.js): j/k through the
 * rendered rows, Enter opens the reader, `/` is left to the host,
 * `[` toggles the rail, `c` opens a new entry and `r` a follow-up of the
 * selection, and `Esc` closes exactly one layer — skipping events that were
 * defaultPrevented or arrived while a popover or drawer was open.
 *
 *   npx vitest run tests/interfaces/ariel/keyboard.test.mjs
 *
 * Popovers and drawers are plain
 * elements with the design system's open markers, and a document-level
 * bubble listener stands in for the design system closing them on Escape.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

/** @typedef {typeof import('../../../src/osprey/interfaces/ariel/static/js/store.js')} StoreMod */
/** @typedef {typeof import('../../../src/osprey/interfaces/ariel/static/js/keyboard.js')} KeyboardMod */

/** @type {StoreMod} */ let store;
/** @type {KeyboardMod} */ let keyboard;
/** @type {{destroy: () => void} | null} */ let api = null;
/** @type {((e: KeyboardEvent) => void) | null} */ let designSystemEsc = null;

function mountFixture() {
  document.body.innerHTML = `
    <div class="app-container">
      <div id="ariel-rail"></div>
      <div id="ariel-list">
        <input id="search" class="osprey-search" />
        <div class="list-body">
          <div class="osprey-list-row" data-entry-id="e1"></div>
          <div class="osprey-list-row" data-entry-id="e2"></div>
          <div class="osprey-list-row" data-entry-id="e3"></div>
        </div>
        <div id="ariel-compose"><input id="entry-subject" /><textarea id="entry-body"></textarea><select id="sel"></select><div id="rich" contenteditable="true"><span id="rich-inner"></span></div></div>
      </div>
      <div id="ariel-reader"><button id="back">Back</button></div>
    </div>
    <div class="osprey-popover" id="gear-pop" hidden></div>
    <osprey-drawer id="status-drawer"></osprey-drawer>
  `;
  // Stand-in for the design system: popovers and drawers close themselves on
  // Escape from a bubble-phase document listener, without preventDefault.
  designSystemEsc = (e) => {
    if (e.key !== 'Escape') return;
    document.querySelectorAll('.osprey-popover').forEach((p) => {
      /** @type {HTMLElement} */ (p).hidden = true;
    });
    document.querySelector('osprey-drawer')?.removeAttribute('open');
  };
  document.addEventListener('keydown', designSystemEsc);
}

/**
 * @param {string} key
 * @param {EventTarget} [target]
 * @param {KeyboardEventInit} [init]
 * @returns {KeyboardEvent}
 */
function press(key, target = document.body, init = {}) {
  const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
  target.dispatchEvent(event);
  return event;
}

async function boot() {
  vi.resetModules();
  store = await import('../../../src/osprey/interfaces/ariel/static/js/store.js');
  keyboard = await import('../../../src/osprey/interfaces/ariel/static/js/keyboard.js');
  api = keyboard.initKeyboard(document);
}

beforeEach(async () => {
  mountFixture();
  await boot();
});

afterEach(() => {
  api?.destroy();
  api = null;
  if (designSystemEsc) document.removeEventListener('keydown', designSystemEsc);
  designSystemEsc = null;
  document.body.innerHTML = '';
});

describe('j / k', () => {
  test('j with nothing selected selects the first rendered row', () => {
    press('j');
    expect(store.getState().sel).toBe('e1');
  });

  test('j and k walk the rendered rows in order and stop at the ends', () => {
    press('j');
    press('j');
    press('j');
    expect(store.getState().sel).toBe('e3');
    press('j');
    expect(store.getState().sel).toBe('e3');
    press('k');
    expect(store.getState().sel).toBe('e2');
    press('k');
    press('k');
    expect(store.getState().sel).toBe('e1');
  });

  test.each(['j', 'k'])(
    '%s while an attachment is open clears att from state and hash',
    async (key) => {
      const router = await import('../../../src/osprey/interfaces/ariel/static/js/router.js');
      store.setState({ sel: 'e2', att: 'a1' });
      press(key);
      expect(store.getState().sel).not.toBe('e2');
      expect(store.getState().att).toBeNull();
      expect(router.serializeState(store.getState())).not.toContain('att=');
    }
  );

  test('j/k set only sel, never narrowView', () => {
    store.setState({ stacked: true });
    press('j');
    expect(store.getState().narrowView).toBe('list');
  });

  test('a selection that is no longer rendered restarts at the first row', () => {
    store.setState({ sel: 'gone' });
    press('k');
    expect(store.getState().sel).toBe('e1');
  });

  test('no rows: j is a no-op', () => {
    document.querySelectorAll('[data-entry-id]').forEach((r) => r.remove());
    const e = press('j');
    expect(store.getState().sel).toBeNull();
    expect(e.defaultPrevented).toBe(false);
  });

  test('ignored while typing in an input, textarea or select', () => {
    for (const id of ['search', 'entry-body', 'sel']) {
      press('j', /** @type {HTMLElement} */ (document.getElementById(id)));
    }
    expect(store.getState().sel).toBeNull();
  });

  test('ignored with a modifier held', () => {
    press('j', document.body, { ctrlKey: true });
    press('j', document.body, { metaKey: true });
    expect(store.getState().sel).toBeNull();
  });
});

describe('Enter', () => {
  test('opens the selection in the reader', () => {
    store.setState({ sel: 'e2' });
    press('Enter');
    expect(store.getState()).toMatchObject({ sel: 'e2', narrowView: 'reader' });
  });

  test('does nothing without a selection', () => {
    press('Enter');
    expect(store.getState().narrowView).toBe('list');
  });

  test('is left to a focused button', () => {
    store.setState({ sel: 'e2' });
    press('Enter', /** @type {HTMLElement} */ (document.getElementById('back')));
    expect(store.getState().narrowView).toBe('list');
  });

  test('ignored while typing', () => {
    store.setState({ sel: 'e2' });
    press('Enter', /** @type {HTMLElement} */ (document.getElementById('search')));
    expect(store.getState().narrowView).toBe('list');
  });
});

describe('/', () => {
  test("is unbound: it stays the host's", () => {
    const e = press('/');
    expect(e.defaultPrevented).toBe(false);
  });
});

describe('[', () => {
  test('toggles railCollapsed both ways', () => {
    press('[');
    expect(store.getState().railCollapsed).toBe(true);
    press('[');
    expect(store.getState().railCollapsed).toBe(false);
  });

  test('ignored while typing', () => {
    press('[', /** @type {HTMLElement} */ (document.getElementById('entry-body')));
    expect(store.getState().railCollapsed).toBe(false);
  });
});

describe('c / r', () => {
  test('c opens a new entry and swallows the key', () => {
    const e = press('c');
    expect(store.getState().compose).toEqual({ draft: null });
    expect(e.defaultPrevented).toBe(true);
  });

  test('c with a selection still opens a plain new entry, not a follow-up', () => {
    store.setState({ sel: 'e2' });
    press('c');
    expect(store.getState().compose).toEqual({ draft: null });
    expect(store.getState().sel).toBe('e2');
  });

  test('r opens a follow-up of the selected entry', () => {
    store.setState({ sel: 'e2' });
    const e = press('r');
    expect(store.getState().compose).toEqual({ draft: null, followsUp: 'e2' });
    expect(e.defaultPrevented).toBe(true);
  });

  test('r with no selection does nothing', () => {
    const e = press('r');
    expect(store.getState().compose).toBeNull();
    expect(e.defaultPrevented).toBe(false);
  });

  test.each(['c', 'r'])('%s is ignored in an input, textarea, select or contenteditable', (key) => {
    store.setState({ sel: 'e2' });
    for (const id of ['search', 'entry-subject', 'entry-body', 'sel', 'rich', 'rich-inner']) {
      const e = press(key, /** @type {HTMLElement} */ (document.getElementById(id)));
      expect(e.defaultPrevented).toBe(false);
    }
    expect(store.getState().compose).toBeNull();
  });

  test.each(['c', 'r'])('%s is ignored with a modifier held', (key) => {
    store.setState({ sel: 'e2' });
    press(key, document.body, { ctrlKey: true });
    press(key, document.body, { metaKey: true });
    press(key, document.body, { altKey: true });
    expect(store.getState().compose).toBeNull();
  });

  test('upper-case C / R are not bound', () => {
    store.setState({ sel: 'e2' });
    press('C', document.body, { shiftKey: true });
    press('R', document.body, { shiftKey: true });
    expect(store.getState().compose).toBeNull();
  });

  test('r on a lightbox-open selection opens the follow-up for that entry', () => {
    store.setState({ sel: 'e1', att: 'a1' });
    press('r');
    expect(store.getState().compose).toEqual({ draft: null, followsUp: 'e1' });
  });
});

describe('Esc', () => {
  test('closes the lightbox', () => {
    store.setState({ sel: 'e1', att: 'a1' });
    const e = press('Escape');
    expect(store.getState()).toMatchObject({ sel: 'e1', att: null });
    expect(e.defaultPrevented).toBe(true);
  });

  test('closes compose', () => {
    store.setState({ compose: { draft: null } });
    press('Escape');
    expect(store.getState().compose).toBeNull();
  });

  test('else goes Back in the stacked layout', () => {
    store.setState({ stacked: true, narrowView: 'reader', sel: 'e1' });
    press('Escape');
    expect(store.getState()).toMatchObject({ narrowView: 'list', sel: 'e1' });
  });

  test('wide layout with nothing open: nothing to close', () => {
    store.setState({ narrowView: 'reader', sel: 'e1' });
    const e = press('Escape');
    expect(store.getState().narrowView).toBe('reader');
    expect(e.defaultPrevented).toBe(false);
  });

  test('still acts while typing (closes compose from its textarea)', () => {
    store.setState({ compose: { draft: null } });
    press('Escape', /** @type {HTMLElement} */ (document.getElementById('entry-body')));
    expect(store.getState().compose).toBeNull();
  });

  test('skips a defaultPrevented event', () => {
    store.setState({ compose: { draft: null } });
    const ta = /** @type {HTMLElement} */ (document.getElementById('entry-body'));
    ta.addEventListener('keydown', (e) => e.preventDefault(), { once: true });
    press('Escape', ta);
    expect(store.getState().compose).toEqual({ draft: null });
  });

  test('skips an Esc that arrived while a popover was open, even though it closed first', () => {
    store.setState({ att: 'a1' });
    const pop = /** @type {HTMLElement} */ (document.getElementById('gear-pop'));
    pop.hidden = false;
    press('Escape');
    expect(pop.hidden).toBe(true);
    expect(store.getState().att).toBe('a1');
  });

  test('skips an Esc that arrived while a drawer was open', () => {
    store.setState({ compose: { draft: null } });
    document.getElementById('status-drawer')?.setAttribute('open', '');
    press('Escape');
    expect(document.getElementById('status-drawer')?.hasAttribute('open')).toBe(false);
    expect(store.getState().compose).toEqual({ draft: null });
  });

  test('from a sheet field: lightbox, then sheet close (compose:null), then stacked Back', () => {
    store.setState({ sel: 'e1', stacked: true, narrowView: 'reader' });
    press('r');
    store.setState({ att: 'a1' });
    const field = /** @type {HTMLElement} */ (document.getElementById('entry-subject'));

    const e1 = press('Escape', field);
    expect(e1.defaultPrevented).toBe(true);
    expect(store.getState()).toMatchObject({
      att: null,
      compose: { draft: null, followsUp: 'e1' },
      narrowView: 'reader',
    });

    const e2 = press('Escape', field);
    expect(e2.defaultPrevented).toBe(true);
    expect(store.getState()).toMatchObject({ compose: null, narrowView: 'reader', sel: 'e1' });

    press('Escape', field);
    expect(store.getState()).toMatchObject({ narrowView: 'list', sel: 'e1' });
  });

  test('closes the sheet from a contenteditable field', () => {
    press('c');
    press('Escape', /** @type {HTMLElement} */ (document.getElementById('rich-inner')));
    expect(store.getState().compose).toBeNull();
  });

  test('one Esc closes one layer', () => {
    store.setState({
      sel: 'e1',
      att: 'a1',
      compose: { draft: null },
      stacked: true,
      narrowView: 'reader',
    });
    const pop = /** @type {HTMLElement} */ (document.getElementById('gear-pop'));
    const drawer = /** @type {HTMLElement} */ (document.getElementById('status-drawer'));
    pop.hidden = false;
    drawer.setAttribute('open', '');

    // 1: the popover and the drawer close (design system); the store is untouched.
    press('Escape');
    expect(pop.hidden).toBe(true);
    expect(store.getState()).toMatchObject({
      att: 'a1',
      compose: { draft: null },
      narrowView: 'reader',
    });

    // 2: the lightbox.
    press('Escape');
    expect(store.getState()).toMatchObject({
      att: null,
      compose: { draft: null },
      narrowView: 'reader',
    });

    // 3: compose.
    press('Escape');
    expect(store.getState()).toMatchObject({ att: null, compose: null, narrowView: 'reader' });

    // 4: Back.
    press('Escape');
    expect(store.getState()).toMatchObject({ narrowView: 'list', sel: 'e1' });

    // 5: nothing left.
    const e = press('Escape');
    expect(e.defaultPrevented).toBe(false);
    expect(store.getState()).toMatchObject({ narrowView: 'list', sel: 'e1' });
  });
});

describe('e', () => {
  test('is left to the reader seam: not defaultPrevented, no store write', () => {
    store.setState({ sel: 'e2' });
    const before = { ...store.getState() };
    const e = press('e');
    expect(e.defaultPrevented).toBe(false);
    expect(store.getState()).toEqual(before);
  });
});

describe('scope', () => {
  test('keydowns from another document are not handled', () => {
    const other = document.implementation.createHTMLDocument('host');
    const el = other.createElement('div');
    other.body.append(el);
    // Re-dispatch on this window with a foreign target is not possible in the
    // DOM, so assert the handler only listens on the panel's own window.
    el.dispatchEvent(new KeyboardEvent('keydown', { key: 'j', bubbles: true }));
    expect(store.getState().sel).toBeNull();
  });

  test('destroy removes every shortcut', () => {
    api?.destroy();
    api = null;
    press('j');
    press('[');
    store.setState({ sel: 'e2' });
    press('r');
    expect(store.getState().compose).toBeNull();
    press('c');
    expect(store.getState().compose).toBeNull();
    store.setState({ compose: { draft: null } });
    press('Escape');
    expect(store.getState()).toMatchObject({ sel: 'e2', railCollapsed: false });
    expect(store.getState().compose).toEqual({ draft: null });
  });
});
