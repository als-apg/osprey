// @ts-check
/**
 * Tab shows or hides the sidebar labels (sidebar-tab-key.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-tab-key.test.mjs
 *
 * A plain Tab flips the sidebar unless focus is somewhere Tab already means
 * something (typing, the terminal, an open layer), the key carries a
 * modifier, another handler took it, or the operator turned it off.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { byId } from '../_support/dom.mjs';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';
const { initSidebarTabKey, setTabToggle, tabToggleOn, TAB_TOGGLE_KEY } = await import(
  `${JS}/sidebar-tab-key.js`
);

const flip = vi.fn();
/** @type {() => void} */
let stop;

/**
 * Press a key with focus on `el` (the body when null).
 * @param {HTMLElement | null} el
 * @param {KeyboardEventInit} [init]
 * @returns {KeyboardEvent}
 */
function press(el, init = {}) {
  el?.focus();
  const event = new KeyboardEvent('keydown', {
    key: 'Tab',
    bubbles: true,
    cancelable: true,
    ...init,
  });
  (el ?? document.body).dispatchEvent(event);
  return event;
}

beforeEach(() => {
  localStorage.clear();
  flip.mockClear();
  document.body.innerHTML = `
    <button id="row">Knowledge</button>
    <textarea id="message"></textarea>
    <div class="terminal xterm"><textarea class="xterm-helper-textarea" id="xterm"></textarea></div>
    <input id="palette">
    <div contenteditable="true" id="editable"></div>
    <div class="osprey-popover" role="menu"><button id="menu-row">Light</button></div>
    <osprey-drawer><button id="drawer-tab">Behavior</button></osprey-drawer>`;
  stop = initSidebarTabKey(flip);
});

afterEach(() => {
  stop();
  document.body.replaceChildren();
});

describe('a plain Tab flips the sidebar', () => {
  test('from the page and from a hub control, and keeps focus where it was', () => {
    expect(press(null).defaultPrevented).toBe(true);
    expect(press(byId('row')).defaultPrevented).toBe(true);
    expect(flip).toHaveBeenCalledTimes(2);
    expect(document.activeElement).toBe(byId('row'));
  });

  test('is on before the operator has chosen', () => {
    expect(tabToggleOn()).toBe(true);
  });
});

describe('in a panel frame on the same origin', () => {
  test('a Tab from the panel empty space flips; a Tab in its field does not', () => {
    stop();
    const frame = document.createElement('iframe');
    document.body.append(frame);
    const doc = /** @type {Document} */ (frame.contentDocument);
    doc.body.innerHTML = '<input id="filter">';
    stop = initSidebarTabKey(flip);

    const tabIn = () =>
      doc.body.dispatchEvent(
        new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true })
      );
    tabIn();
    expect(flip).toHaveBeenCalledTimes(1);

    /** @type {HTMLElement} */ (doc.getElementById('filter')).focus();
    tabIn();
    expect(flip).toHaveBeenCalledTimes(1);
  });
});

describe('Tab keeps its own meaning', () => {
  test.each(['message', 'xterm', 'palette', 'editable', 'menu-row', 'drawer-tab'])(
    'with focus in #%s',
    (id) => {
      expect(press(byId(id)).defaultPrevented).toBe(false);
      expect(flip).not.toHaveBeenCalled();
    }
  );

  test('with a modifier: Shift+Tab still moves focus backwards', () => {
    for (const mod of ['shiftKey', 'altKey', 'ctrlKey', 'metaKey'])
      press(byId('row'), { [mod]: true });
    expect(flip).not.toHaveBeenCalled();
  });

  test('when another handler already took the key', () => {
    byId('row').addEventListener('keydown', (e) => e.preventDefault());
    press(byId('row'));
    expect(flip).not.toHaveBeenCalled();
  });

  test('once the operator turns it off, and again after turning it on', () => {
    setTabToggle(false);
    expect(localStorage.getItem(TAB_TOGGLE_KEY)).toBe('off');
    press(byId('row'));
    expect(flip).not.toHaveBeenCalled();
    setTabToggle(true);
    press(byId('row'));
    expect(flip).toHaveBeenCalledTimes(1);
  });
});
