/**
 * Unit tests for the shared focus trap (design_system/static/js/focus-trap.js).
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/design_system/focus-trap.test.mjs
 *
 * happy-dom has no layout engine, so `offsetParent` is null for every element
 * and the trap's visibility filter would otherwise reject the whole fixture.
 * `setVisible()` below defines an own `offsetParent` data property per element,
 * which is exactly the signal the filter reads.
 */

import { test, expect, describe, beforeEach, afterEach } from 'vitest';

import {
  focusableElements,
  installFocusTrap,
  removeFocusTrap,
} from '../../../src/osprey/interfaces/design_system/static/js/focus-trap.js';
import { qs } from '../_support/dom.mjs';

/**
 * Give `el` the `offsetParent` the trap's visibility filter looks at.
 *
 * @param {HTMLElement} el
 * @param {boolean} visible
 */
function setVisible(el, visible) {
  Object.defineProperty(el, 'offsetParent', {
    value: visible ? document.body : null,
    configurable: true,
  });
}

/**
 * Mount `html` inside a root div on <body>, marking every element visible.
 *
 * @param {string} html
 * @returns {HTMLElement}
 */
function mountRoot(html) {
  const root = document.createElement('div');
  root.innerHTML = html;
  document.body.appendChild(root);
  for (const node of Array.from(root.querySelectorAll('*'))) {
    if (node instanceof HTMLElement) setVisible(node, true);
  }
  return root;
}

/**
 * Dispatch a bubbling, cancelable Tab keydown from `target`.
 *
 * @param {HTMLElement} target
 * @param {{ shift?: boolean, key?: string }} [options]
 * @returns {KeyboardEvent}
 */
function pressTab(target, options = {}) {
  const event = new KeyboardEvent('keydown', {
    key: options.key ?? 'Tab',
    shiftKey: options.shift ?? false,
    bubbles: true,
    cancelable: true,
  });
  target.dispatchEvent(event);
  return event;
}

const THREE_BUTTONS = `
  <button id="first" type="button">First</button>
  <button id="middle" type="button">Middle</button>
  <button id="last" type="button">Last</button>
`;

/** @type {HTMLElement} */
let outside;

beforeEach(() => {
  document.body.replaceChildren();
  outside = document.createElement('button');
  outside.id = 'outside';
  document.body.appendChild(outside);
  setVisible(outside, true);
});

afterEach(() => {
  document.body.replaceChildren();
});

describe('focusableElements', () => {
  test('returns rendered focusables in document order', () => {
    const root = mountRoot(THREE_BUTTONS);

    expect(focusableElements(root).map((el) => el.id)).toEqual(['first', 'middle', 'last']);
  });

  test('skips elements with no offsetParent', () => {
    const root = mountRoot(THREE_BUTTONS);
    setVisible(qs(root, '#middle'), false);

    expect(focusableElements(root).map((el) => el.id)).toEqual(['first', 'last']);
  });

  test('skips disabled controls and tabindex="-1"', () => {
    const root = mountRoot(`
      <button id="first" type="button">First</button>
      <button id="off" type="button" disabled>Disabled</button>
      <input id="offinput" disabled>
      <div id="programmatic" tabindex="-1">Not tabbable</div>
      <a id="link" href="#somewhere">Link</a>
    `);

    expect(focusableElements(root).map((el) => el.id)).toEqual(['first', 'link']);
  });

  test('includes inputs, selects, textareas, positive tabindex and contenteditable', () => {
    const root = mountRoot(`
      <input id="text">
      <select id="select"></select>
      <textarea id="area"></textarea>
      <div id="tabbable" tabindex="0">Tabbable</div>
      <div id="editable" contenteditable="true">Editable</div>
    `);

    expect(focusableElements(root).map((el) => el.id)).toEqual([
      'text',
      'select',
      'area',
      'tabbable',
      'editable',
    ]);
  });

  test('returns an empty list for a root with no focusables', () => {
    const root = mountRoot('<p id="prose">Nothing to focus</p>');

    expect(focusableElements(root)).toEqual([]);
  });
});

describe('installFocusTrap wraparound', () => {
  test('Tab on the last focusable wraps to the first', () => {
    const root = mountRoot(THREE_BUTTONS);
    installFocusTrap(root);
    const last = qs(root, '#last');
    last.focus();

    const event = pressTab(last);

    expect(event.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(qs(root, '#first'));
  });

  test('Shift+Tab on the first focusable wraps to the last', () => {
    const root = mountRoot(THREE_BUTTONS);
    installFocusTrap(root);
    const first = qs(root, '#first');
    first.focus();

    const event = pressTab(first, { shift: true });

    expect(event.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(qs(root, '#last'));
  });

  test('Tab in the middle of the list is left to the browser', () => {
    const root = mountRoot(THREE_BUTTONS);
    installFocusTrap(root);
    const middle = qs(root, '#middle');
    middle.focus();

    const event = pressTab(middle);

    expect(event.defaultPrevented).toBe(false);
    expect(document.activeElement).toBe(middle);
  });

  test('a hidden last control is skipped, so the last visible one wraps', () => {
    const root = mountRoot(THREE_BUTTONS);
    setVisible(qs(root, '#last'), false);
    installFocusTrap(root);
    const middle = qs(root, '#middle');
    middle.focus();

    const event = pressTab(middle);

    expect(event.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(qs(root, '#first'));
  });

  test('a hidden first control is skipped, so the first visible one wraps back', () => {
    const root = mountRoot(THREE_BUTTONS);
    setVisible(qs(root, '#first'), false);
    installFocusTrap(root);
    const middle = qs(root, '#middle');
    middle.focus();

    const event = pressTab(middle, { shift: true });

    expect(event.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(qs(root, '#last'));
  });

  test('Tab with focus outside the root pulls focus to the first focusable', () => {
    const root = mountRoot(THREE_BUTTONS);
    installFocusTrap(root);
    outside.focus();

    const event = pressTab(root);

    expect(event.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(qs(root, '#first'));
  });

  test('Shift+Tab with focus outside the root pulls focus to the last focusable', () => {
    const root = mountRoot(THREE_BUTTONS);
    installFocusTrap(root);
    outside.focus();

    const event = pressTab(root, { shift: true });

    expect(event.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(qs(root, '#last'));
  });

  test('Tab is swallowed when the root has no focusables at all', () => {
    const root = mountRoot('<p id="prose">Nothing to focus</p>');
    installFocusTrap(root);

    const event = pressTab(root);

    expect(event.defaultPrevented).toBe(true);
  });

  test('keys other than Tab pass straight through', () => {
    const root = mountRoot(THREE_BUTTONS);
    installFocusTrap(root);
    const last = qs(root, '#last');
    last.focus();

    const event = pressTab(last, { key: 'Enter' });

    expect(event.defaultPrevented).toBe(false);
    expect(document.activeElement).toBe(last);
  });

  test('keydowns from outside the root never reach the trap', () => {
    const root = mountRoot(THREE_BUTTONS);
    installFocusTrap(root);
    outside.focus();

    const event = pressTab(outside);

    expect(event.defaultPrevented).toBe(false);
    expect(document.activeElement).toBe(outside);
  });
});

describe('install/remove idempotency', () => {
  test('a second install registers no second listener', () => {
    const root = mountRoot(THREE_BUTTONS);
    installFocusTrap(root);
    installFocusTrap(root);

    // A single removal must therefore disarm the trap completely; if the second
    // install had added its own listener, this Tab would still wrap.
    removeFocusTrap(root);
    const last = qs(root, '#last');
    last.focus();
    const event = pressTab(last);

    expect(event.defaultPrevented).toBe(false);
    expect(document.activeElement).toBe(last);
  });

  test('removing twice is a no-op', () => {
    const root = mountRoot(THREE_BUTTONS);
    installFocusTrap(root);
    removeFocusTrap(root);

    expect(() => removeFocusTrap(root)).not.toThrow();
  });

  test('removing a trap that was never installed is a no-op', () => {
    const root = mountRoot(THREE_BUTTONS);

    expect(() => removeFocusTrap(root)).not.toThrow();
  });

  test('a trap can be reinstalled after removal', () => {
    const root = mountRoot(THREE_BUTTONS);
    installFocusTrap(root);
    removeFocusTrap(root);
    installFocusTrap(root);
    const last = qs(root, '#last');
    last.focus();

    const event = pressTab(last);

    expect(event.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(qs(root, '#first'));
  });

  test('two roots trap independently', () => {
    const one = mountRoot('<button id="one-a" type="button">A</button>');
    const two = mountRoot('<button id="two-a" type="button">A</button>');
    installFocusTrap(one);
    installFocusTrap(two);
    removeFocusTrap(one);

    const onlyInTwo = qs(two, '#two-a');
    onlyInTwo.focus();
    const trapped = pressTab(onlyInTwo);
    const freed = pressTab(qs(one, '#one-a'));

    expect(trapped.defaultPrevented).toBe(true);
    expect(freed.defaultPrevented).toBe(false);
  });
});
