/**
 * Unit tests for the design-system agent-activity flash helper
 * (static/js/highlight.js).
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/design_system/highlight.test.mjs
 *
 * happy-dom does not run CSS animations, so `animationend` is dispatched
 * synthetically; the helper tolerates events without an `animationName`
 * (and ignores mismatching names / bubbled child targets).
 */

import { test, expect, describe, beforeEach } from 'vitest';

import { flashElement } from '../../../src/osprey/interfaces/design_system/static/js/highlight.js';

/** @type {HTMLElement} */
let el;

beforeEach(() => {
  document.body.replaceChildren();
  el = document.createElement('div');
  document.body.appendChild(el);
});

describe('flashElement', () => {
  test('adds the agent-flash class', () => {
    flashElement(el);

    expect(el.classList.contains('agent-flash')).toBe(true);
  });

  test('reflow-restart: a second call while still flashing leaves the class applied', () => {
    flashElement(el);
    flashElement(el);

    // The remove -> offsetWidth -> add cycle must land back on "applied";
    // a naive second `add` no-op and the restart are indistinguishable by
    // class list alone, so this guards the invariant that restarting never
    // strands the element without the class.
    expect(el.classList.contains('agent-flash')).toBe(true);
  });

  test('removes the class on animationend (self-cleaning)', () => {
    flashElement(el);

    el.dispatchEvent(new Event('animationend'));

    expect(el.classList.contains('agent-flash')).toBe(false);
  });

  test('still self-cleans after a reflow-restart', () => {
    flashElement(el);
    flashElement(el);

    el.dispatchEvent(new Event('animationend'));

    expect(el.classList.contains('agent-flash')).toBe(false);
  });

  test('ignores animationend for a different animation name', () => {
    flashElement(el);

    const ev = new Event('animationend');
    // Plain Event so happy-dom does not need AnimationEvent support.
    Object.defineProperty(ev, 'animationName', { value: 'some-other-anim' });
    el.dispatchEvent(ev);

    expect(el.classList.contains('agent-flash')).toBe(true);
  });

  test('ignores animationend bubbling from a child element', () => {
    const child = document.createElement('span');
    el.appendChild(child);
    flashElement(el);

    child.dispatchEvent(new Event('animationend', { bubbles: true }));

    expect(el.classList.contains('agent-flash')).toBe(true);
  });

  test('can flash again after a completed cycle', () => {
    flashElement(el);
    el.dispatchEvent(new Event('animationend'));

    flashElement(el);

    expect(el.classList.contains('agent-flash')).toBe(true);
  });
});
