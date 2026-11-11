/** Unit tests for the OPTIMIZER panel's element builder (hyperscript.js). */

import { test, expect, describe, vi } from 'vitest';

import {
  fmt,
  h,
  replaceChildren,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/hyperscript.js';

describe('h', () => {
  test('text reaches the DOM as text, never markup', () => {
    const node = h('p', { text: '<b>x</b>' });
    expect(node.textContent).toBe('<b>x</b>');
    expect(node.querySelector('b')).toBeNull();
  });

  test('props, attributes, listeners, and skipped falsy values', () => {
    const onclick = vi.fn();
    const node = /** @type {HTMLButtonElement} */ (
      h(
        'button',
        { class: 'osprey-btn', hidden: true, 'aria-label': 'Go', title: null, onclick },
        'Go',
        null,
        false,
        3
      )
    );
    expect(node.className).toBe('osprey-btn');
    expect(node.hidden).toBe(true);
    expect(node.getAttribute('aria-label')).toBe('Go');
    expect(node.hasAttribute('title')).toBe(false);
    expect(node.textContent).toBe('Go3');
    node.click();
    expect(onclick).toHaveBeenCalledOnce();
  });

  test('a false prop is skipped, so `disabled: false` leaves a button enabled', () => {
    const node = /** @type {HTMLButtonElement} */ (h('button', { disabled: false }));
    expect(node.disabled).toBe(false);
  });
});

test('replaceChildren drops null children', () => {
  const parent = h('div', null, 'old');
  replaceChildren(parent, h('span', { text: 'a' }), null, false, h('span', { text: 'b' }));
  expect(parent.textContent).toBe('ab');
});

test('fmt keeps significant figures and dashes non-numbers', () => {
  expect(fmt(1.23456789)).toBe('1.23457');
  expect(fmt(0.1 + 0.2)).toBe('0.3');
  expect(fmt(null)).toBe('—');
  expect(fmt(Number.NaN)).toBe('—');
  expect(fmt('3')).toBe('—');
});
