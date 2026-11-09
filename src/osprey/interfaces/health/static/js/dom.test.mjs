// @ts-check
/* global KeyboardEvent, document -- vitest runs these in happy-dom */
/**
 * DOM builders (dom.js):
 *   npx vitest run src/osprey/interfaces/health/static/js/dom.test.mjs
 */

import { test, expect, describe, vi } from 'vitest';

import { h, svg, must, led, enterClick, pulse, note } from './dom.js';

describe('h', () => {
  test('sets class, text, data, listeners and attributes; skips null and false', () => {
    const fn = vi.fn();
    const n = h('button', {
      class: 'rp',
      text: '<b>x</b>',
      data: { name: 'a' },
      on: { click: fn },
      title: 't',
      hidden: false,
      id: null,
      tabindex: 0,
    });
    expect(n.className).toBe('rp');
    expect(n.textContent).toBe('<b>x</b>');
    expect(n.querySelector('b')).toBeNull();
    expect(n.dataset.name).toBe('a');
    expect(n.getAttribute('title')).toBe('t');
    expect(n.hasAttribute('hidden')).toBe(false);
    expect(n.hasAttribute('id')).toBe(false);
    expect(n.getAttribute('tabindex')).toBe('0');
    n.click();
    expect(fn).toHaveBeenCalledOnce();
  });

  test('true becomes an empty attribute; children flatten, strings become text', () => {
    const n = h('div', { hidden: true }, 'a', [h('span', null, 'b'), null, false], undefined);
    expect(n.getAttribute('hidden')).toBe('');
    expect(n.textContent).toBe('ab');
    expect(n.childNodes.length).toBe(2);
  });
});

describe('svg / must / led', () => {
  test('svg builds a namespaced element with attributes', () => {
    const c = svg('circle', { r: 80, cx: 100 });
    expect(c.namespaceURI).toBe('http://www.w3.org/2000/svg');
    expect(c.getAttribute('r')).toBe('80');
  });

  test('must throws for a missing skeleton element', () => {
    document.body.textContent = '';
    expect(() => must('nope')).toThrow('missing element: nope');
    document.body.append(h('div', { id: 'here' }));
    expect(must('here').id).toBe('here');
  });

  test('led carries the severity class', () => {
    expect(led('MINOR').className).toBe('led s-MINOR');
  });
});

describe('effects', () => {
  test('enterClick clicks the focused element on Enter', () => {
    const fn = vi.fn();
    const n = h('div', { on: { click: fn, keydown: enterClick } });
    n.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    n.dispatchEvent(new KeyboardEvent('keydown', { key: 'a' }));
    expect(fn).toHaveBeenCalledOnce();
  });

  test('pulse re-adds the flash class', () => {
    const n = h('div', { class: 'row flash' });
    pulse(n);
    expect(n.classList.contains('flash')).toBe(true);
  });

  test('note shows a transient line and hides it again', () => {
    vi.useFakeTimers();
    document.body.textContent = '';
    document.body.append(h('span', { id: 'note', hidden: true }));
    note('refresh suppressed');
    expect(must('note').hidden).toBe(false);
    expect(must('note').textContent).toBe('refresh suppressed');
    vi.advanceTimersByTime(6000);
    expect(must('note').hidden).toBe(true);
    vi.useRealTimers();
  });
});
