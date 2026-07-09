// @ts-check
/**
 * Unit tests for the OKF Knowledge Panel's pure DOM-leaf helpers
 * (helpers.js) -- the interface's first JS unit test:
 *   npx vitest run tests/interfaces/okf_panel/helpers.test.mjs
 */

import { test, expect, describe, beforeEach } from 'vitest';

import {
  STRUCTURE_MARKER,
  el,
  isFallback,
  readPanelParams,
} from '../../../src/osprey/interfaces/okf_panel/static/js/helpers.js';

describe('el', () => {
  test('creates an element with the given tag', () => {
    const node = el('span');
    expect(node.tagName).toBe('SPAN');
  });

  test('the "class" key sets className, not a class attribute', () => {
    const node = el('div', { class: 'health-dot ok' });
    expect(node.className).toBe('health-dot ok');
  });

  test('the "text" key sets textContent', () => {
    const node = el('span', { text: 'hello world' });
    expect(node.textContent).toBe('hello world');
  });

  test('text is set via textContent, never parsed as HTML', () => {
    const node = el('div', { text: '<b>x</b>' });
    expect(node.textContent).toBe('<b>x</b>');
    expect(node.querySelector('b')).toBe(null);
    expect(node.children.length).toBe(0);
  });

  test('other keys are set as attributes, readable via getAttribute', () => {
    const node = el('a', { title: 'A title', href: '#control-system/x' });
    expect(node.getAttribute('title')).toBe('A title');
    expect(node.getAttribute('href')).toBe('#control-system/x');
  });

  test('children are appended in order', () => {
    const a = el('span', { text: 'first' });
    const b = el('span', { text: 'second' });
    const parent = el('div', null, [a, b]);
    expect(parent.children.length).toBe(2);
    expect(parent.children[0]).toBe(a);
    expect(parent.children[1]).toBe(b);
  });

  test('falsy children entries are skipped without throwing', () => {
    const a = el('span', { text: 'only' });
    expect(() => el('div', null, [null, a, undefined])).not.toThrow();
    const parent = el('div', null, [null, a, undefined]);
    expect(parent.children.length).toBe(1);
    expect(parent.children[0]).toBe(a);
  });

  test('attrs may be null, as in el("li", null, [link])', () => {
    const link = el('a', { href: '#x', text: 'link' });
    const node = el('li', null, [link]);
    expect(node.tagName).toBe('LI');
    expect(node.children[0]).toBe(link);
  });

  test('children may be omitted entirely, as in el("span", { class: "health-dot ok" })', () => {
    const node = el('span', { class: 'health-dot ok' });
    expect(node.tagName).toBe('SPAN');
    expect(node.className).toBe('health-dot ok');
    expect(node.children.length).toBe(0);
  });
});

describe('isFallback', () => {
  test('true when the title equals the last "/"-separated segment of the id', () => {
    expect(isFallback('control-system/channel-finding', 'channel-finding')).toBe(true);
  });

  test('false when the title is a real human title', () => {
    expect(isFallback('control-system/channel-finding', 'Channel Finding')).toBe(false);
  });

  test('false for empty or missing id/title (guard clause)', () => {
    expect(isFallback('', 'channel-finding')).toBe(false);
    expect(isFallback('control-system/channel-finding', '')).toBe(false);
    expect(isFallback(null, 'channel-finding')).toBe(false);
    expect(isFallback('control-system/channel-finding', null)).toBe(false);
    expect(isFallback(undefined, undefined)).toBe(false);
  });
});

describe('readPanelParams', () => {
  beforeEach(() => {
    location.hash = '';
  });

  test('empty hash yields concept: null, structure: false, raw: ""', () => {
    location.hash = '';
    expect(readPanelParams()).toEqual({ concept: null, structure: false, raw: '' });
  });

  test('the structure marker hash yields structure: true, concept: null', () => {
    location.hash = `#${STRUCTURE_MARKER}`;
    const params = readPanelParams();
    expect(params.structure).toBe(true);
    expect(params.concept).toBe(null);
  });

  test('a concept hash preserves slashes and sets structure: false', () => {
    location.hash = '#control-system/channel-finding';
    const params = readPanelParams();
    expect(params.concept).toBe('control-system/channel-finding');
    expect(params.structure).toBe(false);
  });

  test('a malformed percent-escape falls back to the raw hash via the catch branch', () => {
    // Verified to throw URIError in decodeURIComponent, and happy-dom does
    // not normalize it away when assigned to location.hash:
    //   node -e 'decodeURIComponent("%E0%A4%A")' -> URIError: URI malformed
    const malformed = '%E0%A4%A';
    expect(() => decodeURIComponent(malformed)).toThrow(URIError);
    location.hash = `#${malformed}`;
    const params = readPanelParams();
    expect(params.concept).toBe(malformed);
    expect(params.raw).toBe(malformed);
  });
});
