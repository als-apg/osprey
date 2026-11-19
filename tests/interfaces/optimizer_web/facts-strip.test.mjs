/**
 * Unit tests for the facts strip (facts-strip.js): which facts it shows for
 * given values, and that an update changes values in place.
 */

import { test, expect } from 'vitest';

import {
  factsStrip,
  shownFacts,
  updateFacts,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/facts-strip.js';

/** @param {HTMLElement} strip */
const read = (strip) =>
  Array.from(strip.children).map((node) => [
    /** @type {HTMLElement} */ (node).dataset.fact,
    node.querySelector('.osprey-kicker')?.textContent,
    node.querySelector('.osprey-fact-value')?.textContent,
  ]);

test('a fact with no value is left out, never shown as a placeholder', () => {
  const facts = [
    { key: 'best', label: 'Best so far', value: '3.98', accent: true },
    { key: 'before', label: 'Before the run', value: null },
    { key: 'left', label: 'Left', value: undefined },
    { key: 'note', label: 'Note', value: '' },
    { key: 'writes', label: 'Evaluations', value: 0 },
  ];
  expect(shownFacts(facts).map((f) => f.key)).toEqual(['best', 'writes']);
  const strip = factsStrip(facts, { label: 'Run' });
  expect(read(strip)).toEqual([
    ['best', 'Best so far', '3.98'],
    ['writes', 'Evaluations', '0'],
  ]);
  expect(strip.getAttribute('aria-label')).toBe('Run');
  expect(strip.children[0].classList.contains('accent')).toBe(true);
});

test('a value may be a node', () => {
  const value = document.createElement('b');
  value.textContent = 'op1';
  const strip = factsStrip([{ key: 'who', label: 'Approver', value }]);
  expect(strip.querySelector('b')).toBe(value);
});

test('new values are written in place; a fact that comes or goes redraws the strip', () => {
  const strip = factsStrip([
    { key: 'best', label: 'Best', value: '1' },
    { key: 'left', label: 'Left', value: null },
  ]);
  const first = strip.children[0];
  updateFacts(strip, [
    { key: 'best', label: 'Best', value: '2', accent: true },
    { key: 'left', label: 'Left', value: null },
  ]);
  expect(strip.children[0]).toBe(first);
  expect(read(strip)).toEqual([['best', 'Best', '2']]);
  expect(first.classList.contains('accent')).toBe(true);
  updateFacts(strip, [
    { key: 'best', label: 'Best', value: '2' },
    { key: 'left', label: 'Left', value: 'about 4 min' },
  ]);
  expect(read(strip)).toEqual([
    ['best', 'Best', '2'],
    ['left', 'Left', 'about 4 min'],
  ]);
});
