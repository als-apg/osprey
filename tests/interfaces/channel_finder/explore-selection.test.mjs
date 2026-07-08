// @ts-check
/**
 * Unit tests for the hierarchical selection state machine (explore-selection.js),
 * extracted from explore-hierarchical.js during the P3 retrofit. Pure logic:
 *   npx vitest run tests/interfaces/channel_finder/explore-selection.test.mjs
 */

import { test, expect } from 'vitest';

import { computeSelection, isTreeLevel } from '../../../src/osprey/interfaces/channel_finder/static/js/explore-selection.js';

test('non-terminal levels are single-select: a new value replaces the old', () => {
  // Fresh selection loads the next level.
  const first = computeSelection([], 'A', false);
  expect(first.selectedValues).toEqual(['A']);
  expect(first.selectionValue).toBe('A');
  expect(first.loadNext).toBe(true);

  // Clicking a different value clears the prior single selection.
  const replaced = computeSelection(['A'], 'B', false);
  expect(replaced.selectedValues).toEqual(['B']);
  expect(replaced.selectionValue).toBe('B');
  expect(replaced.loadNext).toBe(true);
});

test('clicking the selected value again deselects it (no drill-down)', () => {
  const off = computeSelection(['A'], 'A', false);
  expect(off.selectedValues).toEqual([]);
  expect(off.selectionValue).toBeNull();
  expect(off.loadNext).toBe(false);
});

test('terminal level is multi-select and never drills down', () => {
  // A second value is added, not replaced.
  const two = computeSelection(['A'], 'B', true);
  expect(two.selectedValues).toEqual(['A', 'B']);
  expect(two.selectionValue).toEqual(['A', 'B']);
  expect(two.loadNext).toBe(false);

  // Toggling one off on a terminal level leaves the remaining scalar selection.
  const one = computeSelection(['A', 'B'], 'A', true);
  expect(one.selectedValues).toEqual(['B']);
  expect(one.selectionValue).toBe('B');
  expect(one.loadNext).toBe(false);
});

test('selectionValue is null for empty, a scalar for one, an array for many', () => {
  expect(computeSelection(['A'], 'A', true).selectionValue).toBeNull();
  expect(computeSelection([], 'A', true).selectionValue).toBe('A');
  expect(computeSelection(['A', 'B'], 'C', true).selectionValue).toEqual(['A', 'B', 'C']);
});

test('isTreeLevel defaults unknown levels to tree and honors explicit config', () => {
  expect(isTreeLevel(undefined, 'system')).toBe(true);
  expect(isTreeLevel({}, 'system')).toBe(true);
  expect(isTreeLevel({ system: { type: 'tree' } }, 'system')).toBe(true);
  expect(isTreeLevel({ sector: { type: 'instance' } }, 'sector')).toBe(false);
});
