// @ts-check
/**
 * Unit tests for the graph finder's kept arrangement (graph-finder-prefs.js):
 * the column layout and the view preferences, stored per render, and the
 * fallbacks that keep the finder working when storage is refused or corrupt.
 *   npx vitest run tests/interfaces/channel_finder/graph-finder-prefs.test.mjs
 */

import { test, expect, beforeEach, afterEach, vi } from 'vitest';

import {
  COLUMN_SPECS,
  COLUMNS_KEY,
  PREFS_KEY,
  loadColumns,
  loadPrefs,
  saveColumns,
  savePrefs,
} from '../../../src/osprey/interfaces/channel_finder/static/js/graph-finder-prefs.js';
import { defaultLayout, moveColumn, resizeColumn } from '/design-system/js/table-columns.js';

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute('data-osprey-storage-scope');
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  document.documentElement.removeAttribute('data-osprey-storage-scope');
});

test('every data column is a server sort key, and chrome is pinned', () => {
  const data = COLUMN_SPECS.filter((s) => !s.pin)
    .map((s) => s.key)
    .sort();
  expect(data).toEqual(['address', 'description', 'device', 'dir', 'section', 'signal', 'system']);
  expect(COLUMN_SPECS[0]).toMatchObject({ key: 'chk', pin: 'start' });
  expect(COLUMN_SPECS.at(-1)).toMatchObject({ key: 'act', pin: 'end' });
  expect(COLUMN_SPECS.find((s) => s.key === 'system')?.hidden).toBe(true);
});

test('nothing stored is the default layout and comfortable rows', () => {
  expect(loadColumns()).toEqual(defaultLayout(COLUMN_SPECS));
  expect(loadPrefs()).toEqual({ density: 'comfortable', collapsed: [] });
});

test('a saved layout comes back', () => {
  const layout = defaultLayout(COLUMN_SPECS);
  moveColumn(layout, 'description', 'device', true);
  resizeColumn(COLUMN_SPECS, layout, 'address', 300);
  saveColumns(layout);
  expect(loadColumns()).toEqual(layout);
});

test('saved preferences come back, and junk in them is dropped', () => {
  savePrefs({ density: 'compact', collapsed: ['cls', 'dir'] });
  expect(loadPrefs()).toEqual({ density: 'compact', collapsed: ['cls', 'dir'] });
  localStorage.setItem(PREFS_KEY, JSON.stringify({ density: 'huge', collapsed: ['dir', 7, null] }));
  expect(loadPrefs()).toEqual({ density: 'comfortable', collapsed: ['dir'] });
  localStorage.setItem(PREFS_KEY, '{not json');
  expect(loadPrefs()).toEqual({ density: 'comfortable', collapsed: [] });
});

test('a multi-user render keeps its arrangement under its own key', () => {
  document.documentElement.setAttribute('data-osprey-storage-scope', 'alice');
  savePrefs({ density: 'compact', collapsed: [] });
  expect(localStorage.getItem(`${PREFS_KEY}--alice`)).not.toBeNull();
  expect(localStorage.getItem(PREFS_KEY)).toBeNull();
  document.documentElement.removeAttribute('data-osprey-storage-scope');
  expect(loadPrefs().density).toBe('comfortable');
});

test('refused storage forgets quietly instead of breaking the finder', () => {
  const deny = () => {
    throw new Error('denied');
  };
  vi.stubGlobal('localStorage', { getItem: deny, setItem: deny, removeItem: deny, clear: deny });
  expect(() => saveColumns(defaultLayout(COLUMN_SPECS))).not.toThrow();
  expect(() => savePrefs({ density: 'compact', collapsed: [] })).not.toThrow();
  expect(loadColumns()).toEqual(defaultLayout(COLUMN_SPECS));
  expect(loadPrefs().density).toBe('comfortable');
  expect(COLUMNS_KEY).toBe('cf-graph-columns');
});
