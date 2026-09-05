// @ts-check
/**
 * Unit tests for the graph finder state model (graph-finder-state.js). Pure
 * logic, no DOM:
 *   npx vitest run tests/interfaces/channel_finder/graph-finder-state.test.mjs
 */

import { test, expect } from 'vitest';

import {
  createFinderState,
  toggleFacet,
  setQuery,
  toSearchParams,
  activeChips,
  removeChip,
  clampPage,
  toggleSelection,
  togglePageSelection,
  clearSelection,
  sendText,
  copyText,
} from '../../../src/osprey/interfaces/channel_finder/static/js/graph-finder-state.js';

test('createFinderState starts empty on page 1', () => {
  const s = createFinderState();
  expect(s.q).toBe('');
  expect(s.section).toBeInstanceOf(Set);
  expect(s.section.size).toBe(0);
  expect(s.system.size).toBe(0);
  expect(s.signal.size).toBe(0);
  expect(s.dir.size).toBe(0);
  expect(s.cls).toBeNull();
  expect(s.page).toBe(1);
  expect(s.selected.size).toBe(0);
});

test('createFinderState returns independent states', () => {
  const a = createFinderState();
  const b = createFinderState();
  toggleFacet(a, 'section', 'SR');
  expect(b.section.size).toBe(0);
});

// --- facets -----------------------------------------------------------------

test('multi facets accumulate values and toggle off when re-clicked', () => {
  const s = createFinderState();
  toggleFacet(s, 'section', 'SR');
  toggleFacet(s, 'section', 'BR');
  expect([...s.section].sort()).toEqual(['BR', 'SR']);
  toggleFacet(s, 'section', 'SR');
  expect([...s.section]).toEqual(['BR']);
});

test('every multi facet is independent', () => {
  const s = createFinderState();
  toggleFacet(s, 'section', 'SR');
  toggleFacet(s, 'system', 'BPM');
  toggleFacet(s, 'signal', 'X');
  toggleFacet(s, 'dir', 'R');
  expect([...s.section]).toEqual(['SR']);
  expect([...s.system]).toEqual(['BPM']);
  expect([...s.signal]).toEqual(['X']);
  expect([...s.dir]).toEqual(['R']);
});

test('class is single-select and toggles off when re-clicked', () => {
  const s = createFinderState();
  toggleFacet(s, 'cls', 'https://example.org/ont#BPM');
  expect(s.cls).toBe('https://example.org/ont#BPM');
  // A different class replaces rather than accumulating.
  toggleFacet(s, 'cls', 'https://example.org/ont#Corrector');
  expect(s.cls).toBe('https://example.org/ont#Corrector');
  // Re-clicking the active class clears it.
  toggleFacet(s, 'cls', 'https://example.org/ont#Corrector');
  expect(s.cls).toBeNull();
});

test('toggleFacet rejects an unknown facet', () => {
  const s = createFinderState();
  expect(() => toggleFacet(s, /** @type {any} */ ('nope'), 'x')).toThrow(/unknown facet/i);
});

test('every filter change and every query change resets to page 1', () => {
  const s = createFinderState();

  s.page = 4;
  toggleFacet(s, 'section', 'SR');
  expect(s.page).toBe(1);

  s.page = 4;
  toggleFacet(s, 'section', 'SR'); // toggling OFF also resets
  expect(s.page).toBe(1);

  s.page = 4;
  toggleFacet(s, 'system', 'BPM');
  expect(s.page).toBe(1);

  s.page = 4;
  toggleFacet(s, 'signal', 'X');
  expect(s.page).toBe(1);

  s.page = 4;
  toggleFacet(s, 'dir', 'none');
  expect(s.page).toBe(1);

  s.page = 4;
  toggleFacet(s, 'cls', 'urn:c');
  expect(s.page).toBe(1);

  s.page = 4;
  setQuery(s, 'quad');
  expect(s.page).toBe(1);
});

test('setQuery stores the query verbatim', () => {
  const s = createFinderState();
  setQuery(s, '  quad corrector ');
  expect(s.q).toBe('  quad corrector ');
  setQuery(s, '');
  expect(s.q).toBe('');
});

// --- query string -----------------------------------------------------------

test('toSearchParams always carries q and page, even when empty', () => {
  const s = createFinderState();
  expect(toSearchParams(s).toString()).toBe('q=&page=1');
});

test('multi facets serialise as repeated params in sorted order', () => {
  const s = createFinderState();
  toggleFacet(s, 'section', 'SR');
  toggleFacet(s, 'section', 'BR');
  const params = toSearchParams(s);
  expect(params.getAll('section')).toEqual(['BR', 'SR']);
  expect(params.toString()).toBe('q=&section=BR&section=SR&page=1');
});

test('serialisation is deterministic regardless of click order', () => {
  const a = createFinderState();
  toggleFacet(a, 'section', 'SR');
  toggleFacet(a, 'section', 'BR');
  const b = createFinderState();
  toggleFacet(b, 'section', 'BR');
  toggleFacet(b, 'section', 'SR');
  expect(toSearchParams(a).toString()).toBe(toSearchParams(b).toString());
});

test('dir serialises every value including "none"', () => {
  const s = createFinderState();
  toggleFacet(s, 'dir', 'none');
  toggleFacet(s, 'dir', 'RW');
  toggleFacet(s, 'dir', 'R');
  const params = toSearchParams(s);
  expect(params.getAll('dir')).toEqual(['R', 'RW', 'none']);
});

test('cls is emitted once when set and omitted when null', () => {
  const s = createFinderState();
  expect(toSearchParams(s).has('cls')).toBe(false);
  toggleFacet(s, 'cls', 'https://example.org/ont#BPM');
  const params = toSearchParams(s);
  expect(params.getAll('cls')).toEqual(['https://example.org/ont#BPM']);
});

test('empty facets are omitted entirely', () => {
  const s = createFinderState();
  setQuery(s, 'bpm');
  const params = toSearchParams(s);
  expect(params.has('section')).toBe(false);
  expect(params.has('system')).toBe(false);
  expect(params.has('signal')).toBe(false);
  expect(params.has('dir')).toBe(false);
  expect(params.has('cls')).toBe(false);
});

test('full round trip: query + every facet + page', () => {
  const s = createFinderState();
  setQuery(s, 'bpm x');
  toggleFacet(s, 'section', 'SR');
  toggleFacet(s, 'section', 'BR');
  toggleFacet(s, 'system', 'Diagnostics');
  toggleFacet(s, 'signal', 'X');
  toggleFacet(s, 'dir', 'R');
  toggleFacet(s, 'dir', 'none');
  toggleFacet(s, 'cls', 'urn:cls:bpm');
  s.page = 3;

  const params = toSearchParams(s);
  expect(params.get('q')).toBe('bpm x');
  expect(params.getAll('section')).toEqual(['BR', 'SR']);
  expect(params.getAll('system')).toEqual(['Diagnostics']);
  expect(params.getAll('signal')).toEqual(['X']);
  expect(params.getAll('dir')).toEqual(['R', 'none']);
  expect(params.getAll('cls')).toEqual(['urn:cls:bpm']);
  expect(params.get('page')).toBe('3');
  // Values are URL-encoded by URLSearchParams, not hand-escaped.
  expect(params.toString()).toContain('q=bpm+x');
});

// --- chips ------------------------------------------------------------------

test('activeChips is empty for a fresh state', () => {
  expect(activeChips(createFinderState())).toEqual([]);
});

test('a non-empty query gets its own chip; a blank one does not', () => {
  const s = createFinderState();
  setQuery(s, '   ');
  expect(activeChips(s)).toEqual([]);
  setQuery(s, 'quad');
  const chips = activeChips(s);
  expect(chips).toHaveLength(1);
  expect(chips[0].key).toBe('q');
  expect(chips[0].label).toBe('Search: quad');
});

test('activeChips yields one chip per active facet value, query first', () => {
  const s = createFinderState();
  setQuery(s, 'bpm');
  toggleFacet(s, 'section', 'SR');
  toggleFacet(s, 'section', 'BR');
  toggleFacet(s, 'dir', 'none');
  toggleFacet(s, 'cls', 'urn:cls:bpm');

  expect(activeChips(s)).toEqual([
    { key: 'q', label: 'Search: bpm' },
    { key: 'section:BR', label: 'Section: BR' },
    { key: 'section:SR', label: 'Section: SR' },
    { key: 'dir:none', label: 'Direction: none' },
    { key: 'cls:urn:cls:bpm', label: 'Device class: urn:cls:bpm' },
  ]);
});

test('every chip key round-trips through removeChip', () => {
  const s = createFinderState();
  setQuery(s, 'bpm');
  toggleFacet(s, 'section', 'SR');
  toggleFacet(s, 'system', 'Diagnostics');
  toggleFacet(s, 'signal', 'X');
  toggleFacet(s, 'dir', 'RW');
  toggleFacet(s, 'cls', 'https://example.org/ont#BPM');

  for (const chip of activeChips(s)) {
    removeChip(s, chip.key);
  }

  expect(s.q).toBe('');
  expect(s.section.size).toBe(0);
  expect(s.system.size).toBe(0);
  expect(s.signal.size).toBe(0);
  expect(s.dir.size).toBe(0);
  expect(s.cls).toBeNull();
  expect(activeChips(s)).toEqual([]);
});

test('removeChip drops only the named value and resets the page', () => {
  const s = createFinderState();
  toggleFacet(s, 'section', 'SR');
  toggleFacet(s, 'section', 'BR');
  s.page = 5;
  removeChip(s, 'section:SR');
  expect([...s.section]).toEqual(['BR']);
  expect(s.page).toBe(1);
});

test('removeChip on a class key whose uri contains colons clears the class', () => {
  const s = createFinderState();
  toggleFacet(s, 'cls', 'https://example.org/ont#BPM');
  const [chip] = activeChips(s);
  removeChip(s, chip.key);
  expect(s.cls).toBeNull();
});

test('removeChip ignores an unknown key without throwing', () => {
  const s = createFinderState();
  toggleFacet(s, 'section', 'SR');
  s.page = 3;
  removeChip(s, 'section:NOPE');
  removeChip(s, 'garbage');
  expect([...s.section]).toEqual(['SR']);
  expect(s.page).toBe(3);
});

test('removeChip leaves the selection untouched', () => {
  const s = createFinderState();
  toggleFacet(s, 'section', 'SR');
  toggleSelection(s, 'SR:BPM1');
  removeChip(s, 'section:SR');
  expect([...s.selected]).toEqual(['SR:BPM1']);
});

// --- paging -----------------------------------------------------------------

test('clampPage pulls the page back into range and reports the change', () => {
  const s = createFinderState();
  s.page = 7;
  expect(clampPage(s, 3)).toBe(true);
  expect(s.page).toBe(3);
});

test('clampPage leaves an in-range page alone', () => {
  const s = createFinderState();
  s.page = 2;
  expect(clampPage(s, 5)).toBe(false);
  expect(s.page).toBe(2);
});

test('clampPage floors at 1 for empty or nonsensical page counts', () => {
  const s = createFinderState();
  s.page = 4;
  expect(clampPage(s, 0)).toBe(true);
  expect(s.page).toBe(1);

  s.page = 4;
  clampPage(s, -3);
  expect(s.page).toBe(1);

  s.page = 0;
  clampPage(s, 10);
  expect(s.page).toBe(1);
});

// --- selection --------------------------------------------------------------

test('toggleSelection adds then removes, reporting the new state', () => {
  const s = createFinderState();
  expect(toggleSelection(s, 'SR:BPM1')).toBe(true);
  expect(s.selected.has('SR:BPM1')).toBe(true);
  expect(toggleSelection(s, 'SR:BPM1')).toBe(false);
  expect(s.selected.has('SR:BPM1')).toBe(false);
});

test('toggleSelection does not touch the page', () => {
  const s = createFinderState();
  s.page = 3;
  toggleSelection(s, 'SR:BPM1');
  expect(s.page).toBe(3);
});

test('togglePageSelection adds or removes a whole page at once', () => {
  const s = createFinderState();
  togglePageSelection(s, ['a', 'b', 'c'], true);
  expect([...s.selected]).toEqual(['a', 'b', 'c']);

  // Selecting a second page keeps the first page's rows (selection is
  // preserved across pages within a mount).
  togglePageSelection(s, ['d', 'e'], true);
  expect([...s.selected]).toEqual(['a', 'b', 'c', 'd', 'e']);

  togglePageSelection(s, ['a', 'b'], false);
  expect([...s.selected]).toEqual(['c', 'd', 'e']);
});

test('togglePageSelection on is idempotent', () => {
  const s = createFinderState();
  togglePageSelection(s, ['a', 'b'], true);
  togglePageSelection(s, ['a', 'b'], true);
  expect([...s.selected]).toEqual(['a', 'b']);
});

test('clearSelection empties the selection and nothing else', () => {
  const s = createFinderState();
  setQuery(s, 'bpm');
  toggleFacet(s, 'section', 'SR');
  s.page = 2;
  togglePageSelection(s, ['a', 'b'], true);

  clearSelection(s);
  expect(s.selected.size).toBe(0);
  expect(s.q).toBe('bpm');
  expect([...s.section]).toEqual(['SR']);
  expect(s.page).toBe(2);
});

test('selection survives a page change', () => {
  const s = createFinderState();
  toggleSelection(s, 'SR:BPM1');
  s.page = 2;
  toggleSelection(s, 'SR:BPM2');
  expect([...s.selected]).toEqual(['SR:BPM1', 'SR:BPM2']);
});

// --- text output ------------------------------------------------------------

test('sendText joins addresses with spaces on one line, no trailing newline', () => {
  const s = createFinderState();
  togglePageSelection(s, ['SR:BPM1', 'SR:BPM2', 'SR:BPM3'], true);
  const text = sendText(s);
  expect(text).toBe('SR:BPM1 SR:BPM2 SR:BPM3');
  expect(text.includes('\n')).toBe(false);
  expect(text.endsWith(' ')).toBe(false);
});

test('copyText joins addresses with newlines, no trailing newline', () => {
  const s = createFinderState();
  togglePageSelection(s, ['SR:BPM1', 'SR:BPM2'], true);
  const text = copyText(s);
  expect(text).toBe('SR:BPM1\nSR:BPM2');
  expect(text.endsWith('\n')).toBe(false);
});

test('text output follows insertion order, not sort order', () => {
  const s = createFinderState();
  toggleSelection(s, 'SR:zeta');
  toggleSelection(s, 'SR:alpha');
  expect(sendText(s)).toBe('SR:zeta SR:alpha');
  expect(copyText(s)).toBe('SR:zeta\nSR:alpha');
});

test('re-selecting a row moves it to the end of the order', () => {
  const s = createFinderState();
  togglePageSelection(s, ['a', 'b'], true);
  toggleSelection(s, 'a'); // off
  toggleSelection(s, 'a'); // on again
  expect(copyText(s)).toBe('b\na');
});

test('an empty selection produces empty text', () => {
  const s = createFinderState();
  expect(sendText(s)).toBe('');
  expect(copyText(s)).toBe('');
});

test('a single selection produces no separator', () => {
  const s = createFinderState();
  toggleSelection(s, 'SR:BPM1');
  expect(sendText(s)).toBe('SR:BPM1');
  expect(copyText(s)).toBe('SR:BPM1');
});
