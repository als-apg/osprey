// @ts-check
/**
 * Saved-search storage: saved-searches.js keeps `{name, query, mode, filter,
 * filters}` records in browser storage and migrates P1 records on read.
 *
 * Run: npx vitest run tests/interfaces/ariel/saved-searches.test.mjs
 */
import { test, expect, describe, beforeEach, vi } from 'vitest';

const SAVED = '../../../src/osprey/interfaces/ariel/static/js/saved-searches.js';

/** @type {typeof import('../../../src/osprey/interfaces/ariel/static/js/saved-searches.js')} */
let saved;

beforeEach(async () => {
  localStorage.clear();
  document.documentElement.removeAttribute('data-osprey-storage-scope');
  vi.resetModules();
  saved = await import(SAVED);
});

/** @returns {any} */
function stored() {
  return JSON.parse(localStorage.getItem(saved.SAVED_KEY) || 'null');
}

const RECORD = {
  name: 'K',
  query: 'kicker',
  mode: 'semantic',
  filter: { author: ['Bob'] },
  filters: { max_results: 3 },
};

describe('saveSearch / readSaved / findSaved', () => {
  test('stores a record and reads it back in the current shape', () => {
    expect(saved.saveSearch(RECORD)).toBe(true);
    expect(stored()).toEqual([RECORD]);
    expect(saved.readSaved()).toEqual([RECORD]);
    expect(saved.findSaved('K')).toEqual(RECORD);
    expect(saved.findSaved('nope')).toBeUndefined();
  });

  test('the name is trimmed; a blank name is ignored', () => {
    expect(saved.saveSearch({ ...RECORD, name: '   ' })).toBe(false);
    expect(localStorage.getItem(saved.SAVED_KEY)).toBeNull();
    saved.saveSearch({ ...RECORD, name: '  K  ' });
    expect(saved.readSaved().map((s) => s.name)).toEqual(['K']);
  });

  test('saving an existing name replaces it in place', () => {
    saved.saveSearch({ ...RECORD, name: 'a', query: '1' });
    saved.saveSearch({ ...RECORD, name: 'b', query: '2' });
    saved.saveSearch({ ...RECORD, name: 'a', query: '3' });
    expect(saved.readSaved().map((s) => [s.name, s.query])).toEqual([
      ['a', '3'],
      ['b', '2'],
    ]);
  });

  test('the stored record is a copy', () => {
    const filter = { author: ['Bob'] };
    const filters = { max_results: 3 };
    saved.saveSearch({ ...RECORD, filter, filters });
    filter.author.push('Eve');
    filters.max_results = 9;
    expect(saved.findSaved('K')).toMatchObject({
      filter: { author: ['Bob'] },
      filters: { max_results: 3 },
    });
  });

  test('time is kept in a saved filter', () => {
    const time = { start: '2026-01-01', end: '2026-02-01' };
    saved.saveSearch({ ...RECORD, filter: { time } });
    expect(saved.findSaved('K')?.filter).toEqual({ time });
  });

  test('removeSearch deletes only that name', () => {
    saved.saveSearch({ ...RECORD, name: 'gone' });
    saved.saveSearch({ ...RECORD, name: 'kept' });
    saved.removeSearch('gone');
    expect(saved.readSaved().map((s) => s.name)).toEqual(['kept']);
  });
});

describe('readSaved: what storage holds', () => {
  test('a P1 record is migrated: author/source_system/dates move into filter', () => {
    localStorage.setItem(
      saved.SAVED_KEY,
      JSON.stringify([
        {
          name: 'Old',
          query: 'beam',
          mode: 'keyword',
          filters: {
            author: 'Bob',
            source_system: 'ops',
            start_date: '2026-01-01',
            end_date: '2026-02-01',
            max_results: 5,
          },
        },
      ])
    );
    expect(saved.readSaved()).toEqual([
      {
        name: 'Old',
        query: 'beam',
        mode: 'keyword',
        filter: {
          author: ['Bob'],
          source_system: ['ops'],
          time: { start: '2026-01-01', end: '2026-02-01' },
        },
        filters: { max_results: 5 },
      },
    ]);
  });

  test("a record's own filter wins over what is migrated", () => {
    localStorage.setItem(
      saved.SAVED_KEY,
      JSON.stringify([{ name: 'X', filter: { author: ['Eve'] }, filters: { author: 'Bob' } }])
    );
    expect(saved.findSaved('X')?.filter).toEqual({ author: ['Eve'] });
  });

  test('corrupt or non-list storage is an empty list', () => {
    localStorage.setItem(saved.SAVED_KEY, '{not json');
    expect(saved.readSaved()).toEqual([]);
    localStorage.setItem(saved.SAVED_KEY, '{"a":1}');
    expect(saved.readSaved()).toEqual([]);
  });

  test('malformed items and repeated names are skipped; missing fields default', () => {
    localStorage.setItem(
      saved.SAVED_KEY,
      JSON.stringify([
        { nope: 1 },
        'str',
        null,
        { name: '  ok  ', query: 7, filters: [1] },
        { name: 'ok', query: 'dup' },
      ])
    );
    expect(saved.readSaved()).toEqual([
      { name: 'ok', query: '', mode: null, filter: {}, filters: {} },
    ]);
  });

  test('the key honours the storage scope', () => {
    document.documentElement.setAttribute('data-osprey-storage-scope', 'acme');
    saved.saveSearch(RECORD);
    expect(localStorage.getItem(`${saved.SAVED_KEY}--acme`)).not.toBeNull();
    expect(localStorage.getItem(saved.SAVED_KEY)).toBeNull();
    expect(saved.findSaved('K')).toEqual(RECORD);
  });

  test('blocked storage never throws', () => {
    const get = vi.spyOn(localStorage, 'getItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    const set = vi.spyOn(localStorage, 'setItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    try {
      expect(() => saved.saveSearch(RECORD)).not.toThrow();
      expect(saved.readSaved()).toEqual([]);
      expect(() => saved.removeSearch('K')).not.toThrow();
    } finally {
      get.mockRestore();
      set.mockRestore();
    }
  });
});
