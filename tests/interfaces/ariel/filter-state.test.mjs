// @ts-check
/**
 * Panel filter state (C-STATE): the C-FILTER object kept under the store's
 * `filter` key and the helpers every writer of it goes through. What is pinned:
 *
 *   - the canonical form: sorted keys, sorted values deduplicated ignoring
 *     case, empty dimensions and an unset time window dropped;
 *   - the request scope: Mine replaces `author`, Drafts makes no request;
 *   - sanitizing against the declared dimensions always keeps `time`, and
 *     without capabilities returns the filter unchanged;
 *   - toggling and merging, including clearing a dimension;
 *   - the split between flat Filters & Options values and the filter/knobs
 *     keys, and its inverse;
 *   - migrating a P1 saved search.
 *
 *   npx vitest run tests/interfaces/ariel/filter-state.test.mjs
 */
import { test, expect, describe } from 'vitest';

import {
  normalizeFilter,
  requestFilter,
  sanitizeFilter,
  toggleValue,
  mergeFilter,
  splitFilters,
  joinFilters,
  migrateSaved,
} from '../../../src/osprey/interfaces/ariel/static/js/filter-state.js';

describe('normalizeFilter', () => {
  test('sorts keys and values, dedupes ignoring case, drops empties', () => {
    const f = normalizeFilter({
      tags: ['rf', 'RF', ' beam ', ''],
      author: ['bob', 'Alice'],
      shift: [],
      time: { start: null, end: '' },
    });
    expect(f).toEqual({ author: ['Alice', 'bob'], tags: ['beam', 'rf'] });
    expect(Object.keys(f)).toEqual(['author', 'tags']);
  });

  test('keeps a time window with one bound and nulls the other', () => {
    expect(normalizeFilter({ time: { start: '2026-01-01' } })).toEqual({
      time: { start: '2026-01-01', end: null },
    });
  });

  test('a lone string is a list of one; junk is nothing', () => {
    expect(normalizeFilter({ author: 'alice', x: 3 })).toEqual({ author: ['alice'] });
    expect(normalizeFilter(null)).toEqual({});
    expect(normalizeFilter([1])).toEqual({});
  });
});

describe('requestFilter', () => {
  const filter = { author: ['alice'], tags: ['rf'] };

  test('All sends the filter as it is', () => {
    expect(requestFilter({ filter, mailbox: 'all', mineAuthor: 'bob' })).toEqual(filter);
  });

  test('Mine replaces the author dimension with the Mine name', () => {
    expect(requestFilter({ filter, mailbox: 'mine', mineAuthor: 'bob' })).toEqual({
      author: ['bob'],
      tags: ['rf'],
    });
  });

  test('Drafts makes no request', () => {
    expect(requestFilter({ filter, mailbox: 'drafts', mineAuthor: null })).toBeNull();
  });

  test('a saved box sends the filter too', () => {
    expect(requestFilter({ filter, mailbox: { saved: 'x' }, mineAuthor: null })).toEqual(filter);
  });
});

describe('sanitizeFilter', () => {
  const dims = [{ name: 'author' }, { name: 'tags' }];

  test('drops undeclared dimensions, names them, and keeps time', () => {
    const { filter, dropped } = sanitizeFilter(
      { author: ['a'], shift: ['owl'], zone: ['x'], time: { start: '2026-01-01', end: null } },
      dims
    );
    expect(filter).toEqual({ author: ['a'], time: { start: '2026-01-01', end: null } });
    expect(dropped).toEqual(['shift', 'zone']);
  });

  test('without capabilities the filter comes back unchanged', () => {
    const f = { shift: ['owl'] };
    expect(sanitizeFilter(f, null)).toEqual({ filter: f, dropped: [] });
  });
});

describe('toggleValue and mergeFilter', () => {
  test('toggling adds, then removes ignoring case, then drops the dimension', () => {
    const one = toggleValue({}, 'tags', 'rf');
    expect(one).toEqual({ tags: ['rf'] });
    expect(toggleValue(one, 'tags', 'beam')).toEqual({ tags: ['beam', 'rf'] });
    expect(toggleValue(one, 'tags', 'RF')).toEqual({});
  });

  test('merging replaces keys; null or [] clears one', () => {
    const base = { author: ['a'], tags: ['rf'] };
    expect(mergeFilter(base, { author: ['b'] })).toEqual({ author: ['b'], tags: ['rf'] });
    expect(mergeFilter(base, { author: [], tags: null })).toEqual({});
    expect(mergeFilter(base, { time: { start: '2026-01-01', end: null } })).toEqual({
      ...base,
      time: { start: '2026-01-01', end: null },
    });
  });
});

describe('splitFilters and joinFilters', () => {
  const flat = {
    author: 'alice',
    source_system: 'elog',
    start_date: '2026-01-01',
    end_date: '2026-02-01',
    fuzzy: true,
    max_results: 25,
  };

  test('split maps author, source_system and the dates; the rest are knobs', () => {
    expect(splitFilters(flat)).toEqual({
      filter: {
        author: ['alice'],
        source_system: ['elog'],
        time: { start: '2026-01-01', end: '2026-02-01' },
      },
      knobs: { fuzzy: true, max_results: 25 },
    });
  });

  test('blank values are not dimensions', () => {
    expect(splitFilters({ author: '', start_date: null, fuzzy: false })).toEqual({
      filter: {},
      knobs: { fuzzy: false },
    });
    expect(splitFilters(null)).toEqual({ filter: {}, knobs: {} });
  });

  test('join is the inverse of split', () => {
    const { filter, knobs } = splitFilters(flat);
    expect(joinFilters(filter, knobs)).toEqual(flat);
  });

  test('a multi-value dimension joins as a list and splits back', () => {
    const values = joinFilters({ author: ['a', 'b'], tags: ['rf'] }, {});
    expect(values).toEqual({ author: ['a', 'b'] });
    expect(splitFilters(values).filter).toEqual({ author: ['a', 'b'] });
  });
});

describe('migrateSaved', () => {
  test('a P1 record moves its dimensions into filter', () => {
    const p1 = {
      name: 'quench',
      query: 'quench',
      mode: 'keyword',
      filters: { author: 'alice', start_date: '2026-01-01', fuzzy: true },
    };
    expect(migrateSaved(p1)).toEqual({
      name: 'quench',
      query: 'quench',
      mode: 'keyword',
      filter: { author: ['alice'], time: { start: '2026-01-01', end: null } },
      filters: { fuzzy: true },
    });
  });

  test('a current record keeps its own filter, which wins over a moved one', () => {
    const rec = {
      name: 'n',
      query: '',
      mode: null,
      filter: { author: ['bob'], tags: ['rf'] },
      filters: { author: 'alice', fuzzy: true },
    };
    expect(migrateSaved(rec)).toEqual({
      name: 'n',
      query: '',
      mode: null,
      filter: { author: ['bob'], tags: ['rf'] },
      filters: { fuzzy: true },
    });
  });

  test('migration is idempotent', () => {
    const once = migrateSaved({
      name: 'n',
      query: 'q',
      mode: 'semantic',
      filters: { author: 'a' },
    });
    expect(migrateSaved(once)).toEqual(once);
  });
});
