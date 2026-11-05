// @ts-check
/**
 * ARIEL search operators (query-ops.js): lifting `dim:value` tokens out of the
 * search text into the C-FILTER object. What is pinned:
 *
 *   - `dim:value` and `dim:"quoted value"` for every declared dimension,
 *     matched ignoring case, several values OR-ed;
 *   - a prefix that is not a dimension stays in the query and is reported;
 *   - `date:YYYY-MM-DD` is one day, `date:YYYY-MM` the whole month; an invalid
 *     date stays in the query;
 *   - completeness: without `final` a trailing token is stripped, not lifted;
 *   - merging is additive and idempotent, taking back a previous contribution
 *     keeps the other values, and an author lifted under Mine switches to All;
 *   - the standalone Enter write (`liftSearch`).
 *
 *   npx vitest run tests/interfaces/ariel/query-ops.test.mjs
 */
import { test, expect, describe } from 'vitest';

import {
  addLifted,
  dateWindow,
  holdsLifted,
  liftOperators,
  liftSearch,
  liftedPatch,
  removeLifted,
} from '../../../src/osprey/interfaces/ariel/static/js/query-ops.js';

const DIMS = [{ name: 'author' }, { name: 'source_system' }, { name: 'logbook' }];
const FINAL = { final: true };

describe('lifting', () => {
  test('a complete dim:value token is lifted and stripped', () => {
    expect(liftOperators('author:bob beam dump', DIMS)).toEqual({
      query: 'beam dump',
      filters: { author: ['bob'] },
      unknown: [],
    });
  });

  test('tokens anywhere in the text are lifted; repeats of a dimension are OR-ed', () => {
    const r = liftOperators('kicker author:bob trip author:alice logbook:ops', DIMS, FINAL);
    expect(r.query).toBe('kicker trip');
    expect(r.filters).toEqual({ author: ['alice', 'bob'], logbook: ['ops'] });
  });

  test('dimension names match ignoring case and map to the declared name', () => {
    expect(liftOperators('Author:bob x', DIMS).filters).toEqual({ author: ['bob'] });
  });

  test('a token with an empty value is stripped and applies nothing', () => {
    expect(liftOperators('author: kicker', DIMS)).toMatchObject({ query: 'kicker', filters: {} });
  });

  test('whitespace in the query is collapsed and trimmed', () => {
    expect(liftOperators('  beam   dump  ', DIMS, FINAL).query).toBe('beam dump');
  });
});

describe('quoting', () => {
  test('a closed quote is a complete token holding whitespace', () => {
    const r = liftOperators('author:"bob beam" dump', DIMS);
    expect(r).toMatchObject({ query: 'dump', filters: { author: ['bob beam'] } });
  });

  test('a closed quote at the end of the text is complete without final', () => {
    expect(liftOperators('x source_system:"RF ctl"', DIMS).filters).toEqual({
      source_system: ['RF ctl'],
    });
  });

  test('an unclosed quote is stripped but not lifted until final', () => {
    expect(liftOperators('x author:"bob be', DIMS)).toMatchObject({ query: 'x', filters: {} });
    expect(liftOperators('x author:"bob be', DIMS, FINAL).filters).toEqual({
      author: ['bob be'],
    });
  });
});

describe('unknown prefixes', () => {
  test('a prefix that is not a dimension stays in the query and is reported', () => {
    expect(liftOperators('color:red author:bob rest', DIMS)).toEqual({
      query: 'color:red rest',
      filters: { author: ['bob'] },
      unknown: ['color'],
    });
  });

  test('without dimensions only date: lifts', () => {
    const r = liftOperators('author:bob date:2026-03-04 x', null);
    expect(r).toEqual({
      query: 'author:bob x',
      filters: { time: { start: '2026-03-04', end: '2026-03-04' } },
      unknown: ['author'],
    });
  });

  test('a URL-like or plain word is left alone', () => {
    expect(liftOperators('see 10:30 ok', DIMS, FINAL).query).toBe('see 10:30 ok');
  });
});

describe('date: mapping', () => {
  test('date:YYYY-MM-DD is that one day', () => {
    expect(liftOperators('date:2026-02-03 x', DIMS).filters).toEqual({
      time: { start: '2026-02-03', end: '2026-02-03' },
    });
  });

  test('date:YYYY-MM is the first to the last day of the month', () => {
    expect(dateWindow('2026-02')).toEqual({ start: '2026-02-01', end: '2026-02-28' });
    expect(dateWindow('2024-02')).toEqual({ start: '2024-02-01', end: '2024-02-29' });
    expect(dateWindow('2026-12')).toEqual({ start: '2026-12-01', end: '2026-12-31' });
    expect(dateWindow('2026-04')).toEqual({ start: '2026-04-01', end: '2026-04-30' });
  });

  test('an invalid date stays in the query', () => {
    expect(dateWindow('2026-02-30')).toBeNull();
    expect(dateWindow('2026-13')).toBeNull();
    const r = liftOperators('date:yesterday x', DIMS);
    expect(r).toMatchObject({ query: 'date:yesterday x', filters: {} });
  });

  test('the last date wins', () => {
    expect(liftOperators('date:2026-01 date:2026-05-06', DIMS, FINAL).filters).toEqual({
      time: { start: '2026-05-06', end: '2026-05-06' },
    });
  });
});

describe('completeness', () => {
  test('a trailing token without whitespace is stripped but not lifted', () => {
    expect(liftOperators('kicker author:bo', DIMS)).toMatchObject({
      query: 'kicker',
      filters: {},
    });
  });

  test('final lifts the trailing token', () => {
    expect(liftOperators('kicker author:bo', DIMS, FINAL).filters).toEqual({ author: ['bo'] });
  });

  test('an incomplete trailing dim: is stripped and applies nothing, even final', () => {
    expect(liftOperators('kicker author:', DIMS, FINAL)).toMatchObject({
      query: 'kicker',
      filters: {},
    });
  });

  test('a trailing token followed by whitespace is complete', () => {
    expect(liftOperators('author:bob ', DIMS).filters).toEqual({ author: ['bob'] });
  });
});

describe('merging', () => {
  test('addLifted unions values and is idempotent', () => {
    const once = addLifted({ author: ['alice'] }, { author: ['bob'] });
    expect(once).toEqual({ author: ['alice', 'bob'] });
    expect(addLifted(once, { author: ['BOB'] })).toEqual(once);
  });

  test('a lifted window replaces the window', () => {
    const t = { start: '2026-01-01', end: '2026-01-31' };
    expect(addLifted({ time: { start: '2025-01-01', end: null } }, { time: t })).toEqual({
      time: t,
    });
  });

  test('removeLifted takes back only the contribution', () => {
    const t = { start: '2026-01-01', end: '2026-01-01' };
    const f = { author: ['alice', 'bob'], time: t };
    expect(removeLifted(f, { author: ['bob'], time: t })).toEqual({ author: ['alice'] });
    expect(removeLifted({ time: { start: 'x', end: null } }, { time: t })).toEqual({
      time: { start: 'x', end: null },
    });
  });

  test('holdsLifted notices a removed lifted value', () => {
    expect(holdsLifted({ author: ['Bob'] }, { author: ['bob'] })).toBe(true);
    expect(holdsLifted({}, { author: ['bob'] })).toBe(false);
    expect(holdsLifted({ author: ['a'] }, {})).toBe(true);
  });

  test('liftedPatch replaces the previous contribution and keeps facet values', () => {
    const state = { filter: { author: ['alice', 'J.'] }, mailbox: 'all' };
    expect(liftedPatch(state, { author: ['J. Smith'] }, { author: ['J.'] })).toEqual({
      filter: { author: ['alice', 'J. Smith'] },
    });
  });

  test('liftedPatch writes no filter when nothing changes', () => {
    expect(liftedPatch({ filter: { author: ['bob'] } }, { author: ['bob'] })).toEqual({});
  });
});

describe('Mine switch', () => {
  test('an author lifted under Mine switches the mailbox to all', () => {
    expect(liftedPatch({ filter: {}, mailbox: 'mine' }, { author: ['bob'] })).toEqual({
      filter: { author: ['bob'] },
      mailbox: 'all',
    });
  });

  test('another dimension under Mine keeps the mailbox', () => {
    expect(liftedPatch({ filter: {}, mailbox: 'mine' }, { logbook: ['ops'] })).toEqual({
      filter: { logbook: ['ops'] },
    });
  });
});

describe('liftSearch (standalone Enter)', () => {
  test('returns the stripped query, merged filter, mailbox and rerank in one patch', () => {
    const caps = { filter_dimensions: DIMS };
    const state = { filter: { logbook: ['ops'] }, mailbox: 'mine' };
    expect(liftSearch('  trip author:bob', state, caps, 'now')).toEqual({
      query: 'trip',
      rerank: 'now',
      filter: { author: ['bob'], logbook: ['ops'] },
      mailbox: 'all',
    });
  });

  test('plain text writes only the query', () => {
    expect(liftSearch(' beam ', { filter: {} }, null, 'now')).toEqual({
      query: 'beam',
      rerank: 'now',
    });
  });
});
