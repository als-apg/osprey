// @ts-check
/**
 * The panel's route grammar (route.js): parseRoute and its inverse formatRoute.
 *   npx vitest run tests/interfaces/channel_finder/route.test.mjs
 */

import { test, expect, describe } from 'vitest';
import {
  FACET_KEYS,
  allPvsHash,
  familyOf,
  formatRoute,
  parseRoute,
} from '../../../src/osprey/interfaces/channel_finder/static/js/route.js';

describe('parseRoute', () => {
  test.each([
    ['', { view: 'explore', params: {} }],
    ['#', { view: 'explore', params: {} }],
    ['#explore', { view: 'explore', params: {} }],
    ['#feedback', { view: 'feedback', params: {} }],
    ['#allpvs', { view: 'allpvs', params: {} }],
    ['#/allpvs', { view: 'allpvs', params: {} }],
    ['#/allpvs?q=beam%20current', { view: 'allpvs', params: { q: 'beam current' } }],
    ['#/allpvs?q=', { view: 'allpvs', params: { q: '' } }],
    ['#/allpvs?other=1&q=a', { view: 'allpvs', params: { q: 'a' } }],
    ['#/allpvs?ioc=IOC1', { view: 'allpvs', params: { ioc: ['IOC1'] } }],
    ['#/allpvs?ioc=demo-ioc-bpm', { view: 'allpvs', params: { ioc: ['demo-ioc-bpm'] } }],
    [
      '#/allpvs?q=a&ioc=I1&ioc=I2&origin=curated&record_type=ai',
      {
        view: 'allpvs',
        params: { q: 'a', ioc: ['I1', 'I2'], origin: ['curated'], record_type: ['ai'] },
      },
    ],
    ['#/allpvs?other=1&name=x', { view: 'allpvs', params: {} }],
    ['#/allpvs?name=x&page=2', { view: 'allpvs', params: {} }],
    ['#/pv/SR01%3AX', { view: 'pv', params: { name: 'SR01:X' } }],
    ['#/ioc/ioc-sr01', { view: 'ioc', params: { name: 'ioc-sr01' } }],
    ['#nonsense', { view: 'nonsense', params: {} }],
  ])('%s', (hash, route) => {
    expect(parseRoute(hash)).toEqual(route);
  });

  test.each([
    '#/pv',
    '#/pv/',
    '#pv',
    '#/ioc',
    '#/pv/a/b',
    '#/explore/x',
    '#/pv/%zz',
    '#/allpvs?q=%zz',
    '#/allpvs?ioc=%zz',
  ])('%s has no route', (hash) => {
    expect(parseRoute(hash)).toBeNull();
  });

  test.each(['SR01C:BPM1:X.VAL', 'a/b', 'odd name+1%', 'ÄÖ?&=#'])(
    'the name %s round-trips as one encoded segment',
    (name) => {
      expect(parseRoute(`#/pv/${encodeURIComponent(name)}`)).toEqual({
        view: 'pv',
        params: { name },
      });
      expect(parseRoute(`#/ioc/${encodeURIComponent(name)}`)).toEqual({
        view: 'ioc',
        params: { name },
      });
    }
  );

  test.each(['beam current', 'a+b', 'x&q=y', '100%', 'ÄÖ?#'])(
    'the words %s round-trip through q, with %%20 for a space',
    (q) => {
      expect(encodeURIComponent('a b')).toBe('a%20b');
      expect(parseRoute(`#/allpvs?q=${encodeURIComponent(q)}`)).toEqual({
        view: 'allpvs',
        params: { q },
      });
    }
  );

  test.each(['a+b', 'x&ioc=y', 'ÄÖ'])(
    'the filter value %s round-trips, encoded the same way',
    (value) => {
      const encoded = encodeURIComponent(value);
      expect(parseRoute(`#/allpvs?ioc=${encoded}&origin=${encoded}`)).toEqual({
        view: 'allpvs',
        params: { ioc: [value], origin: [value] },
      });
    }
  );

  test.each(FACET_KEYS)('the filter key %s parses to a list', (key) => {
    expect(parseRoute(`#/allpvs?${key}=v`)).toEqual({ view: 'allpvs', params: { [key]: ['v'] } });
  });
});

/** @type {[import('../../../src/osprey/interfaces/channel_finder/static/js/route.js').Route, string][]} */
const FORMATS = [
  [{ view: 'pv', params: { name: 'SR01:X' } }, '#/pv/SR01%3AX'],
  [{ view: 'ioc', params: { name: 'ioc-sr01' } }, '#/ioc/ioc-sr01'],
  [{ view: 'allpvs', params: {} }, '#/allpvs'],
  [{ view: 'allpvs', params: { q: '' } }, '#/allpvs?q='],
  [{ view: 'allpvs', params: { q: 'beam current' } }, '#/allpvs?q=beam%20current'],
  [{ view: 'allpvs', params: { ioc: ['demo-ioc-bpm'] } }, '#/allpvs?ioc=demo-ioc-bpm'],
  [
    {
      view: 'allpvs',
      params: { q: 'a', origin: ['curated'], ioc: ['I1', 'I2'], record_type: ['ai'] },
    },
    '#/allpvs?q=a&origin=curated&ioc=I1&ioc=I2&record_type=ai',
  ],
  [{ view: 'feedback', params: {} }, '#/feedback'],
];

describe('formatRoute', () => {
  test.each(FORMATS)('%j is %s', (route, hash) => {
    expect(formatRoute(route)).toBe(hash);
  });

  test.each(FORMATS)('%j: the hash %s parses back and formats to itself', (route, hash) => {
    expect(parseRoute(hash)).toEqual(route);
    expect(formatRoute(/** @type {any} */ (parseRoute(hash)))).toBe(hash);
  });

  test.each(['beam current', 'a+b', 'x&q=y', '100%', 'ÄÖ?#', ''])(
    'the words %j round-trip',
    (q) => {
      const route = { view: 'allpvs', params: { q } };
      expect(parseRoute(formatRoute(route))).toEqual(route);
    }
  );

  test.each(
    FACET_KEYS.flatMap((key) => [
      [key, ['a+b', 'x&ioc=y', 'ÄÖ']],
      [key, ['I1', 'I2']],
    ])
  )('the filter %s with %j round-trips', (key, values) => {
    const route = { view: 'allpvs', params: { [key]: values } };
    expect(parseRoute(formatRoute(route))).toEqual(route);
  });

  test.each(['SR01C:BPM1:X.VAL', 'a/b', 'odd name+1%', 'ÄÖ?&=#'])(
    'the name %s round-trips on pv and ioc',
    (name) => {
      for (const view of ['pv', 'ioc']) {
        const route = { view, params: { name } };
        expect(parseRoute(formatRoute(route))).toEqual(route);
      }
    }
  );

  test.each(['explore', 'feedback'])('the view %s round-trips', (view) => {
    const route = { view, params: {} };
    expect(parseRoute(formatRoute(route))).toEqual(route);
  });

  test('a pv route without a name throws', () => {
    expect(() => formatRoute({ view: 'pv', params: {} })).toThrow(TypeError);
  });
});

describe('allPvsHash', () => {
  test('trimmed words become q, blank words no params', () => {
    expect(allPvsHash(' BPM01 ')).toBe('#/allpvs?q=BPM01');
    expect(allPvsHash('  ')).toBe('#/allpvs');
  });
});

describe('familyOf', () => {
  test.each([
    ['pv', 'allpvs'],
    ['ioc', 'allpvs'],
    ['allpvs', 'allpvs'],
    ['explore', 'explore'],
    ['feedback', 'feedback'],
  ])('%s belongs to %s', (view, family) => {
    expect(familyOf(view)).toBe(family);
  });
});
