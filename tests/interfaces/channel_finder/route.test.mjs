// @ts-check
/**
 * The panel's route grammar (route.js): parseRoute and its inverse formatRoute.
 *   npx vitest run tests/interfaces/channel_finder/route.test.mjs
 */

import { test, expect, describe } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  FACET_KEYS,
  channelsHash,
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
    ['#/alarms', { view: 'alarms', params: {} }],
    ['#/alarms?severity=MAJOR', { view: 'alarms', params: {} }],
    ['#/archive', { view: 'archive', params: {} }],
    ['#channels', { view: 'channels', params: {} }],
    ['#/channels', { view: 'channels', params: {} }],
    ['#/channels?q=beam%20current', { view: 'channels', params: { q: 'beam current' } }],
    ['#/channels?q=', { view: 'channels', params: { q: '' } }],
    ['#/channels?other=1&q=a', { view: 'channels', params: { q: 'a' } }],
    ['#/channels?server=IOC1', { view: 'channels', params: { server: ['IOC1'] } }],
    ['#/channels?server=demo-ioc-bpm', { view: 'channels', params: { server: ['demo-ioc-bpm'] } }],
    [
      '#/channels?q=a&server=I1&server=I2&origin=curated&record_type=ai',
      {
        view: 'channels',
        params: { q: 'a', server: ['I1', 'I2'], origin: ['curated'], record_type: ['ai'] },
      },
    ],
    [
      '#/channels?alarm=MAJOR&server=I1&alarm=none',
      { view: 'channels', params: { server: ['I1'], alarm: ['MAJOR', 'none'] } },
    ],
    [
      '#/channels?server=sim-SR&archived=no',
      { view: 'channels', params: { server: ['sim-SR'], archived: ['no'] } },
    ],
    ['#/channels?other=1&name=x', { view: 'channels', params: {} }],
    ['#/channels?name=x&page=2', { view: 'channels', params: {} }],
    ['#/channel/SR01%3AX', { view: 'channel', params: { name: 'SR01:X' } }],
    ['#/server/ioc-sr01', { view: 'server', params: { name: 'ioc-sr01' } }],
    ['#nonsense', { view: 'nonsense', params: {} }],
  ])('%s', (hash, route) => {
    expect(parseRoute(hash)).toEqual(route);
  });

  test.each([
    '#/channel',
    '#/channel/',
    '#channel',
    '#/server',
    '#/channel/a/b',
    '#/explore/x',
    '#/channel/%zz',
    '#/channels?q=%zz',
    '#/channels?server=%zz',
  ])('%s has no route', (hash) => {
    expect(parseRoute(hash)).toBeNull();
  });

  test.each(['SR01C:BPM1:X.VAL', 'a/b', 'odd name+1%', 'ÄÖ?&=#'])(
    'the name %s round-trips as one encoded segment',
    (name) => {
      expect(parseRoute(`#/channel/${encodeURIComponent(name)}`)).toEqual({
        view: 'channel',
        params: { name },
      });
      expect(parseRoute(`#/server/${encodeURIComponent(name)}`)).toEqual({
        view: 'server',
        params: { name },
      });
    }
  );

  test.each(['beam current', 'a+b', 'x&q=y', '100%', 'ÄÖ?#'])(
    'the words %s round-trip through q, with %%20 for a space',
    (q) => {
      expect(encodeURIComponent('a b')).toBe('a%20b');
      expect(parseRoute(`#/channels?q=${encodeURIComponent(q)}`)).toEqual({
        view: 'channels',
        params: { q },
      });
    }
  );

  test.each(['a+b', 'x&server=y', 'ÄÖ'])(
    'the filter value %s round-trips, encoded the same way',
    (value) => {
      const encoded = encodeURIComponent(value);
      expect(parseRoute(`#/channels?server=${encoded}&origin=${encoded}`)).toEqual({
        view: 'channels',
        params: { server: [value], origin: [value] },
      });
    }
  );

  test.each(FACET_KEYS)('the filter key %s parses to a list', (key) => {
    expect(parseRoute(`#/channels?${key}=v`)).toEqual({
      view: 'channels',
      params: { [key]: ['v'] },
    });
  });
});

/**
 * The shared example links, also read by test_panel_routes.py against the server's link builder.
 * @type {[import('../../../src/osprey/interfaces/channel_finder/static/js/route.js').Route, string][]}
 */
const FORMATS = JSON.parse(
  readFileSync(join(import.meta.dirname, 'fixtures/panel_routes.json'), 'utf8')
).map((/** @type {{ route: any, hash: string }} */ entry) => [entry.route, entry.hash]);

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
      const route = { view: 'channels', params: { q } };
      expect(parseRoute(formatRoute(route))).toEqual(route);
    }
  );

  test.each(
    FACET_KEYS.flatMap((key) => [
      [key, ['a+b', 'x&server=y', 'ÄÖ']],
      [key, ['I1', 'I2']],
    ])
  )('the filter %s with %j round-trips', (key, values) => {
    const route = { view: 'channels', params: { [key]: values } };
    expect(parseRoute(formatRoute(route))).toEqual(route);
  });

  test.each(['SR01C:BPM1:X.VAL', 'a/b', 'odd name+1%', 'ÄÖ?&=#'])(
    'the name %s round-trips on pv and ioc',
    (name) => {
      for (const view of ['channel', 'server']) {
        const route = { view, params: { name } };
        expect(parseRoute(formatRoute(route))).toEqual(route);
      }
    }
  );

  test.each(['explore', 'feedback', 'alarms', 'archive'])('the view %s round-trips', (view) => {
    const route = { view, params: {} };
    expect(parseRoute(formatRoute(route))).toEqual(route);
  });

  test('a pv route without a name throws', () => {
    expect(() => formatRoute({ view: 'channel', params: {} })).toThrow(TypeError);
  });
});

describe('channelsHash', () => {
  test('trimmed words become q, blank words no params', () => {
    expect(channelsHash(' BPM01 ')).toBe('#/channels?q=BPM01');
    expect(channelsHash('  ')).toBe('#/channels');
  });
});

describe('familyOf', () => {
  test.each([
    ['channel', 'channels'],
    ['server', 'channels'],
    ['channels', 'channels'],
    ['explore', 'explore'],
    ['feedback', 'feedback'],
    ['alarms', 'alarms'],
    ['archive', 'archive'],
  ])('%s belongs to %s', (view, family) => {
    expect(familyOf(view)).toBe(family);
  });
});
