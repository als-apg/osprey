/**
 * The bar item catalog's closed sets:
 *   npx vitest run tests/interfaces/web_terminal/js/bar-catalog.test.js
 *
 * bar-catalog.js is the single declaration the hosts, the layout model and the
 * customize UI all read. The facts that are load-bearing far from the file
 * that states them are pinned here as EXACT sets rather than membership
 * checks, so both directions of drift fail:
 *
 *   - no placement axis — every type may sit in either bar, so no entry may
 *     grow a `hosts` list that would quietly refuse one of them;
 *   - foldable — a truthy `overflowLabel(ctx)` IS the ladder's fold domain, so
 *     an accidental label on a chrome item silently makes it foldable;
 *   - single-node and baseline types, and the gated types with the fact each
 *     one asks.
 *
 * The type list itself is pinned against the server's copy by
 * test_bar_items_ssr.py, which is where a new type has to be declared too.
 */

import { test, expect, describe } from 'vitest';

import {
  BAR_CATALOG,
  barItemType,
  defaultOptions,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/bar-catalog.js';

/** @typedef {import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-catalog.js').BarItemType} BarItemType */

/** Every entry, as [type, entry] pairs. */
const entries = () => Object.entries(BAR_CATALOG);

/**
 * Types whose entries satisfy `predicate`, sorted for set comparison.
 * @param {(entry: BarItemType) => boolean} predicate
 * @returns {string[]}
 */
const typesWhere = (predicate) =>
  entries()
    .filter(([, entry]) => predicate(entry))
    .map(([type]) => type)
    .sort();

/**
 * @param {readonly string[]} list
 * @returns {string[]}
 */
const sorted = (list) => [...list].sort();

describe('type set', () => {
  test('exactly the types with a JS-built or empty body may be placed twice', () => {
    // Everything else is one server-rendered node or one id-owning dot; a
    // second shell for it could only ever be empty.
    expect(typesWhere((entry) => entry.multi)).toEqual(
      sorted(['clock', 'stopwatch', 'space', 'separator'])
    );
  });

  test('barItemType resolves known types and returns null for unknown ones', () => {
    expect(barItemType('clock')).toBe(BAR_CATALOG.clock);
    expect(barItemType('no-such-item')).toBeNull();
    // Inherited object members must not resolve as types.
    expect(barItemType('toString')).toBeNull();
    expect(barItemType('constructor')).toBeNull();
  });
});

describe('placement', () => {
  test('no entry declares a placement axis — every type may sit in either bar', () => {
    // The axis used to exist and refused five types from the status bar; an
    // entry that grows one back would refuse a bar the operator was promised.
    for (const [, entry] of entries()) {
      expect('hosts' in entry).toBe(false);
      expect('densities' in entry).toBe(false);
    }
  });

});

describe('foldable set', () => {
  const FOLDABLE = ['bluesky-queue', 'clock', 'docs', 'feedback', 'stopwatch', 'system-health'];

  test('exactly six types return an overflow label', () => {
    // Truthiness, as the ladder reads it: an empty label never folds, so it
    // must not count as foldable here either.
    expect(typesWhere((entry) => Boolean(entry.overflowLabel({})))).toEqual(sorted(FOLDABLE));
  });

});

describe('available', () => {
  test.each([
    ['identity', 'identityAvailable'],
    ['system-health', 'systemHealthAvailable'],
    ['bluesky-queue', 'blueskyAvailable'],
  ])('%s is offered exactly where %s holds', (type, fact) => {
    const entry = BAR_CATALOG[type];
    expect(entry.available({})).toBe(false);
    expect(entry.available({ [fact]: false })).toBe(false);
    expect(entry.available({ [fact]: true })).toBe(true);
  });

  test('every other type is available on a bare deployment', () => {
    const gated = ['identity', 'bluesky-queue', 'system-health'];
    for (const [type, entry] of entries()) {
      if (gated.includes(type)) continue;
      expect(entry.available({})).toBe(true);
    }
  });
});

describe('align', () => {
  test('exactly logo and identity share a baseline run', () => {
    expect(typesWhere((entry) => entry.align === 'baseline')).toEqual(sorted(['logo', 'identity']));
  });

});

describe('flex hints', () => {
  test('a space at width 0 fills, and at any other width holds that width', () => {
    expect(BAR_CATALOG.space.flex({})).toEqual({ flex: '1 1 0' });
    expect(BAR_CATALOG.space.flex({ width: 0 })).toEqual({ flex: '1 1 0' });
    // Held, but shrinking before any real item is touched.
    expect(BAR_CATALOG.space.flex({ width: 400 })).toEqual({ flex: '0 1 400px', minWidth: '0' });
    // A non-numeric stored value falls back to the declared default.
    expect(BAR_CATALOG.space.flex({ width: 'wide' })).toEqual({ flex: '1 1 0' });
  });

  test('every other type stamps nothing on its shell', () => {
    const stamped = ['space'];
    for (const [type, entry] of entries()) {
      if (stamped.includes(type)) continue;
      expect(entry.flex({})).toBeNull();
    }
  });
});

describe('option spec', () => {
  test('every option spec declares a default of its own scalar kind', () => {
    for (const [, entry] of entries()) {
      for (const spec of Object.values(entry.options)) {
        if (spec.kind === 'number') {
          expect(typeof spec.default).toBe('number');
          expect(spec.min).toBeLessThan(spec.max);
        } else if (spec.kind === 'boolean') {
          expect(typeof spec.default).toBe('boolean');
        } else {
          expect(spec.kind).toBe('enum');
          expect(spec.values).toContain(spec.default);
        }
      }
    }
  });

  test('defaultOptions returns a fresh, mutable object of the declared defaults', () => {
    expect(defaultOptions('space')).toEqual({ width: 0 });
    expect(defaultOptions('clock')).toEqual({ zone: 'none', format: '24h', seconds: false });
    expect(defaultOptions('logo')).toEqual({});
    expect(defaultOptions('no-such-item')).toEqual({});
    const first = defaultOptions('space');
    first.width = 99;
    expect(defaultOptions('space').width).toBe(0);
  });
});
