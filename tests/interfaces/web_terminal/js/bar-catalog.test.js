/**
 * The bar item catalog's closed sets:
 *   npx vitest run tests/interfaces/web_terminal/js/bar-catalog.test.js
 *
 * bar-catalog.js is the single declaration the hosts, the layout model and the
 * customize UI all read. Four of its facts are load-bearing far from the file
 * that states them, and each is pinned here as an EXACT set rather than a
 * membership check, so both directions of drift fail:
 *
 *   - the 13 types themselves;
 *   - no placement axis — every type may sit in either bar, so no entry may
 *     grow a `hosts` list that would quietly refuse one of them;
 *   - foldable — `overflowLabel(ctx) !== null` IS the ladder's fold domain, so
 *     an accidental label on a chrome item silently makes it foldable.
 *
 */

import { test, expect, describe } from 'vitest';

import {
  BAR_CATALOG,
  BAR_GROUPS,
  BAR_HOSTS,
  BAR_ITEM_TYPES,
  DENSITY_BY_HOST,
  barItemType,
  defaultOptions,
  densityForHost,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/bar-catalog.js';

const EXPECTED_TYPES = [
  'logo',
  'identity',
  'control-target',
  'search',
  'display',
  'docs',
  'feedback',
  'clock',
  'system-health',
  'bluesky-queue',
  'stopwatch',
  'space',
  'separator',
];

/** @typedef {import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-catalog.js').BarItemType} BarItemType */
/** @typedef {import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-catalog.js').BarOptionSpec} BarOptionSpec */

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

/**
 * Narrow an option spec to its number member, failing the test rather than the
 * type-checker when a spec changes kind.
 * @param {BarOptionSpec} spec
 * @returns {Extract<BarOptionSpec, {kind: 'number'}>}
 */
function asNumberSpec(spec) {
  if (spec.kind !== 'number') throw new Error(`expected a number option, got "${spec.kind}"`);
  return spec;
}

/**
 * @param {BarOptionSpec} spec
 * @returns {Extract<BarOptionSpec, {kind: 'enum'}>}
 */
function asEnumSpec(spec) {
  if (spec.kind !== 'enum') throw new Error(`expected an enum option, got "${spec.kind}"`);
  return spec;
}

describe('type set', () => {
  test('declares exactly 13 types', () => {
    expect(BAR_ITEM_TYPES).toHaveLength(13);
    expect(sorted(BAR_ITEM_TYPES)).toEqual(sorted(EXPECTED_TYPES));
  });

  test('every type files under one of the sheet headings', () => {
    for (const [, entry] of entries()) expect(BAR_GROUPS).toContain(entry.group);
    expect(Object.isFrozen(BAR_GROUPS)).toBe(true);
  });

  test('exactly the types with a JS-built or empty body may be placed twice', () => {
    // Everything else is one server-rendered node or one id-owning dot; a
    // second shell for it could only ever be empty.
    expect(typesWhere((entry) => entry.multi)).toEqual(
      sorted(['clock', 'stopwatch', 'space', 'separator'])
    );
  });

  test('every entry states its own type as its key', () => {
    for (const [type, entry] of entries()) expect(entry.type).toBe(type);
  });

  test('every entry carries a non-empty label', () => {
    for (const [, entry] of entries()) expect(entry.label.length).toBeGreaterThan(0);
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

  test('each host renders at exactly one density', () => {
    expect(Object.keys(DENSITY_BY_HOST).sort()).toEqual(sorted(BAR_HOSTS));
    expect(densityForHost('header')).toBe('comfortable');
    expect(densityForHost('status')).toBe('compact');
  });
});

describe('foldable set', () => {
  const FOLDABLE = ['bluesky-queue', 'clock', 'docs', 'feedback', 'stopwatch', 'system-health'];

  test('exactly six types return an overflow label', () => {
    expect(typesWhere((entry) => entry.overflowLabel({}) !== null)).toEqual(sorted(FOLDABLE));
  });

  test('foldable labels are non-empty strings', () => {
    for (const type of FOLDABLE) {
      const label = BAR_CATALOG[type].overflowLabel({});
      expect(typeof label).toBe('string');
      expect(label ?? '').not.toBe('');
    }
  });

  test('search never folds — it collapses to its magnifier and stops there', () => {
    expect(BAR_CATALOG.search.overflowLabel({})).toBeNull();
  });

  test('spacing never folds — it yields through flex-shrink instead', () => {
    for (const type of ['space', 'separator']) {
      expect(BAR_CATALOG[type].overflowLabel({})).toBeNull();
    }
  });
});

describe('available', () => {
  test('identity is absent on a deployment with no identity block', () => {
    expect(BAR_CATALOG.identity.available({})).toBe(false);
    expect(BAR_CATALOG.identity.available({ identityAvailable: true })).toBe(true);
  });

  test('system-health is absent where the SYSTEM panel is not enabled', () => {
    expect(BAR_CATALOG['system-health'].available({})).toBe(false);
    expect(BAR_CATALOG['system-health'].available({ systemHealthAvailable: false })).toBe(false);
    expect(BAR_CATALOG['system-health'].available({ systemHealthAvailable: true })).toBe(true);
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

  test('every other type is centred', () => {
    for (const [type, entry] of entries()) {
      if (type === 'logo' || type === 'identity') continue;
      expect(entry.align).toBe('center');
    }
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
  test('space width is 0 (fill) to 2000 px', () => {
    const spec = asNumberSpec(BAR_CATALOG.space.options.width);
    expect(spec.min).toBe(0);
    expect(spec.max).toBe(2000);
    expect(spec.unit).toBe('px');
    expect(spec.default).toBe(0);
  });

  test('clock offers none/local/UTC/both, 24h/12h, plus seconds', () => {
    const zone = asEnumSpec(BAR_CATALOG.clock.options.zone);
    expect(zone.values).toEqual(['none', 'local', 'utc', 'both']);
    // The plain clock is the default: no zone suffix until one is asked for.
    expect(zone.default).toBe('none');
    const format = asEnumSpec(BAR_CATALOG.clock.options.format);
    expect(format.values).toEqual(['24h', '12h']);
    expect(format.default).toBe('24h');
    expect(BAR_CATALOG.clock.options.seconds.kind).toBe('boolean');
  });

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

  test('only clock, space, system health and the plan queue take options', () => {
    expect(typesWhere((entry) => Object.keys(entry.options).length > 0)).toEqual(
      sorted(['bluesky-queue', 'clock', 'space', 'system-health'])
    );
  });

  test('system health is quiet by default: a dot alone, one row per category', () => {
    const text = asEnumSpec(BAR_CATALOG['system-health'].options.text);
    expect(text.values).toEqual(['none', 'status']);
    expect(text.default).toBe('none');
    const detail = asEnumSpec(BAR_CATALOG['system-health'].options.detail);
    expect(detail.values).toEqual(['categories', 'checks']);
    expect(detail.default).toBe('categories');
  });

  test('the plan queue offers its controls in three steps, quiet by default', () => {
    const controls = asEnumSpec(BAR_CATALOG['bluesky-queue'].options.controls);
    expect(controls.values).toEqual(['none', 'stop', 'full']);
    expect(controls.default).toBe('none');
    expect(BAR_CATALOG['bluesky-queue'].options.progress).toEqual({ kind: 'boolean', default: true });
    expect(BAR_CATALOG['bluesky-queue'].options.count).toEqual({ kind: 'boolean', default: true });
  });

  test('the plan queue is offered exactly where the Bluesky panel is declared', () => {
    const entry = BAR_CATALOG['bluesky-queue'];
    expect(entry.available({ blueskyAvailable: true })).toBe(true);
    expect(entry.available({ blueskyAvailable: false })).toBe(false);
    expect(entry.available({})).toBe(false);
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

describe('priority', () => {
  test('every type declares a finite priority', () => {
    for (const [, entry] of entries()) expect(Number.isFinite(entry.priority)).toBe(true);
  });

  test('every foldable type folds before any never-folding one', () => {
    const foldable = entries().filter(([, entry]) => entry.overflowLabel({}) !== null);
    const fixed = entries().filter(([, entry]) => entry.overflowLabel({}) === null);
    const highestFoldable = Math.max(...foldable.map(([, entry]) => entry.priority));
    const lowestFixed = Math.min(...fixed.map(([, entry]) => entry.priority));
    expect(highestFoldable).toBeLessThan(lowestFixed);
  });
});
