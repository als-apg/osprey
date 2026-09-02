/**
 * The bar item catalog's closed sets:
 *   npx vitest run tests/interfaces/web_terminal/js/bar-catalog.test.js
 *
 * bar-catalog.js is the single declaration the hosts, the layout model and the
 * customize UI all read. Four of its facts are load-bearing far from the file
 * that states them, and each is pinned here as an EXACT set rather than a
 * membership check, so both directions of drift fail:
 *
 *   - the 17 types themselves;
 *   - `hosts` header-only — a status-bar shell is 24–26 px tall, and an item
 *     whose body cannot render there must be refused as a hard capability, not
 *     merely defaulted away from the footer;
 *   - `locked` — exactly the four ids a deployment may never let a user remove;
 *   - foldable — `overflowLabel(ctx) !== null` IS the ladder's fold domain, so
 *     an accidental label on a locked item silently makes it foldable.
 *
 * Plus the panel-health status-bar id set, which is derived from the shipped
 * panel catalog: this suite asserts the derivation rather than a retyped list,
 * because a panel gaining or losing a `statusBarId` must move the item's set
 * with it and must never leave a dot the deployment cannot fill.
 */

import { test, expect, describe } from 'vitest';

import {
  BAR_CATALOG,
  BAR_HOSTS,
  BAR_ITEM_TYPES,
  DENSITY_BY_HOST,
  LOCKED_BAR_ITEM_TYPES,
  PANEL_HEALTH_STATUS_BAR_IDS,
  barItemType,
  defaultOptions,
  densityForHost,
  supportsHost,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/bar-catalog.js';
import { PANELS } from '../../../../src/osprey/interfaces/web_terminal/static/js/panel-catalog.js';

const EXPECTED_TYPES = [
  'logo',
  'identity',
  'control-target',
  'search',
  'display',
  'docs',
  'feedback',
  'activity',
  'clock',
  'connection',
  'terminal-size',
  'panel-health',
  'bluesky-queue',
  'stopwatch',
  'space',
  'gap',
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
  test('declares exactly 17 types', () => {
    expect(BAR_ITEM_TYPES).toHaveLength(17);
    expect(sorted(BAR_ITEM_TYPES)).toEqual(sorted(EXPECTED_TYPES));
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

describe('hosts', () => {
  const HEADER_ONLY = ['logo', 'identity', 'control-target', 'search', 'display'];

  test('exactly five types are header-only', () => {
    const headerOnly = typesWhere(
      (entry) => entry.hosts.includes('header') && !entry.hosts.includes('status')
    );
    expect(headerOnly).toEqual(sorted(HEADER_ONLY));
  });

  test('every other type is placeable in both bars', () => {
    for (const [type, entry] of entries()) {
      if (HEADER_ONLY.includes(type)) continue;
      expect(sorted(entry.hosts)).toEqual(sorted(BAR_HOSTS));
    }
  });

  test('no type declares a host outside the two bars, or none at all', () => {
    for (const [, entry] of entries()) {
      expect(entry.hosts.length).toBeGreaterThan(0);
      for (const host of entry.hosts) expect(BAR_HOSTS).toContain(host);
    }
  });

  test('supportsHost refuses header-only types in the status bar', () => {
    expect(supportsHost('logo', 'header')).toBe(true);
    expect(supportsHost('logo', 'status')).toBe(false);
    expect(supportsHost('clock', 'status')).toBe(true);
    expect(supportsHost('no-such-item', 'header')).toBe(false);
  });

  test('declared densities are exactly the densities of the declared hosts', () => {
    for (const [, entry] of entries()) {
      const fromHosts = entry.hosts.map((host) => DENSITY_BY_HOST[host]);
      expect(sorted(entry.densities)).toEqual(sorted(fromHosts));
    }
    expect(densityForHost('header')).toBe('comfortable');
    expect(densityForHost('status')).toBe('compact');
  });
});

describe('locked', () => {
  test('exactly four types are locked', () => {
    expect(typesWhere((entry) => entry.locked)).toEqual(
      sorted(['logo', 'identity', 'control-target', 'display'])
    );
    expect(sorted(LOCKED_BAR_ITEM_TYPES)).toEqual(
      sorted(['logo', 'identity', 'control-target', 'display'])
    );
  });

  test('no locked type folds — locked chrome must stay visible in the bar', () => {
    for (const type of LOCKED_BAR_ITEM_TYPES) {
      expect(BAR_CATALOG[type].overflowLabel({})).toBeNull();
    }
  });
});

describe('foldable set', () => {
  const FOLDABLE = ['clock', 'terminal-size', 'docs', 'feedback', 'stopwatch', 'bluesky-queue'];

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
    for (const type of ['space', 'gap', 'separator']) {
      expect(BAR_CATALOG[type].overflowLabel({})).toBeNull();
    }
  });
});

describe('panel-health status bar ids', () => {
  test('the set is exactly the built-in panels that declare a statusBarId', () => {
    const declared = PANELS.map((panel) => panel.statusBarId).filter((id) => id !== null);
    expect(sorted(PANEL_HEALTH_STATUS_BAR_IDS)).toEqual(sorted(declared));
    expect(PANEL_HEALTH_STATUS_BAR_IDS.length).toBeGreaterThan(0);
  });

  test('the set is closed — no id from outside the panel catalog', () => {
    // #ws-dot and #term-dims are catalog items of their own now, and
    // #operator-status is dead markup; none is a panel-health dot.
    for (const stray of ['ws-dot', 'term-dims', 'operator-status', 'status-clock']) {
      expect(PANEL_HEALTH_STATUS_BAR_IDS).not.toContain(stray);
    }
  });

  test('panel-health is unavailable when no enabled panel declares an id', () => {
    expect(BAR_CATALOG['panel-health'].available({ statusBarIds: [] })).toBe(false);
    expect(BAR_CATALOG['panel-health'].available({})).toBe(false);
    expect(BAR_CATALOG['panel-health'].available({ statusBarIds: ['operator-status'] })).toBe(
      false
    );
    expect(
      BAR_CATALOG['panel-health'].available({ statusBarIds: [PANEL_HEALTH_STATUS_BAR_IDS[0]] })
    ).toBe(true);
  });
});

describe('available', () => {
  test('identity is absent on a deployment with no identity block', () => {
    expect(BAR_CATALOG.identity.available({})).toBe(false);
    expect(BAR_CATALOG.identity.available({ identityAvailable: true })).toBe(true);
  });

  test('bluesky-queue is absent without a bluesky bridge', () => {
    expect(BAR_CATALOG['bluesky-queue'].available({})).toBe(false);
    expect(BAR_CATALOG['bluesky-queue'].available({ blueskyAvailable: true })).toBe(true);
  });

  test('every other type is available on a bare deployment', () => {
    const gated = ['identity', 'bluesky-queue', 'panel-health'];
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
  test('activity absorbs the spare space and may ellipsize', () => {
    expect(BAR_CATALOG.activity.flex({})).toEqual({ flex: '1 1 0', minWidth: '0' });
  });

  test('space grows by its share', () => {
    expect(BAR_CATALOG.space.flex({})).toEqual({ flex: '1 1 0' });
    expect(BAR_CATALOG.space.flex({ share: 3 })).toEqual({ flex: '3 1 0' });
    // A non-numeric stored value falls back to the declared default.
    expect(BAR_CATALOG.space.flex({ share: 'wide' })).toEqual({ flex: '1 1 0' });
  });

  test('gap holds its size but shrinks before any real item is touched', () => {
    expect(BAR_CATALOG.gap.flex({})).toEqual({ flex: '0 1 12px', minWidth: '0' });
    expect(BAR_CATALOG.gap.flex({ size: 400 })).toEqual({ flex: '0 1 400px', minWidth: '0' });
  });

  test('every other type stamps nothing on its shell', () => {
    const stamped = ['activity', 'space', 'gap'];
    for (const [type, entry] of entries()) {
      if (stamped.includes(type)) continue;
      expect(entry.flex({})).toBeNull();
    }
  });
});

describe('option spec', () => {
  test('gap size is bounded 4–400 px', () => {
    const spec = asNumberSpec(BAR_CATALOG.gap.options.size);
    expect(spec.min).toBe(4);
    expect(spec.max).toBe(400);
    expect(spec.unit).toBe('px');
    expect(spec.default).toBeGreaterThanOrEqual(spec.min);
    expect(spec.default).toBeLessThanOrEqual(spec.max);
  });

  test('space share is bounded 1–3', () => {
    const spec = asNumberSpec(BAR_CATALOG.space.options.share);
    expect(spec.min).toBe(1);
    expect(spec.max).toBe(3);
  });

  test('clock offers local/UTC/both plus seconds', () => {
    const zone = asEnumSpec(BAR_CATALOG.clock.options.zone);
    expect(zone.values).toEqual(['local', 'utc', 'both']);
    expect(zone.values).toContain(zone.default);
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

  test('only clock, space and gap take options', () => {
    expect(typesWhere((entry) => Object.keys(entry.options).length > 0)).toEqual(
      sorted(['clock', 'space', 'gap'])
    );
  });

  test('defaultOptions returns a fresh, mutable object of the declared defaults', () => {
    expect(defaultOptions('gap')).toEqual({ size: 12 });
    expect(defaultOptions('clock')).toEqual({ zone: 'local', seconds: false });
    expect(defaultOptions('logo')).toEqual({});
    expect(defaultOptions('no-such-item')).toEqual({});
    const first = defaultOptions('gap');
    first.size = 99;
    expect(defaultOptions('gap').size).toBe(12);
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
