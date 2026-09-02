/**
 * The bar layout model — schema v1 and `normalize()`:
 *   npx vitest run tests/interfaces/web_terminal/js/bar-layout.test.js
 *
 * `normalize()` is the boundary every untrusted layout crosses: a document
 * saved by a NEWER deployment, a hand-edited config block, a half-written file.
 * It never throws and always returns a renderable document, so the only thing
 * standing between a silent drop and a DESTROYED user layout is the flag it
 * returns with it. That is what this suite pins hardest:
 *
 *   - `readonly` — content was dropped, a stored value was overwritten, or the
 *     version was unknown. The client renders what it got and issues ZERO PUTs
 *     until an explicit reset (FR5). A rollback to a build with fewer item
 *     types must not let the next write truncate the good document.
 *   - `changed` — the weaker question: is this identical to the input. True for
 *     lossless completions too (an option arriving with its declared default),
 *     which is exactly why it may NOT be the flag that gates writes.
 *
 * Plus the four drop rules — unknown type, host mismatch, unavailable,
 * per-host overflow — and the clean round-trip that proves normalization is not
 * quietly rewriting documents that were already fine.
 *
 * `merge()` is pinned on the other half of that contract: a LOCKED item is one
 * the user was never allowed to remove, so a document missing one is repaired
 * from the deployment default rather than honoured. What must not happen is a
 * repair that duplicates an item the user merely MOVED, or one that keeps
 * growing across repeated merges — both would corrupt a layout silently.
 *
 * `PRESETS` is pinned on one property that makes it safe to hand to a user:
 * every preset normalizes CLEAN. A preset that arrived `changed` would make a
 * fresh document look edited; one that arrived `readonly` could not be applied.
 */

import { test, expect, describe } from 'vitest';

import {
  BAR_CATALOG,
  LOCKED_BAR_ITEM_TYPES,
  PANEL_HEALTH_STATUS_BAR_IDS,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/bar-catalog.js';
import {
  BAR_LAYOUT_VERSION,
  MAX_ITEMS_PER_HOST,
  PRESETS,
  emptyLayout,
  merge,
  normalize,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/bar-layout.js';

/** @typedef {import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-catalog.js').BarItemType} BarItemType */
/** @typedef {import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-layout.js').BarLayoutItem} BarLayoutItem */
/** @typedef {import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-layout.js').BarLayout} BarLayout */
/** @typedef {import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-layout.js').BarLayoutContext} BarLayoutContext */

/**
 * A layout document, with the boilerplate fields defaulted.
 * @param {Partial<{version: unknown, rev: unknown, header: unknown,
 *                  status: unknown, status_visible: unknown}>} [fields]
 * @returns {Record<string, unknown>}
 */
function doc(fields = {}) {
  return { version: 1, rev: 0, header: [], status: [], status_visible: true, ...fields };
}

/**
 * @param {string} type
 * @param {Record<string, unknown>} [options]
 * @returns {Record<string, unknown>}
 */
function item(type, options = {}) {
  return { type, options };
}

/**
 * @param {readonly BarLayoutItem[]} items
 * @returns {string[]}
 */
const types = (items) => items.map((entry) => entry.type);

/**
 * A deployment that renders everything the catalog declares. Presets and merge
 * fixtures are read in this context: an availability drop would otherwise say
 * more about the fixture deployment than about the document under test.
 * @type {BarLayoutContext}
 */
const EQUIPPED = {
  identityAvailable: true,
  blueskyAvailable: true,
  statusBarIds: PANEL_HEALTH_STATUS_BAR_IDS,
};

/**
 * A NORMALIZED document — the only kind `merge()` is given.
 * @param {Partial<{rev: unknown, header: unknown, status: unknown,
 *                  status_visible: unknown}>} [fields]
 * @returns {BarLayout}
 */
const layoutOf = (fields = {}) => normalize(doc(fields), BAR_CATALOG, EQUIPPED).layout;

describe('schema', () => {
  test('declares v1 and a per-host cap of 20', () => {
    expect(BAR_LAYOUT_VERSION).toBe(1);
    expect(MAX_ITEMS_PER_HOST).toBe(20);
  });

  test('the empty layout is a valid, frozen v1 document', () => {
    const layout = emptyLayout();
    expect(layout).toEqual({
      version: 1,
      rev: 0,
      header: [],
      status: [],
      status_visible: true,
    });
    expect(Object.isFrozen(layout)).toBe(true);
    expect(Object.isFrozen(layout.header)).toBe(true);
  });
});

describe('a clean document', () => {
  const clean = doc({
    rev: 7,
    header: [item('logo'), item('control-target'), item('activity'), item('docs')],
    status: [item('connection'), item('clock', { zone: 'utc', seconds: true })],
    status_visible: false,
  });

  test('round-trips unchanged and is flagged clean', () => {
    const result = normalize(clean, BAR_CATALOG, {});
    expect(result.changed).toBe(false);
    expect(result.readonly).toBe(false);
    expect(result.dropped).toEqual([]);
    expect(result.layout).toEqual({
      version: 1,
      rev: 7,
      header: [
        { type: 'logo', options: {} },
        { type: 'control-target', options: {} },
        { type: 'activity', options: {} },
        { type: 'docs', options: {} },
      ],
      status: [
        { type: 'connection', options: {} },
        { type: 'clock', options: { zone: 'utc', seconds: true } },
      ],
      status_visible: false,
    });
  });

  test('keeps the stored order — order IS the layout', () => {
    const reversed = doc({ header: [item('docs'), item('activity'), item('logo')] });
    const result = normalize(reversed, BAR_CATALOG, {});
    expect(types(result.layout.header)).toEqual(['docs', 'activity', 'logo']);
    expect(result.readonly).toBe(false);
  });

  test('is returned frozen, and the input document is not mutated', () => {
    const input = doc({ header: [item('logo')] });
    const before = structuredClone(input);
    const result = normalize(input, BAR_CATALOG, {});
    expect(input).toEqual(before);
    expect(Object.isFrozen(result.layout)).toBe(true);
    expect(Object.isFrozen(result.layout.header[0])).toBe(true);
    expect(Object.isFrozen(result.layout.header[0].options)).toBe(true);
  });
});

describe('dropping', () => {
  test('an unknown type is dropped and the document goes read-only', () => {
    const result = normalize(
      doc({ header: [item('logo'), item('quantum-flux'), item('docs')] }),
      BAR_CATALOG,
      {}
    );
    expect(types(result.layout.header)).toEqual(['logo', 'docs']);
    expect(result.dropped).toEqual([
      { host: 'header', index: 1, type: 'quantum-flux', reason: 'unknown-type' },
    ]);
    expect(result.changed).toBe(true);
    expect(result.readonly).toBe(true);
  });

  test('an inherited object member is not an item type', () => {
    const inherited = doc({ header: [item('toString'), item('constructor')] });
    const result = normalize(inherited, BAR_CATALOG, {});
    expect(result.layout.header).toEqual([]);
    expect(result.dropped.map((drop) => drop.reason)).toEqual(['unknown-type', 'unknown-type']);
  });

  test('a header-only type placed in the status bar is dropped', () => {
    const result = normalize(
      doc({ status: [item('connection'), item('logo'), item('search'), item('clock')] }),
      BAR_CATALOG,
      {}
    );
    expect(types(result.layout.status)).toEqual(['connection', 'clock']);
    expect(result.dropped).toEqual([
      { host: 'status', index: 1, type: 'logo', reason: 'host-mismatch' },
      { host: 'status', index: 2, type: 'search', reason: 'host-mismatch' },
    ]);
    expect(result.readonly).toBe(true);
  });

  test('the same type survives in the host that can render it', () => {
    const result = normalize(doc({ header: [item('logo'), item('search')] }), BAR_CATALOG, {});
    expect(types(result.layout.header)).toEqual(['logo', 'search']);
    expect(result.readonly).toBe(false);
  });

  test('an item this deployment cannot render is dropped', () => {
    const layout = doc({ header: [item('identity'), item('bluesky-queue'), item('docs')] });

    const bare = normalize(layout, BAR_CATALOG, {});
    expect(types(bare.layout.header)).toEqual(['docs']);
    expect(bare.dropped.map((drop) => drop.reason)).toEqual(['unavailable', 'unavailable']);
    expect(bare.readonly).toBe(true);

    const equipped = normalize(layout, BAR_CATALOG, {
      identityAvailable: true,
      blueskyAvailable: true,
    });
    expect(types(equipped.layout.header)).toEqual(['identity', 'bluesky-queue', 'docs']);
    expect(equipped.readonly).toBe(false);
  });

  test('availability is asked per host — the host is part of the context', () => {
    // panel-health reads the enabled built-in panels, not the placement, but the
    // host must still reach `available()` so a host-sensitive type can use it.
    const result = normalize(doc({ status: [item('panel-health')] }), BAR_CATALOG, {
      statusBarIds: [PANEL_HEALTH_STATUS_BAR_IDS[0]],
    });
    expect(types(result.layout.status)).toEqual(['panel-health']);
    expect(result.readonly).toBe(false);
  });

  test('a malformed entry is dropped without a type', () => {
    const result = normalize(
      doc({ header: [null, 'docs', {}, { type: 42 }, item('docs')] }),
      BAR_CATALOG,
      {}
    );
    expect(types(result.layout.header)).toEqual(['docs']);
    expect(result.dropped.map((drop) => drop.reason)).toEqual([
      'malformed',
      'malformed',
      'malformed',
      'malformed',
    ]);
    expect(result.dropped.every((drop) => drop.type === '')).toBe(true);
    expect(result.readonly).toBe(true);
  });

  test('a list that is not a list is an empty bar the client must not write back', () => {
    const result = normalize(doc({ header: [item('docs')], status: 'nope' }), BAR_CATALOG, {});
    expect(types(result.layout.header)).toEqual(['docs']);
    expect(result.layout.status).toEqual([]);
    expect(result.readonly).toBe(true);
  });
});

describe('the per-host cap', () => {
  /**
   * @param {number} count
   * @returns {Record<string, unknown>[]}
   */
  const gaps = (count) => Array.from({ length: count }, () => item('gap', { size: 12 }));

  test('the 21st item in a host is dropped', () => {
    const result = normalize(doc({ status: gaps(21) }), BAR_CATALOG, {});
    expect(result.layout.status).toHaveLength(MAX_ITEMS_PER_HOST);
    expect(result.dropped).toEqual([
      { host: 'status', index: 20, type: 'gap', reason: 'overflow' },
    ]);
    expect(result.readonly).toBe(true);
  });

  test('exactly 20 in a host is clean', () => {
    const result = normalize(doc({ status: gaps(20) }), BAR_CATALOG, {});
    expect(result.layout.status).toHaveLength(20);
    expect(result.changed).toBe(false);
    expect(result.readonly).toBe(false);
  });

  test('the cap is per host, not per document — 20 in each survive', () => {
    const result = normalize(doc({ header: gaps(20), status: gaps(20) }), BAR_CATALOG, {});
    expect(result.layout.header).toHaveLength(20);
    expect(result.layout.status).toHaveLength(20);
    expect(result.readonly).toBe(false);
  });

  test('a refused item never consumes a slot a good one could have used', () => {
    const crowded = [item('quantum-flux'), ...gaps(20)];
    const result = normalize(doc({ status: crowded }), BAR_CATALOG, {});
    expect(result.layout.status).toHaveLength(20);
    expect(result.dropped.map((drop) => drop.reason)).toEqual(['unknown-type']);
  });
});

describe('an unknown version', () => {
  const future = doc({
    version: 2,
    rev: 12,
    header: [item('docs'), item('hologram')],
    status_visible: false,
  });

  const deploymentDefault = doc({
    header: [item('logo'), item('control-target'), item('activity')],
    status: [item('connection')],
  });

  test('falls back to the deployment default and goes read-only', () => {
    const result = normalize(future, BAR_CATALOG, { defaultLayout: deploymentDefault });
    expect(result.layout.version).toBe(BAR_LAYOUT_VERSION);
    expect(types(result.layout.header)).toEqual(['logo', 'control-target', 'activity']);
    expect(types(result.layout.status)).toEqual(['connection']);
    expect(result.changed).toBe(true);
    expect(result.readonly).toBe(true);
  });

  test('carries none of the future document — not its items, rev or visibility', () => {
    const result = normalize(future, BAR_CATALOG, { defaultLayout: deploymentDefault });
    expect(types(result.layout.header)).not.toContain('hologram');
    expect(result.layout.rev).toBe(0);
    expect(result.layout.status_visible).toBe(true);
  });

  test('a missing version is an unknown version', () => {
    const result = normalize({ header: [item('docs')] }, BAR_CATALOG, {});
    expect(result.layout).toEqual(emptyLayout());
    expect(result.readonly).toBe(true);
  });

  test('with no deployment default, the fallback is the empty document', () => {
    const result = normalize(future, BAR_CATALOG, {});
    expect(result.layout).toEqual(emptyLayout());
    expect(result.readonly).toBe(true);
  });

  test('a default that itself carries an unrenderable item is cleaned, and still read-only', () => {
    const result = normalize(future, BAR_CATALOG, {
      defaultLayout: doc({ header: [item('identity'), item('docs')] }),
    });
    expect(types(result.layout.header)).toEqual(['docs']);
    expect(result.readonly).toBe(true);
  });

  test('anything that is not a document at all falls back the same way', () => {
    for (const raw of [null, undefined, 'layout', 42, [item('docs')]]) {
      const result = normalize(raw, BAR_CATALOG, {});
      expect(result.layout).toEqual(emptyLayout());
      expect(result.changed).toBe(true);
      expect(result.readonly).toBe(true);
    }
  });
});

describe('options', () => {
  test('a missing option arrives with its default — lossless, still writable', () => {
    const result = normalize(doc({ status: [{ type: 'clock' }] }), BAR_CATALOG, {});
    expect(result.layout.status[0].options).toEqual({ zone: 'local', seconds: false });
    expect(result.changed).toBe(true);
    expect(result.readonly).toBe(false);
  });

  test('an out-of-bounds number is clamped, and that IS a loss', () => {
    const result = normalize(doc({ status: [item('gap', { size: 900 })] }), BAR_CATALOG, {});
    expect(result.layout.status[0].options).toEqual({ size: 400 });
    expect(result.readonly).toBe(true);
  });

  test('a value of the wrong kind falls back to the default and goes read-only', () => {
    const result = normalize(
      doc({ status: [item('clock', { zone: 'mars', seconds: 'yes' })] }),
      BAR_CATALOG,
      {}
    );
    expect(result.layout.status[0].options).toEqual({ zone: 'local', seconds: false });
    expect(result.readonly).toBe(true);
  });

  test('an unknown option key is discarded and goes read-only', () => {
    const result = normalize(
      doc({ status: [item('gap', { size: 12, colour: 'red' })] }),
      BAR_CATALOG,
      {}
    );
    expect(result.layout.status[0].options).toEqual({ size: 12 });
    expect(result.readonly).toBe(true);
  });

  test('a type with no options keeps none, whatever was stored', () => {
    const result = normalize(doc({ status: [item('connection', { size: 4 })] }), BAR_CATALOG, {});
    expect(result.layout.status[0].options).toEqual({});
    expect(result.readonly).toBe(true);
  });
});

describe('document fields', () => {
  test('a nonsense rev reads as never-saved without making the document read-only', () => {
    for (const rev of [-1, 1.5, '3', null, Number.NaN]) {
      const result = normalize(doc({ rev }), BAR_CATALOG, {});
      expect(result.layout.rev).toBe(0);
      expect(result.changed).toBe(true);
      expect(result.readonly).toBe(false);
    }
  });

  test('a nonsense status_visible reads as visible, and the bar stays writable', () => {
    const result = normalize(doc({ status_visible: 'yes' }), BAR_CATALOG, {});
    expect(result.layout.status_visible).toBe(true);
    expect(result.changed).toBe(true);
    expect(result.readonly).toBe(false);
  });

  test('a hidden status bar keeps its items', () => {
    const result = normalize(
      doc({ status: [item('connection')], status_visible: false }),
      BAR_CATALOG,
      {}
    );
    expect(result.layout.status_visible).toBe(false);
    expect(types(result.layout.status)).toEqual(['connection']);
    expect(result.changed).toBe(false);
  });
});

describe('the catalog is a parameter', () => {
  /**
   * @param {string} type
   * @param {readonly ('header' | 'status')[]} hosts
   * @returns {BarItemType}
   */
  const fixtureType = (type, hosts) => ({
    type,
    label: type,
    hosts,
    densities: ['comfortable', 'compact'],
    options: {},
    priority: 10,
    locked: false,
    align: 'center',
    flex: () => null,
    overflowLabel: () => null,
    available: () => true,
  });

  test('validation follows the catalog it was handed, not a shipped one', () => {
    const fixture = { widget: fixtureType('widget', ['header']) };
    const result = normalize(
      doc({ header: [item('widget'), item('docs')], status: [item('widget')] }),
      fixture,
      {}
    );
    // `docs` is real, but not in THIS catalog; `widget` is not real, but is.
    expect(types(result.layout.header)).toEqual(['widget']);
    expect(result.layout.status).toEqual([]);
    expect(result.dropped.map((drop) => drop.reason)).toEqual(['unknown-type', 'host-mismatch']);
  });
});

/**
 * A deployment default with all four locked items in the header and a status
 * bar whose clock sits at index 3 — the index a repair has to reproduce.
 */
const DEPLOYMENT_DEFAULT = layoutOf({
  header: [item('logo'), item('identity'), item('control-target'), item('search'), item('display')],
  status: [item('connection'), item('activity'), item('docs'), item('clock')],
});

/** What the catalog locks, before any deployment adds to it. */
const LOCKED = LOCKED_BAR_ITEM_TYPES;

describe('merge', () => {
  test('a locked item the user removed comes back at its default index', () => {
    const user = layoutOf({
      header: [item('logo'), item('identity'), item('search'), item('display')],
    });
    const merged = merge(DEPLOYMENT_DEFAULT, user, LOCKED);
    expect(types(merged.header)).toEqual([
      'logo',
      'identity',
      'control-target',
      'search',
      'display',
    ]);
  });

  test('several missing locked items come back in default-relative order', () => {
    const user = layoutOf({ header: [item('search')] });
    const merged = merge(DEPLOYMENT_DEFAULT, user, LOCKED);
    expect(types(merged.header)).toEqual([
      'logo',
      'identity',
      'control-target',
      'search',
      'display',
    ]);
  });

  test("the user's own order and options survive the repair", () => {
    const user = layoutOf({
      header: [item('display'), item('control-target'), item('identity')],
      status: [item('clock', { zone: 'utc', seconds: true }), item('docs'), item('activity')],
    });
    const merged = merge(DEPLOYMENT_DEFAULT, user, [...LOCKED, 'clock']);
    // Only `logo` was missing; the four the user kept stay in the order they
    // were left in, and the clock keeps the zone the user chose.
    expect(types(merged.header)).toEqual(['logo', 'display', 'control-target', 'identity']);
    expect(types(merged.status)).toEqual(['clock', 'docs', 'activity']);
    expect(merged.status[0].options).toEqual({ zone: 'utc', seconds: true });
  });

  test('a type this deployment locked on top of the catalog is restored too', () => {
    const user = layoutOf({
      header: types(DEPLOYMENT_DEFAULT.header).map((type) => item(type)),
      status: [item('connection'), item('activity'), item('docs')],
    });
    expect(types(merge(DEPLOYMENT_DEFAULT, user, LOCKED).status)).toEqual([
      'connection',
      'activity',
      'docs',
    ]);
    expect(types(merge(DEPLOYMENT_DEFAULT, user, [...LOCKED, 'clock']).status)).toEqual([
      'connection',
      'activity',
      'docs',
      'clock',
    ]);
  });

  test('a locked item the user MOVED to the other bar is not duplicated', () => {
    const user = layoutOf({
      header: [...types(DEPLOYMENT_DEFAULT.header).map((type) => item(type)), item('clock')],
      status: [item('connection'), item('activity'), item('docs')],
    });
    const merged = merge(DEPLOYMENT_DEFAULT, user, [...LOCKED, 'clock']);
    const placed = [...types(merged.header), ...types(merged.status)].filter(
      (type) => type === 'clock'
    );
    expect(placed).toEqual(['clock']);
    expect(types(merged.header)).toContain('clock');
    expect(types(merged.status)).not.toContain('clock');
  });

  test('a locked type the default never places is not invented', () => {
    const user = layoutOf({ header: [item('search')], status: [item('activity')] });
    const merged = merge(DEPLOYMENT_DEFAULT, user, [...LOCKED, 'stopwatch']);
    expect(types(merged.header)).not.toContain('stopwatch');
    expect(types(merged.status)).not.toContain('stopwatch');
    expect(merged).toEqual(merge(DEPLOYMENT_DEFAULT, user, LOCKED));
  });

  test('merging an already-merged document changes nothing', () => {
    const user = layoutOf({ header: [item('search')], status: [item('docs')] });
    const once = merge(DEPLOYMENT_DEFAULT, user, [...LOCKED, 'clock']);
    const twice = merge(DEPLOYMENT_DEFAULT, once, [...LOCKED, 'clock']);
    expect(twice).toEqual(once);
  });

  test('with nothing locked, the user document comes back as it was', () => {
    const user = layoutOf({ header: [item('search')], status: [item('docs')] });
    const merged = merge(DEPLOYMENT_DEFAULT, user);
    expect(merged).toEqual(user);
    expect(merged).not.toBe(user);
  });

  test('rev and status_visible come from the user document, not the default', () => {
    const user = layoutOf({ rev: 9, header: [item('search')], status_visible: false });
    const merged = merge(DEPLOYMENT_DEFAULT, user, LOCKED);
    expect(merged.rev).toBe(9);
    expect(merged.status_visible).toBe(false);
    expect(DEPLOYMENT_DEFAULT.rev).toBe(0);
  });

  test('a repair never pushes a host past the cap, so the result stays writable', () => {
    const gaps = Array.from({ length: MAX_ITEMS_PER_HOST }, () => item('gap', { size: 12 }));
    const user = layoutOf({ header: [item('search')], status: gaps });
    const merged = merge(DEPLOYMENT_DEFAULT, user, [...LOCKED, 'clock']);
    expect(merged.status).toHaveLength(MAX_ITEMS_PER_HOST);
    expect(types(merged.status)[3]).toBe('clock');
    expect(normalize(merged, BAR_CATALOG, EQUIPPED).readonly).toBe(false);
  });

  test('the result is frozen and neither input is mutated', () => {
    const user = layoutOf({ header: [item('search')] });
    const before = { user: structuredClone(user), fallback: structuredClone(DEPLOYMENT_DEFAULT) };
    const merged = merge(DEPLOYMENT_DEFAULT, user, LOCKED);
    expect(user).toEqual(before.user);
    expect(DEPLOYMENT_DEFAULT).toEqual(before.fallback);
    expect(Object.isFrozen(merged)).toBe(true);
    expect(Object.isFrozen(merged.header)).toBe(true);
    expect(Object.isFrozen(merged.header[0])).toBe(true);
  });
});

describe('PRESETS', () => {
  test('every preset normalizes clean against the catalog', () => {
    for (const entry of PRESETS) {
      const result = normalize(entry.layout, BAR_CATALOG, EQUIPPED);
      expect(result.dropped, entry.id).toEqual([]);
      expect(result.changed, entry.id).toBe(false);
      expect(result.readonly, entry.id).toBe(false);
      expect(result.layout, entry.id).toEqual(entry.layout);
    }
  });

  test('every preset carries every locked type, so none depends on a repair', () => {
    for (const entry of PRESETS) {
      const placed = [...types(entry.layout.header), ...types(entry.layout.status)];
      for (const type of LOCKED) expect(placed, `${entry.id}/${type}`).toContain(type);
      expect(merge(DEPLOYMENT_DEFAULT, entry.layout, LOCKED), entry.id).toEqual(entry.layout);
    }
  });

  test('the ids are unique and the table is frozen', () => {
    const ids = PRESETS.map((entry) => entry.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toContain('clean');
    expect(Object.isFrozen(PRESETS)).toBe(true);
    expect(PRESETS.every((entry) => Object.isFrozen(entry) && Object.isFrozen(entry.layout))).toBe(
      true
    );
  });

  test('the clean preset is the status bar the deployment default describes', () => {
    const clean = PRESETS.find((entry) => entry.id === 'clean');
    expect(clean).toBeDefined();
    expect(types(clean?.layout.status ?? [])).toEqual([
      'activity',
      'panel-health',
      'docs',
      'clock',
    ]);
    expect(clean?.layout.status_visible).toBe(true);
  });

  test('a deployment that renders less drops only what it cannot render', () => {
    // Single-user, no bluesky bridge, no built-in panel with a status dot: the
    // preset still applies, minus the items this deployment has no body for.
    for (const entry of PRESETS) {
      const result = normalize(entry.layout, BAR_CATALOG, {});
      const reasons = new Set(result.dropped.map((drop) => drop.reason));
      expect([...reasons].filter((reason) => reason !== 'unavailable'), entry.id).toEqual([]);
      expect(types(result.layout.header), entry.id).toContain('logo');
    }
  });
});
