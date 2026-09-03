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
 * A second copy of a single-node type is dropped WITHOUT going read-only: it
 * never rendered anything, so removing it loses nothing, and a document that
 * carries one must stay writable or the operator could never fix it.
 */

import { test, expect, describe } from 'vitest';

import { BAR_CATALOG } from '../../../../src/osprey/interfaces/web_terminal/static/js/bar-catalog.js';
import {
  BAR_LAYOUT_VERSION,
  MAX_ITEMS_PER_HOST,
  emptyLayout,
  normalize,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/bar-layout.js';

/** @typedef {import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-catalog.js').BarItemType} BarItemType */
/** @typedef {import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-layout.js').BarLayoutItem} BarLayoutItem */
/** @typedef {import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-layout.js').BarLayoutContext} BarLayoutContext */

/**
 * A layout document, with the boilerplate fields defaulted.
 * @param {Partial<{version: unknown, rev: unknown, header: unknown,
 *                  status: unknown, header_visible: unknown,
 *                  status_visible: unknown}>} [fields]
 * @returns {Record<string, unknown>}
 */
function doc(fields = {}) {
  return {
    version: 1,
    rev: 0,
    header: [],
    status: [],
    header_visible: true,
    status_visible: true,
    ...fields,
  };
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
      header_visible: true,
      status_visible: true,
    });
    expect(Object.isFrozen(layout)).toBe(true);
    expect(Object.isFrozen(layout.header)).toBe(true);
  });
});

describe('a clean document', () => {
  const clean = doc({
    rev: 7,
    header: [item('logo'), item('control-target'), item('feedback'), item('docs')],
    status: [item('stopwatch'), item('clock', { zone: 'utc', format: '24h', seconds: true })],
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
        { type: 'feedback', options: {} },
        { type: 'docs', options: {} },
      ],
      status: [
        { type: 'stopwatch', options: {} },
        { type: 'clock', options: { zone: 'utc', format: '24h', seconds: true } },
      ],
      header_visible: true,
      status_visible: false,
    });
  });

  test('keeps the stored order — order IS the layout', () => {
    const reversed = doc({ header: [item('docs'), item('feedback'), item('logo')] });
    const result = normalize(reversed, BAR_CATALOG, {});
    expect(types(result.layout.header)).toEqual(['docs', 'feedback', 'logo']);
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

  test('every known type may be placed in the status bar', () => {
    // The wordmark, the search trigger and the display menu were once refused
    // here as `host-mismatch`; that reason no longer exists.
    const result = normalize(
      doc({ status: [item('stopwatch'), item('logo'), item('search'), item('clock')] }),
      BAR_CATALOG,
      {}
    );
    expect(types(result.layout.status)).toEqual(['stopwatch', 'logo', 'search', 'clock']);
    expect(result.dropped).toEqual([]);
    expect(result.readonly).toBe(false);
  });

  test('and in the header', () => {
    const result = normalize(doc({ header: [item('logo'), item('search')] }), BAR_CATALOG, {});
    expect(types(result.layout.header)).toEqual(['logo', 'search']);
    expect(result.readonly).toBe(false);
  });

  test('an item this deployment cannot render is dropped from a stored document', () => {
    // rev 3: an operator saved this, so the drop is THEIR content going missing.
    const layout = doc({
      rev: 3,
      header: [item('identity'), item('system-health'), item('docs')],
    });

    const bare = normalize(layout, BAR_CATALOG, {});
    expect(types(bare.layout.header)).toEqual(['docs']);
    expect(bare.dropped.map((drop) => drop.reason)).toEqual(['unavailable', 'unavailable']);
    expect(bare.readonly).toBe(true);

    const equipped = normalize(layout, BAR_CATALOG, {
      identityAvailable: true,
      systemHealthAvailable: true,
    });
    expect(types(equipped.layout.header)).toEqual(['identity', 'system-health', 'docs']);
    expect(equipped.readonly).toBe(false);
  });

  test('the same drop from a rev-0 document is not a loss', () => {
    // rev 0 is the deployment default: nobody authored it and the server
    // already declined to paint the item, so dropping it here takes nothing
    // away from anyone. The drop is still reported, and the document is still
    // `changed`; it just does not latch (#863).
    const layout = doc({ header: [item('identity'), item('system-health'), item('docs')] });

    const result = normalize(layout, BAR_CATALOG, {});
    expect(types(result.layout.header)).toEqual(['docs']);
    expect(result.dropped.map((drop) => drop.reason)).toEqual(['unavailable', 'unavailable']);
    expect(result.changed).toBe(true);
    expect(result.readonly).toBe(false);
  });

  test('a rev-0 document still latches on an unknown type', () => {
    const result = normalize(
      doc({ header: [item('identity'), item('quantum-flux'), item('docs')] }),
      BAR_CATALOG,
      {}
    );
    expect(result.dropped.map((drop) => drop.reason)).toEqual(['unavailable', 'unknown-type']);
    expect(result.readonly).toBe(true);
  });

  test('a rev-0 document still latches on an unreadable version', () => {
    const result = normalize(doc({ version: 99, header: [item('logo')] }), BAR_CATALOG, {});
    expect(result.readonly).toBe(true);
  });

  test('a rev that is not a number reads as unsaved, and its unavailable drops do not latch', () => {
    const result = normalize(
      doc({ rev: 'seven', header: [item('identity'), item('docs')] }),
      BAR_CATALOG,
      {}
    );
    expect(result.layout.rev).toBe(0);
    expect(result.dropped.map((drop) => drop.reason)).toEqual(['unavailable']);
    expect(result.readonly).toBe(false);
  });

  test('a second copy of a single-node type is dropped, and that is NOT a loss', () => {
    const result = normalize(
      doc({ header: [item('logo'), item('docs')], status: [item('docs'), item('clock')] }),
      BAR_CATALOG,
      {}
    );
    expect(types(result.layout.header)).toEqual(['logo', 'docs']);
    expect(types(result.layout.status)).toEqual(['clock']);
    expect(result.dropped).toEqual([{ host: 'status', index: 0, type: 'docs', reason: 'duplicate' }]);
    expect(result.changed).toBe(true);
    expect(result.readonly).toBe(false);
  });

  test('a type the catalog marks multi may be placed as often as the cap allows', () => {
    const result = normalize(
      doc({ header: [item('clock'), item('separator'), item('clock')], status: [item('separator')] }),
      BAR_CATALOG,
      {}
    );
    expect(types(result.layout.header)).toEqual(['clock', 'separator', 'clock']);
    expect(types(result.layout.status)).toEqual(['separator']);
    expect(result.dropped).toEqual([]);
  });

  test('availability is asked per host — the host is part of the context', () => {
    // system-health reads whether the SYSTEM panel is enabled, not the
    // placement, but the host must still reach `available()` so a
    // host-sensitive type can use it.
    const result = normalize(doc({ status: [item('system-health')] }), BAR_CATALOG, {
      systemHealthAvailable: true,
    });
    expect(types(result.layout.status)).toEqual(['system-health']);
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
  const gaps = (count) => Array.from({ length: count }, () => item('separator'));

  test('the 21st item in a host is dropped', () => {
    const result = normalize(doc({ status: gaps(21) }), BAR_CATALOG, {});
    expect(result.layout.status).toHaveLength(MAX_ITEMS_PER_HOST);
    expect(result.dropped).toEqual([
      { host: 'status', index: 20, type: 'separator', reason: 'overflow' },
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
    header: [item('logo'), item('control-target'), item('feedback')],
    status: [item('stopwatch')],
  });

  test('falls back to the deployment default and goes read-only', () => {
    const result = normalize(future, BAR_CATALOG, { defaultLayout: deploymentDefault });
    expect(result.layout.version).toBe(BAR_LAYOUT_VERSION);
    expect(types(result.layout.header)).toEqual(['logo', 'control-target', 'feedback']);
    expect(types(result.layout.status)).toEqual(['stopwatch']);
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
    expect(result.layout.status[0].options).toEqual({
      zone: 'none',
      format: '24h',
      seconds: false,
    });
    expect(result.changed).toBe(true);
    expect(result.readonly).toBe(false);
  });

  test('an out-of-bounds number is clamped, and that IS a loss', () => {
    const result = normalize(doc({ status: [item('space', { width: 9000 })] }), BAR_CATALOG, {});
    expect(result.layout.status[0].options).toEqual({ width: 2000 });
    expect(result.readonly).toBe(true);
  });

  test('a value of the wrong kind falls back to the default and goes read-only', () => {
    const result = normalize(
      doc({ status: [item('clock', { zone: 'mars', seconds: 'yes' })] }),
      BAR_CATALOG,
      {}
    );
    expect(result.layout.status[0].options).toEqual({
      zone: 'none',
      format: '24h',
      seconds: false,
    });
    expect(result.readonly).toBe(true);
  });

  test('an unknown option key is discarded and goes read-only', () => {
    const result = normalize(
      doc({ status: [item('space', { width: 12, colour: 'red' })] }),
      BAR_CATALOG,
      {}
    );
    expect(result.layout.status[0].options).toEqual({ width: 12 });
    expect(result.readonly).toBe(true);
  });

  test('a type with no options keeps none, whatever was stored', () => {
    const result = normalize(doc({ status: [item('stopwatch', { size: 4 })] }), BAR_CATALOG, {});
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

  test('header_visible follows the same rules, on its own', () => {
    const nonsense = normalize(doc({ header_visible: 'yes' }), BAR_CATALOG, {});
    expect(nonsense.layout.header_visible).toBe(true);
    expect(nonsense.changed).toBe(true);
    expect(nonsense.readonly).toBe(false);

    const hidden = normalize(
      doc({ header: [item('logo')], header_visible: false }),
      BAR_CATALOG,
      {}
    );
    expect(hidden.layout.header_visible).toBe(false);
    expect(hidden.layout.status_visible).toBe(true);
    expect(types(hidden.layout.header)).toEqual(['logo']);
    expect(hidden.changed).toBe(false);
  });

  test('a hidden status bar keeps its items', () => {
    const result = normalize(
      doc({ status: [item('stopwatch')], status_visible: false }),
      BAR_CATALOG,
      {}
    );
    expect(result.layout.status_visible).toBe(false);
    expect(types(result.layout.status)).toEqual(['stopwatch']);
    expect(result.changed).toBe(false);
  });
});

describe('the catalog is a parameter', () => {
  /**
   * @param {string} type
   * @returns {BarItemType}
   */
  const fixtureType = (type) => ({
    type,
    label: type,
    group: 'System',
    multi: false,
    options: {},
    priority: 10,
    align: 'center',
    flex: () => null,
    overflowLabel: () => null,
    available: () => true,
  });

  test('validation follows the catalog it was handed, not a shipped one', () => {
    const fixture = { widget: fixtureType('widget') };
    const result = normalize(
      doc({ header: [item('widget'), item('docs')], status: [item('widget')] }),
      fixture,
      {}
    );
    // `docs` is real, but not in THIS catalog; `widget` is not real, but is —
    // once, because the fixture type is single-node.
    expect(types(result.layout.header)).toEqual(['widget']);
    expect(result.layout.status).toEqual([]);
    expect(result.dropped.map((drop) => drop.reason)).toEqual(['unknown-type', 'duplicate']);
  });
});
