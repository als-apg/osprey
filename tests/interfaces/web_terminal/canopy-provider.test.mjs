/**
 * Unit tests for the command palette's CANOPY provider (canopy-provider.js).
 *
 *   npx vitest run tests/interfaces/web_terminal/canopy-provider.test.mjs
 *
 * The matcher port runs every case of the shared vectors file, a byte copy of
 * canopy's tests/canopy_mcp/fixtures/collections/match-vectors.json held at
 * tests/integrations/canopy/fixtures/match-vectors.json. A case whose kind is
 * neither `rank` nor `include` fails the run.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  CANOPY_CAP,
  COLLECTIONS_PATH,
  MANIFEST_PATH,
  createCanopyProvider,
  rankEntries,
  searchable,
  words,
} from '../../../src/osprey/interfaces/web_terminal/static/js/canopy-provider.js';

const VECTORS_PATH = join(
  import.meta.dirname,
  '../../integrations/canopy/fixtures/match-vectors.json'
);
const VECTORS = JSON.parse(readFileSync(VECTORS_PATH, 'utf8'));
/** @type {Array<Record<string, any>>} */
const CASES = VECTORS.cases;

/** @param {Record<string, any>} c */
function runCase(c) {
  switch (c.kind) {
    case 'rank':
      return rankEntries(c.candidates, c.query).map((e) => e.label);
    case 'include':
      return c.manifest
        .filter((/** @type {any} */ m) => searchable(m, { showSubScreens: c.showSubScreens }))
        .map((/** @type {any} */ m) => m.id);
    default:
      throw new Error(`unknown vector kind ${c.kind}`);
  }
}

describe('shared match vectors', () => {
  test('both kinds are present and every case is one of them', () => {
    const rank = CASES.filter((c) => c.kind === 'rank').length;
    const include = CASES.filter((c) => c.kind === 'include').length;
    expect(rank).toBeGreaterThan(0);
    expect(include).toBeGreaterThan(0);
    expect(rank + include).toBe(CASES.length);
  });

  test('an unknown kind fails', () => {
    expect(() => runCase({ name: 'x', kind: 'fuzzy', expect: [] })).toThrow(/unknown vector kind/);
  });

  test.each(CASES.map((c) => [c.name, c]))('%s', (_name, c) => {
    expect(runCase(c)).toEqual(c.expect);
  });

  test('words splits camelCase, acronyms and digits', () => {
    expect(words('GTLView')).toEqual(['gtl', 'view']);
    expect(words('SR01C')).toEqual(['sr', '01', 'c']);
    expect(words('BPMTable')).toEqual(['bpm', 'table']);
  });
});

/** @type {Record<string, unknown>} */
const MANIFEST = {
  screens: [
    { id: 'gtl/GTLView', category: 'library', kind: 'ir', title: 'GTLView', path: 'x' },
    { id: 'sr/embedded/row', category: 'library', kind: 'ir', title: 'GTL Row', path: 'x' },
    { id: 'tools/trace', category: '@tools', kind: 'ir', title: 'GTL Trace', path: 'x' },
    { id: 'sr/overview', category: 'library', kind: 'ir', title: 'SR Overview', path: 'x' },
    { id: 'bts/My View', category: 'drafts', kind: 'ir', title: 'BTS view', path: 'x' },
  ],
};

const COLLECTIONS = {
  collections: [
    {
      source: 'facility',
      ref: 'facility:injector',
      id: 'injector',
      name: 'Injector GTL',
      members: [{ screen: 'library/gtl/GTLView' }, { screen: 'library/sr/overview' }],
    },
  ],
};

/** @returns {Promise<void>} */
function flush() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * A provider over the given answers; `load` records each path read.
 * @param {{ manifest?: () => Promise<unknown>, collections?: () => Promise<unknown> }} [over]
 */
function makeProvider(over = {}) {
  const navigate = vi.fn();
  const load = vi.fn(async (/** @type {string} */ path) => {
    if (path === MANIFEST_PATH) return (over.manifest ?? (async () => MANIFEST))();
    if (path === COLLECTIONS_PATH) return (over.collections ?? (async () => COLLECTIONS))();
    throw new Error(`unexpected read ${path}`);
  });
  const provider = createCanopyProvider({ load, navigate });
  return { provider, navigate, load };
}

describe('provider', () => {
  beforeEach(() => {
    delete (/** @type {any} */ (window).__OSPREY_PREFIX__);
  });
  afterEach(() => {
    delete (/** @type {any} */ (window).__OSPREY_PREFIX__);
  });

  test('reads the manifest and the collections from the CANOPY panel', async () => {
    const { provider, load } = makeProvider();
    provider.start();
    await flush();
    expect(load.mock.calls.map((c) => c[0]).sort()).toEqual(
      ['/panel/canopy/drafts/manifest.json', '/panel/canopy/embed/collections'].sort()
    );
  });

  test('answers null while the read is in flight, then rows; onChange fires once', async () => {
    const { provider } = makeProvider();
    const onChange = vi.fn();
    provider.start(onChange);
    expect(provider.search('gtl')).toBeNull();
    await flush();
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(provider.search('gtl')).not.toBeNull();
  });

  test('ranks screens and collections; sub-screens and @ categories are left out', async () => {
    const { provider } = makeProvider();
    provider.start();
    await flush();
    const rows = /** @type {any[]} */ (provider.search('gtl'));
    expect(rows.map((r) => r.label)).toEqual([
      'Open screen GTLView',
      'Open collection Injector GTL (2)',
    ]);
    expect(rows[0].detail).toBe('library/gtl/GTLView');
  });

  test('a blank query yields no rows', async () => {
    const { provider } = makeProvider();
    provider.start();
    await flush();
    expect(provider.search('')).toEqual([]);
    expect(provider.search('   ')).toEqual([]);
  });

  test('a screen opens at its embed URL in the canopy panel', async () => {
    const { provider, navigate } = makeProvider();
    provider.start();
    await flush();
    /** @type {any[]} */ (provider.search('gtlview'))[0].run();
    expect(navigate).toHaveBeenCalledWith('canopy', '/panel/canopy/embed/library/gtl/GTLView');
  });

  test('a screen id with a space is encoded per segment', async () => {
    const { provider, navigate } = makeProvider();
    provider.start();
    await flush();
    /** @type {any[]} */ (provider.search('bts'))[0].run();
    expect(navigate).toHaveBeenCalledWith('canopy', '/panel/canopy/embed/drafts/bts/My%20View');
  });

  test('a collection opens through the _canopyOpen deep link', async () => {
    const { provider, navigate } = makeProvider();
    provider.start();
    await flush();
    const rows = /** @type {any[]} */ (provider.search('injector'));
    expect(rows.map((r) => r.label)).toEqual(['Open collection Injector GTL (2)']);
    rows[0].run();
    expect(navigate).toHaveBeenCalledWith(
      'canopy',
      '/panel/canopy/embed/?_canopyOpen=facility%3Ainjector'
    );
  });

  test('both URLs carry the per-user prefix', async () => {
    /** @type {any} */ (window).__OSPREY_PREFIX__ = '/u/alice';
    const { provider, navigate } = makeProvider();
    provider.start();
    await flush();
    /** @type {any[]} */ (provider.search('gtlview'))[0].run();
    /** @type {any[]} */ (provider.search('injector'))[0].run();
    expect(navigate.mock.calls).toEqual([
      ['canopy', '/u/alice/panel/canopy/embed/library/gtl/GTLView'],
      ['canopy', '/u/alice/panel/canopy/embed/?_canopyOpen=facility%3Ainjector'],
    ]);
  });

  test('a failed read contributes no rows; the other source still does', async () => {
    const { provider } = makeProvider({
      manifest: async () => {
        throw new Error('HTTP 404');
      },
    });
    provider.start();
    await flush();
    expect(/** @type {any[]} */ (provider.search('gtl')).map((r) => r.label)).toEqual([
      'Open collection Injector GTL (2)',
    ]);
  });

  test('both reads failing yields no rows, never null', async () => {
    const fail = async () => {
      throw new Error('down');
    };
    const { provider } = makeProvider({ manifest: fail, collections: fail });
    provider.start();
    await flush();
    expect(provider.search('gtl')).toEqual([]);
  });

  test('rows are capped', async () => {
    const screens = Array.from({ length: CANOPY_CAP + 4 }, (_, i) => ({
      id: `gtl/s${i}`,
      category: 'library',
      kind: 'ir',
      title: `GTL ${i}`,
    }));
    const { provider } = makeProvider({
      manifest: async () => ({ screens }),
      collections: async () => ({ collections: [] }),
    });
    provider.start();
    await flush();
    expect(/** @type {any[]} */ (provider.search('gtl')).length).toBe(CANOPY_CAP);
  });

  test('a read superseded by a newer start is dropped', async () => {
    /** @type {(v: unknown) => void} */
    let releaseOld = () => {};
    let call = 0;
    const { provider } = makeProvider({
      manifest: () => {
        call += 1;
        if (call === 1) return new Promise((r) => (releaseOld = r));
        return Promise.resolve({ screens: [] });
      },
    });
    const first = vi.fn();
    const second = vi.fn();
    provider.start(first);
    provider.start(second);
    await flush();
    expect(second).toHaveBeenCalledTimes(1);
    releaseOld(MANIFEST);
    await flush();
    expect(first).not.toHaveBeenCalled();
    // The newer (empty) manifest stands: no screen rows, only the collection.
    expect(/** @type {any[]} */ (provider.search('gtl')).map((r) => r.label)).toEqual([
      'Open collection Injector GTL (2)',
    ]);
  });
});
