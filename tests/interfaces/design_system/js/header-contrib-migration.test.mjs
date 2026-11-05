/**
 * Protocol v3 migration guard (design-system header-contrib.js, §3 of the
 * panel-chrome standard): a first-party panel sends ONLY `text`, `icon` and
 * `menu` through contributeHeader / contributeLocal. The legacy kinds `nav`,
 * `button`, `search`, `crumb` and `sep` stay rendered for CANOPY (external)
 * and for a panel not yet migrated, which must be listed in EXEMPT with the
 * follow-up that migrates it. A panel is either fully on the standard or
 * untouched: never half.
 *
 * The check reads each source file, drops its comments and scans the item
 * literals (`kind: '…'`). A kind built from a variable is not caught — this
 * is a guard against drift, not a parser. Run:
 *   npx vitest run tests/interfaces/design_system/js/header-contrib-migration.test.mjs
 */
import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { test, expect, describe } from 'vitest';

const ROOT = resolve(import.meta.dirname, '../../../..');

/** What a migrated panel may send. */
const V3_KINDS = new Set(['text', 'icon', 'menu']);

/**
 * Every first-party contribution source: each panel's own sender, and the
 * hub-side senders (contributeLocal) for the tiles whose header the hub owns.
 */
const SOURCES = [
  'src/osprey/interfaces/channel_finder/static/js/app.js',
  'src/osprey/interfaces/channel_finder/static/js/graph-finder-verbs.js',
  'src/osprey/interfaces/okf_panel/static/js/app.js',
  'src/osprey/interfaces/bluesky_web/panels/bluesky/panel.js',
  'src/osprey/dispatch/dashboard.html',
  'src/osprey/interfaces/health/static/js/chrome.js',
  'src/osprey/interfaces/web_terminal/static/js/artifacts-pane-header.js',
  'src/osprey/interfaces/web_terminal/static/js/jupyter-chrome.js',
];

/**
 * Sources still on the legacy kinds, each with the work that migrates it.
 * An entry here must still NEED its exemption: once the file sends only v3
 * kinds the test asks for the entry to go.
 * @type {Record<string, string>}
 */
const EXEMPT = {
  // The Logbook (#2391) landed after the standard was cut; its migration is
  // WP13 of the panel-chrome standard, a follow-up pull request.
  'src/osprey/interfaces/ariel/static/js/header.js': 'WP13: Logbook on the standard',
};

/** @param {string} source @returns {string} the source without its comments */
function withoutComments(source) {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:'"`])\/\/[^\n]*/g, '$1');
}

/** @param {string} relPath @returns {string[]} the item kinds the file names, in order */
function kindsIn(relPath) {
  const text = withoutComments(readFileSync(join(ROOT, relPath), 'utf8'));
  return [...text.matchAll(/\bkind:\s*['"]([a-z-]+)['"]/g)].map((m) => m[1]);
}

describe('first-party panels send only protocol v3 kinds', () => {
  test.each(SOURCES)('%s', (relPath) => {
    expect(existsSync(join(ROOT, relPath)), `${relPath} moved: update SOURCES`).toBe(true);
    const legacy = kindsIn(relPath).filter((kind) => !V3_KINDS.has(kind));
    expect(legacy, `${relPath} still sends legacy kinds`).toEqual([]);
  });

  test.each(Object.entries(EXEMPT))('%s is exempt until %s, and still needs it', (relPath) => {
    expect(existsSync(join(ROOT, relPath)), `${relPath} moved: update EXEMPT`).toBe(true);
    const legacy = kindsIn(relPath).filter((kind) => !V3_KINDS.has(kind));
    expect(
      legacy.length,
      `${relPath} sends only v3 kinds now: remove its exemption`
    ).toBeGreaterThan(0);
  });

  test('no source is both required and exempt', () => {
    expect(SOURCES.filter((relPath) => relPath in EXEMPT)).toEqual([]);
  });
});
