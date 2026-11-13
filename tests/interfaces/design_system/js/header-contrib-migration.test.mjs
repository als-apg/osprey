/**
 * Protocol v3 guard (design-system header-contrib.js): a first-party panel sends
 * `icon`, `menu` or one of the legacy kinds the hub still renders, and nothing
 * else. `text` is retired: the bar shows the tile's name and no subtitle.
 *
 * The check reads each file, drops its comments and scans the item literals
 * (`kind: '…'`), a `showToast(` line aside. A kind built from a variable is not
 * caught: this is a guard against drift, not a parser. Run:
 *   npx vitest run tests/interfaces/design_system/js/header-contrib-migration.test.mjs
 */
import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { test, expect, describe } from 'vitest';

const ROOT = resolve(import.meta.dirname, '../../../..');

const V3_KINDS = ['icon', 'menu'];
const LEGACY_KINDS = ['nav', 'button', 'search', 'crumb', 'sep'];
/** Hub-only: `contributeLocal` carries it, a panel's message cannot. */
const HUB_ONLY_KINDS = ['count'];

const ALLOWED = new Set([...V3_KINDS, ...LEGACY_KINDS, ...HUB_ONLY_KINDS]);

/**
 * Every first-party contribution source: each panel's own sender, and the
 * hub-side senders for the tiles whose header the hub owns.
 */
const SOURCES = [
  'src/osprey/interfaces/channel_finder/static/js/app.js',
  'src/osprey/interfaces/channel_finder/static/js/graph-finder-verbs.js',
  'src/osprey/interfaces/okf_panel/static/js/app.js',
  'src/osprey/interfaces/bluesky_web/panels/bluesky/panel.js',
  'src/osprey/dispatch/dashboard.html',
  'src/osprey/interfaces/web_terminal/static/js/artifacts-pane-header.js',
  'src/osprey/interfaces/web_terminal/static/js/jupyter-chrome.js',
];

/**
 * Sources that may still send `text`: each lost it in the change that drops the
 * subtitle (5e02527cde). Once that is in this tree, this list goes.
 */
const TEXT_UNTIL_SUBTITLE_REMOVAL = new Set([
  'src/osprey/dispatch/dashboard.html',
  'src/osprey/interfaces/web_terminal/static/js/artifacts-pane-header.js',
]);

/**
 * Not guarded, with the work that puts them on the standard. An entry must still
 * need its exemption: once the file sends only v3 kinds the test asks for it to go.
 * @type {Record<string, string>}
 */
const EXEMPT = {
  'src/osprey/interfaces/ariel/static/js/header.js': 'the Logbook moves to its own toolbar',
};

/** @param {string} source @returns {string} the source without its comments */
function withoutComments(source) {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:'"`])\/\/[^\n]*/g, '$1');
}

/**
 * A toast's options name a `kind` too (design-system toast.js): that is the
 * toast's severity, shown in the panel's own page, and never reaches the shell.
 * @param {string} source @returns {string} the source without its `showToast(` lines
 */
function withoutToasts(source) {
  return source
    .split('\n')
    .filter((line) => !line.includes('showToast('))
    .join('\n');
}

/** @param {string} relPath @returns {string[]} the item kinds the file names, in order */
function kindsIn(relPath) {
  const text = withoutToasts(withoutComments(readFileSync(join(ROOT, relPath), 'utf8')));
  return [...text.matchAll(/\bkind:\s*['"]([a-z-]+)['"]/g)].map((m) => m[1]);
}

describe('first-party panels send only the kinds the protocol allows', () => {
  test.each(SOURCES)('%s', (relPath) => {
    expect(existsSync(join(ROOT, relPath)), `${relPath} moved: update SOURCES`).toBe(true);
    const allowed = TEXT_UNTIL_SUBTITLE_REMOVAL.has(relPath)
      ? new Set([...ALLOWED, 'text'])
      : ALLOWED;
    expect(kindsIn(relPath).filter((kind) => !allowed.has(kind))).toEqual([]);
  });

  test.each(Object.entries(EXEMPT))('%s is exempt until %s, and still needs it', (relPath) => {
    expect(existsSync(join(ROOT, relPath)), `${relPath} moved: update EXEMPT`).toBe(true);
    const beyondV3 = kindsIn(relPath).filter((kind) => !V3_KINDS.includes(kind));
    expect(beyondV3.length, `${relPath} sends only v3 kinds now: remove it`).toBeGreaterThan(0);
  });

  test('no source is both required and exempt', () => {
    expect(SOURCES.filter((relPath) => relPath in EXEMPT)).toEqual([]);
  });
});
