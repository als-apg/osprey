// @ts-check
// @vitest-environment node
/**
 * OSPREY's dock policy (dock-policy.js): a tile holds ONE panel.
 *   npx vitest run tests/interfaces/web_terminal/dock-policy.test.mjs
 *
 * `crowdedTile` names the first tile, in tree order, that holds several
 * panels: the tile the layout holder refuses an operation's result for, and
 * an arrangement is refused for, by the name the server uses too.
 * `onePanelPerTile` migrates a layout stored by the release that had tabs:
 * each tile keeps the panel it showed, the others leave the dock and are
 * listed in tree order, and a layout that already keeps the rule comes back
 * as the same object.
 */

import { describe, expect, test } from 'vitest';

import {
  ONE_PANEL_PER_TILE,
  crowdedTile,
  onePanelPerTile,
} from '../../../src/osprey/interfaces/web_terminal/static/js/dock-policy.js';
import { validateLayout } from '/static/vendor/dock-engine.mjs';

/** @typedef {import('/static/vendor/dock-engine.mjs').Layout} Layout */

/** Every tile holds one panel: a column of two services beside the terminal. */
/** @type {Layout} */
const KEPT = {
  split: 'row',
  items: [
    {
      split: 'column',
      size: 0.6,
      items: [
        { tile: 't1', tabs: ['plots'], showing: 'plots', size: 0.5 },
        { tile: 't3', tabs: ['logbook'], showing: 'logbook', size: 0.5, from: 't1' },
      ],
    },
    { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
  ],
};

/**
 * A dock stored by the release that had tabs. Two tiles break the rule: t3,
 * nested in the column, and t2, a direct child of the root. In tree order t3
 * comes first; a breadth-first walk would meet t2 first. Neither shows its
 * first tab.
 * @type {Layout}
 */
const STORED = {
  split: 'row',
  items: [
    {
      split: 'column',
      size: 0.6,
      items: [
        { tile: 't1', tabs: ['plots'], showing: 'plots', size: 0.5 },
        {
          tile: 't3',
          tabs: ['logbook', 'alarms'],
          showing: 'alarms',
          size: 0.5,
          from: 't1',
        },
      ],
    },
    { tile: 't2', tabs: ['ariel', 'terminal', 'okf'], showing: 'terminal', size: 0.4 },
  ],
};

describe('crowdedTile', () => {
  test('null when every tile holds one panel, and for an empty dock', () => {
    expect(crowdedTile(KEPT)).toBeNull();
    expect(crowdedTile({ tile: 't1', tabs: ['terminal'], showing: 'terminal' })).toBeNull();
    expect(crowdedTile(null)).toBeNull();
  });

  test('the first tile holding several panels, in tree order', () => {
    expect(crowdedTile(STORED)).toMatchObject({ tile: 't3', tabs: ['logbook', 'alarms'] });
  });

  test('a root tile holding several panels is itself the crowded tile', () => {
    const root = { tile: 't1', tabs: ['plots', 'logbook'], showing: 'plots' };
    expect(crowdedTile(root)).toBe(root);
  });

  test('the refusal is named as the server names it', () => {
    expect(ONE_PANEL_PER_TILE).toBe('one_panel_per_tile');
  });
});

describe('onePanelPerTile — the migration of a dock stored with tabs', () => {
  test('each tile keeps the panel it showed; the panels that left are listed in tree order', () => {
    const out = onePanelPerTile(STORED);
    expect(out.layout).toEqual({
      split: 'row',
      items: [
        {
          split: 'column',
          size: 0.6,
          items: [
            { tile: 't1', tabs: ['plots'], showing: 'plots', size: 0.5 },
            { tile: 't3', tabs: ['alarms'], showing: 'alarms', size: 0.5, from: 't1' },
          ],
        },
        { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
      ],
    });
    expect(out.evicted).toEqual(['logbook', 'ariel', 'okf']);
    expect(crowdedTile(out.layout)).toBeNull();
    expect(validateLayout(out.layout, { form: 'dock' }).valid).toBe(true);
  });

  test('a layout that keeps the rule comes back as the same object, nothing evicted', () => {
    const out = onePanelPerTile(KEPT);
    expect(out.layout).toBe(KEPT);
    expect(out.evicted).toEqual([]);
  });

  test('the input is never mutated', () => {
    const input = structuredClone(STORED);
    onePanelPerTile(input);
    expect(input).toEqual(STORED);
  });

  test('an empty dock stays empty', () => {
    expect(onePanelPerTile(null)).toEqual({ layout: null, evicted: [] });
  });

  test('a root tile keeps the panel it showed', () => {
    const root = { tile: 't1', tabs: ['plots', 'logbook', 'alarms'], showing: 'logbook' };
    expect(onePanelPerTile(root)).toEqual({
      layout: { tile: 't1', tabs: ['logbook'], showing: 'logbook' },
      evicted: ['plots', 'alarms'],
    });
  });
});
