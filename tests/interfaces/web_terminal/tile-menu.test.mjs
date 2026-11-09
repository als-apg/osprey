/**
 * Contract tests for the tile ⋯ menu's shape (tile-menu.js): two sections,
 * the empty-section rule, `separator` dividers, danger rows last, and tidy
 * dividers. Pure logic; happy-dom not needed. Run:
 *   npx vitest run tests/interfaces/web_terminal/tile-menu.test.mjs
 */
import { test, expect, describe } from 'vitest';

import {
  AGENT_HEADING,
  composeTileMenu,
  THIS_PANEL_HEADING,
  TILE_HEADING,
} from '../../../src/osprey/interfaces/web_terminal/static/js/tile-menu.js';

const noop = () => {};

/** @param {import('/design-system/js/context-menu.js').MenuItem[]} rows */
const shape = (rows) =>
  rows.map((r) => (r.divider ? '—' : r.heading !== undefined ? `#${r.heading}` : r.label));

// The service tile bar's Tile section, as the policy hands it over: the
// destructive row already last behind its divider.
const TILE = [
  { label: 'Open in new window', run: noop },
  { label: 'Reload', run: noop },
  { label: 'Help', run: noop },
  { divider: true },
  { label: 'Close tile', glyph: '×', danger: true, run: noop },
];

describe('composeTileMenu', () => {
  test('with panel rows: "This panel" then "Tile", each under its heading', () => {
    const rows = composeTileMenu({
      thisPanel: [
        { label: 'Compact rows', checked: true, run: noop },
        { label: 'Export CSV', run: noop },
      ],
      tile: TILE,
    });
    expect(shape(rows)).toEqual([
      `#${THIS_PANEL_HEADING}`,
      'Compact rows',
      'Export CSV',
      `#${TILE_HEADING}`,
      'Open in new window',
      'Reload',
      'Help',
      '—',
      'Close tile',
    ]);
    // Rows pass through untouched: checked, run and the rest survive.
    expect(rows[1]).toMatchObject({ label: 'Compact rows', checked: true, run: noop });
  });

  test('a named heading captions the content section: the Agent pane reads "Agent"', () => {
    const rows = composeTileMenu({
      thisPanel: [{ label: 'Export conversation', run: noop }],
      tile: TILE,
      heading: AGENT_HEADING,
    });
    expect(AGENT_HEADING).toBe('Agent');
    expect(shape(rows).slice(0, 3)).toEqual(['#Agent', 'Export conversation', `#${TILE_HEADING}`]);
    expect(shape(composeTileMenu({ thisPanel: [], tile: TILE, heading: AGENT_HEADING }))).toEqual(
      shape(TILE)
    );
  });

  test('with nothing for the panel: the tile rows alone, no heading at all', () => {
    const rows = composeTileMenu({ thisPanel: [], tile: TILE });
    expect(rows).toEqual(TILE);
    expect(rows.some((r) => r.heading !== undefined)).toBe(false);
  });

  test('a row with `separator` gets a divider before it, and the flag does not reach the menu', () => {
    const rows = composeTileMenu({
      thisPanel: [
        { label: 'Density: comfortable', checked: true, run: noop },
        { label: 'Columns…', separator: true, run: noop },
      ],
      tile: TILE,
    });
    expect(shape(rows).slice(0, 4)).toEqual([
      `#${THIS_PANEL_HEADING}`,
      'Density: comfortable',
      '—',
      'Columns…',
    ]);
    expect(rows.some((r) => 'separator' in r)).toBe(false);
  });

  test('destructive rows go last, after one divider, in their own order', () => {
    const rows = composeTileMenu({
      thisPanel: [
        { label: 'Delete', danger: true, run: noop },
        { label: 'Pin', run: noop },
        { label: 'Clear filters', danger: true, separator: true, run: noop },
        { label: 'Print', separator: true, run: noop },
      ],
      tile: TILE,
    });
    expect(shape(rows).slice(0, 7)).toEqual([
      `#${THIS_PANEL_HEADING}`,
      'Pin',
      '—',
      'Print',
      '—',
      'Delete',
      'Clear filters',
    ]);
    expect(rows[5].danger).toBe(true);
    expect(rows[6].danger).toBe(true);
  });

  test('a section of only danger rows needs no leading divider', () => {
    const rows = composeTileMenu({
      thisPanel: [{ label: 'Delete', danger: true, run: noop }],
      tile: TILE,
    });
    expect(shape(rows).slice(0, 3)).toEqual([
      `#${THIS_PANEL_HEADING}`,
      'Delete',
      `#${TILE_HEADING}`,
    ]);
  });

  test('dividers that separate nothing are dropped: leading, doubled, trailing', () => {
    const rows = composeTileMenu({
      thisPanel: [
        { label: 'A', separator: true, run: noop },
        { label: 'B', separator: true, run: noop },
      ],
      tile: [
        { divider: true },
        { divider: true },
        { label: 'Reload', run: noop },
        { divider: true },
      ],
    });
    expect(shape(rows)).toEqual([
      `#${THIS_PANEL_HEADING}`,
      'A',
      '—',
      'B',
      `#${TILE_HEADING}`,
      'Reload',
    ]);
  });

  test('a destructive Tile row goes last, behind a divider, whatever order it came in', () => {
    const rows = composeTileMenu({
      thisPanel: [],
      tile: [
        { label: 'Close tile', glyph: '×', danger: true, run: noop },
        { label: 'Open in new window', run: noop },
        { label: 'Reload', run: noop },
        { label: 'Pin to rail', run: noop },
      ],
    });
    expect(shape(rows)).toEqual(['Open in new window', 'Reload', 'Pin to rail', '—', 'Close tile']);
  });

  test('a "This panel" section that tidies away to nothing counts as empty', () => {
    const rows = composeTileMenu({ thisPanel: [{ divider: true }], tile: TILE });
    expect(rows).toEqual(TILE);
  });
});
