// @ts-check
/**
 * Unit tests for the graph finder's right-click menu builders
 * (graph-finder-menus.js). The builders are pure — what a surface offers, and
 * when a verb is disabled — so every assertion reads the returned rows and runs
 * the bound actions against spies:
 *   npx vitest run tests/interfaces/channel_finder/graph-finder-menus.test.mjs
 */

import { describe, test, expect, vi } from 'vitest';

import {
  bindingMenu,
  chipMenu,
  facetHeadMenu,
  facetValueMenu,
  hasFilters,
  headerMenu,
  inspectorMenu,
  moreMenu,
  rowMenu,
} from '../../../src/osprey/interfaces/channel_finder/static/js/graph-finder-menus.js';
import {
  createFinderState,
  toggleFacet,
  toggleSelection,
} from '../../../src/osprey/interfaces/channel_finder/static/js/graph-finder-state.js';
import { COLUMN_SPECS } from '../../../src/osprey/interfaces/channel_finder/static/js/graph-finder-prefs.js';
import { defaultLayout, resizeColumn, setColumnHidden } from '/design-system/js/table-columns.js';

/** @typedef {import('/design-system/js/context-menu.js').MenuItem} MenuItem */
/** @typedef {import('../../../src/osprey/interfaces/channel_finder/static/js/graph-finder-menus.js').MenuContext} MenuContext */

/** @returns {Record<string, any>} every action, as a spy */
function spies() {
  const names = [
    'copy',
    'send',
    'openDevice',
    'closeDevice',
    'toggleSelected',
    'clearSelection',
    'onlyFacet',
    'toggleFacet',
    'clearFacet',
    'clearFilters',
    'removeChip',
    'collapseAllFacets',
    'sort',
    'nudgeColumn',
    'setColumnHidden',
    'autosize',
    'autosizeAll',
    'resetColumns',
    'setDensity',
    'toggleRail',
  ];
  return Object.fromEntries(names.map((n) => [n, vi.fn()]));
}

/**
 * @param {Partial<MenuContext>} [over]
 * @returns {MenuContext & {actions: Record<string, any>}}
 */
function ctx(over = {}) {
  return /** @type {any} */ ({
    finder: createFinderState(),
    specs: COLUMN_SPECS,
    layout: defaultLayout(COLUMN_SPECS),
    embedded: false,
    openDevice: null,
    density: 'comfortable',
    railCollapsed: false,
    actions: spies(),
    ...over,
  });
}

/** @param {MenuItem[]} items @param {string} label @returns {MenuItem} */
function row(items, label) {
  const found = items.find((i) => i.label === label);
  if (!found)
    throw new Error(
      `no row "${label}" in ${items.map((i) => i.label ?? i.heading ?? '—').join(' | ')}`
    );
  return found;
}

/** @param {MenuItem[]} items @returns {string[]} */
const labels = (items) => items.filter((i) => i.label).map((i) => /** @type {string} */ (i.label));

const ROW = {
  fullPv: 'SR01C:QF1:SP',
  device: 'SR01C_QF1',
  device_uri: 'urn:QF1',
  section: 'SR01C',
  system: 'MG',
  description: 'current\tsetpoint',
  edges: ['WRITESSIGNAL'],
  signals: [{ name: 'current' }, { name: 'voltage' }],
};

describe('a result row', () => {
  test('offers the address verbs, the device, the narrowing verbs and selection', () => {
    const items = rowMenu(ctx(), ROW);
    expect(labels(items)).toEqual([
      'Copy address',
      'Copy device name',
      'Copy row as TSV',
      'Open device',
      'Only section',
      'Only system',
      'Only signal',
      'Only signal',
      'Select',
    ]);
    expect(items.filter((i) => i.label === 'Only signal').map((i) => i.code)).toEqual([
      'current',
      'voltage',
    ]);
  });

  test('the row TSV is one line in table order, with tabs and newlines flattened', () => {
    const c = ctx();
    row(rowMenu(c, ROW), 'Copy row as TSV').run?.();
    expect(c.actions.copy).toHaveBeenCalledWith(
      'SR01C_QF1\tSR01C\tSR01C:QF1:SP\tW\tcurrent, voltage\tcurrent setpoint',
      'the row'
    );
  });

  test('on a row inside a larger selection, copy and send act on the selection', () => {
    const finder = createFinderState();
    toggleSelection(finder, 'SR01C:QF1:SP');
    toggleSelection(finder, 'SR01C:QF1:RB');
    const c = ctx({ finder, embedded: true });
    const items = rowMenu(c, ROW);
    row(items, 'Copy 2 addresses').run?.();
    expect(c.actions.copy).toHaveBeenCalledWith('SR01C:QF1:SP\nSR01C:QF1:RB', '2 addresses');
    row(items, 'Send 2 addresses to assistant').run?.();
    expect(c.actions.send).toHaveBeenCalledWith(['SR01C:QF1:SP', 'SR01C:QF1:RB']);
    expect(row(items, 'Deselect')).toBeTruthy();
    expect(row(items, 'Clear selection (2)').danger).toBe(true);
  });

  test('a selected row on its own copies just itself', () => {
    const finder = createFinderState();
    toggleSelection(finder, 'SR01C:QF1:SP');
    const items = rowMenu(ctx({ finder }), ROW);
    expect(labels(items)).toContain('Copy address');
  });

  test('send exists only when an assistant hosts the panel', () => {
    expect(labels(rowMenu(ctx(), ROW))).not.toContain('Send to assistant');
    expect(labels(rowMenu(ctx({ embedded: true }), ROW))).toContain('Send to assistant');
  });

  test('the device verb closes the device the inspector already shows', () => {
    const c = ctx({ openDevice: 'urn:QF1' });
    row(rowMenu(c, ROW), 'Close device').run?.();
    expect(c.actions.closeDevice).toHaveBeenCalled();
  });

  test('only-verbs narrow through the facet actions', () => {
    const c = ctx();
    const items = rowMenu(c, ROW);
    row(items, 'Only system').run?.();
    expect(c.actions.onlyFacet).toHaveBeenCalledWith('system', 'MG');
  });

  test('a row with no section, system or signal offers no narrowing at all', () => {
    const bare = { fullPv: 'X', device: '', device_uri: '', edges: [] };
    const items = rowMenu(ctx(), bare);
    expect(labels(items).some((l) => l.startsWith('Only'))).toBe(false);
    expect(row(items, 'Copy device name').disabled).toBe(true);
    expect(row(items, 'Open device').disabled).toBe(true);
  });
});

describe('a column header', () => {
  test('pinned chrome columns have no menu', () => {
    expect(headerMenu(ctx(), 'chk')).toEqual([]);
    expect(headerMenu(ctx(), 'nope')).toEqual([]);
  });

  test('sort rows mark the active direction and clear sort follows the state', () => {
    const finder = createFinderState();
    finder.sort = { key: 'device', dir: 'desc' };
    const items = headerMenu(ctx({ finder }), 'device');
    expect(row(items, 'Sort descending').glyph).toBe('✓');
    expect(row(items, 'Sort ascending').glyph).toBe('▲');
    expect(row(items, 'Clear sort').disabled).toBe(false);
    expect(row(headerMenu(ctx(), 'device'), 'Clear sort').disabled).toBe(true);
  });

  test('sorting runs through the sort action', () => {
    const c = ctx();
    row(headerMenu(c, 'section'), 'Sort ascending').run?.();
    expect(c.actions.sort).toHaveBeenCalledWith('section', 'asc');
  });

  test('move left and right are disabled at the visible edges', () => {
    const first = headerMenu(ctx(), 'device');
    expect(row(first, 'Move left').disabled).toBe(true);
    expect(row(first, 'Move right').disabled).toBe(false);
    const last = headerMenu(ctx(), 'description');
    expect(row(last, 'Move right').disabled).toBe(true);
  });

  test('the column checklist reflects the layout and keeps the last column', () => {
    const layout = defaultLayout(COLUMN_SPECS);
    const items = headerMenu(ctx({ layout }), 'device');
    expect(row(items, 'System').checked).toBe(false);
    expect(row(items, 'Device').checked).toBe(true);
    expect(items.some((i) => i.heading === 'Columns')).toBe(true);
    for (const key of ['section', 'address', 'dir', 'signal', 'description'])
      setColumnHidden(COLUMN_SPECS, layout, key, true);
    const lone = headerMenu(ctx({ layout }), 'device');
    expect(row(lone, 'Device').disabled).toBe(true);
    expect(row(lone, 'Hide Device').disabled).toBe(true);
  });

  test('reset is offered only once something moved', () => {
    expect(row(headerMenu(ctx(), 'device'), 'Reset columns').disabled).toBe(true);
    const layout = defaultLayout(COLUMN_SPECS);
    resizeColumn(COLUMN_SPECS, layout, 'device', 222);
    expect(row(headerMenu(ctx({ layout }), 'device'), 'Reset columns').disabled).toBe(false);
  });
});

describe('the rail', () => {
  test('a multi facet value offers only, add and copy', () => {
    const c = ctx();
    const items = facetValueMenu(c, 'section', 'SR01C', 'SR01C');
    expect(labels(items)).toEqual(['Only', 'Add to filter', 'Copy value', 'Clear this facet']);
    expect(row(items, 'Clear this facet').disabled).toBe(true);
    row(items, 'Add to filter').run?.();
    expect(c.actions.toggleFacet).toHaveBeenCalledWith('section', 'SR01C');
  });

  test('a selected value offers remove instead of add', () => {
    const finder = createFinderState();
    toggleFacet(finder, 'section', 'SR01C');
    expect(labels(facetValueMenu(ctx({ finder }), 'section', 'SR01C', 'SR01C'))).toContain(
      'Remove from filter'
    );
  });

  test('a class copies its URI under its name, and offers no add', () => {
    const c = ctx();
    const items = facetValueMenu(c, 'cls', 'urn:Dipole', 'Dipole');
    expect(labels(items)).not.toContain('Add to filter');
    row(items, 'Copy value').run?.();
    expect(c.actions.copy).toHaveBeenCalledWith('urn:Dipole', 'Dipole');
  });

  test('a facet heading clears it, all filters, or folds every facet', () => {
    const finder = createFinderState();
    toggleFacet(finder, 'dir', 'R');
    const c = ctx({ finder });
    const items = facetHeadMenu(c, 'dir');
    expect(row(items, 'Clear this facet').disabled).toBe(false);
    expect(row(items, 'Clear all filters').danger).toBe(true);
    row(items, 'Collapse all facets').run?.();
    expect(c.actions.collapseAllFacets).toHaveBeenCalledWith(true);
    expect(row(facetHeadMenu(ctx(), 'dir'), 'Clear all filters').disabled).toBe(true);
  });

  test('a chip removes itself or every filter', () => {
    const c = ctx();
    const items = chipMenu(c, 'section:SR01C');
    row(items, 'Remove filter').run?.();
    expect(c.actions.removeChip).toHaveBeenCalledWith('section:SR01C');
  });
});

describe('the inspector', () => {
  test('its heading copies, sends and closes the whole device', () => {
    const c = ctx({ embedded: true });
    const items = inspectorMenu(c, 'QF1', ['A', 'B']);
    row(items, 'Copy 2 addresses').run?.();
    expect(c.actions.copy).toHaveBeenCalledWith('A\nB', '2 addresses');
    row(items, 'Send 2 addresses to assistant').run?.();
    expect(c.actions.send).toHaveBeenCalledWith(['A', 'B']);
    expect(labels(items)).toContain('Close');
  });

  test('a device with nothing bound copies nothing', () => {
    expect(row(inspectorMenu(ctx(), 'EMPTY', []), 'Copy 0 addresses').disabled).toBe(true);
  });

  test('a channel row copies, sends and selects its address', () => {
    const c = ctx({ embedded: true });
    const items = bindingMenu(c, 'A');
    expect(labels(items)).toEqual(['Copy address', 'Send to assistant', 'Add to selection']);
    row(items, 'Add to selection').run?.();
    expect(c.actions.toggleSelected).toHaveBeenCalledWith('A');
  });
});

describe('the toolbar ⋯ menu', () => {
  test('density is a pair of toggles reflecting the preference', () => {
    const items = moreMenu(ctx({ density: 'compact' }));
    expect(row(items, 'Compact').checked).toBe(true);
    expect(row(items, 'Comfortable').checked).toBe(false);
  });

  test('the rail verb names what it will do', () => {
    expect(labels(moreMenu(ctx()))).toContain('Hide filters');
    expect(labels(moreMenu(ctx({ railCollapsed: true })))).toContain('Show filters');
  });

  test('bulk verbs are disabled when there is nothing to clear', () => {
    const items = moreMenu(ctx());
    expect(row(items, 'Clear all filters').disabled).toBe(true);
    expect(row(items, 'Clear selection').disabled).toBe(true);
  });
});

test('hasFilters sees the query and every facet', () => {
  const finder = createFinderState();
  expect(hasFilters(finder)).toBe(false);
  finder.q = '  ';
  expect(hasFilters(finder)).toBe(false);
  toggleFacet(finder, 'signal', 'current');
  expect(hasFilters(finder)).toBe(true);
});
