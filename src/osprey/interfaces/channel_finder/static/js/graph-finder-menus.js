// @ts-check
/**
 * OSPREY Channel Finder — the graph finder's right-click menus.
 *
 * One builder per surface: a result row, a column header, a facet value, a
 * facet heading, an active-filter chip, the device inspector and its channel
 * rows, and the toolbar's ⋯ menu. Each takes what the surface shows plus an
 * {@link Actions} object of bound verbs, and returns the rows the shared
 * context menu (/design-system/js/context-menu.js) draws. Nothing here touches
 * the document or the network, so which verb a surface offers — and when it
 * is disabled — is tested without a browser.
 *
 * The verbs are the same ones the visible controls run: a row's "Copy address"
 * is the row's ⎘, "Only section SR01C" is a rail click, "Sort ascending" is a
 * header click. A menu adds reach, never a second code path.
 */

import { directionOf } from './graph-finder-render.js';
import { facetCount, hasFacet } from './graph-finder-state.js';
import { isDefaultLayout, visibleColumns } from '/design-system/js/table-columns.js';

/** @typedef {import('/design-system/js/context-menu.js').MenuItem} MenuItem */
/** @typedef {import('./graph-finder-state.js').FinderState} FinderState */
/** @typedef {import('./graph-finder-state.js').Facet} Facet */
/** @typedef {import('./graph-finder-state.js').SortKey} SortKey */
/** @typedef {import('/design-system/js/table-columns.js').ColumnSpec} ColumnSpec */
/** @typedef {import('/design-system/js/table-columns.js').ColumnLayout} ColumnLayout */

/**
 * The verbs a menu row can run, bound by the mount.
 * @typedef {object} Actions
 * @property {(text: string, what: string) => void} copy - copy text, naming it in the toast
 * @property {(pvs: string[]) => void} send - post addresses to the assistant prompt
 * @property {(uri: string) => void} openDevice
 * @property {() => void} closeDevice
 * @property {(pv: string) => void} toggleSelected
 * @property {() => void} clearSelection
 * @property {(facet: Facet, value: string) => void} onlyFacet
 * @property {(facet: Facet, value: string) => void} toggleFacet
 * @property {(facet: Facet) => void} clearFacet
 * @property {() => void} clearFilters
 * @property {(key: string) => void} removeChip
 * @property {(collapsed: boolean) => void} collapseAllFacets
 * @property {(key: SortKey, dir: 'asc'|'desc'|null) => void} sort
 * @property {(key: string, delta: -1|1) => void} nudgeColumn
 * @property {(key: string, hidden: boolean) => void} setColumnHidden
 * @property {(key: string) => void} autosize
 * @property {() => void} autosizeAll
 * @property {() => void} resetColumns
 * @property {(density: 'comfortable'|'compact') => void} setDensity
 * @property {() => void} toggleRail
 */

/**
 * What every menu needs to know about the finder at the moment it opens.
 * @typedef {object} MenuContext
 * @property {FinderState} finder
 * @property {ColumnSpec[]} specs
 * @property {ColumnLayout} layout
 * @property {boolean} embedded - an assistant session hosts the panel, so Send exists
 * @property {string|null} openDevice - the URI the inspector shows, if any
 * @property {'comfortable'|'compact'} density
 * @property {boolean} railCollapsed
 * @property {Actions} actions
 */

/** @param {number} n @param {string} one @param {string} many */
const plural = (n, one, many) => `${n.toLocaleString()} ${n === 1 ? one : many}`;

/**
 * The column toggles, under a caption. Pinned chrome columns are not offered.
 * @param {MenuContext} ctx
 * @returns {MenuItem[]}
 */
function columnToggles(ctx) {
  const shown = visibleColumns(ctx.specs, ctx.layout).filter((c) => !c.pin);
  return [
    { heading: 'Columns' },
    ...ctx.specs
      .filter((s) => !s.pin)
      .map((s) => {
        const visible = !ctx.layout.hidden.has(s.key);
        return {
          label: s.label,
          checked: visible,
          // The last visible column stays: a table of no columns is not a view.
          disabled: visible && shown.length <= 1,
          run: () => ctx.actions.setColumnHidden(s.key, visible),
        };
      }),
  ];
}

/**
 * A result row. On a row that is part of a larger selection the copy and send
 * verbs act on the whole selection, and say so in their label.
 * @param {MenuContext} ctx
 * @param {Record<string, any>} row - the search payload row
 * @returns {MenuItem[]}
 */
export function rowMenu(ctx, row) {
  const { finder, actions } = ctx;
  const pv = String(row.fullPv ?? '');
  const device = String(row.device ?? '');
  const uri = String(row.device_uri ?? '');
  const selected = finder.selected.has(pv);
  const many = selected && finder.selected.size > 1;
  const targets = many ? [...finder.selected] : [pv];
  const n = finder.selected.size;
  const signals = Array.isArray(row.signals)
    ? row.signals.map((/** @type {any} */ s) => String((s && s.name) ?? '')).filter(Boolean)
    : [];
  const dirLabel = { R: 'R', W: 'W', RW: 'RW', none: '' }[directionOf(row)];
  const tsv = [device, row.section ?? '', pv, dirLabel, signals.join(', '), row.description ?? '']
    .map((v) => String(v).replace(/[\t\n]/g, ' '))
    .join('\t');

  /** @type {MenuItem[]} */
  const items = [
    many
      ? {
          label: `Copy ${plural(n, 'address', 'addresses')}`,
          glyph: '⎘',
          run: () => actions.copy(targets.join('\n'), plural(n, 'address', 'addresses')),
        }
      : { label: 'Copy address', glyph: '⎘', run: () => actions.copy(pv, pv) },
    { label: 'Copy device name', disabled: !device, run: () => actions.copy(device, device) },
    { label: 'Copy row as TSV', run: () => actions.copy(tsv, 'the row') },
  ];
  if (ctx.embedded) {
    items.push({
      label: many ? `Send ${plural(n, 'address', 'addresses')} to assistant` : 'Send to assistant',
      glyph: '➤',
      run: () => actions.send(targets),
    });
  }
  items.push({ divider: true });
  items.push(
    ctx.openDevice === uri && uri
      ? { label: 'Close device', run: () => actions.closeDevice() }
      : { label: 'Open device', disabled: !uri, run: () => actions.openDevice(uri) }
  );

  /** @type {MenuItem[]} */
  const narrow = [];
  if (row.section)
    narrow.push({
      label: 'Only section',
      code: String(row.section),
      run: () => actions.onlyFacet('section', String(row.section)),
    });
  if (row.system)
    narrow.push({
      label: 'Only system',
      code: String(row.system),
      run: () => actions.onlyFacet('system', String(row.system)),
    });
  for (const name of signals.slice(0, 3)) {
    narrow.push({ label: 'Only signal', code: name, run: () => actions.onlyFacet('signal', name) });
  }
  if (narrow.length) items.push({ divider: true }, ...narrow);

  items.push({ divider: true });
  items.push({ label: selected ? 'Deselect' : 'Select', run: () => actions.toggleSelected(pv) });
  if (n > 0)
    items.push({
      label: `Clear selection (${n.toLocaleString()})`,
      danger: true,
      run: () => actions.clearSelection(),
    });
  return items;
}

/**
 * One channel row of the device inspector. The inspector carries addresses,
 * not search rows, so the menu is the address verbs alone.
 * @param {MenuContext} ctx
 * @param {string} pv
 * @returns {MenuItem[]}
 */
export function bindingMenu(ctx, pv) {
  const { actions, finder } = ctx;
  /** @type {MenuItem[]} */
  const items = [{ label: 'Copy address', glyph: '⎘', run: () => actions.copy(pv, pv) }];
  if (ctx.embedded)
    items.push({ label: 'Send to assistant', glyph: '➤', run: () => actions.send([pv]) });
  items.push(
    { divider: true },
    {
      label: finder.selected.has(pv) ? 'Deselect' : 'Add to selection',
      run: () => actions.toggleSelected(pv),
    }
  );
  return items;
}

/**
 * The device inspector's heading.
 * @param {MenuContext} ctx
 * @param {string} device - the device name
 * @param {string[]} pvs - every address the device binds
 * @returns {MenuItem[]}
 */
export function inspectorMenu(ctx, device, pvs) {
  const { actions } = ctx;
  const what = plural(pvs.length, 'address', 'addresses');
  /** @type {MenuItem[]} */
  const items = [
    {
      label: `Copy ${what}`,
      glyph: '⎘',
      disabled: pvs.length === 0,
      run: () => actions.copy(pvs.join('\n'), what),
    },
    { label: 'Copy device name', disabled: !device, run: () => actions.copy(device, device) },
  ];
  if (ctx.embedded)
    items.push({
      label: `Send ${what} to assistant`,
      glyph: '➤',
      disabled: pvs.length === 0,
      run: () => actions.send(pvs),
    });
  items.push({ divider: true }, { label: 'Close', glyph: '✕', run: () => actions.closeDevice() });
  return items;
}

/**
 * A column header: sort, move, size, and which columns show.
 * @param {MenuContext} ctx
 * @param {string} key - the column's key
 * @returns {MenuItem[]}
 */
export function headerMenu(ctx, key) {
  const { finder, actions, specs, layout } = ctx;
  const spec = specs.find((s) => s.key === key);
  if (!spec || spec.pin) return [];
  const sortKey = /** @type {SortKey} */ (key);
  const sorted = finder.sort && finder.sort.key === key ? finder.sort.dir : null;
  const shown = visibleColumns(specs, layout)
    .filter((c) => !c.pin)
    .map((c) => c.key);
  const at = shown.indexOf(key);

  return [
    {
      label: 'Sort ascending',
      glyph: sorted === 'asc' ? '✓' : '▲',
      run: () => actions.sort(sortKey, 'asc'),
    },
    {
      label: 'Sort descending',
      glyph: sorted === 'desc' ? '✓' : '▼',
      run: () => actions.sort(sortKey, 'desc'),
    },
    { label: 'Clear sort', disabled: !finder.sort, run: () => actions.sort(sortKey, null) },
    { divider: true },
    { label: 'Move left', glyph: '←', disabled: at <= 0, run: () => actions.nudgeColumn(key, -1) },
    {
      label: 'Move right',
      glyph: '→',
      disabled: at === -1 || at >= shown.length - 1,
      run: () => actions.nudgeColumn(key, 1),
    },
    {
      label: `Hide ${spec.label}`,
      disabled: shown.length <= 1,
      run: () => actions.setColumnHidden(key, true),
    },
    { label: 'Fit column to content', run: () => actions.autosize(key) },
    { label: 'Fit all columns', run: () => actions.autosizeAll() },
    { divider: true },
    ...columnToggles(ctx),
    { divider: true },
    {
      label: 'Reset columns',
      disabled: isDefaultLayout(specs, layout),
      run: () => actions.resetColumns(),
    },
  ];
}

/**
 * One facet value in the rail.
 * @param {MenuContext} ctx
 * @param {Facet} facet
 * @param {string} value - the facet's value (a class URI for `cls`)
 * @param {string} label - what the rail shows for it
 * @returns {MenuItem[]}
 */
export function facetValueMenu(ctx, facet, value, label) {
  const { finder, actions } = ctx;
  const on = hasFacet(finder, facet, value);
  /** @type {MenuItem[]} */
  const items = [{ label: 'Only', code: label, run: () => actions.onlyFacet(facet, value) }];
  // The class facet is single-select: a click already replaces, so there is
  // no "add" to offer beside "only".
  if (facet !== 'cls') {
    items.push({
      label: on ? 'Remove from filter' : 'Add to filter',
      run: () => actions.toggleFacet(facet, value),
    });
  }
  items.push(
    { label: 'Copy value', glyph: '⎘', run: () => actions.copy(value, label) },
    { divider: true },
    {
      label: 'Clear this facet',
      disabled: facetCount(finder, facet) === 0,
      run: () => actions.clearFacet(facet),
    }
  );
  return items;
}

/**
 * A facet heading.
 * @param {MenuContext} ctx
 * @param {Facet} facet
 * @returns {MenuItem[]}
 */
export function facetHeadMenu(ctx, facet) {
  const { finder, actions } = ctx;
  return [
    {
      label: 'Clear this facet',
      disabled: facetCount(finder, facet) === 0,
      run: () => actions.clearFacet(facet),
    },
    {
      label: 'Clear all filters',
      danger: true,
      disabled: !hasFilters(finder),
      run: () => actions.clearFilters(),
    },
    { divider: true },
    { label: 'Collapse all facets', run: () => actions.collapseAllFacets(true) },
    { label: 'Expand all facets', run: () => actions.collapseAllFacets(false) },
  ];
}

/**
 * An active-filter chip.
 * @param {MenuContext} ctx
 * @param {string} key - the chip key
 * @returns {MenuItem[]}
 */
export function chipMenu(ctx, key) {
  return [
    { label: 'Remove filter', run: () => ctx.actions.removeChip(key) },
    { label: 'Clear all filters', danger: true, run: () => ctx.actions.clearFilters() },
  ];
}

/**
 * The toolbar's ⋯ menu: view settings and the bulk verbs.
 * @param {MenuContext} ctx
 * @returns {MenuItem[]}
 */
export function moreMenu(ctx) {
  const { finder, actions } = ctx;
  return [
    { heading: 'Rows' },
    {
      label: 'Comfortable',
      checked: ctx.density === 'comfortable',
      run: () => actions.setDensity('comfortable'),
    },
    {
      label: 'Compact',
      checked: ctx.density === 'compact',
      run: () => actions.setDensity('compact'),
    },
    { divider: true },
    ...columnToggles(ctx),
    {
      label: 'Reset columns',
      disabled: isDefaultLayout(ctx.specs, ctx.layout),
      run: () => actions.resetColumns(),
    },
    { divider: true },
    { label: ctx.railCollapsed ? 'Show filters' : 'Hide filters', run: () => actions.toggleRail() },
    {
      label: 'Clear all filters',
      disabled: !hasFilters(finder),
      run: () => actions.clearFilters(),
    },
    {
      label: 'Clear selection',
      disabled: finder.selected.size === 0,
      run: () => actions.clearSelection(),
    },
  ];
}

/**
 * Whether any query or facet narrows the result.
 * @param {FinderState} finder
 * @returns {boolean}
 */
export function hasFilters(finder) {
  return (
    finder.q.trim() !== '' ||
    finder.cls !== null ||
    finder.section.size > 0 ||
    finder.system.size > 0 ||
    finder.signal.size > 0 ||
    finder.dir.size > 0
  );
}
