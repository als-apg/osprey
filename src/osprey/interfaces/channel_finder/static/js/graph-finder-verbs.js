// @ts-check
/**
 * OSPREY Channel Finder — the graph finder's verbs.
 *
 * Every action the finder offers, written once and reached four ways: a
 * visible control, a right-click menu row, a keyboard shortcut, and the hub's
 * tile bar. The mount (explore-graph.js) owns state, network and rendering and
 * hands this module a {@link FinderHost} — a narrow, live view of itself — so a
 * verb reads the state of the moment it runs and asks the mount to redraw only
 * what it changed.
 *
 * Also here: the tile-bar contribution (the search box and the ⋯ menu the
 * finder adds to the hub's bar when embedded) and the routing of its actions,
 * because both are nothing but verbs under other names.
 */

import { state } from './state.js';
import { openContextMenuAt } from '/design-system/js/context-menu.js';
import {
  defaultLayout,
  measureColumn,
  nudgeColumn,
  resizeColumn,
  setColumnHidden,
  visibleColumns,
} from '/design-system/js/table-columns.js';
import {
  clearFacet,
  clearFilters,
  clearSelection,
  cycleSort,
  onlyFacet,
  removeChip,
  setQuery,
  setSort,
  SORT_KEYS,
  toggleFacet,
  toggleSelection,
} from './graph-finder-state.js';
import { deviceAddresses } from './graph-finder-render.js';
import { hasFilters, headerMenu } from './graph-finder-menus.js';
import { COLUMN_SPECS, savePrefs } from './graph-finder-prefs.js';

/** @typedef {import('./graph-finder-state.js').FinderState} FinderState */
/** @typedef {import('./graph-finder-state.js').Facet} Facet */
/** @typedef {import('./graph-finder-state.js').SortKey} SortKey */
/** @typedef {import('./graph-finder-menus.js').Actions} Actions */
/** @typedef {import('./graph-finder-menus.js').MenuContext} MenuContext */
/** @typedef {import('./graph-finder-prefs.js').ViewPrefs} ViewPrefs */
/** @typedef {import('/design-system/js/table-columns.js').ColumnLayout} ColumnLayout */
/** @typedef {import('/design-system/js/splitter.js').SplitterApi} SplitterApi */

/**
 * The mount, as the verbs and the event handlers see it. Every getter is live.
 * @typedef {object} FinderHost
 * @property {() => FinderState} finder
 * @property {() => any} results - the newest search payload, or null
 * @property {() => {uri: string, data: any}|null} card - the device the inspector shows
 * @property {() => ColumnLayout} layout
 * @property {(layout: ColumnLayout) => void} setLayout
 * @property {() => ViewPrefs} prefs
 * @property {() => boolean} embedded
 * @property {() => SplitterApi|null} railSplit
 * @property {() => HTMLElement|null} pane - the mounted pane, or null
 * @property {(id: string) => HTMLElement|null} part - an element of the drawn finder, by id
 * @property {string} searchInputId
 * @property {() => Promise<void>} runSearch
 * @property {() => void} renderTable
 * @property {() => void} renderRail
 * @property {() => void} commitColumns - persist the layout and redraw the table
 * @property {(uri: string) => Promise<void>} toggleDevice
 * @property {(uri: string) => Promise<void>} openDevice
 * @property {() => void} closeCard
 * @property {(text: string, what: string) => Promise<void>} copy
 * @property {(pvs: string[], text?: string) => void} send
 * @property {() => void} copySelection
 * @property {(direction: string|null) => void} turnPage
 */

/** The facet groups, in rail order. */
const FACET_KEYS = /** @type {Facet[]} */ (['section', 'system', 'cls', 'signal', 'dir']);

/** Header-contribution ids, routed back through {@link onHeaderAction}. */
export const HEADER_SEARCH = 'graph-search';
export const HEADER_MENU = 'graph-more';

/** @param {FinderHost} host @returns {HTMLTableElement|null} */
function tableOf(host) {
  return /** @type {HTMLTableElement|null} */ (host.pane()?.querySelector('.result-table') ?? null);
}

/** @param {FinderHost} host @returns {void} */
function clearSearchBox(host) {
  const input = /** @type {HTMLInputElement|null} */ (host.part(host.searchInputId));
  if (input) input.value = '';
}

/**
 * The verbs every menu row, control and key binds. Built per use, so each
 * closure reads the state of the moment it runs.
 * @param {FinderHost} host
 * @returns {Actions}
 */
export function createActions(host) {
  const finder = host.finder();
  /** A selection change redraws the rows and the tile-bar menu's enablement. */
  const selectionChanged = () => {
    host.renderTable();
    contributeHeader(host);
  };
  return {
    copy: (text, what) => void host.copy(text, what),
    send: (pvs) => host.send(pvs),
    openDevice: (uri) => void host.openDevice(uri),
    closeDevice: () => host.closeCard(),
    toggleSelected: (pv) => {
      toggleSelection(finder, pv);
      selectionChanged();
    },
    clearSelection: () => {
      clearSelection(finder);
      selectionChanged();
    },
    onlyFacet: (facet, value) => {
      if (onlyFacet(finder, facet, value)) void host.runSearch();
    },
    toggleFacet: (facet, value) => {
      toggleFacet(finder, facet, value);
      void host.runSearch();
    },
    clearFacet: (facet) => {
      if (clearFacet(finder, facet)) void host.runSearch();
    },
    clearFilters: () => {
      if (!clearFilters(finder)) return;
      clearSearchBox(host);
      void host.runSearch();
    },
    removeChip: (key) => {
      if (!removeChip(finder, key)) return;
      if (key === 'q') clearSearchBox(host);
      void host.runSearch();
    },
    collapseAllFacets: (collapsed) => {
      const prefs = host.prefs();
      prefs.collapsed = collapsed ? [...FACET_KEYS] : [];
      savePrefs(prefs);
      host.renderRail();
    },
    sort: (key, dir) => {
      if (setSort(finder, dir === null ? null : { key, dir })) void host.runSearch();
    },
    nudgeColumn: (key, delta) => {
      if (nudgeColumn(host.layout(), key, delta)) host.commitColumns();
    },
    setColumnHidden: (key, hidden) => {
      if (setColumnHidden(COLUMN_SPECS, host.layout(), key, hidden)) host.commitColumns();
    },
    autosize: (key) => {
      const table = tableOf(host);
      const width = table ? measureColumn(table, key) : null;
      if (width === null) return;
      resizeColumn(COLUMN_SPECS, host.layout(), key, width);
      host.commitColumns();
    },
    autosizeAll: () => {
      const table = tableOf(host);
      if (!table) return;
      for (const col of visibleColumns(COLUMN_SPECS, host.layout())) {
        if (col.pin) continue;
        const width = measureColumn(table, col.key);
        if (width !== null) resizeColumn(COLUMN_SPECS, host.layout(), col.key, width);
      }
      host.commitColumns();
    },
    resetColumns: () => {
      host.setLayout(defaultLayout(COLUMN_SPECS));
      host.commitColumns();
    },
    setDensity: (density) => {
      const prefs = host.prefs();
      prefs.density = density;
      savePrefs(prefs);
      host
        .pane()
        ?.querySelector('.graph-explorer')
        ?.classList.toggle('compact', density === 'compact');
      for (const btn of host.pane()?.querySelectorAll('[data-density]') ?? []) {
        btn.setAttribute('aria-pressed', String(btn.getAttribute('data-density') === density));
      }
      contributeHeader(host);
    },
    toggleRail: () => {
      host.railSplit()?.toggleCollapse();
      contributeHeader(host);
    },
  };
}

/**
 * What a menu builder needs to know at the moment it opens.
 * @param {FinderHost} host
 * @returns {MenuContext}
 */
export function menuContext(host) {
  const card = host.card();
  return {
    finder: host.finder(),
    specs: COLUMN_SPECS,
    layout: host.layout(),
    embedded: host.embedded(),
    openDevice: card ? card.uri : null,
    density: host.prefs().density,
    railCollapsed: host.railSplit()?.isCollapsed() ?? false,
    actions: createActions(host),
  };
}

/**
 * Fold or unfold one facet group, and remember it.
 * @param {FinderHost} host
 * @param {string} key
 * @returns {void}
 */
export function toggleFold(host, key) {
  const prefs = host.prefs();
  const folded = new Set(prefs.collapsed);
  if (folded.has(key)) folded.delete(key);
  else folded.add(key);
  prefs.collapsed = [...folded];
  savePrefs(prefs);
  const group = host.pane()?.querySelector(`.facet[data-facet-group="${CSS.escape(key)}"]`);
  if (group) {
    group.classList.toggle('collapsed', folded.has(key));
    group
      .querySelector('[data-facet-toggle]')
      ?.setAttribute('aria-expanded', String(!folded.has(key)));
  }
}

/**
 * The header click: cycle a column through ascending, descending and address
 * order. Only the server's sortable columns respond.
 * @param {FinderHost} host
 * @param {string} key
 * @returns {void}
 */
export function sortBy(host, key) {
  /** @type {readonly string[]} */
  const sortKeys = SORT_KEYS;
  if (!sortKeys.includes(key)) return;
  cycleSort(host.finder(), /** @type {SortKey} */ (key));
  void host.runSearch();
}

/**
 * Run one footer or inspector action.
 * @param {FinderHost} host
 * @param {string|null} action - `copy`, `send`, `clear`, `close-card`, `copy-device` or `send-device`.
 * @returns {void}
 */
export function runAction(host, action) {
  const finder = host.finder();
  if (action === 'copy') host.copySelection();
  else if (action === 'send') host.send([...finder.selected]);
  else if (action === 'clear') createActions(host).clearSelection();
  else if (action === 'close-card') host.closeCard();
  else if (action === 'copy-device' || action === 'send-device') {
    const card = host.card();
    const pvs = card ? deviceAddresses(card.data) : [];
    const what = `${pvs.length.toLocaleString()} ${pvs.length === 1 ? 'address' : 'addresses'}`;
    if (action === 'copy-device') void host.copy(pvs.join('\n'), what);
    else host.send(pvs);
  }
}

/**
 * Publish the search box and the ⋯ menu to the hub's tile bar, in Expert and
 * Simple alike. A no-op standalone (header-contrib.js ignores it outside a
 * frame), where the in-body toolbar stays.
 * @param {FinderHost} host
 * @returns {void}
 */
export function contributeHeader(host) {
  if (!host.pane()) return;
  const finder = host.finder();
  const prefs = host.prefs();
  state.setViewHeader({
    items: [
      {
        kind: 'search',
        id: HEADER_SEARCH,
        placeholder: 'Search devices, addresses and descriptions',
        value: finder.q,
        priority: 2,
      },
      {
        kind: 'menu',
        id: HEADER_MENU,
        label: 'Finder options',
        priority: 1,
        items: [
          {
            id: 'density-comfortable',
            label: 'Comfortable rows',
            checked: prefs.density === 'comfortable',
          },
          { id: 'density-compact', label: 'Compact rows', checked: prefs.density === 'compact' },
          { id: 'columns', label: 'Choose columns…' },
          { id: 'reset-columns', label: 'Reset columns' },
          {
            id: 'toggle-rail',
            label: host.railSplit()?.isCollapsed() ? 'Show filters' : 'Hide filters',
          },
          { id: 'clear-filters', label: 'Clear all filters', disabled: !hasFilters(finder) },
          { id: 'clear-selection', label: 'Clear selection', disabled: finder.selected.size === 0 },
        ],
      },
    ],
    onAction: (id, value) => onHeaderAction(host, id, value),
  });
}

/**
 * An action from the hub's tile bar.
 * @param {FinderHost} host
 * @param {string} id - The contributed item's id.
 * @param {string|undefined} value - The search text, or the menu entry's id.
 * @returns {void}
 */
export function onHeaderAction(host, id, value) {
  if (id === HEADER_SEARCH) {
    // The hub debounces its own search box, so the text is final here.
    const input = /** @type {HTMLInputElement|null} */ (host.part(host.searchInputId));
    if (input) input.value = value ?? '';
    setQuery(host.finder(), value ?? '');
    void host.runSearch();
    return;
  }
  if (id !== HEADER_MENU) return;
  const a = createActions(host);
  if (value === 'density-comfortable') a.setDensity('comfortable');
  else if (value === 'density-compact') a.setDensity('compact');
  else if (value === 'reset-columns') a.resetColumns();
  else if (value === 'toggle-rail') a.toggleRail();
  else if (value === 'clear-filters') a.clearFilters();
  else if (value === 'clear-selection') a.clearSelection();
  else if (value === 'columns') {
    // The tile bar lives in the hub's document; the column checklist opens
    // here, against the table header it changes.
    const head = host.pane()?.querySelector('.result-table thead tr');
    const first = visibleColumns(COLUMN_SPECS, host.layout()).find((c) => !c.pin)?.key ?? 'device';
    if (head instanceof HTMLElement) {
      openContextMenuAt(head, {
        ariaLabel: 'Columns',
        items: headerMenu(menuContext(host), first),
      });
    }
  }
}
