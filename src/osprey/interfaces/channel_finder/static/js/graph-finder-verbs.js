// @ts-check
/**
 * OSPREY Channel Finder — the graph finder's verbs.
 *
 * Every action the finder offers, written once and reached four ways: a
 * visible control, a right-click menu row, a keyboard shortcut, and the
 * panel's ⋯ entries. The mount (explore-graph.js) owns state, network and
 * rendering and hands this module a {@link FinderHost} — a narrow, live view
 * of itself — so a verb reads the state of the moment it runs and asks the
 * mount to redraw only what it changed.
 *
 * Also here: the ⋯ entries' one handler ({@link runPanelMenu}) and the v3
 * `menu` contribution that puts them in the hub's tile ⋯ when embedded
 * ({@link publishMenu}), because both are nothing but verbs under other
 * names. Standalone, the panel toolbar's ⋯ lists the same entries
 * (panel-toolbar.js).
 */

import { contributeHeader, onHeaderAction } from '/design-system/js/header-contrib.js';
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
  setSort,
  SORT_KEYS,
  toggleFacet,
  toggleSelection,
} from './graph-finder-state.js';
import { deviceAddresses } from './graph-finder-render.js';
import { headerMenu, panelMenuEntries } from './graph-finder-menus.js';
import { COLUMN_SPECS, savePrefs } from './graph-finder-prefs.js';

/** @typedef {import('./graph-finder-state.js').FinderState} FinderState */
/** @typedef {import('./graph-finder-state.js').Facet} Facet */
/** @typedef {import('./graph-finder-state.js').SortKey} SortKey */
/** @typedef {import('./graph-finder-menus.js').Actions} Actions */
/** @typedef {import('./graph-finder-menus.js').MenuContext} MenuContext */
/** @typedef {import('./graph-finder-prefs.js').ViewPrefs} ViewPrefs */
/** @typedef {import('/design-system/js/table-columns.js').ColumnLayout} ColumnLayout */

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
 * @property {() => HTMLElement|null} pane - the mounted pane, or null
 * @property {(id: string) => HTMLElement|null} part - an element of the drawn finder, by id
 * @property {() => HTMLInputElement|null} searchInput - the panel toolbar's search field
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

/** The id of the panel's ⋯ entries in the hub's tile ⋯, routed back to {@link runPanelMenu}. */
const PANEL_MENU = 'finder';

/** @param {FinderHost} host @returns {HTMLTableElement|null} */
function tableOf(host) {
  return /** @type {HTMLTableElement|null} */ (host.pane()?.querySelector('.result-table') ?? null);
}

/** @param {FinderHost} host @returns {void} */
function clearSearchBox(host) {
  const input = host.searchInput();
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
  /** A selection change redraws the rows and the ⋯ entries' enablement. */
  const selectionChanged = () => {
    host.renderTable();
    publishMenu(host);
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
      publishMenu(host);
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

/** Whether the hub's ⋯ actions reach {@link runPanelMenu} yet: bound once, at the first publish. */
let menuRouted = false;

/**
 * Offer the panel's ⋯ entries to the hub as one v3 `menu` contribution (the
 * tile ⋯'s "This panel" section), re-sent whole whenever an entry's check
 * or enablement can change. The finder sends nothing else to the bar. A
 * no-op standalone, where header-contrib.js sends nothing.
 * @param {FinderHost} host
 * @returns {void}
 */
export function publishMenu(host) {
  if (!host.pane()) return;
  if (!menuRouted) {
    menuRouted = true;
    // header-contrib.js has no unsubscribe, so the route lives with the page;
    // the host is the mount's one live view and refuses when nothing is mounted.
    onHeaderAction((id, entry) => {
      if (id === PANEL_MENU && entry && host.pane()) runPanelMenu(host, entry);
    });
  }
  const items = panelMenuEntries(host.finder(), host.prefs().density);
  contributeHeader([{ kind: 'menu', id: PANEL_MENU, items }]);
}

/**
 * Take the ⋯ entries back out of the hub's tile ⋯: the finder unmounted.
 * @returns {void}
 */
export function withdrawMenu() {
  contributeHeader([]);
}

/**
 * Run one of the panel's ⋯ entries, from the hub's tile ⋯ or the panel
 * toolbar's standalone ⋯.
 * @param {FinderHost} host
 * @param {string} entry - The entry's id (graph-finder-menus.js panelMenuEntries).
 * @returns {void}
 */
export function runPanelMenu(host, entry) {
  const a = createActions(host);
  if (entry === 'density-comfortable') a.setDensity('comfortable');
  else if (entry === 'density-compact') a.setDensity('compact');
  else if (entry === 'reset-columns') a.resetColumns();
  else if (entry === 'clear-filters') a.clearFilters();
  else if (entry === 'clear-selection') a.clearSelection();
  else if (entry === 'columns') {
    // The ⋯ that asked has closed (the hub's lives in another document); the
    // column checklist opens here, against the table header it changes.
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
