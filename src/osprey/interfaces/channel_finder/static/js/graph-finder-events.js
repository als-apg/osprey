// @ts-check
/**
 * OSPREY Channel Finder — the graph finder's input handling.
 *
 * Every click, change, keystroke and right-click the finder answers, delegated
 * from the mounted pane (which outlives every re-render inside it, so nothing
 * is re-bound) plus the two document-level keys: `/` to search and Escape to
 * close the inspector. Each handler resolves its target to a verb from
 * graph-finder-verbs.js and runs it against the mount's {@link FinderHost};
 * none of them draws or fetches on its own.
 *
 * Keyboard model: the result table is one Tab stop. Arrows, Home and End walk
 * its rows, Space selects, Enter opens the device, ⌘C/Ctrl+C copies the row's
 * address, and Shift+F10 or the menu key opens the focused surface's menu
 * (macOS fires no `contextmenu` for those keys). A focused column header sorts
 * on Enter/Space and moves on Alt+←/→.
 */

import { isContextMenuKey, openContextMenu, openContextMenuAt } from '/design-system/js/context-menu.js';
import { nudgeColumn } from '/design-system/js/table-columns.js';
import { setQuery, togglePageSelection, toggleSelection } from './graph-finder-state.js';
import { deviceAddresses } from './graph-finder-render.js';
import {
  bindingMenu,
  chipMenu,
  facetHeadMenu,
  facetValueMenu,
  headerMenu,
  inspectorMenu,
  moreMenu,
  rowMenu,
} from './graph-finder-menus.js';
import { contributeHeader, createActions, menuContext, runAction, sortBy, toggleFold } from './graph-finder-verbs.js';

/** @typedef {import('./graph-finder-verbs.js').FinderHost} FinderHost */
/** @typedef {import('./graph-finder-state.js').Facet} Facet */
/** @typedef {import('/design-system/js/context-menu.js').MenuItem} MenuItem */

/** How long typing settles before the query is sent, in ms. */
export const SEARCH_DEBOUNCE_MS = 200;

const ROWS = '.result-table tbody tr[data-pv]';
const DATA_HEADS = '.result-table thead th[data-col]:not([data-pinned])';

/**
 * Wire every listener the finder needs. Returns the teardown, which also
 * cancels a query still settling in its debounce.
 * @param {HTMLElement} pane - The mounted pane.
 * @param {FinderHost} host
 * @returns {() => void}
 */
export function bindFinderEvents(pane, host) {
  /** @type {number|undefined} */
  let debounce;
  const controller = new AbortController();
  const { signal } = controller;

  pane.addEventListener('input', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement) || target.id !== host.searchInputId) return;
    const value = target.value;
    window.clearTimeout(debounce);
    debounce = window.setTimeout(() => {
      setQuery(host.finder(), value);
      void host.runSearch();
    }, SEARCH_DEBOUNCE_MS);
  }, { signal });

  pane.addEventListener('change', (event) => onChange(event, host), { signal });
  pane.addEventListener('click', (event) => onClick(event, host), { signal });
  pane.addEventListener('contextmenu', (event) => onContextMenu(event, host), { signal });
  pane.addEventListener('keydown', (event) => onKeydown(event, host), { signal });
  pane.addEventListener('focusin', (event) => onFocusIn(event, host), { signal });
  document.addEventListener('keydown', (event) => onDocKeydown(event, host), { signal });

  return () => {
    window.clearTimeout(debounce);
    controller.abort();
  };
}

/**
 * Selection: one row, or every row on the page. Neither refetches — the
 * selection is the operator's, not the server's.
 * @param {Event} event
 * @param {FinderHost} host
 * @returns {void}
 */
function onChange(event, host) {
  const target = event.target;
  if (!(target instanceof HTMLInputElement)) return;
  const finder = host.finder();
  if (target.hasAttribute('data-select-page')) {
    const rows = Array.isArray(host.results()?.rows) ? host.results().rows : [];
    togglePageSelection(finder, rows.map((/** @type {any} */ r) => String(r.fullPv ?? '')), target.checked);
  } else if (target.type === 'checkbox' && target.hasAttribute('data-pv')) {
    toggleSelection(finder, target.getAttribute('data-pv') || '');
  } else {
    return;
  }
  host.renderTable();
  contributeHeader(host);
}

/**
 * Every click the finder answers, in the order the render module's contract
 * lists them.
 * @param {Event} event
 * @param {FinderHost} host
 * @returns {void}
 */
function onClick(event, host) {
  const target = event.target;
  if (!(target instanceof Element)) return;
  const act = createActions(host);

  const facet = target.closest('button.facet-item[data-facet][data-value]');
  if (facet) {
    return act.toggleFacet(/** @type {Facet} */ (facet.getAttribute('data-facet')), facet.getAttribute('data-value') || '');
  }
  const fold = target.closest('button[data-facet-toggle]');
  if (fold) return toggleFold(host, fold.getAttribute('data-facet-toggle') || '');
  const clear = target.closest('button[data-clear-facet]');
  if (clear) return act.clearFacet(/** @type {Facet} */ (clear.getAttribute('data-clear-facet')));
  const chip = target.closest('button.active-filter[data-chip]');
  if (chip) return act.removeChip(chip.getAttribute('data-chip') || '');
  if (target.closest('[data-clear-all]')) return act.clearFilters();

  const density = target.closest('button[data-density]');
  if (density) return act.setDensity(density.getAttribute('data-density') === 'compact' ? 'compact' : 'comfortable');

  const more = target.closest('button[data-more]');
  if (more instanceof HTMLElement) {
    const rect = more.getBoundingClientRect();
    openContextMenu({
      x: rect.right - 220,
      y: rect.bottom + 4,
      anchorEl: more,
      ariaLabel: 'Finder options',
      items: moreMenu(menuContext(host)),
    });
    return;
  }

  const copy = target.closest('button.copy-btn[data-copy]');
  if (copy) return void host.copy(copy.getAttribute('data-copy') || '', copy.getAttribute('data-copy') || '');
  const dev = target.closest('button.dev[data-uri]');
  if (dev) return void host.toggleDevice(dev.getAttribute('data-uri') || '');
  const head = target.closest(DATA_HEADS);
  if (head && !target.closest('.col-resize')) return sortBy(host, head.getAttribute('data-col') || '');
  const pager = target.closest('[data-page]');
  if (pager) return host.turnPage(pager.getAttribute('data-page'));
  const action = target.closest('[data-action]');
  if (action) return runAction(host, action.getAttribute('data-action'));

  // A click anywhere else on a row makes it the keyboard's row.
  const row = target.closest(ROWS);
  if (row instanceof HTMLElement && !target.closest('input, button, a')) row.focus();
}

/**
 * Every right-click the finder answers. A surface with verbs opens its menu and
 * suppresses the browser's; anywhere else — the search box above all, where the
 * native menu pastes — the browser's own menu shows.
 * @param {MouseEvent} event
 * @param {FinderHost} host
 * @returns {void}
 */
function onContextMenu(event, host) {
  const target = event.target;
  if (!(target instanceof Element) || target.closest('input[type="search"]')) return;
  const menu = menuFor(target, host);
  if (!menu) return;
  event.preventDefault();
  openContextMenu({ x: event.clientX, y: event.clientY, anchorEl: host.pane(), ...menu });
}

/**
 * The menu a surface offers, or null for a surface with none.
 * @param {Element} target
 * @param {FinderHost} host
 * @returns {{ariaLabel: string, items: MenuItem[]}|null}
 */
export function menuFor(target, host) {
  const ctx = menuContext(host);

  const head = target.closest(DATA_HEADS);
  if (head) {
    const key = head.getAttribute('data-col') || '';
    return { ariaLabel: `${key} column`, items: headerMenu(ctx, key) };
  }
  const row = target.closest(ROWS);
  if (row) {
    const pv = row.getAttribute('data-pv') || '';
    const data = (host.results()?.rows || []).find((/** @type {any} */ r) => String(r.fullPv ?? '') === pv);
    if (data) return { ariaLabel: `${pv} actions`, items: rowMenu(ctx, data) };
  }
  const binding = target.closest('.device-card .sig-table tr[data-pv]');
  if (binding) {
    const pv = binding.getAttribute('data-pv') || '';
    return { ariaLabel: `${pv} actions`, items: bindingMenu(ctx, pv) };
  }
  const card = host.card();
  if (target.closest('.device-card-head') && card) {
    const name = String(card.data?.device ?? '');
    return { ariaLabel: `${name || 'Device'} actions`, items: inspectorMenu(ctx, name, deviceAddresses(card.data)) };
  }
  const facet = target.closest('button.facet-item[data-facet][data-value]');
  if (facet) {
    const name = /** @type {Facet} */ (facet.getAttribute('data-facet'));
    const value = facet.getAttribute('data-value') || '';
    const label = facet.querySelector('.k')?.textContent || value;
    return { ariaLabel: `${label} filter`, items: facetValueMenu(ctx, name, value, label) };
  }
  const facetHead = target.closest('.facet-head');
  if (facetHead) {
    const key = /** @type {Facet} */ (facetHead.closest('.facet')?.getAttribute('data-facet-group') || 'section');
    return { ariaLabel: 'Filter group', items: facetHeadMenu(ctx, key) };
  }
  const chip = target.closest('button.active-filter[data-chip]');
  if (chip) return { ariaLabel: 'Filter', items: chipMenu(ctx, chip.getAttribute('data-chip') || '') };
  return null;
}

/**
 * Keys inside the panel; see the module docstring for the model.
 * @param {KeyboardEvent} event
 * @param {FinderHost} host
 * @returns {void}
 */
function onKeydown(event, host) {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;

  if (isContextMenuKey(event)) {
    const menu = menuFor(target, host);
    if (!menu) return;
    event.preventDefault();
    openContextMenuAt(target, menu);
    return;
  }

  if (target.matches(DATA_HEADS)) {
    const key = target.getAttribute('data-col') || '';
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      sortBy(host, key);
    } else if (event.altKey && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {
      event.preventDefault();
      if (nudgeColumn(host.layout(), key, event.key === 'ArrowLeft' ? -1 : 1)) {
        host.commitColumns();
        /** @type {HTMLElement|null} */ (host.pane()?.querySelector(`.result-table thead th[data-col="${CSS.escape(key)}"]`) ?? null)?.focus();
      }
    }
    return;
  }

  if (!target.matches(ROWS)) return;
  const pv = target.getAttribute('data-pv') || '';
  const rows = /** @type {HTMLElement[]} */ ([...(host.pane()?.querySelectorAll(ROWS) ?? [])]);
  const at = rows.indexOf(target);
  const go = (/** @type {number} */ i) => rows[Math.max(0, Math.min(rows.length - 1, i))]?.focus();

  if (event.key === 'ArrowDown') go(at + 1);
  else if (event.key === 'ArrowUp') go(at - 1);
  else if (event.key === 'Home') go(0);
  else if (event.key === 'End') go(rows.length - 1);
  else if (event.key === ' ') createActions(host).toggleSelected(pv);
  else if (event.key === 'Enter') void host.toggleDevice(target.getAttribute('data-uri') || '');
  else if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'c'
    && !String(window.getSelection?.() ?? '')) void host.copy(pv, pv);
  else return;
  event.preventDefault();
}

/**
 * A row taking focus becomes the table's Tab stop, so Tab back into the table
 * lands where the operator left it.
 * @param {FocusEvent} event
 * @param {FinderHost} host
 * @returns {void}
 */
function onFocusIn(event, host) {
  const target = event.target;
  if (!(target instanceof HTMLElement) || !target.matches(ROWS)) return;
  for (const row of host.pane()?.querySelectorAll(ROWS) ?? []) {
    row.setAttribute('tabindex', row === target ? '0' : '-1');
  }
}

/**
 * Document-level keys while the finder is mounted: `/` jumps to the search box
 * and Escape closes the inspector. Neither fires while the operator is typing,
 * and an Escape the context menu already handled never reaches here.
 * @param {KeyboardEvent} event
 * @param {FinderHost} host
 * @returns {void}
 */
function onDocKeydown(event, host) {
  if (!host.pane() || event.defaultPrevented) return;
  const active = document.activeElement;
  if (active instanceof HTMLElement && (active.matches('input, textarea, select') || active.isContentEditable)) return;
  if (event.key === '/' && !event.metaKey && !event.ctrlKey && !event.altKey) {
    const input = /** @type {HTMLInputElement|null} */ (host.part(host.searchInputId));
    // Hidden when the hub's tile bar carries the search box instead.
    if (input && input.getClientRects().length > 0) {
      event.preventDefault();
      input.focus();
      input.select();
    }
  } else if (event.key === 'Escape' && host.card()) {
    event.preventDefault();
    host.closeCard();
  }
}
