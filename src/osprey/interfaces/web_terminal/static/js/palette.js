// @ts-check
/* OSPREY Web Terminal — Command Palette Overlay
 *
 * The Cmd/Ctrl+K modal: a body-appended scrim + dialog that fuzzy-searches the
 * grouped registry (Settings / Panels / Layouts / Actions) and runs the picked
 * item. UI-only — the non-config app dependencies (panel/preset getters, action
 * closures, navigation callbacks) are injected via `openPalette(deps)`; this
 * module supplies just the /api/config fetch (its ONLY network call, injectable
 * as `deps.fetchConfig` for tests). The config load runs concurrently with the
 * open: the palette renders immediately with a "Loading settings…" decoration,
 * then rebuilds against the CURRENT query once the fetch settles.
 *
 * Highlight-vs-rank split: items are RANKED by fuzzyMatch(query, searchText)
 * (panels carry extra tokens like the panel id in searchText), but the visible
 * highlight spans come from fuzzyMatch(query, label) so only characters that
 * actually appear in the label get marked. A query that matches searchText but
 * not the label simply renders an un-highlighted label.
 *
 * The keydown handler is installed on the document in CAPTURE phase and calls
 * stopPropagation for the keys it owns, so palette Escape/arrows never reach
 * osprey-drawer's Escape-closes-drawers handler or the terminal underneath.
 */

import { fuzzyMatch } from './fuzzy.js';
import { buildRegistry } from './palette-registry.js';
import { fetchJSON } from './api.js';
import { el } from '/design-system/js/dom.js';

/** @typedef {import('./palette-registry.js').Item} Item */

/**
 * A navigable registry item (the non-status variant), duplicated locally so the
 * render path can be strongly typed after the status rows are filtered out.
 * @typedef {{
 *   group: 'Settings' | 'Panels' | 'Layouts' | 'Actions',
 *   label: string,
 *   detail?: string,
 *   searchText: string,
 *   run: () => void,
 * }} NavItem
 */

/**
 * The NON-config dependency bundle the app injects. The config portion is
 * supplied by this module via the fetch flow; `fetchConfig` overrides the
 * default GET /api/config (used by tests to avoid the network).
 * @typedef {{
 *   getHiddenPanels?: () => Array<{ id: string, label: string }>,
 *   getVisiblePanels?: () => Array<{ id: string, label: string }>,
 *   getPresets?: () => Array<{ name: string, panels: string[] }>,
 *   showPanel?: (id: string) => void,
 *   focusPanel?: (id: string) => void,
 *   applyPreset?: (panels: string[]) => void,
 *   revealSetting?: (dotKey: string) => void,
 *   actions?: Array<{ label: string, detail?: string, run: () => void }>,
 *   fetchConfig?: () => Promise<any>,
 * }} OpenDeps
 */

/** @typedef {{ state: 'loading' } | { state: 'ok', sections: Record<string, unknown> } | { state: 'error' }} ConfigState */

/** Fixed outer group order — matches the registry and the stylesheet. */
const GROUP_ORDER = /** @type {const} */ (['Settings', 'Panels', 'Layouts', 'Actions']);

/** The module's ONLY network call: fetch the config snapshot. */
const defaultFetchConfig = () => fetchJSON('/api/config');

/** @type {HTMLElement | null} */
let overlayEl = null;
/** @type {HTMLInputElement | null} */
let inputEl = null;
/** @type {HTMLElement | null} */
let listEl = null;

let opened = false;
/** @type {HTMLElement | null} */
let previousFocus = null;
/** @type {OpenDeps} */
let currentDeps = {};
/** @type {() => Promise<any>} */
let fetchConfigFn = defaultFetchConfig;
/** @type {ConfigState} */
let configState = { state: 'loading' };
/** @type {Item[]} */
let registryItems = [];
/** @type {Array<{ key: string, run: () => void, el: HTMLElement }>} */
let navItems = [];
let activeIndex = -1;
/** @type {string | null} */
let activeKey = null;
let currentQuery = '';
let fetchSeq = 0;
let closeSeq = 0;

/** @returns {boolean} True when the first-visit welcome modal is present + visible. */
function isWelcomeVisible() {
  const w = document.getElementById('welcome-overlay');
  return !!w && !w.classList.contains('hidden');
}

/**
 * Stable identity for an item across registry rebuilds (group + label). The
 * \x1f (unit separator) delimiter cannot occur in a group name or label, so
 * the key is unambiguous.
 */
function itemKey(/** @type {NavItem} */ item) {
  return `${item.group}\x1f${item.label}`;
}

/** Lazily create the single reused overlay node and (re)attach it to <body>. */
function ensureOverlay() {
  if (!overlayEl) {
    overlayEl = el('div', 'command-palette-overlay');
    overlayEl.addEventListener('mousedown', (e) => {
      if (e.target === overlayEl) closePalette();
    });
  }
  if (!overlayEl.parentNode) document.body.appendChild(overlayEl);
  return overlayEl;
}

/** Rebuild the dialog's inner content (input + list) from scratch on each open. */
function buildDialog() {
  const root = ensureOverlay();
  root.textContent = '';

  const dialog = el('div', 'command-palette-dialog');
  dialog.setAttribute('role', 'dialog');
  dialog.setAttribute('aria-modal', 'true');
  dialog.setAttribute('aria-label', 'Command palette');

  const input = /** @type {HTMLInputElement} */ (el('input', 'command-palette-input'));
  input.type = 'text';
  input.setAttribute('role', 'combobox');
  input.setAttribute('aria-expanded', 'true');
  input.setAttribute('aria-autocomplete', 'list');
  input.setAttribute('placeholder', 'Search settings, panels, layouts, actions…');
  const listId = 'command-palette-listbox';
  input.setAttribute('aria-controls', listId);
  input.addEventListener('input', () => {
    currentQuery = input.value;
    render();
  });
  inputEl = input;

  const list = el('div', 'command-palette-list');
  list.id = listId;
  list.setAttribute('role', 'listbox');
  listEl = list;

  dialog.appendChild(input);
  dialog.appendChild(list);
  root.appendChild(dialog);
}

/** Rebuild the registry from the current deps + config state. */
function rebuildRegistry() {
  registryItems = buildRegistry({
    config: configState,
    getHiddenPanels: currentDeps.getHiddenPanels,
    getVisiblePanels: currentDeps.getVisiblePanels,
    getPresets: currentDeps.getPresets,
    showPanel: currentDeps.showPanel,
    focusPanel: currentDeps.focusPanel,
    applyPreset: currentDeps.applyPreset,
    revealSetting: currentDeps.revealSetting,
    actions: currentDeps.actions,
  });
}

/**
 * Append `text` to `container` with `.command-palette-match` spans over the
 * fuzzy-matched LABEL ranges (empty query or a label that does not match falls
 * back to a plain text node).
 * @param {HTMLElement} container
 * @param {string} text
 * @param {string} query
 */
function appendHighlighted(container, text, query) {
  const res = query ? fuzzyMatch(query, text) : null;
  if (!res || res.spans.length === 0) {
    container.textContent = text;
    return;
  }
  let cursor = 0;
  for (const [start, end] of res.spans) {
    if (start > cursor) container.appendChild(document.createTextNode(text.slice(cursor, start)));
    const mark = el('span', 'command-palette-match');
    mark.textContent = text.slice(start, end);
    container.appendChild(mark);
    cursor = end;
  }
  if (cursor < text.length) container.appendChild(document.createTextNode(text.slice(cursor)));
}

/**
 * Build one navigable option row (role="option", unique id, label + optional
 * detail, click-to-run, hover-to-activate).
 * @param {NavItem} item
 * @param {string} id
 * @param {string} query
 * @returns {HTMLElement}
 */
function buildItemRow(item, id, query) {
  const row = el('div', 'command-palette-item');
  row.id = id;
  row.setAttribute('role', 'option');
  row.setAttribute('aria-selected', 'false');

  const label = el('span', 'command-palette-item-label');
  appendHighlighted(label, item.label, query);
  row.appendChild(label);

  if (item.detail) {
    const detail = el('span', 'command-palette-item-detail');
    detail.textContent = item.detail;
    row.appendChild(detail);
  }

  const run = item.run;
  row.addEventListener('click', () => {
    closePalette();
    run();
  });
  row.addEventListener('mouseenter', () => {
    const i = navItems.findIndex((n) => n.el === row);
    if (i !== -1 && i !== activeIndex) setActive(i);
  });
  return row;
}

/**
 * Recompute the filtered/sorted results for the current query and repaint the
 * list. Group order (Settings → Panels → Layouts → Actions) is the outer
 * ordering; within each group navigable items sort by descending score. Status
 * decorations always render under Settings but are excluded from navigation.
 */
function render() {
  if (!listEl) return;
  listEl.textContent = '';
  navItems = [];
  const q = currentQuery.trim();
  let optSeq = 0;
  let renderedStatus = false;

  for (const group of GROUP_ORDER) {
    /** @type {Array<{ node: HTMLElement, nav?: { key: string, run: () => void } }>} */
    const rows = [];
    /** @type {Array<{ item: NavItem, score: number }>} */
    const matches = [];

    for (const item of registryItems) {
      if (item.group !== group) continue;
      if ('status' in item) {
        const status = el('div', 'command-palette-status');
        status.textContent = item.label;
        rows.push({ node: status });
        renderedStatus = true;
        continue;
      }
      const res = fuzzyMatch(q, item.searchText);
      if (res) matches.push({ item, score: res.score });
    }

    matches.sort((a, b) => b.score - a.score);
    for (const { item } of matches) {
      const id = `command-palette-opt-${optSeq++}`;
      rows.push({ node: buildItemRow(item, id, q), nav: { key: itemKey(item), run: item.run } });
    }

    if (rows.length === 0) continue;
    const heading = el('div', 'command-palette-group-heading');
    heading.textContent = group;
    listEl.appendChild(heading);
    for (const row of rows) {
      listEl.appendChild(row.node);
      if (row.nav) navItems.push({ key: row.nav.key, run: row.nav.run, el: row.node });
    }
  }

  if (navItems.length === 0 && !renderedStatus) {
    const empty = el('div', 'command-palette-empty');
    empty.textContent = 'No matches';
    listEl.appendChild(empty);
  }

  let idx = activeKey ? navItems.findIndex((n) => n.key === activeKey) : -1;
  if (idx === -1) idx = navItems.length ? 0 : -1;
  setActive(idx);
}

/**
 * Mark `index` as the active navigable row: toggle the active class + aria on
 * every row, scroll it into view, and point aria-activedescendant at its id.
 * @param {number} index
 */
function setActive(index) {
  activeIndex = index;
  activeKey = index >= 0 && navItems[index] ? navItems[index].key : null;
  for (let i = 0; i < navItems.length; i++) {
    const on = i === index;
    navItems[i].el.classList.toggle('command-palette-item--active', on);
    navItems[i].el.setAttribute('aria-selected', on ? 'true' : 'false');
  }
  if (!inputEl) return;
  if (index >= 0 && navItems[index]) {
    inputEl.setAttribute('aria-activedescendant', navItems[index].el.id);
    navItems[index].el.scrollIntoView({ block: 'nearest' });
  } else {
    inputEl.removeAttribute('aria-activedescendant');
  }
}

/**
 * Capture-phase document keydown handler. Owns Escape (close), Arrow Up/Down
 * (wrap-move the active row) and Enter (run + close); for each it stops
 * propagation + default so the event never reaches the drawer or terminal.
 * @param {KeyboardEvent} e
 */
function onKeydown(e) {
  if (!opened) return;
  const len = navItems.length;
  switch (e.key) {
    case 'Escape':
      e.preventDefault();
      e.stopPropagation();
      closePalette();
      return;
    case 'ArrowDown':
      e.preventDefault();
      e.stopPropagation();
      if (len) setActive((activeIndex + 1 + len) % len);
      return;
    case 'ArrowUp':
      e.preventDefault();
      e.stopPropagation();
      if (len) setActive((activeIndex - 1 + len) % len);
      return;
    case 'Enter':
      e.preventDefault();
      e.stopPropagation();
      if (activeIndex >= 0 && navItems[activeIndex]) {
        // Close (which restores focus) BEFORE running, matching the click path,
        // so an action that moves focus is not undone by focus restore.
        const run = navItems[activeIndex].run;
        closePalette();
        run();
      }
      return;
    case 'Tab':
      // The overlay's only focusable node is the input, so trap Tab to keep
      // focus inside the modal — honoring aria-modal without a full focus trap.
      e.preventDefault();
      e.stopPropagation();
      return;
    default:
  }
}

/**
 * Kick off the config fetch concurrently with the open. On resolve/reject it
 * rebuilds the registry and re-renders against the CURRENT query, but only if
 * this is still the newest fetch and the palette is still open.
 */
function startConfigFetch() {
  const token = ++fetchSeq;
  Promise.resolve()
    .then(() => fetchConfigFn())
    .then((payload) => {
      if (token !== fetchSeq || !opened) return;
      const sections = payload && typeof payload === 'object' && payload.sections ? payload.sections : {};
      configState = { state: 'ok', sections };
      rebuildRegistry();
      render();
    })
    .catch(() => {
      if (token !== fetchSeq || !opened) return;
      configState = { state: 'error' };
      rebuildRegistry();
      render();
    });
}

/** Restore focus to the element that was focused before the palette opened. */
function restoreFocus() {
  if (previousFocus && document.contains(previousFocus)) previousFocus.focus();
  previousFocus = null;
}

/**
 * Open the command palette. No-op (re-focuses input) if already open; a no-op
 * entirely if the welcome modal is present + visible (its any-Enter dismiss
 * handler would collide).
 * @param {OpenDeps} [deps]
 */
export function openPalette(deps) {
  if (isWelcomeVisible()) return;
  if (opened) {
    if (inputEl) inputEl.focus();
    return;
  }
  currentDeps = deps || {};
  fetchConfigFn = typeof currentDeps.fetchConfig === 'function'
    ? currentDeps.fetchConfig
    : defaultFetchConfig;
  previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  configState = { state: 'loading' };
  currentQuery = '';
  activeKey = null;
  opened = true;

  buildDialog();
  const root = ensureOverlay();
  requestAnimationFrame(() => root.classList.add('visible'));
  document.addEventListener('keydown', onKeydown, true);
  if (inputEl) inputEl.focus();

  rebuildRegistry();
  render();
  startConfigFetch();
}

/** Close the palette and restore focus to the previously-focused element. */
export function closePalette() {
  if (!opened) return;
  opened = false;
  document.removeEventListener('keydown', onKeydown, true);

  const node = overlayEl;
  const seq = ++closeSeq;
  if (node) {
    node.classList.remove('visible');
    const removeNode = () => {
      if (seq === closeSeq && !opened && node.parentNode) node.remove();
    };
    node.addEventListener('transitionend', removeNode, { once: true });
    setTimeout(removeNode, 300);
  }
  restoreFocus();
}

/** @returns {boolean} Whether the palette is currently open. */
export function isOpen() {
  return opened;
}
