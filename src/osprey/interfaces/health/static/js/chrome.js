// @ts-check
/*
 * System Health dashboard — the panel chrome (DESIGN.md §12, "Panel chrome
 * standard").
 *
 * - The toolbar: the All / OK / Warn / Err / Skip status filter (an
 *   `.osprey-seg` with counts) and the Scan now icon, whose tooltip carries
 *   the countdown to the next automatic scan.
 * - The categories sidebar: one `.osprey-list-row` per check category with
 *   its worst-status dot and its passed/total count.
 * - The tile ⋯ "This panel" entries (one protocol-v3 `menu` contribution):
 *   Expand all details, Copy summary, Copy on-demand CLI commands. The
 *   panel contributes nothing else to the hub's tile bar. A standalone or
 *   popped-out page has no tile bar, so there the toolbar ends in its own ⋯
 *   that lists the same entries and runs the same handlers.
 *
 * This module builds and updates that chrome and the clipboard texts; the
 * dashboard (dashboard.js) owns the data and decides what each action does.
 * Status colours stay CSS classes (`dot s-ok` …), never inline colours.
 */

import { openContextMenu } from '/design-system/js/context-menu.js';
import { contributeHeader, onHeaderAction } from '/design-system/js/header-contrib.js';
import { icon } from '/design-system/js/icons.js';
import { writeTitle } from '/design-system/js/tooltip.js';
import { el, fmtName, worstStatus } from './helpers.js';

/** The id of the panel's one `menu` contribution. */
export const MENU_ID = 'health';

/** The ids of the menu entries; `onHeaderAction` delivers them as `value`. */
export const MENU_ENTRY = Object.freeze({
  expand: 'expand-all',
  summary: 'copy-summary',
  cli: 'copy-cli',
});

/**
 * The filter buttons, in order. `status` null is All. `key` names the
 * envelope count shown beside the label.
 *
 * @type {ReadonlyArray<{status: string | null, label: string, key: 'total' | 'ok' | 'warnings' | 'errors' | 'skips'}>}
 */
export const FILTERS = Object.freeze([
  { status: null, label: 'All', key: 'total' },
  { status: 'ok', label: 'OK', key: 'ok' },
  { status: 'warning', label: 'Warn', key: 'warnings' },
  { status: 'error', label: 'Err', key: 'errors' },
  { status: 'skip', label: 'Skip', key: 'skips' },
]);

/** @type {Record<string, string>} */
const DOT_CLS = { ok: 's-ok', warning: 's-wn', error: 's-er', skip: 's-sk' };

/**
 * @typedef {{total: number, ok: number, warnings: number, errors: number, skips: number}} Counts
 * @typedef {{name: string, category: string, status: string, message: string}} Check
 * @typedef {{cat: string, checks: Check[], onDemand: boolean}} Group
 */

// -- toolbar -----------------------------------------------------------------

/**
 * Fill the `.osprey-seg` with the five filter buttons. A press calls
 * `onPick(status)` (null = All); the dashboard re-renders and calls
 * `update` with the new state. The buttons are built once, so keyboard
 * focus survives every refresh.
 *
 * @param {HTMLElement} seg
 * @param {(status: string | null) => void} onPick
 * @returns {{ update(counts: Counts | null, active: string | null): void }}
 */
export function initFilter(seg, onPick) {
  /** @type {Array<{button: HTMLElement, count: HTMLElement, filter: typeof FILTERS[number]}>} */
  const parts = FILTERS.map((filter) => {
    const count = el('span', { class: 'health-seg-count' });
    const button = el('button', { type: 'button', 'data-status': filter.status || 'all' }, [
      document.createTextNode(filter.label),
      count,
    ]);
    button.addEventListener('click', () => onPick(filter.status));
    seg.appendChild(button);
    return { button, count, filter };
  });
  /** @type {{ update(counts: Counts | null, active: string | null): void }} */
  const api = {
    update(counts, active) {
      for (const { button, count, filter } of parts) {
        button.setAttribute('aria-pressed', String(filter.status === active));
        count.textContent = counts ? String(counts[filter.key]) : '';
      }
    },
  };
  api.update(null, null);
  return api;
}

/**
 * The Scan now tooltip.
 *
 * @param {number | null} seconds to the next automatic scan; null when unknown
 * @returns {string}
 */
export function scanTitle(seconds) {
  if (seconds == null) return 'Scan now';
  return 'Scan now (next automatic scan in ' + Math.max(0, seconds) + ' s)';
}

/**
 * Make the toolbar's Scan now icon button. Its accessible name stays
 * "Scan now"; the countdown goes in the tooltip through `writeTitle`.
 *
 * @param {HTMLElement} button
 * @param {() => void} onScan
 * @returns {{ setCountdown(seconds: number | null): void }}
 */
export function initScanButton(button, onScan) {
  if (!button.querySelector('svg')) button.appendChild(icon('refresh'));
  button.setAttribute('aria-label', 'Scan now');
  writeTitle(button, scanTitle(null));
  button.addEventListener('click', () => onScan());
  return {
    setCountdown(seconds) {
      writeTitle(button, scanTitle(seconds));
    },
  };
}

// -- sidebar -----------------------------------------------------------------

/**
 * A category id as a sidebar label: `file_system` → "File system".
 *
 * @param {string} cat
 * @returns {string}
 */
export function catLabel(cat) {
  const text = cat.replace(/_/g, ' ');
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/**
 * The count a category row shows: passed/total, or "on demand".
 *
 * @param {Group} group
 * @returns {string}
 */
export function catCount({ checks, onDemand }) {
  if (onDemand) return 'on demand';
  return checks.filter((c) => c.status === 'ok').length + '/' + checks.length;
}

/**
 * Replace the categories list with one row per group.
 *
 * @param {HTMLElement} list
 * @param {Group[]} groups
 * @param {string | null} selected the selected category, if any
 * @param {(cat: string) => void} onSelect
 */
export function renderCategories(list, groups, selected, onSelect) {
  list.textContent = '';
  for (const group of groups) {
    const status = group.onDemand ? 'skip' : worstStatus(group.checks);
    const row = el(
      'button',
      {
        type: 'button',
        class: 'osprey-list-row',
        'data-category': group.cat,
        'aria-selected': String(group.cat === selected),
      },
      [
        el('span', { class: 'dot ' + (DOT_CLS[status] || 's-sk') }),
        el('span', { class: 'osprey-list-title', text: catLabel(group.cat) }),
        el('span', { class: 'health-cat-count', text: catCount(group) }),
      ]
    );
    row.addEventListener('click', () => onSelect(group.cat));
    list.appendChild(row);
  }
}

// -- tile ⋯ menu ----------------------------------------------------------------

/**
 * @typedef {object} MenuState
 * @property {boolean} hasData a report is shown
 * @property {boolean} hasDetails at least one row has expandable details
 * @property {boolean} allOpen every expandable row is open
 * @property {boolean} hasCli at least one on-demand category is listed
 */

/**
 * The "This panel" entries for the current state.
 *
 * @param {MenuState} state
 * @returns {MenuEntry[]}
 */
export function menuItems({ hasData, hasDetails, allOpen, hasCli }) {
  return [
    {
      id: MENU_ENTRY.expand,
      label: hasDetails && allOpen ? 'Collapse all details' : 'Expand all details',
      disabled: !hasDetails,
    },
    { id: MENU_ENTRY.summary, label: 'Copy summary', disabled: !hasData },
    { id: MENU_ENTRY.cli, label: 'Copy on-demand CLI commands', disabled: !hasCli },
  ];
}

/**
 * @typedef {{id: string, label: string, checked?: boolean, disabled?: boolean,
 *   danger?: boolean, separator?: boolean}} MenuEntry
 */

/** @type {string | null} */
let lastMenu = null;
/** The entries of the last contribution: what the standalone ⋯ lists. */
let currentEntries = menuItems({
  hasData: false,
  hasDetails: false,
  allOpen: false,
  hasCli: false,
});
/** @type {Partial<Record<string, () => void>>} */
let entryHandlers = {};

/**
 * Send the panel's whole contribution: the one `menu`. A contribution equal
 * to the last one is not sent again. No-op standalone (no tile bar).
 *
 * @param {MenuState} state
 */
export function contributeMenu(state) {
  currentEntries = menuItems(state);
  const items = [{ kind: /** @type {'menu'} */ ('menu'), id: MENU_ID, items: currentEntries }];
  const wire = JSON.stringify(items);
  if (wire === lastMenu) return;
  lastMenu = wire;
  contributeHeader(items);
}

/** @param {string} entryId */
function runEntry(entryId) {
  const handler = entryHandlers[entryId];
  if (handler) handler();
}

/**
 * Route the menu-entry clicks, from the hub's tile ⋯ or the standalone ⋯,
 * to `handlers[entryId]`.
 *
 * @param {Partial<Record<string, () => void>>} handlers keyed by MENU_ENTRY values
 */
export function onMenuAction(handlers) {
  entryHandlers = handlers;
  onHeaderAction((id, value) => {
    if (id !== MENU_ID || !value) return;
    runEntry(value);
  });
}

/**
 * The context-menu rows for the menu entries: the same labels and states,
 * with a divider before an entry that asks for a separator.
 *
 * @param {MenuEntry[]} entries
 * @returns {import('/design-system/js/context-menu.js').MenuItem[]}
 */
export function contextItems(entries) {
  return entries.flatMap((entry) => {
    const row = {
      label: entry.label,
      checked: entry.checked,
      disabled: entry.disabled,
      danger: entry.danger,
      run: () => runEntry(entry.id),
    };
    return entry.separator ? [{ divider: true }, row] : [row];
  });
}

/**
 * On a standalone or popped-out page (no hub tile bar), append the panel's
 * own ⋯ to `host`, the toolbar's end. Embedded in the hub it adds nothing:
 * a tile has one ⋯, the hub's.
 *
 * @param {HTMLElement} host
 * @returns {HTMLElement | null} the button, or null when embedded
 */
export function initStandaloneMenu(host) {
  if (document.body.classList.contains('embedded')) return null;
  const button = el('button', {
    type: 'button',
    class: 'osprey-icon-btn',
    title: 'Panel menu',
    'aria-label': 'Panel menu',
    'aria-haspopup': 'menu',
  });
  button.appendChild(icon('more'));
  button.addEventListener('click', () => {
    const rect = button.getBoundingClientRect();
    openContextMenu({
      x: rect.right,
      y: rect.bottom,
      anchorEl: button,
      triggerEl: button,
      ariaLabel: 'Panel menu',
      items: contextItems(currentEntries),
    });
  });
  host.appendChild(button);
  return button;
}

// -- clipboard texts -------------------------------------------------------------

/**
 * "12 warnings, 1 error, 6 skipped"; empty when everything passed.
 *
 * @param {Counts} counts
 * @returns {string}
 */
export function countsText({ warnings, errors, skips }) {
  /** @type {string[]} */
  const parts = [];
  if (warnings) parts.push(warnings + ' warning' + (warnings !== 1 ? 's' : ''));
  if (errors) parts.push(errors + ' error' + (errors !== 1 ? 's' : ''));
  if (skips) parts.push(skips + ' skipped');
  return parts.join(', ');
}

/**
 * The plain-text report Copy summary puts on the clipboard: the title and
 * summary line, the stale and deadline notes, then one line per error and
 * per warning.
 *
 * @param {Counts & {title?: string, summary: string, stale?: boolean, deadline_hit?: boolean, results: Check[]}} d
 * @returns {string}
 */
export function summaryText(d) {
  const lines = [(d.title || 'System Health') + ': ' + d.summary];
  if (d.stale) lines.push('Data may be stale.');
  if (d.deadline_hit) lines.push('The scan hit its deadline.');
  /** @type {Array<[string, string]>} */
  const order = [
    ['error', 'Error'],
    ['warning', 'Warning'],
  ];
  /** @type {string[]} */
  const problems = [];
  for (const [status, label] of order) {
    for (const c of d.results) {
      if (c.status !== status) continue;
      problems.push(
        label +
          ': ' +
          catLabel(c.category) +
          ' · ' +
          fmtName(c.name, c.category) +
          ' — ' +
          c.message
      );
    }
  }
  if (problems.length) lines.push('', ...problems);
  return lines.join('\n');
}

/**
 * The CLI command that runs an on-demand category.
 *
 * @param {string} cat
 * @returns {string}
 */
export function cliCommand(cat) {
  return 'osprey health --full --category ' + cat;
}

/**
 * One CLI command per on-demand category, one per line.
 *
 * @param {Group[]} groups
 * @returns {string}
 */
export function cliCommands(groups) {
  return groups
    .filter((g) => g.onDemand)
    .map((g) => cliCommand(g.cat))
    .join('\n');
}
