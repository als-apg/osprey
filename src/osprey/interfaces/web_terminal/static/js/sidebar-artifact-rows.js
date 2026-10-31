// @ts-check
/* OSPREY Web Terminal — the rows of the sidebar's artifact lists (layout F)
 *
 * Builds what the Artifacts and Script runs lists draw: group heads, one row
 * per artifact (kind icon, title, time) and the line an empty list shows. The
 * rows and heads are the shared list primitives of primitives.css
 * (`.osprey-list-row`, `.osprey-list-head`, `.osprey-list-empty`); a row is a
 * listbox option whose `aria-selected` marks the artifact the viewer shows.
 *
 * Two groupings: Types maps the artifact type to Reports, Plots, Tables,
 * Notebooks, Data or Other; Activity sorts by the facility day: Today,
 * Yesterday, Earlier. Inside a group the newest comes first. All text goes in
 * through textContent.
 *
 * @module sidebar-artifact-rows
 */

import { icon } from '/design-system/js/icons.js';
import { facilityDayKey, formatFacilityTime } from '/design-system/js/facility-time.js';

/** @typedef {import('./sidebar-artifact-listing.js').ArtifactEntry} ArtifactEntry */

/** @typedef {'types' | 'activity'} ListView */

/**
 * The Types groups, in the order they are drawn, with the artifact types each
 * holds. A type no group names (a newer one) falls into Other.
 * @type {readonly {label: string, icon: string, types: readonly string[]}[]}
 */
export const KINDS = [
  { label: 'Reports', icon: 'file-text', types: ['markdown', 'text', 'html'] },
  {
    label: 'Plots',
    icon: 'chart-line',
    types: ['plot_html', 'plot_png', 'image', 'dashboard_html'],
  },
  { label: 'Tables', icon: 'table', types: ['table_html'] },
  { label: 'Notebooks', icon: 'notebook', types: ['notebook'] },
  { label: 'Data', icon: 'database', types: ['json', 'file', 'binary'] },
  { label: 'Other', icon: 'layers', types: [] },
];

/** The icon of every Script runs row. */
export const RUN_ICON = 'bash';

/**
 * @typedef {object} RowsState
 * @property {string | null} shownId  the artifact the viewer shows
 * @property {boolean} loaded         a first page has arrived
 * @property {boolean} failed         the last first page failed
 */

/**
 * @param {ArtifactEntry} entry
 * @returns {string}
 */
export function titleOf(entry) {
  return entry.title || entry.filename || entry.id;
}

/**
 * The Types group of an artifact.
 * @param {ArtifactEntry} entry
 */
export function kindOf(entry) {
  const type = entry.artifact_type ?? '';
  return KINDS.find((kind) => kind.types.includes(type)) ?? KINDS[KINDS.length - 1];
}

/**
 * Group newest-first entries for a view: by type group in KINDS order, or by
 * facility day. Empty groups are kept, so callers can rely on the order.
 * @param {ArtifactEntry[]} entries
 * @param {ListView} view
 * @returns {{label: string, entries: ArtifactEntry[]}[]}
 */
export function groupEntries(entries, view) {
  if (view === 'types') {
    return KINDS.map((kind) => ({
      label: kind.label,
      entries: entries.filter((a) => kindOf(a) === kind),
    }));
  }
  const today = facilityDayKey(new Date());
  const yesterday = dayBefore(today);
  /** @type {ArtifactEntry[]} */
  const todays = [];
  /** @type {ArtifactEntry[]} */
  const yesterdays = [];
  /** @type {ArtifactEntry[]} */
  const earlier = [];
  for (const entry of entries) {
    const day = facilityDayKey(entry.timestamp);
    if (day === today) todays.push(entry);
    else if (day === yesterday) yesterdays.push(entry);
    else earlier.push(entry);
  }
  return [
    { label: 'Today', entries: todays },
    { label: 'Yesterday', entries: yesterdays },
    { label: 'Earlier', entries: earlier },
  ];
}

/**
 * The entries in the order a view draws them.
 * @param {ArtifactEntry[]} entries  newest first
 * @param {ListView} view
 * @returns {ArtifactEntry[]}
 */
export function drawnOrder(entries, view) {
  return groupEntries(entries, view).flatMap((group) => group.entries);
}

/**
 * The civil day before a `YYYY-MM-DD` key, by calendar arithmetic: counting
 * back 24 hours lands two days back on the morning after a 23-hour day.
 * @param {string} dayKey
 * @returns {string}
 */
function dayBefore(dayKey) {
  const [y, m, d] = dayKey.split('-').map(Number);
  return new Date(Date.UTC(y, m - 1, d - 1)).toISOString().slice(0, 10);
}

/**
 * The Artifacts list: group heads and rows, narrowed by a title filter.
 * @param {ArtifactEntry[]} entries  newest first
 * @param {RowsState & {view: ListView, filter: string}} options
 * @returns {Node[]}
 */
export function artifactRows(entries, { view, filter, shownId, loaded, failed }) {
  const query = filter.trim().toLowerCase();
  const visible = query ? entries.filter((a) => titleOf(a).toLowerCase().includes(query)) : entries;
  /** @type {Node[]} */
  const nodes = [];
  for (const group of groupEntries(visible, view)) {
    if (!group.entries.length) continue;
    nodes.push(listHead(group.label));
    for (const entry of group.entries) nodes.push(listRow(entry, kindOf(entry).icon, shownId));
  }
  if (nodes.length || !loaded) return nodes;
  if (query && entries.length) return [listEmpty('No artifacts match')];
  return [listEmpty(failed ? 'Artifacts are unavailable' : 'No artifacts yet')];
}

/**
 * The Script runs list: its rows, newest first, with no groups.
 * @param {ArtifactEntry[]} entries  newest first
 * @param {RowsState} options
 * @returns {Node[]}
 */
export function runRows(entries, { shownId, loaded, failed }) {
  if (entries.length) return entries.map((entry) => listRow(entry, RUN_ICON, shownId));
  if (!loaded) return [];
  return [listEmpty(failed ? 'Script runs are unavailable' : 'No script runs yet')];
}

/**
 * One row: the kind icon, the title (also the tooltip, since a long one
 * ellipsizes) and the time.
 * @param {ArtifactEntry} entry
 * @param {string} iconName
 * @param {string | null} shownId
 * @returns {HTMLButtonElement}
 */
function listRow(entry, iconName, shownId) {
  const row = document.createElement('button');
  row.type = 'button';
  row.className = 'osprey-list-row';
  row.setAttribute('role', 'option');
  row.setAttribute('aria-selected', String(entry.id === shownId));
  row.dataset.id = entry.id;
  row.title = titleOf(entry);
  const title = document.createElement('span');
  title.className = 'osprey-list-title';
  title.textContent = titleOf(entry);
  const time = document.createElement('time');
  if (entry.timestamp) {
    time.setAttribute('datetime', entry.timestamp);
    time.textContent = rowTime(entry.timestamp);
  }
  row.append(icon(iconName), title, time);
  return row;
}

/**
 * A row's time on the facility clock: the time of day for today's artifacts,
 * the date for older ones.
 * @param {string} timestamp
 * @returns {string}
 */
function rowTime(timestamp) {
  if (facilityDayKey(timestamp) === facilityDayKey(new Date())) {
    return formatFacilityTime(timestamp, { hour: '2-digit', minute: '2-digit' });
  }
  return formatFacilityTime(timestamp, { month: 'short', day: 'numeric' });
}

/*
 * The lists are listboxes, which own only options: a group label or the empty
 * note takes role="presentation" so it is not an unnamed child of the listbox.
 */

/** @param {string} label */
function listHead(label) {
  const head = document.createElement('div');
  head.className = 'osprey-list-head';
  head.setAttribute('role', 'presentation');
  head.textContent = label;
  return head;
}

/** @param {string} text */
function listEmpty(text) {
  const empty = document.createElement('div');
  empty.className = 'osprey-list-empty';
  empty.setAttribute('role', 'presentation');
  empty.textContent = text;
  return empty;
}

/**
 * The row of an artifact in a list, found by comparing the data attribute
 * rather than through a selector, so no id needs escaping.
 * @param {Element} list
 * @param {string} id
 * @returns {HTMLElement | null}
 */
export function rowIn(list, id) {
  for (const row of list.querySelectorAll('.osprey-list-row')) {
    if (row instanceof HTMLElement && row.dataset.id === id) return row;
  }
  return null;
}

/**
 * The Artifacts ⋯ menu, built once into the skeleton's empty popover: a
 * Group by section with the Types and Activity radios, a separator, then
 * All sessions (a checkbox) and Refresh. The caller keeps the radios' and the
 * checkbox's `aria-checked` in step with the grouping and the scope.
 * @param {HTMLElement} menu
 */
export function fillArtifactMenu(menu) {
  if (menu.querySelector('[data-art-all-sessions]')) return;
  if (!menu.hasAttribute('aria-label')) menu.setAttribute('aria-label', 'Artifact options');
  const title = document.createElement('div');
  title.className = 'osprey-popover-title';
  title.id = 'sidebar-artifacts-menu-group-label';
  title.textContent = 'Group by';
  const group = document.createElement('div');
  group.setAttribute('role', 'group');
  group.setAttribute('aria-labelledby', title.id);
  group.append(
    title,
    viewItem('types', 'types', 'Types'),
    viewItem('activity', 'history', 'Activity')
  );
  const sep = document.createElement('div');
  sep.className = 'osprey-popover-sep';
  sep.setAttribute('role', 'separator');
  const allSessions = menuItem('menuitemcheckbox', 'history', 'All sessions');
  allSessions.dataset.artAllSessions = '';
  allSessions.setAttribute('aria-checked', 'false');
  allSessions.append(checkMark());
  const refresh = menuItem('menuitem', 'refresh', 'Refresh');
  refresh.dataset.artRefresh = '';
  menu.append(group, sep, allSessions, refresh);
}

/**
 * A Group by radio of the ⋯ menu.
 * @param {string} view
 * @param {string} iconName
 * @param {string} label
 */
function viewItem(view, iconName, label) {
  const item = menuItem('menuitemradio', iconName, label);
  item.dataset.artView = view;
  item.setAttribute('aria-checked', 'false');
  item.append(checkMark());
  return item;
}

/** The check a menu's radio or checkbox shows while it is on. */
function checkMark() {
  const check = icon('check');
  check.classList.add('osprey-popover-check');
  return check;
}

/**
 * @param {string} role
 * @param {string} iconName
 * @param {string} label
 * @returns {HTMLButtonElement}
 */
function menuItem(role, iconName, label) {
  const item = document.createElement('button');
  item.type = 'button';
  item.className = 'osprey-popover-item';
  item.setAttribute('role', role);
  item.append(icon(iconName), document.createTextNode(label));
  return item;
}
