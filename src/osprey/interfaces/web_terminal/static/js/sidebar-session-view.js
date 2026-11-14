// @ts-check
/* OSPREY Web Terminal — the session list's view menu
 *
 * The view button beside the Sessions row's "+" opens a menu of how the
 * session list is shown (the hub's shared context menu, a value at each
 * row's end):
 *
 *   Last activity  ▸ Today · 3 days · 7 days · 30 days · All
 *   Group by       ▸ Date · State · None
 *   Artifacts by   ▸ Activity · Kind        (sidebar-artifacts.js's tree mode)
 *   With artifacts only ·  Artifact counts
 *   Reset to defaults
 *
 * The choices are the operator's, kept in this browser under a key scoped per
 * user; the tree mode keeps its own key (sidebar-artifact-listing.js). The
 * button shows a dot while any choice differs from its default.
 *
 * {@link arrangeSessions} applies them to the loaded rows, which come newest
 * activity first, a page at a time (session-model.js): a filter hides rows
 * and, past its cutoff, the "More sessions" row; Date groups follow that
 * order (Today, Yesterday, Last 7 days, Last 30 days, Older), State puts the
 * sessions whose agent runs now first. The session in use is always shown,
 * and a row that cannot be judged (no time yet, no artifact count yet) stays.
 * The pinned group's list is not filtered.
 */

import { facilityDayStart } from '/design-system/js/facility-time.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { readTreeView } from './sidebar-artifact-listing.js';
import { openContextMenu } from './panel-context-menu.js';
import { labelRow } from './sidebar-artifact-rows.js';

/** @typedef {import('./panel-context-menu.js').MenuItem} MenuItem */
/** @typedef {import('./session-model.js').SessionRow & {last_ts?: string|null, first_ts?: string|null}} ViewRow */

/** Base key of the persisted view, scoped per user. */
export const SESSION_VIEW_STORAGE_KEY = 'osprey-sidebar-session-view';

/** The view button's id (index.html). */
export const VIEW_BUTTON_ID = 'sidebar-session-view';

/** @typedef {'1d' | '3d' | '7d' | '30d' | 'all'} Activity */
/** @typedef {'date' | 'state' | 'none'} Grouping */
/** @typedef {{activity: Activity, group: Grouping, withArtifacts: boolean, counts: boolean}} SessionView */

/** @type {Readonly<SessionView>} */
export const DEFAULT_VIEW = Object.freeze({
  activity: 'all',
  group: 'none',
  withArtifacts: false,
  counts: true,
});

/** @type {[Activity, string][]} */
const ACTIVITY = [
  ['1d', 'Today'],
  ['3d', '3 days'],
  ['7d', '7 days'],
  ['30d', '30 days'],
  ['all', 'All'],
];

/** @type {[Grouping, string][]} */
const GROUPING = [
  ['date', 'Date'],
  ['state', 'State'],
  ['none', 'None'],
];

const DAY_MS = 24 * 60 * 60 * 1000;

/** @type {SessionView} */
let view = { ...DEFAULT_VIEW };
/** @type {() => void} */
let changed = () => {};

/**
 * @param {Record<string, unknown>} raw
 * @returns {SessionView} the stored view, each unknown value at its default
 */
function cleanView(raw) {
  const activity = ACTIVITY.find(([v]) => v === raw.activity)?.[0] ?? DEFAULT_VIEW.activity;
  const group = GROUPING.find(([v]) => v === raw.group)?.[0] ?? DEFAULT_VIEW.group;
  return {
    activity,
    group,
    withArtifacts: raw.withArtifacts === true,
    counts: raw.counts !== false,
  };
}

/** @returns {SessionView} the persisted view, the defaults when there is none */
export function readSessionView() {
  try {
    const raw = JSON.parse(
      localStorage.getItem(scopedStorageKey(SESSION_VIEW_STORAGE_KEY)) ?? '{}'
    );
    return cleanView(raw && typeof raw === 'object' ? raw : {});
  } catch {
    return { ...DEFAULT_VIEW };
  }
}

/** @returns {SessionView} the view in force */
export function sessionView() {
  return { ...view };
}

/**
 * Persist and apply a change to the view.
 * @param {Partial<SessionView>} change
 */
export function setSessionView(change) {
  view = cleanView({ ...view, ...change });
  try {
    localStorage.setItem(scopedStorageKey(SESSION_VIEW_STORAGE_KEY), JSON.stringify(view));
  } catch {
    /* storage blocked: the view holds for this page */
  }
  paintButton();
  changed();
}

/** The tree mode through the controller, imported lazily (it imports this module). */
async function applyTreeView(/** @type {'kind' | 'activity'} */ mode) {
  (await import('./sidebar-artifacts.js')).setTreeView(mode);
  paintButton();
}

/**
 * The "Artifacts by" row, how a session's artifacts are grouped; the Sessions
 * row's right-click menu shows it too.
 * @returns {MenuItem}
 */
export function artifactsByItem() {
  const mode = readTreeView();
  return {
    label: 'Artifacts by',
    value: mode === 'kind' ? 'Kind' : 'Activity',
    submenu: [
      {
        label: 'Activity',
        checked: mode === 'activity',
        run: () => void applyTreeView('activity'),
      },
      { label: 'Kind', checked: mode === 'kind', run: () => void applyTreeView('kind') },
    ],
  };
}

/**
 * @template {string} V
 * @param {string} label  @param {[V, string][]} choices  @param {V} current
 * @param {(value: V) => void} pick
 * @returns {MenuItem}
 */
function choiceItem(label, choices, current, pick) {
  return {
    label,
    value: choices.find(([v]) => v === current)?.[1],
    submenu: choices.map(([value, name]) => ({
      label: name,
      checked: value === current,
      run: () => pick(value),
    })),
  };
}

/** @returns {MenuItem[]} the view menu's rows */
export function viewMenuItems() {
  return [
    choiceItem('Last activity', ACTIVITY, view.activity, (activity) =>
      setSessionView({ activity })
    ),
    choiceItem('Group by', GROUPING, view.group, (group) => setSessionView({ group })),
    artifactsByItem(),
    { divider: true },
    {
      label: 'With artifacts only',
      checked: view.withArtifacts,
      run: () => setSessionView({ withArtifacts: !view.withArtifacts }),
    },
    {
      label: 'Artifact counts',
      checked: view.counts,
      run: () => setSessionView({ counts: !view.counts }),
    },
    { divider: true },
    {
      label: 'Reset to defaults',
      disabled: isDefault(),
      run: () => {
        setSessionView({ ...DEFAULT_VIEW });
        if (readTreeView() !== 'activity') void applyTreeView('activity');
      },
    },
  ];
}

/** @returns {boolean} whether every choice, the tree mode too, is at its default */
function isDefault() {
  const keys = /** @type {(keyof SessionView)[]} */ (Object.keys(DEFAULT_VIEW));
  return keys.every((k) => view[k] === DEFAULT_VIEW[k]) && readTreeView() === 'activity';
}

/** The button's dot, and the counts' visibility (a page attribute the CSS reads). */
function paintButton() {
  const button = document.getElementById(VIEW_BUTTON_ID);
  if (button) button.toggleAttribute('data-changed', !isDefault());
  document.documentElement.toggleAttribute('data-session-counts-off', !view.counts);
}

/**
 * Wire the view button: load the stored view and redraw through `onChange`.
 * @param {AbortSignal} signal  @param {() => void} onChange
 */
export function watchViewButton(signal, onChange) {
  view = readSessionView();
  changed = onChange;
  paintButton();
  const button = document.getElementById(VIEW_BUTTON_ID);
  button?.addEventListener(
    'click',
    () => {
      const box = button.getBoundingClientRect();
      openContextMenu({
        x: box.right,
        y: box.bottom,
        anchorEl: button,
        triggerEl: button,
        side: 'right',
        ariaLabel: 'Session list view',
        items: viewMenuItems(),
      });
    },
    { signal }
  );
  signal.addEventListener('abort', () => {
    view = { ...DEFAULT_VIEW };
    changed = () => {};
  });
}

/** @param {ViewRow} row @returns {number | null} its last activity, ms, else its start */
function timeOf(row) {
  const at = Date.parse(row.last_ts ?? row.first_ts ?? '');
  return Number.isNaN(at) ? null : at;
}

/**
 * The earliest last activity the filter keeps: the start of today on the
 * facility clock, or N days back from now; null for All.
 * @param {Activity} activity  @param {Date} now
 */
function cutoffOf(activity, now) {
  if (activity === 'all') return null;
  if (activity === '1d') return facilityDayStart(now);
  return now.getTime() - Number.parseInt(activity, 10) * DAY_MS;
}

/**
 * The Date group of a time: Today and Yesterday by calendar day on the
 * facility clock, then by age.
 * @param {number | null} at  @param {Date} now
 */
function dateGroup(at, now) {
  if (at === null) return 'Today';
  const today = facilityDayStart(now);
  if (at >= today) return 'Today';
  if (at >= facilityDayStart(today - 1)) return 'Yesterday';
  if (at >= now.getTime() - 7 * DAY_MS) return 'Last 7 days';
  if (at >= now.getTime() - 30 * DAY_MS) return 'Last 30 days';
  return 'Older';
}

/**
 * The loaded rows as the view shows them.
 * @param {ViewRow[]} rows  newest activity first
 * @param {{idOf: (row: ViewRow) => string, activeId: string | null,
 *   countOf: (id: string) => number | undefined, now?: Date}} options
 * @returns {{items: ({label: string} | {row: ViewRow})[], more: boolean, empty: string | null}}
 *   `more`: whether rows past the loaded pages can still match
 */
export function arrangeSessions(rows, { idOf, activeId, countOf, now = new Date() }) {
  const cutoff = cutoffOf(view.activity, now);
  const kept = rows.filter((row) => {
    if (idOf(row) === activeId) return true;
    const at = timeOf(row);
    if (cutoff !== null && at !== null && at < cutoff) return false;
    const count = countOf(idOf(row));
    return !view.withArtifacts || count === undefined || count > 0;
  });
  const lastAt =
    rows
      .map(timeOf)
      .filter((at) => at !== null)
      .at(-1) ?? null;
  const more = cutoff === null || lastAt === null || lastAt >= cutoff;
  /** @type {({label: string} | {row: ViewRow})[]} */
  const items = [];
  if (view.group === 'date') {
    let current = '';
    for (const row of kept) {
      const at = row.synthesized ? null : timeOf(row);
      // A listed row with no time yet stays in the group of the row above it.
      const label = at === null && current && !row.synthesized ? current : dateGroup(at, now);
      if (label !== current) items.push({ label: (current = label) });
      items.push({ row });
    }
  } else if (view.group === 'state') {
    const live = kept.filter((row) => row.synthesized || row.state === 'live');
    const rest = kept.filter((row) => !live.includes(row));
    if (live.length) items.push({ label: 'Open now' }, ...live.map((row) => ({ row })));
    if (rest.length) items.push({ label: 'Earlier' }, ...rest.map((row) => ({ row })));
  } else {
    items.push(...kept.map((row) => ({ row })));
  }
  return { items, more, empty: rows.length && !kept.length ? emptyText() : null };
}

/**
 * The session list's nodes as the view shows them, and what the "More
 * sessions" row needs.
 * @param {ViewRow[]} rows  @param {(row: ViewRow) => string} idOf
 * @param {string | null} activeId  @param {(id: string) => number | undefined} countOf
 * @param {(row: ViewRow) => Node[]} nodesOf  a session's row and its box
 * @returns {{nodes: Node[], more: boolean, loadedTotal: number}}
 */
export function sessionListNodes(rows, idOf, activeId, countOf, nodesOf) {
  const { items, more, empty } = arrangeSessions(rows, { idOf, activeId, countOf });
  const nodes = items.flatMap((item) =>
    'label' in item ? [labelRow(item.label)] : nodesOf(item.row)
  );
  if (empty) nodes.push(labelRow(empty));
  const loadedTotal = rows.reduce(
    (n, row) => n + ((row.transcript_id && countOf(idOf(row))) || 0),
    0
  );
  return { nodes, more, loadedTotal };
}

/** @returns {string} the notice for a view that hides every loaded row */
function emptyText() {
  const span = ACTIVITY.find(([v]) => v === view.activity)?.[1] ?? '';
  const when = view.activity === 'all' ? '' : view.activity === '1d' ? ' today' : ` in ${span}`;
  return `No sessions${view.withArtifacts ? ' with artifacts' : ''}${when}`;
}
