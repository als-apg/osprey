// @ts-check
/* OSPREY Web Terminal — the rows of the sidebar's artifact tree
 *
 * Pure DOM builders for the session tree: the Sessions row's disclosure, one row
 * per agent session (its title, or "naming…" while the name is pending, and
 * the server's artifact count), a per-session scroll box, form group heads
 * with the server's counts, run rows (a status dot for succeeded, failed and
 * not run, none while in flight; the output count; the parts underneath) and
 * artifact rows (marked "run" or "interactive"). After the sessions come the
 * "More sessions" row and the notice rows. Activity mode bands a session's
 * rows by the facility day: Today, Yesterday, Older. Indents follow a 4/16/28/40 px ladder
 * (.osprey-tree-d0 … d3, sidebar-artifacts.css).
 *
 * Artifact rows are the shared list primitives of primitives.css
 * (`.osprey-list-row`): a listbox option whose `aria-selected` marks the
 * artifact the viewer shows. Session rows, heads, bands and notices are
 * role="presentation", so the listbox owns only options. The builders keep no
 * state: what is open and what is shown is passed in. All text goes in
 * through textContent.
 *
 * @module sidebar-artifact-rows
 */

import { icon } from '/design-system/js/icons.js';
import { facilityDayKey, formatFacilityTime } from '/design-system/js/facility-time.js';
import { isMacPlatform } from '/design-system/js/platform.js';

/** The session peek's element id (sidebar-session-peek.js draws it). */
export const SESSION_PEEK_ID = 'sidebar-session-peek';

/** @typedef {import('./sidebar-artifact-listing.js').ArtifactEntry} ArtifactEntry */

/** The icon of every run row. */
export const RUN_ICON = 'bash';

/**
 * @param {ArtifactEntry} entry
 * @returns {string}
 */
export function titleOf(entry) {
  return entry.title || entry.filename || entry.id;
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

/**
 * A row the listbox does not own as an option: role="presentation".
 * @param {string} className
 * @param {string} [text]
 * @returns {HTMLDivElement}
 */
function plainRow(className, text) {
  const row = document.createElement('div');
  row.className = className;
  row.setAttribute('role', 'presentation');
  if (text !== undefined) row.textContent = text;
  return row;
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

/* ------------------------------------------------------------------------ */
/* The session tree                                                          */
/* ------------------------------------------------------------------------ */

/**
 * A session row as `GET /api/sessions` lists it (the fields the row draws).
 * @typedef {object} SessionRow
 * @property {string} transcript_id
 * @property {string | null} [title]
 * @property {'user' | 'auto' | 'prompt' | null} [title_source]
 * @property {'pending' | 'done' | 'failed'} [naming]
 * @property {'live' | 'idle'} [state]
 * @property {boolean} [truncated]
 */

/**
 * A part of a run, as a run head's `run_parts` lists it.
 * @typedef {object} RunPart
 * @property {string} id
 * @property {'preview' | 'code' | 'output' | 'result'} role
 * @property {string} [title]
 * @property {string | null} [form]
 */

/**
 * The listing payload fields the tree reads on top of {@link ArtifactEntry}.
 * @typedef {ArtifactEntry & {
 *   form?: string | null,
 *   exec_id?: string,
 *   run_part?: string,
 *   run_status?: 'succeeded' | 'failed' | 'not_run' | null,
 *   run_parts?: RunPart[],
 *   interactive?: boolean,
 * }} TreeEntry
 */

/** @typedef {{key: string, label: string, icon: string}} TreeForm */

/**
 * The base forms in their fixed order (artifact_forms.BASE_FORMS), each with
 * the icon its head and rows draw. Labels come from the registry when it has
 * them; these are the fallback while it has not loaded.
 * @type {readonly TreeForm[]}
 */
const BASE_FORMS = [
  { key: 'runs', label: 'Runs', icon: 'bash' },
  { key: 'plots', label: 'Plots', icon: 'chart-line' },
  { key: 'trends', label: 'Trends', icon: 'activity' },
  { key: 'tables', label: 'Tables', icon: 'table' },
  { key: 'documents', label: 'Documents', icon: 'file-text' },
  { key: 'dashboards', label: 'Dashboards', icon: 'workspace' },
  { key: 'images', label: 'Images', icon: 'crosshair' },
  { key: 'data', label: 'Data', icon: 'database' },
];

/** The icon of a form no group names (an unregistered extension type). */
const FALLBACK_ICON = 'layers';

/** The titles of a run's parts other than its outputs, in the order drawn. */
const PART_TITLES = /** @type {const} */ ({
  preview: 'Approval preview',
  code: 'Code',
  result: 'Result',
});

/** The dot of each settled run status; a run in flight draws none. */
const RUN_DOTS = /** @type {const} */ ({
  succeeded: { cls: 'ok', label: 'Succeeded' },
  failed: { cls: 'err', label: 'Failed' },
  not_run: { cls: 'hollow', label: 'Not run' },
});

/**
 * The tree's groups, in the order they are drawn: the base forms, then the
 * registered extension types (each drawing its base form's icon), from the
 * `/api/type-registry` answer.
 * @param {{base_forms?: {key: string, label: string}[],
 *          extension_types?: {key: string, label: string, base_form: string}[]} | null} registry
 * @returns {TreeForm[]}
 */
export function treeForms(registry) {
  const iconOf = (/** @type {string} */ key) =>
    BASE_FORMS.find((f) => f.key === key)?.icon ?? FALLBACK_ICON;
  const base = registry?.base_forms?.length
    ? registry.base_forms.map((f) => ({ key: f.key, label: f.label, icon: iconOf(f.key) }))
    : BASE_FORMS.map((f) => ({ ...f }));
  const extensions = (registry?.extension_types ?? []).map((t) => ({
    key: t.key,
    label: t.label,
    icon: iconOf(t.base_form),
  }));
  return [...base, ...extensions];
}

/**
 * The Agent row's disclosure: a chevron that says whether the sessions show.
 * @param {boolean} open
 * @returns {HTMLElement}
 */
export function agentDisclosure(open) {
  const disc = document.createElement('span');
  disc.className = 'osprey-tree-agent-disc';
  disc.dataset.treeAgentDisc = '';
  disc.setAttribute('aria-expanded', String(open));
  disc.append(icon(open ? 'chevron-down' : 'chevron-right'));
  return disc;
}

/**
 * The modifier click that expands or collapses every session, as this
 * platform names it.
 * @returns {string}
 */
export function allSessionsClick() {
  return isMacPlatform() ? '⌥-click' : 'Alt+click';
}

/**
 * One session row: a disclosure (on the active row, and on any row the
 * operator opened with expand-all; no other session opens without switching
 * to it), the live dot, the title (or "naming…"), the artifact count from the
 * session summary (`data-count` once it is summarized). The title's tooltip
 * says what a click does and names the expand-all click.
 *
 * Only the active row is bold (`.active`). The accent bar and tint
 * (`.accent`) sit on the active row, or on the row just clicked while the
 * switch to it is under way. `.is-busy` marks the session the agent is
 * working in: its bar breathes. A folded row's title names the peek
 * (sidebar-session-peek.js) in `aria-controls`, so the tooltip layer stays
 * quiet while the peek shows.
 * @param {SessionRow} session
 * @param {{active: boolean, open: boolean, count: number, summarized?: boolean,
 *   paneOpen?: boolean, accent?: boolean, busy?: boolean}} options
 * @returns {HTMLElement}
 */
export function sessionRow(
  session,
  { active, open, count, summarized = false, paneOpen = true, accent = active, busy = false }
) {
  const pending = session.naming === 'pending';
  const title = treeSpan(
    'osprey-tree-title',
    session.title || (pending ? 'naming…' : 'Untitled session')
  );
  if (pending) title.classList.add('pending');
  title.title = title.textContent ?? '';
  title.dataset.tipDetail = `${active ? (paneOpen ? 'Click to fold or unfold' : 'Click to reopen the Agent pane') : 'Click to switch to it'} · ${allSessionsClick()} to expand or collapse all`;
  const dot = document.createElement('span');
  dot.className = `osprey-dot ${session.state === 'live' ? 'ok' : 'hollow'}`;
  const row = topRow('osprey-tree-session', open, [dot, title], active || open);
  row.dataset.sessionId = session.transcript_id;
  if (session.title_source) row.dataset.titleSource = session.title_source;
  if (active) row.classList.add('active');
  if (accent) row.classList.add('accent');
  if (busy) row.classList.add('is-busy');
  if (!open) title.setAttribute('aria-controls', SESSION_PEEK_ID);
  if (summarized) row.dataset.count = String(count);
  if (pending && session.title) row.append(treeSpan('osprey-tree-naming', 'naming…'));
  row.append(treeSpan('osprey-tree-n', String(count)));
  return row;
}

/**
 * The "More sessions" row, ending a tree that holds one page of sessions.
 * @param {number} count  sessions with artifacts not loaded yet
 * @returns {HTMLElement}
 */
export function moreSessionsRow(count) {
  const row = plainRow('osprey-tree-more osprey-tree-d0');
  row.dataset.treeMore = '';
  row.append(
    treeSpan('osprey-tree-label', 'More sessions'),
    treeSpan('osprey-tree-n', String(count))
  );
  return row;
}

/** The text of each notice row. */
const NOTICES = /** @type {const} */ ({
  incomplete: 'Session history may be incomplete',
  unavailable: 'Session history unavailable',
});

/**
 * A notice row: the session history was cut short, or could not be read.
 * @param {'incomplete' | 'unavailable'} kind
 * @returns {HTMLElement}
 */
export function noticeRow(kind) {
  const notice = plainRow('osprey-tree-notice', NOTICES[kind]);
  notice.dataset.treeNotice = kind;
  return notice;
}

/**
 * A session's rows inside its own scroll box, or the line an empty session
 * shows.
 * @param {string} sessionId  the tree id: a transcript id, or `key:<K>`
 * @param {Node[]} children
 * @returns {HTMLElement}
 */
export function sessionBox(sessionId, children) {
  const box = document.createElement('div');
  box.className = 'osprey-tree-box';
  box.setAttribute('role', 'group');
  box.dataset.sessionBox = sessionId;
  box.append(...(children.length ? children : [plainRow('osprey-tree-empty', 'No artifacts yet')]));
  return box;
}

/**
 * A form group's head: disclosure, icon, label and the server's count.
 * @param {TreeForm} form
 * @param {{sessionId: string, count: number, open: boolean}} options
 * @returns {HTMLElement}
 */
export function groupHead(form, { sessionId, count, open }) {
  const node = plainRow('osprey-tree-head osprey-tree-d1');
  node.setAttribute('aria-expanded', String(open));
  node.dataset.groupHead = `${sessionId}:${form.key}`;
  node.append(
    icon(open ? 'chevron-down' : 'chevron-right'),
    treeSpan('osprey-tree-label', form.label),
    treeSpan('osprey-tree-n', String(count))
  );
  return node;
}

/**
 * @typedef {object} TreeRowState
 * @property {string | null} shownId  the artifact the viewer shows, null while it is off view
 */

/**
 * An artifact row, marked "interactive" (a plotly figure) and "run" (an
 * output of a script run, listed in its form group too).
 * @param {TreeEntry} entry
 * @param {TreeRowState & {iconName?: string, depth?: number}} options
 * @returns {HTMLButtonElement}
 */
export function artifactRow(entry, { shownId, iconName, depth = 2 }) {
  const row = treeRow(entry, iconName ?? iconOfForm(entry.form), shownId, depth);
  const markers = [];
  if (entry.interactive) markers.push('interactive');
  if (entry.run_part === 'output' && entry.exec_id) markers.push('run');
  for (const text of markers) row.querySelector('time')?.before(treeSpan('osprey-tree-meta', text));
  return row;
}

/**
 * A run's rows: its head (a status dot unless in flight, the output count or
 * "not run", a disclosure when it has parts) and, open, its outputs then its
 * Approval preview and Code.
 * @param {TreeEntry} head
 * @param {TreeRowState & {open: boolean}} options
 * @returns {HTMLElement[]}
 */
export function runTreeRows(head, { shownId, open }) {
  const row = treeRow(head, RUN_ICON, shownId, 2);
  row.classList.add('osprey-tree-run');
  row.dataset.run = head.exec_id || head.id;
  const status = head.run_status ?? null;
  row.dataset.runStatus = status ?? 'running';
  const title = row.querySelector('.osprey-list-title');
  if (status && RUN_DOTS[status]) {
    const dot = document.createElement('span');
    dot.className = `osprey-dot ${RUN_DOTS[status].cls}`;
    dot.title = RUN_DOTS[status].label;
    title?.before(dot);
  }
  const parts = (head.run_parts ?? []).filter((p) => p.id !== head.id);
  const outputs = parts.filter((p) => p.role === 'output');
  const meta = status === 'not_run' ? 'not run' : outputs.length ? String(outputs.length) : '';
  if (meta) row.querySelector('time')?.before(treeSpan('osprey-tree-meta', meta));
  if (!parts.length) return [row];
  row.setAttribute('aria-expanded', String(open));
  const disc = icon(open ? 'chevron-down' : 'chevron-right');
  disc.classList.add('osprey-tree-disc');
  disc.setAttribute('data-tree-disc', '');
  row.prepend(disc);
  if (!open) return [row];
  const others = Object.keys(PART_TITLES).flatMap((role) => parts.filter((p) => p.role === role));
  /** @type {HTMLElement[]} */
  const kids = outputs.map((part) =>
    treeRow({ id: part.id, title: part.title ?? part.id }, iconOfForm(part.form), shownId, 3)
  );
  for (const part of others) {
    const kid = treeRow(
      { id: part.id, title: PART_TITLES[/** @type {keyof typeof PART_TITLES} */ (part.role)] },
      'notebook',
      shownId,
      3
    );
    kid.classList.add('osprey-tree-part');
    kids.push(kid);
  }
  return [row, ...kids];
}

/**
 * A session's rows in Kind mode: one head per form with rows (the server's
 * count when it has one), in form order; the open groups list their rows,
 * Runs as run rows. Run parts (`form` null) are never top-level rows.
 * @param {TreeEntry[]} entries  newest first, as session-list returns them
 * @param {TreeRowState & {
 *   forms: TreeForm[],
 *   sessionId: string,
 *   counts: Record<string, number> | null,
 *   openGroups: Set<string>,
 *   openRuns: Set<string>,
 * }} options
 * @returns {HTMLElement[]}
 */
export function kindRows(entries, { forms, sessionId, counts, openGroups, openRuns, shownId }) {
  /** @type {Map<string, TreeEntry[]>} */
  const byForm = new Map();
  for (const entry of entries) {
    if (!entry.form) continue;
    const list = byForm.get(entry.form) ?? [];
    list.push(entry);
    byForm.set(entry.form, list);
  }
  const known = new Set(forms.map((f) => f.key));
  const drawn = [
    ...forms,
    ...[...byForm.keys()]
      .filter((key) => !known.has(key))
      .map((key) => ({ key, label: key, icon: FALLBACK_ICON })),
  ];
  /** @type {HTMLElement[]} */
  const nodes = [];
  for (const form of drawn) {
    const mine = byForm.get(form.key) ?? [];
    const count = counts ? (counts[form.key] ?? 0) : mine.length;
    if (!count) continue;
    const open = openGroups.has(form.key);
    nodes.push(groupHead(form, { sessionId, count, open }));
    if (!open) continue;
    for (const entry of mine) {
      if (form.key === 'runs') {
        nodes.push(...runTreeRows(entry, { shownId, open: openRuns.has(runKey(entry)) }));
      } else {
        nodes.push(artifactRow(entry, { shownId, iconName: form.icon }));
      }
    }
  }
  return nodes;
}

/**
 * A session's rows in Activity mode: banded Today, Yesterday, Older by the
 * facility day, newest first, a run per row (its outputs and parts listed
 * only under it).
 * @param {TreeEntry[]} entries  newest first
 * @param {TreeRowState & {openRuns: Set<string>}} options
 * @returns {HTMLElement[]}
 */
export function activityRows(entries, { shownId, openRuns }) {
  const today = facilityDayKey(new Date());
  const yesterday = dayBefore(today);
  /** @type {HTMLElement[]} */
  const nodes = [];
  const labelOf = (/** @type {TreeEntry} */ e) => {
    const day = e.timestamp ? facilityDayKey(e.timestamp) : '';
    return day === today ? 'Today' : day === yesterday ? 'Yesterday' : 'Older';
  };
  // A heading only separates bands: none while every row falls in one.
  const banded = new Set(entries.filter((e) => e.form).map(labelOf)).size > 1;
  let band = '';
  for (const entry of entries) {
    if (!entry.form) continue;
    const isHead = entry.form === 'runs';
    if (!isHead && entry.exec_id && entry.run_part === 'output') continue;
    const label = labelOf(entry);
    if (banded && label !== band) {
      band = label;
      nodes.push(plainRow('osprey-tree-day osprey-tree-d1', label));
    }
    if (isHead) {
      nodes.push(...runTreeRows(entry, { shownId, open: openRuns.has(runKey(entry)) }));
    } else {
      nodes.push(artifactRow(entry, { shownId }));
    }
  }
  return nodes;
}

/**
 * The key a run's open state is kept under: its `exec_id`, else its id.
 * @param {TreeEntry} head
 */
export function runKey(head) {
  return head.exec_id || head.id;
}

/**
 * @param {string | null | undefined} form
 * @returns {string}
 */
function iconOfForm(form) {
  return BASE_FORMS.find((f) => f.key === form)?.icon ?? FALLBACK_ICON;
}

/**
 * A tree row: a list row at an indent, selected when shown.
 * @param {ArtifactEntry} entry
 * @param {string} iconName
 * @param {string | null} shownId
 * @param {number} depth
 * @returns {HTMLButtonElement}
 */
function treeRow(entry, iconName, shownId, depth) {
  const row = listRow(entry, iconName, shownId);
  row.classList.add('osprey-tree-row', `osprey-tree-d${depth}`);
  return row;
}

/**
 * A session-level row: disclosure first, then `kids`.
 * Without `foldable` an empty slot of the disclosure's width keeps the titles
 * aligned.
 * @param {string} className
 * @param {boolean} open
 * @param {Node[]} kids
 * @param {boolean} [foldable]
 * @returns {HTMLElement}
 */
function topRow(className, open, kids, foldable = true) {
  const row = plainRow(`${className} osprey-tree-d0`);
  row.setAttribute('aria-expanded', String(open));
  /** @type {Element} */
  let disc;
  if (foldable) {
    disc = icon(open ? 'chevron-down' : 'chevron-right');
    disc.setAttribute('data-tree-disc', '');
  } else {
    disc = document.createElement('span');
    disc.setAttribute('aria-hidden', 'true');
  }
  disc.classList.add('osprey-tree-disc');
  row.append(disc, ...kids);
  return row;
}

/**
 * @param {string} className
 * @param {string} text
 * @returns {HTMLSpanElement}
 */
function treeSpan(className, text) {
  const span = document.createElement('span');
  span.className = className;
  span.textContent = text;
  return span;
}

/**
 * @param {string} value  `sid:form`; the session id may itself hold colons
 * @returns {[string, string]}
 */
export function splitGroup(value) {
  const at = value.lastIndexOf(':');
  return [value.slice(0, at), value.slice(at + 1)];
}

/** Toggle a key's membership. @param {Set<string>} set @param {string} key */
export function flip(set, key) {
  if (!set.delete(key)) set.add(key);
}
