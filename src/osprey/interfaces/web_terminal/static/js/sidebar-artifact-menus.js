// @ts-check
/* OSPREY Web Terminal — Session-tree context menus
 *
 * The right-click (and menu key / Shift+F10) menus of the sidebar's session
 * tree, one item list per row kind, opened through the hub's shared context
 * menu (panel-context-menu.js):
 *
 *   Agent row   New session · Expand all / Collapse all
 *   Session     Open in Agent · Browse · Group by Kind / Activity · Rename… ·
 *               Use automatic name · Export transcript · Copy ID
 *   Group head  Expand / Collapse
 *   Run         Open result · Open approval preview · Open code · Delete…
 *   Artifact    Open · Pin · Delete… · Copy ID
 *
 * Where a verb is a gesture the tree already answers (open a session, browse
 * it, open a group or a run, show an artifact) the item clicks that row, so a
 * menu pick runs the controller's own closure (sidebar-artifacts.js) and is
 * indistinguishable downstream from the click. The Agent row is a rail entry:
 * panel-menu-policy asks {@link buildAgentRowMenuItems} for its rows.
 *
 * session-model.js (and through it chat.js) is imported lazily, at a pick:
 * panel-menu-policy imports this module, and a static edge to the model would
 * close an import cycle through the chat back to panel-manager.
 */

/** @returns {Promise<typeof import('./session-model.js')>} */
const sessionModel = () => import('./session-model.js');

import { openContextMenu } from './panel-context-menu.js';
import { ARTIFACTS_BASE, fetchArtifact, readTreeView } from './sidebar-artifact-listing.js';
import { showConfirm } from './posture-confirm.js';

/** @typedef {import('./panel-context-menu.js').MenuItem} MenuItem */
/** @typedef {import('./sidebar-artifact-listing.js').ArtifactEntry} ArtifactEntry */
/** @typedef {import('./sidebar-artifact-rows.js').TreeEntry} TreeEntry */

const TREE_SELECTOR = '[data-session-tree]';
const SESSION_SELECTOR = '.osprey-tree-session[data-session-id]';
const GROUP_SELECTOR = '[data-group-head]';
const RUN_SELECTOR = '.osprey-list-row.osprey-tree-run[data-id]';
const ARTIFACT_SELECTOR = '.osprey-list-row[data-id]';
/** Top-level rows Expand/Collapse all acts on: sessions and Earlier. */
const TOP_SELECTOR = `${SESSION_SELECTOR}, [data-tree-earlier]`;

// ---- Row lookups ----

/**
 * @param {Element | null} scope  @param {string} selector  @param {string} key
 * @param {string} value
 * @returns {HTMLElement | null}
 */
function rowBy(scope, selector, key, value) {
  for (const row of (scope ?? document).querySelectorAll(selector)) {
    if (row instanceof HTMLElement && row.dataset[key] === value) return row;
  }
  return null;
}

/** Click a row's disclosure (or the row itself when it has none). @param {HTMLElement} row */
function clickDisc(row) {
  const disc = row.querySelector('[data-tree-disc]');
  (disc instanceof HTMLElement ? disc : row).click();
}

/** @param {HTMLElement} row */
const isOpen = (row) => row.getAttribute('aria-expanded') === 'true';

// ---- Agent row ----

/** @returns {boolean} whether any top-level tree row is open */
function anyOpen() {
  return [...document.querySelectorAll(TOP_SELECTOR)].some(
    (row) => row instanceof HTMLElement && isOpen(row)
  );
}

/**
 * Open or close every top-level row. Each click re-renders the tree, so the
 * rows are looked up afresh each round.
 * @param {boolean} open
 */
export function setAllOpen(open) {
  const limit = document.querySelectorAll(TOP_SELECTOR).length + 1;
  for (let i = 0; i < limit; i += 1) {
    const row = [...document.querySelectorAll(TOP_SELECTOR)].find(
      (r) => r instanceof HTMLElement && isOpen(r) !== open
    );
    if (!(row instanceof HTMLElement)) return;
    clickDisc(row);
  }
}

/**
 * The Agent row's menu: a new session, and one Expand all / Collapse all row
 * worded for what it will do.
 * @returns {MenuItem[]}
 */
export function buildAgentRowMenuItems() {
  const open = anyOpen();
  return [
    { label: 'New session', run: () => void sessionModel().then((m) => m.switchTo({ new: true })) },
    { divider: true },
    open
      ? { label: 'Collapse all', run: () => setAllOpen(false) }
      : { label: 'Expand all', run: () => setAllOpen(true) },
  ];
}

// ---- Session ----

/**
 * Put the operator's title; an empty one reverts to the automatic name.
 * @param {string} transcriptId  @param {string} title
 */
export async function putTitle(transcriptId, title) {
  try {
    await fetch(`/api/sessions/${encodeURIComponent(transcriptId)}/title`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
  } finally {
    (await sessionModel()).refresh();
  }
}

/**
 * Edit a session row's title in place: Enter or blur saves, Escape cancels,
 * an empty name means the automatic one.
 * @param {HTMLElement} row  @param {string} transcriptId
 */
export function startRename(row, transcriptId) {
  const title = row.querySelector('.osprey-tree-title');
  if (!(title instanceof HTMLElement)) return;
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'osprey-tree-rename';
  input.value = title.classList.contains('pending') ? '' : (title.textContent ?? '');
  input.setAttribute('aria-label', 'Session name');
  let settled = false;
  /** @param {boolean} save */
  const finish = (save) => {
    if (settled) return;
    settled = true;
    input.replaceWith(title);
    if (save) void putTitle(transcriptId, input.value.trim());
  };
  input.addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (e.key === 'Enter') finish(true);
    else if (e.key === 'Escape') finish(false);
  });
  input.addEventListener('click', (e) => e.stopPropagation());
  input.addEventListener('blur', () => finish(true));
  title.replaceWith(input);
  input.focus();
  input.select();
}

/** Download a session's transcript (GET /api/sessions/{id}/transcript.md). @param {string} id */
export function exportTranscript(id) {
  const a = document.createElement('a');
  a.href = `/api/sessions/${encodeURIComponent(id)}/transcript.md`;
  a.download = `transcript-${id}.md`;
  document.body.append(a);
  a.click();
  a.remove();
}

/** @param {string} text */
function copyText(text) {
  void navigator.clipboard?.writeText(text).catch(() => {});
}

/**
 * A session row's menu.
 * @param {HTMLElement} row
 * @returns {MenuItem[]}
 */
export function sessionMenuItems(row) {
  const treeId = row.dataset.sessionId ?? '';
  // A synthesized row (`key:<K>`) has no transcript yet: nothing to name,
  // export or copy.
  const tid = treeId && !treeId.startsWith('key:') ? treeId : null;
  const view = readTreeView();
  return [
    { label: 'Open in Agent', run: () => row.click() },
    { label: 'Browse', run: () => (isOpen(row) ? undefined : clickDisc(row)) },
    { divider: true },
    { label: 'Group by Kind', checked: view === 'kind', run: () => void applyView('kind') },
    {
      label: 'Group by Activity',
      checked: view === 'activity',
      run: () => void applyView('activity'),
    },
    { divider: true },
    { label: 'Rename…', disabled: !tid, run: () => tid && startRename(row, tid) },
    { label: 'Use automatic name', disabled: !tid, run: () => tid && void putTitle(tid, '') },
    { label: 'Export transcript', disabled: !tid, run: () => tid && exportTranscript(tid) },
    { label: 'Copy ID', disabled: !tid, run: () => tid && copyText(tid) },
  ];
}

/**
 * Apply a tree mode through the controller. Imported lazily: the controller
 * imports panel-menu-policy, which imports this module.
 * @param {'kind' | 'activity'} view
 */
async function applyView(view) {
  const { setTreeView } = await import('./sidebar-artifacts.js');
  setTreeView(view);
}

// ---- Group head ----

/** @param {HTMLElement} head @returns {MenuItem[]} */
export function groupMenuItems(head) {
  return [{ label: isOpen(head) ? 'Collapse' : 'Expand', run: () => head.click() }];
}

// ---- Artifact and run ----

/** @param {string} id */
const artifactUrl = (id) => `${ARTIFACTS_BASE}/api/artifacts/${encodeURIComponent(id)}`;

/** @param {ArtifactEntry} entry */
async function setPinned(entry) {
  await fetch(`${artifactUrl(entry.id)}/pin`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pinned: !entry.pinned }),
  });
}

/**
 * Ask first, then delete. Deleting a run's head deletes the run and its
 * parts (the server's delete_with_group), and the question says so.
 * @param {TreeEntry} entry  @param {boolean} isRun
 */
export function confirmDelete(entry, isRun) {
  const name = entry.title || entry.id;
  const parts = (entry.run_parts ?? []).filter((p) => p.id !== entry.id).length;
  const body = isRun
    ? [['This deletes the run ', { em: name }, ` and its parts (${parts}).`]]
    : [['This deletes ', { em: name }, '.']];
  showConfirm({
    title: isRun ? 'Delete run?' : 'Delete artifact?',
    body,
    live: null,
    confirmLabel: 'Delete',
    focus: 'cancel',
    onConfirm: async (ui) => {
      try {
        const res = await fetch(artifactUrl(entry.id), { method: 'DELETE' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        ui.done();
      } catch (err) {
        ui.error.textContent = `Delete failed: ${err instanceof Error ? err.message : err}`;
        ui.error.hidden = false;
      }
    },
  });
}

/**
 * The menu of an artifact row (a run's part included).
 * @param {HTMLElement} row  @param {ArtifactEntry} entry
 * @returns {MenuItem[]}
 */
export function artifactMenuItems(row, entry) {
  return [
    { label: 'Open', run: () => row.click() },
    { label: entry.pinned ? 'Unpin' : 'Pin', run: () => void setPinned(entry) },
    { label: 'Delete…', danger: true, run: () => confirmDelete(entry, false) },
    { divider: true },
    { label: 'Copy ID', run: () => copyText(entry.id) },
  ];
}

/**
 * Show one part of a run: open the run so its parts are drawn, then click the
 * part's row (the controller's own pick).
 * @param {HTMLElement} head  @param {string} partId
 */
function openPart(head, partId) {
  const runKey = head.dataset.run ?? '';
  const tree = head.closest(TREE_SELECTOR);
  if (!isOpen(head)) clickDisc(head);
  const current = rowBy(tree, RUN_SELECTOR, 'run', runKey) ?? head;
  const box = current.parentElement ?? tree;
  rowBy(box, ARTIFACT_SELECTOR, 'id', partId)?.click();
}

/**
 * A run head's menu. A part the run does not have is offered disabled.
 * @param {HTMLElement} head  @param {TreeEntry} entry
 * @returns {MenuItem[]}
 */
export function runMenuItems(head, entry) {
  const parts = (entry.run_parts ?? []).filter((p) => p.id !== entry.id);
  /** @param {string} role */
  const part = (role) => parts.find((p) => p.role === role) ?? null;
  const result = part('result') ?? part('output');
  const preview = part('preview');
  const code = part('code');
  /** @param {string} label  @param {{id: string} | null} p @returns {MenuItem} */
  const opener = (label, p) => ({
    label,
    disabled: !p,
    run: () => p && openPart(head, p.id),
  });
  return [
    opener('Open result', result),
    opener('Open approval preview', preview),
    opener('Open code', code),
    { divider: true },
    { label: 'Delete…', danger: true, run: () => confirmDelete(entry, true) },
  ];
}

// ---- Opening ----

/**
 * The menu a tree row offers, or null when it offers none (Earlier, "more").
 * Artifact and run rows read their entry first (pin state, run parts).
 * @param {Element} target
 * @returns {Promise<{row: HTMLElement, label: string, items: MenuItem[]} | null>}
 */
export async function menuFor(target) {
  const head = target.closest(GROUP_SELECTOR);
  if (head instanceof HTMLElement) {
    return { row: head, label: 'Group actions', items: groupMenuItems(head) };
  }
  const session = target.closest(SESSION_SELECTOR);
  if (session instanceof HTMLElement) {
    return { row: session, label: 'Session actions', items: sessionMenuItems(session) };
  }
  const row = target.closest(ARTIFACT_SELECTOR);
  if (!(row instanceof HTMLElement)) return null;
  const entry = await fetchArtifact(row.dataset.id ?? '');
  if (!entry) return null;
  if (row.matches(RUN_SELECTOR)) {
    return { row, label: 'Run actions', items: runMenuItems(row, entry) };
  }
  return { row, label: 'Artifact actions', items: artifactMenuItems(row, entry) };
}

/**
 * Open a tree row's menu at a viewport position.
 * @param {Element} target  @param {number} x  @param {number} y
 * @returns {Promise<boolean>} whether a menu opened
 */
export async function openTreeMenu(target, x, y) {
  const menu = await menuFor(target);
  if (!menu) return false;
  openContextMenu({ x, y, anchorEl: menu.row, ariaLabel: menu.label, items: menu.items });
  return true;
}

/** @param {EventTarget | null} t @returns {Element | null} */
function inTree(t) {
  const el = t instanceof Element ? t : null;
  if (!el?.closest(TREE_SELECTOR)) return null;
  return el.closest(`${GROUP_SELECTOR}, ${SESSION_SELECTOR}, ${ARTIFACT_SELECTOR}`) ? el : null;
}

/** @param {MouseEvent} e */
function onContextMenu(e) {
  const el = inTree(e.target);
  if (!el) return;
  e.preventDefault();
  void openTreeMenu(el, e.clientX, e.clientY);
}

/** The menu key and Shift+F10 open the focused row's menu at its centre. @param {KeyboardEvent} e */
function onKeyDown(e) {
  if (e.key !== 'ContextMenu' && !(e.key === 'F10' && e.shiftKey)) return;
  const el = inTree(e.target);
  if (!el) return;
  e.preventDefault();
  const rect = el.getBoundingClientRect();
  void openTreeMenu(el, rect.left + rect.width / 2, rect.top + rect.height / 2);
}

/** @type {AbortController | null} */
let listeners = null;

/** Listen for tree-row menu gestures on the document. Idempotent. */
export function initTreeMenus() {
  if (listeners) return;
  listeners = new AbortController();
  const { signal } = listeners;
  document.addEventListener('contextmenu', onContextMenu, { signal });
  document.addEventListener('keydown', onKeyDown, { signal });
}

/** Stop listening (tests). */
export function teardownTreeMenus() {
  listeners?.abort();
  listeners = null;
}
