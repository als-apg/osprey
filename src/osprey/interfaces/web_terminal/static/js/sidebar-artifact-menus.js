// @ts-check
/* OSPREY Web Terminal — Session-tree context menus
 *
 * The right-click (and menu key / Shift+F10) menus of the sidebar's session
 * tree, one item list per row kind, opened through the hub's shared context
 * menu (panel-context-menu.js):
 *
 *   Agent row   New session · Group by ▸ Kind / Activity · Show pinned sessions ·
 *               Collapse all / Expand all
 *   Session     Open (Focus on the active one) · Pin / Unpin · Rename… · Use
 *               automatic name (only on a hand-named session) · Export transcript ·
 *               Copy ID · Delete… (disabled on the active one)
 *   Group head  Collapse / Expand · Collapse others
 *   Run         artifact-verbs.js's run rows
 *   Artifact    artifact-verbs.js's artifact rows, Open being the row's click
 *
 * The pinned group's heading has its own menu (sidebar-session-pins.js).
 *
 * Every menu is built synchronously from what the tree already holds
 * (heldEntry), so it opens offline; a run's part row resolves through its run
 * head's `run_parts`. A row with neither is left to the browser's menu.
 *
 * Where a verb is a gesture the tree already answers (open a session, browse
 * it, open a group, show an artifact) the item clicks that row, so a
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

import { withPrefix } from './api.js';
import { declinesToBrowser } from '/design-system/js/context-menu.js';
import { openContextMenu } from './panel-context-menu.js';
import { heldEntry, readTreeView } from './sidebar-artifact-listing.js';
import { artifactMenuItems, copyText, runMenuItems } from './artifact-verbs.js';
import { showConfirm } from './posture-confirm.js';
import { setSessionPinned, showPinnedItem } from './sidebar-session-pins.js';

/** @typedef {import('./panel-context-menu.js').MenuItem} MenuItem */
/** @typedef {import('./sidebar-artifact-listing.js').ArtifactEntry} ArtifactEntry */
/** @typedef {import('./sidebar-artifact-rows.js').TreeEntry} TreeEntry */

const TREE_SELECTOR = '[data-session-tree]';
const SESSION_SELECTOR = '.osprey-tree-session[data-session-id]';
const GROUP_SELECTOR = '[data-group-head]';
const RUN_SELECTOR = '.osprey-list-row.osprey-tree-run[data-id]';
const ARTIFACT_SELECTOR = '.osprey-list-row[data-id]';
const RUN_HEAD_CLASS = 'osprey-tree-run';
/** Top-level rows Expand/Collapse all acts on: the sessions of the session
 * list (not their pinned copies). Only a row with a disclosure folds (the
 * session in use); another session opens only by switching to it. */
const TOP_SELECTOR = `${SESSION_SELECTOR}:not([data-where="pinned"])`;

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

/** @param {HTMLElement} row */
const isOpen = (row) => row.getAttribute('aria-expanded') === 'true';

// ---- Agent row ----

/** @returns {boolean} whether every top-level tree row is open */
function allOpen() {
  const rows = [...document.querySelectorAll(TOP_SELECTOR)];
  return rows.length > 0 && rows.every((row) => row instanceof HTMLElement && isOpen(row));
}

/**
 * The same gesture as a ⌥-click on the Agent row (sidebar.js, imported lazily
 * like the session model: it sits behind panel-menu-policy).
 * @param {boolean} open
 */
async function setAllSessions(open) {
  (await import('./sidebar.js')).toggleEverySession(open);
}

/**
 * The Agent row's menu: a new session, Group by ▸ Kind / Activity, Show
 * pinned sessions, and one Collapse all / Expand all row worded for what it
 * will do (Collapse all when every session is open).
 * @returns {MenuItem[]}
 */
export function buildAgentRowMenuItems() {
  const view = readTreeView();
  return [
    { label: 'New session', run: () => void sessionModel().then((m) => m.switchTo({ new: true })) },
    { divider: true },
    {
      label: 'Group by',
      submenu: [
        { label: 'Kind', checked: view === 'kind', run: () => void applyView('kind') },
        { label: 'Activity', checked: view === 'activity', run: () => void applyView('activity') },
      ],
    },
    showPinnedItem(),
    { divider: true },
    allOpen()
      ? { label: 'Collapse all', run: () => void setAllSessions(false) }
      : { label: 'Expand all', run: () => void setAllSessions(true) },
  ];
}

// ---- Session ----

/**
 * Put the operator's title; an empty one reverts to the automatic name.
 * @param {string} transcriptId  @param {string} title
 */
export async function putTitle(transcriptId, title) {
  try {
    await fetch(withPrefix(`/api/sessions/${encodeURIComponent(transcriptId)}/title`), {
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
  a.href = withPrefix(`/api/sessions/${encodeURIComponent(id)}/transcript.md`);
  a.download = `transcript-${id}.md`;
  document.body.append(a);
  a.click();
  a.remove();
}

/** The tip on a disabled session Delete…: the Agent is on that session. */
export const DELETE_ACTIVE_TIP = 'Switch to another session to delete this one';

/**
 * Ask first, then delete a past session with its artifacts
 * (DELETE /api/sessions/{id}). The count is the row's, once the tree has
 * summarized it. On success the row leaves the model at once; other tabs drop it on the
 * hub's `session_deleted` frame.
 * @param {HTMLElement} row  @param {string} transcriptId
 */
export function confirmDeleteSession(row, transcriptId) {
  const name = row.querySelector('.osprey-tree-title')?.textContent || 'Untitled session';
  const n = row.dataset.count;
  const what = n === undefined ? 'its artifacts' : `its ${n} artifact${n === '1' ? '' : 's'}`;
  showConfirm({
    title: 'Delete session?',
    body: [['This deletes the session ', { em: name }, ` and ${what}.`]],
    live: null,
    confirmLabel: 'Delete',
    focus: 'cancel',
    onConfirm: async (ui) => {
      try {
        const res = await fetch(withPrefix(`/api/sessions/${encodeURIComponent(transcriptId)}`), {
          method: 'DELETE',
        });
        if (!res.ok) {
          const detail = await res.json().then(
            (b) => b?.detail,
            () => null
          );
          throw new Error(typeof detail === 'string' ? detail : `HTTP ${res.status}`);
        }
        ui.done();
        (await sessionModel()).forgetSession(transcriptId);
      } catch (err) {
        ui.error.textContent = `Delete failed: ${err instanceof Error ? err.message : err}`;
        ui.error.hidden = false;
      }
    },
  });
}

/**
 * Bring the Agent pane forward (panel-menu-policy, imported lazily: it
 * imports this module).
 */
async function focusAgent() {
  (await import('./panel-menu-policy.js')).focusAgentPane();
}

/**
 * A session row's menu. Open is the row's click (the controller switches the
 * Agent to it); on the session the Agent is on it is Focus.
 * @param {HTMLElement} row
 * @returns {MenuItem[]}
 */
export function sessionMenuItems(row) {
  const treeId = row.dataset.sessionId ?? '';
  // A synthesized row (`key:<K>`) has no transcript yet: nothing to name,
  // export or copy.
  const tid = treeId && !treeId.startsWith('key:') ? treeId : null;
  // The session the Agent is on draws `.active`.
  const active = row.classList.contains('active');
  const pinned = Boolean(row.dataset.pinned);
  return [
    active
      ? { label: 'Focus', run: () => void focusAgent() }
      : { label: 'Open', run: () => row.click() },
    ...(tid
      ? [
          {
            label: pinned ? 'Unpin' : 'Pin',
            run: () => void setSessionPinned(tid, !pinned),
          },
        ]
      : []),
    { divider: true },
    { label: 'Rename…', disabled: !tid, run: () => tid && startRename(row, tid) },
    // Only a hand-set name has an automatic one to go back to.
    ...(tid && row.dataset.titleSource === 'user'
      ? [{ label: 'Use automatic name', run: () => void putTitle(tid, '') }]
      : []),
    { label: 'Export transcript', disabled: !tid, run: () => tid && exportTranscript(tid) },
    { label: 'Copy ID', disabled: !tid, run: () => tid && copyText(tid) },
    { divider: true },
    {
      label: 'Delete…',
      danger: true,
      disabled: !tid || active,
      tip: active ? DELETE_ACTIVE_TIP : undefined,
      run: () => tid && confirmDeleteSession(row, tid),
    },
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

/** @param {HTMLElement} head  @returns {string} the session part of `${treeId}:${form}` */
const groupSession = (head) => {
  const key = head.dataset.groupHead ?? '';
  return key.slice(0, key.lastIndexOf(':'));
};

/**
 * The other open group heads of a head's session.
 * @param {HTMLElement} head
 * @returns {string[]} their `data-group-head` keys
 */
function otherOpenGroups(head) {
  const session = groupSession(head);
  const scope = head.closest(TREE_SELECTOR) ?? document;
  return [...scope.querySelectorAll(GROUP_SELECTOR)]
    .filter(
      (h) => h instanceof HTMLElement && h !== head && isOpen(h) && groupSession(h) === session
    )
    .map((h) => /** @type {HTMLElement} */ (h).dataset.groupHead ?? '');
}

/**
 * A group head's menu: Collapse / Expand, and Collapse others (each other
 * open group of the same session, by its own click; every click redraws the
 * tree, so each head is looked up afresh).
 * @param {HTMLElement} head
 * @returns {MenuItem[]}
 */
export function groupMenuItems(head) {
  const others = otherOpenGroups(head);
  const tree = head.closest(TREE_SELECTOR);
  return [
    { label: isOpen(head) ? 'Collapse' : 'Expand', run: () => head.click() },
    {
      label: 'Collapse others',
      disabled: others.length === 0,
      run: () => {
        for (const key of others) {
          const h = rowBy(tree, GROUP_SELECTOR, 'groupHead', key);
          if (h && isOpen(h)) h.click();
        }
      },
    },
  ];
}

// ---- Artifact and run ----
// The rows are artifact-verbs.js's, the one producer every place shares.

// ---- Opening ----

/**
 * A run part row's entry, from its run head's `run_parts`: part rows are the
 * head's following siblings, not its children.
 * @param {HTMLElement} row
 * @returns {ArtifactEntry | null}
 */
function partEntry(row) {
  const id = row.dataset.id ?? '';
  for (let el = row.previousElementSibling; el; el = el.previousElementSibling) {
    if (!(el instanceof HTMLElement) || !el.classList.contains(RUN_HEAD_CLASS)) continue;
    const head = heldEntry(el.dataset.id);
    const part = /** @type {TreeEntry | null} */ (head)?.run_parts?.find((p) => p.id === id);
    return part
      ? /** @type {ArtifactEntry} */ ({ id: part.id, title: part.title ?? part.id })
      : null;
  }
  return null;
}

/**
 * The menu a tree row offers, or null when it offers none (the browser's).
 * Built synchronously from what the tree holds: nothing is fetched.
 * @param {Element} target
 * @returns {{row: HTMLElement, label: string, items: MenuItem[]} | null}
 */
export function menuFor(target) {
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
  const isRun = row.matches(RUN_SELECTOR);
  const entry = heldEntry(row.dataset.id) ?? (isRun ? null : partEntry(row));
  if (!entry) return null;
  if (isRun) {
    const execId = entry.exec_id || entry.id;
    return { row, label: 'Run actions', items: runMenuItems(execId, entry) };
  }
  const open = () => row.click();
  return { row, label: 'Artifact actions', items: artifactMenuItems(entry, { open }) };
}

/**
 * Open a tree row's menu at a viewport position.
 * @param {Element} target  @param {number} x  @param {number} y
 * @returns {boolean} whether a menu opened
 */
export function openTreeMenu(target, x, y) {
  const menu = menuFor(target);
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
  if (declinesToBrowser(e)) return;
  const el = inTree(e.target);
  if (el && openTreeMenu(el, e.clientX, e.clientY)) e.preventDefault();
}

/** The menu key and Shift+F10 open the focused row's menu at its centre. @param {KeyboardEvent} e */
function onKeyDown(e) {
  if (e.key !== 'ContextMenu' && !(e.key === 'F10' && e.shiftKey)) return;
  const el = inTree(e.target);
  if (!el) return;
  const rect = el.getBoundingClientRect();
  if (openTreeMenu(el, rect.left + rect.width / 2, rect.top + rect.height / 2)) e.preventDefault();
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
