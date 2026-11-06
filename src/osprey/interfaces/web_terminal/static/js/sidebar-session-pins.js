// @ts-check
/* OSPREY Web Terminal — the sidebar's pinned sessions
 *
 * The operator pins sessions (the pin button on a session row, or Pin in its
 * menu); the pinned ones show in a group of their own directly above the
 * Sessions row, first pinned on top, and stay in the session list as well
 * (with a pin marker). The pin is the session record's (`pinned`,
 * `PUT /api/sessions/{id}/pin`, session-model.js); the agent is not told.
 *
 * THE GROUP is not one of sidebar-groups.js's: nothing moves above that
 * module's first group, and this one has no Move up / down or New group
 * below. It is a `.sidebar-pinned` element this module keeps as the first
 * child of `#panel-rail` (panel-rail.js leaves a child that is not a
 * `.sidebar-group` where it is), drawn while a session is pinned and the group
 * is not hidden. Its heading holds the name, the count and a chevron; a click,
 * Enter or Space folds or unfolds it. Its list is a `[data-session-tree]` with
 * `data-tree-where="pinned"`, which sidebar-artifacts.js fills with the same
 * session rows as the session list.
 *
 * MENUS. The heading's right-click (or Shift+F10 / the menu key): Expand or
 * Collapse, Rename, then Hide pinned sessions. Show pinned sessions, a
 * checkable item greyed while nothing is pinned, is on the Sessions row's menu
 * (sidebar-artifact-menus.js) and on a right-click of empty sidebar space.
 *
 * VIEW STATE. Hidden, folded and the name are this browser's, per user, under
 * `scopedStorageKey('osprey-sidebar-pinned')`. The first pin shows the group
 * folded; a pin made while the group is hidden shows it again; unpinning the
 * last session takes the group away and keeps its name.
 *
 * @module sidebar-session-pins
 */

import { icon } from '/design-system/js/icons.js';
import { isContextMenuKey } from '/design-system/js/context-menu.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { openContextMenu } from './panel-context-menu.js';
import { FOLD_EVENT } from './sidebar-fold.js';

/** @typedef {import('./panel-context-menu.js').MenuItem} MenuItem */

/** The view state's storage key. */
export const PINS_STORAGE_KEY = 'osprey-sidebar-pinned';

/** The group's name until the operator renames it. */
export const PINNED_LABEL = 'Pinned';

/** The longest group name, in characters (as sidebar-groups.js). */
const MAX_LABEL_LENGTH = 40;

/** The rail the group leads. */
const RAIL_ID = 'panel-rail';

/** @typedef {{hidden: boolean, folded: boolean, name: string}} PinsView */

/** @returns {PinsView} */
function readView() {
  /** @type {PinsView} */
  const view = { hidden: false, folded: true, name: PINNED_LABEL };
  try {
    const raw = window.localStorage.getItem(scopedStorageKey(PINS_STORAGE_KEY));
    const parsed = raw === null ? null : JSON.parse(raw);
    if (parsed && typeof parsed === 'object') {
      if (typeof parsed.hidden === 'boolean') view.hidden = parsed.hidden;
      if (typeof parsed.folded === 'boolean') view.folded = parsed.folded;
      if (typeof parsed.name === 'string' && parsed.name.trim())
        view.name = parsed.name.trim().slice(0, MAX_LABEL_LENGTH);
    }
  } catch {
    // Unreadable or unavailable storage: the defaults.
  }
  return view;
}

function writeView() {
  try {
    window.localStorage.setItem(scopedStorageKey(PINS_STORAGE_KEY), JSON.stringify(view));
  } catch {
    // Storage can be unavailable (private mode, quota); the page keeps the state.
  }
}

let view = readView();
/** Pinned sessions at the last paint. */
let count = 0;
/** @type {Set<() => void>} */
const listeners = new Set();

/**
 * The group's height changed (shown, hidden, folded, unfolded): the Sessions
 * divider re-measures the room it leaves on the fold event (sidebar-divider.js).
 */
function resized() {
  document.dispatchEvent(new CustomEvent(FOLD_EVENT));
}

/** Store the view and redraw. @param {Partial<PinsView>} change */
function update(change) {
  view = { ...view, ...change };
  writeView();
  for (const fn of [...listeners]) fn();
  resized();
}

/**
 * Redraw whenever the view changes (sidebar-artifacts.js renders the tree).
 * @param {() => void} fn
 * @returns {() => void} stops the calls
 */
export function onPinsChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** @returns {string} the group's name */
export const pinnedName = () => view.name;

/** @returns {boolean} whether the group is drawn (a session is pinned, the group not hidden) */
export const pinnedShown = () => count > 0 && !view.hidden;

/** @returns {boolean} whether the group's sessions are on view */
export const pinnedOpen = () => pinnedShown() && !view.folded;

/** @param {boolean} hidden */
export function setPinnedHidden(hidden) {
  update({ hidden });
}

/** @param {boolean} folded */
export function setPinnedFolded(folded) {
  update({ folded });
}

/**
 * Pin or unpin a session: the group shows (folded on the first pin), then the
 * model sets the pin. session-model.js is imported lazily: the menus import
 * this module, and a static edge to the model would close a cycle through the
 * chat (sidebar-artifact-menus.js says why).
 * @param {string} transcriptId
 * @param {boolean} pinned
 * @returns {Promise<boolean>} whether the hub took it
 */
export async function setSessionPinned(transcriptId, pinned) {
  if (pinned && (count === 0 || view.hidden))
    update({ hidden: false, folded: count === 0 ? true : view.folded });
  const model = await import('./session-model.js');
  return model.setPinned(transcriptId, pinned);
}

/**
 * The checkable Show pinned sessions item: checked while the group shows,
 * greyed while nothing is pinned.
 * @returns {MenuItem}
 */
export function showPinnedItem() {
  return {
    label: 'Show pinned sessions',
    checked: pinnedShown(),
    disabled: count === 0,
    tip: count === 0 ? 'No pinned sessions' : undefined,
    run: () => setPinnedHidden(!view.hidden),
  };
}

// ---- The group ----

/** @returns {HTMLElement | null} the group, made on first use */
function groupEl() {
  const rail = document.getElementById(RAIL_ID);
  if (!rail) return null;
  let group = /** @type {HTMLElement | null} */ (rail.querySelector(':scope > .sidebar-pinned'));
  if (!group) group = buildGroup();
  if (rail.firstElementChild !== group) rail.prepend(group);
  return group;
}

/** @returns {HTMLElement} */
function buildGroup() {
  const group = document.createElement('div');
  group.className = 'sidebar-pinned';
  group.setAttribute('role', 'listitem');
  group.setAttribute('aria-labelledby', 'sidebar-pinned-label');
  group.hidden = true;
  const head = document.createElement('div');
  head.className = 'osprey-section-label sidebar-group-head sidebar-pinned-head';
  head.setAttribute('role', 'button');
  head.tabIndex = 0;
  const name = document.createElement('span');
  name.className = 'sidebar-group-name';
  name.id = 'sidebar-pinned-label';
  const n = document.createElement('span');
  n.className = 'osprey-tree-n sidebar-pinned-count';
  const disc = icon('chevron-down');
  disc.classList.add('sidebar-pinned-disc');
  head.append(name, n, disc);
  const list = document.createElement('div');
  list.className = 'sidebar-kids osprey-tree sidebar-pinned-list';
  list.dataset.sessionTree = '';
  list.dataset.treeWhere = 'pinned';
  list.setAttribute('role', 'tree');
  list.setAttribute('aria-labelledby', 'sidebar-pinned-label');
  group.append(head, list);
  wireHead(head);
  return group;
}

/**
 * Draw the group for `pinned` sessions: shown or not, its name, count and fold.
 * The tree's own render fills its list.
 * @param {number} pinned
 */
export function paintPinnedGroup(pinned) {
  count = pinned;
  const group = groupEl();
  if (!group) return;
  const wasHidden = group.hidden;
  group.hidden = !pinnedShown();
  if (group.hidden !== wasHidden) resized();
  const head = /** @type {HTMLElement} */ (group.querySelector('.sidebar-pinned-head'));
  const name = /** @type {HTMLElement} */ (head.querySelector('.sidebar-group-name'));
  name.textContent = view.name;
  /** @type {HTMLElement} */ (head.querySelector('.sidebar-pinned-count')).textContent =
    String(pinned);
  head.setAttribute('aria-expanded', String(!view.folded));
  head.dataset.tip = view.name;
  head.dataset.tipDetail = view.folded ? 'Click to expand' : 'Click to collapse';
  /** @type {HTMLElement} */ (group.querySelector('.sidebar-pinned-list')).hidden = view.folded;
}

/** @param {HTMLElement} head */
function wireHead(head) {
  const renaming = () => Boolean(head.querySelector('.sidebar-group-rename'));
  head.addEventListener('click', () => {
    if (!renaming()) setPinnedFolded(!view.folded);
  });
  head.addEventListener('contextmenu', (event) => {
    if (renaming()) return;
    event.preventDefault();
    openHeadMenu(head, event.clientX, event.clientY);
  });
  head.addEventListener('keydown', (event) => {
    if (renaming()) return;
    if (isContextMenuKey(event)) {
      event.preventDefault();
      const rect = head.getBoundingClientRect();
      openHeadMenu(head, rect.left + 8, rect.bottom);
    } else if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      setPinnedFolded(!view.folded);
    }
  });
}

/**
 * The heading's menu items.
 * @param {HTMLElement} head
 * @returns {MenuItem[]}
 */
export function headMenuItems(head) {
  return [
    { label: view.folded ? 'Expand' : 'Collapse', run: () => setPinnedFolded(!view.folded) },
    { label: 'Rename', run: () => startPinnedRename(head) },
    { divider: true },
    { label: 'Hide pinned sessions', run: () => setPinnedHidden(true) },
  ];
}

/** @param {HTMLElement} head @param {number} x @param {number} y */
function openHeadMenu(head, x, y) {
  openContextMenu({
    x,
    y,
    anchorEl: head,
    ariaLabel: `${view.name} group`,
    items: headMenuItems(head),
  });
}

/**
 * Swap the name for a field: Enter or leaving it keeps the new name, Escape
 * and an empty name keep the old one (sidebar-groups.js's rename).
 * @param {HTMLElement} head
 * @returns {boolean} whether the field opened
 */
export function startPinnedRename(head) {
  const name = /** @type {HTMLElement | null} */ (head.querySelector('.sidebar-group-name'));
  if (!name || head.querySelector('.sidebar-group-rename')) return false;
  const field = document.createElement('input');
  field.type = 'text';
  field.className = 'sidebar-group-rename';
  field.value = view.name;
  field.maxLength = MAX_LABEL_LENGTH;
  field.setAttribute('aria-label', 'Group name');
  name.hidden = true;
  name.after(field);
  field.focus();
  field.select();
  let finished = false;
  /** @param {boolean} keep @param {boolean} refocus */
  const finish = (keep, refocus) => {
    if (finished) return;
    finished = true;
    const value = field.value.trim().slice(0, MAX_LABEL_LENGTH);
    field.remove();
    name.hidden = false;
    if (keep && value && value !== view.name) update({ name: value });
    if (refocus) head.focus();
  };
  field.addEventListener('keydown', (event) => {
    event.stopPropagation();
    if (event.key !== 'Enter' && event.key !== 'Escape') return;
    event.preventDefault();
    finish(event.key === 'Enter', true);
  });
  field.addEventListener('click', (event) => event.stopPropagation());
  field.addEventListener('blur', () => finish(true, false));
  return true;
}

// ---- Empty sidebar space ----

/** What counts as empty sidebar space: the containers themselves, not a row in them. */
const EMPTY_SPACE = '#sidebar, #panel-rail, #panel-rail > .sidebar-group, .sidebar-fold-spacer';

/** @param {MouseEvent} event */
function onSidebarContextMenu(event) {
  const target = event.target instanceof Element ? event.target : null;
  if (!target?.matches(EMPTY_SPACE)) return;
  if (document.documentElement.getAttribute('data-sidebar') === 'collapsed') return;
  event.preventDefault();
  openContextMenu({
    x: event.clientX,
    y: event.clientY,
    anchorEl: /** @type {HTMLElement} */ (target),
    ariaLabel: 'Sidebar',
    items: [showPinnedItem()],
  });
}

/** @type {AbortController | null} */
let wired = null;

/** Listen for right-clicks on empty sidebar space. Idempotent. */
export function initPinnedMenus() {
  if (wired) return;
  wired = new AbortController();
  document.addEventListener('contextmenu', onSidebarContextMenu, { signal: wired.signal });
}

/** Forget everything (tests). */
export function resetSessionPins() {
  wired?.abort();
  wired = null;
  listeners.clear();
  count = 0;
  view = readView();
  document.querySelector(`#${RAIL_ID} > .sidebar-pinned`)?.remove();
}
