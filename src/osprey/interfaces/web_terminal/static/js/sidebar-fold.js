// @ts-check
/* OSPREY Web Terminal — folding a sidebar group
 *
 * A double-click on a group's heading folds the group to its heading; another
 * unfolds it. The heading's chevron (shown while folded, and on hover while
 * open) and Enter / Space on the group's name do the same. Any group folds,
 * built-in or the operator's; membership and order are untouched, and the
 * Agent row's own session-list fold (sidebar.js) is a separate thing.
 *
 * Folded group ids are kept per user under
 * `scopedStorageKey('osprey-sidebar-folded')`, a JSON array. A folded group is
 * `data-folded` and its name `aria-expanded="false"`; terminal.css hides the
 * rest of the group. The listeners sit on the rail, so a render that rebuilds
 * a heading loses nothing; the chevron and the name's button role are
 * re-applied after every render. With the sidebar collapsed to icons nothing
 * folds.
 */

import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { SIDEBAR_RENDERED_EVENT } from './panel-rail.js';

/** The folded-groups storage key (a JSON array of group ids). */
export const FOLDED_STORAGE_KEY = 'osprey-sidebar-folded';

/** Fired on `document` after a group folds or unfolds. */
export const FOLD_EVENT = 'osprey-sidebar-fold-change';

/** @returns {Set<string>} the folded group ids, as stored */
function readFolded() {
  try {
    const raw = window.localStorage.getItem(scopedStorageKey(FOLDED_STORAGE_KEY));
    const parsed = raw === null ? null : JSON.parse(raw);
    if (Array.isArray(parsed)) return new Set(parsed.filter((id) => typeof id === 'string'));
  } catch {
    // Unreadable or unavailable storage: nothing folded.
  }
  return new Set();
}

/** @param {Set<string>} folded */
function writeFolded(folded) {
  try {
    window.localStorage.setItem(scopedStorageKey(FOLDED_STORAGE_KEY), JSON.stringify([...folded]));
  } catch {
    // Storage can be unavailable (private mode, quota); the page keeps the state.
  }
}

/** @type {Set<string>} */
let folded = new Set();

/**
 * @param {string} id
 * @returns {boolean} whether group `id` is folded
 */
export function isGroupFolded(id) {
  return folded.has(id);
}

/**
 * Give every heading its chevron and button role, and every group its fold state.
 * @param {HTMLElement} rail
 */
function paint(rail) {
  for (const group of rail.querySelectorAll(':scope > .sidebar-group')) {
    const head = group.querySelector(':scope > .sidebar-group-head');
    if (!head) continue;
    const isFolded = folded.has(group.getAttribute('data-group') ?? '');
    group.toggleAttribute('data-folded', isFolded);
    let chevron = head.querySelector(':scope > .sidebar-group-chevron');
    if (!chevron) {
      chevron = document.createElement('span');
      chevron.className = 'sidebar-group-chevron';
      chevron.setAttribute('aria-hidden', 'true');
      head.prepend(chevron);
    }
    chevron.textContent = isFolded ? '▸' : '▾';
    const name = head.querySelector('.sidebar-group-name');
    if (name) {
      name.setAttribute('role', 'button');
      name.setAttribute('tabindex', '0');
      name.setAttribute('aria-expanded', String(!isFolded));
    }
  }
}

/**
 * Fold or unfold a group and remember it.
 * @param {HTMLElement} rail
 * @param {string} id
 */
function toggle(rail, id) {
  if (folded.has(id)) folded.delete(id);
  else folded.add(id);
  writeFolded(folded);
  paint(rail);
  document.dispatchEvent(new CustomEvent(FOLD_EVENT));
}

/**
 * Wire group folding on the rail. Call once; no-op without a rail.
 * @param {HTMLElement | null} rail
 * @returns {void}
 */
export function initSidebarFold(rail) {
  if (!rail) return;
  folded = readFolded();
  paint(rail);
  document.addEventListener(SIDEBAR_RENDERED_EVENT, () => paint(rail));

  /**
   * The id of the group whose heading an event landed on, or null when it
   * landed on the ⋯ button or a rename field, or the sidebar shows icons only.
   * @param {Event} event
   * @returns {string | null}
   */
  const groupIdOf = (event) => {
    const target = event.target instanceof Element ? event.target : null;
    if (!target || target.closest('[data-group-menu], input')) return null;
    const group = target.closest('.sidebar-group-head')?.closest('.sidebar-group');
    if (!group || !rail.contains(group)) return null;
    if (document.documentElement.getAttribute('data-sidebar') === 'collapsed') return null;
    return group.getAttribute('data-group') ?? '';
  };

  rail.addEventListener('dblclick', (event) => {
    const id = groupIdOf(event);
    if (id === null) return;
    event.preventDefault();
    toggle(rail, id);
  });
  rail.addEventListener('click', (event) => {
    const target = event.target instanceof Element ? event.target : null;
    if (!target?.closest('.sidebar-group-chevron')) return;
    const id = groupIdOf(event);
    if (id !== null) toggle(rail, id);
  });
  rail.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    const target = event.target instanceof Element ? event.target : null;
    if (!target?.matches('.sidebar-group-name')) return;
    const id = groupIdOf(event);
    if (id === null) return;
    event.preventDefault();
    toggle(rail, id);
  });
}
