// @ts-check
/* OSPREY Web Terminal — the Workspace divider
 *
 * A faint horizontal line below the group that holds the Agent row (the
 * Workspace group, unless the operator moved the groups) is a drag handle for
 * that group's height, so the session tree can be made taller on demand; the
 * groups after it move down and the rail scrolls when they no longer fit.
 *
 *   - LIMITS. At least 120 px (heading and about two rows); at most what
 *     leaves 56 px of the rail for the next group.
 *   - DRAG / KEYS. A pointer drag sets the height live; ArrowUp / ArrowDown
 *     step 16 px (64 with Shift). A double-click goes back to automatic sizing.
 *   - STORAGE. The chosen height is kept per user under
 *     `scopedStorageKey('osprey-sidebar-workspace-height')`, whole pixels,
 *     removed on reset. It is applied as `--sidebar-group-height` on the
 *     group, which terminal.css reads only while the sidebar shows labels and
 *     the group is not folded. The divider is hidden while that group is
 *     folded and while the sidebar is collapsed to icons.
 */

import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { SIDEBAR_RENDERED_EVENT } from './panel-rail.js';
import { FOLD_EVENT, isGroupFolded } from './sidebar-fold.js';

/** The height's storage key (whole pixels). */
export const HEIGHT_STORAGE_KEY = 'osprey-sidebar-workspace-height';

export const MIN_HEIGHT = 120;
const NEXT_GROUP_ROOM = 56;
const STEP = 16;
const BIG_STEP = 64;

/** @returns {number | null} the stored height, or null */
function readHeight() {
  try {
    const raw = window.localStorage.getItem(scopedStorageKey(HEIGHT_STORAGE_KEY));
    if (raw !== null && /^\d+$/.test(raw.trim()) && Number(raw) >= MIN_HEIGHT) return Number(raw);
  } catch {
    // Unreadable or unavailable storage: automatic sizing.
  }
  return null;
}

/** @param {number | null} height */
function writeHeight(height) {
  try {
    const key = scopedStorageKey(HEIGHT_STORAGE_KEY);
    if (height === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, String(height));
  } catch {
    // Storage can be unavailable (private mode, quota); the page keeps the height.
  }
}

/**
 * Wire the divider into the rail. Call once; no-op without a rail.
 * @param {HTMLElement | null} railOrNull
 * @returns {{height: () => number | null} | null}
 */
export function initSidebarDivider(railOrNull) {
  if (!railOrNull) return null;
  const rail = railOrNull;
  const root = document.documentElement;
  /** @type {number | null} */
  let height = readHeight();

  const divider = document.createElement('div');
  divider.className = 'sidebar-divider';
  divider.setAttribute('role', 'separator');
  divider.setAttribute('aria-orientation', 'horizontal');
  divider.setAttribute('aria-label', 'Workspace height');
  divider.setAttribute('tabindex', '0');
  divider.dataset.tip = 'Resize workspace';
  divider.dataset.tipDetail = 'Drag to set the Workspace height; double-click resets it';

  /** @returns {HTMLElement | null} the group holding the Agent row */
  const workspace = () =>
    /** @type {HTMLElement | null} */ (
      rail.querySelector(':scope > .sidebar-group:has(#sidebar-agent-row)')
    );

  /**
   * The height the rail can have: the sidebar's content box minus its in-flow
   * siblings. The rail itself is content-sized, so its own clientHeight moves
   * when a group folds, and a cap taken from it would re-clamp (and shift) the
   * Workspace group on every fold. Without a laid-out sidebar, the rail's own.
   * @returns {number}
   */
  const railRoom = () => {
    const side = rail.parentElement;
    if (!side || side.clientHeight <= 0) return rail.clientHeight;
    const cs = window.getComputedStyle(side);
    let room =
      side.clientHeight - (parseFloat(cs.paddingTop) || 0) - (parseFloat(cs.paddingBottom) || 0);
    for (const el of Array.from(side.children)) {
      if (el === rail || !(el instanceof HTMLElement)) continue;
      const pos = window.getComputedStyle(el).position;
      if (pos === 'absolute' || pos === 'fixed') continue;
      room -= el.offsetHeight;
    }
    return room;
  };

  /** @returns {number} the tallest the group may be, px */
  const maxHeight = () => Math.max(MIN_HEIGHT, railRoom() - NEXT_GROUP_ROOM);
  /** @param {number} px */
  const clamp = (px) => Math.min(maxHeight(), Math.max(MIN_HEIGHT, Math.round(px)));

  function paint() {
    const group = workspace();
    for (const el of rail.querySelectorAll(':scope > .sidebar-group[data-sized]')) {
      if (el !== group) {
        el.removeAttribute('data-sized');
        /** @type {HTMLElement} */ (el).style.removeProperty('--sidebar-group-height');
      }
    }
    if (!group) {
      divider.remove();
      return;
    }
    if (divider.previousElementSibling !== group) group.after(divider);
    group.toggleAttribute('data-sized', height !== null);
    if (height !== null) group.style.setProperty('--sidebar-group-height', `${clamp(height)}px`);
    else group.style.removeProperty('--sidebar-group-height');
    divider.hidden =
      isGroupFolded(group.getAttribute('data-group') ?? '') ||
      root.getAttribute('data-sidebar') === 'collapsed';
    const now = height === null ? group.offsetHeight : clamp(height);
    divider.setAttribute('aria-valuemin', String(MIN_HEIGHT));
    divider.setAttribute('aria-valuemax', String(maxHeight()));
    divider.setAttribute('aria-valuenow', String(now));
    divider.setAttribute('aria-valuetext', height === null ? 'Automatic' : `${now} pixels`);
  }

  /** @param {number | null} next */
  function commit(next) {
    height = next === null ? null : clamp(next);
    paint();
    writeHeight(height);
  }

  divider.addEventListener('pointerdown', (event) => {
    const group = workspace();
    if (event.button !== 0 || !group) return;
    event.preventDefault();
    try {
      divider.setPointerCapture(event.pointerId);
    } catch {
      // Without capture the drag still tracks while the pointer is on the divider.
    }
    divider.classList.add('dragging');
    root.setAttribute('data-sidebar-dividing', '');
    const startY = event.clientY;
    const startHeight = group.offsetHeight;
    let moved = false;
    /** @param {PointerEvent} ev */
    const move = (ev) => {
      if (ev.clientY !== startY) moved = true;
      height = clamp(startHeight + ev.clientY - startY);
      paint();
    };
    const end = () => {
      divider.removeEventListener('pointermove', move);
      divider.removeEventListener('pointerup', end);
      divider.removeEventListener('pointercancel', end);
      divider.classList.remove('dragging');
      root.removeAttribute('data-sidebar-dividing');
      // A press that never moved is half of a double-click: nothing to store.
      if (moved && height !== null) writeHeight(height);
    };
    divider.addEventListener('pointermove', move);
    divider.addEventListener('pointerup', end);
    divider.addEventListener('pointercancel', end);
  });

  divider.addEventListener('dblclick', () => commit(null));

  divider.addEventListener('keydown', (event) => {
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
    const group = workspace();
    if (!group) return;
    event.preventDefault();
    const step = (event.shiftKey ? BIG_STEP : STEP) * (event.key === 'ArrowDown' ? 1 : -1);
    commit((height === null ? group.offsetHeight : clamp(height)) + step);
  });

  document.addEventListener(SIDEBAR_RENDERED_EVENT, paint);
  document.addEventListener(FOLD_EVENT, paint);
  document.addEventListener('osprey-collapsible-change', paint);
  window.addEventListener('resize', paint);
  paint();

  return { height: () => height };
}
