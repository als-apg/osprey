// @ts-check
/* OSPREY Web Terminal — the Workspace divider
 *
 * A faint horizontal line below the group that holds the Agent row (the
 * Workspace group, unless the operator moved the groups) is a drag handle for
 * that group's height, so the session tree can be made taller on demand; the
 * groups after it move down and the rail scrolls when they no longer fit.
 *
 *   - AUTOMATIC. With nothing stored the group is as tall as its content, up
 *     to a cap: the room the rail leaves once every other group is whole,
 *     kept between 240 and 480 px. Past the cap the tree scrolls inside the
 *     group, so a small project keeps the line high and a big one stops it
 *     at the cap. The cap is `--sidebar-group-cap` on the group.
 *   - LIMITS. At least 120 px (heading and about two rows); at most what
 *     leaves 56 px of the rail for the next group. The cap obeys the same
 *     maximum.
 *   - DRAG / KEYS. A pointer drag sets the height live; ArrowUp / ArrowDown
 *     step 16 px (64 with Shift). While dragging, a dashed tick marks the
 *     automatic height and the line snaps to it within 10 px; ending a drag
 *     on it, or a double-click, goes back to automatic sizing.
 *   - STORAGE. The chosen height is kept per user under
 *     `scopedStorageKey('osprey-sidebar-workspace-height')` as a fraction of
 *     the rail's room (`0.4583`), so it holds when the window is resized;
 *     removed on reset. A whole-pixel value from an earlier build is read
 *     once and rewritten as a fraction. The height is applied as
 *     `--sidebar-group-height` on the group, which terminal.css reads only
 *     while the sidebar shows labels and the group is not folded. The divider
 *     is hidden while that group is folded and while the sidebar is collapsed
 *     to icons.
 *   - ROWS KEEP THEIR HEIGHT. When the group runs short of room only an open
 *     session's artifact box gives way, down to min(its content, 3 rows);
 *     past that the group scrolls. The box floor is in terminal.css; this
 *     module adds up the tree's and the Agent row's floors from it
 *     (`--sidebar-tree-floor`, `--sidebar-row-floor`) whenever the tree
 *     changes, so neither shrinks below what it holds.
 */

import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { SIDEBAR_RENDERED_EVENT } from './panel-rail.js';
import { FOLD_EVENT, isGroupFolded } from './sidebar-fold.js';

/** The height's storage key (a fraction of the rail's room). */
export const HEIGHT_STORAGE_KEY = 'osprey-sidebar-workspace-height';

export const MIN_HEIGHT = 120;
const NEXT_GROUP_ROOM = 56;
export const CAP_MIN = 240;
export const CAP_MAX = 480;
export const SNAP = 10;
const STEP = 16;
/** The custom properties this module sets on the group. */
const GROUP_PROPERTIES = [
  '--sidebar-group-height',
  '--sidebar-group-cap',
  '--sidebar-tree-floor',
  '--sidebar-row-floor',
];
const BIG_STEP = 64;

/**
 * The stored height: a fraction of the rail's room, or whole pixels written
 * by an earlier build (converted on the first paint with a laid-out rail).
 * @typedef {{fraction: number} | {px: number}} Stored
 */

/** @returns {Stored | null} */
function readStored() {
  try {
    const raw = window.localStorage.getItem(scopedStorageKey(HEIGHT_STORAGE_KEY))?.trim();
    if (raw && /^\d+$/.test(raw) && Number(raw) >= MIN_HEIGHT) return { px: Number(raw) };
    if (raw && /^\d*\.\d+$/.test(raw) && Number(raw) > 0 && Number(raw) <= 1)
      return { fraction: Number(raw) };
  } catch {
    // Unreadable or unavailable storage: automatic sizing.
  }
  return null;
}

/** @param {number | null} fraction */
function writeFraction(fraction) {
  try {
    const key = scopedStorageKey(HEIGHT_STORAGE_KEY);
    if (fraction === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, fraction.toFixed(4));
  } catch {
    // Storage can be unavailable (private mode, quota); the page keeps the height.
  }
}

/** @param {Element} el @returns {number} the element's vertical margins, px */
function marginsOf(el) {
  const cs = window.getComputedStyle(el);
  return (parseFloat(cs.marginTop) || 0) + (parseFloat(cs.marginBottom) || 0);
}

/**
 * The least height a column flex container can take without squashing a
 * child: its padding and borders, every child that does not shrink at its
 * own height, and every child that does at its floor (`floors`, or its CSS
 * min-height).
 * @param {HTMLElement} el
 * @param {Map<Element, number>} [floors]
 * @returns {number}
 */
function floorOf(el, floors = new Map()) {
  const cs = window.getComputedStyle(el);
  let px = ['paddingTop', 'paddingBottom', 'borderTopWidth', 'borderBottomWidth'].reduce(
    (sum, prop) => sum + (parseFloat(/** @type {any} */ (cs)[prop]) || 0),
    0
  );
  for (const kid of Array.from(el.children)) {
    if (!(kid instanceof HTMLElement)) continue;
    const kcs = window.getComputedStyle(kid);
    if (kcs.display === 'none' || kcs.position === 'absolute' || kcs.position === 'fixed') continue;
    const floor = floors.get(kid);
    if (floor !== undefined) px += floor;
    else if (kcs.flexShrink === '0') px += kid.offsetHeight;
    else px += Math.min(kid.offsetHeight, parseFloat(kcs.minHeight) || 0);
    px += marginsOf(kid);
  }
  return Math.ceil(px);
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
  let stored = readStored();
  /** The live height while a drag runs, px. @type {number | null} */
  let dragHeight = null;

  const divider = document.createElement('div');
  divider.className = 'sidebar-divider';
  divider.setAttribute('role', 'separator');
  divider.setAttribute('aria-orientation', 'horizontal');
  divider.setAttribute('aria-label', 'Workspace height');
  divider.setAttribute('tabindex', '0');
  divider.dataset.tip = 'Resize workspace';
  divider.dataset.tipDetail =
    'Drag to set the Workspace height; drag onto the mark or double-click to reset it';
  // The automatic height's mark, drawn while a drag runs.
  const tick = document.createElement('div');
  tick.className = 'sidebar-divider-default';
  tick.setAttribute('aria-hidden', 'true');
  tick.hidden = true;
  divider.append(tick);

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

  /**
   * The automatic height's ceiling: what the rail leaves once every other
   * group is whole, kept in [CAP_MIN, CAP_MAX] and under the maximum.
   * @param {HTMLElement} group
   * @returns {number}
   */
  const capOf = (group) => {
    const cs = window.getComputedStyle(rail);
    let rest = (parseFloat(cs.paddingTop) || 0) + (parseFloat(cs.paddingBottom) || 0);
    for (const el of Array.from(rail.children)) {
      if (el !== group && el !== divider && el instanceof HTMLElement) rest += el.offsetHeight;
    }
    const cap = Math.max(CAP_MIN, Math.min(CAP_MAX, Math.round(railRoom() - rest)));
    return Math.min(maxHeight(), cap);
  };

  /** @returns {number | null} the stored height in px, or null for automatic */
  const storedHeight = () => {
    if (stored === null) return null;
    if ('px' in stored) return clamp(stored.px);
    return clamp(stored.fraction * railRoom());
  };

  /**
   * The group's automatic height, measured with any set height taken off.
   * @param {HTMLElement} group
   * @returns {number}
   */
  const automaticHeight = (group) => {
    const sized = group.hasAttribute('data-sized');
    group.removeAttribute('data-sized');
    const px = group.offsetHeight;
    if (sized) group.setAttribute('data-sized', '');
    return px;
  };

  /**
   * Keep the tree and the Agent row from shrinking below what they hold
   * (only the artifact boxes give, to their CSS floor), and the cap current.
   */
  function measure() {
    const group = workspace();
    if (!group) return;
    group.style.setProperty('--sidebar-group-cap', `${capOf(group)}px`);
    const tree = group.querySelector(':scope > .sidebar-row > [data-session-tree]');
    const row = tree?.parentElement;
    if (!(tree instanceof HTMLElement) || !row) return;
    const treeFloor = floorOf(tree);
    group.style.setProperty('--sidebar-tree-floor', `${treeFloor}px`);
    group.style.setProperty(
      '--sidebar-row-floor',
      `${floorOf(row, new Map([[tree, treeFloor]]))}px`
    );
  }

  function paint() {
    const group = workspace();
    for (const el of rail.querySelectorAll(':scope > .sidebar-group[data-capped]')) {
      if (el === group) continue;
      el.removeAttribute('data-sized');
      el.removeAttribute('data-capped');
      for (const prop of GROUP_PROPERTIES)
        /** @type {HTMLElement} */ (el).style.removeProperty(prop);
    }
    if (!group) {
      divider.remove();
      return;
    }
    if (divider.previousElementSibling !== group) group.after(divider);
    // A whole-pixel height from an earlier build keeps its pixels for the rest
    // of this page and is stored as their fraction of the rail, rewritten while
    // the sidebar's head settles; the next load reads the fraction.
    if (stored && 'px' in stored && railRoom() > 0) writeFraction(clamp(stored.px) / railRoom());
    const height = dragHeight ?? storedHeight();
    group.setAttribute('data-capped', '');
    group.toggleAttribute('data-sized', height !== null);
    if (height !== null) group.style.setProperty('--sidebar-group-height', `${height}px`);
    else group.style.removeProperty('--sidebar-group-height');
    measure();
    divider.hidden =
      isGroupFolded(group.getAttribute('data-group') ?? '') ||
      root.getAttribute('data-sidebar') === 'collapsed';
    updateAria();
  }

  /**
   * The separator's value, read from the group as it renders now (a short
   * tree can render under the minimum, so it is kept in range; a set height
   * or a drag's live height is used as is). Called after every paint and
   * whenever the tree changes under an unchanged paint, so it never goes
   * stale. Aria writes are not observed, so this never feeds itself.
   */
  function updateAria() {
    const group = workspace();
    if (!group) return;
    const height = dragHeight ?? storedHeight();
    const now = height ?? clamp(group.offsetHeight);
    divider.setAttribute('aria-valuemin', String(MIN_HEIGHT));
    divider.setAttribute('aria-valuemax', String(maxHeight()));
    divider.setAttribute('aria-valuenow', String(now));
    divider.setAttribute('aria-valuetext', height === null ? 'Automatic' : `${now} pixels`);
  }

  /** @param {number | null} px  a height to keep, or null for automatic */
  function commit(px) {
    const room = railRoom();
    if (px === null) stored = null;
    else stored = room > 0 ? { fraction: clamp(px) / room } : { px: clamp(px) };
    paint();
    if (stored === null) writeFraction(null);
    else if ('fraction' in stored) writeFraction(stored.fraction);
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
    const startHeight = storedHeight() ?? group.offsetHeight;
    const automatic = clamp(automaticHeight(group));
    let moved = false;
    /** @param {number} px */
    const place = (px) => {
      dragHeight = px;
      paint();
      tick.style.top = `${automatic - px}px`;
      tick.hidden = false;
    };
    /** @param {PointerEvent} ev */
    const move = (ev) => {
      if (ev.clientY !== startY) moved = true;
      const px = clamp(startHeight + ev.clientY - startY);
      place(Math.abs(px - automatic) <= SNAP ? automatic : px);
    };
    const end = () => {
      divider.removeEventListener('pointermove', move);
      divider.removeEventListener('pointerup', end);
      divider.removeEventListener('pointercancel', end);
      divider.classList.remove('dragging');
      root.removeAttribute('data-sidebar-dividing');
      tick.hidden = true;
      const px = dragHeight;
      dragHeight = null;
      // A press that never moved is half of a double-click: nothing to store.
      if (!moved || px === null) paint();
      else commit(px === automatic ? null : px);
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
    commit((storedHeight() ?? group.offsetHeight) + step);
  });

  document.addEventListener(SIDEBAR_RENDERED_EVENT, paint);
  document.addEventListener(FOLD_EVENT, paint);
  document.addEventListener('osprey-collapsible-change', paint);
  window.addEventListener('resize', () => {
    // A resized window keeps the share of the rail, an earlier build's pixels included.
    if (stored && 'px' in stored && railRoom() > 0)
      stored = { fraction: clamp(stored.px) / railRoom() };
    paint();
  });
  // The rail's room follows the sidebar's other blocks (the control target
  // and the footer render late): the stored fraction and the cap follow it.
  if (typeof ResizeObserver !== 'undefined' && rail.parentElement) {
    const sizes = new ResizeObserver(paint);
    for (const el of Array.from(rail.parentElement.children)) if (el !== rail) sizes.observe(el);
  }
  // The tree re-renders on its own (sessions arrive, a box loads its rows):
  // its floors and the separator's value follow. Not paint(): that sets
  // `hidden`, which this observer watches. Style and aria writes are not
  // observed, so measuring never feeds itself.
  new MutationObserver(() => {
    measure();
    updateAria();
  }).observe(rail, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['hidden'],
  });
  paint();

  return { height: () => storedHeight() };
}
