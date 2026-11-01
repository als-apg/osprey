// @ts-check
/* OSPREY Web Terminal — the sidebar's resize handle
 *
 * The handle (`#sidebar-resize` in index.html, a `role="separator"` on the
 * sidebar's right edge) sets the labelled sidebar's width. The width is the
 * `--sidebar-user-width` property on <html>, which `.shell-body` reads in
 * place of the `--wt-sidebar-width` token (terminal.css); the collapsed rule
 * overrides both, so a stored width never reaches the 56 px icon rail.
 *
 *   - LIMITS. 180–400 px, and never so wide that the workspace keeps less than
 *     480 px; the cap follows the window, the stored width does not.
 *   - DRAG. A pointer drag (captured) sets the width live, with the grid's
 *     transition off (`html[data-sidebar-resizing]`). Dragging more than 40 px
 *     below the minimum collapses the sidebar, and dragging a collapsed one
 *     40 px outward expands it to the stored width; both go through the labels
 *     toggle's collapsible.js, so there is one collapsed state. A collapse by
 *     drag keeps the width the drag started from.
 *   - KEYS. ArrowLeft / ArrowRight step 8 px (32 px with Shift), Home and End
 *     go to the limits; any of them expands a collapsed sidebar. A
 *     double-click goes back to the token's width.
 *   - STORAGE. The width is stored when a drag ends, a key steps it or a
 *     double-click resets it, under `scopedStorageKey('osprey-sidebar-width')`
 *     — the key sidebar-boot.js reads before first paint, with the same limits.
 *     A stored value that is not a whole number in range is ignored.
 *
 * A drag closes every sidebar popover and the tooltip, as a width change by
 * the toggle does (sidebar.js).
 */

import { closeAllPopovers } from '/design-system/js/popover.js';
import { hideTooltip } from '/design-system/js/tooltip.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';

/** The width's storage key, shared with sidebar-boot.js (pre-paint). */
export const SIDEBAR_WIDTH_STORAGE_KEY = 'osprey-sidebar-width';

/** The <html> property `.shell-body` reads for the labelled width. */
const WIDTH_PROPERTY = '--sidebar-user-width';

/** The <html> attribute that turns the grid's transition off for a drag. */
const RESIZING_ATTRIBUTE = 'data-sidebar-resizing';

/** The narrowest and widest labelled sidebar, px (mirrored in sidebar-boot.js). */
export const MIN_WIDTH = 180;
export const MAX_WIDTH = 400;

/** The workspace never gets narrower than this to make room for the sidebar. */
export const WORKSPACE_MIN_WIDTH = 480;

/** How far past a limit a drag goes before it collapses or expands. */
export const SNAP_MARGIN = 40;

/** The icon rail's width (the `--wt-sidebar-collapsed-width` token). */
const COLLAPSED_WIDTH = 56;

/** The token's width, read when the stylesheet is not there to ask. */
const FALLBACK_DEFAULT_WIDTH = 232;

/** Arrow-key steps, px. */
const STEP = 8;
const BIG_STEP = 32;

/** @returns {number} the `--wt-sidebar-width` token, px */
function defaultWidth() {
  const token = parseInt(
    getComputedStyle(document.documentElement).getPropertyValue('--wt-sidebar-width'),
    10
  );
  return Number.isFinite(token) && token > 0 ? token : FALLBACK_DEFAULT_WIDTH;
}

/**
 * The widest sidebar this window allows: {@link MAX_WIDTH}, less whatever the
 * workspace needs, never below {@link MIN_WIDTH}.
 * @param {number} [viewport] the window's width, px
 * @returns {number}
 */
export function maxWidth(viewport = window.innerWidth) {
  return Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, viewport - WORKSPACE_MIN_WIDTH));
}

/**
 * Clamp a width into the limits, rounded to whole pixels.
 * @param {number} width
 * @param {number} [viewport] the window's width, px
 * @returns {number}
 */
export function clampWidth(width, viewport = window.innerWidth) {
  return Math.min(maxWidth(viewport), Math.max(MIN_WIDTH, Math.round(width)));
}

/** @returns {number | null} the stored width, or null when none is usable */
export function readStoredWidth() {
  try {
    const raw = window.localStorage.getItem(scopedStorageKey(SIDEBAR_WIDTH_STORAGE_KEY));
    if (raw !== null && /^\d+$/.test(raw.trim())) {
      const width = Number(raw);
      if (width >= MIN_WIDTH && width <= MAX_WIDTH) return width;
    }
  } catch {
    // Unreadable or unavailable storage: the default.
  }
  return null;
}

/** @param {number | null} width the width to store, or null to forget it */
function writeStoredWidth(width) {
  try {
    const key = scopedStorageKey(SIDEBAR_WIDTH_STORAGE_KEY);
    if (width === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, String(width));
  } catch {
    // Storage can be unavailable (private mode, quota); the page keeps the width.
  }
}

/**
 * @typedef {object} SidebarResize
 * @property {() => number} width the labelled width on screen, px
 * @property {() => void} destroy detach every listener (a re-rendered shell)
 */

/**
 * Wire the resize handle. No-op (returns null) without a handle.
 * @param {object} options
 * @param {HTMLElement | null} options.handle the separator
 * @param {import('/design-system/js/collapsible.js').Collapsible} options.collapsible
 *   the labels toggle's state, which a snap drives
 * @returns {SidebarResize | null}
 */
export function initSidebarResize({ handle, collapsible }) {
  if (!handle) return null;
  const grip = handle;
  const root = document.documentElement;
  const controller = new AbortController();
  const { signal } = controller;

  /** The labelled width the operator chose; the window may show less. */
  let width = readStoredWidth() ?? defaultWidth();

  /** @returns {number} the labelled width this window shows */
  const shown = () => clampWidth(width);

  function paint() {
    const px = shown();
    root.style.setProperty(WIDTH_PROPERTY, `${px}px`);
    const collapsed = collapsible.isCollapsed();
    grip.setAttribute('aria-valuemin', String(MIN_WIDTH));
    grip.setAttribute('aria-valuemax', String(maxWidth()));
    grip.setAttribute('aria-valuenow', String(collapsed ? MIN_WIDTH : px));
    grip.setAttribute('aria-valuetext', collapsed ? 'Icons only' : `${px} pixels`);
  }

  /** @param {number} next the new labelled width; stored as shown */
  function commit(next) {
    width = clampWidth(next);
    if (collapsible.isCollapsed()) collapsible.set('expanded');
    paint();
    writeStoredWidth(width);
  }

  grip.addEventListener(
    'pointerdown',
    (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      try {
        grip.setPointerCapture(event.pointerId);
      } catch {
        // Without capture the drag still tracks while the pointer is on the handle.
      }
      closeAllPopovers();
      hideTooltip();
      root.setAttribute(RESIZING_ATTRIBUTE, '');
      grip.classList.add('dragging');

      const startX = event.clientX;
      const startWidth = width;
      // The drag measures from an origin that moves on every snap, so the
      // sidebar never jumps under the pointer by the snap's distance.
      let originX = startX;
      let originWidth = collapsible.isCollapsed() ? COLLAPSED_WIDTH : shown();
      let moved = false;

      /** @param {PointerEvent} ev */
      const move = (ev) => {
        if (ev.clientX !== startX) moved = true;
        const candidate = originWidth + ev.clientX - originX;
        if (collapsible.isCollapsed()) {
          if (candidate < COLLAPSED_WIDTH + SNAP_MARGIN) return;
          collapsible.set('expanded');
          paint();
          originX = ev.clientX;
          originWidth = shown();
          return;
        }
        if (candidate < MIN_WIDTH - SNAP_MARGIN) {
          width = startWidth;
          collapsible.set('collapsed');
          paint();
          originX = ev.clientX;
          originWidth = COLLAPSED_WIDTH;
          return;
        }
        width = clampWidth(candidate);
        paint();
      };

      const end = () => {
        grip.removeEventListener('pointermove', move);
        grip.removeEventListener('pointerup', end);
        grip.removeEventListener('pointercancel', end);
        // Settle the last width while the transition is still off, or a move
        // and a release in one frame would animate it.
        void grip.offsetWidth;
        root.removeAttribute(RESIZING_ATTRIBUTE);
        grip.classList.remove('dragging');
        // A press that never moved is a click (half of a double-click), and a
        // drag that ends collapsed keeps the width it started from.
        if (moved && !collapsible.isCollapsed()) writeStoredWidth(width);
      };

      grip.addEventListener('pointermove', move);
      grip.addEventListener('pointerup', end);
      grip.addEventListener('pointercancel', end);
    },
    { signal }
  );

  grip.addEventListener(
    'dblclick',
    () => {
      width = defaultWidth();
      if (collapsible.isCollapsed()) collapsible.set('expanded');
      paint();
      writeStoredWidth(null);
    },
    { signal }
  );

  grip.addEventListener(
    'keydown',
    (event) => {
      const step = event.shiftKey ? BIG_STEP : STEP;
      const collapsed = collapsible.isCollapsed();
      /** @type {number | null} */
      let next = null;
      if (event.key === 'ArrowRight') next = collapsed ? width : shown() + step;
      else if (event.key === 'ArrowLeft') next = collapsed ? null : shown() - step;
      else if (event.key === 'Home') next = MIN_WIDTH;
      else if (event.key === 'End') next = maxWidth();
      else return;
      event.preventDefault();
      if (next !== null) commit(next);
    },
    { signal }
  );

  window.addEventListener('resize', paint, { signal });
  document.addEventListener('osprey-collapsible-change', paint, { signal });
  paint();

  return { width: shown, destroy: () => controller.abort() };
}
