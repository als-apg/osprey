// @ts-check
/*
 * OSPREY Design System — the panel sidebar.
 *
 * The optional sidebar of a panel (DESIGN.md §12, "Panel chrome standard"):
 * left of the content, below the panel toolbar, 260 px by default and
 * resizable with the shared splitter. Its toggle is a `panel-left` icon
 * button at the toolbar's left end. The visual half is
 * `.osprey-panel-sidebar` in primitives.css; this module is the behaviour.
 *
 * Markup, inside the panel root (`[data-panel-root]`, the size container
 * named `panel`):
 *
 *   <button class="osprey-icon-btn" id="…-toggle"></button>   (in the toolbar)
 *   <div class="…body">                                       (one row)
 *     <aside class="osprey-panel-sidebar" id="…" data-open="false">…</aside>
 *     <div class="osprey-splitter" tabindex="0"></div>
 *     <main>…</main>
 *   </div>
 *
 * Stamp `data-open="false"` on the sidebar so the first paint never shows a
 * sidebar the boot then hides. The handle must follow the sidebar directly:
 * the stylesheet hides it with the sidebar.
 *
 * ---- One state, mirrored ----
 *
 * The shared splitter's collapse is the state; collapsed means gone (0 px,
 * `display: none`), never an icon rail. After every change the state is
 * mirrored to `[data-open]` on the sidebar and on the panel root (the
 * panel's own stylesheet hook), and to the toggle's `aria-expanded`, title
 * and accessible name ("Hide sidebar" / "Show sidebar", through
 * `writeTitle`). The toggle gets `aria-controls` when it has none.
 *
 * ---- What is stored ----
 *
 * The width and the open state live under `scopedStorageKey(storageKey)` as
 * `{size, collapsed}`, the splitter's record shape. Only the operator's own
 * gestures are stored: the toggle button, and on the handle a drag, an
 * arrow-key nudge, a double-click or Enter. Everything else is automatic and
 * never stored: `open()`, `close()` and `toggle()` from code, a change of
 * default (`setDefaultOpen`, `applyDefault`), and the narrow rule. This
 * generalises the Knowledge panel's rule (okf_panel/static/js/contents.js).
 *
 * `defaultOpen` is the panel's choice for the current view: a boolean, or a
 * function the module evaluates again on `applyDefault()`. Where the default
 * is open, the operator's stored choice wins, and a toggle there is stored.
 * Where the default is closed (Knowledge's overview), the sidebar starts
 * closed whatever is stored, and a toggle there holds only until the next
 * automatic change. A width change is always stored.
 *
 * ---- Narrow tiles ----
 *
 * When the panel root's content box matches `narrowQuery` (container width,
 * read with a ResizeObserver: a tile is narrower than its window), the
 * sidebar is an overlay over the content: it carries `[data-overlay]`, a
 * scrim covers the content, and it starts closed. It closes on Escape and on
 * a press outside it (a press inside a popover does not count). Entering the
 * narrow width closes it and leaving it applies the default again; neither
 * is stored, and nor is a toggle on a narrow tile.
 *
 * @module panel-sidebar
 */

import { icon } from './icons.js';
import { initSplitter } from './splitter.js';
import { scopedStorageKey } from './storage-scope.js';
import { writeTitle } from './tooltip.js';

const HIDE_LABEL = 'Hide sidebar';
const SHOW_LABEL = 'Show sidebar';
const NARROW_QUERY_RE = /^\(\s*max-width\s*:\s*(\d+(?:\.\d+)?)px\s*\)$/;
const WIDTH_PROPERTY = '--osprey-panel-sidebar-width';

let idSequence = 0;

/**
 * @typedef {object} PanelSidebarOptions
 * @property {HTMLElement} sidebar the `.osprey-panel-sidebar` element
 * @property {HTMLElement | null} [handle] the `.osprey-splitter` after it
 * @property {HTMLElement | null} [toggle] the toolbar's `panel-left` button;
 *   it gets the icon when it has no element child
 * @property {string} storageKey the bare key; the module scopes it
 * @property {boolean | (() => boolean)} [defaultOpen] the panel's choice for
 *   the current view. Default: true
 * @property {number} [min] smallest width, px. Default: 180
 * @property {number} [max] largest width, px. Default: 480
 * @property {number} [width] width with nothing stored, px. Default: 260
 * @property {string} [narrowQuery] `(max-width: <n>px)`, matched against the
 *   panel root's content box. Default: `(max-width: 720px)`
 * @property {HTMLElement | null} [root] the panel root. Default: the nearest
 *   `[data-panel-root]` around the sidebar, else its parent
 */

/**
 * @typedef {object} PanelSidebarApi
 * @property {() => void} open Show the sidebar; not stored.
 * @property {() => void} close Hide the sidebar; not stored.
 * @property {() => void} toggle Show or hide it; not stored.
 * @property {() => boolean} isOpen Whether it shows.
 * @property {(next: boolean | (() => boolean)) => void} setDefaultOpen Replace
 *   the default and apply it; not stored.
 * @property {() => void} applyDefault Evaluate the default again and apply it
 *   (call it when the panel changes view); not stored.
 * @property {() => void} resetWidth Go back to the `width` option and forget
 *   the stored width (an operator's reset, from a panel menu entry).
 * @property {() => void} destroy Remove every listener, the scrim and the
 *   observer. Nothing is stored.
 */

/**
 * The px bound of a `(max-width: <n>px)` query.
 * @param {string} query
 * @returns {number}
 * @throws {Error} on any other query, so a typo fails at the call site
 */
function parseNarrowQuery(query) {
  const match = NARROW_QUERY_RE.exec(query.trim());
  if (!match)
    throw new Error(`panel-sidebar: narrowQuery must be "(max-width: <n>px)", got "${query}"`);
  return Number(match[1]);
}

/**
 * The stored record. `open` is null when the operator made no choice: a
 * record may hold a width only. Reads the splitter's `{size, collapsed}`
 * shape, a bare width and a bare "true"/"false" collapsed flag.
 * @param {string} key the scoped key
 * @returns {{size: number | null, open: boolean | null}}
 */
function readRecord(key) {
  let raw = null;
  try {
    raw = localStorage.getItem(key);
  } catch {
    /* storage blocked: no stored choice */
  }
  if (raw === null || raw === '') return { size: null, open: null };
  if (raw === 'true' || raw === 'false') return { size: null, open: raw === 'false' };
  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed === 'number')
      return { size: Number.isFinite(parsed) ? parsed : null, open: null };
    if (parsed && typeof parsed === 'object') {
      return {
        size: Number.isFinite(parsed.size) ? parsed.size : null,
        open: typeof parsed.collapsed === 'boolean' ? !parsed.collapsed : null,
      };
    }
  } catch {
    /* unparseable: no stored choice */
  }
  return { size: null, open: null };
}

/**
 * The content-box width of `el`, or 0 when it is not laid out.
 * @param {HTMLElement} el
 * @returns {number}
 */
function contentWidth(el) {
  const width = el.clientWidth;
  if (!width) return 0;
  const style = getComputedStyle(el);
  return width - (parseFloat(style.paddingLeft) || 0) - (parseFloat(style.paddingRight) || 0);
}

/**
 * Wire a panel sidebar.
 * @param {PanelSidebarOptions} options
 * @returns {PanelSidebarApi}
 */
export function initPanelSidebar({
  sidebar,
  handle = null,
  toggle = null,
  storageKey,
  defaultOpen = true,
  min = 180,
  max = 480,
  width = 260,
  narrowQuery = '(max-width: 720px)',
  root = null,
}) {
  const narrowBound = parseNarrowQuery(narrowQuery);
  const panelRoot =
    root ??
    /** @type {HTMLElement | null} */ (sidebar.closest('[data-panel-root]')) ??
    sidebar.parentElement ??
    sidebar;
  const key = scopedStorageKey(storageKey);
  const record = readRecord(key);
  let defaultRule = defaultOpen;
  let narrow = false;
  /** True while the module itself opens or closes the sidebar: nothing is stored. */
  let automatic = false;
  let lastOpen = false;

  const controller = new AbortController();
  const { signal } = controller;

  if (!sidebar.id) sidebar.id = `osprey-panel-sidebar-${++idSequence}`;
  sidebar.style.setProperty(WIDTH_PROPERTY, `${width}px`);

  // The overlay and its scrim are placed against the sidebar's row.
  const row = sidebar.parentElement;
  const rowPosition = row ? getComputedStyle(row).position : '';
  const rowPositioned = rowPosition !== '' && rowPosition !== 'static';
  if (row && !rowPositioned) row.style.position = 'relative';
  const scrim = document.createElement('div');
  scrim.className = 'osprey-panel-sidebar-scrim';
  scrim.hidden = true;
  // Before the sidebar, so the handle still follows the sidebar directly.
  sidebar.before(scrim);

  if (toggle) {
    if (!toggle.firstElementChild) toggle.append(icon('panel-left'));
    if (!toggle.hasAttribute('aria-controls')) toggle.setAttribute('aria-controls', sidebar.id);
  }

  const split = initSplitter({
    handle,
    pane: sidebar,
    storageKey: null,
    min,
    max,
    axis: 'x',
    anchor: 'start',
    sizing: 'box',
    collapsedSize: 0,
    initialSize: record.size ?? width,
    initialCollapsed: true,
    onCommit,
  });

  const resolveDefault = () => (typeof defaultRule === 'function' ? defaultRule() : defaultRule);

  /** A toggle is the operator's choice to keep where the default is open on a wide tile. */
  const storesChoice = () => !narrow && resolveDefault();

  /** @returns {boolean} whether the current view, on its own, wants the sidebar. */
  function wanted() {
    if (narrow || !resolveDefault()) return false;
    return record.open ?? true;
  }

  /** Write the record; a record with no width and no choice is removed. */
  function persist() {
    /** @type {{size?: number, collapsed?: boolean}} */
    const value = {};
    if (record.size !== null) value.size = record.size;
    if (record.open !== null) value.collapsed = !record.open;
    try {
      if (Object.keys(value).length) localStorage.setItem(key, JSON.stringify(value));
      else localStorage.removeItem(key);
    } catch {
      /* storage blocked: the choice holds for this page lifetime */
    }
  }

  /**
   * The splitter settled a change. One the operator made is stored: a drag
   * or a nudge as the width, a toggle as the open state where the default is
   * open on a wide tile.
   * @param {number} size
   * @param {boolean} collapsed
   */
  function onCommit(size, collapsed) {
    const open = !collapsed;
    const toggled = open !== lastOpen;
    lastOpen = open;
    reflect();
    if (automatic) return;
    if (!toggled) record.size = size;
    else if (storesChoice()) record.open = open;
    else return;
    persist();
  }

  /** Mirror the state onto the sidebar, the panel root, the scrim and the toggle. */
  function reflect() {
    const open = !split.isCollapsed();
    const state = open ? 'true' : 'false';
    sidebar.setAttribute('data-open', state);
    panelRoot.setAttribute('data-open', state);
    sidebar.toggleAttribute('data-overlay', narrow);
    scrim.hidden = !(narrow && open);
    if (!toggle) return;
    const label = open ? HIDE_LABEL : SHOW_LABEL;
    toggle.setAttribute('aria-expanded', state);
    toggle.setAttribute('aria-label', label);
    writeTitle(toggle, label);
  }

  /**
   * Show or hide without storing. Focus inside a sidebar that goes away moves
   * to the toggle, so it never stays on a hidden element.
   * @param {boolean} open
   */
  function apply(open) {
    const hadFocus = !open && sidebar.contains(document.activeElement);
    automatic = true;
    try {
      if (open && split.isCollapsed()) split.expand();
      else if (!open && !split.isCollapsed()) split.collapse();
    } finally {
      automatic = false;
    }
    reflect();
    if (hadFocus && toggle && !toggle.hasAttribute('disabled'))
      toggle.focus({ preventScroll: true });
  }

  /** @param {number} measured the root's content width; 0 = not laid out */
  function measure(measured) {
    if (!(measured > 0)) return;
    const next = measured <= narrowBound;
    if (next === narrow) return;
    narrow = next;
    apply(wanted());
  }

  const observer =
    typeof ResizeObserver === 'function'
      ? new ResizeObserver((entries) => {
          const entry = entries[entries.length - 1];
          if (entry) measure(entry.contentRect.width);
        })
      : null;
  if (observer) observer.observe(panelRoot);
  else window.addEventListener('resize', () => measure(contentWidth(panelRoot)), { signal });

  toggle?.addEventListener(
    'click',
    () => {
      // The operator's toggle: through the splitter, so onCommit stores it.
      split.toggleCollapse();
    },
    { signal }
  );

  document.addEventListener(
    'keydown',
    (event) => {
      if (event.key !== 'Escape' || !narrow || split.isCollapsed()) return;
      // Escape inside a popover closes that popover first.
      const active = document.activeElement;
      if (active instanceof Element && active.closest('.osprey-popover')) return;
      apply(false);
    },
    { signal }
  );

  document.addEventListener(
    'pointerdown',
    (event) => {
      if (!narrow || split.isCollapsed()) return;
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (sidebar.contains(target) || toggle?.contains(target)) return;
      if (target instanceof Element && target.closest('.osprey-popover')) return;
      apply(false);
    },
    { capture: true, signal }
  );

  narrow = (() => {
    const measured = contentWidth(panelRoot);
    return measured > 0 && measured <= narrowBound;
  })();
  apply(wanted());

  return {
    open: () => apply(true),
    close: () => apply(false),
    toggle: () => apply(split.isCollapsed()),
    isOpen: () => !split.isCollapsed(),
    setDefaultOpen(next) {
      defaultRule = next;
      apply(wanted());
    },
    applyDefault: () => apply(wanted()),
    resetWidth() {
      const open = !split.isCollapsed();
      automatic = true;
      try {
        split.applySize(width);
        if (!open) split.collapse();
      } finally {
        automatic = false;
      }
      record.size = null;
      persist();
      reflect();
    },
    destroy() {
      controller.abort();
      observer?.disconnect();
      split.destroy();
      scrim.remove();
      sidebar.removeAttribute('data-overlay');
      if (row && !rowPositioned) row.style.position = '';
    },
  };
}
