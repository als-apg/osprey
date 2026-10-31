// @ts-check
/* OSPREY Web Terminal — the sidebar shell
 *
 * The labelled sidebar (`nav#sidebar` in index.html) is mostly markup the
 * page ships and rows panel-rail.js renders. This module gives it the
 * behaviour that belongs to the shell rather than to any one row:
 *
 *   - ICONS. The page's own glyphs are empty `svg[data-icon]` placeholders,
 *     sized by `.osprey-ic` from the first paint; {@link hydrateIcons} fills
 *     them from icons.js, the one source of the chrome's paths.
 *   - WIDTH. The labels toggle and Cmd/Ctrl+\ (collapsible.js), stored under
 *     `scopedStorageKey('osprey-sidebar')` — the key sidebar-boot.js reads
 *     before first paint. The rows' tips come from the page's one tooltip
 *     layer (`data-tip`, `data-tip-detail`, tooltip.js; started in app.js).
 *   - DISCLOSURE. The Artifacts and Script runs rows open and close their
 *     lists (`aria-expanded` on the row, `hidden` on the list). Which lists are
 *     open is kept per user under `scopedStorageKey('osprey-sidebar-open')`, a
 *     JSON array of row ids, `["artifacts"]` by default. A click on the
 *     Artifacts row opens its list (the row's other half, activating the
 *     panel, is panel-rail.js's); a click on its chevron (`[data-disc]`)
 *     toggles the list only; ArrowRight / ArrowLeft open and close it from the
 *     keyboard. While the sidebar is collapsed the Artifacts row opens the
 *     same list as a flyout beside the sidebar (`#sidebar-artifacts-flyout`),
 *     and Script runs, which has no flyout, does nothing.
 *   - THE AGENT ROW. Its open/closed look comes from `html[data-agent]`
 *     (dock-workspace.js); this module keeps its tooltip and description in
 *     step and shows how many cards wait on the operator (the chat bus's
 *     `attention` count) on its badge while the pane is closed — the one time
 *     the operator cannot see them.
 *   - FADES. Each list, and the group that holds the lists (the one section
 *     of the sidebar that gives up height), fade at an edge that has more to
 *     scroll to (`.fade`, `.fade-top`), so a cut-off row reads as scrolled
 *     rather than clipped. A group made later is watched from the render
 *     that adds it.
 *
 * Changing the width closes every sidebar popover and the tooltip, since each
 * is placed against a sidebar edge that is about to move.
 */

import { icon } from '/design-system/js/icons.js';
import { initCollapsible } from '/design-system/js/collapsible.js';
import { closeAllPopovers, togglePopover } from '/design-system/js/popover.js';
import { hideTooltip } from '/design-system/js/tooltip.js';
import { isMacPlatform } from '/design-system/js/platform.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { onChat } from './chat-events.js';
import { SIDEBAR_RENDERED_EVENT } from './panel-rail.js';

/** The width's storage key, shared with sidebar-boot.js (pre-paint). */
export const SIDEBAR_STORAGE_KEY = 'osprey-sidebar';

/** The open-lists storage key (a JSON array of row ids). */
export const OPEN_LISTS_STORAGE_KEY = 'osprey-sidebar-open';

/** The lists open on a first visit. */
const DEFAULT_OPEN = Object.freeze(['artifacts']);

/**
 * The rows that disclose a list: the row's button and the list it controls.
 * @type {Readonly<Record<string, {toggle: string, kids: string}>>}
 */
const DISCLOSURES = Object.freeze({
  artifacts: { toggle: '#sidebar-artifacts-row > .osprey-nav-row', kids: 'sidebar-artifacts-kids' },
  runs: { toggle: '#sidebar-runs-toggle', kids: 'sidebar-runs-kids' },
});

/** The collapsed sidebar's Artifacts flyout. */
const FLYOUT_ID = 'sidebar-artifacts-flyout';

/** The Agent row (panel-rail.js's `terminal` entry, shipped by index.html). */
const AGENT_ROW_ID = 'sidebar-agent-row';

/**
 * Fill every empty `svg[data-icon]` placeholder under `root` with its icons.js
 * glyph, in place, so the element (and any reference to it) is kept.
 * @param {ParentNode} [root]
 * @returns {void}
 */
export function hydrateIcons(root = document) {
  for (const svg of root.querySelectorAll('svg[data-icon]')) {
    if (svg.childElementCount > 0) continue;
    const name = svg.getAttribute('data-icon') ?? '';
    const glyph = icon(name);
    for (const attr of ['viewBox', 'aria-hidden', 'focusable']) {
      svg.setAttribute(attr, glyph.getAttribute(attr) ?? '');
    }
    svg.replaceChildren(...glyph.childNodes);
  }
}

/**
 * Write each `[data-chord]` key hint under `root` for this platform: `⌘` on
 * macOS, `Ctrl+` elsewhere, then the key the attribute names.
 * @param {ParentNode} [root]
 * @returns {void}
 */
export function paintChords(root = document) {
  const prefix = isMacPlatform() ? '⌘' : 'Ctrl+';
  for (const el of root.querySelectorAll('[data-chord]')) {
    el.textContent = `${prefix}${/** @type {HTMLElement} */ (el).dataset.chord}`;
  }
}

/** @returns {boolean} whether the sidebar shows icons only */
function isCollapsed() {
  return document.documentElement.getAttribute('data-sidebar') === 'collapsed';
}

// ---- Open lists ----

/** @returns {Set<string>} the row ids whose lists are open, as stored */
function readOpenLists() {
  try {
    const raw = window.localStorage.getItem(scopedStorageKey(OPEN_LISTS_STORAGE_KEY));
    if (raw !== null) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        return new Set(
          parsed.filter((id) => typeof id === 'string' && Object.hasOwn(DISCLOSURES, id))
        );
      }
    }
  } catch {
    // Unreadable or unavailable storage: the defaults below.
  }
  return new Set(DEFAULT_OPEN);
}

/** @param {Set<string>} open */
function writeOpenLists(open) {
  try {
    window.localStorage.setItem(
      scopedStorageKey(OPEN_LISTS_STORAGE_KEY),
      JSON.stringify([...open])
    );
  } catch {
    // Storage can be unavailable (private mode, quota); the page keeps the state.
  }
}

/** @param {string} id @returns {{toggle: HTMLElement | null, kids: HTMLElement | null}} */
function disclosure(id) {
  const spec = DISCLOSURES[id];
  return {
    toggle: /** @type {HTMLElement | null} */ (document.querySelector(spec.toggle)),
    kids: document.getElementById(spec.kids),
  };
}

/**
 * Paint one row's list open or closed: `hidden` on the list, and on the row
 * `aria-expanded` with `aria-controls` naming the list (the collapsed
 * Artifacts row points both at its flyout while that is open; this puts them
 * back).
 * @param {string} id
 * @param {boolean} open
 */
function paintDisclosure(id, open) {
  const { toggle, kids } = disclosure(id);
  if (kids) kids.hidden = !open;
  if (toggle) {
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-controls', DISCLOSURES[id].kids);
  }
}

/** @type {Set<string>} */
let openLists = new Set(DEFAULT_OPEN);

/**
 * Open or close a row's list and remember it.
 * @param {string} id
 * @param {boolean} open
 */
function setListOpen(id, open) {
  if (open) openLists.add(id);
  else openLists.delete(id);
  paintDisclosure(id, open);
  writeOpenLists(openLists);
}

/** Paint every row's list from {@link openLists}. */
function paintAllDisclosures() {
  for (const id of Object.keys(DISCLOSURES)) paintDisclosure(id, openLists.has(id));
}

/**
 * Wire a disclosing row: `primary` runs on a click on the row itself, a click
 * on its chevron toggles the list only, and ArrowRight / ArrowLeft open and
 * close the list from the keyboard.
 * @param {string} id
 * @param {(event: MouseEvent) => void} primary
 */
function wireDisclosure(id, primary) {
  const { toggle } = disclosure(id);
  if (!toggle) return;
  toggle.addEventListener('click', primary);
  const disc = toggle.querySelector('[data-disc]');
  disc?.addEventListener('click', (event) => {
    // The chevron is the list's own control: the row's activation (and its
    // flyout) never ride along.
    event.stopPropagation();
    if (!isCollapsed()) setListOpen(id, !openLists.has(id));
  });
  toggle.addEventListener('keydown', (event) => {
    if (isCollapsed() || (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft')) return;
    event.preventDefault();
    setListOpen(id, event.key === 'ArrowRight');
  });
}

/** @param {HTMLElement} sidebar */
function initDisclosures(sidebar) {
  openLists = readOpenLists();
  paintAllDisclosures();

  wireDisclosure('artifacts', (event) => {
    const row = /** @type {HTMLElement} */ (event.currentTarget);
    if (isCollapsed()) {
      const flyout = document.getElementById(FLYOUT_ID);
      if (flyout) togglePopover(row, flyout, { place: 'beside-right', edge: sidebar });
      return;
    }
    setListOpen('artifacts', true);
  });
  wireDisclosure('runs', () => {
    if (!isCollapsed()) setListOpen('runs', !openLists.has('runs'));
  });
}

// ---- The Agent row ----

/** Cards waiting on the operator, as the chat bus last reported. */
let waiting = 0;

/** Paint the Agent row from `html[data-agent]` and the waiting count. */
function paintAgentRow() {
  const row = document.getElementById(AGENT_ROW_ID);
  if (!row) return;
  const closed = document.documentElement.getAttribute('data-agent') === 'closed';
  const badge = /** @type {HTMLElement | null} */ (row.querySelector('[data-agent-badge]'));
  const showBadge = closed && waiting > 0;
  if (badge) {
    badge.hidden = !showBadge;
    badge.textContent = showBadge ? String(waiting) : '';
  }
  if (!closed) {
    row.dataset.tip = 'Agent';
    row.removeAttribute('aria-description');
    return;
  }
  const state = showBadge ? `${waiting} waiting` : 'closed';
  row.dataset.tip = `Agent · ${state} — click to open`;
  row.setAttribute(
    'aria-description',
    showBadge
      ? `The Agent pane is closed and ${waiting} ${waiting === 1 ? 'card waits' : 'cards wait'} on you; activate to open it`
      : 'The Agent pane is closed; activate to open it'
  );
}

function initAgentRow() {
  paintAgentRow();
  document.addEventListener('osprey-agent-pane', paintAgentRow);
  onChat('attention', (detail) => {
    waiting = Number.isFinite(detail?.waiting) ? Math.max(0, detail.waiting) : 0;
    paintAgentRow();
  });
}

// ---- Fades ----

/**
 * Mark the edges of a scroller that have more to scroll to.
 * @param {HTMLElement} el
 * @param {boolean} [top] also mark the top edge
 */
function paintFade(el, top = false) {
  el.classList.toggle('fade', el.scrollHeight - el.scrollTop - el.clientHeight > 2);
  if (top) el.classList.toggle('fade-top', el.scrollTop > 2);
}

/**
 * Keep a scroller's fades current: on scroll, on a size change, and when its
 * rows change. A group watches its whole subtree: a row added inside one of
 * its lists, or a list opened, changes its scroll height without changing
 * its own children or size.
 * @param {HTMLElement} el
 * @param {boolean} top
 */
function watchFade(el, top) {
  const paint = () => paintFade(el, top);
  el.addEventListener('scroll', paint, { passive: true });
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(paint).observe(el);
  new MutationObserver(paint).observe(el, {
    childList: true,
    subtree: true,
    attributeFilter: ['hidden'],
  });
  paint();
}

/** The sidebar groups whose fades are watched. @type {WeakSet<Element>} */
const fadingGroups = new WeakSet();

/**
 * Watch the fades of every sidebar group not watched yet. Only the group
 * that holds the lists scrolls (terminal.css), and which group that is
 * changes when the operator moves the Artifacts row, so every group is
 * watched; a group that does not scroll never shows a fade.
 */
function watchGroupFades() {
  for (const group of document.querySelectorAll('#panel-rail > .sidebar-group')) {
    if (fadingGroups.has(group)) continue;
    fadingGroups.add(group);
    watchFade(/** @type {HTMLElement} */ (group), true);
  }
}

/** @param {HTMLElement} sidebar */
function initFades(sidebar) {
  watchGroupFades();
  const lists = new Set([
    ...sidebar.querySelectorAll('.sidebar-kids-list'),
    ...(document.getElementById(FLYOUT_ID)?.querySelectorAll('.sidebar-kids-list') ?? []),
  ]);
  for (const list of lists) watchFade(/** @type {HTMLElement} */ (list), false);
}

// ---- Boot ----

/**
 * Wire the sidebar. Call once, before the panel manager renders its rows.
 * No-op (returns null) on a page without `#sidebar`.
 * @returns {{isCollapsed: () => boolean} | null}
 */
export function initSidebar() {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return null;

  hydrateIcons(document);
  paintChords(document);
  const collapsible = initCollapsible({
    storageKey: SIDEBAR_STORAGE_KEY,
    toggle: document.getElementById('sidebar-toggle'),
  });
  initDisclosures(sidebar);
  initAgentRow();
  initFades(sidebar);

  document.addEventListener('osprey-collapsible-change', (event) => {
    if (/** @type {CustomEvent} */ (event).detail?.attribute !== 'data-sidebar') return;
    closeAllPopovers();
    hideTooltip();
    paintAllDisclosures();
  });
  // panel-rail.js re-labels the Agent row on every render, and a render may
  // add a group.
  document.addEventListener(SIDEBAR_RENDERED_EVENT, () => {
    paintAgentRow();
    watchGroupFades();
  });

  return { isCollapsed: collapsible.isCollapsed };
}
