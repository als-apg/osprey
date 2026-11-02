// @ts-check
/* OSPREY Web Terminal — the sidebar's panel rows (DOM renderer)
 *
 * The labelled sidebar lists the panels inside `#panel-rail`, sorted into the
 * operator's groups (sidebar-groups.js): `Workspace` and `Tools` to begin
 * with, then renamed, reordered or joined by groups of the operator's own.
 * The Agent row is always the first row of the first group. Every other row
 * sits in the group the model places it in, and within a group the rows keep
 * the server's rail order, so a group filters the rail and never reorders
 * it. `+ Add panel` is always the last row of the last group. This module
 * builds and places those rows and exposes small imperative mutators — set
 * active, enable, attention, append, remove — so `panel-manager.js`'s state
 * machine can drive the sidebar without owning any DOM specifics.
 *
 * The rows deliberately carry no per-poll readout — coarse reachability only.
 * Backend liveness is reported in one place — the SYSTEM panel's
 * `web_panels` health category — so the sidebar stays a navigation surface
 * rather than a status board. Health reaches a row as two coarse states: a row
 * whose backend has never answered is `.disabled` (dimmed and inert), and one
 * that answered before but has stopped is `.unreachable` (dimmed, still
 * clickable, its tooltip saying since when). The one status a row names is a
 * sidecar that failed to start: its row is `.failed` and carries the server's
 * sentence as its tooltip, because a click on it is the retry.
 *
 * The rows are a curated MEMBERSHIP list: a panel has a row iff it is a rail
 * member. Which member holds the workspace tile is per-client layout state,
 * reflected only by `.active` + `aria-current="page"` — never by dimming.
 *
 * PAGE-OWNED ROWS. Two rows ship ready-made in index.html because other
 * modules hang state on them: the Agent row (`#sidebar-agent-row`, the
 * `terminal` entry, whose dot and badge sidebar.js keeps) and the Artifacts
 * row (`#sidebar-artifacts-row`, whose lists sidebar-artifacts.js renders).
 * Beside them sit two page-owned nodes that are not entries at all: the
 * Script runs row (`#sidebar-runs-row`, which follows the Artifacts row in
 * and out of view, and into whichever group the Artifacts row moves to,
 * because its records come from the same server) and `#panel-add-btn`. A
 * render MOVES these nodes into place and never rebuilds them, so their
 * listeners, their lists and their live state survive every render. A
 * page-owned entry whose panel is not a member stays in the document,
 * `hidden`, and counts as absent for every query below. A rail element
 * without the page's skeleton (a test fixture) gets its groups built here and
 * plain rows for every entry.
 *
 * GROUPS. The groups follow the model, in its order: each one is a
 * `.sidebar-group[data-group]` led by the heading that sidebar-groups.js
 * paints. A group with no row in view shows an empty line instead, so it
 * stays a place that a row can be moved or dropped into. A change to the
 * model moves the rows into their new groups at once (moved, never rebuilt,
 * so their live state goes with them).
 *
 * This module holds NO panel state and issues NO fetches or POSTs. The caller
 * passes closures for the interactions (activate, context menu, drag, reorder).
 * After every full render and every change of the groups it dispatches
 * `osprey-sidebar-rendered` on `document`, and again whenever a page-owned
 * row comes into or out of view, so the modules that fill those rows
 * re-render against the placed nodes.
 *
 * DOM contract (stable — the browser suite, the tour and the dock select on it):
 *
 *   <div id="panel-rail" role="list">
 *     <div class="sidebar-group" data-group="workspace" role="listitem"
 *          aria-labelledby="panel-rail-workspace-label">
 *       <div class="osprey-section-label sidebar-group-head">
 *         <span class="sidebar-group-name" id="panel-rail-workspace-label">Workspace</span>
 *         <button class="osprey-mini" type="button" data-group-menu …>…</button>
 *       </div>
 *       <div class="osprey-list-empty sidebar-group-empty" data-group-empty hidden>…</div>
 *       <button class="osprey-nav-row panel-rail-button disabled" data-panel-id="events"
 *               type="button" title="Events" aria-label="Events" data-tip="Events"
 *               data-tip-detail="Event dispatcher dashboard"
 *               draggable="true">                      (title gains " · hint" when panel.hint is given)
 *         <svg class="osprey-ic osprey-nav-icon">…</svg>
 *         <span class="osprey-nav-label">Events</span>
 *       </button>
 *     </div>
 *     <div class="sidebar-group" data-group="tools" role="listitem" …>…</div>
 *   </div>
 *
 * A row's tooltip detail (`data-tip-detail`) is the panel's sentence from
 * panel-catalog.js; a panel without one has none.
 *
 * Beyond activation, a row's verbs (open beside, pop out, move to another
 * group, remove) live in the caller's right-click context menu — the rail
 * only forwards the contextmenu event through `onContextMenu` and suppresses
 * the browser menu when the caller reports it handled. A caller-supplied
 * `panel.hint` is appended to the tooltip so the menu is advertised.
 *
 * That menu is reachable from the keyboard, and explicitly so: a focused row
 * handles the `ContextMenu` key and Shift+F10 itself and asks the caller to
 * open the menu centred on the row's own rect. macOS fires no `contextmenu`
 * event from the keyboard at all, so leaving this to the platform would strand
 * those operators; on the platforms that DO fire one (Windows menu key,
 * Firefox Shift+F10) it is that keydown's default action, and suppressing the
 * default is what stops a single keypress opening two menus.
 *
 * DRAG. A row dragged and dropped among the rows of a group asks the caller
 * to put it directly above the row it was dropped on (`onReorder`); dropped
 * into another group, it moves to that group first (sidebar-groups.js
 * moveToGroup) and then takes the same position there. The Agent row never
 * moves.
 *
 * State classes on a row: `.active` (surfaced panel), `.disabled` (backend
 * never answered), `.unreachable` (backend stopped answering), `.failed` (a
 * sidecar that failed to start; dimmed but clickable), `.agent-attention`
 * (the agent touched the panel and the operator has not looked yet; drawn as
 * an `.osprey-nav-badge` dot inside the row), `.popped-out` (the panel runs
 * in its own window; drawn as a trailing `maximize` glyph). While a badge
 * time, an unreachable notice or a status message owns the tooltip, the row
 * carries a transient `data-title-base` holding the tooltip it borrowed;
 * clearing all of them restores it and removes the attribute.
 */

import { icon } from '/design-system/js/icons.js';
import { TERMINAL_RAIL_ID, describePanel } from './panel-catalog.js';
import { BUTTON_SELECTOR, getEntry, parked, rowOf, writeRowTips } from './rail-row-state.js';
import {
  groupOf,
  loadGroups,
  moveToGroup,
  onGroupsChange,
  paintGroupHeading,
} from './sidebar-groups.js';

// ---- Types ----

/**
 * The minimal panel descriptor the sidebar renders — a structural subset of
 * the `Panel` records `panel-manager.js` holds and of the `/api/panels`
 * payload. Only id + label are needed to draw a row; health and active state
 * are applied by the mutators.
 * @typedef {object} RailPanel
 * @property {string} id
 * @property {string} label - as the sidebar shows it (title case)
 * @property {string} [icon] - icons.js name of the row's glyph; default
 *   {@link DEFAULT_ICON}
 * @property {string} [hint] - tooltip suffix advertising the context menu
 *   (rendered as "Label · hint"; aria-label stays the bare label)
 */

/**
 * Interaction closures injected by the caller so the sidebar stays a dumb
 * view. Every callback is optional: omit `onDragStart` to render
 * non-draggable rows. `onDragStart` may return false to cancel the drag for
 * that row, and `onContextMenu` false to let the browser's native menu
 * through (the caller's policy — e.g. a disabled row); this module stays
 * policy-free.
 * @typedef {object} RailOptions
 * @property {(id: string) => void} [onActivate]   - a row was clicked
 * @property {(id: string, e: MouseEvent) => boolean} [onContextMenu]
 *   - a row was right-clicked, or asked for its menu from the keyboard
 *     (`ContextMenu` / Shift+F10, where `e` is a synthetic contextmenu event
 *     positioned at the row's rect); return true if a menu was opened
 *     (suppresses the browser menu), false to decline and let the native menu
 *     show
 * @property {(id: string, dataTransfer: DataTransfer | null) => boolean} [onDragStart]
 *   - a drag left a row; return false to cancel it
 * @property {(id: string) => void} [onDragEnd]    - the row's drag gesture ended (any outcome)
 * @property {(id: string, beforeId: string | null) => void} [onReorder]
 *   - a row was dropped among the rows of a group: move `id` directly above
 *     `beforeId`, or to the foot of that group when null. A drop into another
 *     group has moved the row into it (sidebar-groups.js) before this runs.
 *     Never called for a drop that would leave the row where it is. Omit to
 *     keep the order fixed; drags into the workspace are unaffected either
 *     way.
 * @property {(id: string) => boolean} [reorderable]
 *   - whether a row takes part in reordering and moving by drag (default:
 *     every one). The Agent row never does: it is pinned first.
 */

// The row lookup and the mutators of a row's live state live in
// rail-row-state.js; the shell imports them from here.
export {
  clockTime,
  getEntry,
  setActive,
  setEntryAttention,
  setEntryEnabled,
  setEntryPoppedOut,
  setEntryReachable,
  setEntryStatus,
} from './rail-row-state.js';

/** The event dispatched on `document` after a render. */
export const SIDEBAR_RENDERED_EVENT = 'osprey-sidebar-rendered';

/** The glyph of a row whose descriptor names none. */
const DEFAULT_ICON = 'workspace';

/** Page-owned rows, by the entry id they hold. */
const PAGE_ROWS = /** @type {Readonly<Record<string, string>>} */ ({
  terminal: 'sidebar-agent-row',
  artifacts: 'sidebar-artifacts-row',
});

/** The entry the Script runs row follows. */
const ARTIFACTS_ID = 'artifacts';

/** The Script runs row: page-owned, shown exactly while Artifacts is, and
 *  always directly after it. */
const RUNS_ROW_ID = 'sidebar-runs-row';

/** The `+ Add panel` row: page-owned, always the last row of the last group. */
const ADD_BUTTON_ID = 'panel-add-btn';

/** Marks a group's empty line. */
const EMPTY_ATTR = 'data-group-empty';

/** What an empty group says, so it still reads as a place a row can go. */
const EMPTY_TEXT = "No panels — move one here from a row's menu";

/** The row nodes of a group (entries, and the page-owned row wrappers). */
const ROW_NODE_SELECTOR = `${BUTTON_SELECTOR}, .sidebar-row`;

/** The options each rail was last fully rendered with, read by its reorder
 *  listeners. @type {WeakMap<HTMLElement, RailOptions>} */
const railOptionsOf = new WeakMap();

/** The options each row was last rendered with, read by its own listeners, so
 *  a page-owned row adopted by many renders always answers to the closures of
 *  its latest one. @type {WeakMap<HTMLElement, RailOptions>} */
const rowOptionsOf = new WeakMap();

/** Page-owned buttons already wired, so a re-render never doubles a listener.
 *  @type {WeakSet<HTMLElement>} */
const wiredButtons = new WeakSet();

/**
 * The rail order each rail's rows follow within their groups: the panel ids
 * of its last full render, kept in step with every addition, removal and
 * placement since, as the server's rail order is. The Agent row is not in
 * it; it is pinned first.
 * @type {WeakMap<HTMLElement, string[]>}
 */
const railOrderOf = new WeakMap();

// ---- Structure ----

/**
 * The rail's group elements, one per group of the model and in its order:
 * each adopted where the page shipped it, built where it did not, and given
 * its heading (sidebar-groups.js). Elements of groups the model no longer
 * has are left for {@link arrange} to empty and remove.
 * @param {HTMLElement} railEl
 * @returns {Map<string, HTMLElement>} group id → element, in model order
 */
function syncGroups(railEl) {
  /** @type {Map<string, HTMLElement>} */
  const found = new Map();
  /** @type {Element | null} */
  let previous = null;
  for (const group of loadGroups().groups) {
    let el = /** @type {HTMLElement | null} */ (
      railEl.querySelector(`:scope > .sidebar-group[data-group="${CSS.escape(group.id)}"]`)
    );
    if (!el) {
      el = document.createElement('div');
      el.className = 'sidebar-group';
      el.setAttribute('data-group', group.id);
      el.setAttribute('role', 'listitem');
    }
    paintGroupHeading(el, group, `${railEl.id || 'panel-rail'}-${group.id}-label`);
    /** @type {Element | null} */
    const slot = previous ? previous.nextElementSibling : railEl.firstElementChild;
    if (slot !== el) railEl.insertBefore(el, slot);
    found.set(group.id, el);
    previous = el;
  }
  return found;
}

/** @param {HTMLElement} group @returns {HTMLElement} the group's heading */
function headingOf(group) {
  return /** @type {HTMLElement} */ (group.querySelector(':scope > .osprey-section-label'));
}

/**
 * A group's empty line, made on first use. It is shown while the group has
 * no row in view (see {@link syncEmptyLines}).
 * @param {HTMLElement} group
 * @returns {HTMLElement}
 */
function emptyLineOf(group) {
  let line = /** @type {HTMLElement | null} */ (group.querySelector(`:scope > [${EMPTY_ATTR}]`));
  if (!line) {
    line = document.createElement('div');
    line.className = 'osprey-list-empty sidebar-group-empty';
    line.setAttribute(EMPTY_ATTR, '');
    line.textContent = EMPTY_TEXT;
    line.hidden = true;
  }
  return line;
}

/**
 * The page-owned row holding entry `id`, or null.
 * @param {HTMLElement} railEl
 * @param {string} id
 * @returns {HTMLElement | null}
 */
function pageRow(railEl, id) {
  const rowId = Object.hasOwn(PAGE_ROWS, id) ? PAGE_ROWS[id] : null;
  return rowId ? /** @type {HTMLElement | null} */ (railEl.querySelector(`#${rowId}`)) : null;
}

/**
 * Show or hide the Script runs row with the Artifacts row: its records come
 * from the same server, so it has nothing to list where Artifacts is absent.
 * @param {HTMLElement} railEl
 */
function syncRunsRow(railEl) {
  const runs = /** @type {HTMLElement | null} */ (railEl.querySelector(`#${RUNS_ROW_ID}`));
  if (runs) runs.hidden = getEntry(railEl, ARTIFACTS_ID) === null;
}

/**
 * Whether `el` is a row node (an entry, or a page-owned row wrapper) in view.
 * @param {Element} el
 * @returns {boolean}
 */
function isRowInView(el) {
  return el.matches(ROW_NODE_SELECTOR) && !(/** @type {HTMLElement} */ (el).hidden);
}

/**
 * Show each group's empty line exactly while the group has no row in view.
 * @param {HTMLElement} railEl
 */
function syncEmptyLines(railEl) {
  for (const group of railEl.querySelectorAll(':scope > .sidebar-group')) {
    const line = /** @type {HTMLElement | null} */ (
      group.querySelector(`:scope > [${EMPTY_ATTR}]`)
    );
    if (line) line.hidden = [...group.children].some(isRowInView);
  }
}

/**
 * The rail order of a rail, made empty on first use.
 * @param {HTMLElement} railEl
 * @returns {string[]}
 */
function orderOf(railEl) {
  let order = railOrderOf.get(railEl);
  if (!order) {
    order = [];
    railOrderOf.set(railEl, order);
  }
  return order;
}

/**
 * Make `kids` the first children of `parent`, in order, moving only the nodes
 * that are out of place. Children after them stay: a row bound for a later
 * group moves when that group's turn comes.
 * @param {HTMLElement} parent
 * @param {HTMLElement[]} kids
 */
function reconcile(parent, kids) {
  kids.forEach((kid, i) => {
    const at = parent.children[i];
    if (at !== kid) parent.insertBefore(kid, at ?? null);
  });
}

/**
 * Put every group and row where the model and the rail order say. Each group
 * holds its heading and its empty line, then its rows in rail order: the
 * Agent row first in the first group, and the Script runs row directly after
 * the Artifacts row in whichever group that is. `+ Add panel` closes the last
 * group. Nodes are moved, never rebuilt, and only where they are out of
 * place, so every row keeps its live state and a node that stays put keeps
 * its focus.
 * @param {HTMLElement} railEl
 */
function arrange(railEl) {
  const groups = syncGroups(railEl);
  const ids = [...groups.keys()];
  const order = orderOf(railEl);
  const rank = (/** @type {string} */ id) => {
    const at = order.indexOf(id);
    return at < 0 ? order.length : at;
  };

  /** @type {Map<string, {id: string, node: HTMLElement}[]>} */
  const rows = new Map();
  for (const id of ids) rows.set(id, []);
  /** @type {HTMLElement | null} */
  let agent = null;
  for (const entry of railEl.querySelectorAll(BUTTON_SELECTOR)) {
    const id = entry.getAttribute('data-panel-id') ?? '';
    const node = rowOf(/** @type {HTMLElement} */ (entry));
    if (id === TERMINAL_RAIL_ID) agent = node;
    else rows.get(groupOf(id))?.push({ id, node });
  }
  const runs = /** @type {HTMLElement | null} */ (railEl.querySelector(`#${RUNS_ROW_ID}`));
  const add = /** @type {HTMLElement | null} */ (railEl.querySelector(`#${ADD_BUTTON_ID}`));

  ids.forEach((groupId, i) => {
    const group = /** @type {HTMLElement} */ (groups.get(groupId));
    const kids = [headingOf(group), emptyLineOf(group)];
    if (i === 0 && agent) kids.push(agent);
    const members = rows.get(groupId) ?? [];
    for (const { id, node } of members.sort((a, b) => rank(a.id) - rank(b.id))) {
      kids.push(node);
      if (id === ARTIFACTS_ID && runs) kids.push(runs);
    }
    if (i === ids.length - 1 && add) kids.push(add);
    reconcile(group, kids);
  });
  // Every row has moved into a group of the model by now, so the element of a
  // deleted group holds nothing that lives on.
  for (const el of railEl.querySelectorAll(':scope > .sidebar-group')) {
    if (!groups.has(el.getAttribute('data-group') ?? '')) el.remove();
  }
  syncRunsRow(railEl);
  syncEmptyLines(railEl);
}

/** Tell the row-filling modules the sidebar's rows have been placed. */
function announceRender() {
  document.dispatchEvent(new CustomEvent(SIDEBAR_RENDERED_EVENT));
}

// ---- Rendering ----

/**
 * Give a row its identity: label, accessible name, tooltips. The tooltip's
 * explanatory line is the panel's sentence (panel-catalog.js), written so a
 * popped-out mark or a borrowed title keeps showing (rail-row-state.js).
 * @param {HTMLElement} btn
 * @param {RailPanel} panel
 */
function applyIdentity(btn, panel) {
  const label = btn.querySelector('.osprey-nav-label');
  if (label) label.textContent = panel.label;
  btn.setAttribute('aria-label', panel.label);
  btn.dataset.tip = panel.label;
  const title = panel.hint ? `${panel.label} · ${panel.hint}` : panel.label;
  writeRowTips(btn, title, describePanel(panel.id));
}

/**
 * Build one row button for a panel. Children are assembled via DOM APIs
 * (never innerHTML) because `panel.label` originates from server JSON / SSE.
 *
 * Rows start `.disabled`; the caller clears it via {@link setEntryEnabled}
 * once the panel's backend is healthy.
 * @param {HTMLElement} railEl
 * @param {RailPanel} panel
 * @param {RailOptions} options
 * @returns {HTMLButtonElement}
 */
function buildRow(railEl, panel, options) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'osprey-nav-row panel-rail-button disabled';
  btn.setAttribute('data-panel-id', panel.id);

  const glyph = icon(panel.icon ?? DEFAULT_ICON);
  glyph.classList.add('osprey-nav-icon');
  const label = document.createElement('span');
  label.className = 'osprey-nav-label';
  btn.append(glyph, label);

  applyIdentity(btn, panel);
  wireButton(railEl, btn, panel.id, options);
  return btn;
}

/**
 * Bind a row to `rowOptions` and attach its listeners, once. The listeners
 * read the options of the row's latest render, so a page-owned row adopted by
 * many renders answers to the current closures and is never wired twice.
 * @param {HTMLElement} railEl
 * @param {HTMLElement} btn
 * @param {string} id
 * @param {RailOptions} rowOptions
 */
function wireButton(railEl, btn, id, rowOptions) {
  rowOptionsOf.set(btn, rowOptions);
  if (wiredButtons.has(btn)) return;
  wiredButtons.add(btn);
  const options = () => rowOptionsOf.get(btn) ?? {};

  btn.addEventListener('click', () => options().onActivate?.(id));

  // Right-click → the caller's context menu. The caller decides per-row
  // whether a menu exists (returns true); declining leaves the event alone so
  // the browser menu shows.
  btn.addEventListener('contextmenu', (e) => {
    const onContextMenu = options().onContextMenu;
    if (onContextMenu?.(id, e)) e.preventDefault();
  });

  // The keyboard route to the same menu, opened by this module rather than by
  // the platform: macOS synthesises no `contextmenu` event from the menu key,
  // so a native-only path would have no keyboard route at all there. The
  // synthetic event is handed straight to the caller and never dispatched, so
  // the listener above cannot also fire for one keypress.
  btn.addEventListener('keydown', (e) => {
    if (e.key !== 'ContextMenu' && !(e.key === 'F10' && e.shiftKey)) return;
    const onContextMenu = options().onContextMenu;
    if (!onContextMenu) return;
    const rect = btn.getBoundingClientRect();
    const synthetic = new MouseEvent('contextmenu', {
      bubbles: true,
      cancelable: true,
      clientX: rect.left + rect.width / 2,
      clientY: rect.top + rect.height / 2,
    });
    // Preventing the default is what keeps Windows/Firefox — where the native
    // menu IS this keydown's default action — from stacking a second menu on
    // the one just opened; a declined row keeps that default.
    if (onContextMenu(id, synthetic)) e.preventDefault();
  });

  // Drag source: dragging a row into the workspace opens the panel as a NEW
  // tile at the drop position (the precise half of the open-beside verb). The
  // caller's onDragStart stamps the dataTransfer payload and may cancel
  // (return false) — e.g. for the Agent row, or with no dock to drop into.
  btn.setAttribute('draggable', 'true');
  btn.addEventListener('dragstart', (e) => {
    const onDragStart = options().onDragStart;
    if (!onDragStart || !onDragStart(id, e.dataTransfer)) {
      e.preventDefault();
      return;
    }
    btn.classList.add('dragging');
  });
  btn.addEventListener('dragend', () => {
    btn.classList.remove('dragging');
    clearDropMarks(railEl);
    options().onDragEnd?.(id);
  });
}

/**
 * The node an entry occupies in its group: a page-owned row (adopted, wired
 * and shown) or a freshly built row.
 * @param {HTMLElement} railEl
 * @param {RailPanel} panel
 * @param {RailOptions} options
 * @returns {HTMLElement}
 */
function entryNode(railEl, panel, options) {
  const row = pageRow(railEl, panel.id);
  if (!row) return buildRow(railEl, panel, options);
  const btn = /** @type {HTMLElement} */ (
    row.matches(BUTTON_SELECTOR) ? row : row.querySelector(BUTTON_SELECTOR)
  );
  applyIdentity(btn, panel);
  wireButton(railEl, btn, panel.id, options);
  row.hidden = false;
  return row;
}

/**
 * Render every row into `railEl`: the groups of the model in its order, the
 * Agent row first, every other entry in its group in `panels` order (the
 * server's rail order), the Script runs row after Artifacts, and `+ Add
 * panel` last. Page-owned rows are moved into place, never rebuilt; one whose
 * panel is not in `panels` stays in its group, hidden.
 *
 * This is the destructive full render: built rows are made anew. Use
 * {@link addEntry} / {@link removeEntry} for runtime membership changes so
 * live rows keep their active / health / enabled state.
 * @param {HTMLElement} railEl
 * @param {RailPanel[]} panels
 * @param {RailOptions} [options]
 */
export function createRail(railEl, panels, options = {}) {
  railEl.setAttribute('role', 'list');
  railOptionsOf.set(railEl, options);
  wireRail(railEl);
  const groups = syncGroups(railEl);

  // Built rows go; page-owned rows are parked until their panel shows up in
  // `panels`. Parking instead of removing is what keeps them (and everything
  // other modules hang on them) alive.
  for (const entry of [...railEl.querySelectorAll(BUTTON_SELECTOR)]) {
    const id = entry.getAttribute('data-panel-id') ?? '';
    if (pageRow(railEl, id)) rowOf(/** @type {HTMLElement} */ (entry)).hidden = true;
    else entry.remove();
  }
  for (const panel of panels) {
    const node = entryNode(railEl, panel, options);
    if (!railEl.contains(node)) groups.get(groupOf(panel.id))?.append(node);
  }
  const ids = panels.map((panel) => panel.id);
  railOrderOf.set(
    railEl,
    ids.filter((id) => id !== TERMINAL_RAIL_ID)
  );
  arrange(railEl);
  announceRender();
}

/**
 * Append a single entry without disturbing existing ones — the non-destructive
 * path for membership additions (agent add_panel_to_rail, "+" menu pick,
 * runtime registration). The row joins the foot of its group, as a new member
 * joins the end of the server's rail order; the Agent row takes its place at
 * the head of the first group.
 *
 * Idempotent by id: an entry that already exists is returned unchanged.
 * @param {HTMLElement} railEl
 * @param {RailPanel} panel
 * @param {RailOptions} [options]
 * @returns {HTMLElement}
 */
export function addEntry(railEl, panel, options = {}) {
  const existing = getEntry(railEl, panel.id);
  if (existing) return existing;
  wireRail(railEl);
  const own = pageRow(railEl, panel.id);
  const node = entryNode(railEl, panel, options);
  if (!railEl.contains(node)) syncGroups(railEl).get(groupOf(panel.id))?.append(node);
  if (panel.id !== TERMINAL_RAIL_ID) {
    const order = orderOf(railEl).filter((id) => id !== panel.id);
    railOrderOf.set(railEl, [...order, panel.id]);
  }
  arrange(railEl);
  if (own) announceRender();
  return /** @type {HTMLElement} */ (getEntry(railEl, panel.id));
}

/**
 * Remove a panel's entry — the membership-removal twin of {@link addEntry}
 * (agent remove_panel_from_rail, "Remove from rail"). A built row leaves the
 * DOM entirely and a later addEntry builds a fresh one; a page-owned row is
 * parked, hidden, in place. No-op for an unknown id.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 */
export function removeEntry(railEl, panelId) {
  const entry = getEntry(railEl, panelId);
  if (!entry) return;
  // A detached (or hidden) drag source can never fire dragend (HTML5 delivers
  // it to the source element only), so a mid-drag removal would strand the
  // caller's drag-end cleanup — the iframe pointer shields raised in
  // onDragStart. End the gesture through the row's own dragend listener first.
  if (entry.classList.contains('dragging')) {
    entry.dispatchEvent(new Event('dragend'));
  }
  railOrderOf.set(
    railEl,
    orderOf(railEl).filter((id) => id !== panelId)
  );
  if (pageRow(railEl, panelId)) {
    rowOf(entry).hidden = true;
    syncRunsRow(railEl);
    syncEmptyLines(railEl);
    announceRender();
    return;
  }
  entry.remove();
  syncEmptyLines(railEl);
}

/**
 * The ids of the rows in view, top to bottom — the order the operator sees.
 * @param {HTMLElement} railEl
 * @returns {string[]}
 */
export function entryIds(railEl) {
  return [...railEl.querySelectorAll(BUTTON_SELECTOR)]
    .filter((el) => !parked(el))
    .map((el) => el.getAttribute('data-panel-id') ?? '');
}

/**
 * Move a row in the rail order so it comes directly before `beforeId`, or
 * last when `beforeId` is null or has no row in view — the client half of
 * the server's rail-order move. The row moves within its own group: it ends
 * up directly above `beforeId`'s row when both are in one group, and among
 * its own group's rows by the rail order otherwise. The Agent row keeps its
 * place. Moving keeps the node, so its live state (active, health, badges)
 * travels with it. No-op for an unknown `panelId`.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @param {string | null} beforeId
 */
export function placeEntry(railEl, panelId, beforeId) {
  if (panelId === TERMINAL_RAIL_ID || !getEntry(railEl, panelId)) return;
  const order = orderOf(railEl).filter((id) => id !== panelId);
  const at = beforeId !== null && getEntry(railEl, beforeId) ? order.indexOf(beforeId) : -1;
  order.splice(at < 0 ? order.length : at, 0, panelId);
  railOrderOf.set(railEl, order);
  arrange(railEl);
}

// ---- Reordering and moving by drag ----

const DROP_BEFORE = 'drop-before';
const DROP_AFTER = 'drop-after';

/** Rails whose own listeners are attached. @type {WeakSet<HTMLElement>} */
const wiredRails = new WeakSet();

/**
 * Attach a rail's own listeners, once per rail element: the dragover /
 * dragleave / drop listeners, which read the options of the rail's last
 * render, and the subscription that moves the rows whenever the groups
 * change. A rail that has left the document drops its subscription at the
 * next change.
 * @param {HTMLElement} railEl
 */
function wireRail(railEl) {
  if (wiredRails.has(railEl)) return;
  wiredRails.add(railEl);
  const stop = onGroupsChange(() => {
    if (!railEl.isConnected) {
      stop();
      return;
    }
    arrange(railEl);
    announceRender();
  });
  railEl.addEventListener('dragover', (e) => {
    const point = insertionPoint(railEl, e);
    if (!point) return;
    e.preventDefault(); // accept the drop here
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'move';
    clearDropMarks(railEl);
    if (point.noop) return;
    if (point.before) point.before.classList.add(DROP_BEFORE);
    else point.after?.classList.add(DROP_AFTER);
  });
  railEl.addEventListener('dragleave', (e) => {
    const to = /** @type {Node | null} */ (e.relatedTarget);
    if (!to || !railEl.contains(to)) clearDropMarks(railEl);
  });
  railEl.addEventListener('drop', (e) => {
    const point = insertionPoint(railEl, e);
    if (!point) return;
    e.preventDefault();
    clearDropMarks(railEl);
    if (point.noop) return;
    // The row joins the group it was dropped into first, so the position
    // below is a position among that group's rows.
    if (groupOf(point.id) !== point.group) moveToGroup(point.id, point.group);
    const beforeId = point.before?.getAttribute('data-panel-id') ?? null;
    railOptionsOf.get(railEl)?.onReorder?.(point.id, beforeId);
  });
}

/**
 * Where a drag over the rail would put the dragged row, or null when the drag
 * is not a reorder: it did not start on a row, the row is fixed (the Agent
 * row, or one the caller marks), or the caller takes no reorders. The target
 * group is the last one whose heading starts above the pointer. Within it,
 * the target is the first movable row whose vertical midpoint lies below the
 * pointer; none means the foot of the group, which the line under its last
 * row (or under its empty line) shows.
 * @param {HTMLElement} railEl
 * @param {DragEvent} e
 * @returns {{ id: string, group: string, before: HTMLElement | null,
 *   after: HTMLElement | null, noop: boolean } | null}
 */
function insertionPoint(railEl, e) {
  const options = railOptionsOf.get(railEl);
  const dragging = /** @type {HTMLElement | null} */ (
    railEl.querySelector(`${BUTTON_SELECTOR}.dragging`)
  );
  if (!options?.onReorder || !dragging) return null;
  const movable = (/** @type {Element} */ el) => {
    const id = el.getAttribute('data-panel-id') ?? '';
    return id !== TERMINAL_RAIL_ID && (!options.reorderable || options.reorderable(id));
  };
  if (!movable(dragging)) return null;

  const groups = /** @type {HTMLElement[]} */ (
    Array.from(railEl.querySelectorAll(':scope > .sidebar-group'))
  );
  const above = groups.filter((g) => headingOf(g).getBoundingClientRect().top <= e.clientY);
  const target = above.at(-1) ?? groups[0];
  if (!target) return null;
  const rows = /** @type {HTMLElement[]} */ (
    [...target.querySelectorAll(BUTTON_SELECTOR)].filter((el) => !parked(el) && movable(el))
  );
  const others = rows.filter((el) => el !== dragging);
  const before =
    others.find((el) => {
      const r = el.getBoundingClientRect();
      return r.top + r.height / 2 > e.clientY;
    }) ?? null;
  const home = target.contains(dragging);
  const next = rows[rows.indexOf(dragging) + 1] ?? null;
  return {
    id: dragging.getAttribute('data-panel-id') ?? '',
    group: target.getAttribute('data-group') ?? '',
    before,
    after: before ? null : footOf(target),
    noop: home && before === next,
  };
}

/**
 * The node a drop at the foot of `group` lands after: its last row in view,
 * or its empty line when it has none.
 * @param {HTMLElement} group
 * @returns {HTMLElement | null}
 */
function footOf(group) {
  const rows = [...group.children].filter(isRowInView);
  return /** @type {HTMLElement | null} */ (
    rows.at(-1) ?? group.querySelector(`:scope > [${EMPTY_ATTR}]`)
  );
}

/** @param {HTMLElement} railEl */
function clearDropMarks(railEl) {
  for (const el of railEl.querySelectorAll(`.${DROP_BEFORE}, .${DROP_AFTER}`)) {
    el.classList.remove(DROP_BEFORE, DROP_AFTER);
  }
}
