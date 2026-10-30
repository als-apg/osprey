// @ts-check
/* OSPREY Web Terminal — the sidebar's panel rows (DOM renderer)
 *
 * The labelled sidebar lists the panels in two groups inside `#panel-rail`:
 * `Workspace` (the Agent pane, Artifacts with its Script runs sibling, and
 * Events when that panel is enabled) and `Tools` (every other member, in the
 * server's rail order, then `+ Add panel`). This module builds and places
 * those rows and exposes small imperative mutators — set active, enable,
 * attention, append, remove — so `panel-manager.js`'s state machine can drive
 * the sidebar without owning any DOM specifics.
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
 * and out of view because its records come from the same server) and
 * `#panel-add-btn`, the last row of Tools. A render MOVES these nodes into
 * place and never rebuilds them, so their listeners, their lists and their
 * live state survive every render. A page-owned entry whose panel is not a
 * member stays in the document, `hidden`, and counts as absent for every
 * query below. A rail element without the page's skeleton (a test fixture)
 * gets the two groups built here and plain rows for every entry.
 *
 * This module holds NO panel state and issues NO fetches or POSTs. The caller
 * passes closures for the interactions (activate, context menu, drag, reorder).
 * After every full render it dispatches `osprey-sidebar-rendered` on
 * `document`, and again whenever a page-owned row comes into or out of view,
 * so the modules that fill those rows re-render against the placed nodes.
 *
 * DOM contract (stable — the browser suite, the tour and the dock select on it):
 *
 *   <div id="panel-rail" role="list">
 *     <div class="sidebar-group" data-group="workspace" role="listitem">
 *       <div class="osprey-section-label">Workspace</div>
 *       <button class="osprey-nav-row panel-rail-button disabled" data-panel-id="events"
 *               type="button" title="Events" aria-label="Events" data-tip="Events"
 *               draggable="true">                      (title gains " · hint" when panel.hint is given)
 *         <svg class="osprey-ic osprey-nav-icon">…</svg>
 *         <span class="osprey-nav-label">Events</span>
 *       </button>
 *     </div>
 *     <div class="sidebar-group" data-group="tools" role="listitem">…</div>
 *   </div>
 *
 * Beyond activation, a row's verbs (open beside, open standalone, remove) live
 * in the caller's right-click context menu — the rail only forwards the
 * contextmenu event through `onContextMenu` and suppresses the browser menu
 * when the caller reports it handled. A caller-supplied `panel.hint` is
 * appended to the tooltip so the menu is advertised.
 *
 * That menu is reachable from the keyboard, and explicitly so: a focused row
 * handles the `ContextMenu` key and Shift+F10 itself and asks the caller to
 * open the menu centred on the row's own rect. macOS fires no `contextmenu`
 * event from the keyboard at all, so leaving this to the platform would strand
 * those operators; on the platforms that DO fire one (Windows menu key,
 * Firefox Shift+F10) it is that keydown's default action, and suppressing the
 * default is what stops a single keypress opening two menus.
 *
 * State classes on a row: `.active` (surfaced panel), `.disabled` (backend
 * never answered), `.unreachable` (backend stopped answering), `.failed` (a
 * sidecar that failed to start; dimmed but clickable), `.agent-attention`
 * (the agent touched the panel and the operator has not looked yet; drawn as
 * an `.osprey-nav-badge` dot inside the row). While a badge time, an
 * unreachable notice or a status message owns the tooltip, the row carries a
 * transient `data-title-base` holding the tooltip it borrowed; clearing all
 * of them restores it and removes the attribute.
 */

import { flashElement } from '/design-system/js/highlight.js';
import { icon } from '/design-system/js/icons.js';
import { formatFacilityTime, viewerSharesFacilityClock } from '/design-system/js/facility-time.js';

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
 *   - a Tools row was dropped among the Tools rows: move `id` directly above
 *     `beforeId`, or to the foot of Tools when null. Never called for a drop
 *     that would leave the row where it is. Omit to keep the order fixed;
 *     drags into the workspace are unaffected either way.
 * @property {(id: string) => boolean} [reorderable]
 *   - whether a Tools row takes part in reordering (default: every one).
 *     Workspace rows never do: their order is the sidebar's, not the rail's.
 */

const BUTTON_SELECTOR = '.panel-rail-button';

/** The event dispatched on `document` after a render. */
export const SIDEBAR_RENDERED_EVENT = 'osprey-sidebar-rendered';

/** The glyph of a row whose descriptor names none. */
const DEFAULT_ICON = 'workspace';

/** The two groups, in sidebar order, with their visible headings. */
const GROUPS = /** @type {const} */ ([
  ['workspace', 'Workspace'],
  ['tools', 'Tools'],
]);

/**
 * Entries whose rows sit in the Workspace group, in this order. Every other
 * entry is a Tool. (The Script runs row, not an entry, follows Artifacts.)
 */
const WORKSPACE_ORDER = ['terminal', 'artifacts', 'events'];

/** Page-owned rows, by the entry id they hold. */
const PAGE_ROWS = /** @type {Readonly<Record<string, string>>} */ ({
  terminal: 'sidebar-agent-row',
  artifacts: 'sidebar-artifacts-row',
});

/** The Script runs row: page-owned, shown exactly while Artifacts is. */
const RUNS_ROW_ID = 'sidebar-runs-row';

/** The `+ Add panel` row: page-owned, always the last row of Tools. */
const ADD_BUTTON_ID = 'panel-add-btn';

/**
 * Where a row's own tooltip is parked while a badge time, an unreachable
 * notice or a status message owns the `title`. Present only while one of them
 * does — see {@link renderTitle}.
 */
const TITLE_BASE_ATTR = 'data-title-base';

/** The badge's "agent touched" clock time, present while the badge carries one. */
const TOUCHED_ATTR = 'data-title-touched';

/** The clock time the backend stopped answering, present while `.unreachable`. */
const UNREACHABLE_ATTR = 'data-title-unreachable';

const TOUCHED_SEPARATOR = ' · agent touched ';

/** A server-reported start status sentence, present while the row shows one. */
const STATUS_ATTR = 'data-title-status';

/** Marks the attention dot a row carries while it is `.agent-attention`. */
const ATTENTION_BADGE_ATTR = 'data-attention-badge';

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

// ---- Structure ----

/**
 * The rail's two group containers, built (heading included) where the page
 * did not ship them.
 * @param {HTMLElement} railEl
 * @returns {{workspace: HTMLElement, tools: HTMLElement}}
 */
function groupsOf(railEl) {
  /** @type {Record<string, HTMLElement>} */
  const found = {};
  for (const [name, heading] of GROUPS) {
    let group = /** @type {HTMLElement | null} */ (
      railEl.querySelector(`:scope > .sidebar-group[data-group="${name}"]`)
    );
    if (!group) {
      group = document.createElement('div');
      group.className = 'sidebar-group';
      group.dataset.group = name;
      group.setAttribute('role', 'listitem');
      const label = document.createElement('div');
      label.className = 'osprey-section-label';
      label.id = `${railEl.id || 'panel-rail'}-${name}-label`;
      label.textContent = heading;
      group.setAttribute('aria-labelledby', label.id);
      group.append(label);
      railEl.append(group);
    }
    found[name] = group;
  }
  return { workspace: found.workspace, tools: found.tools };
}

/** @param {HTMLElement} group @returns {HTMLElement | null} the group's heading */
function headingOf(group) {
  return /** @type {HTMLElement | null} */ (group.querySelector(':scope > .osprey-section-label'));
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
 * The row element that holds `entry`: the entry itself, or the page-owned
 * `.sidebar-row` wrapping it.
 * @param {HTMLElement} entry
 * @returns {HTMLElement}
 */
function rowOf(entry) {
  const wrapper = entry.parentElement;
  return wrapper && wrapper.classList.contains('sidebar-row') ? wrapper : entry;
}

/**
 * Whether an entry is a page-owned one parked out of view (not a member).
 * @param {Element} entry
 * @returns {boolean}
 */
function parked(entry) {
  return rowOf(/** @type {HTMLElement} */ (entry)).hidden === true;
}

/** @param {string} id @returns {boolean} */
function inWorkspace(id) {
  return WORKSPACE_ORDER.includes(id);
}

/**
 * Show or hide the Script runs row with the Artifacts row: its records come
 * from the same server, so it has nothing to list where Artifacts is absent.
 * @param {HTMLElement} railEl
 */
function syncRunsRow(railEl) {
  const runs = /** @type {HTMLElement | null} */ (railEl.querySelector(`#${RUNS_ROW_ID}`));
  if (runs) runs.hidden = getEntry(railEl, 'artifacts') === null;
}

/** Tell the row-filling modules the sidebar's rows have been placed. */
function announceRender() {
  document.dispatchEvent(new CustomEvent(SIDEBAR_RENDERED_EVENT));
}

// ---- Rendering ----

/**
 * Give a row its identity: label, accessible name, tooltips.
 * @param {HTMLElement} btn
 * @param {RailPanel} panel
 */
function applyIdentity(btn, panel) {
  const label = btn.querySelector('.osprey-nav-label');
  if (label) label.textContent = panel.label;
  btn.setAttribute('aria-label', panel.label);
  btn.dataset.tip = panel.label;
  const title = panel.hint ? `${panel.label} · ${panel.hint}` : panel.label;
  if (btn.hasAttribute(TITLE_BASE_ATTR)) btn.setAttribute(TITLE_BASE_ATTR, title);
  else btn.title = title;
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
 * Render every row into `railEl`: the Workspace entries in their fixed order,
 * the rest under Tools in `panels` order, then `+ Add panel`. Page-owned rows
 * are moved into place, never rebuilt; one whose panel is not in `panels`
 * stays in the Workspace group, hidden.
 *
 * This is the destructive full render. Use {@link addEntry} /
 * {@link removeEntry} for runtime membership changes so live rows keep their
 * active / health / enabled state.
 * @param {HTMLElement} railEl
 * @param {RailPanel[]} panels
 * @param {RailOptions} [options]
 */
export function createRail(railEl, panels, options = {}) {
  railEl.setAttribute('role', 'list');
  railOptionsOf.set(railEl, options);
  wireReorder(railEl);
  const { workspace, tools } = groupsOf(railEl);

  // Collect the page-owned nodes before the groups are emptied: moving them
  // back in is what keeps them (and everything hung on them) alive.
  const runsRow = railEl.querySelector(`#${RUNS_ROW_ID}`);
  const addButton = railEl.querySelector(`#${ADD_BUTTON_ID}`);
  const ownRows = WORKSPACE_ORDER.map((id) => pageRow(railEl, id));
  for (const row of ownRows) if (row) row.hidden = true;

  /** @type {Map<string, HTMLElement>} */
  const workspaceNodes = new Map();
  /** @type {HTMLElement[]} */
  const toolNodes = [];
  for (const panel of panels) {
    const node = entryNode(railEl, panel, options);
    if (inWorkspace(panel.id)) workspaceNodes.set(panel.id, node);
    else toolNodes.push(node);
  }

  /** @type {Node[]} */
  const workspaceKids = [];
  WORKSPACE_ORDER.forEach((id, i) => {
    const node = workspaceNodes.get(id) ?? ownRows[i];
    if (node) workspaceKids.push(node);
    if (id === 'artifacts' && runsRow) workspaceKids.push(runsRow);
  });
  const heading = (/** @type {HTMLElement} */ group) => {
    const label = headingOf(group);
    return label ? [label] : [];
  };
  workspace.replaceChildren(...heading(workspace), ...workspaceKids);
  tools.replaceChildren(...heading(tools), ...toolNodes, ...(addButton ? [addButton] : []));
  syncRunsRow(railEl);
  announceRender();
}

/**
 * Append a single entry without disturbing existing ones — the non-destructive
 * path for membership additions (agent add_panel_to_rail, "+" menu pick,
 * runtime registration). A Tools row lands at the foot of Tools; a Workspace
 * entry takes its fixed place.
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
  wireReorder(railEl);
  const { workspace, tools } = groupsOf(railEl);

  const own = pageRow(railEl, panel.id);
  const node = entryNode(railEl, panel, options);
  if (own) {
    syncRunsRow(railEl);
    announceRender();
  } else if (inWorkspace(panel.id)) {
    const rank = WORKSPACE_ORDER.indexOf(panel.id);
    const after = [...workspace.querySelectorAll(`:scope > ${BUTTON_SELECTOR}`)].find(
      (el) => WORKSPACE_ORDER.indexOf(el.getAttribute('data-panel-id') ?? '') > rank
    );
    workspace.insertBefore(node, after ?? null);
  } else {
    tools.insertBefore(node, tools.querySelector(`:scope > #${ADD_BUTTON_ID}`));
  }
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
  if (pageRow(railEl, panelId)) {
    rowOf(entry).hidden = true;
    syncRunsRow(railEl);
    announceRender();
    return;
  }
  entry.remove();
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
 * Move a Tools row so it sits directly above `beforeId`'s row, or at the foot
 * of Tools when `beforeId` is null or has no row. A `beforeId` in the
 * Workspace group (which precedes Tools) puts the row at the head of Tools.
 * Workspace rows keep their fixed places. Moving keeps the node, so its live
 * state (active, health, badges) travels with it. No-op for an unknown
 * `panelId`.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @param {string | null} beforeId
 */
export function placeEntry(railEl, panelId, beforeId) {
  const entry = getEntry(railEl, panelId);
  if (!entry || inWorkspace(panelId)) return;
  const { tools } = groupsOf(railEl);
  const before = beforeId === null ? null : getEntry(railEl, beforeId);
  if (before && tools.contains(before)) {
    tools.insertBefore(entry, before);
  } else if (before) {
    tools.insertBefore(entry, headingOf(tools)?.nextSibling ?? tools.firstChild);
  } else {
    tools.insertBefore(entry, tools.querySelector(`:scope > #${ADD_BUTTON_ID}`));
  }
}

// ---- Reordering ----

const DROP_BEFORE = 'drop-before';
const DROP_AFTER = 'drop-after';

/** Rails whose drop listeners are attached. @type {WeakSet<HTMLElement>} */
const reorderWired = new WeakSet();

/**
 * Attach the rail-level dragover / dragleave / drop listeners once per rail
 * element; they read the options of the rail's last render.
 * @param {HTMLElement} railEl
 */
function wireReorder(railEl) {
  if (reorderWired.has(railEl)) return;
  reorderWired.add(railEl);
  railEl.addEventListener('dragover', (e) => {
    const point = insertionPoint(railEl, e);
    if (!point) return;
    e.preventDefault(); // accept the drop here
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'move';
    clearDropMarks(railEl);
    if (point.noop) return;
    if (point.before) point.before.classList.add(DROP_BEFORE);
    else point.last?.classList.add(DROP_AFTER);
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
    const beforeId = point.before?.getAttribute('data-panel-id') ?? null;
    railOptionsOf.get(railEl)?.onReorder?.(point.id, beforeId);
  });
}

/**
 * Where a drag over the rail would put the dragged row, or null when the drag
 * is not a reorder of the Tools rows: it did not start on one of them, the row
 * is fixed, or the caller takes no reorders. The target is the first movable
 * Tools row whose vertical midpoint lies below the pointer; none means the
 * foot of Tools.
 * @param {HTMLElement} railEl
 * @param {DragEvent} e
 * @returns {{ id: string, before: HTMLElement | null, last: HTMLElement | null, noop: boolean } | null}
 */
function insertionPoint(railEl, e) {
  const options = railOptionsOf.get(railEl);
  const { tools } = groupsOf(railEl);
  const dragging = /** @type {HTMLElement | null} */ (
    tools.querySelector(`${BUTTON_SELECTOR}.dragging`)
  );
  if (!options?.onReorder || !dragging) return null;
  const movable = (/** @type {Element} */ el) =>
    !options.reorderable || options.reorderable(el.getAttribute('data-panel-id') ?? '');
  if (!movable(dragging)) return null;

  const rows = /** @type {HTMLElement[]} */ (
    [...tools.querySelectorAll(BUTTON_SELECTOR)].filter(movable)
  );
  const others = rows.filter((el) => el !== dragging);
  const before =
    others.find((el) => {
      const r = el.getBoundingClientRect();
      return r.top + r.height / 2 > e.clientY;
    }) ?? null;
  const next = rows[rows.indexOf(dragging) + 1] ?? null;
  return {
    id: dragging.getAttribute('data-panel-id') ?? '',
    before,
    last: others.at(-1) ?? null,
    noop: before === next,
  };
}

/** @param {HTMLElement} railEl */
function clearDropMarks(railEl) {
  for (const el of railEl.querySelectorAll(`.${DROP_BEFORE}, .${DROP_AFTER}`)) {
    el.classList.remove(DROP_BEFORE, DROP_AFTER);
  }
}

// ---- Mutators ----

/**
 * Return the row button for a panel id, or null when the panel has no row in
 * view (absent, or a page-owned row parked out of view).
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @returns {HTMLButtonElement | null}
 */
export function getEntry(railEl, panelId) {
  const entry = /** @type {HTMLButtonElement | null} */ (
    railEl.querySelector(`${BUTTON_SELECTOR}[data-panel-id="${CSS.escape(panelId)}"]`)
  );
  return entry && !parked(entry) ? entry : null;
}

/**
 * Mark exactly one row active (`.active` plus `aria-current="page"`, which
 * draws the selected wash and the accent bar) and clear it from every other
 * row. Passing an id with no matching row clears the active state on all.
 * @param {HTMLElement} railEl
 * @param {string | null} panelId
 */
export function setActive(railEl, panelId) {
  for (const el of railEl.querySelectorAll(BUTTON_SELECTOR)) {
    const isActive = el.getAttribute('data-panel-id') === panelId;
    el.classList.toggle('active', isActive);
    if (isActive) el.setAttribute('aria-current', 'page');
    else el.removeAttribute('aria-current');
  }
}

/**
 * Enable or disable a row by toggling `.disabled`. Rows render disabled; the
 * caller enables one once its backend first reports healthy. This is the
 * sidebar's ONLY health-derived state besides reachability. No-op when the
 * row is absent.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @param {boolean} enabled
 */
export function setEntryEnabled(railEl, panelId, enabled) {
  getEntry(railEl, panelId)?.classList.toggle('disabled', !enabled);
}

/**
 * A server or poll instant as the sidebar's tooltips show it: hour and minute
 * on the facility clock, naming its zone when the viewer's clock reads
 * differently. It is the one formatter for every "since" time a panel surface
 * shows, so the sidebar and the tile never disagree.
 * @param {number} ms - epoch milliseconds
 * @returns {string}
 */
export function clockTime(ms) {
  return formatFacilityTime(ms, {
    hour: 'numeric',
    minute: '2-digit',
    ...(viewerSharesFacilityClock(ms) ? {} : { timeZoneName: 'short' }),
  });
}

/**
 * Compose a row's tooltip from its own text and the notices borrowing it.
 *
 * The own text is stashed on the row while any notice is up rather than
 * recomputed, so the restore is exact even if the caller retitled the row,
 * and a repeated notice REPLACES its time instead of appending a second one.
 * A status message stands in for the own text, so a badge time is appended to
 * it. An unreachable notice outranks both: "not answering" is the fact an
 * operator needs before clicking. With no notice the stash is restored and
 * removed, which makes a clear on an untouched row a true no-op.
 * @param {HTMLElement} entry
 */
function renderTitle(entry) {
  const base = entry.getAttribute(TITLE_BASE_ATTR) ?? entry.title;
  const touched = entry.getAttribute(TOUCHED_ATTR);
  const unreachable = entry.getAttribute(UNREACHABLE_ATTR);
  const status = entry.getAttribute(STATUS_ATTR);
  if (touched === null && unreachable === null && status === null) {
    if (entry.hasAttribute(TITLE_BASE_ATTR)) {
      entry.title = base;
      entry.removeAttribute(TITLE_BASE_ATTR);
    }
    return;
  }
  entry.setAttribute(TITLE_BASE_ATTR, base);
  const own = status ?? base;
  if (unreachable !== null) entry.title = `not answering since ${unreachable}`;
  else entry.title = touched !== null ? `${own}${TOUCHED_SEPARATOR}${touched}` : own;
}

/**
 * Show or remove a row's attention dot: an `.osprey-nav-badge` beside the
 * label (before the disclosure chevron, where the row has one).
 * @param {HTMLElement} entry
 * @param {boolean} on
 */
function renderAttentionBadge(entry, on) {
  const badge = entry.querySelector(`:scope > [${ATTENTION_BADGE_ATTR}]`);
  if (!on) {
    badge?.remove();
    return;
  }
  if (badge) return;
  const dot = document.createElement('span');
  dot.className = 'osprey-nav-badge sidebar-attention';
  dot.setAttribute(ATTENTION_BADGE_ATTR, '');
  dot.setAttribute('aria-hidden', 'true');
  entry.insertBefore(dot, entry.querySelector(':scope > .osprey-nav-disc'));
}

/**
 * Set or clear the agent-attention affordance on a row. Turning it on toggles
 * the persistent `agent-attention` state class, draws the row's attention
 * dot, fires the one-shot `agent-flash` glow via {@link flashElement}, and
 * scrolls the row into view: the Workspace section scrolls when its lists are
 * open, and a badge scrolled out of sight reports nothing to the operator.
 * `block: 'nearest'` leaves a visible row exactly where it is.
 *
 * Turning it off removes the dot and restores the tooltip; an in-flight flash
 * is left to finish on its own `animationend`.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @param {boolean} on
 * @param {number} [ts] - the originating event's SERVER timestamp (epoch
 *   seconds, as the `agent_activity` SSE frames carry it), appended to the
 *   row's tooltip as "· agent touched <time>". Omit it — or pass a non-finite
 *   value — for a badge with no time claim; never substitute a client clock,
 *   which would report when the browser rendered rather than when the agent
 *   acted.
 * @returns {boolean} true when the row existed and was updated; false for an
 *   unknown id (safe no-op, so callers can fall back)
 */
export function setEntryAttention(railEl, panelId, on, ts) {
  const entry = getEntry(railEl, panelId);
  if (!entry) return false;
  entry.classList.toggle('agent-attention', on);
  renderAttentionBadge(entry, on);
  if (on && typeof ts === 'number' && Number.isFinite(ts)) {
    entry.setAttribute(TOUCHED_ATTR, clockTime(ts * 1000));
  } else {
    entry.removeAttribute(TOUCHED_ATTR);
  }
  renderTitle(entry);
  if (on) {
    flashElement(entry);
    entry.scrollIntoView({ block: 'nearest' });
  }
  return true;
}

/**
 * Show or clear a server-reported start status on a row. `failed` toggles
 * `.failed`, which replaces `.disabled`: the row stays dimmed but takes a
 * click, the caller's retry. While `message` is set, the tooltip and
 * `aria-description` read exactly that string (with a badge time appended
 * while the row is badged); `message: null` puts the row's own tooltip back.
 * No-op when the row is absent.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @param {{ failed: boolean, message: string | null }} status
 */
export function setEntryStatus(railEl, panelId, { failed, message }) {
  const entry = getEntry(railEl, panelId);
  if (!entry) return;
  entry.classList.toggle('failed', failed);
  if (failed) entry.classList.remove('disabled');
  if (message) {
    entry.setAttribute(STATUS_ATTR, message);
    entry.setAttribute('aria-description', message);
  } else {
    entry.removeAttribute(STATUS_ATTR);
    entry.removeAttribute('aria-description');
  }
  renderTitle(entry);
}

/**
 * Mark a row whose backend answered before but has stopped, or bring it
 * back. Unreachable dims the row like `.disabled` but leaves it clickable —
 * its menu and popout stay live — and points its tooltip at when the backend
 * stopped answering. Reachable restores the row and its tooltip. `.disabled`
 * (never answered) is untouched either way.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @param {boolean} reachable
 * @param {number} [sinceMs] - epoch ms of the first unanswered poll, shown in
 *   the tooltip while unreachable
 * @returns {boolean} true when the row existed and was updated
 */
export function setEntryReachable(railEl, panelId, reachable, sinceMs) {
  const entry = getEntry(railEl, panelId);
  if (!entry) return false;
  entry.classList.toggle('unreachable', !reachable);
  if (reachable) {
    entry.removeAttribute(UNREACHABLE_ATTR);
  } else {
    entry.setAttribute(UNREACHABLE_ATTR, clockTime(sinceMs ?? Date.now()));
  }
  renderTitle(entry);
  return true;
}
