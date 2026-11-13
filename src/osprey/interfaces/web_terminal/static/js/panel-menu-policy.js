// @ts-check
/* OSPREY Web Terminal — Panel Menu Policy
 *
 * WHAT a gesture on a panel surface does. The panel state machine now splits
 * three ways — panel-manager.js owns the core (tile occupancy, iframe
 * lifecycle, the active-panel policy), panel-lifecycle.js owns how a panel
 * comes to exist and stay alive, and panel-sse.js owns the server→client
 * frame dispatcher — and panel-placement.js owns tile geometry; this module
 * owns the policy layer in between — the closures a sidebar row is rendered
 * with, and the context menus a sidebar row or a tile header opens.
 *
 * It is the counterpart of panel-context-menu.js, which is purely
 * presentational (popover DOM, dismissal, focus bookkeeping) and never knows
 * what a verb does. Every decision lives here instead: which menu a surface
 * gets, whether it gets one at all, which rows the Agent tile's surface allows, and
 * which closure each row runs. The rows bind the SAME closures the sidebar
 * click, the row drag and the agent MCP tools use, so a menu pick is
 * indistinguishable downstream from any other path to the verb.
 *
 * Leaf modules are imported directly; everything that lives in panel-manager's
 * private state arrives as injected deps registered once at init
 * ({@link initMenuPolicy}) — the same seam pattern panel-placement.js, the
 * iframe adapter and the tile-close handler use. That injection is what keeps
 * this module out of an import cycle: panel-manager imports it, never the
 * reverse.
 *
 * TWO SURFACES, ONE VERB SET
 * --------------------------
 * A panel's sidebar row and its tile bar (the ⋯ button and a right-click on
 * the bar) build their rows from the same row helpers and bind the same
 * closures, so a verb cannot drift between them. They differ deliberately:
 * the sidebar applies an enabled gate (see openRailContextMenu) and the tile
 * does not; both lead with a "This panel" section (the panel's contributed
 * menu entries, the rows a module owning the panel's content added through
 * {@link setServiceMenuExtras}, and the controls a narrow bar folded), so a
 * panel's own verbs stay reachable from its sidebar row while it has no tile.
 * Below it the tile's ⋯ (tile-menu.js) has the "Tile" verbs
 * ({@link buildTileMenuItems}); the sidebar row has the verbs of the row's
 * state — docked, in its window, tile closed, or failed to start
 * ({@link buildServiceMenuItems}). The Agent is the exception on the
 * sidebar side: its row heads the session tree and gets the tree's menu; its
 * tile gets the pane verbs ({@link buildTerminalMenuItems}).
 *
 * DECLINE BY EMPTY LIST
 * ---------------------
 * Both openers return a boolean their caller turns into `preventDefault()`:
 * true when a menu opened, false to leave the event alone so the browser's own
 * menu still shows. A surface with NO rows to offer declines.
 */

import { openContextMenu } from './panel-context-menu.js';
import { AGENT_HEADING, TILE_HEADING, composeTileMenu } from './tile-menu.js';
import { panelMenuRows } from './tile-header-contrib.js';
import { helpUrl, openHelp } from './panel-help.js';
import { reloadPanel } from './dock-iframe.js';
import { TERMINAL_PANE_LABEL, TERMINAL_RAIL_ID, TERMINAL_RAIL_LABEL } from './panel-catalog.js';
import { setPanelVisibility, setRailOrder } from './panel-commands.js';
import {
  dockedTileOf,
  hasDockedTile,
  openPanelAtTile,
  openPanelBeside,
} from './panel-placement.js';
import { NO_PANELS_LABEL, untiledPanels } from './dock-menus.js';
import {
  openTerminalPanel,
  closeTerminalPanel,
  isTerminalPanelOpen,
  closeDockPanel,
  maximizeTile,
  maximizedTileKey,
  restoreTile,
} from './dock-workspace.js';
import { withEchoSuppressed } from './dock-sync.js';
import { isTerminalSurface, showChatSurface, showTerminalSurface } from './agent-surface.js';
import { terminalOffered } from '/design-system/js/frame-params.js';
import { focusTerminal, restartTerminal, startTerminal } from './terminal.js';
import { buildAgentRowMenuItems, initTreeMenus } from './sidebar-artifact-menus.js';
import { railDragStart, railDragEnd } from './rail-drag.js';
import { getEntry, entryIds, DETACHED_TRAIL_ATTR } from './panel-rail.js';
import {
  isListOpen,
  isSidebarCollapsed,
  setListOpen,
  setSidebarCollapsed,
  toggleEverySession,
} from './sidebar.js';
import { setGroupFolded } from './sidebar-fold.js';
import { flashElement } from '/design-system/js/highlight.js';
import { showUndoNotice } from './undo-notice.js';
import { closeDetached, detach, detachBlocker, focusWindow, reattach } from './pane-window.js';
import { isDetached } from './dock-apply.js';
import { isArtifactTileKey } from './artifact-tiles.js';
import { zoomMenuRow } from './pane-zoom.js';
import {
  NEW_GROUP_LABEL,
  addGroup,
  groupOf,
  loadGroups,
  moveToGroup,
  startGroupRename,
} from './sidebar-groups.js';

/** The Agent row's list (sidebar.js's disclosure id for the session tree). */
const AGENT_LIST = 'agent';

/**
 * The panel-manager state and actions this module's closures act through. Every
 * entry is a live closure over panel-manager's private state — never a
 * snapshot, so a menu built now reads the rail and health of this moment.
 * @typedef {object} MenuPolicyDeps
 * @property {() => HTMLElement} getRailEl - the sidebar list the rows live in
 * @property {(id: string) => boolean} isMember - the panel has rail membership
 * @property {() => string | null} getActiveTabId - the locally surfaced panel id
 * @property {(id: string, options?: {userInitiated?: boolean, auto?: boolean}) => void} activateTab
 * @property {(id: string) => void} showPanel - reveal a NON-member through the visibility POST path
 * @property {(id: string) => void} retireTile - vacate a surfaced tile LOCALLY (membership survives)
 * @property {(id: string) => string} labelOf - the panel's catalog label
 * @property {(id: string) => void} retryStart - start a failed sidecar panel again
 */

/** @type {MenuPolicyDeps | null} */
let deps = null;

/**
 * Register the panel-manager state every closure here reads. Called once from
 * initPanelManager, before the rail is rendered.
 * @param {MenuPolicyDeps} menuPolicyDeps
 */
export function initMenuPolicy(menuPolicyDeps) {
  deps = menuPolicyDeps;
  // The session tree's rows open their menus (sidebar-artifact-menus.js).
  initTreeMenus();
}

/**
 * The registered deps, or a hard failure. A gesture before init is a wiring
 * bug, not a runtime condition to degrade around — panel-manager registers in
 * the same init that mounts the rail.
 * @returns {MenuPolicyDeps}
 */
function ctx() {
  if (!deps) throw new Error('panel-menu-policy: initMenuPolicy() has not run');
  return deps;
}

/**
 * Interaction closures handed to every sidebar render/append call. Routing
 * activation through here keeps a human click and an agent MCP call
 * indistinguishable downstream (both hit activateTab / setPanelVisibility).
 *
 * Every row is a member, so a click is always an activation: show this panel
 * in the service tile used last, replacing the panel it showed, or focus it
 * where it is (the open rule, dock-open.js). The placement happens in the dock
 * adapter and is purely local — the replaced panel keeps its row. The Agent row
 * routes to the dock and its session list instead (its open/closed state is
 * layout state, not membership — see TERMINAL_RAIL_ID and activateAgentRow).
 */
export function railOptions() {
  return {
    onActivate: (/** @type {string} */ id, /** @type {MouseEvent | undefined} */ event) => {
      // The icon on a detached pane's row reattaches it, the Sessions row included.
      if (isDetached(id) && onDetachedTrail(event)) {
        reattach(id);
        return;
      }
      if (id === TERMINAL_RAIL_ID) {
        activateAgentRow(event);
        return;
      }
      // A failed sidecar's row has one verb: start it again.
      if (isFailed(id)) {
        ctx().retryStart(id);
        return;
      }
      // A detached panel's row brings its window to the front.
      if (isDetached(id)) {
        focusWindow(id);
        return;
      }
      // Clicking the row whose tile is ALREADY surfaced retires that tile — a
      // toggle shortcut equivalent to the tile header's own "×". It stays a
      // LOCAL layout change: membership survives, so a second click brings
      // the tile back.
      if (ctx().isMember(id)) {
        if (ctx().getActiveTabId() === id) {
          ctx().retireTile(id);
          return;
        }
        ctx().activateTab(id, { userInitiated: true });
      } else ctx().showPanel(id);
    },
    // Right-click — or the menu key / Shift+F10, which the sidebar forwards
    // through this same closure — opens the row's verbs in words
    // (panel-context-menu.js). openRailContextMenu owns the whole policy
    // (which menu, and whether there is one at all) and reports back with the
    // boolean the sidebar uses to decide whether to suppress the browser's
    // menu.
    onContextMenu: (/** @type {string} */ id, /** @type {MouseEvent} */ e) =>
      openRailContextMenu(id, e.clientX, e.clientY),
    // Row drag source: the Agent row never drags (its tile moves by its own
    // header bar); everything else defers to rail-drag's policy (fallback mode
    // cancels there).
    onDragStart: (/** @type {string} */ id, /** @type {DataTransfer | null} */ dt) =>
      id === TERMINAL_RAIL_ID || isFailed(id) ? false : railDragStart(id, dt),
    onDragEnd: () => railDragEnd(),
    // The same drag, dropped among the sidebar's rows instead of the
    // workspace, reorders them within a group, or moves the row into another
    // group first (panel-rail.js). The Agent row keeps its place.
    onReorder: (/** @type {string} */ id, /** @type {string | null} */ before) => {
      void setRailOrder(id, before);
    },
  };
}

/**
 * Whether a row's click landed on the row's detached icon.
 * @param {MouseEvent | undefined} event
 * @returns {boolean}
 */
function onDetachedTrail(event) {
  const target = event?.target;
  return target instanceof Element && target.closest(`[${DETACHED_TRAIL_ATTR}]`) !== null;
}

/**
 * The Agent row's click. With the Agent pane closed it opens the pane and
 * unfolds the session list; with the pane open it folds or unfolds the list
 * and leaves the pane as it is — the pane closes only from its own header;
 * with the pane in its own window it brings that window to the front.
 * A ⌥-click (altKey, from a pointer: Enter and Space stay a plain click)
 * expands every session, or collapses them all when the list shows them all
 * open (sidebar-artifacts.js, imported lazily: it imports this module),
 * and leaves the pane as it is.
 * The one place that decides: sidebar.js handles only the collapsed
 * sidebar's flyout, and stops the click before it reaches here.
 * @param {MouseEvent} [event]
 */
function activateAgentRow(event) {
  if (event?.altKey && event.detail > 0) {
    // A folded list unfolds onto every session open, never onto a collapse.
    const force = isListOpen(AGENT_LIST) ? undefined : true;
    toggleEverySession(force);
    return;
  }
  if (isDetached(TERMINAL_RAIL_ID)) {
    focusWindow(TERMINAL_RAIL_ID);
    return;
  }
  if (isTerminalPanelOpen()) {
    setListOpen(AGENT_LIST, !isListOpen(AGENT_LIST));
    return;
  }
  setListOpen(AGENT_LIST, true);
  showAgentPane();
}

/**
 * Bring the Agent pane forward: its window when detached, else the pane with
 * the caret in it ({@link showAgentPane}). The session tree's Focus.
 */
export function focusAgentPane() {
  if (isDetached(TERMINAL_RAIL_ID)) focusWindow(TERMINAL_RAIL_ID);
  else showAgentPane();
}

/**
 * Open the Agent pane, then put the caret where the operator types — the chat
 * composer, or the terminal when that is the surface showing.
 */
export function showAgentPane() {
  openTerminalPanel();
  if (!isTerminalPanelOpen()) return;
  if (isTerminalSurface()) focusTerminal();
  else {
    /** @type {HTMLElement | null} */ (
      document.querySelector('#operator-container textarea')
    )?.focus();
  }
}

/**
 * Take a panel off the rail for the operator — the "Remove from rail" row and
 * the row's Delete key — and offer the way back. The Undo re-adds the panel at
 * the position it held among the sidebar's panel rows (the Agent row not
 * counted), which the server carries on its echo so every client restores it
 * there. The removal also closed the panel's tile, so a panel that had one is
 * shown again too — at the default placement, since the tile's old spot in the
 * grid is not something the rail can put back. A panel in its window closes with
 * the window, as a closed window does.
 *
 * The notice belongs to the gesture, not to the membership change: an agent's
 * remove_panel_from_rail and a removal made in another tab arrive here only as
 * SSE echoes, which never pass through this function, so they raise nothing.
 * A refused POST raises a plain notice with no Undo — there is nothing to take
 * back.
 * @param {string} id
 * @returns {Promise<void>}
 */
export async function removeFromRail(id) {
  const label = ctx().labelOf(id);
  const index = entryIds(ctx().getRailEl())
    .filter((e) => e !== TERMINAL_RAIL_ID)
    .indexOf(id);
  // Fallback mode has no tiles, only the one surface the active panel fills.
  const hadTile = hasDockedTile(id) || ctx().getActiveTabId() === id || isDetached(id);
  if (!(await setPanelVisibility(id, false))) {
    showUndoNotice({ message: `Could not remove ${label} from the rail` });
    return;
  }
  closeDetached(id);
  showUndoNotice({
    message: `${label} removed from the rail`,
    undo: async () => {
      const restored = await setPanelVisibility(id, true, index < 0 ? undefined : index);
      if (!restored) showUndoNotice({ message: `Could not put ${label} back on the rail` });
      else if (hadTile) ctx().activateTab(id, { userInitiated: true });
    },
  });
}

/**
 * Close the Agent pane for the operator — the "Close Agent pane" row and the
 * Agent row's Delete key — and offer the way back. The close is local layout,
 * so the notice is this tab's alone; the Undo reopens the pane through the
 * same path as a click on the Agent row. Nothing is shown when there was no
 * pane to close.
 */
export function closeTerminalTile() {
  if (!closeTerminalPanel()) return;
  showUndoNotice({ message: 'Agent pane closed', undo: () => openTerminalPanel() });
}

/**
 * Whether a panel's sidebar row shows a sidecar that failed to start.
 * @param {string} id
 * @returns {boolean}
 */
function isFailed(id) {
  return getEntry(ctx().getRailEl(), id)?.classList.contains('failed') ?? false;
}

// Tooltip suffix advertising the context menu on every service row. The
// Agent row gets none: it is the session tree's head, not a panel row.
export const RAIL_MENU_HINT = 'right-click for actions';

/**
 * Open a sidebar row's context menu at a viewport position, and report
 * whether one was opened — the boolean the sidebar uses to decide whether to
 * suppress the browser's native menu, on the right-click and keyboard routes
 * alike (FR5).
 *
 * The enabled gate lives HERE rather than in the sidebar CSS. `pointer-events:
 * none` on a disabled row stops a right-click landing on the row itself, but
 * it does not stop a menu request fired from the KEYBOARD on that focused
 * button, which would otherwise reach a menu whose verbs act on a panel that
 * has never answered — so a never-started row declines, and the browser's
 * menu shows. A `.failed` row is NOT declined: its menu is the retry and the
 * removal ({@link buildServiceMenuItems}).
 * @param {string} id  @param {number} x  @param {number} y
 * @returns {boolean} true when a menu was opened
 */
export function openRailContextMenu(id, x, y) {
  const entry = getEntry(ctx().getRailEl(), id);
  if (!entry || entry.classList.contains('disabled')) return false;
  // The Agent row heads the session tree, so its menu is the tree's (New
  // session, Artifacts by, Show pinned, Collapse/Expand all), then the pane rows a
  // tileless Agent pane has nowhere else.
  if (id === TERMINAL_RAIL_ID) {
    openContextMenu({
      x,
      y,
      anchorEl: entry,
      ariaLabel: `${TERMINAL_RAIL_LABEL} actions`,
      items: [...buildAgentRowMenuItems(), ...tilelessAgentRows()],
    });
    return true;
  }
  // The row is the anchor: it scopes the menu's scroll dismissal, so the menu
  // survives a panel scrolling its own content underneath it.
  return openSurfaceMenu(id, buildServiceMenuItems(id), { x, y, anchorEl: entry });
}

/**
 * The tile-bar counterpart of openRailContextMenu, registered into
 * dock-tab.js at init and reached from the bar's ⋯ button and from a
 * right-click on the bar alike: the tile's one menu
 * ({@link buildTileMenuItems}, {@link buildTerminalMenuItems}). The same
 * decline-by-empty-list contract: a surface with no rows to offer reports
 * false and keeps the browser's native menu.
 *
 * The sidebar's enabled gate has no counterpart here on purpose. It exists so
 * a panel that has never answered cannot be acted on from an offline row; a
 * tile header only exists for a panel already occupying a tile, and its verbs
 * (focus it, open it in a window, drop it from the rail) stay meaningful even if health
 * polling has since gone quiet.
 *
 * @param {string} id  service panel id, or TERMINAL_RAIL_ID
 * @param {import('./dock-tab.js').TileMenuRequest} opts
 * @returns {boolean} true when a menu was opened
 */
export function openTileContextMenu(id, opts) {
  const { mode = 'tile', ...at } = opts;
  /** @type {MenuSurface} */
  const surface = { mode, noticeIn: opts.anchorEl.ownerDocument };
  const items =
    id === TERMINAL_RAIL_ID ? buildTerminalMenuItems(surface) : buildTileMenuItems(id, surface);
  // The header bar is the anchor — the menu rides out the panel's own scrolls,
  // including the terminal streaming output underneath it.
  return openSurfaceMenu(id, items, at);
}

/**
 * Which bar a menu is for. A bar in the pane's own window (`mode: 'detached'`)
 * offers nothing that acts on the dock or the sidebar's arrangement: there is no
 * tile to focus, move or close from there; nor does the no-dockview fallback
 * bar (`mode: 'fallback'`), which has no dock at all. `noticeIn` is where a refused Reattach
 * says so: the window the gesture came from.
 * @typedef {{mode?: import('./dock-tab.js').TileMode, noticeIn?: Document}} MenuSurface
 */

/**
 * The shared tail of both surface openers: decline by empty list, and open
 * the popover. Label and the decline signal live in this one place.
 * @param {string} id  service panel id, or TERMINAL_RAIL_ID
 * @param {import('./panel-context-menu.js').MenuItem[]} items
 * @param {{x: number, y: number, anchorEl: HTMLElement, triggerEl?: HTMLElement}} at
 *   position, the anchor that scopes the scroll dismissal, and (from the ⋯
 *   button) the trigger the menu hangs under
 * @returns {boolean} true when a menu was opened
 */
function openSurfaceMenu(id, items, at) {
  // No rows is a decline, never an empty popover.
  if (!items.length) return false;
  const label = id === TERMINAL_RAIL_ID ? TERMINAL_PANE_LABEL : ctx().labelOf(id);
  openContextMenu({ ...at, ariaLabel: `${label} actions`, items });
  return true;
}

/**
 * Extra rows per panel, registered by the module that owns the panel's content.
 * @type {Map<string, () => import('./tile-menu.js').PanelRow[]>}
 */
const serviceMenuExtras = new Map();

/**
 * Add rows of a panel's own to its menus: `fn()` is asked at every open, so
 * the rows can follow the panel's state. In the tile ⋯ and on the sidebar row
 * alike they are in the "This panel" section, after the panel's contributed
 * menu entries — on the sidebar row even while the panel has no tile, so a
 * verb like Jupyter's Restart kernel server stays reachable. Registering again replaces the
 * earlier rows; `null` removes them. A row may carry `separator: true` for a
 * divider before it; destructive rows (`danger`) go last in the tile's
 * section.
 * @param {string} panelId
 * @param {(() => import('./tile-menu.js').PanelRow[]) | null} fn
 */
export function setServiceMenuExtras(panelId, fn) {
  if (fn) serviceMenuExtras.set(panelId, fn);
  else serviceMenuExtras.delete(panelId);
}

/**
 * A service panel's SIDEBAR ROW menu, bound to the SAME closures the sidebar
 * click, the row drag and the agent MCP tools use — a menu pick is
 * indistinguishable downstream from any other path to the verb.
 *
 * First, in every state, the "This panel" section ({@link thisPanelSection}):
 * the panel's contributed menu entries and the rows registered through
 * {@link setServiceMenuExtras} — content verbs may repeat the tile's. Then the
 * verbs of the row's state:
 * - docked (a tile shows it): Focus tile, Open in a new tile, Open in new window;
 * - in its window: Show window, Reattach;
 * - tile closed: Open, Open in new window.
 * Then, for a member, "Move to ▸" (a submenu of the other groups and "New
 * group…"), and Remove from rail last.
 *
 * A row whose sidecar failed to start has two verbs only: Retry start — the
 * same retry its click runs — and Remove from rail. A row that never started
 * gets no menu at all; openRailContextMenu declines it before this runs.
 *
 * @param {string} id
 * @param {MenuSurface} [surface]  only `noticeIn` is read: where a refused Reattach says so
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
export function buildServiceMenuItems(id, { noticeIn } = {}) {
  const member = ctx().isMember(id);
  if (isFailed(id)) {
    return [
      { label: 'Retry start', run: () => ctx().retryStart(id) },
      { divider: true },
      membershipRow(id, member),
    ];
  }
  return [
    ...thisPanelSection(id),
    ...stateRows(id, member, noticeIn),
    ...(member ? [moveToRow(id)] : []),
    { divider: true },
    membershipRow(id, member),
  ];
}

/**
 * The rail entry's "This panel" section, closed by a divider: the same rows
 * the tile ⋯ shows under that caption, tidied the same way (separators
 * expanded, destructive rows last). Nothing when the panel contributes no rows.
 * @param {string} id
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
function thisPanelSection(id) {
  const rows = [...panelMenuRows(id), ...(serviceMenuExtras.get(id)?.() ?? [])];
  // composeTileMenu with no tile rows yields the captioned section and a bare
  // "Tile" caption, which a sidebar row has no use for.
  const section = composeTileMenu({ thisPanel: rows, tile: [] }).filter(
    (row) => row.heading !== TILE_HEADING
  );
  return section.length ? [...section, { divider: true }] : [];
}

/**
 * The verbs of a sidebar row's state. Focus tile and Show window bring the
 * panel forward without the row click's retire branch ({@link focusTile}): a
 * row that says "Focus" and made the panel disappear would say the opposite
 * of what it does. Docked is a tile in the dock, or the surfaced panel when
 * there is no dock (the fallback).
 * @param {string} id
 * @param {boolean} member
 * @param {Document} [noticeIn]
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
function stateRows(id, member, noticeIn) {
  if (isDetached(id)) {
    return [{ label: 'Show window', run: () => focusWindow(id) }, paneWindowRow(id, noticeIn)];
  }
  if (hasDockedTile(id) || ctx().getActiveTabId() === id) {
    return [
      { label: 'Focus tile', run: () => focusTile(id) },
      newTileRow(id),
      paneWindowRow(id, noticeIn),
    ];
  }
  return [
    { label: 'Open', run: () => ctx().activateTab(id, { userInitiated: member }) },
    paneWindowRow(id, noticeIn),
  ];
}

/**
 * A service panel's TILE ⋯ menu (tile-menu.js), the same for the ⋯ button
 * and a right-click on the bar: "This panel" — the panel's contributed menu
 * entries, the rows registered through {@link setServiceMenuExtras}, and the
 * controls the narrow bar folded — then "Tile": Open panel ▸
 * ({@link openPanelRow}), Maximize / Restore ({@link maximizeRow}), Zoom ▸
 * (pane-zoom.js), {@link paneWindowRow},
 * Reload, Help (only when the panel has a docs page, panel-help.js), Reveal
 * in sidebar ({@link revealInSidebar}) — or Pin to rail in its place on a
 * tile an agent's focus opened off the rail — and Close tile last.
 * The bar arranges no rail and no layout: moving, unpinning and resetting
 * live on the sidebar.
 *
 * Close tile is the tile's ×: it closes the tile only, and the panel keeps
 * its sidebar row.
 *
 * In the pane's own window (`mode: 'detached'`) — the panel's only bar
 * there, so "This panel" is as docked — the Tile section is Reattach, Zoom ▸, Reload,
 * Help, Pin to rail (off-rail panels), then Close window under a divider
 * ({@link closeWindowRows}); on the no-dockview fallback
 * bar (`mode: 'fallback'`) it is Reload, Help and Pin to rail. Neither acts
 * on a dock. An artifact tile (artifact-tiles.js) is no panel the rail can
 * hold: it gets no Pin to rail.
 * @param {string} id
 * @param {MenuSurface} [surface]
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
export function buildTileMenuItems(id, { mode = 'tile', noticeIn } = {}) {
  const member = ctx().isMember(id);
  const thisPanel = [...panelMenuRows(id), ...(serviceMenuExtras.get(id)?.() ?? [])];
  const reloadHelp = [
    { label: 'Reload', run: () => void reloadPanel(id) },
    ...(helpUrl(id) === null ? [] : [{ label: 'Help', run: () => void openHelp(id) }]),
  ];
  const pin = member || isArtifactTileKey(id) ? [] : [membershipRow(id, false)];
  if (mode === 'detached') {
    const tile = [
      paneWindowRow(id, noticeIn),
      zoomMenuRow(id),
      ...reloadHelp,
      ...pin,
      ...closeWindowRows(id),
    ];
    return composeTileMenu({ thisPanel, tile });
  }
  if (mode === 'fallback') return composeTileMenu({ thisPanel, tile: [...reloadHelp, ...pin] });
  // A member's row is in the sidebar to reveal; an off-rail panel has none,
  // so Pin to rail takes the slot instead.
  const revealOrPin = member ? [revealRow(id)] : pin;
  const tile = [
    openPanelRow(id),
    maximizeRow(id),
    zoomMenuRow(id),
    paneWindowRow(id, noticeIn),
    ...reloadHelp,
    ...revealOrPin,
    closeTileRow(id),
  ];
  return composeTileMenu({ thisPanel, tile });
}

/**
 * A docked tile bar's Maximize / Restore. The label is read from the dock
 * each time the menu opens (dock-workspace.js's maximizedTileKey — view
 * state, never stored): Restore while any tile is maximized, else Maximize,
 * which fills the dock with the tile showing `key`. Pop-out bars and the
 * no-dockview fallback bar have no dock to maximize in and never carry it.
 * @param {string} key  a service key, or TERMINAL_RAIL_ID for the Agent pane
 * @returns {import('./panel-context-menu.js').MenuItem}
 */
function maximizeRow(key) {
  return maximizedTileKey() === null
    ? { label: 'Maximize', run: () => void maximizeTile(key) }
    : { label: 'Restore', run: () => restoreTile() };
}

/** The tile bar's Open panel placements, in menu order. */
const OPEN_PLACEMENTS = /** @type {const} */ ([
  ['Right', 'right'],
  ['Down', 'down'],
  ['Replace here', 'replace'],
]);

/**
 * A docked service tile bar's "Open panel ▸": Right, Down and Replace here,
 * each a submenu of the rail's panels in no tile (dock-menus.js's
 * untiledPanels — the empty dock slot's list), one disabled row when none is
 * left. Each row opens its panel against the bar's own tile, resolved when
 * the row runs (panel-placement.js's openPanelAtTile); Replace here replaces
 * THIS tile, never the tile used last.
 * @param {string} id  the bar's service panel
 * @returns {import('./panel-context-menu.js').MenuItem}
 */
function openPanelRow(id) {
  const panels = untiledPanels();
  /** @param {'right' | 'down' | 'replace'} place */
  const panelRows = (place) =>
    panels.length
      ? panels.map((p) => ({
          label: p.label,
          ...(p.disabled ? { disabled: true, tip: p.tip } : {}),
          run: () => {
            const tile = dockedTileOf(id);
            if (tile !== null) openPanelAtTile(p.id, tile, place);
          },
        }))
      : [{ label: NO_PANELS_LABEL, disabled: true }];
  return {
    label: 'Open panel',
    submenu: OPEN_PLACEMENTS.map(([label, place]) => ({ label, submenu: panelRows(place) })),
  };
}

/**
 * Show a panel's sidebar row: expand the sidebar when it shows icons only,
 * unfold the row's group, then scroll the row into view and flash it. The
 * Agent pane's row is the Sessions row (TERMINAL_RAIL_ID), in the headless
 * group, which never folds. Both mutators are no-ops on a state already held,
 * so a visible row is only scrolled to and flashed.
 *
 * Reduced motion: the scroll jumps (no smooth behaviour) and the flash's own
 * stylesheet swaps the glow for a still hold (design-system highlight.css).
 * @param {string} panelId
 * @returns {boolean} whether a row was found to reveal
 */
export function revealInSidebar(panelId) {
  if (isSidebarCollapsed()) setSidebarCollapsed(false);
  setGroupFolded(groupOf(panelId), false);
  const entry = getEntry(ctx().getRailEl(), panelId);
  if (!entry) return false;
  entry.scrollIntoView?.({ block: 'nearest' });
  flashElement(entry);
  return true;
}

/**
 * Bring a panel's tile forward: its window when the panel is detached, the
 * Agent pane through openTerminalPanel, any other panel through activateTab.
 * Unlike the row click it never retires an already-surfaced tile — focusing
 * the active tile leaves it where it is. An off-rail panel is focused without
 * the report a member's focus sends.
 * @param {string} panelId
 * @returns {void}
 */
export function focusTile(panelId) {
  if (isDetached(panelId)) {
    focusWindow(panelId);
    return;
  }
  if (panelId === TERMINAL_RAIL_ID) {
    openTerminalPanel();
    return;
  }
  ctx().activateTab(panelId, { userInitiated: ctx().isMember(panelId) });
}

/**
 * @param {string} id
 * @returns {import('./panel-context-menu.js').MenuItem} the row that reveals the panel's sidebar row
 */
function revealRow(id) {
  return { label: 'Reveal in sidebar', run: () => void revealInSidebar(id) };
}

/**
 * The tile ×'s verb as a row: the same close, under the same echo guard
 * (dock-tab.js), so dockview re-activating a surviving tile is not reported
 * as a human focus.
 * @param {string} id
 * @returns {import('./panel-context-menu.js').MenuItem}
 */
function closeTileRow(id) {
  return {
    label: 'Close tile',
    glyph: '×',
    danger: true,
    run: () => {
      withEchoSuppressed(() => closeDockPanel(id));
    },
  };
}

/**
 * Close window, under its divider: the window's own close (pane-window.js
 * closeDetached) — the pane closes, the panel stays on the rail, and no tile
 * comes back.
 * @param {string} id  service panel id, or TERMINAL_RAIL_ID
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
function closeWindowRows(id) {
  return [{ divider: true }, { label: 'Close window', run: () => void closeDetached(id) }];
}

/**
 * @param {string} id
 * @returns {import('./panel-context-menu.js').MenuItem} the row that opens the panel beside the current tile
 */
function newTileRow(id) {
  return { label: 'Open in a new tile', glyph: '⊞', run: () => openPanelBeside(id) };
}

/**
 * The one server-side membership change in the menu.
 * @param {string} id
 * @param {boolean} member
 * @returns {import('./panel-context-menu.js').MenuItem}
 */
function membershipRow(id, member) {
  return member
    ? {
        label: 'Remove from rail',
        glyph: '×',
        danger: true,
        run: () => {
          void removeFromRail(id);
        },
      }
    : {
        label: 'Pin to rail',
        run: () => {
          void setPanelVisibility(id, true);
        },
      };
}

/**
 * "Move to ▸": a submenu of every group but the row's own, in sidebar order,
 * then "New group…" under a divider.
 * @param {string} id
 * @returns {import('./panel-context-menu.js').MenuItem}
 */
function moveToRow(id) {
  const here = groupOf(id);
  const others = loadGroups().groups.filter((group) => group.id !== here);
  return {
    label: 'Move to',
    submenu: [
      ...others.map((group) => ({ label: group.label, run: () => moveToGroup(id, group.id) })),
      ...(others.length ? [{ divider: true }] : []),
      { label: 'New group…', run: () => moveToNewGroup(id) },
    ],
  };
}

/**
 * Make a group directly after the row's own, move the row into it, and open
 * the new group's name for editing, so the operator names the group with the
 * row already in it.
 * @param {string} id
 */
function moveToNewGroup(id) {
  const created = addGroup(NEW_GROUP_LABEL, groupOf(id));
  moveToGroup(id, created);
  startGroupRename(ctx().getRailEl(), created);
}

/**
 * The one row that moves a pane between the workspace and its window
 * (pane-window.js): "Open in new window", inert while the pane cannot leave, or
 * "Reattach" while it is in its window. Every menu takes it from here.
 * @param {string} id  service panel id, or TERMINAL_RAIL_ID
 * @param {Document} [noticeIn]  where a refused Reattach says so
 * @returns {import('./panel-context-menu.js').MenuItem}
 */
export function paneWindowRow(id, noticeIn) {
  return isDetached(id)
    ? { label: 'Reattach', glyph: '⊟', run: () => reattach(id, noticeIn) }
    : {
        label: 'Open in new window',
        glyph: '↗',
        disabled: detachBlocker(id) !== null,
        run: () => detach(id),
      };
}

/**
 * What the Sessions row adds while the Agent pane has no tile, closed or in its
 * window: the verbs the tile's bar would otherwise offer.
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
function tilelessAgentRows() {
  if (isDetached(TERMINAL_RAIL_ID)) {
    return [
      { divider: true },
      paneWindowRow(TERMINAL_RAIL_ID),
      { label: 'Close Agent pane', glyph: '×', danger: true, run: () => closeTerminalTile() },
    ];
  }
  if (isTerminalPanelOpen()) return [];
  return [{ divider: true }, paneWindowRow(TERMINAL_RAIL_ID)];
}

/**
 * The Agent tile's ⋯ menu (tile-menu.js). Section "Agent": Switch to
 * terminal / Switch to chat (the session switcher's own handlers; absent when
 * the deployment offers no terminal), then the rows registered through
 * {@link setServiceMenuExtras} under TERMINAL_RAIL_ID (agent-menu.js: Export
 * conversation in chat, Desktop notifications). Section "Tile":
 * Maximize / Restore ({@link maximizeRow}), Zoom ▸ (pane-zoom.js), {@link paneWindowRow}, Reveal in sidebar (the Sessions row), Close Agent
 * pane, and Restart terminal on the terminal surface — the two destructive
 * rows last. The layout lives on the empty dock slot, not here. The Agent row
 * in the sidebar heads the session tree and has its own menu
 * (buildAgentRowMenuItems, sidebar-artifact-menus.js), which is where New
 * session lives.
 *
 * In chat the tile hosts the operator console, so the PTY verb would act on a
 * surface the operator cannot see and is withheld (the palette never offers
 * it).
 *
 * In the pane's own window (`mode: 'detached'`) the "Agent" section holds the
 * registered rows alone, and "Tile" is Reattach, Zoom ▸, then Close window under a
 * divider. On the no-dockview fallback bar there is no sidebar row to reveal
 * and no dock to maximize in.
 * @param {MenuSurface} [surface]
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
export function buildTerminalMenuItems(surface = {}) {
  const detached = surface.mode === 'detached';
  const extras = serviceMenuExtras.get(TERMINAL_RAIL_ID)?.() ?? [];
  return composeTileMenu({
    thisPanel: detached ? extras : [...surfaceSwitchRows(), ...extras],
    tile: terminalTileRows(surface),
    heading: AGENT_HEADING,
  });
}

/**
 * The way between the chat and the terminal, as the session switcher offers
 * it (session-switcher.js surfaceFooter): the same two handlers, none where
 * the deployment has no terminal.
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
function surfaceSwitchRows() {
  if (!terminalOffered()) return [];
  return isTerminalSurface()
    ? [{ label: 'Switch to chat', run: () => showChatSurface() }]
    : [{ label: 'Switch to terminal', run: () => showTerminalSurface() }];
}

/**
 * @param {MenuSurface} surface
 * @returns {import('./panel-context-menu.js').MenuItem[]} the Agent pane's tile verbs
 */
function terminalTileRows({ mode = 'tile', noticeIn }) {
  const windowRow = paneWindowRow(TERMINAL_RAIL_ID, noticeIn);
  if (mode === 'detached') {
    return [windowRow, zoomMenuRow(TERMINAL_RAIL_ID), ...closeWindowRows(TERMINAL_RAIL_ID)];
  }
  const restart = isTerminalSurface()
    ? [
        // restartTerminal tears the PTY down but does NOT reconnect — pairing
        // it with startTerminal is what keeps the card from being left
        // stranded. The palette's "Restart terminal" runs the identical pair.
        {
          label: 'Restart terminal',
          danger: true,
          run: async () => {
            await restartTerminal();
            startTerminal();
          },
        },
      ]
    : [];
  const docked = mode !== 'fallback';
  return [
    ...(docked ? [maximizeRow(TERMINAL_RAIL_ID)] : []),
    zoomMenuRow(TERMINAL_RAIL_ID),
    windowRow,
    ...(docked ? [revealRow(TERMINAL_RAIL_ID)] : []),
    { label: 'Close Agent pane', glyph: '×', danger: true, run: () => closeTerminalTile() },
    ...restart,
  ];
}
