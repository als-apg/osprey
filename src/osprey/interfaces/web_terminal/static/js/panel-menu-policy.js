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
 * does not; the sidebar row leads with "Focus <label>", which a tile already
 * showing the panel has no use for; and the tile's ⋯ is the one tile menu
 * (tile-menu.js): a "This panel" section (the panel's contributed menu
 * entries, the rows a module owning the panel's content added through
 * {@link setServiceMenuExtras}, and the controls a narrow bar folded) above
 * the "Tile" verbs ({@link buildTileMenuItems}). On the sidebar row the
 * extras follow the verbs under a divider. The Agent is the exception on the
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
import { composeTileMenu } from './tile-menu.js';
import { panelMenuRows } from './tile-header-contrib.js';
import { helpUrl, openHelp } from './panel-help.js';
import { reloadPanel } from './dock-iframe.js';
import { TERMINAL_PANE_LABEL, TERMINAL_RAIL_ID, TERMINAL_RAIL_LABEL } from './panel-catalog.js';
import { setPanelVisibility, setRailOrder } from './panel-commands.js';
import { openPanelBeside, hasDockedTile } from './panel-placement.js';
import {
  openTerminalPanel,
  closeTerminalPanel,
  isTerminalPanelOpen,
  resetDockLayout,
} from './dock-workspace.js';
import { isTerminalSurface } from './agent-surface.js';
import { focusTerminal, restartTerminal, startTerminal } from './terminal.js';
import { buildAgentRowMenuItems, initTreeMenus } from './sidebar-artifact-menus.js';
import { railDragStart, railDragEnd } from './rail-drag.js';
import { getEntry, entryIds, DETACHED_TRAIL_ATTR } from './panel-rail.js';
import { isListOpen, setListOpen, toggleEverySession } from './sidebar.js';
import { showUndoNotice } from './undo-notice.js';
import { closeDetached, detach, detachBlocker, focusWindow, reattach } from './pane-window.js';
import { isDetached } from './dock-apply.js';
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
 * has never answered. A `.failed` row is declined the same way: its verbs
 * would act on a panel that is not up.
 * @param {string} id  @param {number} x  @param {number} y
 * @returns {boolean} true when a menu was opened
 */
export function openRailContextMenu(id, x, y) {
  const entry = getEntry(ctx().getRailEl(), id);
  if (!entry || entry.classList.contains('disabled') || entry.classList.contains('failed')) {
    return false;
  }
  // The Agent row heads the session tree, so its menu is the tree's (New
  // session, Expand/Collapse all), not the Agent tile header's pane verbs.
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
 * tile to focus, move or close from there. `noticeIn` is where a refused Reattach
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
 * the rows can follow the panel's state. In the tile ⋯ they are the "This
 * panel" section, after the panel's contributed menu entries; on the sidebar
 * row they follow the verbs under a divider. Registering again replaces the
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
 * indistinguishable downstream from any other path to the verb. Its tile
 * verbs come from the row helpers the tile ⋯ uses too
 * ({@link buildTileMenuItems}), and it is followed by the rows registered
 * through {@link setServiceMenuExtras}.
 *
 * One row moves the panel between the workspace and its window
 * ({@link paneWindowRow}). A panel in its window offers no "Open in a new tile",
 * because the panel has no tile while its window is open.
 *
 * A member's rows also move its sidebar row between the sidebar's groups:
 * "Move to <group>" for every other group, then "Move to new group…".
 *
 * @param {string} id
 * @param {MenuSurface} [surface]
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
export function buildServiceMenuItems(id, { mode = 'tile', noticeIn } = {}) {
  const inWindow = mode === 'detached';
  const label = ctx().labelOf(id);
  const member = ctx().isMember(id);
  const detached = isDetached(id);
  const extras = serviceMenuExtras.get(id)?.() ?? [];
  return [
    // Focus is the ACTIVATION half of the row click only. A click on the row
    // whose tile is already surfaced retires that tile; a row that says
    // "Focus" and made the panel disappear would say the opposite of what it
    // does, so on the already-active panel this row is deliberately a no-op.
    // Off the rail it is focused without the report a member's focus sends.
    ...(inWindow
      ? []
      : [{ label: `Focus ${label}`, run: () => ctx().activateTab(id, { userInitiated: member }) }]),
    ...(detached ? [] : [newTileRow(id)]),
    paneWindowRow(id, noticeIn),
    ...(member && !inWindow ? groupMoveItems(id) : []),
    { divider: true },
    membershipRow(id, member),
    ...(extras.length ? [{ divider: true }, ...extras] : []),
  ];
}

/**
 * A service panel's TILE ⋯ menu (tile-menu.js), the same for the ⋯ button
 * and a right-click on the bar: "This panel" — the panel's contributed menu
 * entries, the rows registered through {@link setServiceMenuExtras}, and the
 * controls the narrow bar folded — then "Tile": open beside, the group moves,
 * {@link paneWindowRow}, Reload, Help (only when the panel has a docs page,
 * panel-help.js), the rail membership row and Reset layout.
 *
 * In the pane's own window (`mode: 'detached'`) the Tile section is
 * Reattach, Reload, Help and the membership row: nothing there acts on the
 * dock.
 *
 * Only a tile can belong to a panel that is NOT a rail member — a tile an
 * agent's focus opened off the rail. Its rows keep it there: the move to a
 * new tile acts on the tile alone, the membership row is "Pin to rail", which
 * adds it where any new member goes, and there is no sidebar row to move.
 * @param {string} id
 * @param {MenuSurface} [surface]
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
export function buildTileMenuItems(id, { mode = 'tile', noticeIn } = {}) {
  const member = ctx().isMember(id);
  const thisPanel = [...panelMenuRows(id), ...(serviceMenuExtras.get(id)?.() ?? [])];
  const panelVerbs = [
    paneWindowRow(id, noticeIn),
    { label: 'Reload', run: () => void reloadPanel(id) },
    ...(helpUrl(id) === null ? [] : [{ label: 'Help', run: () => void openHelp(id) }]),
  ];
  if (mode === 'detached') {
    const tile = [...panelVerbs, { divider: true }, membershipRow(id, member)];
    return composeTileMenu({ thisPanel, tile });
  }
  const tile = [
    ...(isDetached(id) ? [] : [newTileRow(id)]),
    ...(member ? groupMoveItems(id) : []),
    { divider: true },
    ...panelVerbs,
    { divider: true },
    membershipRow(id, member),
    { label: 'Reset layout', run: () => resetDockLayout() },
  ];
  return composeTileMenu({ thisPanel, tile });
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
 * The rows that move a member's sidebar row to another group, under a
 * divider: "Move to <group>" for every group but its own, in sidebar order,
 * then "Move to new group…".
 * @param {string} id
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
function groupMoveItems(id) {
  const here = groupOf(id);
  const others = loadGroups().groups.filter((group) => group.id !== here);
  return [
    { divider: true },
    ...others.map((group) => ({
      label: `Move to ${group.label}`,
      run: () => moveToGroup(id, group.id),
    })),
    { label: 'Move to new group…', run: () => moveToNewGroup(id) },
  ];
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
 * The Agent tile's ⋯ menu (tile-menu.js): "This panel", the rows registered
 * through {@link setServiceMenuExtras} under TERMINAL_RAIL_ID (agent-menu.js:
 * Export conversation in chat, Desktop notifications), then "Tile":
 * {@link paneWindowRow}, Restart terminal on the terminal, Reset layout,
 * Close Agent pane. The Agent row in the sidebar
 * heads the session tree and has its own menu (buildAgentRowMenuItems,
 * sidebar-artifact-menus.js), which is where New session lives.
 *
 * In chat the tile hosts the operator console, so the PTY verb would act on a
 * surface the operator cannot see and is withheld (the palette never offers
 * it). The layout rows are the same on both surfaces.
 * @param {MenuSurface} [surface]
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
export function buildTerminalMenuItems(surface = {}) {
  return composeTileMenu({
    thisPanel: serviceMenuExtras.get(TERMINAL_RAIL_ID)?.() ?? [],
    tile: terminalTileRows(surface),
  });
}

/**
 * @param {MenuSurface} surface
 * @returns {import('./panel-context-menu.js').MenuItem[]} the Agent pane's tile verbs
 */
function terminalTileRows({ mode = 'tile', noticeIn }) {
  const layout =
    mode === 'detached'
      ? []
      : [
          { divider: true },
          { label: 'Reset layout', run: () => resetDockLayout() },
          { label: 'Close Agent pane', glyph: '×', danger: true, run: () => closeTerminalTile() },
        ];
  if (!isTerminalSurface()) return [paneWindowRow(TERMINAL_RAIL_ID, noticeIn), ...layout];
  return [
    paneWindowRow(TERMINAL_RAIL_ID, noticeIn),
    // restartTerminal tears the PTY down but does NOT reconnect — pairing it
    // with startTerminal is what keeps the card from being left stranded. The
    // palette's "Restart terminal" runs the identical pair.
    {
      label: 'Restart terminal',
      run: async () => {
        await restartTerminal();
        startTerminal();
      },
    },
    ...layout,
  ];
}
