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
 * A panel's sidebar row and its tile bar (the ⋯ button and a right-click)
 * build their rows from the same row helpers and bind the same closures, so
 * a verb cannot drift between them. They differ deliberately: the sidebar
 * applies an enabled gate (see openRailContextMenu) and the tile does not;
 * the sidebar row leads with "Focus <label>", which a tile that is already
 * showing the panel has no use for, and keeps the short verb list its
 * redesign owns; the tile's ⋯ is the standard's full menu (tile-menu.js) —
 * a "This panel" section (the panel's contributed menu entries, the rows a
 * module owning the panel's content added through
 * {@link setServiceMenuExtras}, and the controls a narrow bar folded) above
 * the "Tile" verbs, which add Reload, Copy link, Open in browser tab, Help
 * and Reset layout to the row's. On the sidebar row the extras follow the
 * verbs under a divider.
 * The Agent is the exception on the sidebar side: its row heads the session
 * tree and gets the tree's menu (buildAgentRowMenuItems,
 * sidebar-artifact-menus.js); its tile gets the pane verbs
 * ({@link buildTerminalTileMenuItems}).
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
import { foldedControlRows } from './tile-bar-controls.js';
import { copyPanelLink, openPanelInTab, panelLink, reloadPanel } from './panel-links.js';
import { helpUrl, openHelp } from './panel-help.js';
import { TERMINAL_RAIL_ID, TERMINAL_RAIL_LABEL } from './panel-catalog.js';
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
import { getEntry, entryIds } from './panel-rail.js';
import { isListOpen, setListOpen, toggleEverySession } from './sidebar.js';
import { showUndoNotice } from './undo-notice.js';
import { bringBackPanel, canPopOut, focusPopout, isPoppedOut, popOutPanel } from './popout.js';
import { bringBackAttached, canAttach, isAttached, popOutAttached } from './attached-popout.js';
import { isPanelDocked } from './dock-iframe.js';
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
      if (id === TERMINAL_RAIL_ID) {
        activateAgentRow(event);
        return;
      }
      // A failed sidecar's row has one verb: start it again.
      if (isFailed(id)) {
        ctx().retryStart(id);
        return;
      }
      // A popped-out panel's row brings its window to the front; the tile
      // stays concealed until the window closes.
      if (isPoppedOut(id)) {
        focusPopout(id);
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
 * The Agent row's click. With the Agent pane closed it opens the pane and
 * unfolds the session list; with the pane open it folds or unfolds the list
 * and leaves the pane as it is — the pane closes only from its own header.
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
function showAgentPane() {
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
 * grid is not something the rail can put back.
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
  const hadTile = hasDockedTile(id) || ctx().getActiveTabId() === id;
  if (!(await setPanelVisibility(id, false))) {
    showUndoNotice({ message: `Could not remove ${label} from the rail` });
    return;
  }
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
      items: buildAgentRowMenuItems(),
    });
    return true;
  }
  // The row is the anchor: it scopes the menu's scroll dismissal, so the menu
  // survives a panel scrolling its own content underneath it.
  return openSurfaceMenu(id, 'rail', { x, y, anchorEl: entry });
}

/**
 * The tile-bar counterpart of openRailContextMenu, registered into
 * dock-tab.js at init and reached from the bar's ⋯ button and from a
 * right-click on the bar alike. The same tile verbs as the sidebar row, so a
 * panel's two surfaces cannot offer different actions, plus the tile's own
 * "This panel" section; the same decline-by-empty-list contract: a surface
 * with no rows to offer reports false and keeps the browser's native menu.
 *
 * The sidebar's enabled gate has no counterpart here on purpose. It exists so
 * a panel that has never answered cannot be acted on from an offline row; a
 * tile bar only exists for a panel already occupying a tile, and its verbs
 * (pop it out, reload it, drop it from the rail) stay meaningful even if
 * health polling has since gone quiet.
 *
 * @param {string} id  service panel id, or TERMINAL_RAIL_ID
 * @param {import('./dock-tab.js').TileMenuRequest} opts
 * @returns {boolean} true when a menu was opened
 */
export function openTileContextMenu(id, opts) {
  // The header bar is the anchor — the menu rides out the panel's own scrolls,
  // including the terminal streaming output underneath it.
  return openSurfaceMenu(id, 'tile', opts);
}

/**
 * The shared tail of both surface openers: build the id's rows for the
 * surface, decline by empty list, and open the popover. Items, label and the
 * decline signal live in this one place, so the two-surfaces-one-verb-set
 * contract is mechanical rather than maintained in parallel. The Agent
 * reaches here from its tile only (openRailContextMenu gives its row the
 * session tree's menu).
 * @param {string} id  service panel id, or TERMINAL_RAIL_ID
 * @param {'rail' | 'tile'} surface
 * @param {import('./dock-tab.js').TileMenuRequest} opts  position, anchor and
 *   (from the ⋯ button) the trigger the menu hangs under
 * @returns {boolean} true when a menu was opened
 */
function openSurfaceMenu(id, surface, opts) {
  const agent = id === TERMINAL_RAIL_ID;
  const items = agent
    ? buildTerminalTileMenuItems()
    : surface === 'tile'
      ? buildTileMenuItems(id)
      : buildServiceMenuItems(id);
  // No rows is a decline, never an empty popover.
  if (!items.length) return false;
  const label = agent ? TERMINAL_RAIL_LABEL : ctx().labelOf(id);
  openContextMenu({ ...opts, ariaLabel: `${label} actions`, items });
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
 * row they follow the verbs under a divider. The Agent pane's rows are
 * registered under TERMINAL_RAIL_ID. Registering again replaces the earlier
 * rows; `null` removes them.
 *
 * Row shape: the design-system menu's `MenuItem` — `{label, run, checked?,
 * disabled?, danger?, glyph?}`, or `{divider: true}` — plus `separator:
 * true` on a row for a divider before it. Destructive rows (`danger`) are
 * listed last in the tile's section whatever their order here.
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
 * indistinguishable downstream from any other path to the verb. Its rows
 * are the tile verbs the tile ⋯ also shows ({@link buildTileMenuItems}), from
 * the same row helpers, led by "Focus <label>" and followed by the rows
 * registered through {@link setServiceMenuExtras}.
 *
 * A panel with a tile offers "Pop out", the tile button's attached window
 * (attached-popout.js), and "Reattach" while it is out. "Open in new
 * window" stays inert until the panel can open its standalone page
 * (popout.js): its URL has resolved, and it is not an opaque-origin panel. A
 * panel in that window offers "Bring back" in its place and no "Open in a
 * new tile", because the panel has no tile while its window is open.
 * @param {string} id
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
export function buildServiceMenuItems(id) {
  const label = ctx().labelOf(id);
  const member = ctx().isMember(id);
  const popped = isPoppedOut(id);
  const extras = serviceMenuExtras.get(id)?.() ?? [];
  return [
    // Focus is the ACTIVATION half of the row click only. A click on the row
    // whose tile is already surfaced retires that tile; a row that says
    // "Focus" and made the panel disappear would say the opposite of what it
    // does, so on the already-active panel this row is deliberately a no-op.
    // Off the rail it is focused without the report a member's focus sends.
    { label: `Focus ${label}`, run: () => ctx().activateTab(id, { userInitiated: member }) },
    ...(popped ? [] : [newTileRow(id)]),
    ...windowRows(id, popped),
    ...(member ? [{ divider: true }, ...groupMoveItems(id)] : []),
    { divider: true },
    membershipRow(id, member),
    ...(extras.length ? [{ divider: true }, ...extras] : []),
  ];
}

/**
 * A service panel's TILE ⋯ menu (tile-menu.js): "This panel" — the panel's
 * contributed menu entries, the rows registered through
 * {@link setServiceMenuExtras}, and the controls the narrow bar folded —
 * then "Tile", the verbs in the order the standard fixes them.
 *
 * "Reload" reloads the panel's iframe; "Copy link" and "Open in browser tab"
 * act on the panel's deep link — its standalone URL carrying its current
 * state (panel-links.js) — and are inert until the panel has a frame to read
 * it from; "Help" opens the panel's documentation page and is absent when
 * the deployment or the panel has none (panel-help.js).
 *
 * Only a tile can belong to a panel that is NOT a rail member — a tile an
 * agent's focus opened off the rail. Its rows keep it there: the move to a
 * new tile acts on the tile alone, the membership row is "Pin to rail",
 * which adds it where any new member goes, and there is no sidebar row to
 * move.
 * @param {string} id
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
export function buildTileMenuItems(id) {
  const member = ctx().isMember(id);
  const popped = isPoppedOut(id);
  const link = panelLink(id);
  const thisPanel = [
    ...panelMenuRows(id),
    ...(serviceMenuExtras.get(id)?.() ?? []),
    ...foldedControlRows(id),
  ];
  const tile = [
    ...(popped ? [] : [newTileRow(id)]),
    ...(member ? groupMoveItems(id) : []),
    { divider: true },
    ...windowRows(id, popped),
    {
      label: 'Reload',
      run: () => {
        reloadPanel(id);
      },
    },
    {
      label: 'Copy link',
      glyph: '⎘',
      disabled: link === null,
      run: () => {
        void copyPanelLink(id);
      },
    },
    {
      label: 'Open in browser tab',
      disabled: link === null,
      run: () => {
        openPanelInTab(id);
      },
    },
    ...(helpUrl(id) === null
      ? []
      : [
          {
            label: 'Help',
            run: () => {
              openHelp(id);
            },
          },
        ]),
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
 * The window verbs: "Pop out" (or "Reattach"), the attached window of the
 * tile's own button, offered while the panel has a tile; then "Open in new
 * window", the panel's standalone page (popout.js), inert until its URL has
 * resolved and never for an opaque-origin panel. A panel in that window
 * offers "Bring back" in its place.
 * @param {string} id
 * @param {boolean} popped
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
function windowRows(id, popped) {
  return [
    ...(!popped && (isPanelDocked(id) || isAttached(id)) ? [attachedRow(id)] : []),
    popped
      ? { label: 'Bring back', glyph: '⊟', run: () => bringBackPanel(id) }
      : {
          label: 'Open in new window',
          glyph: '↗',
          // False until the config resolves a URL, and for an opaque-origin panel.
          disabled: !canPopOut(id),
          run: () => popOutPanel(id),
        },
  ];
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
 * The rows that move a member's sidebar row to another group: "Move to
 * <group>" for every group but its own, in sidebar order, then "Move to new
 * group…".
 * @param {string} id
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
function groupMoveItems(id) {
  const here = groupOf(id);
  const others = loadGroups().groups.filter((group) => group.id !== here);
  return [
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
 * The attached pop-out's row (the tile header's Pop out button,
 * attached-popout.js): Pop out, or Reattach while the content is in its
 * window.
 * @param {string} id  service panel id, or TERMINAL_RAIL_ID
 * @returns {import('./panel-context-menu.js').MenuItem}
 */
function attachedRow(id) {
  return isAttached(id)
    ? { label: 'Reattach', glyph: '⊟', run: () => bringBackAttached(id) }
    : { label: 'Pop out', glyph: '⧉', disabled: !canAttach(id), run: () => popOutAttached(id) };
}

/**
 * The Agent pane's TILE ⋯ menu: "This panel" — the rows the chat registered
 * through {@link setServiceMenuExtras} under TERMINAL_RAIL_ID — then "Tile",
 * the Agent pane's own verbs. The Agent row in the sidebar heads the session
 * tree and has its own menu (buildAgentRowMenuItems,
 * sidebar-artifact-menus.js), which is where New session lives.
 *
 * In chat the tile hosts the operator console, so the PTY verb would act on a
 * surface the operator cannot see and is withheld (the palette never offers
 * it). The layout rows are the same on both surfaces.
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
export function buildTerminalTileMenuItems() {
  return composeTileMenu({
    thisPanel: serviceMenuExtras.get(TERMINAL_RAIL_ID)?.() ?? [],
    tile: terminalTileRows(),
  });
}

/** @returns {import('./panel-context-menu.js').MenuItem[]} the Agent pane's tile verbs */
function terminalTileRows() {
  const layout = [
    { divider: true },
    { label: 'Reset layout', run: () => resetDockLayout() },
    { label: 'Close Agent pane', glyph: '×', danger: true, run: () => closeTerminalTile() },
  ];
  if (!isTerminalSurface()) {
    return [attachedRow(TERMINAL_RAIL_ID), ...layout];
  }
  return [
    attachedRow(TERMINAL_RAIL_ID),
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
