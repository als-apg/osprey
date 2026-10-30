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
 * gets, whether it gets one at all, which rows the current ui mode allows, and
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
 * A panel's sidebar row and its tile header offer identical rows, built by the
 * same {@link buildServiceMenuItems} / {@link buildTerminalMenuItems}, so the
 * two can never drift apart. They differ in exactly one place, deliberately:
 * the sidebar applies an enabled gate (see openRailContextMenu) and the tile
 * header does not. A module that owns a panel's content can add rows to both
 * through {@link setServiceMenuExtras}.
 *
 * DECLINE BY EMPTY LIST
 * ---------------------
 * Both openers return a boolean their caller turns into `preventDefault()`:
 * true when a menu opened, false to leave the event alone so the browser's own
 * menu still shows. A surface with NO rows to offer declines.
 */

import { openContextMenu } from './panel-context-menu.js';
import { TERMINAL_RAIL_ID, TERMINAL_RAIL_LABEL } from './panel-catalog.js';
import { setPanelVisibility, setRailOrder } from './panel-commands.js';
import { openPanelBeside, hasDockedTile } from './panel-placement.js';
import {
  openTerminalPanel,
  closeTerminalPanel,
  isTerminalPanelOpen,
  resetDockLayout,
} from './dock-workspace.js';
import { focusTerminal, restartTerminal, startTerminal } from './terminal.js';
import { startNewSession } from './sessions.js';
import { railDragStart, railDragEnd } from './rail-drag.js';
import { getEntry, entryIds } from './panel-rail.js';
import { showUndoNotice } from './undo-notice.js';

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
 * @property {(id: string) => string | null} getPanelStandaloneUrl - null until the config fetch resolves
 * @property {(id: string) => void} popoutPanel - open the standalone url in a new browser tab
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
 * in the main tile, taking it over from its current occupant (one panel per
 * tile). The take-over happens in the dock adapter's placement logic and is
 * purely local — the evicted panel keeps its row. The Agent row routes to the
 * dock instead (its open/closed state is layout state, not membership — see
 * TERMINAL_RAIL_ID).
 */
export function railOptions() {
  return {
    onActivate: (/** @type {string} */ id) => {
      if (id === TERMINAL_RAIL_ID) {
        showAgentPane();
        return;
      }
      // A failed sidecar's row has one verb: start it again.
      if (isFailed(id)) {
        ctx().retryStart(id);
        return;
      }
      // Clicking the row whose tile is ALREADY surfaced retires that tile — a
      // toggle shortcut equivalent to the tile header's own "×". It stays a
      // LOCAL layout change: membership survives, so a second click brings
      // the tile back. A parent row (Artifacts) is the exception: the same
      // click opens its list, which is what the operator reached for, so it
      // only ever brings its tile forward.
      if (ctx().isMember(id)) {
        if (ctx().getActiveTabId() === id && !isParentRow(id)) {
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
    // The same drag, dropped among the Tools rows instead of the workspace,
    // reorders them. The Workspace rows keep their places.
    onReorder: (/** @type {string} */ id, /** @type {string | null} */ before) => {
      void setRailOrder(id, before);
    },
    reorderable: (/** @type {string} */ id) => id !== TERMINAL_RAIL_ID,
  };
}

/**
 * The Agent row's click: open the Agent pane when it is closed, or bring the
 * open one forward, then put the caret where the operator types — the chat
 * composer in Simple, the terminal in Expert.
 */
function showAgentPane() {
  openTerminalPanel();
  if (!isTerminalPanelOpen()) return;
  if (isSimpleMode()) {
    /** @type {HTMLElement | null} */ (
      document.querySelector('#operator-container textarea')
    )?.focus();
  } else focusTerminal();
}

/**
 * Whether a panel's row discloses a list (`.parent`, the Artifacts row).
 * @param {string} id
 * @returns {boolean}
 */
function isParentRow(id) {
  return getEntry(ctx().getRailEl(), id)?.classList.contains('parent') ?? false;
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
// Agent row gets none: its verb set changes with the ui view (see
// buildTerminalMenuItems), so one fixed promise would misinform one of them.
export const RAIL_MENU_HINT = 'right-click for actions';

/**
 * Simple mode replaces the xterm card with the operator console — the one fact
 * the terminal menu below turns on. The workspace itself is as editable there
 * as in Expert, so no service-panel verb reads the mode.
 * @returns {boolean}
 */
function isSimpleMode() {
  return document.documentElement.getAttribute('data-ui-mode') === 'simple';
}

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
  // The row is the anchor: it scopes the menu's scroll dismissal, so the menu
  // survives a panel scrolling its own content underneath it.
  return openSurfaceMenu(id, x, y, entry);
}

/**
 * The tile-header counterpart of openRailContextMenu, registered into
 * dock-tab.js at init. Same verb sets, so a panel's two surfaces cannot offer
 * different actions, and the same decline-by-empty-list contract: a surface
 * with no rows to offer reports false and keeps the browser's native menu.
 *
 * The sidebar's enabled gate has no counterpart here on purpose. It exists so
 * a panel that has never answered cannot be acted on from an offline row; a
 * tile header only exists for a panel already occupying a tile, and its verbs
 * (focus it, pop it out, drop it from the rail) stay meaningful even if health
 * polling has since gone quiet.
 *
 * @param {string} id  service panel id, or TERMINAL_RAIL_ID
 * @param {{x: number, y: number, anchorEl: HTMLElement}} opts
 * @returns {boolean} true when a menu was opened
 */
export function openTileContextMenu(id, opts) {
  // The header bar is the anchor — the menu rides out the panel's own scrolls,
  // including the terminal streaming output underneath it.
  return openSurfaceMenu(id, opts.x, opts.y, opts.anchorEl);
}

/**
 * The shared tail of both surface openers: build the id's verb rows, decline
 * by empty list, and open the popover. Items, label and the decline signal
 * live in this one place, so the two-surfaces-one-verb-set contract is
 * mechanical rather than maintained in parallel.
 * @param {string} id  service panel id, or TERMINAL_RAIL_ID
 * @param {number} x  @param {number} y
 * @param {HTMLElement} anchorEl  scopes the menu's scroll dismissal
 * @returns {boolean} true when a menu was opened
 */
function openSurfaceMenu(id, x, y, anchorEl) {
  const items = id === TERMINAL_RAIL_ID ? buildTerminalMenuItems() : buildServiceMenuItems(id);
  // No rows is a decline, never an empty popover.
  if (!items.length) return false;
  const label = id === TERMINAL_RAIL_ID ? TERMINAL_RAIL_LABEL : ctx().labelOf(id);
  openContextMenu({ x, y, anchorEl, ariaLabel: `${label} actions`, items });
  return true;
}

/**
 * Extra rows per panel, registered by the module that owns the panel's content.
 * @type {Map<string, () => import('./panel-context-menu.js').MenuItem[]>}
 */
const serviceMenuExtras = new Map();

/**
 * Add rows of a panel's own to its context menus (sidebar row and tile
 * header): `fn()` is asked at every open, so the rows can follow the panel's
 * state, and its rows are appended under a divider after the panel's verbs.
 * Registering again replaces the earlier rows; `null` removes them.
 * @param {string} panelId
 * @param {(() => import('./panel-context-menu.js').MenuItem[]) | null} fn
 */
export function setServiceMenuExtras(panelId, fn) {
  if (fn) serviceMenuExtras.set(panelId, fn);
  else serviceMenuExtras.delete(panelId);
}

/**
 * A service panel's menu rows, bound to the SAME closures the sidebar click,
 * the row drag and the agent MCP tools use — a menu pick is indistinguishable
 * downstream from any other path to the verb. Shared by the sidebar row and
 * the panel's tile header so the two surfaces cannot drift apart, and followed
 * by the rows registered through {@link setServiceMenuExtras}.
 *
 * Policy carried over from the retired corner glyphs: "Open in a new window"
 * stays inert until the panel's standalone URL resolves. The rows are the same
 * in both ui views.
 *
 * Only a tile header can belong to a panel that is NOT a rail member — a tile
 * an agent's focus opened off the rail. Its rows keep it
 * there: Focus and the move to a new tile act on the tile alone, and the
 * membership row is "Pin to rail", which adds it where any new member goes.
 * @param {string} id
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
export function buildServiceMenuItems(id) {
  const label = ctx().labelOf(id);
  const member = ctx().isMember(id);
  const extras = serviceMenuExtras.get(id)?.() ?? [];
  return [
    // Focus is the ACTIVATION half of the row click only. A click on the row
    // whose tile is already surfaced retires that tile; a row that says
    // "Focus" and made the panel disappear would say the opposite of what it
    // does, so on the already-active panel this row is deliberately a no-op.
    // Off the rail it is focused without the report a member's focus sends.
    { label: `Focus ${label}`, run: () => ctx().activateTab(id, { userInitiated: member }) },
    { label: 'Open in a new tile', glyph: '⊞', run: () => openPanelBeside(id) },
    {
      label: 'Open in a new window',
      glyph: '↗',
      disabled: ctx().getPanelStandaloneUrl(id) === null,
      run: () => ctx().popoutPanel(id),
    },
    { divider: true },
    // The one server-side membership change in the menu.
    member
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
        },
    ...(extras.length ? [{ divider: true }, ...extras] : []),
  ];
}

/**
 * The Agent pane's menu rows — the Agent row and the pane's tile header,
 * which share one verb set.
 *
 * In simple mode the tile hosts the operator console instead of the xterm
 * card, so the two PTY verbs would act on a surface the operator cannot see
 * and are withheld (the palette drops the same actions there, for the same
 * reason). The layout rows are the same in both views: "Reset layout" puts
 * back the CURRENT view's default arrangement, and the pane is as closable
 * there as in Expert.
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
export function buildTerminalMenuItems() {
  const layout = [
    { label: 'Reset layout', run: () => resetDockLayout() },
    { label: 'Close Agent pane', glyph: '×', danger: true, run: () => closeTerminalTile() },
  ];
  if (isSimpleMode()) return layout;
  return [
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
    {
      label: 'New session',
      run: () => {
        startNewSession();
      },
    },
    { divider: true },
    ...layout,
  ];
}
