// @ts-check
/* OSPREY Web Terminal — Panel Menu Policy
 *
 * WHAT a gesture on a panel surface does. The panel state machine now splits
 * three ways — panel-manager.js owns the core (tile occupancy, iframe
 * lifecycle, the active-panel policy), panel-lifecycle.js owns how a panel
 * comes to exist and stay alive, and panel-sse.js owns the server→client
 * frame dispatcher — and panel-placement.js owns tile geometry; this module
 * owns the policy layer in between — the closures a rail entry is rendered
 * with, and the context menus a rail entry or a tile header opens.
 *
 * It is the counterpart of panel-context-menu.js, which is purely
 * presentational (popover DOM, dismissal, focus bookkeeping) and never knows
 * what a verb does. Every decision lives here instead: which menu a surface
 * gets, whether it gets one at all, which rows the current ui mode allows, and
 * which closure each row runs. The rows bind the SAME closures the rail
 * corners, the rail drag and the agent MCP tools use, so a menu pick is
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
 * A panel's rail entry and its tile header offer identical rows, built by the
 * same {@link buildServiceMenuItems} / {@link buildTerminalMenuItems}, so the
 * two can never drift apart. They differ in exactly one place, deliberately:
 * the rail applies an enabled gate (see openRailContextMenu) and the tile
 * header does not.
 *
 * DECLINE BY EMPTY LIST
 * ---------------------
 * Both openers return a boolean their caller turns into `preventDefault()`:
 * true when a menu opened, false to leave the event alone so the browser's own
 * menu still shows. A builder returning NO rows is that decline — the case is
 * the terminal in simple mode, where the xterm card is replaced by the operator
 * console and every terminal verb would act on a surface the operator cannot
 * see.
 */

import { openContextMenu } from './panel-context-menu.js';
import { TERMINAL_RAIL_ID, TERMINAL_RAIL_LABEL } from './panel-catalog.js';
import { setPanelVisibility } from './panel-commands.js';
import { openPanelBeside } from './panel-placement.js';
import { openTerminalPanel, closeTerminalPanel } from './dock-workspace.js';
import { restartTerminal, startTerminal } from './terminal.js';
import { startNewSession } from './sessions.js';
import { railDragStart, railDragEnd } from './rail-drag.js';
import { getEntry } from './panel-rail.js';

/**
 * The panel-manager state and actions this module's closures act through. Every
 * entry is a live closure over panel-manager's private state — never a
 * snapshot, so a menu built now reads the rail and health of this moment.
 * @typedef {object} MenuPolicyDeps
 * @property {() => HTMLElement} getRailEl - the rail nav the entries live in
 * @property {(id: string) => boolean} isMember - the panel has rail membership
 * @property {() => string | null} getActiveTabId - the locally surfaced panel id
 * @property {(id: string, options?: {userInitiated?: boolean, auto?: boolean}) => void} activateTab
 * @property {(id: string) => void} showPanel - reveal a NON-member through the visibility POST path
 * @property {(id: string) => void} retireTile - vacate a surfaced tile LOCALLY (membership survives)
 * @property {(id: string) => string} labelOf - the panel's catalog label
 * @property {(id: string) => string | null} getPanelStandaloneUrl - null until the config fetch resolves
 * @property {(id: string) => void} popoutPanel - open the standalone url in a new browser tab
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
 * Interaction closures handed to every rail render/append call. Routing
 * activation and close through here keeps a human click/"×" and an agent MCP
 * call indistinguishable downstream (both hit activateTab /
 * setPanelVisibility).
 *
 * Every entry is a member, so a click is always an activation: show this panel
 * in the main tile, taking it over from its current occupant (one panel per
 * tile; the rail IS the workspace's tab system). The take-over happens in the
 * dock adapter's placement logic and is purely local — the evicted panel keeps
 * its entry. "×" removes the panel from the rail (membership, a server
 * change). The terminal entry routes to the dock instead (its open/closed
 * state is layout state, not membership — see TERMINAL_RAIL_ID).
 */
export function railOptions() {
  return {
    onActivate: (/** @type {string} */ id) => {
      if (id === TERMINAL_RAIL_ID) { openTerminalPanel(); return; }
      // Clicking the entry whose tile is ALREADY surfaced retires that tile —
      // a toggle shortcut equivalent to the tile header's own "×". It stays a
      // LOCAL layout change: rail membership survives, so a second click
      // brings the tile back. Membership removal remains the separate "×"
      // corner.
      if (ctx().isMember(id)) {
        if (ctx().getActiveTabId() === id) { ctx().retireTile(id); return; }
        ctx().activateTab(id, { userInitiated: true });
      } else ctx().showPanel(id);
    },
    onClose: (/** @type {string} */ id) => {
      if (id === TERMINAL_RAIL_ID) { closeTerminalPanel(); return; }
      setPanelVisibility(id, false);
    },
    // Right-click — or the menu key / Shift+F10, which the rail forwards
    // through this same closure — opens the entry's verbs in words
    // (panel-context-menu.js). openRailContextMenu owns the whole policy
    // (which menu, and whether there is one at all) and reports back with the
    // boolean the rail uses to decide whether to suppress the browser's menu.
    onContextMenu: (/** @type {string} */ id, /** @type {MouseEvent} */ e) =>
      openRailContextMenu(id, e.clientX, e.clientY),
    // Rail drag source: the terminal entry never drags (its tile moves by its
    // own header bar); everything else defers to rail-drag's policy (simple
    // mode and fallback mode cancel there).
    onDragStart: (/** @type {string} */ id, /** @type {DataTransfer | null} */ dt) =>
      id === TERMINAL_RAIL_ID ? false : railDragStart(id, dt),
    onDragEnd: () => railDragEnd(),
  };
}

// Tooltip suffix advertising the context menu on every service entry. The
// terminal entry gets none: its menu exists only in expert mode, so a tooltip
// promising actions would misinform every simple-mode operator.
export const RAIL_MENU_HINT = 'right-click for actions';

/**
 * Simple mode locks the workspace to a single service tile and replaces the
 * xterm card with the operator console — the two facts every menu-policy gate
 * below turns on.
 * @returns {boolean}
 */
function isSimpleMode() {
  return document.documentElement.getAttribute('data-ui-mode') === 'simple';
}

/**
 * Open a rail entry's context menu at a viewport position, and report whether
 * one was opened — the boolean the rail uses to decide whether to suppress the
 * browser's native menu, on the right-click and keyboard routes alike (FR5).
 *
 * The enabled gate lives HERE rather than in the rail CSS. `pointer-events:
 * none` on a disabled entry stops a right-click landing on the entry itself,
 * but it does not stop a menu request fired from the KEYBOARD on that focused
 * button, and it does not cover the offline entry's still-live "×" corner,
 * which stays interactive by design. Both routes would otherwise reach a menu
 * whose verbs act on a panel that has never answered.
 *
 * @param {string} id  @param {number} x  @param {number} y
 * @returns {boolean} true when a menu was opened
 */
export function openRailContextMenu(id, x, y) {
  const entry = getEntry(ctx().getRailEl(), id);
  if (!entry || entry.classList.contains('disabled')) return false;
  // The entry is the anchor: it scopes the menu's scroll dismissal, so the
  // menu survives a panel scrolling its own content underneath it.
  return openSurfaceMenu(id, x, y, entry);
}

/**
 * The tile-header counterpart of openRailContextMenu, registered into
 * dock-tab.js at init. Same verb sets, so a panel's two surfaces cannot offer
 * different actions, and the same decline-by-empty-list contract: the terminal
 * header in simple mode reports false and keeps the browser's native menu.
 *
 * The rail's enabled gate has no counterpart here on purpose. It exists so a
 * panel that has never answered cannot be acted on from an offline entry; a
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
  // No rows is a decline, never an empty popover — simple mode's terminal.
  if (!items.length) return false;
  const label = id === TERMINAL_RAIL_ID ? TERMINAL_RAIL_LABEL : ctx().labelOf(id);
  openContextMenu({ x, y, anchorEl, ariaLabel: `${label} actions`, items });
  return true;
}

/**
 * A service panel's menu rows, bound to the SAME closures the rail, the rail
 * drag and the agent MCP tools use — a menu pick is indistinguishable
 * downstream from any other path to the verb. Shared by the rail entry and the
 * panel's tile header so the two surfaces cannot drift apart.
 *
 * Policy carried over from the retired corner glyphs: "Open in a new tile" is
 * not a simple-mode gesture (the layout is locked to one service tile), and
 * "Open in a new window" stays inert until the panel's standalone URL
 * resolves.
 * @param {string} id
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
export function buildServiceMenuItems(id) {
  const label = ctx().labelOf(id);
  return [
    // Focus is the ACTIVATION half of the entry click only. A click on the
    // entry whose tile is already surfaced retires that tile; a row that says
    // "Focus" and made the panel disappear would say the opposite of what it
    // does, so on the already-active panel this row is deliberately a no-op.
    { label: `Focus ${label}`, run: () => { if (ctx().isMember(id)) ctx().activateTab(id, { userInitiated: true }); else ctx().showPanel(id); } },
    ...(isSimpleMode() ? [] : [{ label: 'Open in a new tile', glyph: '⊞', run: () => openPanelBeside(id) }]),
    { label: 'Open in a new window', glyph: '↗', disabled: ctx().getPanelStandaloneUrl(id) === null, run: () => ctx().popoutPanel(id) },
    { divider: true },
    // The one server-side membership change in the menu — same POST as the "×".
    { label: 'Remove from rail', glyph: '×', danger: true, run: () => setPanelVisibility(id, false) },
  ];
}

/**
 * The terminal surfaces' menu rows — the SESSION rail entry and the terminal
 * tile header, which share one verb set.
 *
 * Empty in simple mode, and that emptiness IS the callers' decline signal:
 * there the terminal card is replaced by the operator console and
 * closeTerminalPanel no-ops, so every row would act on a surface the operator
 * cannot see. (The palette drops the same actions there, for the same reason.)
 * @returns {import('./panel-context-menu.js').MenuItem[]}
 */
export function buildTerminalMenuItems() {
  if (isSimpleMode()) return [];
  return [
    // restartTerminal tears the PTY down but does NOT reconnect — pairing it
    // with startTerminal is what keeps the card from being left stranded. The
    // palette's "Restart terminal" runs the identical pair.
    { label: 'Restart terminal', run: async () => { await restartTerminal(); startTerminal(); } },
    { label: 'New session', run: () => { startNewSession(); } },
    { divider: true },
    { label: 'Close terminal tile', glyph: '×', danger: true, run: () => closeTerminalPanel() },
  ];
}
