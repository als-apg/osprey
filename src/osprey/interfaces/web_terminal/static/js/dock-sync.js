// @ts-check
/* OSPREY Web Terminal — Dock State Sync
 *
 * The bridge between the dockview workspace and the server-owned panel state.
 * The server is authoritative: every panel show/hide/focus lives on the server
 * and is broadcast over the panel_visibility / panel_focus / panel_register SSE
 * echo, so an agent MCP call and a human dock gesture are indistinguishable
 * downstream (see panel-commands.js / panel-manager.js). This module carries the
 * REVERSE half — turning a human's dockview gestures back into the same
 * server POSTs — without letting the server's own echo bounce back out again.
 *
 * TWO DIRECTIONS
 * --------------
 *  - Server → dock: the SSE handler in panel-manager already drives the dock
 *    (activateTab → focusPanel, the visibility fallback → hidePanel, register →
 *    addPanel, all via the dock-iframe adapter). This module does NOT duplicate
 *    that; it only guarantees those applied changes never POST back.
 *  - Dock → server: a human focusing a dock tab POSTs setPanelFocus — the only
 *    new POST source the docked model introduces. A human CLOSING a dock tab is
 *    deliberately NOT a server change: tile occupancy is per-client layout
 *    state under the launcher-rail model (rail membership is the shared state,
 *    and closing a tile does not remove the panel from the rail), so the close
 *    is routed to a locally-registered vacate handler instead.
 *
 * THE ECHO GUARD (core correctness property)
 * ------------------------------------------
 * dockview's onDidActivePanelChange fires for BOTH a human tab click AND a
 * programmatic setActive (the echo of a server-driven focus). We replicate
 * panel-manager's `userInitiated` guard (panel-manager.js activateTab): every
 * server-driven focus application is wrapped in withEchoSuppressed(), which
 * raises a depth counter the onDidActivePanelChange handler checks — so the
 * echo of an applied focus is recognised and never re-POSTed. (dockview's
 * Emitter.fire is synchronous, so the suppressed window covers the echo.)
 *
 * WHY CLOSE USES A CLICK, NOT onDidRemovePanel
 * --------------------------------------------
 * A placeholder panel is removed both when a human closes its tab AND during
 * every programmatic layout rebuild (mode flip, reset, expert-layout restore —
 * all api.clear()/fromJSON in dock-workspace.js, a file this module does not own
 * and cannot wrap in the suppress guard). onDidRemovePanel therefore cannot tell
 * a human close from a rebuild, and treating a rebuild's removals as closes would
 * silently hide every panel on the server. A capture-phase click on the tile
 * bar's close control (`.tile-tab-close`, dock-tab.js — `.dv-default-tab-action`
 * kept as a fallback for any dockview-rendered default tab) fires ONLY on a
 * genuine human gesture, never during a rebuild, so it is the unambiguous close
 * signal. Capture phase resolves the panel id before dockview's own handler
 * removes the panel.
 *
 * Only SERVICE panels participate: their dock ids carry the adapter's `iframe:`
 * placeholder prefix. The native terminal/workspace panels have no server-side
 * visibility/focus, so their tab clicks are ignored.
 *
 * FALLBACK: with no DockviewApi (dock shell absent / failed to init) every entry
 * point no-ops — the pre-dock rail path in panel-manager still POSTs on its own.
 */

import { getDockApi } from './dock-workspace.js';
import { setPanelFocus } from './panel-commands.js';
// dock-iframe.js owns the placeholder lifecycle and resolves panels by these ids;
// dockPanelBesideActive() below pre-creates a placeholder the adapter then adopts
// by the same id, so both modules share the single source of truth for them.
import { PLACEHOLDER_PREFIX, PLACEHOLDER_COMPONENT } from './dock-iframe.js';

/** Depth of nested server-apply windows; >0 means "this dock change is an echo". */
let suppressDepth = 0;
/** True once the dockview listeners are attached (idempotent guard). */
let wired = false;
/** Bounded retry so late DockviewApi arrival still wires (mirrors the adapter). */
let wireAttempts = 0;

/**
 * The service panel id behind a dockview panel id, or null when the id is not a
 * service placeholder (the native terminal/workspace panels, or a nullish id).
 * @param {unknown} panelId
 * @returns {string | null}
 */
export function serviceIdOf(panelId) {
  return typeof panelId === 'string' && panelId.startsWith(PLACEHOLDER_PREFIX)
    ? panelId.slice(PLACEHOLDER_PREFIX.length)
    : null;
}

/**
 * Run `fn` with the echo guard raised: any dockview focus change it causes (a
 * programmatic setActive) is recognised as a server-applied echo and not
 * POSTed back. panel-manager's activateTab wraps its focusPanel() call in this —
 * the single seam that makes server-driven and rail-driven focus applications
 * echo-safe. Re-entrant (depth-counted) and exception-safe.
 * @template T
 * @param {() => T} fn
 * @returns {T}
 */
export function withEchoSuppressed(fn) {
  suppressDepth++;
  try {
    return fn();
  } finally {
    suppressDepth--;
  }
}

/**
 * Handle a dockview active-panel change. Skips while an echo window is open
 * (a server-applied focus) and for the native terminal/workspace panels; a
 * genuine human dock-tab focus of a service panel POSTs setPanelFocus, whose
 * SSE echo then drives the rail + iframe through panel-manager (agent ≡ human).
 */
function onActivePanelChange() {
  if (suppressDepth > 0) return;
  const api = getDockApi();
  if (!api) return;
  const id = serviceIdOf(api.activePanel?.id);
  if (id) setPanelFocus(id);
}

/**
 * Handler a human tile close is routed to, registered by panel-manager (which
 * owns the local active state the close must reconcile — clear the active
 * accent). Called with the closed panel's service id; dockview's own bubble
 * handler performs the actual panel removal.
 * @type {((serviceId: string) => void) | null}
 */
let tileCloseHandler = null;

/** @param {((serviceId: string) => void) | null} fn */
export function setTileCloseHandler(fn) {
  tileCloseHandler = fn;
}

/**
 * Capture-phase click handler on the dock root. Routes a human closing a
 * service panel's dock tab to the registered LOCAL vacate handler (never a
 * server POST — see the module header). Runs in capture so the panel id is
 * resolved before dockview's own bubble handler removes the panel.
 * @param {Event} e
 */
function onDockClickCapture(e) {
  const target = e.target;
  if (!(target instanceof Element)) return;
  // The tile bar's close control (dock-tab.js); `.dv-default-tab-action` kept
  // as a fallback for any dockview-rendered default tab.
  if (!target.closest('.tile-tab-close, .dv-default-tab-action')) return;
  const tab = target.closest('.dv-tab');
  if (!tab) return;
  const id = serviceIdOf(panelIdForTab(tab));
  if (id) tileCloseHandler?.(id);
}

/**
 * Resolve the dockview panel id for a `.dv-tab` element. dock-tab.js stamps the
 * id on every tile bar it renders (`data-panel-id` on the `.tile-tab` root), so
 * the resolution is an exact read — no positional mapping. Returns null when
 * the stamp is absent (resolves to a no-op close).
 * @param {Element} tab
 * @returns {string | null}
 */
function panelIdForTab(tab) {
  const tile = tab.querySelector('.tile-tab');
  return (tile instanceof HTMLElement ? tile.dataset.panelId : null) ?? null;
}

/**
 * Dock a service panel's placeholder at an explicit position as its own tile —
 * the shared placement verb behind "open beside" (rail ⊞ / "+" add-menu) and
 * the rail drag-and-drop drop routing. Creates the placeholder by the
 * adapter's own id + component so the adapter adopts it in place (its
 * ensurePlaceholder no-ops when the id already exists), keeping this position.
 *
 * An ALREADY-DOCKED placeholder is MOVED to the position (remove + re-add —
 * safe because placeholders are empty; the overlay iframe simply re-follows
 * its new rectangle), never duplicated. Two guarded no-op cases: the target
 * group is the placeholder's own group (removing a tile's sole panel destroys
 * the group the re-add would reference), and an already-docked panel with no
 * position at all (nothing meaningful to do).
 *
 * No-op in fallback mode and in SIMPLE mode — the locked simple layout has
 * exactly one service tile, so callers there fall through to the plain show
 * path and take the tile over like a rail click. Wrapped in the echo guard so
 * neither the removal's auto-activation nor the add's active-panel change is
 * read as a human focus — the caller's show path owns the focus POST.
 * @param {string} serviceId  panel-manager rail id
 * @param {string} [title]    dock tab title (defaults to the id)
 * @param {{referenceGroup?: any, direction: string} | null} [position]
 *   referenceGroup omitted = split against the grid edge in `direction`
 */
export function dockPanelAt(serviceId, title = serviceId, position = null) {
  const api = getDockApi();
  if (!api) return;
  if (document.documentElement.getAttribute('data-ui-mode') === 'simple') return;
  const placeholderId = PLACEHOLDER_PREFIX + serviceId;
  const existing = api.getPanel(placeholderId);
  if (existing) {
    if (!position) return;
    if (existing.group && existing.group === position.referenceGroup) return;
  }
  /** @type {any} */
  const opts = { id: placeholderId, component: PLACEHOLDER_COMPONENT, title };
  if (position) opts.position = position;
  withEchoSuppressed(() => {
    try {
      if (existing) api.removePanel(existing);
      api.addPanel(opts);
    } catch (err) {
      console.error('dock-sync: dockPanelAt failed for', placeholderId, err);
    }
  });
}

/**
 * Dock a service panel's placeholder BESIDE the currently-active group as a
 * NEW tile, so a panel opened from the rail's ⊞ corner or the "+" add-menu
 * splits where the operator is working — the discoverable half of the
 * open-beside verb (rail drag-and-drop is the precise half; both land in
 * {@link dockPanelAt}, so an already-open panel is moved, not duplicated). A
 * rail click on the entry itself still REPLACES the focused tile's panel
 * instead (the adapter's ensurePlaceholder).
 * @param {string} serviceId  panel-manager rail id
 * @param {string} [title]    dock tab title (defaults to the id)
 */
export function dockPanelBesideActive(serviceId, title = serviceId) {
  const api = getDockApi();
  if (!api) return;
  const position = api.activeGroup
    ? { referenceGroup: api.activeGroup, direction: 'right' }
    : null;
  dockPanelAt(serviceId, title, position);
}

/**
 * Attach the dockview listeners once the DockviewApi exists. The api arrives
 * after panel-manager inits (dock-workspace runs later in boot), so this polls a
 * bounded number of times, mirroring the dock-iframe adapter's late-api pattern.
 * Idempotent: a no-op once wired or once the shell is confirmed absent.
 */
export function initDockSync() {
  if (wired) return;
  // No document means no shell will ever arrive — the page is gone, or a test
  // environment was disposed while a retry was still queued. Drop the poll
  // instead of dereferencing a global that no longer exists.
  if (typeof document === 'undefined') return;
  const api = getDockApi();
  const root = document.getElementById('dock-root');
  if (api && root) {
    api.onDidActivePanelChange(onActivePanelChange);
    root.addEventListener('click', onDockClickCapture, true);
    wired = true;
    return;
  }
  if (wireAttempts++ > 30) return;  // ~4.5s at 150ms — the shell is genuinely absent
  setTimeout(initDockSync, 150);
}
