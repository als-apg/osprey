// @ts-check
/* OSPREY Web Terminal — Panel Placement
 *
 * WHERE a panel lands in the workspace. panel-manager.js owns the panel state
 * machine (health, membership, iframes, SSE dispatch); this module owns the
 * tile geometry those states are applied onto — the open-beside verb behind the
 * rail's ⊞ corner and drag-and-drop, the polite agent switch, and the
 * whole-workspace rebuild a `panel_arrange` frame requests.
 *
 * It reaches the dock only through the existing seams: dock-sync's placement
 * verbs (dockPanelAt / dockPanelBesideActive) and echo guard, dock-iframe's
 * hidePanel and placeholder-id namespace. Everything it needs from
 * panel-manager's private state arrives as injected effects registered once at
 * init ({@link initPanelPlacement}), the same seam pattern the iframe adapter
 * and the tile-close handler use — so this module holds no panel state of its
 * own and stays unit-testable through panel-manager's own SSE surface.
 *
 * TWO MUTATION CHANNELS (precedence)
 * ----------------------------------
 * `panel_visibility` and `panel_arrange` can both touch the same panel, so the
 * split is deliberate: a visibility HIDE keeps its existing meaning everywhere
 * (rail removal plus tile close, on every client), while an arrange only ever
 * ADDS membership for the tiles it lists — except on the preset path, where
 * `prune_rail` reproduces today's membership-exclusive preset semantics and
 * non-members leave the rail. An arrange therefore never silently un-shows a
 * panel the operator added, and a preset click keeps behaving as it always has.
 *
 * MODE DEGRADATION
 * ----------------
 * The tile rebuild runs only with a dock shell in expert mode. Simple mode has
 * exactly one service tile by construction, and fallback mode (no dock shell)
 * has none at all: both skip the rebuild and let the focus step take the single
 * surface over. Rail MEMBERSHIP is applied in every mode — the launcher rail is
 * not a dock feature, and a preset must prune it wherever it is clicked.
 */

import {
  withEchoSuppressed, dockPanelAt, dockPanelBesideActive, serializeOpenTiles,
} from './dock-sync.js';
import { hidePanel, PLACEHOLDER_PREFIX } from './dock-iframe.js';
import { getDockApi } from './dock-workspace.js';
import { TERMINAL_RAIL_ID } from './panel-catalog.js';

/**
 * The panel-manager state and actions this module places panels through. Every
 * entry is a live closure over panel-manager's private state — never a snapshot.
 * @typedef {object} PlacementDeps
 * @property {(id: string) => boolean} isKnown - panel-manager tracks this id (it has panel state)
 * @property {(id: string) => boolean} isHealthy - the panel can currently fill a tile
 * @property {(id: string) => boolean} isMember - the panel has rail membership
 * @property {() => string[]} members - every rail member id
 * @property {(id: string) => string} label - the panel's dock tab title
 * @property {(id: string) => void} addMember - apply membership + rail entry LOCALLY (no POST)
 * @property {(id: string) => void} dropMember - remove membership + rail entry LOCALLY (no POST)
 * @property {(id: string, options?: {userInitiated?: boolean, auto?: boolean}) => void} activate
 * @property {(id: string) => void} reveal - reveal a NON-member through the visibility POST path
 * @property {() => string | null} getActive - the locally surfaced panel id
 * @property {() => void} clearActive - drop the local active accent/stamp
 * @property {(message: string) => void} renderEmpty - paint the strand-proof empty pane
 * @property {(id: string) => void} glow - transient agent glow on a panel's rail entry and its tile
 * @property {() => void} openTerminal - reopen the native terminal tile
 */

/** @type {PlacementDeps | null} */
let deps = null;

/**
 * Register the panel-manager effects every verb here applies through. Called
 * once from initPanelManager, before any placement can happen.
 * @param {PlacementDeps} placementDeps
 */
export function initPanelPlacement(placementDeps) {
  deps = placementDeps;
}

/**
 * The registered effects, or a hard failure. Placement before init is a wiring
 * bug, not a runtime condition to degrade around — panel-manager registers the
 * deps in the same init that mounts the rail.
 * @returns {PlacementDeps}
 */
function ctx() {
  if (!deps) throw new Error('panel-placement: initPanelPlacement() has not run');
  return deps;
}

/**
 * Whether a service panel currently occupies a dock tile — its placeholder
 * exists in the grid. False in fallback mode, where there are no tiles at all.
 * Resolved through the adapter's public placeholder-id namespace, the same way
 * dock-sync's placement verbs address them.
 * @param {string} panelId
 * @returns {boolean}
 */
export function hasDockedTile(panelId) {
  return !!getDockApi()?.getPanel(PLACEHOLDER_PREFIX + panelId);
}

/**
 * Whether ANY service panel holds a tile right now — read straight off the live
 * panel list, not off the serialized layout.
 *
 * The polarity matters and is not symmetric with the rebuild's. There, a layout
 * that cannot be read means "drop nothing" — the safe direction. Here, a false
 * "no tiles" would send an agent switch down the plain-activation path, whose
 * placement REPLACES the tile the operator is watching: the one outcome this
 * verb exists to prevent. Reading the panel list directly removes the question,
 * since it stays answerable while a layout is mid-rebuild or refuses toJSON.
 * @returns {boolean}
 */
function hasAnyServiceTile() {
  const panels = getDockApi()?.panels;
  return (Array.isArray(panels) ? panels : []).some(
    (/** @type {any} */ p) => typeof p?.id === 'string' && p.id.startsWith(PLACEHOLDER_PREFIX),
  );
}

/**
 * Shared reveal tail for every "open as a new tile" path (rail ⊞, rail drop): a
 * member panel is focused locally + reported (the activation no-ops while the
 * panel is unhealthy — the placeholder tile still appears and fills on the next
 * health settle); a non-member is revealed through the same visibility-POST path
 * the "+" menu uses.
 * @param {string} panelId
 */
function revealOpenedPanel(panelId) {
  const c = ctx();
  if (c.isMember(panelId)) c.activate(panelId, { userInitiated: true });
  else c.reveal(panelId);
}

/**
 * Open a panel as a NEW tile beside the active group — the rail ⊞ corner's
 * action. An already-open panel is MOVED beside the active tile (dockPanelAt's
 * move semantics), never duplicated. In simple mode the placement no-ops in
 * dock-sync and the reveal tail takes the single tile over like a rail click.
 * @param {string} panelId
 */
export function openPanelBeside(panelId) {
  if (panelId === TERMINAL_RAIL_ID) { ctx().openTerminal(); return; }
  dockPanelBesideActive(panelId, ctx().label(panelId));
  revealOpenedPanel(panelId);
}

/**
 * Open (or move) a panel at an explicit drop position — the rail's
 * drag-and-drop landing, the precise half of the open-beside verb.
 * @param {string} panelId
 * @param {{referenceGroup?: any, direction: string} | null} position
 */
export function dropPanelAt(panelId, position) {
  dockPanelAt(panelId, ctx().label(panelId), position);
  revealOpenedPanel(panelId);
}

/**
 * Apply an agent open_panel: bring the panel into view without ever taking a
 * tile away from the operator. A panel that already holds a tile is simply
 * focused (no move); one without a tile opens as a NEW tile beside the active
 * group — the same placement the rail's ⊞ corner uses — so whatever the
 * operator was watching survives. A non-member gains its rail entry first: the
 * switch implies membership, and the server's own visibility broadcast then
 * lands on an already-current rail.
 *
 * Both mode degradations fall out of the placement verb rather than a branch
 * here: dockPanelBesideActive no-ops in simple mode (one service tile by
 * construction) and in fallback mode (no dock shell), leaving exactly the plain
 * activation takeover both had before.
 * @param {string} panelId
 */
export function applyAgentSwitch(panelId) {
  const c = ctx();
  // Unknown id — a panel dropped from the deployment, or the terminal entry,
  // which has no panel state. The activation would refuse it, so refuse before
  // touching membership or the grid.
  if (!c.isKnown(panelId)) return;
  if (!c.isMember(panelId)) c.addMember(panelId);
  // Only a panel that can fill a tile gets one: the activation refuses while
  // unhealthy, and the tile it never claimed would linger empty. An unhealthy
  // target stays the no-op it has always been (attention badge intact).
  //
  // "Beside" also needs something to be beside. With every service tile retired
  // the active group is the TERMINAL's, so splitting off it would open the
  // panel to the terminal's right at a default width — losing the first-tile
  // rule (left of the terminal at the classic split) that the adapter applies
  // on a plain activation. Nothing is open to evict in that state either, so
  // the plain activation below is both correctly anchored and still polite.
  if (c.isHealthy(panelId) && !hasDockedTile(panelId) && hasAnyServiceTile()) {
    dockPanelBesideActive(panelId, c.label(panelId));
  }
  c.activate(panelId);
}

/**
 * Whether the service-tile region can be rebuilt: a dock shell must exist, and
 * the mode must not be the locked single-tile simple layout.
 * @returns {boolean}
 */
function canRebuildTiles() {
  return !!getDockApi() && document.documentElement.getAttribute('data-ui-mode') !== 'simple';
}

/**
 * Rebuild the service-tile region to exactly `tiles`, left to right.
 *
 * Deterministic by construction rather than by diffing: every docked service
 * tile the arrangement does not list is dropped, then each requested tile is
 * docked in order — the first against the grid's left edge (so services sit
 * left of the terminal, as they always open), each next one to the RIGHT of the
 * one before it. dockPanelAt MOVES an already-docked placeholder rather than
 * duplicating it, so a tile that was already open simply slides into position.
 * Prior service-tile geometry (sash positions) is deliberately discarded — an
 * arrangement is a request for a layout, not for the operator's pixel widths.
 *
 * The whole rebuild runs inside the echo guard: every add/remove drives
 * dockview's active-panel change, and each one is a server-applied echo that
 * must not POST focus back. Surfacing each tile afterwards fills the
 * placeholders with their overlay iframes — a docked tile whose panel was never
 * activated has no iframe to show.
 * @param {string[]} tiles
 */
function rebuildTiles(tiles) {
  const c = ctx();
  const api = getDockApi();
  const wanted = new Set(tiles);
  // A layout the serializer cannot read is not the same as an empty workspace:
  // drop nothing rather than assert tiles that may still be there.
  const open = serializeOpenTiles(api) ?? [];
  withEchoSuppressed(() => {
    for (const id of open) if (!wanted.has(id)) hidePanel(id);
    /** @type {string | null} */
    let prev = null;
    for (const id of tiles) {
      // Never OPEN a tile for a panel that cannot fill it — the same rule
      // applyAgentSwitch applies. The activation below refuses while unhealthy,
      // so the tile would stay empty for as long as anything else holds focus,
      // and nothing would come back for it. Membership is applied regardless,
      // so the panel stays one rail click away. A tile that already exists is
      // still positioned: it is filled (its iframe outlives a health dip), and
      // leaving it out of the walk would strand it outside the requested order.
      if (!c.isHealthy(id) && !hasDockedTile(id)) continue;
      const group = prev ? api.getPanel(PLACEHOLDER_PREFIX + prev)?.group : null;
      dockPanelAt(id, c.label(id), group
        ? { referenceGroup: group, direction: 'right' }
        : { direction: 'left' });
      // Advance only for a tile that actually landed: a skipped panel — or a
      // placement that threw — must not become the anchor the next one splits
      // against, which is what could push a tile to the terminal's right.
      if (hasDockedTile(id)) prev = id;
    }
  });
  for (const id of tiles) c.activate(id, { auto: true });
}

/**
 * Apply a `panel_arrange` frame: the declarative end state an `arrange_workspace`
 * call (or a preset click, which is the same server operation) asked for.
 *
 * Order matters. Membership first, so a listed non-member already has its rail
 * entry when its tile appears and a pruned panel's entry is gone before the
 * region is rebuilt. Then the tile rebuild, where the dock shell allows one.
 * Focus last, by the shipped preset rule: the requested panel when it is
 * healthy, else the first healthy listed tile, else no focus change at all —
 * which falls out on its own, since an activation refuses an unhealthy panel.
 *
 * @param {{tiles?: string[], focus?: string, prune_rail?: boolean, source?: string}} frame
 */
export function applyArrange({ tiles = [], focus, prune_rail: pruneRail = false, source }) {
  const c = ctx();
  // The server validates against the same inventory; filtering again keeps a
  // stale client (a panel dropped from the deployment) from arranging a ghost.
  const wanted = tiles.filter((id) => c.isKnown(id));
  const wantedSet = new Set(wanted);

  // MEMBERSHIP — every mode. A preset prunes to its members (today's exclusive
  // semantics); a plain tiles request only adds. A pruned panel loses its tile
  // here too: in simple and fallback mode the rebuild below never runs, and its
  // surface must not outlive its rail entry.
  if (pruneRail) {
    for (const id of c.members()) {
      if (wantedSet.has(id)) continue;
      c.dropMember(id);
      withEchoSuppressed(() => hidePanel(id));
    }
  }
  for (const id of wanted) if (!c.isMember(id)) c.addMember(id);

  if (canRebuildTiles()) rebuildTiles(wanted);

  // FOCUS — healthy-fallback. {auto: true} keeps the health guard and the
  // membership guard; both are satisfied for a target picked here, so it only
  // documents that this is an applied policy, not a human's explicit pick.
  const active = c.getActive();
  const target = focus && c.isHealthy(focus) ? focus : wanted.find((id) => c.isHealthy(id));
  if (target) {
    c.activate(target, { auto: true });
  } else if (active && !wantedSet.has(active)) {
    // Nothing listed can be surfaced and the panel that held the view is not in
    // the arrangement, so the accent is genuinely stale and always clears.
    //
    // The empty PANE is a separate question, and the honest one to ask is
    // OCCUPANCY, not focusability: the rebuild above deliberately keeps an
    // unhealthy panel's EXISTING tile (it is still filled — the overlay iframe
    // outlives a health dip), so an arrangement listing only such a panel would
    // otherwise paint "No panels visible" across a workspace that visibly still
    // holds one. Occupancy is read from the live panel list, so a layout
    // mid-rebuild still reports the tiles it holds; fallback mode has no tiles
    // by construction and does paint, which is the case where a stale surface
    // would otherwise linger behind the cleared accent.
    c.clearActive();
    if (!hasAnyServiceTile()) c.renderEmpty('No panels visible');
  }

  if (source === 'agent') for (const id of wanted) c.glow(id);
}
