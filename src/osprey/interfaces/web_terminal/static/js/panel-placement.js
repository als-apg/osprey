// @ts-check
/* OSPREY Web Terminal — Panel Placement
 *
 * WHERE a panel lands in the workspace. The panel state machine splits three
 * ways: panel-manager.js owns the core (tile occupancy, iframe lifecycle, the
 * active-panel policy), panel-lifecycle.js owns how a panel comes to exist and
 * stay alive, and panel-sse.js owns the server→client frame dispatcher. This
 * module owns the tile geometry: the person's "Open in a new tile" verb, the
 * tile bar's Open panel ▸ placements ({@link openPanelAtTile}), the
 * landing of a sidebar row's drop, and the op runner ({@link runPanelOp})
 * that applies a `panel_op` frame — an agent's drop, close, open, set or
 * arrangement, and a person's preset click, which the server relays the same
 * way — to the layout holder. An open itself follows the open rule in
 * dock-open.js (the service tile used last, replacing the panel it showed).
 *
 * It reaches the dock only through the existing seams: dock-sync's placement
 * verbs (dockPanelBesideActive, dockPanelAt) and echo guard, dock-open.js (the open rule,
 * the set and arrange composers), the dock engine (drops and closes), and the
 * layout holder (dock-apply.js), which answers what the dock holds and applies
 * every change. Everything it needs from panel-manager's private state arrives
 * as injected effects registered once at init ({@link initPanelPlacement}), so
 * this module holds no panel state of its own.
 *
 * ONE MUTATION CHANNEL
 * --------------------
 * An agent changes the dock only through a `panel_op` frame addressed to this
 * client: the runner checks it against the held layout, runs it on the holder
 * and the frame handler POSTs the outcome. Rail membership stays the server's:
 * it follows a recorded result with `panel_visibility` frames (opened panels
 * join the rail; a preset prunes it to its members), so nothing here edits the
 * rail. Fallback mode (no dock shell, no holder) refuses every op.
 *
 * A panel living in its own pop-out window is in no tile: an `open` of it
 * brings its window forward and leaves the layout alone, and every other op
 * naming it is refused `popped_out` — the operator brings it back.
 */

import { dockPanelAt, dockPanelBesideActive, withEchoSuppressed } from './dock-sync.js';
import { getLayoutHolder, isDetached, TERMINAL_KEY } from './dock-apply.js';
import {
  arrange,
  focusKey,
  openPanelSet,
  openPlaced,
  PANEL_UNHEALTHY,
  TERMINAL_IN_LAYOUT,
} from './dock-open.js';
import { ONE_PANEL_PER_TILE } from './dock-policy.js';
import { closeTab, findTile, keysOf, planDrop, tileOf } from '/static/vendor/dock-engine.mjs';
import { TERMINAL_RAIL_ID } from './panel-catalog.js';
import { isArtifactTileKey } from './artifact-tiles.js';

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
 * Whether a service panel currently holds a tile in the dock, read off the
 * layout holder. False with no holder (fallback mode has no tiles at all).
 * @param {string} panelId
 * @returns {boolean}
 */
export function hasDockedTile(panelId) {
  return tileOf(getLayoutHolder()?.layout ?? null, panelId) !== null;
}

/**
 * Shared reveal tail for every "open as a new tile" path (the row menu, a rail
 * drop): a member panel is focused locally + reported (the activation no-ops
 * while the panel is unhealthy — the placeholder tile still appears and fills
 * on the next health settle); a non-member is revealed through the same
 * visibility-POST path the "+" menu uses. The exception is a non-member that
 * already held a tile before the move (a tile opened from a chat link or the
 * palette and then rearranged): moving it is layout, not a request to join the
 * rail, so it is focused without a POST and stays off it. An artifact tile
 * (artifact-tiles.js) is no panel the rail can hold: it is focused the same way.
 * @param {string} panelId
 * @param {boolean} wasOnScreen - the panel held a tile before this placement
 */
function revealOpenedPanel(panelId, wasOnScreen) {
  const c = ctx();
  if (c.isMember(panelId)) c.activate(panelId, { userInitiated: true });
  else if (wasOnScreen || isArtifactTileKey(panelId)) c.activate(panelId);
  else c.reveal(panelId);
}

/**
 * Open a panel as a NEW tile beside the active tile — the row menu's "Open in
 * a new tile", the one open that splits instead of replacing. An
 * already-open panel is MOVED beside the active tile (dockPanelAt's move
 * semantics), never duplicated. In fallback mode the placement no-ops in
 * dock-sync and the reveal tail takes the single surface over like a rail click.
 * @param {string} panelId
 */
export function openPanelBeside(panelId) {
  if (panelId === TERMINAL_RAIL_ID) {
    ctx().openTerminal();
    return;
  }
  const wasOnScreen = hasDockedTile(panelId);
  dockPanelBesideActive(panelId, ctx().label(panelId));
  revealOpenedPanel(panelId, wasOnScreen);
}

/**
 * The tile a service panel is shown in, read off the layout holder; null when
 * it holds no tile (or there is no holder). A service tile bar's own tile.
 * @param {string} panelId
 * @returns {string | null}
 */
export function dockedTileOf(panelId) {
  return tileOf(getLayoutHolder()?.layout ?? null, panelId)?.tile ?? null;
}

/** The engine side each split placement of {@link openPanelAtTile} lands on. */
const SPLIT_SIDE = /** @type {const} */ ({ right: 'right', down: 'bottom' });

/**
 * Open a panel against ONE named tile — the tile bar's Open panel ▸ Right /
 * Down / Replace here, where the tile is the bar's own. Right and Down split
 * that tile (dock-sync's dockPanelAt; an already-docked panel is moved, never
 * duplicated); Replace here swaps that tile's panel for this one through the
 * targeted open (dock-open.js's openPlaced with an explicit tile), NOT the
 * everyday open, which replaces the service tile used last. Every placement
 * ends in the same reveal tail as the other opens, so rail membership,
 * activation and the focus POST match.
 *
 * Refused, with nothing changed and nothing revealed: no layout holder (the
 * fallback has no tiles), a tile the dock does not hold, a placement other
 * than the three, the Agent pane as the panel or as Replace here's target
 * (TERMINAL_NOT_AGENT_MOVABLE), and a panel living in its own window.
 * @param {string} panelId
 * @param {string} tileId  the tile to place against
 * @param {'right' | 'down' | 'replace'} place
 * @returns {string | null} the refusal, or null when the panel was opened
 */
export function openPanelAtTile(panelId, tileId, place) {
  if (panelId === TERMINAL_RAIL_ID || panelId === TERMINAL_KEY) return TERMINAL_NOT_AGENT_MOVABLE;
  const holder = getLayoutHolder();
  if (!holder) return NO_DOCK_CLIENT;
  if (place !== 'right' && place !== 'down' && place !== 'replace') return INVALID_TARGET;
  if (findTile(holder.layout ?? null, tileId) === null) return UNKNOWN_TILE;
  if (isDetached(panelId)) return POPPED_OUT;
  if (place === 'replace' && holdsTerminal(holder.layout, tileId)) {
    return TERMINAL_NOT_AGENT_MOVABLE;
  }
  const wasOnScreen = hasDockedTile(panelId);
  const title = ctx().label(panelId);
  if (place === 'replace') {
    holder.setPanelRecord(panelId, { title });
    withEchoSuppressed(() =>
      holder.run((layout, opCtx) =>
        openPlaced(layout, panelId, { place: 'replace', tile: tileId }, opCtx)
      )
    );
  } else {
    dockPanelAt(panelId, title, { referenceTile: tileId, direction: SPLIT_SIDE[place] });
  }
  revealOpenedPanel(panelId, wasOnScreen);
  return null;
}

/**
 * Reveal a panel a sidebar row's drop has just placed (rail-drag.js's landing):
 * the engine put its tile where the row was dropped and the holder has applied
 * it; what is left is the activation tail every other open uses.
 * @param {string} panelId
 * @param {{ wasOnScreen: boolean }} info  whether the panel held a tile before the drop
 */
export function landDroppedPanel(panelId, { wasOnScreen }) {
  revealOpenedPanel(panelId, wasOnScreen);
}

// ---- The op runner: applies a relayed `panel_op` frame to the holder ----

/** Refusals this runner names itself (snake_case, as /api/panel-op-result takes them). */
export const STALE_BASE = 'stale_base';
export const TERMINAL_NOT_AGENT_MOVABLE = 'terminal_not_agent_movable';
export const POPPED_OUT = 'popped_out';
export const UNKNOWN_KEY = 'unknown_key';
export const UNKNOWN_TILE = 'unknown_tile';
export const INVALID_SOURCE = 'invalid_source';
export const INVALID_TARGET = 'invalid_target';

/** The sides an agent's open may split a tile on. */
const OPEN_EDGES = new Set(['left', 'right', 'top', 'bottom']);
export const INVALID_LAYOUT = 'invalid_layout';
export const UNKNOWN_OP = 'unknown_op';
export const NO_DOCK_CLIENT = 'no_dock_client';

/**
 * One `panel_op` frame as the server relays it (routes/panel_ops.py `_relay_op`).
 * @typedef {object} PanelOpFrame
 * @property {'panel_op'} type
 * @property {string} request_id
 * @property {string} target_client
 * @property {string} op - drop | close | open | open_set | open_arrangement
 * @property {Record<string, any>} args
 * @property {number} [base_rev]
 * @property {string} [source]
 */

/**
 * What running an op came to: a named refusal (nothing changed), or the
 * holder's result — with `popout` naming the panel whose own window an `open`
 * brings forward instead of changing the layout.
 * @typedef {{refusal: string, detail?: string} | {refusal?: undefined, result: import('./dock-apply.js').OpResult, popout?: string}} PanelOpOutcome
 */

/**
 * Whether the tile `id` of `layout` holds the terminal.
 * @param {any} layout
 * @param {string} id
 */
function holdsTerminal(layout, id) {
  return findTile(layout, id)?.tabs.includes(TERMINAL_KEY) ?? false;
}

/**
 * A member-index collection layout (the frames' form) over panel ids.
 * Null when an index names no member.
 * @param {any} node
 * @param {string[]} members
 * @returns {any}
 */
function byMember(node, members) {
  if (node === null || typeof node !== 'object') return null;
  if (Array.isArray(node.items)) {
    const items = node.items.map((/** @type {any} */ item) => byMember(item, members));
    return items.includes(null) ? null : { ...node, items };
  }
  if (!Array.isArray(node.tabs)) return null;
  const tabs = node.tabs.map((/** @type {any} */ idx) => members[idx]);
  if (tabs.some((/** @type {unknown} */ key) => typeof key !== 'string')) return null;
  const out = { ...node, tabs };
  if ('showing' in node) {
    const showing = members[node.showing];
    if (typeof showing !== 'string') return null;
    out.showing = showing;
  }
  return out;
}

/**
 * The tile ids an engine drop target names (a tile zone or edge, or a tile's
 * border).
 * @param {any} target
 * @returns {string[]}
 */
function targetTiles(target) {
  if (target && typeof target.tile === 'string') return [target.tile];
  if (target && target.border && typeof target.border.tile === 'string')
    return [target.border.tile];
  return [];
}

/**
 * Check one op against the held layout and build the holder operation that
 * runs it, or name why it is refused. The terminal invariant is enforced here
 * a second time (the server checked it against its copy of the layout): the
 * terminal, and its tile, is never an agent drop source or close target, and
 * a set never names it. So is the one-panel-per-tile rule (dock-policy.js): a
 * set opened as anything but side by side or a layout, and a drop into a
 * tile's centre or header, are refused by the policy's name. So is the
 * pop-out rule: a panel in its own window is never moved or closed by an op,
 * and an `open` of it brings the window forward.
 * @param {PanelOpFrame} frame
 * @param {any} layout  the held layout
 * @param {(key: string) => boolean} isHealthy
 * @returns {{refusal: string, detail?: string} | {run: (layout: any, ctx: any) => any, popout?: string}}
 */
function planOp(frame, layout, isHealthy) {
  const args = frame.args ?? {};
  /** @param {any} ctx */
  const setCtx = (ctx) => ({ ...ctx, isHealthy });
  switch (frame.op) {
    case 'open': {
      const member = args.member;
      if (member === TERMINAL_KEY) return { refusal: TERMINAL_NOT_AGENT_MOVABLE };
      if (typeof member !== 'string') return { refusal: INVALID_SOURCE };
      if (isDetached(member)) {
        return { run: (l) => ({ layout: l, changed: false }), popout: member };
      }
      if (tileOf(layout, member)) return { run: (l) => focusKey(l, member) };
      if (!isHealthy(member)) return { refusal: PANEL_UNHEALTHY };
      const place = args.place;
      if (place !== undefined && place !== 'split' && place !== 'replace') {
        return { refusal: INVALID_TARGET };
      }
      if (args.edge !== undefined && (place === 'replace' || !OPEN_EDGES.has(args.edge))) {
        return { refusal: INVALID_TARGET };
      }
      if (args.tile !== undefined) {
        if (typeof args.tile !== 'string' || findTile(layout, args.tile) === null) {
          return { refusal: UNKNOWN_TILE };
        }
        if (place === 'replace' && holdsTerminal(layout, args.tile)) {
          return { refusal: TERMINAL_NOT_AGENT_MOVABLE };
        }
      }
      /** @type {import('./dock-open.js').OpenPlacement} */
      const placement = {};
      if (place !== undefined) placement.place = place;
      if (args.tile !== undefined) placement.tile = args.tile;
      if (args.edge !== undefined) placement.edge = args.edge;
      return { run: (l, ctx) => openPlaced(l, member, placement, ctx) };
    }
    case 'open_set':
    case 'open_arrangement': {
      const members = Array.isArray(args.members) ? args.members : [];
      if (members.includes(TERMINAL_KEY)) return { refusal: TERMINAL_IN_LAYOUT };
      if (members.some((/** @type {string} */ key) => isDetached(key))) {
        return { refusal: POPPED_OUT };
      }
      const given = args.layout === undefined ? undefined : byMember(args.layout, members);
      if (given === null) return { refusal: INVALID_LAYOUT };
      if (given !== undefined && keysOf(given).includes(TERMINAL_KEY)) {
        return { refusal: TERMINAL_IN_LAYOUT };
      }
      const as = frame.op === 'open_set' ? args.as : (args.as ?? 'layout');
      if (as === 'layout') {
        if (given === undefined) return { refusal: INVALID_LAYOUT };
        return { run: (l, ctx) => arrange(l, given, setCtx(ctx)) };
      }
      if (as !== 'side') return { refusal: ONE_PANEL_PER_TILE };
      return { run: (l, ctx) => openPanelSet(l, members, 'side', undefined, setCtx(ctx)) };
    }
    case 'drop': {
      const source = args.source ?? {};
      /** @type {{new: string} | {tab: string} | {tile: string} | null} */
      let engineSource = null;
      if (source.new && typeof source.new.panel === 'string') {
        const key = source.new.panel;
        if (key === TERMINAL_KEY) return { refusal: TERMINAL_NOT_AGENT_MOVABLE };
        if (isDetached(key)) return { refusal: POPPED_OUT };
        if (tileOf(layout, key) === null && !isHealthy(key)) return { refusal: PANEL_UNHEALTHY };
        engineSource = tileOf(layout, key) ? { tab: key } : { new: key };
      } else if (typeof source.tab === 'string') {
        if (source.tab === TERMINAL_KEY) return { refusal: TERMINAL_NOT_AGENT_MOVABLE };
        if (isDetached(source.tab)) return { refusal: POPPED_OUT };
        if (tileOf(layout, source.tab) === null) return { refusal: UNKNOWN_KEY };
        engineSource = { tab: source.tab };
      } else if (typeof source.tile === 'string') {
        if (findTile(layout, source.tile) === null) return { refusal: UNKNOWN_TILE };
        if (holdsTerminal(layout, source.tile)) return { refusal: TERMINAL_NOT_AGENT_MOVABLE };
        engineSource = { tile: source.tile };
      }
      if (engineSource === null) return { refusal: INVALID_SOURCE };
      if (targetTiles(args.target).some((id) => findTile(layout, id) === null)) {
        return { refusal: UNKNOWN_TILE };
      }
      // A tile holds one panel: its centre and header are no target
      // (dock-drop.js refuses a person's drop there too).
      if ('zone' in (args.target ?? {})) return { refusal: ONE_PANEL_PER_TILE };
      const src = engineSource;
      return {
        run: (l, ctx) => {
          const out = planDrop(l, src, args.target, ctx.dock);
          if (!out.changed) return out;
          const shown =
            'tile' in src
              ? findTile(out.layout, src.tile)
              : tileOf(out.layout, 'new' in src ? src.new : src.tab);
          return shown ? { ...out, focus: shown.tile } : out;
        },
      };
    }
    case 'close': {
      if (args.key === TERMINAL_KEY) return { refusal: TERMINAL_NOT_AGENT_MOVABLE };
      if (typeof args.key === 'string' && isDetached(args.key)) return { refusal: POPPED_OUT };
      if (tileOf(layout, args.key) === null) return { refusal: UNKNOWN_KEY };
      return { run: (l) => closeTab(l, args.key) };
    }
    default:
      return { refusal: UNKNOWN_OP, detail: `unknown op ${String(frame.op)}` };
  }
}

/**
 * Run one agent op (or a person's preset click, which the server relays the
 * same way) against the layout holder.
 *
 * Refused, with nothing changed, when the op was written against a layout this
 * dock no longer holds — `base_rev` present and different from the holder's
 * acked rev, or the holder dirty since — or when it breaks the terminal or
 * pop-out invariant. Opens, sets and arrangements go through dock-open.js (the
 * open rule, the composers); drops and closes through the engine. The holder
 * applies the result, holding the apply while a person's drag is live.
 *
 * Rail membership is not touched here: the server puts opened panels on the
 * rail (and prunes it for a preset) with `panel_visibility` frames once this
 * dock's result has been recorded.
 * @param {PanelOpFrame} frame
 * @returns {PanelOpOutcome}
 */
export function runPanelOp(frame) {
  const holder = getLayoutHolder();
  if (!holder) return { refusal: NO_DOCK_CLIENT, detail: 'this page has no dock' };
  if (
    frame.base_rev !== undefined &&
    frame.base_rev !== null &&
    (frame.base_rev !== holder.ackedRev || holder.dirty)
  ) {
    return { refusal: STALE_BASE };
  }
  const plan = planOp(frame, holder.layout, ctx().isHealthy);
  if ('refusal' in plan) return plan;
  const result = holder.run(plan.run);
  // An engine or dock-open refusal changed nothing; it is passed on by name.
  const named = /** @type {{refusal?: unknown, notice?: unknown}} */ (result ?? {});
  if (typeof named.refusal === 'string') {
    return {
      refusal: named.refusal,
      ...(typeof named.notice === 'string' ? { detail: named.notice } : {}),
    };
  }
  return plan.popout === undefined ? { result } : { result, popout: plan.popout };
}
