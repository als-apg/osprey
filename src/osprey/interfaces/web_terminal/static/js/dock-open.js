// @ts-check
/* OSPREY Web Terminal — Dock placement rules
 *
 * Where a panel lands when it enters the dock, as pure operations on a dock
 * layout: each takes the held layout and the holder's context and returns the
 * layout after, whether it changed and the tile to focus (dock-apply.js's
 * `run`). Nothing here reads or mutates the dock; the holder applies the
 * result. Every tile holds ONE panel (dock-policy.js): the sidebar is the tab
 * system, and an open switches a tile rather than stacking into it.
 *
 * THE OPEN RULE. A panel that is opened — by a sidebar row click, by a link,
 * by the "+" menu, by a pop-out window coming back — TAKES
 * the service tile the operator used last, REPLACING the panel that tile
 * showed ({@link openPanel}): nothing resizes, and the replaced panel keeps its
 * sidebar row, one click away. A panel already in the dock is focused where
 * it is ({@link focusKey}). With no service tile at all, the first one opens
 * LEFT of the agent pane at the classic 60/40 split, or against the dock's
 * left edge when the agent pane is closed too ({@link firstTile}) — the agent
 * pane is always its own tile and never the target.
 *
 *   - the AGENT's open splits the service tile used last into a new tile by
 *     default, or replaces a tile's panel or splits a named tile on a named
 *     side when it says so ({@link openPlaced});
 *   - OPEN AT (a drop on a tile edge or a dock edge, the row's "Open in a new
 *     tile") splits where the gesture said, and moves a panel that already
 *     holds a tile ({@link openAt}) — the one way to make a new tile on purpose;
 *   - the terminal reopens at the dock's right edge, its home side
 *     ({@link reopenTerminal}), and closes as its tile ({@link closeKey});
 *   - a SET of panels opens side by side beside the service tile the operator
 *     used last ({@link openPanelSet}), and an ARRANGEMENT replaces the
 *     service tiles with a given layout while the agent pane keeps its tile,
 *     place and size ({@link arrange}) — the one place an arrangement is
 *     computed, for a person's preset click and an agent's op alike; one
 *     naming several panels for a tile is refused by the policy's name.
 */

import {
  findTile,
  geometry,
  isTile,
  itemFractions,
  keysOf,
  minimumSize,
  mintTileId,
  normalize,
  openSet,
  planDrop,
  removeTab,
  reopen,
  tileIdsOf,
  tileOf,
  validateLayout,
  LAYOUT_REASONS,
} from '/static/vendor/dock-engine.mjs';
import { TERMINAL_KEY } from './dock-apply.js';
import { ONE_PANEL_PER_TILE, crowdedTile } from './dock-policy.js';

/** @typedef {import('/static/vendor/dock-engine.mjs').Layout} Layout */
/** @typedef {import('/static/vendor/dock-engine.mjs').Tile} Tile */
/** @typedef {import('/static/vendor/dock-engine.mjs').DockPx} DockPx */
/** @typedef {import('/static/vendor/dock-engine.mjs').Side} Side */
/** @typedef {import('/static/vendor/dock-engine.mjs').OpenContext} OpenContext */
/** @typedef {import('/static/vendor/dock-engine.mjs').OpenResult} OpenResult */
/** @typedef {import('/static/vendor/dock-engine.mjs').CollectionLayout<string>} PanelCollection */
/** @typedef {import('./dock-apply.js').OpResult} OpResult */
/** @typedef {import('./dock-apply.js').OpContext} OpContext */

/**
 * What a set or an arrangement is opened with: the holder's context plus the
 * health check (absent: every panel is healthy) and the members the dock
 * cannot open.
 * @typedef {OpContext & {isHealthy?: (key: string) => boolean, skip?: string[]}} SetContext
 */

/**
 * A set's or an arrangement's result: the engine's, plus the open services an
 * arrangement removed (`pruned`, so a preset can keep its prune-to-members rail
 * behaviour) and the members skipped as unhealthy (`skipped`).
 * @typedef {OpResult & {refusal?: string, notice?: string, skipped?: string[], pruned?: string[]}} SetResult
 */

/** Refusals (named, and they change nothing). */
export const TERMINAL_IN_LAYOUT = 'terminal_in_layout';
export const PANEL_UNHEALTHY = 'panel_unhealthy';
export const NO_ROOM = 'no_room';

/**
 * Where a panel is placed relative to a tile, or to the dock when no tile is
 * named: the four sides.
 * @typedef {object} Position
 * @property {string} [referenceTile]  the tile to split; the dock's edge when absent
 * @property {Side} direction
 */

/** Share of the row the first service tile takes beside the terminal — the
 *  service side of the classic split (terminal 40% / services 60%). */
export const SERVICE_SHARE = 0.6;
/** Share of the row the terminal takes when it reopens at the dock's edge. */
export const TERMINAL_SHARE = 0.4;

const SIDES = new Set(['left', 'right', 'top', 'bottom']);

/** The narrowest a tile may become along the split axis when the agent's open
 *  splits by default (no `place` given): a split that would leave either half
 *  under this replaces the target's panel instead. An explicit split goes down
 *  to the dock's own minimum. */
export const READABLE_TILE_PX = 300;

/**
 * @param {Layout|null} layout
 * @returns {OpResult}
 */
function unchanged(layout) {
  return { layout, changed: false };
}

/**
 * A fresh tree with every tile passed through `fn`.
 * @param {Layout} node
 * @param {(tile: Tile) => Tile} fn
 * @returns {Layout}
 */
function mapTiles(node, fn) {
  if (isTile(node)) return fn(node);
  return { ...node, items: node.items.map((item) => mapTiles(item, fn)) };
}

/**
 * The service tile the operator used last — where an open lands: the focused
 * tile when it shows a service, else the last tile that did, else the first
 * tile in tree order showing a service. Null when no tile shows a service.
 * The agent pane's tile is never chosen because the agent pane is in it.
 * @param {Layout|null} layout
 * @param {OpContext} ctx
 * @returns {Tile|null}
 */
export function serviceTileTarget(layout, ctx) {
  for (const id of [ctx.focusedTile, ctx.lastServiceTile]) {
    const tile = id === null ? null : findTile(layout, id);
    if (tile && tile.showing !== TERMINAL_KEY) return tile;
  }
  for (const id of tileIdsOf(layout)) {
    const tile = findTile(layout, id);
    if (tile && tile.showing !== TERMINAL_KEY) return tile;
  }
  return null;
}

/**
 * `layout` with tile `id` taking `share` of its parent split; its siblings
 * keep their proportions over the rest. A root tile has no split to share.
 * @param {Layout} layout
 * @param {string} id
 * @param {number} share
 * @returns {Layout}
 */
function withShare(layout, id, share) {
  if (isTile(layout)) return layout;
  const index = layout.items.findIndex((item) => isTile(item) && item.tile === id);
  if (index === -1) {
    return { ...layout, items: layout.items.map((item) => withShare(item, id, share)) };
  }
  const fractions = itemFractions(layout.items);
  const rest = fractions.reduce((sum, f, i) => (i === index ? sum : sum + f), 0);
  const items = layout.items.map((item, i) => ({
    ...item,
    size: i === index ? share : rest > 0 ? (fractions[i] / rest) * (1 - share) : 0,
  }));
  return { ...layout, items };
}

/**
 * The first service tile: left of the terminal at the classic split, or the
 * dock's left edge when the terminal tile is closed too, or alone in an empty
 * dock.
 * @param {Layout|null} layout
 * @param {string} key
 * @param {DockPx} dock
 * @returns {OpResult}
 */
export function firstTile(layout, key, dock) {
  if (layout === null) {
    const tile = mintTileId(null);
    return { layout: { tile, tabs: [key], showing: key }, changed: true, focus: tile };
  }
  const terminal = tileOf(layout, TERMINAL_KEY);
  const target = terminal
    ? { tile: terminal.tile, edge: /** @type {Side} */ ('left') }
    : { outer: /** @type {Side} */ ('left') };
  const out = planDrop(layout, { new: key }, target, dock);
  const home = out.changed ? tileOf(out.layout, key) : null;
  if (!out.changed || !home || out.layout === null) return unchanged(layout);
  const next = terminal ? withShare(out.layout, home.tile, SERVICE_SHARE) : out.layout;
  return { layout: next, changed: true, focus: home.tile };
}

/**
 * Open `key` by the open rule: a panel already in the dock stays where it is
 * (nothing changes here — the caller focuses it, and the focused tile follows
 * that); otherwise it takes the service tile the operator used last,
 * replacing the panel that tile showed — nothing resizes, and the replaced
 * panel keeps its sidebar row — or, with no service tile, opens as the first
 * service tile beside the agent pane.
 * @param {Layout|null} layout
 * @param {string} key
 * @param {OpContext} ctx
 * @returns {OpResult}
 */
export function openPanel(layout, key, ctx) {
  if (tileOf(layout, key)) return unchanged(layout);
  const target = serviceTileTarget(layout, ctx);
  if (!target || layout === null) return firstTile(layout, key, ctx.dock);
  const next = mapTiles(layout, (tile) =>
    tile.tile === target.tile ? { ...tile, tabs: [key], showing: key } : tile
  );
  return { layout: next, changed: true, focus: target.tile };
}

/**
 * Where an agent's open puts a panel that is not docked yet.
 * @typedef {object} OpenPlacement
 * @property {'split' | 'replace'} [place]  a new tile split off the target, or
 *   the target's panel replaced; absent, a split that keeps both halves at
 *   least {@link READABLE_TILE_PX} along the split axis, else a replace
 * @property {string} [tile]  the target tile; absent, the service tile the
 *   operator used last ({@link serviceTileTarget})
 * @property {Side} [edge]  the side of the target a split puts the panel on;
 *   absent, the target's longer side (right when it is wider than tall, else
 *   bottom)
 */

/**
 * The agent's open: a panel already docked stays where it is (the caller
 * focuses it); otherwise it SPLITS the target tile into a new tile of its own
 * by default, so the panel the operator had there stays on screen, or
 * REPLACES the target's panel when the placement says so. With no target the
 * panel opens as the first service tile beside the agent pane. A split the
 * dock has no room for (each half would fall under the engine's minimum, or,
 * with no `place` given, under {@link READABLE_TILE_PX}) replaces the panel of
 * the target instead, or of the service tile used last
 * when the target is the agent pane's tile; with no service tile either, it is
 * refused `no_room`. A person's opens keep the open rule ({@link openPanel}).
 * @param {Layout|null} layout
 * @param {string} key
 * @param {OpenPlacement} placement
 * @param {OpContext} ctx
 * @returns {OpResult & {refusal?: string, notice?: string}}
 */
export function openPlaced(layout, key, placement, ctx) {
  if (tileOf(layout, key)) return unchanged(layout);
  const target =
    placement.tile === undefined
      ? serviceTileTarget(layout, ctx)
      : findTile(layout, placement.tile);
  if (!target || layout === null) return firstTile(layout, key, ctx.dock);
  const replaceIn = (/** @type {Tile} */ tile) => ({
    layout: mapTiles(layout, (t) =>
      t.tile === tile.tile ? { ...t, tabs: [key], showing: key } : t
    ),
    changed: true,
    focus: tile.tile,
  });
  if (placement.place === 'replace') return replaceIn(target);
  const edge = placement.edge ?? longerSide(layout, target.tile, ctx.dock);
  const readable =
    placement.place === 'split' ||
    halfSpan(layout, target.tile, edge, ctx.dock) >= READABLE_TILE_PX;
  const out = readable
    ? planDrop(layout, { new: key }, { tile: target.tile, edge }, ctx.dock)
    : unchanged(layout);
  const landed = out.changed ? tileOf(out.layout, key) : null;
  if (landed && out.layout !== null)
    return { layout: out.layout, changed: true, focus: landed.tile };
  const fallback = target.showing !== TERMINAL_KEY ? target : serviceTileTarget(layout, ctx);
  if (!fallback) {
    return { ...unchanged(layout), refusal: NO_ROOM, notice: 'no room for a new tile' };
  }
  return replaceIn(fallback);
}

/**
 * The side a default split of tile `id` puts the new tile on: right of a tile
 * wider than tall, below one taller than wide.
 * @param {Layout} layout
 * @param {string} id
 * @param {DockPx} dock
 * @returns {Side}
 */
function longerSide(layout, id, dock) {
  const rect = tileRect(layout, id, dock);
  return rect && rect.width >= rect.height ? 'right' : 'bottom';
}

/**
 * Half of tile `id`'s span along the axis a split on `edge` divides: what each
 * of the two tiles would get.
 * @param {Layout} layout
 * @param {string} id
 * @param {Side} edge
 * @param {DockPx} dock
 * @returns {number}
 */
function halfSpan(layout, id, edge, dock) {
  const rect = tileRect(layout, id, dock);
  if (!rect) return 0;
  return (edge === 'left' || edge === 'right' ? rect.width : rect.height) / 2;
}

/**
 * Tile `id`'s rectangle in the dock, in px; undefined for a tile not in it.
 * @param {Layout} layout
 * @param {string} id
 * @param {DockPx} dock
 */
function tileRect(layout, id, dock) {
  const { width, height } = dock.px;
  return geometry(layout, { x: 0, y: 0, width, height }, dock.minPx).tiles[id];
}

/**
 * Focus `key` where it already is: its tile takes the focus and nothing
 * moves. A key not in the dock changes nothing. The open rule for a panel that
 * is already open.
 * @param {Layout|null} layout
 * @param {string} key
 * @returns {OpResult}
 */
export function focusKey(layout, key) {
  const home = tileOf(layout, key);
  if (!home || layout === null) return unchanged(layout);
  return { layout, changed: false, focus: home.tile };
}

/**
 * Open `key` at an explicit position — a new tile split off the reference
 * tile on the given side, or against the dock's edge when none is named — and
 * MOVE it there when it already holds a tile. Two guarded no-ops: a position
 * naming the panel's own tile (splitting a tile off itself), and a panel
 * already docked with no position at all. Without a position a panel not yet
 * docked opens beside the focused tile, else as the first service tile.
 * @param {Layout|null} layout
 * @param {string} key
 * @param {Position|null} position
 * @param {OpContext} ctx
 * @returns {OpResult}
 */
export function openAt(layout, key, position, ctx) {
  const home = tileOf(layout, key);
  if (!position) {
    if (home) return unchanged(layout);
    const focused = ctx.focusedTile === null ? null : findTile(layout, ctx.focusedTile);
    if (!focused || layout === null) return firstTile(layout, key, ctx.dock);
    position = { referenceTile: focused.tile, direction: 'right' };
  }
  if (!SIDES.has(position.direction)) return unchanged(layout);
  if (home && position.referenceTile === home.tile) return unchanged(layout);
  if (position.referenceTile !== undefined && !findTile(layout, position.referenceTile)) {
    return unchanged(layout);
  }
  const target =
    position.referenceTile === undefined
      ? { outer: position.direction }
      : { tile: position.referenceTile, edge: position.direction };
  const out = planDrop(layout, { new: key }, target, ctx.dock);
  const landed = out.changed ? tileOf(out.layout, key) : null;
  if (!out.changed || !landed) return unchanged(layout);
  return { layout: out.layout, changed: true, focus: landed.tile };
}

/**
 * Bring the terminal back at the dock's right edge, its home side, at its
 * share of the width. An open terminal changes nothing.
 * @param {Layout|null} layout
 * @param {DockPx} dock
 * @returns {OpResult}
 */
export function reopenTerminal(layout, dock) {
  const home = tileOf(layout, TERMINAL_KEY);
  if (home) return { layout, changed: false, focus: home.tile };
  const out = planDrop(layout, { new: TERMINAL_KEY }, { outer: 'right' }, dock);
  const landed = out.changed ? tileOf(out.layout, TERMINAL_KEY) : null;
  if (!out.changed || !landed || out.layout === null) return unchanged(layout);
  return {
    layout: withShare(out.layout, landed.tile, TERMINAL_SHARE),
    changed: true,
    focus: landed.tile,
  };
}

/**
 * Take `key` out of the dock: its tile closes, handing its space to its
 * neighbour (the engine's leave rule). No close record is kept. A key not in
 * the dock changes nothing.
 * @param {Layout|null} layout
 * @param {string} key
 * @returns {OpResult}
 */
export function closeKey(layout, key) {
  if (!tileOf(layout, key)) return unchanged(layout);
  return { layout: removeTab(layout, key).layout, changed: true };
}

/**
 * {@link closeKey} for several keys at once, as one change.
 * @param {Layout|null} layout
 * @param {Iterable<string>} keys
 * @returns {OpResult}
 */
export function closeKeys(layout, keys) {
  let next = layout;
  let changed = false;
  for (const key of keys) {
    const out = closeKey(next, key);
    if (out.changed) {
      next = out.layout;
      changed = true;
    }
  }
  return { layout: next, changed };
}

// ---- a detached pane coming back ----

/**
 * Whether a placement step may be taken: it changed the layout and left one
 * panel per tile, and (`keepDocked`) every panel docked before is still docked.
 * @param {Layout|null} before
 * @param {OpResult|null|undefined} result
 * @param {boolean} keepDocked
 * @returns {boolean}
 */
function acceptsStep(before, result, keepDocked) {
  if (!result || !result.changed || crowdedTile(result.layout) !== null) return false;
  if (!keepDocked) return true;
  const after = new Set(keysOf(result.layout));
  return keysOf(before).every((key) => after.has(key));
}

/**
 * The open rule for a returning pane: the Agent pane at its home side, a
 * service by {@link openPanel}.
 * @param {Layout|null} layout
 * @param {string} key
 * @param {OpContext} ctx
 * @returns {OpResult}
 */
function openReturning(layout, key, ctx) {
  return key === TERMINAL_KEY ? reopenTerminal(layout, ctx.dock) : openPanel(layout, key, ctx);
}

/**
 * Where a detached pane lands (dock-apply.js's `Placement`; dock-workspace
 * hands it to the holder). The entry's record already names its neighbour's
 * current tile, found by key, or null when the neighbour is gone.
 *
 *   - `reattach` (live, R6): beside the old neighbour on the same side — the
 *     engine's `reopen` — and, when there is no room there, nowhere: the
 *     engine's own fallback would crowd a tile, and the open rule would take an
 *     unrelated panel's tile. With the neighbour gone, the open rule, which may
 *     replace the last-used service as any open does (the replaced panel keeps
 *     its sidebar row).
 *   - `persist` (the stored copy, R7): never at a docked panel's cost. Beside
 *     the neighbour; else the Agent pane at its home side, a service right of
 *     the focused tile ({@link openAt}); else at the dock's right edge. Null
 *     when none fits.
 */
export const detachPlacement = Object.freeze({
  /**
   * @param {Layout|null} layout
   * @param {string} key
   * @param {import('./dock-apply.js').DetachedEntry} entry
   * @param {OpContext} ctx
   * @returns {OpResult|null}
   */
  reattach(layout, key, entry, ctx) {
    if (entry.record === null) {
      const opened = openReturning(layout, key, ctx);
      return acceptsStep(layout, opened, false) ? opened : null;
    }
    if (layout === null) {
      const alone = reopen(null, entry.record, ctx);
      return acceptsStep(layout, alone, true) ? alone : null;
    }
    if (neighbourIn(layout, entry)) {
      const beside = reopen(layout, entry.record, ctx);
      return acceptsStep(layout, beside, true) ? beside : null;
    }
    const opened = openReturning(layout, key, ctx);
    return acceptsStep(layout, opened, false) ? opened : null;
  },

  /**
   * @param {Layout|null} layout
   * @param {string} key
   * @param {import('./dock-apply.js').DetachedEntry} entry
   * @param {OpContext} ctx
   * @returns {OpResult|null}
   */
  persist(layout, key, entry, ctx) {
    const { record } = entry;
    const steps = [
      ...(record === null ? [] : [() => reopen(layout, record, ctx)]),
      () =>
        key === TERMINAL_KEY ? reopenTerminal(layout, ctx.dock) : openAt(layout, key, null, ctx),
      () => openAt(layout, key, { direction: 'right' }, ctx),
    ];
    for (const step of steps) {
      const result = step();
      if (acceptsStep(layout, result, true)) return result;
    }
    return null;
  },
});

/**
 * Whether the entry's record names a neighbour tile the layout holds.
 * @param {Layout} layout
 * @param {import('./dock-apply.js').DetachedEntry} entry
 * @returns {boolean}
 */
function neighbourIn(layout, entry) {
  const { record } = entry;
  return (
    record?.kind === 'tile' && record.neighbour !== null && !!findTile(layout, record.neighbour)
  );
}

/**
 * The default arrangement: the terminal card alone in one tile. Services join
 * it left of the terminal at the classic split when they open.
 * @returns {Layout}
 */
export function defaultLayout() {
  const tile = mintTileId(null);
  return { tile, tabs: [TERMINAL_KEY], showing: TERMINAL_KEY };
}

// ---- sets and arrangements ----

/** @param {string} key */
function isService(key) {
  return key !== TERMINAL_KEY;
}

/**
 * @param {SetContext} ctx
 * @param {string} key
 */
function healthy(ctx, key) {
  return ctx.isHealthy === undefined || ctx.isHealthy(key);
}

/**
 * `ctx.skip` plus every member that is not open and not healthy.
 * @param {Layout|null} layout
 * @param {string[]} members
 * @param {SetContext} ctx
 */
function skipOf(layout, members, ctx) {
  const unhealthy = members.filter((key) => !tileOf(layout, key) && !healthy(ctx, key));
  return [...new Set([...(ctx.skip ?? []), ...unhealthy])];
}

/**
 * The engine context of a set: anchored on the service tile the operator used
 * last, with the agent pane never the anchor.
 * @param {Layout|null} layout
 * @param {SetContext} ctx
 * @param {string[]} skip
 * @returns {OpenContext}
 */
function setContext(layout, ctx, skip) {
  return {
    focusedTile: serviceTileTarget(layout, ctx)?.tile ?? null,
    dock: ctx.dock,
    skip,
  };
}

/**
 * The first-tile case of a set: its first member opens as the first service
 * tile ({@link firstTile}) and the rest share its span side by side.
 * @param {Layout} layout
 * @param {string[]} fresh  members to open: none open, all healthy
 * @param {SetContext} ctx
 * @returns {SetResult}
 */
function firstTileSet(layout, fresh, ctx) {
  const first = firstTile(layout, fresh[0], ctx.dock);
  if (!first.changed || !first.focus || first.layout === null) return { layout, changed: false };
  const id = first.focus;
  const rest = fresh.slice(1);
  if (rest.length === 0) return { ...first, focus: id };
  const side = openSet(first.layout, rest, 'side', undefined, { focusedTile: id, dock: ctx.dock });
  if (side.refusal !== undefined) return { layout, changed: false, refusal: side.refusal };
  return { ...side, focus: id };
}

/**
 * Open a set of panels side by side beside the service tile the operator used
 * last; `as: 'layout'` is {@link arrange}. With no service tile the set opens
 * as the first service tile beside the agent pane and the rest beside it.
 * Members not open and not healthy are skipped (`skipped`); a set naming the
 * agent pane is refused.
 * @param {Layout|null} layout
 * @param {string[]} members
 * @param {'side' | 'layout'} as
 * @param {PanelCollection | undefined} given
 * @param {SetContext} ctx
 * @returns {SetResult}
 */
export function openPanelSet(layout, members, as, given, ctx) {
  if (members.includes(TERMINAL_KEY)) {
    return { layout, changed: false, refusal: TERMINAL_IN_LAYOUT };
  }
  if (as === 'layout' && given !== undefined) return arrange(layout, given, ctx);
  const skip = skipOf(layout, members, ctx);
  const engine = setContext(layout, ctx, skip);
  const open = new Set(keysOf(layout));
  const fresh = members.filter((key) => !skip.includes(key) && !open.has(key));
  const terminal = tileOf(layout, TERMINAL_KEY);
  if (layout === null || terminal === null || fresh.length === 0 || engine.focusedTile !== null) {
    return openSet(layout, members, 'side', undefined, engine);
  }
  const skipped = members.filter((key) => skip.includes(key));
  const out = firstTileSet(layout, fresh, ctx);
  return skipped.length > 0 ? { ...out, skipped } : out;
}

/**
 * `services` with any tile whose id the agent pane's tile holds re-taken from
 * the engine's minter, so the composite never repeats an id.
 * @param {Layout} services
 * @param {string} keep
 * @returns {Layout}
 */
function avoidId(services, keep) {
  if (!tileIdsOf(services).includes(keep)) return services;
  const fresh = mintTileId(services);
  return mapTiles(services, (tile) => (tile.tile === keep ? { ...tile, tile: fresh } : tile));
}

/**
 * The arrangement with the agent pane's tile `terminal` kept in place and size:
 * a two-item split of the services and the agent pane along the axis it spans
 * fully (else the root's), on its side, giving it its old extent. An agent
 * pane that was the whole dock gets the classic split, services left.
 * @param {Layout} layout
 * @param {Tile} terminal
 * @param {Layout} services
 * @param {DockPx} dock
 * @returns {Layout}
 */
function compose(layout, terminal, services, dock) {
  const term = { tile: terminal.tile, tabs: [TERMINAL_KEY], showing: TERMINAL_KEY };
  if (isTile(layout)) {
    return normalize({
      split: 'row',
      items: [
        { ...services, size: SERVICE_SHARE },
        { ...term, size: TERMINAL_SHARE },
      ],
    });
  }
  const { width, height } = dock.px;
  const rect = geometry(layout, { x: 0, y: 0, width, height }, dock.minPx).tiles[terminal.tile];
  const fullHeight = rect.height >= height - 1;
  const fullWidth = rect.width >= width - 1;
  const split = fullHeight ? 'row' : fullWidth ? 'column' : layout.split;
  const axis = split === 'row' ? 'width' : 'height';
  const span = axis === 'width' ? width : height;
  const offset = axis === 'width' ? rect.x : rect.y;
  const room = span - minimumSize(services, axis, dock.minPx);
  const share = Math.min(Math.max(rect[axis], dock.minPx), Math.max(room, dock.minPx)) / span;
  const before = offset + rect[axis] / 2 < span / 2;
  const items = [
    { ...services, size: 1 - share },
    { ...term, size: share },
  ];
  return normalize({ split, items: before ? items.reverse() : items });
}

/**
 * The arrange composer: a service-only collection layout of panel ids replaces
 * the dock's service tiles. The agent pane's tile keeps its place and size (it
 * is always its own tile, so no service shares it); with no service tile the
 * layout goes beside it at the classic split; with no agent pane it is the
 * whole dock. A layout giving a tile several panels is refused by the
 * policy's name. Members not open and not healthy are skipped; open services
 * the layout does not name leave the dock and are listed in `pruned`.
 * @param {Layout|null} layout
 * @param {PanelCollection} given
 * @param {SetContext} ctx
 * @returns {SetResult}
 */
export function arrange(layout, given, ctx) {
  const check = validateLayout(given, { form: 'collection' });
  if (!check.valid) return { layout, changed: false, refusal: check.reason };
  const wanted = keysOf(given);
  if (wanted.some((key) => typeof key !== 'string')) {
    return { layout, changed: false, refusal: LAYOUT_REASONS.badKey };
  }
  if (wanted.includes(TERMINAL_KEY)) return { layout, changed: false, refusal: TERMINAL_IN_LAYOUT };
  if (crowdedTile(/** @type {any} */ (given)) !== null) {
    return { layout, changed: false, refusal: ONE_PANEL_PER_TILE };
  }
  const skip = skipOf(layout, wanted, ctx);
  const built = openSet(null, wanted, 'layout', given, { focusedTile: null, dock: ctx.dock, skip });
  const skipped = built.skipped ?? [];
  if (!built.changed || built.layout === null) return { layout, changed: false, skipped };
  const terminal = tileOf(layout, TERMINAL_KEY);
  const services = terminal === null ? built.layout : avoidId(built.layout, terminal.tile);
  const next =
    terminal === null || layout === null ? services : compose(layout, terminal, services, ctx.dock);
  const kept = new Set(keysOf(next));
  const pruned = keysOf(layout).filter((key) => isService(key) && !kept.has(key));
  return {
    layout: next,
    changed: JSON.stringify(next) !== JSON.stringify(layout),
    focus: tileIdsOf(services)[0],
    record: { kind: 'layout', layout },
    skipped,
    pruned,
  };
}
