// @ts-check
/* OSPREY Web Terminal — Panel SSE
 *
 * The server→client half of the panel workspace. Extracted from
 * panel-manager.js, which still owns the panel state machine as a whole (tile
 * occupancy, iframe lifecycle, the command-palette accessors); this module owns
 * the /api/files/events subscription, the frame dispatcher behind it, and the
 * apply paths those frames share with the reconnect resync.
 *
 * The boundary is direction of travel: everything here REACTS to a frame the
 * server sent — nothing in this module is reached by a human gesture. The
 * only POSTs are answers the server asked for: the outcome of a `panel_op`
 * frame addressed to this dock client (/api/panel-op-result), and the forced
 * layout report a `panel_report_request` frame asks dock-sync's reporter for.
 * The command half lives in panel-commands.js.
 *
 * Leaf modules are imported directly; everything that lives in panel-manager's
 * private state arrives as injected deps registered once when the stream opens
 * ({@link subscribePanelEvents}) — the same seam pattern panel-lifecycle.js and
 * panel-placement.js use. That injection is what keeps this module out of an
 * import cycle: panel-manager imports it, never the reverse
 * (panel-manager → panel-sse → panel-lifecycle). It also keeps the dispatcher
 * unit-testable through panel-manager's own surface, since it holds no panel
 * state of its own.
 */

import { fetchJSON, createEventSource, getPageClientId, pageEventsUrl } from './api.js';
import { hidePanel } from './dock-iframe.js';
import { withEchoSuppressed, noteReported, reportLayoutNow, sameReport } from './dock-sync.js';
import { getLayoutHolder, TERMINAL_KEY } from './dock-apply.js';
import { serviceTileTarget } from './dock-open.js';
import {
  flashAgentGlow,
  flashAgentTile,
  badgePanelActivity,
  restoreAgentBadges,
} from './panel-agent-attention.js';
import { AGENT_ACTIVITY_FRAME } from './activity-format.js';
import { PANELS, TERMINAL_RAIL_ID } from './panel-catalog.js';
import {
  ensureRailMembership,
  addPanel,
  applyRailOrder,
  rereadPanelConfig,
} from './panel-lifecycle.js';
import { entryIds, placeEntry } from './panel-rail.js';
import { runPanelOp } from './panel-placement.js';
import { postPanelOpResult } from './panel-commands.js';
import { focusPopout } from './popout.js';
import {
  initWatchdogAlert,
  isWatchdogFrame,
  onWatchdogFrame,
  refetchFlags,
} from './watchdog-alert.js';
import { isSafetySwitchFrame, refreshSafetySwitches } from './safety-switches.js';
import { refreshCodeReviewTab } from './code-review-tab.js';

/** @typedef {import('./panel-manager.js').PanelSSEEvent} PanelSSEEvent */
/** @typedef {import('./panel-manager.js').AgentActivityEvent} AgentActivityEvent */

/**
 * The panel-manager state this module reads and mutates. Containers arrive by
 * live reference (they are never reassigned); reassigned scalars arrive as
 * getters and setters, so a value read here is always the one panel-manager
 * holds now — and `reportActivity` closes over the activity-strip seam rather
 * than capturing whichever handler happened to be registered at init.
 * @typedef {object} SseDeps
 * @property {(id: string) => boolean} isKnown - does this page know the panel at all
 * @property {(id: string) => boolean} isHealthy - may the panel fill a tile
 * @property {() => HTMLElement} getRailEl - the rail nav the entries live in
 * @property {Set<string>} memberSet - the server-owned rail membership set
 * @property {(id: string) => void} dropMember - drop membership AND the rail entry
 * @property {(id: string, options?: {userInitiated?: boolean, auto?: boolean}) => void} activate
 * @property {(id: string, url: string) => void} navigate - point a panel's iframe at `url`
 * @property {() => string | null} getActive - the locally surfaced panel id
 * @property {() => void} resetActive - forget the surfaced panel, leaving the rail accent alone
 * @property {() => void} clearActive - forget the surfaced panel AND clear its rail accent and stamp
 * @property {(message: string) => void} renderEmpty - paint the empty-workspace placeholder
 * @property {(frame: AgentActivityEvent) => void} reportActivity - hand a frame to the strip seam
 */

/** @type {SseDeps | null} */
let deps = null;

/**
 * The registered deps, or a hard failure. A frame arriving before the stream
 * was opened is impossible by construction; reaching this without deps is a
 * wiring bug, not a runtime condition to degrade around.
 * @returns {SseDeps}
 */
function ctx() {
  if (!deps) throw new Error('panel-sse: subscribePanelEvents() has not run');
  return deps;
}

/**
 * Bind the panel-manager state this module works through and open the panel
 * event stream. Called once, as the last statement of initPanelManager — every
 * other init (placement, menu policy, lifecycle, the rail render) must already
 * have run, because the first frame can arrive the moment the stream opens.
 * @param {SseDeps} sseDeps
 */
export function subscribePanelEvents(sseDeps) {
  deps = sseDeps;
  initWatchdogAlert();

  // Listen for SSE events via createEventSource (api.js) so the URL picks up
  // window.__OSPREY_PREFIX__ under multi-user deployments (empty prefix ⇒
  // unchanged behavior). These event types are handled:
  //
  //   panel_op         {type, request_id, target_client, op, args, source, base_rev?}
  //                                             — one dock op (an agent's drop,
  //                                               close, open, set or
  //                                               arrangement, or a person's
  //                                               preset click), addressed to
  //                                               ONE dock client: every other
  //                                               client ignores it. The op
  //                                               runner (panel-placement.js)
  //                                               applies it to the holder and
  //                                               the outcome is POSTed back.
  //   panel_report_request {type}               — the current dock client went
  //                                               away: report this dock's
  //                                               layout now, undeduped, so a
  //                                               surviving dock becomes current.
  //   panel_visibility {type, panel, visible, index?}
  //                                             — add/remove a rail entry (an
  //                                               addition with an index goes
  //                                               to that position); removing
  //                                               also closes the tile (an
  //                                               unlaunchable panel must not be
  //                                               stranded on screen), then switches
  //                                               to the next member+healthy panel
  //                                               or the empty state.
  //   rail_order       {type, order}            — an operator reordered the rail:
  //                                               every member, top to bottom.
  //   panel_register   {type, id, label, url, healthEndpoint, path}
  //                                             — add a runtime panel; do NOT
  //                                               auto-activate (URL may not be ready).
  //   agent_activity   {type, tool, target, ts} — passive "the agent touched X"
  //                                               signal: badge+glow the rail entry
  //                                               for target.kind 'panel', otherwise
  //                                               hand off to the strip seam.
  //   watchdog_*       {type, session_key, ...} — a change notice: handed to
  //                                               watchdog-alert.js, which re-reads
  //                                               the flags from their route.
  //   safety_switch    {type, name, on}         — a runtime safety switch moved:
  //                                               safety-switches.js and the Review
  //                                               tab re-read them.
  //
  // panel_op, panel_visibility and panel_register may carry source:'agent'
  // (agent-originated command): that adds a transient glow — never a
  // persistent badge, the action itself already happened.
  //
  // The stream carries this page's dock client id, so the server knows which
  // stream belongs to the current dock client (and asks the others to report
  // when it closes). It is the page's events URL (api.js pageEventsUrl), the
  // very URL the chip and the activity strip subscribe to: one shared socket.
  let streamOpened = false;
  createEventSource(pageEventsUrl(), {
    // prefixed via createEventSource (api.js)
    // SSE has no replay: anything broadcast while this client was
    // disconnected (or dropped by the server's bounded per-client queue) is
    // gone for good, and every panel command's DOM effect IS its SSE echo —
    // without a resync a client that missed one frame never converges again.
    // The hook fires on every open, including the first; the extra boot-time
    // fetch is a no-op delta.
    // Badges are restored from the history buffer after the membership delta, so
    // an entry the resync just re-added can carry one (restoreAgentBadges).
    onOpen: () => {
      void resyncPanelState().then(restoreAgentBadges);
      void refetchFlags();
      void refreshSafetySwitches();
      // The first open follows the config reads of this page's boot. A later
      // one may be a different hub: what each panel is served at is read again.
      if (streamOpened) for (const panel of PANELS) void rereadPanelConfig(panel);
      streamOpened = true;
    },
    onMessage: (raw) => {
      try {
        // The transport frames are typed here (panel-placement.js's
        // PanelOpFrame); the rest are panel-manager's PanelSSEEvent union.
        const kind = /** @type {{type?: unknown}} */ (raw)?.type;
        if (kind === 'panel_op') {
          const frame = /** @type {import('./panel-placement.js').PanelOpFrame} */ (raw);
          // Only the addressed dock client acts on an op.
          if (frame.target_client === getPageClientId()) void handlePanelOp(frame);
          return;
        }
        if (kind === 'panel_report_request') {
          reportLayoutNow();
          return;
        }
        const data = /** @type {PanelSSEEvent} */ (raw);
        const c = ctx();

        if (data.type === 'panel_visibility' && data.panel) {
          applyPanelVisibility(data.panel, data.visible, data.source, data.index);
        } else if (data.type === 'rail_order' && Array.isArray(data.order)) {
          applyRailOrder(data.order);
        } else if (data.type === 'panel_register' && data.id) {
          // Seed membership before addPanel so the appended entry is a member
          c.memberSet.add(data.id);
          addPanel(data);
          // Do NOT call activateTab — the new panel's URL may not be ready yet;
          // the user activates when they want it.
          if (data.source === 'agent') flashAgentGlow(data.id);
        } else if (data.type === AGENT_ACTIVITY_FRAME && data.target) {
          // kind 'panel' with a live rail entry → persistent badge + glow via
          // badgePanelActivity; everything the rail cannot anchor falls through
          // to the activity-strip seam (no-op until a handler registers).
          const t = data.target;
          if (!(t.kind === 'panel' && t.panel && badgePanelActivity(t.panel, data.ts))) {
            c.reportActivity(data);
          }
        } else if (isWatchdogFrame(data)) {
          onWatchdogFrame(data);
        } else if (isSafetySwitchFrame(data)) {
          void refreshSafetySwitches();
          void refreshCodeReviewTab();
        }
      } catch (err) {
        // A throw mid-dispatch can leave rail membership, the dock layout and
        // the overlay's managed set describing different workspaces — never
        // swallow it without evidence.
        console.error('panel-sse: SSE frame dispatch failed', err, raw);
      }
    },
  });
}

/**
 * Apply a server-visible membership change — the single body behind the
 * panel_visibility SSE branch and the reconnect resync, so a resynced client
 * ends up exactly where a connected one would be.
 *
 * Order matters and is pinned by the test suite: membership + rail entry
 * first (the agent glow runs after the add so a just-added entry can flash,
 * and an agent-origin change reports itself on the activity strip);
 * then, on a hide, the dock tile drop (a closed panel's placeholder would be a
 * ghost tile) under the echo guard (dockview auto-activating a neighbor on
 * removal is a server-applied echo and must not POST focus back), and finally
 * the active-panel fallback so a hidden active panel never strands a blank pane.
 *
 * An addition with an `index` lands at that position among the rail's panel
 * entries, counted without Agent — the same count the operator's Undo took
 * at removal — so every client puts a restored entry back where it was, not
 * only the one that clicked. A panel that was already a member keeps its
 * place.
 * @param {string} panel
 * @param {boolean} visible
 * @param {'agent'} [source]
 * @param {number} [index]
 */
function applyPanelVisibility(panel, visible, source, index) {
  const c = ctx();
  const wasMember = c.memberSet.has(panel);
  if (visible) {
    ensureRailMembership(panel);
    if (!wasMember && index !== undefined) {
      const railEl = c.getRailEl();
      const others = entryIds(railEl).filter((id) => id !== TERMINAL_RAIL_ID && id !== panel);
      placeEntry(railEl, panel, others[index] ?? null);
    }
  } else {
    c.dropMember(panel);
  }
  // An addition can glow its just-created rail entry; a removal has no entry
  // left to glow, so the strip is the only surface that can report it. Both
  // synthesize an activity frame — deliberately straight to the seam, past the
  // agent_activity branch's rail-anchor routing, so the two halves of the
  // agent's rail vocabulary read the same way on the strip.
  if (visible && source === 'agent') flashAgentGlow(panel);
  if (source === 'agent')
    c.reportActivity({
      type: AGENT_ACTIVITY_FRAME,
      ts: Date.now(),
      tool: visible ? 'add_panel_to_rail' : 'remove_panel_from_rail',
      target: { kind: 'panel', panel },
    });

  // A panel the operator can no longer launch must not be stranded on screen.
  // A hide of a panel that was never on the rail changed nothing, so a tile it
  // holds is one opened off the rail on purpose and stays.
  if (!visible && wasMember) closeTile(panel);
}

/**
 * Close `panel`'s tile and, when it held the view, hand the view to a panel
 * the operator can still see, without touching rail membership.
 *
 * The removal half of `panel_visibility` (membership is already gone by the
 * time this runs). The other tiles stay as they are: the view goes to the
 * panel in the service tile the holder picks (an activation of a docked panel moves
 * nothing; one that is not answering keeps its tile and takes no accent), and
 * only a dock left with no service tile at all opens the next member — the
 * fallback search reads the CURRENT membership set, so it naturally skips the
 * panel that was just removed. A fallback into a dock that still shows
 * service tiles would replace one of them (the open rule), taking away a
 * panel the operator did not touch.
 * @param {string} panel
 */
function closeTile(panel) {
  const c = ctx();
  withEchoSuppressed(() => hidePanel(panel));
  if (panel !== c.getActive()) return;
  const holder = getLayoutHolder();
  const docked = holder ? serviceTileTarget(holder.layout, holder.context())?.showing : undefined;
  const fallback =
    docked === undefined
      ? PANELS.find((p) => p.id !== panel && c.memberSet.has(p.id) && c.isHealthy(p.id))?.id
      : c.isHealthy(docked)
        ? docked
        : undefined;
  if (fallback) {
    c.activate(fallback);
  } else {
    c.resetActive();
    if (docked === undefined) c.renderEmpty('No panels visible');
  }
}

/** A refusal for an op the runner threw on (a bug, reported rather than hung). */
const OP_FAILED = 'op_failed';
/** The server caps a refusal's detail at this many characters. */
const DETAIL_CAP = 500;

/**
 * Every tile of a layout, in tree order.
 * @param {any} node
 * @returns {{tile: string, tabs: string[], showing: string}[]}
 */
function tilesOf(node) {
  if (!node) return [];
  return Array.isArray(node.tabs) ? [node] : (node.items ?? []).flatMap(tilesOf);
}

/**
 * Surface what an op put on screen. A placed panel has no iframe until it
 * is activated, so every service panel the op newly SHOWS is activated (the tile
 * the op focused last, so focus ends there), and the focused tile's panel is
 * activated even when it already showed (an open of a panel already on screen
 * still brings it forward). The activation places nothing new: each panel is
 * already where the op put it. A panel the op took off screen while it held
 * the view leaves the accent stale; with nothing surfaced (no activation took
 * effect — an unhealthy panel refuses one) it is cleared, and only an empty
 * service region paints the empty pane: a tile still on screen is never
 * painted over.
 * @param {any} before  the layout before the op
 * @param {any} after   the layout after it
 * @param {string | null | undefined} focus  the tile the op focused
 * @param {boolean} agent  attribute the surfaced panels to the agent
 */
function surfaceOp(before, after, focus, agent) {
  const c = ctx();
  const shownBefore = new Set(tilesOf(before).map((t) => t.showing));
  const tiles = tilesOf(after).filter((t) => t.showing !== TERMINAL_KEY);
  const focused = tiles.find((t) => t.tile === focus) ?? null;
  const surfaced = tiles
    .filter((t) => t !== focused && !shownBefore.has(t.showing))
    .map((t) => t.showing);
  if (focused) surfaced.push(focused.showing);
  // Only an activation that took effect surfaced anything: activate refuses an
  // unhealthy panel, and a refused one must not keep a stale accent alive.
  let tookEffect = 0;
  for (const key of surfaced) {
    c.activate(key);
    if (c.getActive() === key) tookEffect++;
    if (agent) flashAgentTile(key);
  }
  const active = c.getActive();
  const keys = new Set(tilesOf(after).flatMap((t) => t.tabs));
  if (tookEffect > 0 || active === null || keys.has(active)) return;
  // A refused activation leaves the view on a panel the op took away: clear
  // the accent and the stamp too. Nothing tried (a close) keeps the rail
  // accent on the panel that is one click away again.
  if (surfaced.length > 0) c.clearActive();
  else c.resetActive();
  if (tiles.length === 0) c.renderEmpty('No panels visible');
}

/**
 * Run one op on the holder through the op runner; a runner that throws is
 * logged and comes back as the `op_failed` refusal.
 * @param {import('./panel-placement.js').PanelOpFrame} frame
 * @returns {import('./panel-placement.js').PanelOpOutcome}
 */
function runOp(frame) {
  try {
    return runPanelOp(frame);
  } catch (err) {
    console.error('panel-sse: panel op failed', err, frame);
    return { refusal: OP_FAILED, detail: String(err) };
  }
}

/**
 * Show what an op the holder took put on screen: surface the placed panels,
 * or bring a panel's own window forward, and navigate an `open` with a `url`
 * once the panel is placed.
 * @param {import('./panel-placement.js').PanelOpFrame} frame
 * @param {{result: {focus?: string | null}, popout?: string}} outcome
 * @param {any} before  the layout before the op
 * @param {any} after   the layout the holder holds now
 */
function showOutcome(frame, outcome, before, after) {
  const args = frame.args ?? {};
  const url = frame.op === 'open' && typeof args.url === 'string' ? args.url : null;
  if (outcome.popout !== undefined) {
    if (url !== null) ctx().navigate(args.member, url);
    focusPopout(outcome.popout);
    if (frame.source === 'agent') flashAgentTile(outcome.popout);
    return;
  }
  surfaceOp(before, after, outcome.result.focus, frame.source === 'agent');
  if (url !== null) ctx().navigate(args.member, url);
}

/**
 * Run one `panel_op` frame addressed to this dock client and answer it.
 *
 * The op runner (panel-placement.js) refuses a stale or terminal-moving op and
 * otherwise runs it on the layout holder, which applies the result (held while
 * a person's drag is live). The placed panels are then surfaced, an `open` with
 * a `url` navigates the panel's iframe, and the layout the holder now holds is
 * POSTed to /api/panel-op-result with the pop-outs beside it; the `rev` that
 * answer returns is the holder's new ack baseline — unless the layout moved on
 * meanwhile, which dock-sync then reports. An `open` of a panel in its own
 * window brings that window forward and answers the layout unchanged. The
 * answer may come after the server stopped waiting (`no_dock_answer`); the
 * server still records it, as the layout report it is.
 *
 * Every frame is answered. A refusal is POSTed by name only while the holder
 * still holds the layout it had before the op: then nothing changed. Once the
 * holder has taken the op, its layout is the answer whatever fails after — a
 * surfacing, navigation or window error, or a runner that throws after the
 * holder took the result — because the op happened; the error is logged.
 * @param {import('./panel-placement.js').PanelOpFrame} frame
 */
async function handlePanelOp(frame) {
  const answer = { request_id: frame.request_id, client_id: getPageClientId() };
  const holder = getLayoutHolder();
  const before = holder?.layout ?? null;
  const outcome = runOp(frame);
  if (!holder || (outcome.refusal !== undefined && holder.layout === before)) {
    // Nothing was applied, so the refusal is the answer. A page with no holder
    // gets the runner's own refusal; op_failed covers a runner that did not.
    const detail =
      outcome.refusal !== undefined && outcome.detail
        ? { detail: outcome.detail.slice(0, DETAIL_CAP) }
        : {};
    await postPanelOpResult({ ...answer, refusal: outcome.refusal ?? OP_FAILED, ...detail });
    return;
  }
  if (outcome.refusal === undefined) {
    try {
      showOutcome(frame, outcome, before, holder.layout);
    } catch (err) {
      console.error(
        'panel-sse: showing an applied panel op failed; answering its layout',
        err,
        frame
      );
    }
  }
  const layout = holder.layout;
  const popouts = holder.popouts;
  recordAck(holder, { layout, popouts }, await postPanelOpResult({ ...answer, layout, popouts }));
}

/**
 * Record the server's acknowledgement of a layout this module sent. The server
 * holds `sent` now, so it becomes dock-sync's dedupe baseline; its `rev` is the
 * holder's ack baseline only while the holder still holds `sent` (up to
 * dockview's size rounding, as dock-sync compares reports) — a layout that
 * moved on meanwhile is not acked here, and dock-sync reports it
 * (noteReported). An answer that did not land, or carries no `rev`, records
 * nothing.
 * @param {{layout: unknown, popouts: string[], ack: (rev: number) => void}} holder
 * @param {{layout: unknown, popouts: string[]}} sent  what was sent
 * @param {{rev?: number} | null} ack  the server's answer, or null
 */
function recordAck(holder, sent, ack) {
  if (!ack || typeof ack.rev !== 'number') return;
  noteReported(sent);
  if (sameReport({ layout: holder.layout, popouts: holder.popouts }, sent)) holder.ack(ack.rev);
}

/**
 * Re-converge rail membership with the server after the SSE stream (re)opens.
 * The authoritative visible set is re-fetched and applied as a delta through
 * applyPanelVisibility — the same path the panel_visibility frames use.
 * Scope: membership of panels known to this page. A runtime panel whose
 * panel_register frame was missed entirely stays unknown until reload.
 */
async function resyncPanelState() {
  const c = ctx();
  let config = null;
  try {
    config = await fetchJSON('/api/panels');
  } catch {
    return; // offline — the next reconnect retries
  }
  if (!config || !Array.isArray(config.visible)) return;
  const serverVisible = new Set(config.visible);
  for (const id of serverVisible) {
    if (!c.memberSet.has(id) && c.isKnown(id)) applyPanelVisibility(id, true);
  }
  for (const id of [...c.memberSet]) {
    if (!serverVisible.has(id)) applyPanelVisibility(id, false);
  }
  // A missed rail_order frame is converged from the same read.
  applyRailOrder(config.visible);
}
