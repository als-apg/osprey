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
 * server broadcast — nothing in this module is reached by a human gesture, and
 * nothing here POSTs (the command half lives in panel-commands.js, and the DOM
 * effect of every command IS its SSE echo arriving back through here).
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

import { fetchJSON, createEventSource } from './api.js';
import { hidePanel } from './dock-iframe.js';
import { withEchoSuppressed } from './dock-sync.js';
import {
  flashAgentGlow, flashAgentTile, badgePanelActivity, restoreAgentBadges,
} from './panel-agent-attention.js';
import { AGENT_ACTIVITY_FRAME } from './activity-format.js';
import { PANELS } from './panel-catalog.js';
import { ensureRailMembership, addPanel } from './panel-lifecycle.js';
import { applyAgentSwitch, applyArrange } from './panel-placement.js';

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
 * @property {Set<string>} memberSet - the server-owned rail membership set
 * @property {(id: string) => void} dropMember - drop membership AND the rail entry
 * @property {(id: string, options?: {userInitiated?: boolean, auto?: boolean}) => void} activate
 * @property {(id: string, url: string) => void} navigate - point a panel's iframe at `url`
 * @property {() => string | null} getActive - the locally surfaced panel id
 * @property {() => void} resetActive - forget the surfaced panel, leaving the rail accent alone
 * @property {() => boolean} isWorkspaceSuppressed - simple-UX chat-only boot still armed
 * @property {() => void} endWorkspaceSuppression - one-way exit from that boot state
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

  // Listen for SSE events via createEventSource (api.js) so the URL picks up
  // window.__OSPREY_PREFIX__ under multi-user deployments (empty prefix ⇒
  // unchanged behavior). createEventSource also drives the module-level
  // sseState in api.js, but nothing currently reads getConnectionState().sse
  // (only .ws is consumed, by app.js's status dot), so that side effect is
  // harmless. These event types are handled:
  //
  //   panel_focus      {type, panel, url?}      — explicit open_panel MCP call
  //                                               or the echo of a human focus
  //                                               gesture; always honor. An
  //                                               agent-tagged frame surfaces
  //                                               the panel without evicting
  //                                               anything (applyAgentSwitch);
  //                                               a human echo takes the tile
  //                                               over as it always has.
  //   panel_close      {type, panel}            — close_panel MCP call: close the
  //                                               panel's tile and leave rail
  //                                               membership alone, so the panel
  //                                               stays one click away. The
  //                                               on-screen half of panel_visibility.
  //   panel_visibility {type, panel, visible}   — add/remove a rail entry; removing
  //                                               also closes the tile (an
  //                                               unlaunchable panel must not be
  //                                               stranded on screen), then switches
  //                                               to the next member+healthy panel
  //                                               or the empty state.
  //   panel_arrange    {type, tiles, focus?, prune_rail?}
  //                                             — a whole-workspace layout
  //                                               request: exactly these
  //                                               service tiles, left to right
  //                                               (see panel-placement.js).
  //   panel_register   {type, id, label, url, healthEndpoint, path}
  //                                             — add a runtime panel; do NOT
  //                                               auto-activate (URL may not be ready).
  //   agent_activity   {type, tool, target, ts} — passive "the agent touched X"
  //                                               signal: badge+glow the rail entry
  //                                               for target.kind 'panel', otherwise
  //                                               hand off to the strip seam.
  //
  // The first three may carry source:'agent' (agent-originated command): that
  // adds a transient glow on the rail entry — never a persistent badge, the
  // action itself already happened. Untagged frames behave exactly as before.
  createEventSource('/api/files/events', { // prefixed via createEventSource (api.js)
    // SSE has no replay: anything broadcast while this client was
    // disconnected (or dropped by the server's bounded per-client queue) is
    // gone for good, and every panel command's DOM effect IS its SSE echo —
    // without a resync a client that missed one frame never converges again.
    // The hook fires on every open, including the first; the extra boot-time
    // fetch is a no-op delta.
    // Badges are restored from the history ring after the membership delta, so
    // an entry the resync just re-added can carry one (restoreAgentBadges).
    onOpen: () => { void resyncPanelState().then(restoreAgentBadges); },
    onMessage: (raw) => {
      try {
        const data = /** @type {PanelSSEEvent} */ (raw);
        const c = ctx();

        if (data.type === 'panel_focus' && data.panel) {
          // A broadcast switch also ends the simple-UX chat-only suppression,
          // even when the activation still refuses (unhealthy panel): the
          // intent to surface the workspace is clear, so the next health
          // settle may fill the slot.
          c.endWorkspaceSuppression();
          if (data.url) c.navigate(data.panel, data.url);
          // An AGENT switch is polite: focus the panel's own tile, or open one
          // beside the operator's — never take a tile away (applyAgentSwitch).
          // Human focus is never broadcast (the server mirrors it silently and
          // the gesturing client applies it locally), so an unattributed frame
          // can only come from an out-of-contract caller; it keeps the plain
          // activation. The attribution runs after the switch so a just-added
          // entry can flash and the tile glow measures the tile the switch
          // actually surfaced.
          if (data.source === 'agent') {
            applyAgentSwitch(data.panel);
            flashAgentTile(data.panel);
          } else {
            c.activate(data.panel);
          }

        } else if (data.type === 'panel_close' && data.panel) {
          applyPanelClose(data.panel, data.source);

        } else if (data.type === 'panel_visibility' && data.panel) {
          applyPanelVisibility(data.panel, data.visible, data.source);

        } else if (data.type === 'panel_arrange' && Array.isArray(data.tiles)) {
          // A whole-workspace arrangement (agent arrange_workspace, or a human
          // "Layouts" click — one server operation, one apply path). Precedence
          // against the visibility channel above: a hide keeps its existing
          // meaning everywhere, while an arrange only ADDS membership for the
          // tiles it lists — except on the preset path, where prune_rail
          // reproduces today's membership-exclusive semantics. See
          // panel-placement.js's header for the full split.
          applyArrange(data);

        } else if (data.type === 'panel_register' && data.id) {
          // Seed membership before addPanel so the appended entry is a member
          c.memberSet.add(data.id);
          addPanel(data);
          // CC-3: do NOT call activateTab — the new panel's URL may not be ready yet;
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
 * then the simple-UX chat-only reveal (showing a panel while the workspace is
 * suppressed brings the workspace up ON that panel — {auto: true} keeps the
 * health guard); then, on a hide, the dock tile drop (one panel per tile — a
 * closed panel's placeholder would be a ghost tile) under the echo guard
 * (dockview auto-activating a neighbor on removal is a server-applied echo
 * and must not POST focus back), and finally the active-panel fallback so a
 * hidden active panel never strands a blank pane.
 * @param {string} panel
 * @param {boolean} visible
 * @param {'agent'} [source]
 */
function applyPanelVisibility(panel, visible, source) {
  const c = ctx();
  if (visible) {
    ensureRailMembership(panel);
  } else {
    c.dropMember(panel);
  }
  // An addition can glow its just-created rail entry; a removal has no entry
  // left to glow, so the strip is the only surface that can report it. Both
  // synthesize an activity frame — deliberately straight to the seam, past the
  // agent_activity branch's rail-anchor routing, so the two halves of the
  // agent's rail vocabulary read the same way on the strip.
  if (visible && source === 'agent') flashAgentGlow(panel);
  if (source === 'agent') c.reportActivity({ type: AGENT_ACTIVITY_FRAME, ts: Date.now(),
    tool: visible ? 'add_panel_to_rail' : 'remove_panel_from_rail',
    target: { kind: 'panel', panel } });

  if (visible && c.isWorkspaceSuppressed()) {
    c.endWorkspaceSuppression();
    if (!c.getActive()) c.activate(panel, { auto: true });
  }

  // A panel the operator can no longer launch must not be stranded on screen.
  if (!visible) closeTile(panel);
}

/**
 * Close `panel`'s tile and hand the workspace to something the operator can
 * still see, without touching rail membership.
 *
 * Shared by the two frames that take a tile off screen: `panel_close` (the
 * agent's close_panel, membership untouched) and the removal half of
 * `panel_visibility` (where membership is already gone by the time this runs).
 * Keeping one implementation is what makes "closed" mean the same thing on both
 * paths — the fallback search reads the CURRENT membership set, so it naturally
 * skips a panel that was just removed and keeps one that was merely closed.
 * @param {string} panel
 */
function closeTile(panel) {
  const c = ctx();
  withEchoSuppressed(() => hidePanel(panel));
  if (panel !== c.getActive()) return;
  const fallback = PANELS.find(
    p => p.id !== panel && c.memberSet.has(p.id) && c.isHealthy(p.id)
  );
  if (fallback) {
    c.activate(fallback.id);
  } else {
    c.resetActive();
    c.renderEmpty('No panels visible');
  }
}

/**
 * Apply a `panel_close` frame: take the tile off screen, leave the rail alone.
 *
 * The on-screen half of the panel vocabulary. Closing a panel that has no tile
 * open is a no-op in `hidePanel`, which is what makes the verb safe to call
 * without first reading a per-client tile report that may be stale.
 * @param {string} panel
 * @param {string|undefined} source
 */
function applyPanelClose(panel, source) {
  closeTile(panel);
  // No rail entry is created or destroyed, so unlike the visibility path there
  // is an entry left to glow — the operator can see which panel just went away.
  if (source === 'agent') {
    flashAgentGlow(panel);
    ctx().reportActivity({ type: AGENT_ACTIVITY_FRAME, ts: Date.now(),
      tool: 'close_panel', target: { kind: 'panel', panel } });
  }
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
}
