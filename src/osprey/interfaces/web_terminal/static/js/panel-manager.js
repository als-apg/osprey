// @ts-check
/* OSPREY Web Terminal — Panel Manager
 *
 * Manages the left icon rail for the right panel. Each rail entry corresponds
 * to an embedded service (Workspace, ARIEL logbook, etc.) loaded in an iframe.
 * Entries show health LEDs, iframes are lazy-loaded and cached so switching
 * between panels is instant.
 *
 * The rail is a curated LAUNCHER: the server-owned visible set is rail
 * MEMBERSHIP (agent show/hide_panel ≡ human "+"/"×"), and an entry exists iff
 * its panel is a member — always at full brightness, never dimmed. Which
 * member currently holds the workspace tile is per-client layout state,
 * reflected only by the `.active` accent; evicting a panel from the tile (or
 * closing its tile) changes no rail or server state.
 *
 * This module owns the panel state machine (health polling, SSE-driven
 * focus/visibility/registration, iframe lifecycle) and drives the
 * rail's DOM through panel-rail.js's imperative API — it never touches rail
 * markup itself.
 */

import { fetchJSON, createEventSource } from './api.js';
import { sendThemeToIframe, sendSessionToIframe, sendModeToIframe, buildEmbedSrc } from './panel-iframe-sync.js';
import { renderEmptyState as renderEmptyStateInto } from './panel-empty-state.js';
import { applyPreset, wirePanelHeaderControls } from './panel-presets.js';
import { setPanelVisibility, setPanelFocus, registerUrlPanel } from './panel-commands.js';
import {
  initDockIframeAdapter, focusPanel, hidePanel, concealPanel,
  setKnownServicePanels, setServerVisiblePanels,
} from './dock-iframe.js';
import {
  initPanelPlacement, openPanelBeside, dropPanelAt, applyAgentSwitch, applyArrange,
} from './panel-placement.js';
import { createPanelIframe } from './panel-iframe-factory.js';
import {
  PANELS, TERMINAL_RAIL_ID, TERMINAL_RAIL_LABEL, DEFAULT_PANEL_FALLBACK,
} from './panel-catalog.js';
import { initDockSync, withEchoSuppressed, setTileCloseHandler } from './dock-sync.js';
import { initRailDrag, railDragStart, railDragEnd } from './rail-drag.js';
import { startHealthPolling as startPolling } from './panel-health.js';
import { openTerminalPanel, closeTerminalPanel } from './dock-workspace.js';
import { initRailThemeCoupling } from './rail-position.js';
import { flashElement } from '/design-system/js/highlight.js';
import {
  createRail, addEntry, removeEntry, getEntry, setActive,
  setEntryEnabled, setEntryAttention,
} from './panel-rail.js';

// ---- Types ----

/** @typedef {import('./panel-catalog.js').Panel} Panel */

/**
 * @typedef {object} PanelState
 * @property {string | null} url
 * @property {boolean} healthy
 * @property {HTMLIFrameElement | null} iframe
 * @property {ReturnType<typeof setTimeout> | null} pollTimer
 * @property {boolean} polling
 * @property {boolean} configLoaded
 * @property {string | null} [pendingUrl]
 */

/**
 * SSE payloads broadcast on /api/files/events, discriminated on `type`.
 * `source: 'agent'` marks a frame as agent-originated (absent = human/browser
 * origin); absent optional keys are omitted by the server, never null.
 * @typedef {object} PanelFocusEvent
 * @property {'panel_focus'} type
 * @property {string} panel
 * @property {string} [url]
 * @property {'agent'} [source]
 *
 * @typedef {object} PanelVisibilityEvent
 * @property {'panel_visibility'} type
 * @property {string} panel
 * @property {boolean} visible
 * @property {'agent'} [source]
 *
 * @typedef {object} PanelRegisterEvent
 * @property {'panel_register'} type
 * @property {string} id
 * @property {string} [label]
 * @property {string} [url]
 * @property {string} [healthEndpoint]
 * @property {string} [path]
 * @property {'agent'} [source]
 *
 * @typedef {object} PanelArrangeEvent
 * @property {'panel_arrange'} type
 * @property {string[]} tiles      - the service tiles to have open, left to right
 * @property {string} [focus]      - requested focus target, one of `tiles`
 * @property {boolean} [prune_rail] - preset path: rail membership becomes exactly `tiles`
 * @property {'agent'} [source]
 *
 * @typedef {object} AgentActivityEvent
 * @property {'agent_activity'} type
 * @property {string} tool
 * @property {{ kind: 'panel' | 'channel' | 'run' | 'artifact', panel?: string, detail?: string }} target
 * @property {number} [ts]
 *
 * @typedef {PanelFocusEvent | PanelVisibilityEvent | PanelRegisterEvent | PanelArrangeEvent
 *   | AgentActivityEvent} PanelSSEEvent
 */

// ---- State ----

let containerEl = /** @type {HTMLElement | null} */ (null);
// Assigned once in initPanelManager and guarded there; other functions run
// only after that, so the refs are treated as non-null past init.
let railEl = /** @type {HTMLElement} */ (/** @type {unknown} */ (null));
let contentEl = /** @type {HTMLElement} */ (/** @type {unknown} */ (null));
/** @type {string | null} */
let activeTabId = null;

// Per-panel state: { url, healthy, iframe, pollTimer, configLoaded }
/** @type {Record<string, PanelState>} */
const panelState = {};

// Rail MEMBERSHIP — the server-owned visible set, seeded from /api/panels at
// init. An id in here has a rail entry; toggling one is paired with
// addEntry()/removeEntry() on the rail.
const visiblePanels = new Set();

// Simple-UX chat-only first boot: while true, ensureActivePanel leaves the
// workspace slot empty, so no dockview placeholder is ever created and the
// chat keeps the full width. Seeded in initPanelManager (simple mode + empty
// agent workspace, per /api/panels' workspace_has_artifacts); cleared one-way
// by ANY panel activation (agent show/switch, rail click, palette) or a flip
// to expert — once the workspace has appeared, the onboarding state is over
// for this page lifetime. A reload re-derives it from the server flag.
let workspaceSuppressed = false;

// Default panel to activate first (catalog fallback until a profile-pinned
// value arrives via panelConfig.default in initPanelManager).
let DEFAULT_PANEL = DEFAULT_PANEL_FALLBACK;

// Whether the server permits runtime URL-panel registration (web.allow_runtime_panels).
// Read from /api/panels at init; gates the "new panel from URL" row in the add menu.
let allowRuntimePanels = false;

// Config-defined panel presets ("Layouts") from /api/panels (web.presets), in
// config order. Empty unless a facility opts in; feeds the "+" menu's Layouts section.
/** @type {{name: string, panels: string[]}[]} */
let panelPresets = [];

// Activity-strip fallback for agent_activity frames the rail cannot anchor
// (kinds 'channel'/'run'/'artifact', or a 'panel' kind whose id has no rail
// entry). The no-op default keeps panel-manager fully functional standalone;
// the activity-strip module registers the real handler via
// setActivityStripHandler once it exists.
/** @type {(frame: AgentActivityEvent) => void} */
let onAgentActivity = () => {};

/**
 * SEAM: register the activity-strip handler for agent_activity frames that
 * have no rail anchor. Frames arrive verbatim as broadcast (see
 * AgentActivityEvent). Pass null to restore the no-op default.
 * @param {((frame: AgentActivityEvent) => void) | null} handler
 */
export function setActivityStripHandler(handler) { onAgentActivity = handler ?? (() => {}); }

// ---- Public API ----

/**
 * Initialize the tabbed panel manager inside the given container element.
 * @param {string} panelId
 */
export async function initPanelManager(panelId) {
  containerEl = document.getElementById(panelId);
  if (!containerEl) return;

  railEl = /** @type {HTMLElement} */ (document.getElementById('panel-rail'));
  contentEl = /** @type {HTMLElement} */ (containerEl.querySelector('#panel-content') || containerEl.querySelector('.panel-content'));
  if (!railEl || !contentEl) return;

  // Hand the iframe adapter its fallback mount host. When the dockview shell is
  // up, panel iframes live in the adapter's overlay layer instead (dockview
  // re-parents panel content on regroup, which reloads iframes — see the
  // dock-spike verdict and dock-iframe.js); without a shell they mount here.
  initDockIframeAdapter({ fallbackHost: contentEl });

  // Bridge dockview gestures back to the server-owned panel state: a human dock
  // tab focus / close POSTs the same setPanelFocus / setPanelVisibility the rail
  // and agent use. Wires lazily once the dockview shell is up; no-ops without it.
  initDockSync();

  // Hand panel-placement (the tile-geometry half of this module: open-beside,
  // agent switch, arrange rebuild) live access to the state it places panels
  // through. Every entry is a closure over this module's private state, so the
  // placement verbs see the same rail/health/membership the SSE handlers do.
  initPanelPlacement({
    isKnown: (id) => !!panelState[id],
    isHealthy: (id) => !!panelState[id]?.healthy,
    isMember: (id) => visiblePanels.has(id),
    members: () => [...visiblePanels],
    label: labelOf,
    addMember: ensureRailMembership,
    dropMember: (id) => { visiblePanels.delete(id); removeEntry(railEl, id); },
    activate: activateTab,
    reveal: showPanel,
    getActive: () => activeTabId,
    clearActive: clearActivePanel,
    renderEmpty: renderEmptyState,
    glow: flashAgentGlow,
    openTerminal: openTerminalPanel,
  });

  // Rail drag-and-drop: a rail entry dropped on a tile edge opens (or moves)
  // that panel as a new tile at the drop position, then reveals it through the
  // same activate/show tail every other open path uses. Wires lazily like
  // initDockSync; no-ops without a dock shell.
  initRailDrag({ onDropPanel: dropPanelAt });

  // Fetch panel config and filter PANELS before rendering
  let panelConfig = null;
  try {
    panelConfig = await fetchJSON('/api/panels');
    const enabledSet = new Set(panelConfig.enabled || []);

    // Filter built-in panels to only enabled ones
    const activePanels = PANELS.filter(p => enabledSet.has(p.id));

    // Honor a profile-pinned default panel when it resolves to a real tab.
    // Unknown id (typo, dropped panel) silently falls back so the user
    // doesn't end up on a blank tabset.
    if (panelConfig.default) {
      const knownIds = new Set(activePanels.map(p => p.id));
      for (const cp of (panelConfig.custom || [])) knownIds.add(cp.id);
      if (knownIds.has(panelConfig.default)) {
        DEFAULT_PANEL = panelConfig.default;
      } else {
        console.warn(
          `Panel config 'default': ${panelConfig.default} is not an enabled panel; ` +
          `falling back to ${DEFAULT_PANEL_FALLBACK}.`,
        );
      }
    }

    // Add custom panels
    for (const cp of (panelConfig.custom || [])) {
      if (!activePanels.some(p => p.id === cp.id)) {
        activePanels.push({
          id: cp.id,
          label: cp.label || cp.id.toUpperCase(),
          configEndpoint: null,
          healthEndpoint: cp.healthEndpoint,  // null = skip health polling
          statusBarId: null,
          path: cp.path || '/',             // subpath for iframe (e.g. "/panel/")
        });
      }
    }

    // Replace PANELS with filtered list
    PANELS.length = 0;
    PANELS.push(...activePanels);
  } catch (e) {
    console.warn('Could not load panel config, showing all panels:', e);
  }

  // Initialize state for each (now-filtered) panel
  for (const panel of PANELS) {
    panelState[panel.id] = {
      url: null,
      healthy: false,
      iframe: null,
      pollTimer: null,
      polling: false,
      configLoaded: false,
    };
  }

  // Seed visiblePanels from server config ('visible' field added by Task 1.1).
  // Fall back to all enabled panel ids for backward compat when field is absent.
  if (panelConfig?.visible) {
    for (const id of panelConfig.visible) visiblePanels.add(id);
  } else {
    for (const panel of PANELS) visiblePanels.add(panel.id);
  }

  // Simple-UX onboarding: boot chat-only while the agent workspace is empty.
  // html[data-ui-mode] is the resolved runtime mode (mode-boot.js) — it can
  // out-rank the server's ui_mode default, so it is the authority here. Only
  // set when /api/panels answered: a failed fetch keeps today's behavior.
  workspaceSuppressed =
    document.documentElement.getAttribute('data-ui-mode') === 'simple' &&
    panelConfig != null && !panelConfig.workspace_has_artifacts;

  // Whether the human "+" menu may register URL panels (server config gate).
  allowRuntimePanels = !!panelConfig?.allow_runtime_panels;

  // Config-defined layouts for the "+" menu's Layouts section (empty by default).
  panelPresets = panelConfig?.presets || [];

  // Seed the rail's theme coupling (which families imply which rail position,
  // and whether config pinned one) so a later family switch can move the rail.
  // A failed fetch leaves it inert, which is the pre-coupling behavior.
  initRailThemeCoupling(panelConfig || {});

  // A human closing a dock tile is a LOCAL vacate (occupancy is per-client
  // layout state; the panel keeps its rail membership) — reconcile the local
  // active state here, never POST.
  setTileCloseHandler(vacatePanel);

  // Hand the adapter a live reference to the visible set (it prunes restored
  // placeholders of server-closed panels), then finalize the registry — the
  // adapter may now prune any restored placeholder whose service no longer
  // exists (reconcile keeps all iframe:*).
  setServerVisiblePanels(visiblePanels);
  setKnownServicePanels(PANELS.map((p) => p.id));

  // Render the rail entries
  renderRail();

  // Wire the header "+" control (add menu + Layouts). wirePanelHeaderControls
  // owns the getElementById lookups and the initPanelAddMenu call; the menu is a
  // dumb view reading state through these closures and calling back into the same
  // visibility/register paths the agent uses.
  wirePanelHeaderControls({
    getHiddenPanels,
    allowUrlPanels: () => allowRuntimePanels,
    onShowPanel: showPanel,
    onRegisterUrl: registerUrlPanel,
    getPresets,
    onApplyPreset: applyMenuPreset,
  });

  // Keyboard close: Delete/Backspace on a focused entry hides that panel (the
  // "×" is mouse-only/decorative). Delegated — one listener, not one per entry.
  railEl.addEventListener('keydown', (e) => {
    if (e.key !== 'Delete' && e.key !== 'Backspace') return;
    if (!(e.target instanceof HTMLElement)) return;
    const id = e.target.closest('.panel-rail-button')?.getAttribute('data-panel-id');
    if (!id) return;
    e.preventDefault();
    // Terminal entry closes through the dock (no server-side visibility for it).
    if (id === TERMINAL_RAIL_ID) closeTerminalPanel();
    else setPanelVisibility(id, false);
  });

  // Fetch config and start health polling for all panels
  for (const panel of PANELS) {
    initPanel(panel);
  }

  // Handle custom panels that have URLs set directly (from /api/panels)
  if (panelConfig?.custom) {
    for (const cp of panelConfig.custom) {
      const ps = panelState[cp.id];
      if (ps && cp.url) {
        ps.url = cp.url;
        ps.configLoaded = true;
        if (!cp.healthEndpoint) {
          assumeHealthy(cp);
        } else {
          const panel = PANELS.find(p => p.id === cp.id);
          if (panel) startHealthPolling(panel);
        }
      }
    }
  }

  // Listen for SSE events via createEventSource (api.js) so the URL picks up
  // window.__OSPREY_PREFIX__ under multi-user deployments (empty prefix ⇒
  // unchanged behavior). createEventSource also drives the module-level
  // sseState in api.js, but nothing currently reads getConnectionState().sse
  // (only .ws is consumed, by app.js's status dot), so that side effect is
  // harmless. These event types are handled:
  //
  //   panel_focus      {type, panel, url?}      — explicit switch_panel MCP call
  //                                               or the echo of a human focus
  //                                               gesture; always honor. An
  //                                               agent-tagged frame surfaces
  //                                               the panel without evicting
  //                                               anything (applyAgentSwitch);
  //                                               a human echo takes the tile
  //                                               over as it always has.
  //   panel_visibility {type, panel, visible}   — show/hide a rail entry; if the
  //                                               active panel is hidden, switch to
  //                                               the next visible+healthy panel or
  //                                               empty state.
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
    onMessage: (raw) => {
      try {
        const data = /** @type {PanelSSEEvent} */ (raw);

        if (data.type === 'panel_focus' && data.panel) {
          // A switch — agent or human — honor unconditionally. It also ends the
          // simple-UX chat-only suppression, even when the activation still
          // refuses (unhealthy panel): the intent to surface the workspace is
          // clear, so the next health settle may fill the slot.
          workspaceSuppressed = false;
          if (data.url) navigatePanel(data.panel, data.url);
          // An AGENT switch is polite: focus the panel's own tile, or open one
          // beside the operator's — never take a tile away (applyAgentSwitch).
          // Every other frame is the echo of a human gesture (rail click, dock
          // tab focus) whose takeover semantics are the operator's own choice,
          // so it keeps the plain activation. The glow runs after the switch so
          // a just-added entry can flash.
          if (data.source === 'agent') {
            applyAgentSwitch(data.panel);
            flashAgentGlow(data.panel);
          } else {
            activateTab(data.panel);
          }

        } else if (data.type === 'panel_visibility' && data.panel) {
          const { panel, visible } = data;
          // Update the membership set and add/remove the matching rail entry.
          // The agent glow runs after the add so a just-added entry can flash.
          if (visible) {
            ensureRailMembership(panel);
          } else {
            visiblePanels.delete(panel);
            removeEntry(railEl, panel);
          }
          if (data.source === 'agent') flashAgentGlow(panel);

          // Simple-UX reveal: showing a panel while the workspace is chat-only
          // suppressed brings up the workspace ON that panel — this is the
          // "agent produced an artifact, show_panel('artifacts')" onboarding
          // path the panels-context SessionStart hook instructs. {auto: true}
          // keeps the health guard: an unhealthy panel leaves the slot for a
          // later settle (suppression is already cleared).
          if (visible && workspaceSuppressed) {
            workspaceSuppressed = false;
            if (!activeTabId) activateTab(panel, { auto: true });
          }

          if (!visible) {
            // EVERY hide drops the panel's dock tile (one panel per tile — a
            // closed panel's placeholder would be a ghost tile), active or
            // not. The echo guard covers dockview auto-activating a neighbor
            // on removal: that programmatic change is a server-applied echo
            // and must not POST focus back (the fallback below owns focus).
            withEchoSuppressed(() => hidePanel(panel));
          }

          // CC-1: if we just hid the currently active panel, switch away from it
          if (!visible && panel === activeTabId) {
            // Find the first panel that is visible, healthy, and not the one being hidden
            const fallback = PANELS.find(
              p => p.id !== panel && visiblePanels.has(p.id) && panelState[p.id]?.healthy
            );
            if (fallback) {
              activateTab(fallback.id);
            } else {
              // No usable panel remains — strand-proof: clear active state and show empty pane
              activeTabId = null;
              renderEmptyState('No panels visible');
            }
          }

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
          visiblePanels.add(data.id);
          addPanel(data);
          // CC-3: do NOT call activateTab — the new panel's URL may not be ready yet;
          // the user activates when they want it.
          if (data.source === 'agent') flashAgentGlow(data.id);

        } else if (data.type === 'agent_activity' && data.target) {
          // kind 'panel' with a live rail entry → persistent badge + glow via
          // setEntryAttention; everything the rail cannot anchor falls through
          // to the activity-strip seam (no-op until a handler registers).
          const t = data.target;
          if (t.kind !== 'panel' || !t.panel || !setEntryAttention(railEl, t.panel, true)) onAgentActivity(data);
        }

      } catch { /* ignore malformed events */ }
    },
  });
}

// ---- Rail Rendering ----

/**
 * Transient agent glow on a rail entry — flash only, never the persistent
 * badge (that is agent_activity's job via setEntryAttention). No-op for ids
 * without an entry.
 * @param {string} panelId
 */
function flashAgentGlow(panelId) { const entry = getEntry(railEl, panelId); if (entry) flashElement(entry); }

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
function railOptions() {
  return {
    onActivate: (/** @type {string} */ id) => {
      if (id === TERMINAL_RAIL_ID) { openTerminalPanel(); return; }
      // Clicking the entry whose tile is ALREADY surfaced retires that tile —
      // a toggle shortcut equivalent to the tile header's own "×". It stays a
      // LOCAL layout change: rail membership survives, so a second click
      // brings the tile back. Membership removal remains the separate "×"
      // corner.
      if (visiblePanels.has(id)) {
        if (activeTabId === id) { retireTile(id); return; }
        activateTab(id, { userInitiated: true });
      } else showPanel(id);
    },
    onClose: (/** @type {string} */ id) => {
      if (id === TERMINAL_RAIL_ID) { closeTerminalPanel(); return; }
      setPanelVisibility(id, false);
    },
    // Popout stays rail-only (a tile has no standalone-URL affordance).
    // No-ops before the panel's config fetch has resolved a URL; the rail
    // also hides the affordance while the entry is `.disabled`, so this
    // guard is the backstop, not the only gate.
    onPopout: (/** @type {string} */ id) => {
      if (id === TERMINAL_RAIL_ID) return;
      const url = getPanelStandaloneUrl(id);
      if (url) window.open(url, '_blank', 'noopener');
    },
    // "⊞" — open this panel as a NEW tile beside the active one, the
    // discoverable half of the open-beside verb (rail drag is the precise
    // half). The terminal entry routes to its own reopen path; CSS hides its
    // corner regardless.
    onOpenBeside: (/** @type {string} */ id) => openPanelBeside(id),
    // Rail drag source: the terminal entry never drags (its tile moves by its
    // own header bar); everything else defers to rail-drag's policy (simple
    // mode and fallback mode cancel there).
    onDragStart: (/** @type {string} */ id, /** @type {DataTransfer | null} */ dt) =>
      id === TERMINAL_RAIL_ID ? false : railDragStart(id, dt),
    onDragEnd: () => railDragEnd(),
  };
}

/** The catalog label for a panel id (falls back to the id itself).
 *  @param {string} id @returns {string} */
function labelOf(id) {
  return PANELS.find((p) => p.id === id)?.label ?? id;
}

/**
 * Destructive full render of the rail: the terminal entry first (the session
 * tile is the workspace's anchor), then every MEMBER service panel in PANELS
 * order — non-members have no entry at all. The terminal entry is enabled
 * after the render (it never health-polls, so it would stay disabled).
 */
function renderRail() {
  createRail(
    railEl,
    [
      { id: TERMINAL_RAIL_ID, label: TERMINAL_RAIL_LABEL },
      ...PANELS.filter((p) => visiblePanels.has(p.id)).map((p) => ({ id: p.id, label: p.label })),
    ],
    railOptions(),
  );
  setEntryEnabled(railEl, TERMINAL_RAIL_ID, true);
}

/**
 * Clear the locally-tracked active panel (rail accent, container stamp,
 * activeTabId). Used when the active panel's tile goes away without a
 * successor — a human tile close.
 */
function clearActivePanel() {
  activeTabId = null;
  setActive(railEl, null);
  if (containerEl) delete containerEl.dataset.activePanel;
}

/**
 * Reconcile local state after a human closed a service panel's dock tile
 * (registered with dock-sync at init; dockview itself removes the placeholder).
 * The panel keeps its rail membership — only the local occupancy state moves:
 * the overlay iframe is concealed, and the active accent clears when it
 * pointed at the closed tile.
 * @param {string} panelId
 */
function vacatePanel(panelId) {
  concealPanel(panelId);
  if (activeTabId === panelId) clearActivePanel();
}

/**
 * Retire a surfaced panel's dock tile from the rail — the toggle-off half of
 * an entry click. The twin of {@link vacatePanel}: that one reconciles state
 * AFTER dockview has already removed the placeholder (a click on the tile's
 * own close), whereas here nothing has removed it yet, so this must drop the
 * tile itself via hidePanel. Local only — `visiblePanels` is untouched, so the
 * entry keeps its rail membership and no POST fires. The echo guard covers
 * dockview auto-activating a neighbouring tile on the removal.
 * @param {string} panelId
 */
function retireTile(panelId) {
  withEchoSuppressed(() => hidePanel(panelId));
  clearActivePanel();
}

/**
 * Re-apply a rail entry's live state after it was (re)built cold by addEntry —
 * enabled from the panel's health, and the active accent when this panel is the
 * surfaced one (a "+"-menu reveal activates locally BEFORE the membership echo
 * rebuilds the entry).
 * @param {string} panelId
 */
function applyEntryState(panelId) {
  const ps = panelState[panelId];
  if (!ps) return;
  if (ps.healthy) setEntryEnabled(railEl, panelId, true);
  if (activeTabId === panelId) setActive(railEl, panelId);
}

/**
 * Give a panel its rail entry as a MEMBER: record the membership and append the
 * entry (membership IS the rail — there is no dimmed in-between state). The
 * entry is built cold, so its live health/active state is re-applied after the
 * add. Idempotent: addEntry no-ops for an id that already has an entry.
 * @param {string} panelId
 */
function ensureRailMembership(panelId) {
  visiblePanels.add(panelId);
  const spec = PANELS.find((p) => p.id === panelId);
  if (!spec) return;
  addEntry(railEl, { id: spec.id, label: spec.label }, railOptions());
  applyEntryState(panelId);
}

/**
 * Register a runtime panel and append its rail entry without wiping existing ones.
 *
 * spec shape (matches the panel_register SSE broadcast payload):
 *   { id, label, url, healthEndpoint, path }
 *
 * Guard: if panelState[id] already exists (re-register), refresh the url
 * in-place rather than duplicating the entry or state.
 * @param {PanelRegisterEvent} spec
 */
function addPanel(spec) {
  if (panelState[spec.id]) {
    // Re-registration: update url so subsequent navigation stays consistent
    if (spec.url) panelState[spec.id].url = spec.url;
    return;
  }

  const normalized = {
    id: spec.id,
    label: spec.label || spec.id.toUpperCase(),
    configEndpoint: null,
    healthEndpoint: spec.healthEndpoint || null,
    statusBarId: null,
    path: spec.path || '/',
  };
  PANELS.push(normalized);
  // Keep the adapter's known-service set current (never orphan a runtime panel).
  setKnownServicePanels(PANELS.map((p) => p.id));

  panelState[spec.id] = {
    url: null,
    healthy: false,
    iframe: null,
    pollTimer: null,
    polling: false,
    configLoaded: false,
  };

  // Append exactly one entry. addEntry is non-destructive — never a full
  // re-render — so every live entry keeps its active/disabled/LED state, and it
  // is idempotent by id, which also guards the re-register path.
  addEntry(railEl, { id: normalized.id, label: normalized.label }, railOptions());

  // Seed url and health, mirroring the custom-panel block in initPanelManager
  if (spec.url) {
    const ps = panelState[spec.id];
    ps.url = spec.url;
    ps.configLoaded = true;
    if (!spec.healthEndpoint) {
      assumeHealthy(normalized);
    } else {
      startHealthPolling(normalized);
    }
  }
}

// ---- Panel Initialization ----

/** @param {Panel} panel */
async function initPanel(panel) {
  const state = panelState[panel.id];
  // Custom/runtime panels carry no config endpoint; their url arrives via
  // /api/panels. Skip the fetch and leave the panel disabled until then.
  if (!panel.configEndpoint) { state.configLoaded = true; return; }

  try {
    const config = await fetchJSON(panel.configEndpoint);
    // Artifact server returns { url }, ARIEL returns { url, available }
    if (config.url && (config.available === undefined || config.available)) {
      state.url = config.url;
    }
  } catch {
    // Config endpoint not available — panel stays disabled
  } finally {
    state.configLoaded = true;
  }

  if (state.url) {
    // External panels (healthEndpoint === null) skip health polling —
    // mark healthy immediately so the tab is enabled.
    if (panel.healthEndpoint == null) {  // null or undefined → skip polling
      assumeHealthy(panel);
    } else {
      startHealthPolling(panel);
    }
  }
  // Re-evaluate on every settle, including the no-url case: this panel may be
  // the default that another panel's health poll was waiting on.
  ensureActivePanel();
}

/**
 * Give the empty slot to the best panel available, if any.
 *
 * Health-driven, so {auto: true} keeps it from ever surfacing a hidden panel.
 * Safe to call on every settle: it no-ops once something is active.
 *
 * This is deliberately re-entrant rather than a one-shot at each health
 * transition. A panel's FIRST healthy transition can land while the default is
 * still loading its config — decline then and that panel never gets another
 * transition to try again, stranding the pane blank.
 */
function ensureActivePanel() {
  if (activeTabId) return;
  // Simple-UX chat-only boot: nothing auto-claims the empty slot while the
  // workspace is suppressed — an agent reveal, a rail click, or an expert
  // flip ends the suppression and re-runs this policy.
  if (workspaceSuppressed) return;
  const ds = panelState[DEFAULT_PANEL];
  if (!ds?.configLoaded) return;  // default may still claim the slot — wait
  // Hidden disqualifies the default exactly as unhealthy does; activateTab
  // would refuse it anyway, and the slot must not sit empty behind it.
  const target = ds.healthy && visiblePanels.has(DEFAULT_PANEL)
    ? DEFAULT_PANEL
    : PANELS.find(p => visiblePanels.has(p.id) && panelState[p.id]?.healthy)?.id;
  if (target) activateTab(target, { auto: true });
}

// ---- Health Polling ----

/**
 * Poll-settle hook handed to panel-health.js's timing machinery: reflect the
 * new health on the status bar, and on the FIRST healthy settle enable the
 * entry and let the shared policy decide whether the newly-healthy panel
 * should take an empty slot. The rail itself shows no per-poll readout — the
 * SYSTEM panel's `web_panels` category is where liveness is reported.
 * @param {Panel} panel
 * @param {boolean} wasHealthy
 */
function onHealthSettled(panel, wasHealthy) {
  updateStatusBar(panel);
  if (panelState[panel.id].healthy && !wasHealthy) {
    setEntryEnabled(railEl, panel.id, true);
    ensureActivePanel();
  }
}

/** @param {Panel} panel  Start panel-health's polling loop with this module's hook. */
function startHealthPolling(panel) {
  startPolling(panel, panelState[panel.id], onHealthSettled);
}

// ---- Entry State ----

/**
 * A panel with no health endpoint is assumed permanently healthy: mark it so
 * and enable its rail entry. Consolidates the built-in, custom-config, and
 * runtime-addPanel paths so none can leave an entry inert forever.
 * @param {Panel} panel
 */
function assumeHealthy(panel) {
  panelState[panel.id].healthy = true;
  setEntryEnabled(railEl, panel.id, true);
}

/** @param {Panel} panel */
function updateStatusBar(panel) {
  if (!panel.statusBarId) return;

  const statusItem = document.getElementById(panel.statusBarId);
  if (!statusItem) return;

  const state = panelState[panel.id];
  if (state.url) {
    statusItem.style.display = '';
    const dot = statusItem.querySelector('.status-dot');
    if (dot) {
      dot.className = 'status-dot' + (state.healthy ? ' live' : ' error');
    }
  }
}

/**
 * Broadcast the current UI mode to every panel iframe — the hub-role fan-out
 * the header toggle fires after it swaps <html data-ui-mode>. Mirrors
 * theme-manager's own theme _broadcast(); the mode axis has no such manager, so
 * panel-manager drives it over its own iframes.
 */
export function broadcastMode() {
  for (const panel of PANELS) {
    sendModeToIframe(panelState[panel.id]?.iframe ?? null);
  }
}

/**
 * React to the header Expert/Simple toggle — called by app.js's initModeToggle
 * AFTER the html[data-ui-mode] swap and the dock's applyDockMode. The expert
 * surface always shows the full workspace, so flipping to it ends the
 * simple-UX chat-only suppression and lets the default panel claim the still-
 * empty slot. Flipping to simple mid-session changes nothing here: a live
 * workspace stays (suppression is a first-boot state, never re-armed).
 * @param {'expert'|'simple'} mode
 */
export function handleUiModeFlip(mode) {
  if (mode !== 'expert') return;
  workspaceSuppressed = false;
  ensureActivePanel();
}

// ---- Tab Switching ----

/**
 * Focus an already-visible panel. Cross-module callers (the command palette's
 * "Focus panel" pick) MUST pass `{ userInitiated: true }` so the switch is
 * reported to the server via setPanelFocus — omitting it focuses locally only.
 * @param {string} panelId
 * @param {{ userInitiated?: boolean, auto?: boolean }} [options]
 */
export function activateTab(panelId, { userInitiated = false, auto = false } = {}) {
  const state = panelState[panelId];
  if (!state || !state.healthy) return;
  // A panel becoming healthy is not a request to show it. The server owns the
  // visible set, so health-driven activation must never surface a hidden panel
  // — otherwise a panel closed with "×" reappears on its own.
  if (auto && !visiblePanels.has(panelId)) return;

  // Past the guards the panel actually surfaces (rail click, agent focus,
  // palette, dock — any source): its agent-attention badge is served, clear it.
  // The guarded returns above deliberately keep the badge on panels that
  // refused to surface.
  setEntryAttention(railEl, panelId, false);

  // Any surfaced panel means the workspace is open — the simple-UX chat-only
  // suppression (if still armed) is over for this page lifetime.
  workspaceSuppressed = false;

  activeTabId = panelId;

  // Reflect the active entry on the rail
  setActive(railEl, panelId);

  // Stamp the active panel id on the content container so CSS can shape the
  // workspace region per-panel — e.g. a panel that paints its own full-bleed
  // canvas opts out of the hub's card chrome (see files.css [data-active-panel]).
  if (containerEl) containerEl.dataset.activePanel = panelId;

  // Clear any stale empty-state placeholder before revealing a panel. isConnected
  // guards a cached ref that was detached by renderEmptyState's innerHTML wipe
  // (fallback mode, where iframes live in #panel-content) — rebuild rather than
  // re-show a node no longer in the DOM. In overlay mode iframes live outside
  // #panel-content, so the wipe never detaches them and the cached ref is reused.
  contentEl.querySelector('.artifacts-empty-state')?.remove();

  // Create the iframe (first activation) and bring it forward, suppressing the
  // others. Both run inside the dock-sync echo guard: createIframe's adoptIframe
  // adds a dockview placeholder that auto-activates, and focusPanel drives a
  // programmatic active-tab change — each is an applied echo (server- or rail-
  // driven), never a fresh human dock gesture, so neither must POST focus back.
  // The adapter maps focus onto dockview's active-tab geometry in overlay mode,
  // or a plain display toggle in fallback mode.
  withEchoSuppressed(() => {
    if (!state.iframe || !state.iframe.isConnected) {
      createIframe(panelId);
    }
    focusPanel(panelId);
  });

  // Re-send current theme, mode and session ID to the newly visible iframe
  // (handles edge cases where a postMessage was missed while hidden/loading)
  sendThemeToIframe(state.iframe);
  sendModeToIframe(state.iframe);
  sendSessionToIframe(state.iframe);

  // Report user-initiated tab switches to the server (avoids SSE feedback loop)
  if (userInitiated) setPanelFocus(panelId);
}

// ---- Panel Visibility Actions (human "+" / "×") ----
//
// These back the human add/remove controls. The command POSTs live in
// panel-commands.js and the server's SSE echo drives the DOM, so a human
// action and an agent MCP call are indistinguishable downstream. The per-entry
// "×" calls setPanelVisibility(id, false) directly (the rail's onClose closure,
// see railOptions); the "+" menu's reveal path needs a local focus too, so it
// goes through showPanel.

/**
 * Reveal a hidden panel and focus it (a "Show panel" menu pick). The visibility
 * POST un-hides the tab for every client via SSE; activateTab focuses it here
 * when it's healthy (and no-ops otherwise, leaving the tab visible but unfocused).
 * @param {string} panelId
 */
export function showPanel(panelId) {
  setPanelVisibility(panelId, true);
  activateTab(panelId, { userInitiated: true });
}

/**
 * Apply a config-defined preset ("Layout") by name — the "+" menu's and the
 * command palette's Layouts action. One arrange request; the panel_arrange echo
 * opens exactly the preset's tiles and prunes the rail to its members on every
 * client, so nothing is applied locally ahead of it.
 * @param {string} name
 */
export function applyMenuPreset(name) {
  applyPreset(name);
}

// ---- Panel Navigation ----

/**
 * @param {string} panelId
 * @param {string} url
 */
function navigatePanel(panelId, url) {
  const state = panelState[panelId];
  if (!state) return;

  // Store the target URL so that createIframe() picks it up if the iframe
  // hasn't been lazy-loaded yet (e.g. first panel_focus SSE before the user
  // has ever clicked the tab).
  state.pendingUrl = url;

  if (!state.iframe) return;

  // buildEmbedSrc preserves the already-server-prefixed root-relative url
  // verbatim (never strip/re-add window.__OSPREY_PREFIX__ — see its docstring).
  state.iframe.src = buildEmbedSrc(url);
  state.pendingUrl = null;
}

// ---- Iframe Management ----

/** Build + adopt the panel's iframe via panel-iframe-factory.js (which owns
 *  everything about the element); this wrapper only binds the private state.
 *  @param {string} panelId */
function createIframe(panelId) {
  createPanelIframe(PANELS.find(p => p.id === panelId), panelId, panelState[panelId], contentEl);
}

// ---- Command Palette Accessors ----
//
// Thin read-only getters over this module's private panel state (PANELS,
// visiblePanels, activeTabId, panelPresets), letting the command-palette module
// enumerate panels without owning any of that state. They derive from the live
// state on every call — no new module-level variable — and pair with the
// re-exported showPanel / activateTab / applyMenuPreset actions so the palette
// drives the same visibility/focus/layout paths the "+" menu uses.

/**
 * Known-but-hidden panels, in PANELS order. Shared with the "+" add menu (the
 * wirePanelHeaderControls getHiddenPanels closure calls this, so both surfaces
 * enumerate identically).
 * @returns {Array<{id: string, label: string}>}
 */
export function getHiddenPanels() {
  return PANELS.filter(p => !visiblePanels.has(p.id)).map(p => ({ id: p.id, label: p.label }));
}

/**
 * Visible panels excluding the active one, in PANELS order. "Focus" on the
 * already-active panel is a no-op, so activeTabId is filtered out.
 * @returns {Array<{id: string, label: string}>}
 */
export function getVisiblePanels() {
  return PANELS.filter(p => visiblePanels.has(p.id) && p.id !== activeTabId).map(p => ({ id: p.id, label: p.label }));
}

/**
 * Config-defined layout presets ("Layouts"), in config order. Shared with the
 * "+" menu's Layouts section (the wirePanelHeaderControls getPresets closure
 * calls this). Empty unless a facility opts in.
 * @returns {Array<{name: string, panels: string[]}>}
 */
export function getPresets() {
  return panelPresets;
}

/**
 * Standalone (non-embedded) URL for a service panel — the target of the rail
 * entry's popout corner (railOptions' onPopout). state.url is the
 * already-proxied root-relative base; the optional catalog path suffixes custom
 * panels' UI root. Null until the panel's config fetch has resolved a URL.
 * @param {string} panelId
 * @returns {string | null}
 */
export function getPanelStandaloneUrl(panelId) {
  const state = panelState[panelId];
  if (!state?.url) return null;
  const panel = PANELS.find((p) => p.id === panelId);
  const path = panel?.path && panel.path !== '/' ? panel.path : '';
  return state.url + path;
}

/**
 * Currently surfaced panel id — the `data-active-panel` stamp activateTab
 * writes on the container — or null before init / with no active panel.
 * Consumed by the agent-activity suppression table.
 * @returns {string | null}
 */
export function getActivePanel() { return containerEl?.dataset.activePanel ?? null; }

// ---- Empty State ----

/**
 * Thin binding of the extracted placeholder card (panel-empty-state.js) onto
 * this module's private container/content refs.
 * @param {string} message
 */
function renderEmptyState(message) {
  renderEmptyStateInto(containerEl, contentEl, message);
}
