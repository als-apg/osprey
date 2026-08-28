// @ts-check
/* OSPREY Web Terminal — Panel Manager
 *
 * Manages the left icon rail for the right panel. Each rail entry corresponds
 * to an embedded service (Workspace, ARIEL logbook, etc.) loaded in an iframe.
 * Entries show health LEDs, iframes are lazy-loaded and cached so switching
 * between panels is instant.
 *
 * The rail is a curated LAUNCHER: the server-owned visible set is rail
 * MEMBERSHIP (agent add_panel_to_rail/remove_panel_from_rail ≡ human "+"/"×"),
 * and an entry exists iff its panel is a member — always at full brightness,
 * never dimmed. Which member currently holds the workspace tile is per-client
 * layout state, reflected only by the `.active` accent; evicting a panel from
 * the tile (or closing its tile, agent close_panel) changes no rail or server
 * state.
 *
 * This module owns the panel state machine (tile occupancy, iframe lifecycle,
 * the active-panel policy) and holds the private state the whole panel stack
 * reads. It drives the rail's DOM through panel-rail.js's imperative API — it
 * never touches rail markup itself.
 *
 * Two halves of it are extracted. HOW a panel comes to exist and stay alive —
 * cold state, the rail render and its membership entries, config fetch, health
 * polling — is panel-lifecycle.js. The server→client stream — the
 * /api/files/events subscription, its frame dispatcher, and the
 * membership/close apply paths that dispatcher shares with the reconnect
 * resync — is panel-sse.js. What a GESTURE on a rail entry or a tile header
 * does is panel-menu-policy.js (entry closures, both context menus), and where
 * a panel LANDS is panel-placement.js. All of them read this module's private
 * state through deps registered at init, so none imports it back: the import
 * direction is panel-manager → panel-sse → panel-lifecycle.
 */

import { fetchJSON } from './api.js';
import { sendThemeToIframe, sendSessionToIframe, sendModeToIframe, buildEmbedSrc } from './panel-iframe-sync.js';
import { renderEmptyState as renderEmptyStateInto } from './panel-empty-state.js';
import { hiddenPanels, visiblePanelsExcept, standaloneUrl } from './panel-queries.js';
import { applyPreset, wirePanelHeaderControls } from './panel-presets.js';
import { setPanelVisibility, setPanelFocus, registerUrlPanel } from './panel-commands.js';
import { applyConfigTabGate } from './config-tab.js';
import { applyScaffoldWriteGate } from './scaffold/write-gate.js';
import {
  initDockIframeAdapter, focusPanel, hidePanel, concealPanel,
  setKnownServicePanels, setServerVisiblePanels,
} from './dock-iframe.js';
import { initPanelPlacement, dropPanelAt } from './panel-placement.js';
import { createPanelIframe } from './panel-iframe-factory.js';
import {
  PANELS, TERMINAL_RAIL_ID, DEFAULT_PANEL_FALLBACK,
} from './panel-catalog.js';
import { initDockSync, withEchoSuppressed, setTileCloseHandler, setTileFocusHandler } from './dock-sync.js';
import { setTileContextMenuHandler } from './dock-tab.js';
import { initRailDrag } from './rail-drag.js';
import { openTerminalPanel, closeTerminalPanel } from './dock-workspace.js';
import { initRailThemeCoupling } from './rail-position.js';
import { initMenuPolicy, openTileContextMenu } from './panel-menu-policy.js';
import { removeEntry, setActive } from './panel-rail.js';
import {
  initPanelLifecycle, freshPanelState, renderRail, ensureRailMembership,
  initPanel, assumeHealthy, startHealthPolling,
} from './panel-lifecycle.js';
import { initAgentAttention, flashAgentTile, clearBadge } from './panel-agent-attention.js';
import { subscribePanelEvents } from './panel-sse.js';

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
 * @typedef {object} PanelCloseEvent
 * @property {'panel_close'} type
 * @property {string} panel
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
 * @property {typeof import('./activity-format.js').AGENT_ACTIVITY_FRAME} type
 * @property {string} tool
 * @property {{ kind: 'panel' | 'channel' | 'run' | 'artifact' | 'config' | 'ui',
 *   panel?: string, detail?: string }} target
 * @property {number} [ts]
 *
 * @typedef {PanelFocusEvent | PanelVisibilityEvent | PanelCloseEvent | PanelRegisterEvent
 *   | PanelArrangeEvent | AgentActivityEvent} PanelSSEEvent
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
// by ANY panel activation (agent add_panel_to_rail/open_panel, rail click,
// palette) or a flip to expert — once the workspace has appeared, the
// onboarding state is over for this page lifetime. A reload re-derives it
// from the server flag.
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

// ---- Injected State Accessors ----
//
// The private-state verbs the extracted halves (lifecycle, placement, menu
// policy, SSE) are handed at init. Each is defined ONCE here and passed by
// reference into every deps bag that wants it, so the same question asked from
// two modules cannot drift apart. The readers are closures over reassignable
// state by construction — they read railEl/activeTabId at call time, never a
// value captured when the deps were registered.

/** The rail nav element, read live (assigned once in initPanelManager).
 *  @returns {HTMLElement} */
function getRailEl() { return railEl; }

/** The locally surfaced panel id, or null when the workspace slot is empty.
 *  @returns {string | null} */
function getActiveTabId() { return activeTabId; }

/** Does this page know the panel at all (built-in, custom, or runtime-registered)?
 *  @param {string} id @returns {boolean} */
function isPanelKnown(id) { return !!panelState[id]; }

/** May the panel fill a tile — its health poll has settled healthy?
 *  @param {string} id @returns {boolean} */
function isPanelHealthy(id) { return !!panelState[id]?.healthy; }

/** Does the panel hold rail membership (the server-owned visible set)?
 *  @param {string} id @returns {boolean} */
function isRailMember(id) { return visiblePanels.has(id); }

/**
 * Take a panel's rail MEMBERSHIP away: drop it from the server-owned visible
 * set and remove its rail entry (membership IS the rail, so the two always move
 * together). The one mutator in this group, and the drop half of
 * panel-lifecycle's {@link ensureRailMembership} — placement's agent-arrange
 * prune and a panel_visibility removal share this body, so they cannot drift.
 * @param {string} panelId
 */
function dropRailMember(panelId) {
  visiblePanels.delete(panelId);
  removeEntry(railEl, panelId);
}

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

  // Hand panel-lifecycle (the birth-and-liveness half of this module: cold
  // state, the rail render and its membership entries, config fetch, health
  // polling) the private state it works through. Containers go by live
  // reference; the reassigned scalars (railEl, activeTabId) and the shared
  // empty-slot policy go as closures, so it always sees the current value.
  // First of the registrations, because the others hand out its verbs:
  // initPanelPlacement below passes ensureRailMembership as `addMember`, which
  // would throw if placement ran a membership add before this registration.
  initPanelLifecycle({
    panelState,
    memberSet: visiblePanels,
    getRailEl,
    getActive: getActiveTabId,
    ensureActive: ensureActivePanel,
  });

  initAgentAttention(railEl);

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
    isKnown: isPanelKnown,
    isHealthy: isPanelHealthy,
    isMember: isRailMember,
    members: () => [...visiblePanels],
    label: labelOf,
    addMember: ensureRailMembership,
    dropMember: dropRailMember,
    activate: activateTab,
    reveal: showPanel,
    getActive: getActiveTabId,
    clearActive: clearActivePanel,
    renderEmpty: renderEmptyState,
    // An arranged tile is attributed on both surfaces (rail entry flash + tile
    // body glow, one gesture: flashAgentTile). Only the arrange path passes
    // through here, which is the one placement verb an agent drives — the
    // rail ⊞ and drag-and-drop are human gestures and never glow.
    glow: flashAgentTile,
    openTerminal: openTerminalPanel,
  });

  // Hand the menu policy (the rail's interaction closures and both context
  // menus) the same live view of this module's private state. Rendered rows and
  // gates are decided per gesture, so these must be closures, not values.
  initMenuPolicy({
    getRailEl,
    isMember: isRailMember,
    getActiveTabId,
    activateTab, showPanel, retireTile, labelOf, getPanelStandaloneUrl, popoutPanel,
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
    panelState[panel.id] = freshPanelState();
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

  // Withdraw the settings drawer's Config tab where the deployment gated its
  // server surface off (web.config_panel.enabled). The tab is not a dock panel
  // — it is static drawer markup — but the flag rides the payload this module
  // already reads, and applying it here keeps the page to ONE /api/panels
  // round trip. config-tab.js owns the rule; a failed fetch (null) leaves the
  // tab alone, matching every other server-config read above.
  applyConfigTabGate(panelConfig);

  // Record whether this deployment's Scaffold gallery may write
  // (web.scaffold_gallery.write_enabled). The gallery renders its controls
  // lazily, when a drawer tab is first activated, so it reads the posture back
  // at render time; applying it here rides the same single /api/panels round
  // trip as the Config gate above. scaffold/write-gate.js owns the rule; a
  // failed fetch (null) leaves the posture alone, matching every other
  // server-config read in this function.
  applyScaffoldWriteGate(panelConfig);

  // A human closing a dock tile is a LOCAL vacate (occupancy is per-client
  // layout state; the panel keeps its rail membership) — reconcile the local
  // active state here, never POST.
  setTileCloseHandler(vacatePanel);

  // A human focusing a dock tab applies locally through activateTab (rail
  // accent, active-tab state, iframe reveal). dock-sync owns the mirror POST,
  // and the server does not echo human focus back, so this registration is the
  // only thing that keeps the gesturing client's own rail in step.
  setTileFocusHandler(activateTab);

  // A tile header is the panel's second right-click surface, offering the same
  // verbs as its rail entry. dock-tab cannot import this module (cycle via
  // dock-workspace), so the policy arrives by registration.
  setTileContextMenuHandler(openTileContextMenu);

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

  // Open the panel event stream (panel-sse.js: the /api/files/events
  // subscription, its frame dispatcher, and the membership/close apply paths
  // that dispatcher shares with the reconnect resync). LAST statement of init:
  // the first frame can arrive the moment the stream opens, so every other
  // registration and the rail render must already have happened. Reassigned
  // scalars go as getters/setters and the strip seam as a closure, so the
  // dispatcher always sees the value this module holds now.
  subscribePanelEvents({
    isKnown: isPanelKnown,
    isHealthy: isPanelHealthy,
    memberSet: visiblePanels,
    dropMember: dropRailMember,
    activate: activateTab,
    navigate: navigatePanel,
    getActive: getActiveTabId,
    // Forget the surfaced panel WITHOUT clearing the rail accent: a tile closed
    // with no fallback leaves the rail pointing at the panel that is one click
    // away again (clearActivePanel is the human-close path, not this one).
    resetActive: () => { activeTabId = null; },
    isWorkspaceSuppressed: () => workspaceSuppressed,
    endWorkspaceSuppression: () => { workspaceSuppressed = false; },
    renderEmpty: renderEmptyState,
    // Closure, not the value: onAgentActivity is reassigned whenever the
    // activity strip registers (setActivityStripHandler), and the dispatcher
    // must reach the handler current at frame time.
    reportActivity: (frame) => onAgentActivity(frame),
  });
}

// ---- Panel Queries & Active-Panel State ----

/**
 * Open a panel in a new browser tab at its standalone (non-embedded) URL.
 * The target of the context menu's "Open in a new window" row and the
 * palette's "Open <label> in a new window" action — both go through here so
 * the two surfaces cannot drift. No-op until the panel's config fetch has
 * resolved a URL (callers gate on that too).
 * @param {string} id
 */
export function popoutPanel(id) {
  const url = getPanelStandaloneUrl(id);
  if (url) window.open(url, '_blank', 'noopener');
}

/**
 * Rail-member panels that can pop out (standalone URL resolved), in catalog
 * order. Unlike getVisiblePanels this INCLUDES the active panel — popping out
 * the panel you are looking at is the verb's most common use.
 * @returns {Array<{id: string, label: string}>}
 */
export function getPopoutPanels() {
  return PANELS
    .filter((p) => visiblePanels.has(p.id) && getPanelStandaloneUrl(p.id) !== null)
    .map((p) => ({ id: p.id, label: p.label }));
}

/** The catalog label for a panel id (falls back to the id itself).
 *  Exported for the activity strip, which words panel actions with labels.
 *  @param {string} id @returns {string} */
export function labelOf(id) {
  return PANELS.find((p) => p.id === id)?.label ?? id;
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

// ---- UI Mode ----

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
  // palette, dock — any source): its agent-attention badge is served, so clear
  // it AND acknowledge it, or the next reload would restore it from history.
  // The guarded returns above deliberately keep the badge on panels that
  // refused to surface.
  clearBadge(panelId);

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

/** Human-local twin of the panel_focus SSE path: navigate a panel to `url`
 *  and plain-activate it. No agent glow, no server broadcast.
 *  @param {string} panelId
 *  @param {string} url */
export function navigateAndActivatePanel(panelId, url) {
  if (url) navigatePanel(panelId, url);
  activateTab(panelId);
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
export function getHiddenPanels() { return hiddenPanels(PANELS, visiblePanels); }

/**
 * Visible panels excluding the active one, in PANELS order. "Focus" on the
 * already-active panel is a no-op, so activeTabId is filtered out.
 * @returns {Array<{id: string, label: string}>}
 */
export function getVisiblePanels() { return visiblePanelsExcept(PANELS, visiblePanels, activeTabId); }

/**
 * Config-defined layout presets ("Layouts"), in config order. Shared with the
 * "+" menu's Layouts section (the wirePanelHeaderControls getPresets closure
 * calls this). Empty unless a facility opts in.
 * @returns {Array<{name: string, panels: string[]}>}
 */
export function getPresets() { return panelPresets; }

/**
 * Standalone (non-embedded) URL for a service panel — the target of the
 * context menu's and palette's "Open in a new window". state.url is the
 * already-proxied root-relative base; the optional catalog path suffixes custom
 * panels' UI root. Null until the panel's config fetch has resolved a URL.
 * @param {string} panelId
 * @returns {string | null}
 */
export function getPanelStandaloneUrl(panelId) { return standaloneUrl(PANELS, panelState[panelId], panelId); }

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
