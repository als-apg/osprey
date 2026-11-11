// @ts-check
/* OSPREY Web Terminal — Panel Manager
 *
 * Manages the sidebar's panel rows and the workspace tiles they open. Each
 * row corresponds to an embedded service (Artifacts, ARIEL logbook, etc.)
 * loaded in an iframe; iframes are lazy-loaded and cached so switching
 * between panels is instant.
 *
 * The rail is a curated LAUNCHER: the server-owned visible set is rail
 * MEMBERSHIP (agent add_panel_to_rail/remove_panel_from_rail ≡ the human "+"
 * menu and "Remove from rail"), and a row exists iff its panel is a member —
 * always at full brightness, never dimmed. Which member currently holds the
 * workspace tile is per-client layout state, reflected only by the `.active`
 * accent; evicting a panel from the tile (or closing its tile, agent
 * close_panel) changes no rail or server state.
 *
 * This module owns the panel state machine (tile occupancy, iframe lifecycle,
 * the active-panel policy) and holds the private state the whole panel stack
 * reads. It drives the sidebar rows through panel-rail.js's imperative API —
 * it never touches their markup itself.
 *
 * Two halves of it are extracted. HOW a panel comes to exist and stay alive —
 * cold state, the sidebar rows and their membership, config fetch, health
 * polling — is panel-lifecycle.js. The server→client stream — the
 * /api/files/events subscription, its frame dispatcher, and the
 * membership/close apply paths that dispatcher shares with the reconnect
 * resync — is panel-sse.js. What a GESTURE on a sidebar row or a tile header
 * does is panel-menu-policy.js (row closures, both context menus), and where
 * a panel LANDS is panel-placement.js. A panel's time in its own window is
 * pane-window.js. The start-up tail — the payload's other readers and the ready
 * mark — is panel-startup.js. All of them read this module's private state through deps
 * registered at init, so none imports it back: the import direction is
 * panel-manager → panel-sse → panel-lifecycle.
 */

import { fetchJSON } from './api.js';
import {
  sendThemeToIframe,
  sendSessionToIframe,
  buildEmbedSrc,
  isUrlWithinMount,
} from './panel-iframe-sync.js';
import { navigateInPlace, watchNavigateReady } from './panel-navigate.js';
import { renderEmptyState as renderEmptyStateInto } from './panel-empty-state.js';
import { hiddenPanels, visiblePanelsExcept, standaloneUrl } from './panel-queries.js';
import { applyPreset, wirePanelHeaderControls } from './panel-presets.js';
import { setPanelVisibility, setPanelFocus, registerUrlPanel } from './panel-commands.js';
import { finishStartup } from './panel-startup.js';
import {
  initDockIframeAdapter,
  focusPanel,
  hidePanel,
  concealPanel,
  managedIframe,
  panelReturned,
  adoptIframe,
  setKnownServicePanels,
  setServerVisiblePanels,
  setPlaceholdersAppliedHandler,
  PLACEHOLDER_PREFIX,
} from './dock-iframe.js';
import { initPanelPlacement, landDroppedPanel } from './panel-placement.js';
import { createPanelIframe, framePanel } from './panel-iframe-factory.js';
import { PANELS, TERMINAL_RAIL_ID, DEFAULT_PANEL_FALLBACK, titleCase } from './panel-catalog.js';
import {
  initDockSync,
  withEchoSuppressed,
  setTileCloseHandler,
  setTileFocusHandler,
} from './dock-sync.js';
import { setTileContextMenuHandler } from './dock-tab.js';
import { initRailDrag } from './rail-drag.js';
import {
  openTerminalPanel,
  getDockApi,
  bootLayoutSettled,
  setLayoutRestoredHook,
} from './dock-workspace.js';
import { getLayoutHolder } from './dock-apply.js';
import { serviceTileTarget } from './dock-open.js';
import { deriveSurfaced, followDock } from './dock-view.js';
import {
  initMenuPolicy,
  openTileContextMenu,
  removeFromRail,
  closeTerminalTile,
} from './panel-menu-policy.js';
import { removeEntry, setActive } from './panel-rail.js';
import {
  initPanelLifecycle,
  freshPanelState,
  markDetachedRow,
  renderRail,
  ensureRailMembership,
  initPanel,
  getPanelPage,
  assumeHealthy,
  publishPanelUrl,
  startHealthPolling,
  retryPanelStart,
} from './panel-lifecycle.js';
import { initAgentAttention, flashAgentTile, clearBadge } from './panel-agent-attention.js';
import { subscribePanelEvents } from './panel-sse.js';
import { detachBlocker, focusWindow, initPaneWindow, onDetachedChange } from './pane-window.js';
import { noteDetachedNavigation } from './pane-window.js';
import { isDetached } from './dock-apply.js';
import { showUndoNotice } from './undo-notice.js';
import { subscribe as subscribeTheme } from '/design-system/js/theme-manager.js';
import { relayTranscriptMove as relayToPanels } from './panel-transcript-relay.js';

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
 * @property {boolean} settled - the first liveness verdict is in: healthy, unhealthy, failed, or no backend to ask
 * @property {string | null} [pendingUrl]
 * @property {string | null} [failedMessage] - the server's sentence while a sidecar's start has failed
 * @property {boolean} [activateOnHealthy] - surface the panel on its first healthy settle (an operator retry)
 * @property {number} misses - consecutive unanswered polls since the panel last answered
 * @property {number | null} missSince - epoch ms of the first of those misses, or null
 */

/**
 * SSE payloads broadcast on /api/files/events, discriminated on `type`.
 * `source: 'agent'` marks a frame as agent-originated (absent = human/browser
 * origin); absent optional keys are omitted by the server, never null. The
 * agent's dock changes are not broadcast: they arrive as `panel_op` frames
 * addressed to one dock client (panel-placement.js's PanelOpFrame).
 * @typedef {object} PanelVisibilityEvent
 * @property {'panel_visibility'} type
 * @property {string} panel
 * @property {boolean} visible
 * @property {'agent'} [source]
 * @property {number} [index] - where an addition goes among the rail's panel
 *   entries (Agent not counted); absent, it goes to the end
 *
 * @typedef {object} RailOrderEvent
 * @property {'rail_order'} type
 * @property {string[]} order - every rail member, top to bottom (Agent not included)
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
 * @typedef {object} AgentActivityEvent
 * @property {typeof import('./activity-format.js').AGENT_ACTIVITY_FRAME} type
 * @property {string} tool
 * @property {{ kind: 'panel' | 'channel' | 'run' | 'artifact' | 'config' | 'ui' | 'review',
 *   panel?: string, detail?: string }} target
 * @property {number} [ts]
 *
 * @typedef {PanelVisibilityEvent | RailOrderEvent | PanelRegisterEvent | AgentActivityEvent} PanelSSEEvent
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
 * AgentActivityEvent).
 * @param {(frame: AgentActivityEvent) => void} handler
 */
export function setActivityStripHandler(handler) {
  onAgentActivity = handler;
}

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
function getRailEl() {
  return railEl;
}

/** The locally surfaced panel id, or null when the workspace slot is empty.
 *  @returns {string | null} */
function getActiveTabId() {
  return activeTabId;
}

/** Does this page know the panel at all (built-in, custom, or runtime-registered)?
 *  @param {string} id @returns {boolean} */
export function isPanelKnown(id) {
  return !!panelState[id];
}

/** May the panel fill a tile — its health poll has settled healthy?
 *  @param {string} id @returns {boolean} */
export function isPanelHealthy(id) {
  return !!panelState[id]?.healthy;
}

/** Does the panel hold rail membership (the server-owned visible set)?
 *  @param {string} id @returns {boolean} */
function isRailMember(id) {
  return visiblePanels.has(id);
}

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
  contentEl = /** @type {HTMLElement} */ (
    containerEl.querySelector('#panel-content') || containerEl.querySelector('.panel-content')
  );
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
    // A health settle also fills any restored tile the settling panel owns.
    ensureActive: () => {
      ensureActivePanel();
      hydrateRestoredTiles();
    },
    activate: activateTab,
    navigatePending: (id) => {
      const s = panelState[id];
      if (s?.iframe && s.pendingUrl) navigatePanel(id, s.pendingUrl);
    },
  });

  initAgentAttention(railEl);

  // Hand the iframe adapter its fallback mount host. When the dockview shell is
  // up, panel iframes live in the adapter's overlay layer instead (dockview
  // re-parents panel content on regroup, which reloads iframes — see the
  // dock-spike verdict and dock-iframe.js); without a shell they mount here.
  // After every whole-layout apply the adapter hands back here, so a restored
  // placeholder gets its panel's iframe and a reset keeps the panel the
  // operator was on.
  initDockIframeAdapter({ fallbackHost: contentEl });
  setPlaceholdersAppliedHandler(settleAppliedLayout);

  // Bridge dockview gestures back to the server-owned panel state: a human dock
  // tab focus / close POSTs the same setPanelFocus / setPanelVisibility the rail
  // and agent use. Wires lazily once the dockview shell is up; no-ops without it.
  initDockSync();

  // The landing must not claim the workspace slot while a stored layout may
  // still arrive (dock-workspace restores it after its own /api/panels round
  // trip): re-run it once the boot layout is final. Latched, so a layout that
  // settled before this registration fires it at once. From then on the
  // surfaced panel is re-derived on every settled dock change (dock-view.js).
  setLayoutRestoredHook(ensureActivePanel);
  setLayoutRestoredHook(() => followDock(() => deriveSurfaced(isPanelKnown, markSurfaced)));

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
    clearActive: () => markSurfaced(null),
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
    activateTab,
    showPanel,
    retireTile,
    labelOf,
    retryStart: (id) => {
      const panel = PANELS.find((p) => p.id === id);
      if (panel) retryPanelStart(panel);
    },
  });

  // A pane in a window of its own (pane-window.js): this module's part of what
  // it needs. The sidebar row follows every change.
  initPaneWindow({
    managedIframe,
    concealPanel,
    panelReturned,
    getPanelPage,
    buildEmbedSrc,
    sendThemeToIframe,
    subscribeTheme,
    withEchoSuppressed,
    // An address alone is not enough: a panel that has not answered would open
    // a window on an error page, and nothing reloads a frame there.
    canLoadFrame: (id) => !!panelState[id]?.healthy && getPanelStandaloneUrl(id) !== null,
    loadFrame: (id) => createIframe(id, { dock: false }),
    clearAccent: (id) => {
      if (activeTabId === id) markSurfaced(null);
    },
    showNotice: (message, doc) => showUndoNotice({ message, doc }),
  });
  onDetachedChange(markDetachedRow);

  // Rail drag-and-drop: a sidebar row dropped on a tile edge or a dock edge
  // opens (or moves) that panel as a new tile there — the drop bridge places
  // it through the engine and the holder — and the landing reveals it through
  // the same activate/show tail every other open path uses.
  initRailDrag({ onRailDrop: landDroppedPanel, labelOf });

  // The dock boots beside this function, not after it, so the Agent tile is on
  // screen for as long as the panel list below takes (seconds, on a loaded host).
  // Its verbs need no panel list; a service tile's do, and it keeps the browser's
  // menu until the list has come.
  setTileContextMenuHandler((id, opts) => id === TERMINAL_RAIL_ID && openTileContextMenu(id, opts));

  // Fetch panel config and filter PANELS before rendering
  let panelConfig = null;
  try {
    panelConfig = await fetchJSON('/api/panels');
    const enabledSet = new Set(panelConfig.enabled || []);

    // Filter built-in panels to only enabled ones
    const activePanels = PANELS.filter((p) => enabledSet.has(p.id));

    // Honor a profile-pinned default panel when it resolves to a real tab.
    // Unknown id (typo, dropped panel) silently falls back so the user
    // doesn't end up on a blank tabset.
    if (panelConfig.default) {
      const knownIds = new Set(activePanels.map((p) => p.id));
      for (const cp of panelConfig.custom || []) knownIds.add(cp.id);
      if (knownIds.has(panelConfig.default)) {
        DEFAULT_PANEL = panelConfig.default;
      } else {
        console.warn(
          `Panel config 'default': ${panelConfig.default} is not an enabled panel; ` +
            `falling back to ${DEFAULT_PANEL_FALLBACK}.`
        );
      }
    }

    // Add custom panels
    for (const cp of panelConfig.custom || []) {
      if (!activePanels.some((p) => p.id === cp.id)) {
        activePanels.push({
          id: cp.id,
          label: titleCase(cp.label || cp.id.toUpperCase()),
          configEndpoint: null,
          healthEndpoint: cp.healthEndpoint, // null = skip health polling
          path: cp.path || '/', // subpath for iframe (e.g. "/panel/")
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

  // Whether the human "+" menu may register URL panels (server config gate).
  allowRuntimePanels = !!panelConfig?.allow_runtime_panels;

  // Config-defined layouts for the "+" menu's Layouts section (empty by default).
  panelPresets = panelConfig?.presets || [];

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

  // Render the sidebar rows
  renderRail();

  // Wire the sidebar's "+ Add panel" control (add menu + Layouts).
  // wirePanelHeaderControls owns the getElementById lookups and the
  // initPanelAddMenu call; the menu is a dumb view reading state through these
  // closures and calling back into the same visibility/register paths the
  // agent uses.
  wirePanelHeaderControls({
    getHiddenPanels,
    allowUrlPanels: () => allowRuntimePanels,
    onShowPanel: showPanel,
    onRegisterUrl: registerUrlPanel,
    getPresets,
    onApplyPreset: applyMenuPreset,
  });

  // Keyboard remove: Delete/Backspace on a focused row takes that panel off the
  // rail, with the same Undo notice as the context menu's "Remove from rail".
  // Delegated — one listener, not one per row.
  railEl.addEventListener('keydown', (e) => {
    if (e.key !== 'Delete' && e.key !== 'Backspace') return;
    if (!(e.target instanceof HTMLElement)) return;
    const id = e.target.closest('.panel-rail-button')?.getAttribute('data-panel-id');
    if (!id) return;
    e.preventDefault();
    // The Agent row closes its pane through the dock (no server-side
    // visibility for it).
    if (id === TERMINAL_RAIL_ID) closeTerminalTile();
    else void removeFromRail(id);
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
        publishPanelUrl(cp.id, cp.url);
        ps.configLoaded = true;
        if (!cp.healthEndpoint) {
          assumeHealthy(cp);
        } else {
          const panel = PANELS.find((p) => p.id === cp.id);
          if (panel) startHealthPolling(panel);
        }
      } else if (ps) {
        // Declared without an address: there is nothing to ask, so the verdict
        // is in and a default that names this panel falls through at once.
        ps.settled = true;
      }
    }
  }

  // Open the panel event stream (panel-sse.js: the /api/files/events
  // subscription, its frame dispatcher, and the membership/close apply paths
  // that dispatcher shares with the reconnect resync). LAST registration of
  // init: the first frame can arrive the moment the stream opens, so every other
  // registration and the rail render must already have happened. Reassigned
  // scalars go as getters/setters and the strip seam as a closure, so the
  // dispatcher always sees the value this module holds now.
  subscribePanelEvents({
    isKnown: isPanelKnown,
    isHealthy: isPanelHealthy,
    getRailEl,
    memberSet: visiblePanels,
    dropMember: dropRailMember,
    activate: activateTab,
    navigate: navigatePanel,
    getActive: getActiveTabId,
    // Forget the surfaced panel; the rail accent follows the dock once the
    // close settles (dock-view.js).
    resetActive: () => {
      activeTabId = null;
    },
    // Forget the surfaced panel AND its rail accent: an op that took the
    // viewed panel away and could surface nothing in its place.
    clearActive: () => markSurfaced(null),
    renderEmpty: renderEmptyState,
    // Closure, not the value: onAgentActivity is reassigned whenever the
    // activity strip registers (setActivityStripHandler), and the dispatcher
    // must reach the handler current at frame time.
    reportActivity: (frame) => onAgentActivity(frame),
  });

  // Hand the surfaces outside the panel stack the same /api/panels payload,
  // then mark the hub ready (panel-startup.js).
  finishStartup(panelConfig);
}

// ---- Panel Queries & Active-Panel State ----

/**
 * Rail-member panels that can leave for a window (pane-window.js's
 * detachBlocker), and no
 * window open for them yet — in catalog order. Unlike getVisiblePanels this
 * INCLUDES the active panel: moving the panel you are looking at is the
 * verb's most common use. The palette's "Open <label> in new window" rows read
 * this; the context menu's row applies the same two gates per panel.
 * @returns {Array<{id: string, label: string}>}
 */
export function getPopoutPanels() {
  return PANELS.filter(
    (p) => visiblePanels.has(p.id) && !isDetached(p.id) && detachBlocker(p.id) === null
  ).map((p) => ({ id: p.id, label: p.label }));
}

/**
 * Relay a transcript move on `key` to every panel (panel-transcript-relay.js).
 * @param {string} key - The session key whose transcript moved.
 */
export const relayTranscriptMove = (key) => relayToPanels(key, Object.values(panelState));

/** The catalog label for a panel id (falls back to the id itself).
 *  Exported for the activity strip, which words panel actions with labels.
 *  @param {string} id @returns {string} */
export const labelOf = (id) => PANELS.find((p) => p.id === id)?.label ?? id;

/**
 * Set the locally-tracked surfaced panel — activeTabId, the rail accent, and
 * the container stamp CSS shapes the workspace by (files.css
 * [data-active-panel]) — or clear all three with null: the panel's tile went
 * away without a successor. Derived from the dock after every whole-layout
 * apply and every settled dock change (dock-view.js), so no accent outlives
 * the tile it marked; fallback mode keeps activateTab's bookkeeping.
 * @param {string | null} id
 */
function markSurfaced(id) {
  activeTabId = id;
  setActive(railEl, id);
  if (containerEl && id) containerEl.dataset.activePanel = id;
  else if (containerEl) delete containerEl.dataset.activePanel;
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
  if (activeTabId === panelId) markSurfaced(null);
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
  markSurfaced(null);
}

/**
 * Settle the view after a whole-layout apply (a reset, a stored-layout
 * restore): fill the restored tiles, and give a dock left with no service
 * tile the panel the operator was on — a reset keeps what they were looking
 * at, in the default arrangement — else the landing's choice. A dock that
 * still shows a service tile is left as it is: opening anything into it
 * would replace that tile's panel (the open rule). Nothing opens while the
 * boot layout may still arrive (the landing waits for it), nor into a
 * session's own layout (`asLeft`, session-layout.js): it shows as it was left.
 * @param {{ asLeft?: boolean }} [options]
 */
function settleAppliedLayout({ asLeft = false } = {}) {
  hydrateRestoredTiles();
  const keep = asLeft ? null : activeTabId;
  // The accent follows the layout now on screen, never the one it replaced;
  // a dock still showing a service tile (or no dock at all) is left as it is.
  if (deriveSurfaced(isPanelKnown, markSurfaced) !== false) return;
  const kept = keep && visiblePanels.has(keep) && panelState[keep]?.healthy && !isDetached(keep);
  if (kept) activateTab(/** @type {string} */ (keep));
  else if (!asLeft) ensureActivePanel();
}

/**
 * Fill every restored placeholder with its panel's iframe. A stored layout
 * brings back a placeholder for every tile that was open, but only the
 * landing panel's activation creates an iframe, so every other restored tile
 * is an empty card until its row is clicked. This runs after every layout
 * apply (the adapter's placeholders-applied seam) and on every health settle,
 * so a tile is filled as soon as both its placeholder and its panel's health
 * are there. It never moves the active tile or the rail accent and reports
 * nothing: the adapter adopts the iframe into the placeholder that already
 * exists (a cached iframe from a closed tile is concealed, and a concealed
 * entry is never re-materialized on its own), and the echo guard covers the
 * one case where it would have to add one. A panel living in its own pop-out
 * window has no tile to fill; its placeholder is pruned instead.
 */
function hydrateRestoredTiles() {
  const api = getDockApi();
  if (!api) return;
  for (const { id, label } of PANELS) {
    const state = panelState[id];
    if (!state?.healthy || !visiblePanels.has(id) || isDetached(id)) continue;
    if (!api.getPanel(PLACEHOLDER_PREFIX + id)) continue;
    const iframe = state.iframe;
    if (iframe?.isConnected && iframe.style.display !== 'none') continue;
    withEchoSuppressed(() =>
      iframe?.isConnected ? adoptIframe(id, iframe, { title: label }) : createIframe(id)
    );
  }
}

/**
 * The panel the empty slot goes to first: what a restored layout shows in the
 * service tile the operator used last, when the dock holds one and this page
 * knows the panel — so a stored arrangement comes back as it was saved — else
 * the profile-pinned default.
 * @returns {string}
 */
function landingTarget() {
  const holder = getLayoutHolder();
  const key = holder ? serviceTileTarget(holder.layout, holder.context())?.showing : undefined;
  return key && panelState[key] ? key : DEFAULT_PANEL;
}

/**
 * Give the empty slot to the best panel available, if any.
 *
 * Health-driven, so {auto: true} keeps it from ever surfacing a hidden panel.
 * Safe to call on every settle: it no-ops once something is active.
 *
 * This is deliberately re-entrant rather than a one-shot at each health
 * transition. Another panel's FIRST healthy transition can land while the
 * wanted panel is still loading its config or waiting on its first health
 * answer — decline then, and let that panel's own settle re-run the policy.
 */
function ensureActivePanel() {
  if (activeTabId) return;
  // A stored layout is applied after a network round trip and may hold the
  // tabs this session starts with: nothing claims the slot before it is in.
  // The boot-layout hook re-runs this. Fallback mode has no layout to wait for.
  if (getDockApi() && !bootLayoutSettled()) return;
  const wanted = landingTarget();
  const ds = panelState[wanted];
  // It may still claim the slot: its config is loading, or its backend has not
  // answered its first probe (a sidecar still starting, a health poll in
  // flight). Decline until its first verdict is in — every path reaches one
  // (panel-lifecycle's `settled`), so a dead panel releases the slot too.
  if (!ds?.configLoaded || !ds.settled) return;
  // Hidden disqualifies it exactly as unhealthy does; activateTab would refuse
  // it anyway, and the slot must not sit empty behind it. So does a panel
  // living in its own window: its tile is retired on purpose.
  const target =
    ds.healthy && visiblePanels.has(wanted) && !isDetached(wanted)
      ? wanted
      : PANELS.find(
          (p) => visiblePanels.has(p.id) && panelState[p.id]?.healthy && !isDetached(p.id)
        )?.id;
  if (target) activateTab(target, { auto: true, landing: true });
}

// ---- Tab Switching ----

/**
 * Focus an already-visible panel. Cross-module callers (the command palette's
 * "Focus panel" pick) MUST pass `{ userInitiated: true }` so the switch is
 * reported to the server via setPanelFocus — omitting it focuses locally only.
 * `landing` marks the boot/empty-slot landing: a surfacing nobody asked for.
 *
 * A panel living in its own window shows there: a requested surfacing brings that window to the front instead, and a
 * health-driven one does nothing — nobody asked for the window.
 * @param {string} panelId
 * @param {{ userInitiated?: boolean, auto?: boolean, landing?: boolean }} [options]
 */
export function activateTab(
  panelId,
  { userInitiated = false, auto = false, landing = false } = {}
) {
  if (isDetached(panelId)) {
    if (!auto) focusWindow(panelId);
    return;
  }
  const state = panelState[panelId];
  if (!state || !state.healthy) return;
  // A panel becoming healthy is not a request to show it. The server owns the
  // visible set, so health-driven activation must never surface a hidden panel
  // — otherwise a panel closed with "×" reappears on its own.
  if (auto && !visiblePanels.has(panelId)) return;

  // A badge is served only by a surfacing someone asked for (rail click, agent
  // focus, palette, dock, arrange): that clears it AND acknowledges it, or the
  // next reload would restore it from history. The landing is asked for by
  // nobody, so the badge it lands on stays until a requested surfacing; panels
  // that refused to surface at the guarded returns above keep theirs too.
  if (!landing) clearBadge(panelId);

  markSurfaced(panelId);

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

  // Re-send current theme and session ID to the newly visible iframe
  // (handles edge cases where a postMessage was missed while hidden/loading)
  sendThemeToIframe(state.iframe);
  sendSessionToIframe(state.iframe);

  // Report user-initiated tab switches to the server (avoids SSE feedback loop)
  if (userInitiated) setPanelFocus(panelId);
}

// ---- Panel Visibility Actions (human "+" / "Remove from rail") ----
//
// These back the human add/remove controls. The command POSTs live in
// panel-commands.js and the server's SSE echo drives the DOM, so a human
// action and an agent MCP call are indistinguishable downstream. A removal goes
// through removeFromRail (panel-menu-policy.js, which adds the Undo notice);
// the "+" menu's reveal path needs a local focus too, so it goes through
// showPanel.

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
 * command palette's Layouts action. One open-arrangement op through the same
 * route an agent uses; the current dock client's op runner applies it and the
 * server prunes the rail to the preset's members, so nothing is applied locally
 * ahead of it.
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
export function navigatePanel(panelId, url) {
  const state = panelState[panelId];
  if (!state) return;
  // The navigation guard's member rule, applied here as the server applies it
  // to the op: a url that leaves this panel's own mount (another path on this
  // origin, another host, an encoded way out) moves nothing, whatever sent it.
  if (!isUrlWithinMount(url, state.url)) {
    console.warn(`[panels] refused a url for ${panelId} that leaves its mount`);
    return;
  }

  // Store the target URL so that createIframe() picks it up if the iframe
  // hasn't been lazy-loaded yet (e.g. an agent open before the user
  // has ever clicked the tab).
  state.pendingUrl = url;

  if (!state.iframe) return;
  // An iframe exists only for a panel that has answered, so an unhealthy one
  // has just missed a poll. Loading a new address now would trade the last
  // good page for an error page. The address waits in pendingUrl.
  if (!state.healthy) return;

  if (navigateInPlace(state, url)) return; // the document routes it, keeping its state
  noteDetachedNavigation(panelId, url);
  // Through frame resolution, as at first load: an opaque panel is re-framed
  // at a freshly minted cap (its old one may have lapsed), a first-party one
  // at the plain url, which buildEmbedSrc keeps verbatim (already prefixed).
  framePanel(
    PANELS.find((p) => p.id === panelId),
    panelId,
    state.iframe,
    url,
    state.url ?? url
  );
  state.pendingUrl = null;
}

/** Human-local twin of an agent's open with a url: navigate a panel to `url`
 *  and plain-activate it. No agent glow, no server broadcast.
 *  @param {string} panelId
 *  @param {string} url */
export function navigateAndActivatePanel(panelId, url) {
  if (url) {
    // A url that leaves the panel's own mount moves nothing and surfaces
    // nothing: the sender may be an agent-authored page in a sandboxed panel.
    if (!isUrlWithinMount(url, panelState[panelId]?.url)) {
      console.warn(`[panels] refused a url for ${panelId} that leaves its mount`);
      return;
    }
    navigatePanel(panelId, url);
  }
  activateTab(panelId);
}

// ---- Iframe Management ----

/** Build + adopt the panel's iframe via panel-iframe-factory.js (which owns
 *  everything about the element); this wrapper only binds the private state.
 *  @param {string} panelId
 *  @param {{ dock?: boolean }} [options] */
function createIframe(panelId, options) {
  createPanelIframe(
    PANELS.find((p) => p.id === panelId),
    panelId,
    panelState[panelId],
    contentEl,
    options
  );
  watchNavigateReady(panelState[panelId]);
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
  return hiddenPanels(PANELS, visiblePanels);
}

/**
 * Visible panels excluding the active one, in PANELS order. "Focus" on the
 * already-active panel is a no-op, so activeTabId is filtered out.
 * @returns {Array<{id: string, label: string}>}
 */
export function getVisiblePanels() {
  return visiblePanelsExcept(PANELS, visiblePanels, activeTabId);
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
 * Standalone (non-embedded) URL for a service panel. state.url is the already-proxied root-relative base; the optional catalog
 * path suffixes custom panels' UI root. Null until the panel's config fetch
 * has resolved a URL, and for an opaque-origin panel, which has no first-party
 * page.
 * @param {string} panelId
 * @returns {string | null}
 */
export function getPanelStandaloneUrl(panelId) {
  return standaloneUrl(PANELS, panelState[panelId], panelId);
}

/**
 * Currently surfaced panel id — the `data-active-panel` stamp activateTab
 * writes on the container — or null before init / with no active panel.
 * Consumed by the agent-activity suppression table.
 * @returns {string | null}
 */
export function getActivePanel() {
  return containerEl?.dataset.activePanel ?? null;
}

// ---- Empty State ----

/**
 * Thin binding of the extracted placeholder card (panel-empty-state.js) onto
 * this module's private container/content refs.
 * @param {string} message
 */
const renderEmptyState = (message) => renderEmptyStateInto(containerEl, contentEl, message);
