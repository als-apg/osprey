// @ts-check
/* OSPREY Web Terminal — Tabbed Panel Manager
 *
 * Manages horizontal header tabs for the right panel. Each tab corresponds
 * to an embedded service (Workspace, ARIEL logbook, etc.) loaded in an
 * iframe. Tabs show health LEDs, iframes are lazy-loaded and cached so
 * switching between tabs is instant.
 */

import { fetchJSON, createEventSource } from './api.js';
import { getTheme } from '/design-system/js/theme-manager.js';
import { getCurrentSessionId } from './terminal.js';
import { applyPreset, wirePanelHeaderControls } from './panel-presets.js';
import { setPanelVisibility, setPanelFocus, registerUrlPanel } from './panel-commands.js';

// ---- Types ----

/**
 * @typedef {object} Panel
 * @property {string} id
 * @property {string} label
 * @property {string | null} configEndpoint
 * @property {string | null} [healthEndpoint] - null/undefined means skip health polling
 * @property {string | null} statusBarId
 * @property {string} [path] - iframe subpath for custom panels (e.g. "/panel/")
 */

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
 * @typedef {object} PanelFocusEvent
 * @property {'panel_focus'} type
 * @property {string} panel
 * @property {string} [url]
 *
 * @typedef {object} PanelVisibilityEvent
 * @property {'panel_visibility'} type
 * @property {string} panel
 * @property {boolean} visible
 *
 * @typedef {object} PanelRegisterEvent
 * @property {'panel_register'} type
 * @property {string} id
 * @property {string} [label]
 * @property {string} [url]
 * @property {string} [healthEndpoint]
 * @property {string} [path]
 *
 * @typedef {PanelFocusEvent | PanelVisibilityEvent | PanelRegisterEvent} PanelSSEEvent
 */

// ---- Panel Registry ----

/** @type {Panel[]} */
const PANELS = [
  {
    id: 'artifacts',
    label: 'WORKSPACE',
    configEndpoint: '/api/artifact-server', // data string; fetchJSON prefixes it in initPanel()
    healthEndpoint: null,    // embedded same-origin — skip health polling
    statusBarId: null,       // no dedicated status-bar item
  },
  {
    id: 'ariel',
    label: 'ARIEL',
    configEndpoint: '/api/ariel-server', // data string; fetchJSON prefixes it in initPanel()
    statusBarId: 'ariel-status',
  },
  {
    id: 'channel-finder',
    label: 'CHANNELS',
    configEndpoint: '/api/channel-finder-server', // data string; fetchJSON prefixes it in initPanel()
    statusBarId: 'channel-finder-status',
  },
  {
    id: 'lattice',
    label: 'LATTICE',
    configEndpoint: '/api/lattice-server', // data string; fetchJSON prefixes it in initPanel()
    statusBarId: null,
  },
  {
    id: 'okf',
    label: 'KNOWLEDGE',
    configEndpoint: '/api/okf-server', // data string; fetchJSON prefixes it in initPanel()
    statusBarId: null,
  },
];

// ---- State ----

let containerEl = null;
// Assigned once in initPanelManager and guarded there; other functions run
// only after that, so the refs are treated as non-null past init.
let tabsEl = /** @type {HTMLElement} */ (/** @type {unknown} */ (null));
let contentEl = /** @type {HTMLElement} */ (/** @type {unknown} */ (null));
/** @type {string | null} */
let activeTabId = null;

// Per-panel state: { url, healthy, iframe, pollTimer, configLoaded }
/** @type {Record<string, PanelState>} */
const panelState = {};

// Visible panel ids — seeded from /api/panels at init; controls tab-hidden visibility.
// Task 3.2 toggles individual ids here and flips .tab-hidden on the button.
const visiblePanels = new Set();

// Default panel to activate first. The hardcoded value is the fallback used
// when /api/panels doesn't pin one (kept in sync with
// osprey.profiles.web_panels.DEFAULT_PANEL_FALLBACK on the backend).
// Profile-pinned values arrive via panelConfig.default in initPanelManager
// and replace this at startup.
const DEFAULT_PANEL_FALLBACK = 'artifacts';
let DEFAULT_PANEL = DEFAULT_PANEL_FALLBACK;

// Whether the server permits runtime URL-panel registration (web.allow_runtime_panels).
// Read from /api/panels at init; gates the "new panel from URL" row in the add menu.
let allowRuntimePanels = false;

// Config-defined panel presets ("Layouts") from /api/panels (web.presets), in
// config order. Empty unless a facility opts in; feeds the "+" menu's Layouts section.
/** @type {{name: string, panels: string[]}[]} */
let panelPresets = [];

// ---- Public API ----

/**
 * Initialize the tabbed panel manager inside the given container element.
 * @param {string} panelId
 */
export async function initPanelManager(panelId) {
  containerEl = document.getElementById(panelId);
  if (!containerEl) return;

  tabsEl = /** @type {HTMLElement} */ (document.getElementById('header-tabs'));
  contentEl = /** @type {HTMLElement} */ (containerEl.querySelector('#panel-content') || containerEl.querySelector('.panel-content'));
  if (!tabsEl || !contentEl) return;

  // Fetch panel config and filter PANELS before rendering
  let panelConfig = null;
  try {
    panelConfig = await fetchJSON('/api/panels'); // fetchJSON prefixes this internally
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

  // Whether the human "+" menu may register URL panels (server config gate).
  allowRuntimePanels = !!panelConfig?.allow_runtime_panels;

  // Config-defined layouts for the "+" menu's Layouts section (empty by default).
  panelPresets = panelConfig?.presets || [];

  // Render tab buttons
  renderTabs();

  // Wire the header "+" control (add menu + Layouts). wirePanelHeaderControls
  // owns the getElementById lookups and the initPanelAddMenu call; the menu is a
  // dumb view reading state through these closures and calling back into the same
  // visibility/register paths the agent uses.
  wirePanelHeaderControls({
    getHiddenPanels: () => PANELS.filter(p => !visiblePanels.has(p.id)).map(p => ({ id: p.id, label: p.label })),
    allowUrlPanels: () => allowRuntimePanels,
    onShowPanel: showPanel,
    onRegisterUrl: registerUrlPanel,
    getPresets: () => panelPresets,
    onApplyPreset: applyMenuPreset,
  });

  // Keyboard close: Delete/Backspace on a focused tab hides that panel (the "×"
  // is mouse-only/decorative). Delegated — one listener rather than one per tab.
  tabsEl.addEventListener('keydown', (e) => {
    if (e.key !== 'Delete' && e.key !== 'Backspace') return;
    if (!(e.target instanceof HTMLElement)) return;
    const id = e.target.closest('.header-tab')?.getAttribute('data-panel-id');
    if (id) { e.preventDefault(); setPanelVisibility(id, false); }
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
        if (cp.healthEndpoint) {
          const panel = PANELS.find(p => p.id === cp.id);
          if (panel) startHealthPolling(panel);
        } else {
          ps.healthy = true;
          updateTabState(cp.id);  // no poll loop will ever repaint this tab
        }
      }
    }
  }

  // Listen for SSE events via createEventSource (api.js) so the URL picks up
  // window.__OSPREY_PREFIX__ under multi-user deployments (empty prefix ⇒
  // unchanged behavior). createEventSource also drives the module-level
  // sseState in api.js, but nothing currently reads getConnectionState().sse
  // (only .ws is consumed, by app.js's status dot), so that side effect is
  // harmless. Three event types are handled:
  //
  //   panel_focus      {type, panel, url?}      — explicit switch_panel MCP call;
  //                                               always honor (user asked for it).
  //   panel_visibility {type, panel, visible}   — show/hide a tab; if the active
  //                                               tab is hidden, switch to the next
  //                                               visible+healthy panel or empty state.
  //   panel_register   {type, id, label, url, healthEndpoint, path}
  //                                             — add a runtime panel; do NOT
  //                                               auto-activate (URL may not be ready).
  createEventSource('/api/files/events', { // prefixed via createEventSource (api.js)
    onMessage: (raw) => {
      try {
        const data = /** @type {PanelSSEEvent} */ (raw);

        if (data.type === 'panel_focus' && data.panel) {
          // Agent asked the panel to switch — honor unconditionally
          if (data.url) navigatePanel(data.panel, data.url);
          activateTab(data.panel);

        } else if (data.type === 'panel_visibility' && data.panel) {
          const { panel, visible } = data;
          // Update the visibility set and flip .tab-hidden on the button
          if (visible) {
            visiblePanels.add(panel);
          } else {
            visiblePanels.delete(panel);
          }
          const btn = tabsEl.querySelector(`[data-panel-id="${panel}"]`);
          if (btn) btn.classList.toggle('tab-hidden', !visible);

          // CC-1: if we just hid the currently active tab, switch away from it
          if (!visible && panel === activeTabId) {
            // Conceal the outgoing iframe immediately so it doesn't bleed through
            panelState[panel]?.iframe?.classList.add('hidden');
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

        } else if (data.type === 'panel_register' && data.id) {
          // Seed visibility before addPanel so buildTabButton produces a visible tab
          visiblePanels.add(data.id);
          addPanel(data);
          // Guarantee no .tab-hidden in case the set was already populated (re-register path)
          const btn = tabsEl.querySelector(`[data-panel-id="${data.id}"]`);
          if (btn) btn.classList.remove('tab-hidden');
          // CC-3: do NOT call activateTab — the new panel's URL may not be ready yet;
          // the user activates when they want it.
        }

      } catch { /* ignore malformed events */ }
    },
  });
}

// ---- Tab Rendering ----

/**
 * Build a single tab button element for a given panel spec.
 * Shared by renderTabs() (initial render) and addPanel() (runtime additions)
 * so both paths produce byte-identical DOM nodes.
 *
 * Applies .tab-hidden when the panel id is absent from visiblePanels.
 * Task 3.2 toggles that class (and the visiblePanels set) on show/hide SSE events.
 * @param {Panel} panel
 */
function buildTabButton(panel) {
  const tab = document.createElement('button');
  tab.className = 'header-tab disabled';
  tab.dataset.panelId = panel.id;
  tab.title = panel.label;
  // Build children via DOM to avoid XSS — panel.label comes from server JSON / SSE
  const led = document.createElement('span');
  led.className = 'tab-led offline';
  tab.appendChild(led);
  tab.appendChild(document.createTextNode(panel.label));
  tab.addEventListener('click', () => activateTab(panel.id, { userInitiated: true }));
  // Per-tab close "×" (browser-tab style, revealed on hover via CSS). Decorative
  // (aria-hidden) so it is not an interactive control nested inside the tab
  // button; keyboard users close via Delete on the focused tab (see init).
  const close = document.createElement('span');
  close.className = 'tab-close';
  close.setAttribute('aria-hidden', 'true');
  close.title = `Close ${panel.label}`;
  close.textContent = '×';
  close.addEventListener('click', (e) => {
    e.stopPropagation();  // don't activate the tab we're closing
    setPanelVisibility(panel.id, false);
  });
  tab.appendChild(close);
  if (!visiblePanels.has(panel.id)) tab.classList.add('tab-hidden');
  return tab;
}

function renderTabs() {
  tabsEl.innerHTML = '';
  for (const panel of PANELS) {
    tabsEl.appendChild(buildTabButton(panel));
  }
}

/**
 * Register a runtime panel and append its tab without wiping existing tabs.
 *
 * spec shape (matches the panel_register SSE broadcast payload):
 *   { id, label, url, healthEndpoint, path }
 *
 * Guard: if panelState[id] already exists (re-register), refresh the url
 * in-place rather than duplicating the tab or state.
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

  panelState[spec.id] = {
    url: null,
    healthy: false,
    iframe: null,
    pollTimer: null,
    polling: false,
    configLoaded: false,
  };

  // Append exactly one tab. NEVER call renderTabs() here — it does
  // innerHTML='' which wipes every live tab's active/disabled/LED state.
  tabsEl.appendChild(buildTabButton(normalized));

  // Seed url and health, mirroring the custom-panel block in initPanelManager
  if (spec.url) {
    const ps = panelState[spec.id];
    ps.url = spec.url;
    ps.configLoaded = true;
    if (!spec.healthEndpoint) {
      ps.healthy = true;
      updateTabState(spec.id);
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
      state.healthy = true;
      updateTabState(panel.id);
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

/** @param {Panel} panel */
function startHealthPolling(panel) {
  const state = panelState[panel.id];
  pollHealth(panel);

  // Fast retry during startup (500ms), slow down to 10s once healthy
  let delay = 500;
  function scheduleNext() {
    state.pollTimer = setTimeout(() => {
      pollHealth(panel).then(() => {
        if (state.healthy) {
          // Switch to slow maintenance polling
          state.pollTimer = setInterval(() => pollHealth(panel), 10000);
        } else {
          delay = Math.min(delay * 1.5, 5000);
          scheduleNext();
        }
      });
    }, delay);
  }
  scheduleNext();
}

/** @param {Panel} panel */
async function pollHealth(panel) {
  const state = panelState[panel.id];
  if (!state.url || state.polling) return;
  state.polling = true;

  try {
    // Use the panel's configured health endpoint — hardcoding /health only
    // worked while every panel happened to use it (tiled's is /api/v1/).
    // state.url is already the server-prefixed `<prefix>/panel/<id>` (routes/
    // panels.py's compute_url_prefix()) — a root-relative path — so this string
    // concat is safe: fetch() resolves it against the current origin,
    // preserving the prefix as-is. Do not re-derive or re-prefix it here.
    const resp = await fetch(`${state.url}${panel.healthEndpoint || '/health'}`, {
      signal: AbortSignal.timeout(2000),
    });
    const wasHealthy = state.healthy;
    state.healthy = resp.ok;
    updateTabState(panel.id);
    updateStatusBar(panel);

    // First time healthy — let the shared policy decide whether this
    // newly-healthy panel should take an empty slot.
    if (state.healthy && !wasHealthy) {
      ensureActivePanel();
    }
  } catch {
    state.healthy = false;
    updateTabState(panel.id);
    updateStatusBar(panel);
  } finally {
    state.polling = false;
  }
}

// ---- Tab State ----

/**
 * Repaint a tab's visual state from panelState: the LED class, and — once
 * healthy — the initial 'disabled' state (tabs are never re-disabled).
 * @param {string} panelId
 */
function updateTabState(panelId) {
  const tab = tabsEl.querySelector(`[data-panel-id="${panelId}"]`);
  if (!tab) return;

  if (panelState[panelId].healthy) tab.classList.remove('disabled');
  const led = tab.querySelector('.tab-led');
  if (led) {
    led.className = 'tab-led ' + (panelState[panelId].healthy ? 'healthy' : 'offline');
  }
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

// ---- Theme Sync ----

/**
 * Send the hub's current theme to one iframe. Always sends — never
 * conditioned on whether the id differs from the last send — because a
 * hidden iframe can read empty custom properties on Firefox and needs a
 * fresh, unconditional resend once it's visible again (theme-manager's
 * hidden-iframe protocol; see that module's docstring). Broadcasting to
 * every iframe on every hub theme change is theme-manager's own job (hub
 * role); this is only the two per-iframe trigger points panel-manager
 * owns: iframe creation and tab activation.
 *
 * 'osprey-theme-change' is the one message type theme-manager's follower
 * role (and every embedded interface) actually listens for.
 * @param {HTMLIFrameElement | null} iframe
 */
function sendThemeToIframe(iframe) {
  if (!iframe?.contentWindow) return;
  try {
    iframe.contentWindow.postMessage({ type: 'osprey-theme-change', theme: getTheme() }, window.location.origin);
  } catch { /* cross-origin */ }
}

/**
 * Send the hub's active session id to one iframe — the twin of
 * sendThemeToIframe(), owned by the same two per-iframe trigger points
 * (iframe creation and tab activation). Guards on a missing contentWindow
 * (a not-yet-loaded or detached iframe) and on there being no active
 * session yet, and swallows the cross-origin postMessage throw the same
 * way its theme twin does.
 *
 * 'osprey-session-change' is the message type embedded interfaces listen
 * for to scope their view to the hub's active session.
 * @param {HTMLIFrameElement | null} iframe
 */
function sendSessionToIframe(iframe) {
  if (!iframe?.contentWindow) return;
  const sid = getCurrentSessionId();
  if (!sid) return;
  try {
    iframe.contentWindow.postMessage({ type: 'osprey-session-change', session_id: sid }, window.location.origin);
  } catch { /* cross-origin */ }
}

// ---- Tab Switching ----

/**
 * @param {string} panelId
 * @param {{ userInitiated?: boolean, auto?: boolean }} [options]
 */
function activateTab(panelId, { userInitiated = false, auto = false } = {}) {
  const state = panelState[panelId];
  if (!state || !state.healthy) return;
  // A panel becoming healthy is not a request to show it. The server owns the
  // visible set, so health-driven activation must never surface a hidden panel
  // — otherwise a panel closed with "×" reappears on its own.
  if (auto && !visiblePanels.has(panelId)) return;

  activeTabId = panelId;

  // Update tab active states
  for (const tab of tabsEl.querySelectorAll('.header-tab')) {
    tab.classList.toggle('active', /** @type {HTMLElement} */ (tab).dataset.panelId === panelId);
  }

  // Hide all iframes
  for (const panel of PANELS) {
    const ps = panelState[panel.id];
    if (ps.iframe) ps.iframe.classList.add('hidden');
  }

  // Show or create the selected iframe. isConnected guards a cached ref that was
  // detached by renderEmptyState's innerHTML wipe — rebuild (and clear the stale
  // empty-state placeholder) rather than re-showing a node no longer in the DOM.
  if (state.iframe && state.iframe.isConnected) {
    state.iframe.classList.remove('hidden');
  } else {
    contentEl.querySelector('.artifacts-empty-state')?.remove();
    createIframe(panelId);
  }

  // Re-send current theme and session ID to the newly visible iframe
  // (handles edge cases where a postMessage was missed while hidden/loading)
  sendThemeToIframe(state.iframe);
  sendSessionToIframe(state.iframe);

  // Report user-initiated tab switches to the server (avoids SSE feedback loop)
  if (userInitiated) setPanelFocus(panelId);
}

// ---- Panel Visibility Actions (human "+" / "×") ----
//
// These back the human add/remove controls. The command POSTs live in
// panel-commands.js and the server's SSE echo drives the DOM, so a human
// action and an agent MCP call are indistinguishable downstream. The per-tab
// "×" calls setPanelVisibility(id, false) directly (see buildTabButton); the
// "+" menu's reveal path needs a local focus too, so it goes through showPanel.

/**
 * Reveal a hidden panel and focus it (a "Show panel" menu pick). The visibility
 * POST un-hides the tab for every client via SSE; activateTab focuses it here
 * when it's healthy (and no-ops otherwise, leaving the tab visible but unfocused).
 * @param {string} panelId
 */
function showPanel(panelId) {
  setPanelVisibility(panelId, true);
  activateTab(panelId, { userInitiated: true });
}

/**
 * Apply a config-defined preset ("Layout") exclusively: show its members, hide
 * every visible non-member, focus the first healthy member locally. Feeds
 * panel-manager's live state into the pure applyPreset() orchestrator; each
 * show/hide rides the same setPanelVisibility POST + SSE echo the "+"/"×" use.
 * focus is a purely-local activateTab (no visibility re-POST for the primary).
 * @param {string[]} panels
 */
function applyMenuPreset(panels) {
  applyPreset(panels, {
    getVisible: () => visiblePanels,
    getKnown: () => new Set(PANELS.map(p => p.id)),
    isHealthy: (id) => !!panelState[id]?.healthy,
    setVisibility: setPanelVisibility,
    focus: (id) => activateTab(id, { userInitiated: true }),
  });
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

  // `url` (from the panel_focus SSE payload) is root-relative. `new URL(path,
  // origin)` for a leading-slash path keeps the full path verbatim — it does
  // not resolve against the origin's own path — so an already-prefixed path
  // stays prefixed and an unprefixed one stays unprefixed. Do not strip or
  // re-add window.__OSPREY_PREFIX__ here.
  const embedUrl = new URL(url, window.location.origin);
  embedUrl.searchParams.set('embedded', 'true');
  embedUrl.searchParams.set('theme', /** @type {string} */ (getTheme()));
  state.iframe.src = embedUrl.toString();
  state.pendingUrl = null;
}

// ---- Iframe Management ----

/** @param {string} panelId */
function createIframe(panelId) {
  const state = panelState[panelId];
  if (!state.url) return;

  const iframe = document.createElement('iframe');
  iframe.className = 'panel-iframe';
  iframe.dataset.panelId = panelId;
  // Use pendingUrl (from navigatePanel) if available, otherwise base URL.
  // For custom panels with a subpath (e.g. path: "/panel/"), append it so
  // the iframe loads the UI root rather than the API root.
  const panel = PANELS.find(p => p.id === panelId);
  const panelPath = panel?.path && panel.path !== '/' ? panel.path : '';
  const baseUrl = state.pendingUrl || (state.url + panelPath);
  const targetUrl = baseUrl;
  state.pendingUrl = null;
  // state.url is already the server-prefixed `<prefix>/panel/<id>` (2.2), so
  // targetUrl is root-relative and already carries the prefix. `new URL(path,
  // origin)` for a leading-slash path preserves the full path verbatim —
  // see the matching comment in navigatePanel() above.
  const embedUrl = new URL(targetUrl, window.location.origin);
  embedUrl.searchParams.set('embedded', 'true');
  embedUrl.searchParams.set('theme', /** @type {string} */ (getTheme()));
  iframe.src = embedUrl.toString();
  iframe.sandbox = 'allow-scripts allow-same-origin allow-popups allow-forms allow-modals';

  iframe.addEventListener('load', () => {
    iframe.classList.add('loaded');
    // Sync theme + session immediately so there's no flash of the wrong
    // theme and the embedded app scopes to the active session.
    sendThemeToIframe(iframe);
    sendSessionToIframe(iframe);
  });

  contentEl.appendChild(iframe);
  state.iframe = iframe;

  // Forward resize events to the iframe so embedded apps re-render
  const observer = new ResizeObserver(() => {
    if (iframe.contentWindow) {
      try {
        iframe.contentWindow.dispatchEvent(new Event('resize'));
      } catch {
        // cross-origin — nothing we can do
      }
    }
  });
  observer.observe(contentEl);
}

// ---- Empty State ----

/** @param {string} message */
function renderEmptyState(message) {
  if (!contentEl) return;
  contentEl.innerHTML = `
    <div class="artifacts-empty-state">
      <div class="artifacts-empty-icon">
        <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1">
          <path d="M12 2L2 7l10 5 10-5-10-5z" transform="translate(12 8) scale(1.2)"/>
          <path d="M2 17l10 5 10-5" transform="translate(12 8) scale(1.2)"/>
          <path d="M2 12l10 5 10-5" transform="translate(12 8) scale(1.2)"/>
        </svg>
      </div>
      <div class="artifacts-empty-title">Services</div>
      <div class="artifacts-empty-text">${message}</div>
    </div>
  `;
}
