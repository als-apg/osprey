/* OSPREY Web Terminal — Tabbed Panel Manager
 *
 * Manages horizontal header tabs for the right panel. Each tab corresponds
 * to an embedded service (Workspace, ARIEL logbook, etc.) loaded in an
 * iframe. Tabs show health LEDs, iframes are lazy-loaded and cached so
 * switching between tabs is instant.
 */

import { fetchJSON } from './api.js';
import { getTheme } from '/design-system/js/theme-manager.js';
import { getCurrentSessionId } from './terminal.js';

// ---- Panel Registry ----

const PANELS = [
  {
    id: 'artifacts',
    label: 'WORKSPACE',
    configEndpoint: '/api/artifact-server',
    healthEndpoint: null,    // embedded same-origin — skip health polling
    statusBarId: null,       // no dedicated status-bar item
  },
  {
    id: 'ariel',
    label: 'ARIEL',
    configEndpoint: '/api/ariel-server',
    statusBarId: 'ariel-status',
  },
  {
    id: 'tuning',
    label: 'TUNING',
    configEndpoint: '/api/tuning-server',
    statusBarId: 'tuning-status',
  },
  {
    id: 'channel-finder',
    label: 'CHANNELS',
    configEndpoint: '/api/channel-finder-server',
    statusBarId: 'channel-finder-status',
  },
  {
    id: 'lattice',
    label: 'LATTICE',
    configEndpoint: '/api/lattice-server',
    statusBarId: null,
  },
  {
    id: 'okf',
    label: 'KNOWLEDGE',
    configEndpoint: '/api/okf-server',
    statusBarId: null,
  },
];

// ---- State ----

let containerEl = null;
let tabsEl = null;
let contentEl = null;
let activeTabId = null;
let userSelectedTab = false;

// Per-panel state: { url, healthy, iframe, pollTimer, configLoaded }
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

// ---- Public API ----

/**
 * Initialize the tabbed panel manager inside the given container element.
 */
export async function initPanelManager(panelId) {
  containerEl = document.getElementById(panelId);
  if (!containerEl) return;

  tabsEl = document.getElementById('header-tabs');
  contentEl = containerEl.querySelector('#panel-content') || containerEl.querySelector('.panel-content');
  if (!tabsEl || !contentEl) return;

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

  // Render tab buttons
  renderTabs();

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
          ps.healthy = true;
          enableTab(cp.id);
        } else {
          const panel = PANELS.find(p => p.id === cp.id);
          if (panel) startHealthPolling(panel);
        }
      }
    }
  }

  // Listen for SSE events (uses raw EventSource to avoid conflicts with the
  // module-level sseState in api.js). Three event types are handled:
  //
  //   panel_focus      {type, panel, url?}      — explicit switch_panel MCP call;
  //                                               always honor (user asked for it).
  //   panel_visibility {type, panel, visible}   — show/hide a tab; if the active
  //                                               tab is hidden, switch to the next
  //                                               visible+healthy panel or empty state.
  //   panel_register   {type, id, label, url, healthEndpoint, path}
  //                                             — add a runtime panel; do NOT
  //                                               auto-activate (URL may not be ready).
  const es = new EventSource('/api/files/events');
  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);

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

    } catch { /* ignore parse errors */ }
  };
}

// ---- Tab Rendering ----

/**
 * Build a single tab button element for a given panel spec.
 * Shared by renderTabs() (initial render) and addPanel() (runtime additions)
 * so both paths produce byte-identical DOM nodes.
 *
 * Applies .tab-hidden when the panel id is absent from visiblePanels.
 * Task 3.2 toggles that class (and the visiblePanels set) on show/hide SSE events.
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
  if (!visiblePanels.has(panel.id)) {
    tab.classList.add('tab-hidden');
  }
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
      enableTab(spec.id);
    } else {
      startHealthPolling(normalized);
    }
  }
}

// ---- Panel Initialization ----

async function initPanel(panel) {
  const state = panelState[panel.id];

  try {
    const config = await fetchJSON(panel.configEndpoint);
    // Artifact server returns { url }, ARIEL returns { url, available }
    if (config.url && (config.available === undefined || config.available)) {
      state.url = config.url;
    }
    // Some panels return a project name for session filtering
    if (config.project) {
      state.project = config.project;
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
      enableTab(panel.id);
      updateTabState(panel);
      if (panel.id === DEFAULT_PANEL) {
        activateTab(panel.id);
      } else if (!activeTabId) {
        activateTab(panel.id);
      }
    } else {
      startHealthPolling(panel);
    }
  }
}

// ---- Health Polling ----

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

async function pollHealth(panel) {
  const state = panelState[panel.id];
  if (!state.url || state.polling) return;
  state.polling = true;

  try {
    // Use the panel's configured health endpoint — hardcoding /health only
    // worked while every panel happened to use it (tiled's is /api/v1/).
    const resp = await fetch(`${state.url}${panel.healthEndpoint || '/health'}`, {
      signal: AbortSignal.timeout(2000),
    });
    const wasHealthy = state.healthy;
    state.healthy = resp.ok;
    updateTabState(panel);
    updateStatusBar(panel);

    // First time healthy — enable tab and auto-activate
    if (state.healthy && !wasHealthy) {
      enableTab(panel.id);
      // Auto-activate: prefer the DEFAULT_PANEL. Only activate a
      // non-default panel if nothing is active yet and the default
      // panel has already been polled and isn't healthy.
      if (panel.id === DEFAULT_PANEL) {
        activateTab(panel.id);
      } else if (!activeTabId) {
        const defaultState = panelState[DEFAULT_PANEL];
        // Only fall back if default panel finished loading config and isn't healthy
        if (defaultState?.configLoaded && !defaultState.healthy) {
          activateTab(panel.id);
        }
      }
    }
  } catch {
    state.healthy = false;
    updateTabState(panel);
    updateStatusBar(panel);
  } finally {
    state.polling = false;
  }
}

// ---- Tab State ----

function enableTab(panelId) {
  const tab = tabsEl.querySelector(`[data-panel-id="${panelId}"]`);
  if (tab) {
    tab.classList.remove('disabled');
  }
}

function updateTabState(panel) {
  const tab = tabsEl.querySelector(`[data-panel-id="${panel.id}"]`);
  if (!tab) return;

  const led = tab.querySelector('.tab-led');
  if (led) {
    led.className = 'tab-led ' + (panelState[panel.id].healthy ? 'healthy' : 'offline');
  }
}

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
 */
function sendThemeToIframe(iframe) {
  if (!iframe?.contentWindow) return;
  try {
    iframe.contentWindow.postMessage({ type: 'osprey-theme-change', theme: getTheme() }, '*');
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
 */
function sendSessionToIframe(iframe) {
  if (!iframe?.contentWindow) return;
  const sid = getCurrentSessionId();
  if (!sid) return;
  try {
    iframe.contentWindow.postMessage({ type: 'osprey-session-change', session_id: sid }, '*');
  } catch { /* cross-origin */ }
}

// ---- Tab Switching ----

function activateTab(panelId, { userInitiated = false } = {}) {
  const state = panelState[panelId];
  if (!state || !state.healthy) return;

  if (userInitiated) userSelectedTab = true;
  activeTabId = panelId;

  // Update tab active states
  for (const tab of tabsEl.querySelectorAll('.header-tab')) {
    tab.classList.toggle('active', tab.dataset.panelId === panelId);
  }

  // Hide all iframes
  for (const panel of PANELS) {
    const ps = panelState[panel.id];
    if (ps.iframe) {
      ps.iframe.classList.add('hidden');
    }
  }

  // Show or create the selected iframe
  if (state.iframe) {
    state.iframe.classList.remove('hidden');
  } else {
    createIframe(panelId);
  }

  // Re-send current theme and session ID to the newly visible iframe
  // (handles edge cases where a postMessage was missed while hidden/loading)
  sendThemeToIframe(state.iframe);
  sendSessionToIframe(state.iframe);

  // Report user-initiated tab switches to the server (avoids SSE feedback loop)
  if (userInitiated) {
    fetch('/api/panel-focus', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ panel: panelId }),
    }).catch(() => {});
  }
}

// ---- Panel Navigation ----

function navigatePanel(panelId, url) {
  const state = panelState[panelId];
  if (!state) return;

  // Store the target URL so that createIframe() picks it up if the iframe
  // hasn't been lazy-loaded yet (e.g. first panel_focus SSE before the user
  // has ever clicked the tab).
  state.pendingUrl = url;

  if (!state.iframe) return;

  const embedUrl = new URL(url, window.location.origin);
  embedUrl.searchParams.set('embedded', 'true');
  embedUrl.searchParams.set('theme', getTheme());
  if (state.project) {
    embedUrl.hash = `#/sessions?project=${encodeURIComponent(state.project)}`;
  }
  state.iframe.src = embedUrl.toString();
  state.pendingUrl = null;
}

// ---- Iframe Management ----

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
  const embedUrl = new URL(targetUrl, window.location.origin);
  embedUrl.searchParams.set('embedded', 'true');
  embedUrl.searchParams.set('theme', getTheme());
  if (state.project) {
    embedUrl.hash = `#/sessions?project=${encodeURIComponent(state.project)}`;
  }
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
