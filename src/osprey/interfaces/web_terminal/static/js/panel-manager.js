/* OSPREY Web Terminal — Tabbed Panel Manager
 *
 * Manages horizontal header tabs for the right panel. Each tab corresponds
 * to an embedded service (Workspace, ARIEL logbook, etc.) loaded in an
 * iframe. Tabs show health LEDs, iframes are lazy-loaded and cached so
 * switching between tabs is instant.
 */

import { fetchJSON } from './api.js';
import { getTheme } from './theme.js';

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
    id: 'monitoring',
    label: 'MONITORING',
    configEndpoint: '/api/monitoring-server',
    healthEndpoint: null,    // backend verifies Grafana health before advertising URL
    statusBarId: 'monitoring-status',
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

// Default panel to activate first (must match an id in PANELS)
const DEFAULT_PANEL = 'artifacts';

// ---- Public API ----

/**
 * Initialize the tabbed panel manager inside the given container element.
 */
export function initPanelManager(panelId) {
  containerEl = document.getElementById(panelId);
  if (!containerEl) return;

  tabsEl = document.getElementById('header-tabs');
  contentEl = containerEl.querySelector('#panel-content') || containerEl.querySelector('.panel-content');
  if (!tabsEl || !contentEl) return;

  // Initialize state for each panel
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

  // Render tab buttons
  renderTabs();

  // Fetch config and start health polling for all panels
  for (const panel of PANELS) {
    initPanel(panel);
  }

  // Listen for panel_focus events via SSE (uses raw EventSource to avoid
  // conflicts with the module-level sseState in api.js)
  const es = new EventSource('/api/files/events');
  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data.type === 'panel_focus' && data.panel) {
        if (data.url) navigatePanel(data.panel, data.url);
        if (!userSelectedTab) {
          activateTab(data.panel);
        }
      }
    } catch { /* ignore parse errors */ }
  };
}

// ---- Tab Rendering ----

function renderTabs() {
  tabsEl.innerHTML = '';
  for (const panel of PANELS) {
    const tab = document.createElement('button');
    tab.className = 'header-tab disabled';
    tab.dataset.panelId = panel.id;
    tab.title = panel.label;
    tab.innerHTML = `
      <span class="tab-led offline"></span>
      ${panel.label}
    `;
    tab.addEventListener('click', () => activateTab(panel.id, { userInitiated: true }));
    tabsEl.appendChild(tab);
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
  } catch {
    // Config endpoint not available — panel stays disabled
  } finally {
    state.configLoaded = true;
  }

  if (state.url) {
    // External panels (healthEndpoint === null) skip health polling —
    // mark healthy immediately so the tab is enabled.
    if (panel.healthEndpoint === null) {
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
    const resp = await fetch(`${state.url}/health`, {
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

  // Re-send current theme to the newly visible iframe (handles edge cases
  // where a postMessage was missed while the iframe was hidden or loading)
  if (state.iframe?.contentWindow) {
    try {
      state.iframe.contentWindow.postMessage(
        { type: 'osprey-theme-change', theme: getTheme() },
        '*'
      );
    } catch { /* cross-origin */ }
  }

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
  if (!state?.iframe) return;
  const embedUrl = new URL(url);
  embedUrl.searchParams.set('embedded', 'true');
  embedUrl.searchParams.set('theme', getTheme());
  state.iframe.src = embedUrl.toString();
}

// ---- Iframe Management ----

function createIframe(panelId) {
  const state = panelState[panelId];
  if (!state.url) return;

  const iframe = document.createElement('iframe');
  iframe.className = 'panel-iframe';
  const embedUrl = new URL(state.url);
  embedUrl.searchParams.set('embedded', 'true');
  embedUrl.searchParams.set('theme', getTheme());
  iframe.src = embedUrl.toString();
  iframe.sandbox = 'allow-scripts allow-same-origin allow-popups allow-forms allow-modals';

  iframe.addEventListener('load', () => {
    iframe.classList.add('loaded');
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
