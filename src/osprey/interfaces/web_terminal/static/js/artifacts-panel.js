/* OSPREY Web Terminal — Artifact Gallery Panel
 *
 * Loads the standalone Artifact Gallery in an iframe inside the right panel.
 * Polls the artifact server health endpoint until it's available, then
 * replaces the empty-state message with the iframe.
 */

import { fetchJSON } from './api.js';

let panelEl = null;
let artifactServerUrl = null;
let pollTimer = null;
let polling = false;
let connected = false;

/**
 * Initialize the artifacts panel in the right-hand pane.
 */
export async function initArtifactsPanel(panelId) {
  panelEl = document.getElementById(panelId);
  if (!panelEl) return;

  // Show empty state while we wait
  renderEmptyState('Connecting to artifact server\u2026');

  // Get the artifact server URL from our backend
  try {
    const config = await fetchJSON('/api/artifact-server');
    artifactServerUrl = config.url;
  } catch {
    artifactServerUrl = 'http://127.0.0.1:8086';
  }

  // Start polling for the artifact server
  pollHealth();
}

function pollHealth() {
  if (connected || polling) return;
  polling = true;

  checkHealth().then((healthy) => {
    polling = false;
    if (healthy) {
      if (pollTimer) clearTimeout(pollTimer);
      pollTimer = null;
      connected = true;
      renderIframe();
    } else {
      pollTimer = setTimeout(pollHealth, 2000);
    }
  });
}

async function checkHealth() {
  try {
    const resp = await fetch(`${artifactServerUrl}/health`, {
      signal: AbortSignal.timeout(1500),
    });
    return resp.ok;
  } catch {
    return false;
  }
}

function renderEmptyState(message) {
  if (!panelEl) return;
  panelEl.innerHTML = `
    <div class="artifacts-empty-state">
      <div class="artifacts-empty-icon">
        <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1">
          <path d="M12 2L2 7l10 5 10-5-10-5z" transform="translate(12 8) scale(1.2)"/>
          <path d="M2 17l10 5 10-5" transform="translate(12 8) scale(1.2)"/>
          <path d="M2 12l10 5 10-5" transform="translate(12 8) scale(1.2)"/>
        </svg>
      </div>
      <div class="artifacts-empty-title">Artifact Gallery</div>
      <div class="artifacts-empty-text">${message}</div>
    </div>
  `;
}

function renderIframe() {
  if (!panelEl) return;
  panelEl.innerHTML = `
    <iframe
      id="artifacts-iframe"
      class="artifacts-iframe"
      src="${artifactServerUrl}"
      sandbox="allow-scripts allow-same-origin allow-popups allow-forms allow-modals"
    ></iframe>
  `;

  const iframe = document.getElementById('artifacts-iframe');
  iframe.addEventListener('load', () => {
    iframe.classList.add('loaded');
  });

  // When the panel resizes (browser window resize, split-pane drag), the
  // iframe element gets new CSS dimensions but no `resize` event fires on
  // the iframe's inner window.  Forward it so the gallery's ResizeObservers
  // pick up the change and re-render charts/images.
  const observer = new ResizeObserver(() => {
    if (iframe.contentWindow) {
      try {
        iframe.contentWindow.dispatchEvent(new Event('resize'));
      } catch {
        // cross-origin — nothing we can do
      }
    }
  });
  observer.observe(panelEl);
}
