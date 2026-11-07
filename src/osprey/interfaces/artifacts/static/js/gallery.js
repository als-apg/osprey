// @ts-check
/**
 * OSPREY Artifact Gallery — the page entry: a viewer of ONE artifact.
 *
 * The Web Terminal hub's sidebar owns the artifact list and the hub renders the
 * pane header, so this page renders the preview without a header and shows the
 * artifact the hub selects (viewer.js). It opens no event stream: every stream
 * holds one of the browser's six connections to the hub's origin, and the hub
 * already follows the panel's stream, so it forwards the shown artifact's
 * updates and deletes (viewer.js) and acts on the agent's focus itself.
 *
 * The page holds the session-scoped artifact list, because the logbook
 * composer's artifact picker reads it; it is fetched again before the composer
 * opens.
 */
import { getTheme, initTheme, subscribe } from '/design-system/js/theme-manager.js';
import { applyEmbedded } from '/design-system/js/frame-params.js';
import {
  getArtifacts,
  setArtifacts,
  getSelectedArtifact,
  setSelectedArtifact,
  getCurrentSessionId,
  setCurrentSessionId,
  addArtifact,
  removeArtifact,
  inCurrentScope,
  fetchArtifacts,
} from './state.js';
import { initTypeRegistry } from './types.js';
import { createPreviewRenderer } from './preview.js';
import { createViewer } from './viewer.js';
import { printArtifact } from './print.js';
import { openComposeModal } from './logbook.js';
import { renderTimeseriesView, restyleMountedCharts } from './timeseries.js';

// ---- State ----
// artifacts/selectedArtifact/currentSessionId live in state.js behind explicit
// accessors, and typeRegistry lives in types.js (behind getTypeRegistry()) — see
// the imports above. No closure vars here.

const previewRenderer = createPreviewRenderer({
  onTimeseriesNeeded: (container, artifact) => renderTimeseriesView(container, artifact),
});

/**
 * Fetch one artifact the held pages do not include. An entry inside the
 * current session scope is held from then on; one outside it is returned but never held, because the
 * held list is what the scope admits.
 * @param {string} id
 * @returns {Promise<any|null>} the held copy, or the entry, or null when the server has none
 */
async function fetchEntry(id) {
  const resp = await fetch(`/api/artifacts/${encodeURIComponent(id)}`);
  if (!resp.ok) return null;
  const entry = await resp.json();
  if (!inCurrentScope(entry)) return entry;
  addArtifact(entry);
  return getArtifacts().find((a) => a.id === id) || entry;
}

// ---- Viewer ----

/**
 * Show what the hub selected: an entry, or the empty state for null. A
 * repeat select of the entry already on screen leaves the pane alone, because
 * a re-render would reload the preview's frame for nothing.
 * @param {any|null} entry
 */
function showInViewer(entry) {
  if (entry !== null && entry === getSelectedArtifact()) return;
  setSelectedArtifact(entry);
  previewRenderer.renderPreview();
}

// ---- Artifact changes (forwarded by the hub) ----

/**
 * Take the server's new copy of an artifact: merge it into the held entry,
 * and re-render the preview when it is the artifact shown.
 * @param {any} fields
 */
function applyUpdated(fields) {
  const held = getArtifacts();
  const idx = held.findIndex((a) => a.id === fields.id);
  if (idx >= 0) {
    held[idx] = { ...held[idx], ...fields };
    setArtifacts(held);
  }
  const selected = getSelectedArtifact();
  if (selected?.id === fields.id) {
    setSelectedArtifact(idx >= 0 ? held[idx] : { ...selected, ...fields });
    previewRenderer.renderPreview();
  }
}

/**
 * Forget a deleted artifact. Deleting the shown one empties the viewer; the hub
 * then selects another.
 * @param {string} id
 */
function applyDeleted(id) {
  removeArtifact(id);
  if (getSelectedArtifact()?.id === id) {
    setSelectedArtifact(null);
    previewRenderer.renderPreview();
  }
}

// ---- Theme: forward to nested previews + re-style plots ----
//
// initTheme() replaces the old hand-rolled 'osprey-theme-change' listener
// and data-theme MutationObserver: the theme-manager runtime already applies
// broadcasts from the hub and whatever ?theme=/localStorage/data-theme
// theme-boot.js resolved pre-paint. subscribe() below is the one thing still
// gallery-specific: re-forwarding to nested preview iframes (Plotly HTML
// artifacts) and re-styling the visible timeseries chart. It fires on every
// apply, even one that re-applies an unchanged id (the hidden-iframe repair
// path), which is exactly what a hidden preview iframe needs on tab
// activation.
// theme-boot.js already applied data-theme pre-paint, and this attaches the
// postMessage listener for the hub's live broadcasts.
initTheme({ role: 'follower' });

// Every panel page calls it: the frame contract the hub reads.
applyEmbedded();

/**
 * Send a theme to one embedded artifact page.
 * @param {Element} iframe
 * @param {string} theme
 */
function _sendThemeToFrame(iframe, theme) {
  // Intentional '*' (same-origin contract exception): nested preview iframe may be null/cross-origin.
  try {
    /** @type {any} */ (iframe).contentWindow.postMessage(
      { type: 'osprey-theme-change', theme },
      '*'
    );
  } catch {
    // PostMessage to a stale/cross-origin frame is best-effort.
  }
}

subscribe((theme) => {
  // Every iframe the gallery mounts is a served artifact page (the preview
  // pane's viewport) running the same follower bridge, so there is no selector
  // here to keep in step with the markup.
  document.querySelectorAll('iframe').forEach((iframe) => _sendThemeToFrame(iframe, theme));
  restyleMountedCharts();
});

// The load-time half of the same contract: an artifact page that finishes
// loading after the last apply (a newly selected preview) gets the theme in
// force right away instead of waiting for the next change. `load` doesn't bubble, so it is
// caught in the capture phase at the document.
document.addEventListener(
  'load',
  (e) => {
    const theme = getTheme();
    if (theme && e.target instanceof HTMLIFrameElement) _sendThemeToFrame(e.target, theme);
  },
  true
);

// Session changes are unrelated to theming and stay a plain message
// listener (theme-manager owns the 'osprey-theme-change' type now).
window.addEventListener('message', (e) => {
  if (e.origin !== window.location.origin) return;
  if (e.data && e.data.type === 'osprey-session-change' && e.data.session_id) {
    // The same key again means its transcript moved (contract C11): read the
    // list again.
    if (e.data.session_id !== getCurrentSessionId()) setCurrentSessionId(e.data.session_id);
    fetchArtifacts();
  }
});

// ---- Init ----

// Nothing is selected yet: the pane opens on its centred empty state.
previewRenderer.renderPreview();

// The hub says what to show, what to do with it and when it changed (viewer.js).
createViewer({
  getHeld: getArtifacts,
  fetchEntry,
  show: showInViewer,
  update: applyUpdated,
  remove: applyDeleted,
  actions: {
    print: printArtifact,
    // No stream keeps the held list current here, and the composer's
    // picker offers it, so it is fetched again first.
    logbook: (a) => {
      fetchArtifacts().finally(() => openComposeModal({ artifact_id: a.id }));
    },
    delete: (a) => previewRenderer.deleteArtifact(a),
  },
});

initTypeRegistry().then(() => {
  fetchArtifacts();
});
