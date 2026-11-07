// @ts-check
/**
 * OSPREY Artifact Gallery — preview pane shell and delete.
 *
 * Owns `renderPreview` (the pane's description, meta strip and viewport markup)
 * and deleting an artifact. The hub renders this pane's header (title, pager,
 * Open full size, Save) and its actions arrive as messages (viewer.js); the
 * action behind Delete is exposed (`deleteArtifact`) for the viewer to run.
 *
 * The viewport inside the pane — which markup an artifact of a given type renders
 * as, and what has to be mounted into it afterwards — belongs to
 * artifact-viewport.js, the one dispatch for how an artifact renders full-size,
 * and this module calls it: `artifactViewportHtml(a)` for the markup, then
 * `mountArtifactViewport(...)` once it is live. That dispatch in turn owns the
 * calls into preview-content.js's markdown+KaTeX pipeline and JSON viewer.
 *
 * Rendering the timeseries chart/table is timeseries.js's job, supplied as an
 * injected callback.
 *
 * @module preview
 */

import { removeArtifact, getSelectedArtifact, setSelectedArtifact } from './state.js';
import { escapeHtml, formatSize, formatFullTime, artifactPath } from './types.js';
import { artifactViewportHtml, mountArtifactViewport } from './artifact-viewport.js';

/**
 * @typedef {object} PreviewRenderCallbacks
 * @property {(container: HTMLElement, artifact: any) => void} onTimeseriesNeeded - forwarded to artifact-viewport.js's `mountArtifactViewport`; fired with the mounted `.ts-viewport-container` for a timeseries artifact, which the caller (gallery.js, via timeseries.js's renderTimeseriesView) renders the chart/table into
 */

/**
 * Create the gallery's preview renderer: the preview pane (`renderPreview`) and
 * deleting. Bound to the injected timeseries rendering, which is owned elsewhere.
 * @param {PreviewRenderCallbacks} callbacks
 */
export function createPreviewRenderer(callbacks) {
  // ---- Preview Pane ----

  /** @returns {void} */
  function renderPreview() {
    const previewEmpty = document.getElementById('preview-empty');
    const previewContent = document.getElementById('preview-content');
    if (!previewEmpty || !previewContent) return;

    if (!getSelectedArtifact()) {
      previewEmpty.classList.remove('hidden');
      previewContent.classList.add('hidden');
      return;
    }

    previewEmpty.classList.add('hidden');
    previewContent.classList.remove('hidden');

    const a = getSelectedArtifact();

    // The viewport itself is the shared dispatch (artifact-viewport.js). Only
    // the chrome around it is this pane's own.
    const viewportHtml = artifactViewportHtml(a);

    previewContent.innerHTML = `
      ${a.description ? `<div class="preview-desc">${escapeHtml(a.description)}</div>` : ''}
      <div class="preview-meta-strip">
        <span class="preview-meta-item">
          <span class="preview-meta-label">Size</span>
          <span class="preview-meta-value">${formatSize(a.size_bytes)}</span>
        </span>
        <span class="preview-meta-item">
          <span class="preview-meta-label">Created</span>
          <span class="preview-meta-value">${formatFullTime(a.timestamp)}</span>
        </span>
        ${
          a.tool_source
            ? `
        <span class="preview-meta-item">
          <span class="preview-meta-label">Source</span>
          <span class="preview-meta-value">${escapeHtml(a.tool_source)}</span>
        </span>`
            : ''
        }
        <span class="preview-meta-path" id="preview-copy-path" title="Click to copy path">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
          <span class="preview-path-text">${escapeHtml(artifactPath(a))}</span>
        </span>
      </div>
      <div class="preview-viewport">
        ${viewportHtml}
      </div>
    `;

    // Copy path to clipboard
    const copyPathBtn = document.getElementById('preview-copy-path');
    if (copyPathBtn) {
      copyPathBtn.addEventListener('click', () => {
        const path = artifactPath(a);
        navigator.clipboard.writeText(path).then(() => {
          copyPathBtn.classList.add('copied');
          setTimeout(() => copyPathBtn.classList.remove('copied'), 1500);
        });
      });
    }

    // Post-mount step for the types whose content is fetched (timeseries,
    // markdown, JSON), scoped to this pane's own content element.
    mountArtifactViewport(previewContent, a, {
      onTimeseriesNeeded: callbacks.onTimeseriesNeeded,
    });
  }

  // ---- Delete ----

  /**
   * Delete an artifact once the operator confirms: DELETE it, drop it from
   * the held list and the selection, and re-render this pane. A declined confirm does nothing; a
   * failed request leaves every state as it was.
   * @param {any} artifact
   * @returns {Promise<void>}
   */
  async function deleteArtifact(artifact) {
    if (!confirm(`Delete "${artifact.title}"? This cannot be undone.`)) return;
    try {
      const resp = await fetch(`/api/artifacts/${artifact.id}`, { method: 'DELETE' });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      await resp.json();
      removeArtifact(artifact.id);
      if (getSelectedArtifact()?.id === artifact.id) setSelectedArtifact(null);
      renderPreview();
    } catch (err) {
      console.error('Delete failed:', err);
    }
  }

  return { renderPreview, deleteArtifact };
}
