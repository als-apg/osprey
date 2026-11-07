// @ts-check
/**
 * OSPREY Artifact Gallery — preview pane shell and delete.
 *
 * Owns `renderPreview` (the artifact alone fills the frame) and deleting an
 * artifact. The hub renders this pane's tile bar (Info, Download; its Info
 * popover holds the details) and its actions arrive as messages (viewer.js); the
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

import { renderState } from '/design-system/js/panel-state.js';
import { removeArtifact, getSelectedArtifact, setSelectedArtifact } from './state.js';
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
      // The one centred state look (panel-state.js); the list to pick from is
      // the hub's sidebar.
      renderState(previewEmpty, {
        icon: 'file-text',
        message: 'No artifact selected',
        detail: 'Pick one in the sidebar.',
      });
      return;
    }

    previewEmpty.classList.add('hidden');
    previewContent.classList.remove('hidden');

    const a = getSelectedArtifact();

    // The viewport itself is the shared dispatch (artifact-viewport.js).
    previewContent.innerHTML = `
      <div class="preview-viewport">
        ${artifactViewportHtml(a)}
      </div>
    `;

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
