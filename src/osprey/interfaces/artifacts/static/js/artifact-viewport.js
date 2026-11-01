// @ts-check
/**
 * OSPREY Artifact Gallery — the per-type viewport dispatch.
 *
 * One artifact in, one rendered viewport out. This is the single answer to
 * "how does an artifact of type X render?", and it is deliberately the *only*
 * one: the preview pane (preview.js's `renderPreview`) calls these two
 * functions for every artifact it shows, standalone inside its header/meta/
 * action chrome, and embedded as the hub's single-artifact viewer, where the
 * hub renders the header.
 *
 * The chrome stays with the pane and only this dispatch decides what the
 * viewport holds. A full-size surface once borrowed types.js's
 * `thumbnailHtml()` instead — the *sidebar card* builder, which knows only
 * the `<img>`/`<iframe>` types and drops markdown/JSON/text/PDF/timeseries to
 * a summary dump or a type-icon placeholder. Two dispatch tables cannot stay
 * in parity; there is now one. `thumbnailHtml` is back to meaning only what
 * its name says.
 *
 * Split in two because the caller needs the halves at different moments:
 * `renderPreview` interpolates the markup into a larger `innerHTML` template
 * (the pane's chrome and the viewport in one assignment), so it needs the
 * markup as a string before anything is in the DOM, and the post-mount step
 * after. Hence `artifactViewportHtml()` (pure, returns a string) and
 * `mountArtifactViewport()` (effects, takes the now-live container).
 *
 * **The markup carries classes, no ids.** Every post-mount lookup below is
 * scoped to the container passed in, never `document`, so the dispatch never
 * assumes its viewport is the only one in the document: a second mounted
 * viewport can never pull the first one's renderer into its element. The
 * classes were already carrying all the CSS; the ids only ever served as
 * lookup handles.
 *
 * Timeseries rendering arrives as an injected callback rather than a direct
 * import, preserving the seam preview.js already had: timeseries.js pulls in
 * the lazy Plotly loader and the theme bridge, which the pane should not drag
 * into its import graph just to dispatch a type. gallery.js wires the real
 * `renderTimeseriesView` through the preview renderer's callbacks.
 *
 * @module artifact-viewport
 */

import { getTheme } from '/design-system/js/theme-manager.js';
import { fileUrl } from './state.js';
import { escapeHtml, hasTimeseriesData, sandboxAttr, withTheme } from './types.js';
import { renderMarkdownView, renderJsonView } from './preview-content.js';

/**
 * The src for an embedded artifact page: the URL stamped with the theme the
 * gallery is showing right now, so the page first-paints in it (its own
 * theme-boot.js honors `?theme=` first). Without this a freshly mounted
 * preview resolves a theme of its own -- from storage this follower never
 * writes, or the OS -- and can boot dark inside a light gallery until the
 * next theme change reaches it.
 * @param {string} url
 * @returns {string}
 */
function embedSrc(url) {
  return withTheme(url, getTheme());
}

/**
 * Build the viewport markup for one artifact.
 *
 * Timeseries is checked ahead of the type switch: an artifact carrying
 * timeseries data renders as a chart regardless of the `artifact_type` it was
 * saved under (archiver data is routinely a `plot_html` or `image` with a
 * sidecar data file, and the live chart beats the frozen render). Types the
 * browser can't display inline fall back to a download link. Every iframe
 * takes its `sandbox` from {@link sandboxAttr}: the C-OPAQUE tokens when the
 * server serves the page opaque, none for an inert type (PDF, text).
 *
 * Returns a string rather than nodes so `renderPreview` can interpolate it
 * into its one-shot header+meta+viewport `innerHTML`. Callers must follow up
 * with {@link mountArtifactViewport} for the types that need one.
 *
 * @param {any} a - artifact record
 * @returns {string} viewport markup, class-hooked and id-free
 */
export function artifactViewportHtml(a) {
  if (hasTimeseriesData(a)) {
    return `<div class="ts-viewport-container"></div>`;
  }

  switch (a.artifact_type) {
    case 'plot_html':
    case 'table_html':
    case 'dashboard_html':
    case 'html':
      return `<iframe src="${embedSrc(fileUrl(a))}" class="preview-iframe-light"${sandboxAttr(a)}></iframe>`;
    case 'notebook':
      return `<iframe src="${embedSrc(`/api/notebooks/${encodeURIComponent(a.id)}/rendered`)}" class="preview-iframe-light"${sandboxAttr(a)}></iframe>`;
    case 'plot_png':
    case 'image':
      return `<img src="${fileUrl(a)}" alt="${escapeHtml(a.title)}" />`;
    case 'markdown':
      return `<div class="md-preview-container"></div>`;
    case 'json':
      return `<div class="json-viewer"></div>`;
    case 'text':
      return `<iframe src="${fileUrl(a)}" class="preview-iframe-dark"${sandboxAttr(a)}></iframe>`;
    default:
      if (a.mime_type === 'application/pdf') {
        return `<iframe src="${fileUrl(a)}" class="preview-iframe-light"${sandboxAttr(a)}></iframe>`;
      }
      return `<div class="preview-download">
        <a href="${fileUrl(a)}" target="_blank" class="btn btn-secondary">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
          </svg>
          Download ${escapeHtml(a.filename)}
        </a>
      </div>`;
  }
}

/**
 * @typedef {object} MountOptions
 * @property {(container: HTMLElement, artifact: any) => void} [onTimeseriesNeeded] - called with the mounted `.ts-viewport-container` for a timeseries artifact; the preview pane wires this to timeseries.js's `renderTimeseriesView` (see this module's docstring for why it is injected rather than imported). Omitting it simply leaves the container empty.
 */

/**
 * Run the post-mount step for the markup {@link artifactViewportHtml} just
 * produced: the three types whose content is fetched asynchronously rather
 * than named by a `src` attribute. A no-op for every other type.
 *
 * Every lookup is scoped to `container` — never `document` — so two mounted
 * viewports can never land one's renderer in the other's element.
 *
 * @param {HTMLElement} container - the element `artifactViewportHtml`'s markup was written into
 * @param {any} a - the same artifact that markup was built from
 * @param {MountOptions} options
 * @returns {void}
 */
export function mountArtifactViewport(container, a, options) {
  if (hasTimeseriesData(a)) {
    const tsEl = /** @type {HTMLElement|null} */ (
      container.querySelector('.ts-viewport-container')
    );
    if (tsEl && options.onTimeseriesNeeded) options.onTimeseriesNeeded(tsEl, a);
    return;
  }

  if (a.artifact_type === 'markdown') {
    const mdEl = /** @type {HTMLElement|null} */ (container.querySelector('.md-preview-container'));
    if (mdEl) renderMarkdownView(mdEl, a);
    return;
  }

  if (a.artifact_type === 'json') {
    const jsonEl = /** @type {HTMLElement|null} */ (container.querySelector('.json-viewer'));
    if (jsonEl) renderJsonView(jsonEl, a);
  }
}
