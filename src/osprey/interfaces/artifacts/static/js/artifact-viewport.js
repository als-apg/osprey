// @ts-check
/**
 * OSPREY Artifact Gallery — the per-type viewport dispatch.
 *
 * One artifact in, one rendered viewport out. This is the single answer to
 * "how does an artifact of type X render?", and it is deliberately the *only*
 * one: both surfaces that show an artifact full-size call these two functions.
 *
 *   - Expert's preview pane (preview.js's `renderPreview`) — inside its
 *     header/meta/action chrome
 *   - Simple's latest-result card (gallery.js's `renderSimple`) — inside its
 *     own friendlier header (title, NEW badge, Open full size / Save)
 *
 * The two panes differ in the chrome around the viewport, never in the
 * viewport itself, so the chrome stays with each pane and only this dispatch
 * is shared. Simple previously borrowed types.js's `thumbnailHtml()` for its
 * card — the *sidebar card* builder, which knows only the `<img>`/`<iframe>`
 * types and drops markdown/JSON/text/PDF/timeseries to a summary dump or a
 * type-icon placeholder. Two dispatch tables cannot stay in parity; there is
 * now one. `thumbnailHtml` is back to meaning only what its name says.
 *
 * Split in two because the call sites need the halves at different moments:
 * `renderPreview` interpolates the markup into a larger `innerHTML` template
 * (header + meta strip + viewport in one assignment), so it needs the markup
 * as a string before anything is in the DOM, and the post-mount step after.
 * Simple's card writes the markup straight into its own slot and mounts
 * immediately. Hence `artifactViewportHtml()` (pure, returns a string) and
 * `mountArtifactViewport()` (effects, takes the now-live container).
 *
 * **The markup carries classes, no ids.** Both panes exist in the DOM at once
 * — `renderPreview` still runs while Simple is on screen (SSE focus/update
 * events call it) — so a `document.getElementById('md-viewport')` lookup would
 * resolve *both* mounts to whichever pane came first in document order. Every
 * post-mount lookup below is therefore scoped to the container passed in. The
 * classes were already carrying all the CSS; the ids only ever served as
 * lookup handles.
 *
 * Timeseries rendering arrives as an injected callback rather than a direct
 * import, preserving the seam preview.js already had: timeseries.js pulls in
 * the lazy Plotly loader and the theme bridge, which neither pane should drag
 * into its import graph just to dispatch a type. gallery.js wires the real
 * `renderTimeseriesView` at both call sites.
 *
 * @module artifact-viewport
 */

import { fileUrl } from "./state.js";
import { escapeHtml, hasTimeseriesData } from "./types.js";
import { renderMarkdownView, renderJsonView } from "./preview-content.js";

/**
 * Build the viewport markup for one artifact.
 *
 * Timeseries is checked ahead of the type switch: an artifact carrying
 * timeseries data renders as a chart regardless of the `artifact_type` it was
 * saved under (archiver data is routinely a `plot_html` or `image` with a
 * sidecar data file, and the live chart beats the frozen render). Types the
 * browser can't display inline fall back to a download link.
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
    case "plot_html":
    case "table_html":
    case "dashboard_html":
    case "html":
      return `<iframe src="${fileUrl(a)}" class="preview-iframe-light" sandbox="allow-scripts allow-same-origin"></iframe>`;
    case "notebook":
      return `<iframe src="/api/notebooks/${encodeURIComponent(a.id)}/rendered" class="preview-iframe-light" sandbox="allow-scripts allow-same-origin"></iframe>`;
    case "plot_png":
    case "image":
      return `<img src="${fileUrl(a)}" alt="${escapeHtml(a.title)}" />`;
    case "markdown":
      return `<div class="md-preview-container"></div>`;
    case "json":
      return `<div class="json-viewer"></div>`;
    case "text":
      return `<iframe src="${fileUrl(a)}" class="preview-iframe-dark"></iframe>`;
    default:
      if (a.mime_type === "application/pdf") {
        return `<iframe src="${fileUrl(a)}" class="preview-iframe-light"></iframe>`;
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
 * @property {(container: HTMLElement, artifact: any) => void} [onTimeseriesNeeded] - called with the mounted `.ts-viewport-container` for a timeseries artifact; both panes wire this to timeseries.js's `renderTimeseriesView` (see this module's docstring for why it is injected rather than imported). Omitting it simply leaves the container empty.
 */

/**
 * Run the post-mount step for the markup {@link artifactViewportHtml} just
 * produced: the three types whose content is fetched asynchronously rather
 * than named by a `src` attribute. A no-op for every other type.
 *
 * Every lookup is scoped to `container` — never `document` — so the Expert
 * pane and the Simple card can hold a mounted viewport simultaneously without
 * either one's renderer landing in the other's element.
 *
 * @param {HTMLElement} container - the element `artifactViewportHtml`'s markup was written into
 * @param {any} a - the same artifact that markup was built from
 * @param {MountOptions} options
 * @returns {void}
 */
export function mountArtifactViewport(container, a, options) {
  if (hasTimeseriesData(a)) {
    const tsEl = /** @type {HTMLElement|null} */ (container.querySelector(".ts-viewport-container"));
    if (tsEl && options.onTimeseriesNeeded) options.onTimeseriesNeeded(tsEl, a);
    return;
  }

  if (a.artifact_type === "markdown") {
    const mdEl = /** @type {HTMLElement|null} */ (container.querySelector(".md-preview-container"));
    if (mdEl) renderMarkdownView(mdEl, a);
    return;
  }

  if (a.artifact_type === "json") {
    const jsonEl = /** @type {HTMLElement|null} */ (container.querySelector(".json-viewer"));
    if (jsonEl) renderJsonView(jsonEl, a);
  }
}
