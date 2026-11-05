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
 * **The viewer hook.** An entry whose `kind` is an extension type with a
 * viewer (`viewer_url` in the type registry, a "has a viewer" flag only) is
 * rendered by that type's own ES module: `render(container, artifact, ctx)`
 * with `ctx = {fileUrl, theme, tokens}`. The import URL is the literal
 * viewers path built in `mountViewer`, never the registry's string, so the hub's
 * proxy can rebase it for the embedded gallery. A failed import, a missing
 * `render` export or a throw logs one `console.warn` and re-fills the viewer's
 * own element through the base dispatch below.
 *
 * **The content strip.** The base dispatch's post-mount step also draws the
 * strip of view controls over the types that have one (artifact-strip.js:
 * a plot's zoom, pan and export, a table's copy and CSV export, a note's
 * contents, find and copy), in the standalone preview and in the embedded
 * viewer alike.
 *
 * @module artifact-viewport
 */

import { getTheme } from '/design-system/js/theme-manager.js';
import { fileUrl } from './state.js';
import { escapeHtml, getTypeRegistry, hasTimeseriesData, sandboxAttr, withTheme } from './types.js';
import { renderMarkdownView, renderJsonView } from './preview-content.js';
import { mountArtifactStrip } from './artifact-strip.js';

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
  if (viewerKey(a) !== null) return `<div class="ext-viewer-container"></div>`;
  return baseViewportHtml(a);
}

/**
 * The base dispatch's markup: what {@link artifactViewportHtml} returns for an
 * artifact whose `kind` has no viewer, and what a failed viewer falls back to.
 * @param {any} a
 * @returns {string}
 */
function baseViewportHtml(a) {
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
 * than named by a `src` attribute, and the content strip of the types that
 * have one. A no-op for every other type.
 *
 * Every lookup is scoped to `container` — never `document` — so two mounted
 * viewports can never land one's renderer in the other's element.
 *
 * @param {HTMLElement} container - the element `artifactViewportHtml`'s markup was written into
 * @param {any} a - the same artifact that markup was built from
 * @param {MountOptions} options
 * @returns {Promise<void> | void} a promise settling once a viewer has rendered (or fallen back); nothing for the base types
 */
export function mountArtifactViewport(container, a, options) {
  const key = viewerKey(a);
  if (key !== null) {
    const el = /** @type {HTMLElement|null} */ (container.querySelector('.ext-viewer-container'));
    if (el) return mountViewer(el, a, key, options);
    return;
  }
  mountBaseViewport(container, a, options);
}

/**
 * The base dispatch's post-mount step.
 * @param {HTMLElement} container
 * @param {any} a
 * @param {MountOptions} options
 * @returns {void}
 */
function mountBaseViewport(container, a, options) {
  if (hasTimeseriesData(a)) {
    const tsEl = /** @type {HTMLElement|null} */ (
      container.querySelector('.ts-viewport-container')
    );
    if (tsEl && options.onTimeseriesNeeded) options.onTimeseriesNeeded(tsEl, a);
    return;
  }
  mountArtifactStrip(container, a);

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

// ---- The viewer hook ---- //

/**
 * The design-system colour custom properties a viewer receives as
 * `ctx.tokens`, read off the document at render time.
 */
const VIEWER_TOKENS = [
  '--bg-primary',
  '--bg-secondary',
  '--bg-panel',
  '--bg-elevated',
  '--text-primary',
  '--text-secondary',
  '--text-muted',
  '--border-default',
  '--surface-canvas',
  '--surface-card',
  '--surface-well',
  '--color-accent',
  '--color-accent-light',
  '--color-on-accent',
  '--color-accent-secondary',
  '--color-accent-secondary-light',
  '--color-accent-secondary-hover',
  '--color-success',
  '--color-warning',
  '--color-error',
  '--color-error-hover',
  '--chart-series-1',
  '--chart-series-2',
  '--chart-series-3',
  '--chart-series-4',
  '--chart-series-5',
  '--chart-series-6',
];

/** @type {(url: string) => Promise<any>} */
const defaultImporter = (url) => import(/* @vite-ignore */ url);

/** @type {(url: string) => Promise<any>} */
let importViewer = defaultImporter;

/**
 * Replace the dynamic `import()` a viewer is loaded with (tests); `null`
 * restores the real one.
 * @param {((url: string) => Promise<any>) | null} importer
 * @returns {void}
 */
export function setViewerImporter(importer) {
  importViewer = importer ?? defaultImporter;
}

/**
 * The extension-type key whose viewer renders `a`, or `null` for the base
 * dispatch (no `kind`, an unregistered one, or a type without a viewer).
 * @param {any} a
 * @returns {string|null}
 */
function viewerKey(a) {
  const kind = a && a.kind;
  if (!kind) return null;
  const types = getTypeRegistry().extension_types;
  if (!Array.isArray(types)) return null;
  const ext = types.find((t) => t && t.key === kind);
  return ext && ext.viewer_url ? kind : null;
}

/** @returns {Record<string, string>} */
function readTokens() {
  const styles = getComputedStyle(document.documentElement);
  /** @type {Record<string, string>} */
  const tokens = {};
  for (const name of VIEWER_TOKENS) tokens[name] = styles.getPropertyValue(name).trim();
  return tokens;
}

/**
 * Import `key`'s viewer and render `a` into `el`; on any failure warn once and
 * re-fill `el` through the base dispatch.
 * @param {HTMLElement} el
 * @param {any} a
 * @param {string} key
 * @param {MountOptions} options
 * @returns {Promise<void>}
 */
async function mountViewer(el, a, key, options) {
  try {
    const mod = await importViewer(`/viewers/${encodeURIComponent(key)}.js`);
    if (!mod || typeof mod.render !== 'function') {
      throw new Error('the module exports no render function');
    }
    await mod.render(el, a, { fileUrl: fileUrl(a), theme: getTheme(), tokens: readTokens() });
  } catch (err) {
    console.warn(`Viewer for "${key}" failed; showing the base view.`, err);
    el.innerHTML = baseViewportHtml(a);
    mountBaseViewport(el, a, options);
  }
}
