// @ts-check
/**
 * OSPREY Artifact Gallery — type registry and formatting utilities.
 *
 * Stateless/self-contained: the only mutable state is the type registry
 * fetched from `/api/type-registry` (behind a getter, since ES modules only
 * give importers a read-only live view of an exported binding). Everything
 * else here is a pure function of its arguments.
 *
 * @module types
 */

import { fileUrl } from './state.js';
import { formatFacilityTime, viewerSharesFacilityClock } from '/design-system/js/facility-time.js';

// ---- Type Registry ---- //

/** @type {any} */
let typeRegistry = {};

/** @returns {any} */
export function getTypeRegistry() {
  return typeRegistry;
}

/**
 * Fetch the type registry from the API. Silent on failure (matches the
 * original: console-only), leaving `typeRegistry` at its previous value.
 * @returns {Promise<void>}
 */
export async function initTypeRegistry() {
  try {
    const resp = await fetch('/api/type-registry');
    typeRegistry = await resp.json();
  } catch (err) {
    console.error('Failed to load type registry:', err);
  }
}

// ---- Utilities ---- //

// HTML-escaping lives in the design-system's canonical helper — this module
// re-exports it so existing importers (preview.js, preview-content.js) keep
// working unchanged.
export { escapeHtml } from '/design-system/js/dom.js';

/**
 * Parse an artifact timestamp string to a Date, or null if it isn't a
 * usable ISO-shaped date. This is the single source of truth for the
 * timestamp guard: bare `new Date(x)` coercion turns null/numbers/numeric
 * strings like "0" into fabricated epoch/year-2000 dates, so every display
 * formatter that renders an artifact timestamp — the three below, plus
 * print.js's `fmtTime` — routes through this rather than coercing directly.
 * Nullish, non-string, non-ISO-shaped, or NaN input yields null and the
 * caller supplies its own empty/"Unknown"/raw fallback instead of an
 * invented timestamp.
 * @param {any} iso
 * @returns {Date|null}
 */
export function isoToDate(iso) {
  if (iso === null || iso === undefined) return null;
  if (typeof iso !== 'string' || !/^\d{4}-\d{2}-\d{2}/.test(iso)) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

/**
 * Human-readable byte size (B/KB/MB/GB).
 * @param {number} bytes
 * @returns {string}
 */
export function formatSize(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let size = bytes;
  while (size >= 1024 && i < units.length - 1) {
    size /= 1024;
    i++;
  }
  return `${size.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

/**
 * Full facility-clock date + time (e.g. "Jul 3, 2026, 3:45 PM"), naming the
 * zone when the viewer's clock reads differently. Empty string for falsy or
 * non-ISO input (see `isoToDate`).
 * @param {string} [iso]
 * @returns {string}
 */
export function formatFullTime(iso) {
  const d = isoToDate(iso);
  if (!d) return '';
  return formatFacilityTime(d, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    ...(viewerSharesFacilityClock(d) ? {} : { timeZoneName: 'short' }),
  });
}

/**
 * Whether an artifact is a timeseries: its metadata declares the
 * `timeseries` data_type (an archiver read, an array channel read), or the
 * archiver wrote it before `data_type` existed (`tool_source` is
 * `archiver_read`). Never the category: a scalar channel snapshot is not a
 * timeseries. This
 * is the artifact *type* question only — see `hasTimeseriesData` for the
 * "and it actually has a data file to render" variant. Single source of
 * truth so print.js's strategy dispatch and preview.js's viewport choice
 * can never drift on the definition.
 * @param {any} a
 * @returns {boolean}
 */
export function isTimeseries(a) {
  return (
    (!!a.metadata && a.metadata.data_type === 'timeseries') || a.tool_source === 'archiver_read'
  );
}

/**
 * Whether an artifact is a timeseries (see `isTimeseries`) *and* has a data
 * file to fetch and render — the condition preview.js uses to pick the
 * timeseries viewport over the generic type dispatch. print.js's print
 * strategy deliberately uses the looser `isTimeseries` instead: it captures
 * the already-rendered on-screen chart rather than refetching, so it doesn't
 * need the data file.
 * @param {any} a
 * @returns {boolean}
 */
export function hasTimeseriesData(a) {
  return isTimeseries(a) && !!(a.data_file || (a.metadata && a.metadata.data_file));
}

/**
 * URL for "Open in new tab" — uses rendered endpoints for types that
 * browsers can't display natively (markdown, notebook).
 *
 * A page opened in its own tab has no hub parent to follow, so a theme id
 * passed here rides along as `?theme=` (see {@link withTheme}) and the new tab
 * opens in exactly the theme the operator is looking at.
 * @param {any} a
 * @param {string|null} [theme] - concrete theme id to carry, if any
 * @returns {string}
 */
export function openUrl(a, theme) {
  switch (a.artifact_type) {
    case 'markdown':
      return withTheme(`/api/markdown/${encodeURIComponent(a.id)}/rendered`, theme);
    case 'notebook':
      return withTheme(`/api/notebooks/${encodeURIComponent(a.id)}/rendered`, theme);
    default:
      return withTheme(fileUrl(a), theme);
  }
}

/**
 * Stamp a concrete theme id onto an artifact-page URL as `?theme=` — the
 * load-time half of the gallery's theme contract with the pages it embeds or
 * opens (the same shape the web terminal's `buildEmbedSrc()` uses for its
 * panels): theme-boot.js honors `?theme=` ahead of storage and the OS, so the
 * page first-paints in the theme the gallery is showing instead of resolving
 * one of its own. An empty/`null` theme (theme-manager not initialised yet)
 * leaves the URL bare.
 * @param {string} url - a root-relative artifact-page URL
 * @param {string|null} [theme] - concrete theme id to carry, if any
 * @returns {string}
 */
export function withTheme(url, theme) {
  if (!theme) return url;
  const sep = url.includes('?') ? '&' : '?';
  return `${url}${sep}theme=${encodeURIComponent(theme)}`;
}

// ---- Opaque origin (C-OPAQUE mirror) ---- //
//
// The server serves every agent-written document that is not on a closed inert
// list under `Content-Security-Policy: sandbox <tokens>` (opaque_origin.py).
// These mirror that module so the gallery frames and prints an artifact the
// same way the server serves it; tests/interfaces/test_opaque_origin.py holds
// them to it.

/** The sandbox tokens of an opaque document — never `allow-same-origin`. */
export const OPAQUE_SANDBOX = 'allow-scripts allow-modals allow-popups allow-downloads';

/** The closed inert list; a type not named here is opaque. */
const INERT_TYPES = new Set([
  'application/pdf',
  'text/plain',
  'application/json',
  'text/csv',
  'image/png',
  'image/jpeg',
  'image/gif',
  'image/webp',
]);

/**
 * Whether a response of type `mime` is served without the sandbox. Compared
 * case-insensitively with parameters removed; empty or unknown is not inert.
 * @param {string|null|undefined} mime
 * @returns {boolean}
 */
export function isInert(mime) {
  if (!mime) return false;
  return INERT_TYPES.has(mime.split(';', 1)[0].trim().toLowerCase());
}

/**
 * The type the server serves an artifact's {@link openUrl} as: `text` is
 * always text/plain, markdown and notebooks are rendered HTML, everything
 * else goes out under its own `mime_type`.
 * @param {any} a
 * @returns {string}
 */
export function servedMime(a) {
  switch (a.artifact_type) {
    case 'text':
      return 'text/plain';
    case 'markdown':
    case 'notebook':
      return 'text/html';
    default:
      return a.mime_type || '';
  }
}

/**
 * Whether an artifact's page is served opaque (sandboxed, origin "null").
 * @param {any} a
 * @returns {boolean}
 */
export function isOpaque(a) {
  return !isInert(servedMime(a));
}

/**
 * The ` sandbox="…"` attribute for an iframe showing `a`: the C-OPAQUE tokens
 * for an opaque artifact, none for an inert one (Chrome's PDF viewer does not
 * render sandboxed). The attribute and the response CSP intersect, so print
 * needs `allow-modals` in both.
 * @param {any} a
 * @returns {string}
 */
export function sandboxAttr(a) {
  return isOpaque(a) ? ` sandbox="${OPAQUE_SANDBOX}"` : '';
}

/**
 * The shipped default layout's artifact directory, used only when the page
 * carries no `osprey-artifact-dir` meta.
 */
const DEFAULT_ARTIFACTS_DIR = 'var/agent_data/artifacts';

/**
 * The directory the server stamped into index.html as the
 * `osprey-artifact-dir` meta: where the store writes artifacts, repo-relative
 * or absolute. Read per call so this module stays stateless.
 * @returns {string}
 */
function artifactsDir() {
  return (
    document.querySelector('meta[name="osprey-artifact-dir"]')?.getAttribute('content') ||
    DEFAULT_ARTIFACTS_DIR
  );
}

/**
 * The store's path of an artifact's file — the spelling handed to the agent
 * (drag-to-terminal), shown in the preview header, and copied to the clipboard.
 * One definition so those three never drift apart.
 * @param {{filename: string}} a
 * @returns {string}
 */
export function artifactPath(a) {
  return `${artifactsDir()}/${a.filename}`;
}
