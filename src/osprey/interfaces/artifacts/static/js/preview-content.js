// @ts-check
/**
 * OSPREY Artifact Gallery — markdown+KaTeX view rendering and the JSON viewer.
 *
 * The "fetch an artifact's file, render its content into a given container"
 * half of the preview pane, kept separate from preview.js purely to stay
 * under the 450-line module cap — see that module's docstring for the
 * fuller picture of how the preview pane fits together. The marked + hljs +
 * KaTeX pipeline itself lives in md-render.js (shared with the standalone
 * server-rendered markdown page); `renderMathInMarkdown` is re-exported
 * here so existing importers keep one path. Both pieces here are stateless:
 * no shared mutable state with preview.js, just an artifact object and a
 * container element passed in as args, mirroring how preview.js already
 * consumes escapeHtml/typeBadge from types.js.
 *
 * The JSON viewer (`renderJsonHtml`) predates its call site — it originally
 * shipped unwired, with case "json" falling through to the same read-only
 * iframe as case "text". `renderJsonView` wires it live, mirroring
 * `renderMarkdownView`'s fetch-then-render shape; preview.js's
 * `renderPreview` calls it for the `case "json"` viewport.
 *
 * @module preview-content
 */

import { fileUrl } from "./state.js";
import { escapeHtml } from "./types.js";
import { configureMarked, renderMathInMarkdown } from "./md-render.js";

// ---- Markdown Rendering ---- //

// Re-exported so existing importers (and the tests pinning the pipeline)
// keep the established path; the implementation lives in md-render.js.
export { renderMathInMarkdown };

/**
 * @param {HTMLElement} container
 * @param {any} artifact
 * @returns {Promise<void>}
 */
export async function renderMarkdownView(container, artifact) {
  container.innerHTML = '<div style="padding:16px;color:var(--text-muted)">Loading...</div>';
  try {
    const url = fileUrl(artifact);
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Fetch failed: ${resp.status}`);
    const text = await resp.text();

    configureMarked();

    const mdDiv = document.createElement("div");
    mdDiv.className = "osprey-md-rendered";
    if (typeof marked !== "undefined") {
      // Uses innerHTML with marked.parse() + katex.renderToString() output,
      // both of which produce sanitized HTML from trusted local artifact files.
      mdDiv.innerHTML = renderMathInMarkdown(text);
    } else {
      mdDiv.textContent = text;
    }
    container.innerHTML = "";
    container.appendChild(mdDiv);
  } catch (err) {
    console.error("Markdown render failed:", err);
    container.innerHTML = '<span style="color:var(--text-muted)">Failed to load markdown</span>';
  }
}

// ---- JSON Viewer ---- //

/**
 * Recursively render a JSON value as syntax-highlighted HTML, truncating
 * past depth 6, long strings past 200 chars, long arrays past 20 items, and
 * objects with more than 50 keys (hardening added at review time — the
 * array/string/depth caps were already there, but an object with thousands
 * of keys had no bound and would render every one of them).
 * @param {any} obj
 * @param {number} depth
 * @returns {string}
 */
export function renderJsonHtml(obj, depth) {
  if (depth > 6) return '<span class="json-truncated">[...]</span>';
  if (obj === null) return '<span class="json-null">null</span>';
  if (typeof obj === "boolean") return `<span class="json-bool">${obj}</span>`;
  if (typeof obj === "number") return `<span class="json-num">${obj}</span>`;
  if (typeof obj === "string") {
    if (obj.length > 200) return `<span class="json-str">"${escapeHtml(obj.substring(0, 200))}..."</span>`;
    return `<span class="json-str">"${escapeHtml(obj)}"</span>`;
  }
  if (Array.isArray(obj)) {
    if (obj.length === 0) return '<span class="json-bracket">[]</span>';
    if (obj.length > 20) {
      const items = obj.slice(0, 20).map((v) => `<div class="json-item">${renderJsonHtml(v, depth + 1)}</div>`).join("");
      return `<span class="json-bracket">[</span><div class="json-indent">${items}<div class="json-item"><span class="json-truncated">... ${obj.length - 20} more</span></div></div><span class="json-bracket">]</span>`;
    }
    const items = obj.map((v) => `<div class="json-item">${renderJsonHtml(v, depth + 1)}</div>`).join("");
    return `<span class="json-bracket">[</span><div class="json-indent">${items}</div><span class="json-bracket">]</span>`;
  }
  if (typeof obj === "object") {
    const keys = Object.keys(obj);
    if (keys.length === 0) return '<span class="json-bracket">{}</span>';
    if (keys.length > 50) {
      const items = keys.slice(0, 50).map((k) => `<div class="json-item"><span class="json-key">"${escapeHtml(k)}"</span>: ${renderJsonHtml(obj[k], depth + 1)}</div>`).join("");
      return `<span class="json-bracket">{</span><div class="json-indent">${items}<div class="json-item"><span class="json-truncated">... ${keys.length - 50} more</span></div></div><span class="json-bracket">}</span>`;
    }
    const items = keys.map((k) => `<div class="json-item"><span class="json-key">"${escapeHtml(k)}"</span>: ${renderJsonHtml(obj[k], depth + 1)}</div>`).join("");
    return `<span class="json-bracket">{</span><div class="json-indent">${items}</div><span class="json-bracket">}</span>`;
  }
  return escapeHtml(String(obj));
}

/**
 * Fetch a JSON artifact's file and render it via `renderJsonHtml`, mirroring
 * `renderMarkdownView`'s loading/error shape. Wires the previously-dead
 * `renderJsonHtml` to the `case "json"` viewport in preview.js's
 * `renderPreview`.
 * @param {HTMLElement} container
 * @param {any} artifact
 * @returns {Promise<void>}
 */
export async function renderJsonView(container, artifact) {
  container.innerHTML = '<div style="padding:16px;color:var(--text-muted)">Loading...</div>';
  try {
    const url = fileUrl(artifact);
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Fetch failed: ${resp.status}`);
    const data = await resp.json();
    container.innerHTML = renderJsonHtml(data, 0);
  } catch (err) {
    console.error("JSON render failed:", err);
    container.innerHTML = '<span style="color:var(--text-muted)">Failed to load JSON</span>';
  }
}
