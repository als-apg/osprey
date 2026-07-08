// @ts-check
/**
 * OSPREY Artifact Gallery — Print / Save as PDF
 *
 * Injects a print icon button into .preview-header-actions following
 * the same injection pattern as logbook.js.
 *
 * Strategy: opens a dedicated print window for each artifact type so
 * the print output is clean (no gallery chrome) and WYSIWYG:
 *
 *   - iframe-based (plot_html, table_html, dashboard_html, html,
 *     notebook, markdown, text, json): opens a rendered URL in a new
 *     window, injects print-cleanup styles, calls window.print().
 *     Markdown and notebook use server-side rendered endpoints.
 *
 *   - image / plot_png: wraps the image in minimal HTML with a title
 *     header, opens via about:blank window for printing.
 *
 *   - timeseries (Plotly.js in DOM): opens an about:blank window
 *     synchronously (to avoid popup blocking), captures chart via
 *     Plotly.toImage() as PNG, then populates the window and prints.
 */

import { escapeHtml as esc } from "/design-system/js/dom.js";
import { getSelectedArtifact, fileUrl } from "./state.js";
import { openUrl, isTimeseries, isoToDate } from "./types.js";

export { esc };

// ---- SVG icon (printer) ----

const PRINT_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
  '<polyline points="6 9 6 2 18 2 18 9"/>' +
  '<path d="M6 18H4a2 2 0 01-2-2v-5a2 2 0 012-2h16a2 2 0 012 2v5a2 2 0 01-2 2h-2"/>' +
  '<rect x="6" y="14" width="12" height="8"/>' +
  "</svg>";

// ---- Shared styles for self-built print windows ----
//
// Printed output stays light-on-white regardless of the gallery's active
// theme (matches the fleet-wide "@media print never goes dark" rule) — the
// couple of hardcoded grays below are intentional, not migration debt.
// hygiene-allow-color-start

const PRINT_STYLES =
  "body { margin: 0; padding: 24px; font-family: system-ui, -apple-system, sans-serif; " +
  "background: white; color: black; }\n" +
  ".print-header { margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #ccc; }\n" +
  ".print-header h1 { font-size: 18px; font-weight: 600; margin: 0 0 4px; }\n" +
  ".print-header .print-meta { font-size: 11px; color: #666; }\n" +
  ".print-body { width: 100%; }\n" +
  ".print-body img { max-width: 100%; height: auto; display: block; }\n" +
  "@media print { .print-header { page-break-after: avoid; } }\n";

// hygiene-allow-color-end

const MSG_POPUP_BLOCKED = "Print blocked \u2014 please allow pop-ups for this site and try again.";

// ---- Helpers ----

/**
 * Full locale timestamp for the print header. Deliberately keeps the
 * with-seconds `toLocaleString()` form (a print/PDF header wants precision,
 * unlike the gallery's compact `formatFullTime`), but shares types.js's
 * `isoToDate` guard so a malformed/non-ISO timestamp renders "" rather than
 * a fabricated year-2000/epoch date.
 * @param {any} ts
 */
export function fmtTime(ts) {
  const d = isoToDate(ts);
  return d ? d.toLocaleString() : "";
}

/** @param {any} a */
export function headerHtml(a) {
  let meta = "";
  if (a.timestamp) meta += "<span>" + esc(fmtTime(a.timestamp)) + "</span>";
  if (a.tool_source) meta += " &middot; <span>Source: " + esc(a.tool_source) + "</span>";
  if (a.filename) meta += " &middot; <span>" + esc(a.filename) + "</span>";
  return '<div class="print-header">' +
    "<h1>" + esc(a.title || a.filename || "Artifact") + "</h1>" +
    (meta ? '<div class="print-meta">' + meta + "</div>" : "") +
    "</div>";
}

/**
 * Replace a target window's document with parsed HTML via DOM adoption.
 *
 * Resets the window to a minimal shell, parses `html` in the opener's
 * document, then moves the parsed head/body nodes across with adoptNode.
 * Adoption (rather than `document.write(html)`) keeps the about:blank
 * window on the opener's origin, so relative image URLs in `html` resolve
 * and iframed openers aren't blocked. `html` is always assembled from
 * `esc()`-escaped fields by the callers.
 *
 * Shared by `openBlobAndPrint` (image path) and `printTimeseries` (chart
 * capture) so this security-sensitive adoption path has a single copy.
 * @param {Window} w
 * @param {string} html
 */
function populateWindow(w, html) {
  const parsed = new DOMParser().parseFromString(html, "text/html");

  // Initialize the target document with a minimal shell.
  // Safe: writing a hardcoded empty document to our own window.
  w.document.open();
  w.document.write("<!DOCTYPE html><html><head></head><body></body></html>");  // static content only
  w.document.close();

  // Transfer parsed content into the live document via DOM adoption
  while (parsed.head.childNodes.length) {
    w.document.head.appendChild(w.document.adoptNode(parsed.head.childNodes[0]));
  }
  while (parsed.body.childNodes.length) {
    w.document.body.appendChild(w.document.adoptNode(parsed.body.childNodes[0]));
  }
}

/**
 * Opens a new window from an HTML string and triggers the print dialog.
 *
 * Uses about:blank instead of Blob URLs because Blob URLs get their own
 * origin, which (a) breaks relative image URLs embedded in the HTML and
 * (b) gets blocked by browsers when opened from an iframed page.
 * about:blank windows inherit the opener's origin, avoiding both issues.
 * @param {string} html
 */
function openBlobAndPrint(html) {
  const w = window.open("about:blank", "_blank", "width=900,height=700");
  if (!w) {
    alert(MSG_POPUP_BLOCKED);
    return;
  }

  populateWindow(w, html);

  // Wait for images/stylesheets to load before printing
  w.addEventListener("load", function () { w.focus(); w.print(); });
  if (w.document.readyState === "complete") { w.focus(); w.print(); }
}

// ---- Per-type print strategies ----

/** @param {any} a */
function printIframe(a) {
  // openUrl() owns the type -> URL mapping (rendered endpoints for
  // markdown/notebook, raw file URL otherwise).
  const url = openUrl(a);

  const w = window.open(url, "_blank", "width=900,height=700");
  if (!w) {
    alert(MSG_POPUP_BLOCKED);
    return;
  }

  // `onReady` is a hoisted function declaration, so TS can't carry the
  // `if (!w)` narrowing above into its body (unlike the inline function
  // expressions used elsewhere in this file) — re-bind to a typed local.
  const win = /** @type {Window} */ (w);

  function onReady() {
    try {
      const s = win.document.createElement("style");
      s.textContent =
        "@media print { " +
        ".plotly-notifier, .modebar, .modebar-container { display: none !important; } " +
        ".jp-Toolbar, .jp-mod-toolbar { display: none !important; } " +
        "body { background: white !important; } " +
        "}";
      win.document.head.appendChild(s);
    } catch (_) { /* cross-origin guard — print as-is */ }
    win.focus();
    win.print();
  }

  w.addEventListener("load", onReady);
  if (w.document.readyState === "complete") onReady();
}

/** @param {any} a */
function printImage(a) {
  const imgUrl = fileUrl(a);
  const html = "<!DOCTYPE html><html><head><meta charset='UTF-8'>" +
    "<title>" + esc(a.title) + "</title>" +
    "<style>" + PRINT_STYLES + "</style></head><body>" +
    headerHtml(a) +
    '<div class="print-body"><img src="' + esc(imgUrl) +
    '" alt="' + esc(a.title) + '"></div></body></html>';
  openBlobAndPrint(html);
}

/** @param {any} a */
function printTimeseries(a) {
  const chartEl = document.querySelector("#ts-viewport [data-ts-chart] .js-plotly-plot");

  if (!chartEl || typeof Plotly === "undefined") {
    alert("Chart not yet rendered. Please wait for it to load, then try again.");
    return;
  }

  // Open the window synchronously during the click handler so it isn't
  // blocked as a popup.  Plotly.toImage() is async — by the time its
  // .then() runs we've left the user-gesture context.
  const w = window.open("about:blank", "_blank", "width=900,height=700");
  if (!w) {
    alert(MSG_POPUP_BLOCKED);
    return;
  }

  // Show loading message while capturing
  // Safe: writing static HTML to our own about:blank window.
  // This capture window is part of the always-light print flow (see the
  // PRINT_STYLES comment above) \u2014 the hardcoded gray is intentional.
  w.document.open();
  w.document.write(  // static content only
    "<!DOCTYPE html><html><head><style>body{display:flex;align-items:center;" +
    "justify-content:center;height:100vh;font-family:system-ui;color:#666;}" + // hygiene-allow-color
    "</style></head><body><p>Capturing chart\u2026</p></body></html>"
  );
  w.document.close();

  Plotly.toImage(chartEl, { format: "png", width: 1200, height: 600 })
    .then(function (/** @type {string} */ dataUrl) {
      const html = "<!DOCTYPE html><html><head><meta charset='UTF-8'>" +
        "<title>" + esc(a.title) + "</title>" +
        "<style>" + PRINT_STYLES + "</style></head><body>" +
        headerHtml(a) +
        '<div class="print-body"><img src="' + dataUrl +
        '" alt="' + esc(a.title) + ' chart"></div></body></html>';

      // Replace the loading content with the captured chart.
      populateWindow(w, html);
      w.focus();
      w.print();
    })
    .catch(function (/** @type {unknown} */ err) {
      console.error("[print.js] Plotly.toImage failed:", err);
      // eslint-disable-next-line no-empty -- intentional empty catch: closing an already-gone print window is best-effort
      try { w.close(); } catch (_) {}
      alert("Could not capture chart for printing.");
    });
}

// ---- Dispatch ----

/** @param {any} a */
export function printArtifact(a) {
  if (!a) return;

  if (isTimeseries(a)) { printTimeseries(a); return; }

  switch (a.artifact_type) {
    case "plot_html":
    case "table_html":
    case "dashboard_html":
    case "html":
    case "notebook":
    case "markdown":
    case "text":
    case "json":
      printIframe(a);
      break;
    case "plot_png":
    case "image":
      printImage(a);
      break;
    default:
      printIframe(a);
  }
}

// ---- Button injection (mirrors logbook.js pattern) ----

/** @param {any} a */
function createPrintBtn(a) {
  const btn = document.createElement("button");
  btn.className = "btn-action-icon print-action-btn";
  btn.title = "Print / Save as PDF";
  // PRINT_ICON is a hardcoded SVG constant — safe for innerHTML
  btn.innerHTML = PRINT_ICON;  // PRINT_ICON is a trusted module-level SVG constant (no user input) — safe innerHTML assignment
  btn.addEventListener("click", function (e) {
    e.stopPropagation();
    printArtifact(a);
  });
  return btn;
}

function injectPrintButton() {
  // Remove stale buttons from previous render
  document.querySelectorAll(".print-action-btn").forEach(function (b) { b.remove(); });

  const selected = getSelectedArtifact();
  if (!selected) return;

  const bar = document.querySelector("#preview-content .preview-header-actions");
  if (!bar) return;

  const btn = createPrintBtn(selected);
  // Insert before Delete — destructive action stays rightmost
  const deleteBtn = bar.querySelector(".btn-action-danger");
  if (deleteBtn) {
    bar.insertBefore(btn, deleteBtn);
  } else {
    bar.appendChild(btn);
  }
}

export { injectPrintButton };
