// @ts-check
/**
 * OSPREY Artifact Gallery — Print / Save as PDF
 *
 * Injects a print icon button into .preview-header-actions following
 * the same injection pattern as logbook.js. `printArtifact` is exported
 * too: the embedded viewer, which renders no preview header, runs it for
 * the hub's `print` action.
 *
 * Strategy: opens a dedicated print window for each artifact type so
 * the print output is clean (no gallery chrome) and WYSIWYG:
 *
 *   - inert types (text, json, PDF and the rest of C-OPAQUE's inert
 *     list): opens the file URL in a new same-origin window, injects
 *     print-cleanup styles, calls window.print().
 *
 *   - opaque types (plot_html, table_html, dashboard_html, html,
 *     notebook, markdown, and any other type served sandboxed): the
 *     popup has origin "null", so the gallery never touches its DOM.
 *     C-MSG handshake: wait for {type: "osprey-artifact-ready"} from
 *     that popup, answer {type: "osprey-artifact-print"}; the page
 *     prints itself. A page with no injected listener (raw SVG/XML)
 *     stays open in its tab with a note to use the browser's print.
 *
 *   - image / plot_png: wraps the image in minimal HTML with a title
 *     header, opens via about:blank window for printing.
 *
 *   - timeseries (Plotly.js in DOM): opens an about:blank window
 *     synchronously (to avoid popup blocking), captures chart via
 *     Plotly.toImage() as PNG, then populates the window and prints.
 */

import { escapeHtml as esc } from '/design-system/js/dom.js';
import { getSelectedArtifact, fileUrl } from './state.js';
import { openUrl, isOpaque, isTimeseries, isoToDate } from './types.js';
import { formatFacilityTime } from '/design-system/js/facility-time.js';

export { esc };

// ---- SVG icon (printer) ----

const PRINT_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
  '<polyline points="6 9 6 2 18 2 18 9"/>' +
  '<path d="M6 18H4a2 2 0 01-2-2v-5a2 2 0 012-2h16a2 2 0 012 2v5a2 2 0 01-2 2h-2"/>' +
  '<rect x="6" y="14" width="12" height="8"/>' +
  '</svg>';

// ---- Shared styles for self-built print windows ----
//
// Printed output stays light-on-white regardless of the gallery's active
// theme (matches the fleet-wide "@media print never goes dark" rule) — the
// couple of hardcoded grays below are intentional, not migration debt.
// hygiene-allow-color-start

const PRINT_STYLES =
  'body { margin: 0; padding: 24px; font-family: system-ui, -apple-system, sans-serif; ' +
  'background: white; color: black; }\n' +
  '.print-header { margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #ccc; }\n' +
  '.print-header h1 { font-size: 18px; font-weight: 600; margin: 0 0 4px; }\n' +
  '.print-header .print-meta { font-size: 11px; color: #666; }\n' +
  '.print-body { width: 100%; }\n' +
  '.print-body img { max-width: 100%; height: auto; display: block; }\n' +
  '@media print { .print-header { page-break-after: avoid; } }\n';

// hygiene-allow-color-end

const MSG_POPUP_BLOCKED = 'Print blocked \u2014 please allow pop-ups for this site and try again.';
const MSG_PRINT_FALLBACK =
  "This artifact can't start printing by itself. It is open in its own tab \u2014 " +
  "use the browser's print there.";

/** How long an opaque print window has to announce itself (C-MSG). */
export const READY_TIMEOUT_MS = 3000;

// ---- Helpers ----

/**
 * Full timestamp for the print header, on the facility clock. It always names
 * its zone, because a printed page leaves the screen that could say which
 * clock it is. It keeps seconds (a print/PDF header wants precision, unlike
 * the gallery's compact `formatFullTime`), and shares types.js's `isoToDate`
 * guard so a malformed/non-ISO timestamp renders "" rather than a fabricated
 * year-2000/epoch date.
 * @param {any} ts
 */
export function fmtTime(ts) {
  const d = isoToDate(ts);
  if (!d) return '';
  return formatFacilityTime(d, {
    year: 'numeric',
    month: 'numeric',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
    timeZoneName: 'short',
  });
}

/** @param {any} a */
export function headerHtml(a) {
  let meta = '';
  if (a.timestamp) meta += '<span>' + esc(fmtTime(a.timestamp)) + '</span>';
  if (a.tool_source) meta += ' &middot; <span>Source: ' + esc(a.tool_source) + '</span>';
  if (a.filename) meta += ' &middot; <span>' + esc(a.filename) + '</span>';
  return (
    '<div class="print-header">' +
    '<h1>' +
    esc(a.title || a.filename || 'Artifact') +
    '</h1>' +
    (meta ? '<div class="print-meta">' + meta + '</div>' : '') +
    '</div>'
  );
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
  const parsed = new DOMParser().parseFromString(html, 'text/html');

  // Initialize the target document with a minimal shell.
  // Safe: writing a hardcoded empty document to our own window.
  w.document.open();
  w.document.write('<!DOCTYPE html><html><head></head><body></body></html>'); // static content only
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
  const w = window.open('about:blank', '_blank', 'width=900,height=700');
  if (!w) {
    alert(MSG_POPUP_BLOCKED);
    return;
  }

  populateWindow(w, html);

  // Wait for images/stylesheets to load before printing
  w.addEventListener('load', function () {
    w.focus();
    w.print();
  });
  if (w.document.readyState === 'complete') {
    w.focus();
    w.print();
  }
}

// ---- Per-type print strategies ----

/**
 * Print an inert-type artifact by opening it same-origin and printing it
 * directly (C-MSG: inert types keep this path).
 * @param {any} a
 */
function printIframe(a) {
  // openUrl() owns the type -> URL mapping (rendered endpoints for
  // markdown/notebook, raw file URL otherwise). The theme is pinned to
  // light, not the gallery's: printed output stays light-on-white (see
  // PRINT_STYLES), and the `@media print` body repaint below cannot undo
  // colors a themed Plotly page has already written into its SVG.
  const url = openUrl(a, 'light');

  const w = window.open(url, '_blank', 'width=900,height=700');
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
      const s = win.document.createElement('style');
      s.textContent =
        '@media print { ' +
        '.plotly-notifier, .modebar, .modebar-container { display: none !important; } ' +
        '.jp-Toolbar, .jp-mod-toolbar { display: none !important; } ' +
        'body { background: white !important; } ' +
        '}';
      win.document.head.appendChild(s);
    } catch {
      /* cross-origin guard — print as-is */
    }
    win.focus();
    win.print();
  }

  w.addEventListener('load', onReady);
  let loaded = false;
  try {
    loaded = w.document.readyState === 'complete';
  } catch {
    /* not same-origin — wait for load */
  }
  if (loaded) onReady();
}

/**
 * Print an opaque-type artifact through the C-MSG handshake. The popup's
 * origin is "null", so only its WindowProxy is used — never its document:
 * wait for its ready message (matched by `event.source`), then ask it to
 * print. No ready within READY_TIMEOUT_MS (a raw SVG/XML file carries no
 * listener) leaves the tab open and tells the operator to print from there.
 * @param {any} a
 */
function printOpaque(a) {
  // Light theme for the same reason as printIframe.
  const opened = window.open(openUrl(a, 'light'), '_blank', 'width=900,height=700');
  if (!opened) {
    alert(MSG_POPUP_BLOCKED);
    return;
  }
  // Typed local: the narrowing above does not reach the hoisted handler.
  const w = /** @type {Window} */ (opened);

  /** @param {MessageEvent} event */
  function onMessage(event) {
    if (event.source !== w) return;
    const data = event.data;
    if (!data || data.type !== 'osprey-artifact-ready') return;
    window.removeEventListener('message', onMessage);
    clearTimeout(timer);
    w.focus();
    // Intentional "*": the popup's origin is opaque ("null") and cannot be
    // named; the target is pinned by the WindowProxy and the message is bare.
    w.postMessage({ type: 'osprey-artifact-print' }, '*');
  }
  const timer = setTimeout(function () {
    window.removeEventListener('message', onMessage);
    if (!w.closed) alert(MSG_PRINT_FALLBACK);
  }, READY_TIMEOUT_MS);
  window.addEventListener('message', onMessage);
}

/** @param {any} a */
function printImage(a) {
  const imgUrl = fileUrl(a);
  const html =
    "<!DOCTYPE html><html><head><meta charset='UTF-8'>" +
    '<title>' +
    esc(a.title) +
    '</title>' +
    '<style>' +
    PRINT_STYLES +
    '</style></head><body>' +
    headerHtml(a) +
    '<div class="print-body"><img src="' +
    esc(imgUrl) +
    '" alt="' +
    esc(a.title) +
    '"></div></body></html>';
  openBlobAndPrint(html);
}

/** @param {any} a */
function printTimeseries(a) {
  // Scoped to the preview pane, the one place the gallery mounts a chart: the
  // artifact printed is the one shown there (standalone, the Print button sits
  // in that pane's header; embedded, the hub's print action names the
  // artifact its viewer shows).
  const chartEl = document.querySelector(
    '#preview-content .ts-viewport-container [data-ts-chart] .js-plotly-plot'
  );

  if (!chartEl || typeof Plotly === 'undefined') {
    alert('Chart not yet rendered. Please wait for it to load, then try again.');
    return;
  }

  // Open the window synchronously during the click handler so it isn't
  // blocked as a popup.  Plotly.toImage() is async — by the time its
  // .then() runs we've left the user-gesture context.
  const w = window.open('about:blank', '_blank', 'width=900,height=700');
  if (!w) {
    alert(MSG_POPUP_BLOCKED);
    return;
  }

  // Show loading message while capturing
  // Safe: writing static HTML to our own about:blank window.
  // This capture window is part of the always-light print flow (see the
  // PRINT_STYLES comment above) \u2014 the hardcoded gray is intentional.
  w.document.open();
  w.document.write(
    // static content only
    '<!DOCTYPE html><html><head><style>body{display:flex;align-items:center;' +
      'justify-content:center;height:100vh;font-family:system-ui;color:#666;}' + // hygiene-allow-color
      '</style></head><body><p>Capturing chart\u2026</p></body></html>'
  );
  w.document.close();

  Plotly.toImage(chartEl, { format: 'png', width: 1200, height: 600 })
    .then(function (/** @type {string} */ dataUrl) {
      const html =
        "<!DOCTYPE html><html><head><meta charset='UTF-8'>" +
        '<title>' +
        esc(a.title) +
        '</title>' +
        '<style>' +
        PRINT_STYLES +
        '</style></head><body>' +
        headerHtml(a) +
        '<div class="print-body"><img src="' +
        dataUrl +
        '" alt="' +
        esc(a.title) +
        ' chart"></div></body></html>';

      // Replace the loading content with the captured chart.
      populateWindow(w, html);
      w.focus();
      w.print();
    })
    .catch(function (/** @type {unknown} */ err) {
      console.error('[print.js] Plotly.toImage failed:', err);
      try {
        w.close();
      } catch {
        // Closing an already-gone print window is best-effort.
      }
      alert('Could not capture chart for printing.');
    });
}

// ---- Dispatch ----

/** @param {any} a */
export function printArtifact(a) {
  if (!a) return;

  if (isTimeseries(a)) {
    printTimeseries(a);
    return;
  }

  switch (a.artifact_type) {
    case 'plot_png':
    case 'image':
      printImage(a);
      break;
    default:
      // Same rule the server serves by (C-OPAQUE): inert types print
      // directly, everything else through the handshake.
      if (isOpaque(a)) printOpaque(a);
      else printIframe(a);
  }
}

// ---- Button injection (mirrors logbook.js pattern) ----

/** @param {any} a */
function createPrintBtn(a) {
  const btn = document.createElement('button');
  btn.className = 'btn-action-icon print-action-btn';
  btn.title = 'Print / Save as PDF';
  // PRINT_ICON is a hardcoded SVG constant — safe for innerHTML
  btn.innerHTML = PRINT_ICON; // PRINT_ICON is a trusted module-level SVG constant (no user input) — safe innerHTML assignment
  btn.addEventListener('click', function (e) {
    e.stopPropagation();
    printArtifact(a);
  });
  return btn;
}

function injectPrintButton() {
  // Remove stale buttons from previous render
  document.querySelectorAll('.print-action-btn').forEach(function (b) {
    b.remove();
  });

  const selected = getSelectedArtifact();
  if (!selected) return;

  const bar = document.querySelector('#preview-content .preview-header-actions');
  if (!bar) return;

  const btn = createPrintBtn(selected);
  // Insert before Delete — destructive action stays rightmost
  const deleteBtn = bar.querySelector('.btn-action-danger');
  if (deleteBtn) {
    bar.insertBefore(btn, deleteBtn);
  } else {
    bar.appendChild(btn);
  }
}

export { injectPrintButton };
