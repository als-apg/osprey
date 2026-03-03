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
 *     notebook, text, json): opens the source URL in a new window,
 *     injects print-cleanup styles, calls window.print().
 *
 *   - image / plot_png: wraps the image in minimal HTML with a title
 *     header, opens via Blob URL in a print window.
 *
 *   - markdown (rendered in DOM with KaTeX): clones the rendered
 *     .osprey-md-rendered node into a Blob-based print window that
 *     includes the KaTeX and highlight.js stylesheets.
 *
 *   - timeseries (Plotly.js in DOM): captures the chart via
 *     Plotly.toImage() as PNG, embeds in a print window.
 */
(function () {
  "use strict";

  // ---- SVG icon (printer) ----

  const PRINT_ICON =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<polyline points="6 9 6 2 18 2 18 9"/>' +
    '<path d="M6 18H4a2 2 0 01-2-2v-5a2 2 0 012-2h16a2 2 0 012 2v5a2 2 0 01-2 2h-2"/>' +
    '<rect x="6" y="14" width="12" height="8"/>' +
    "</svg>";

  // ---- Shared styles for self-built print windows ----

  const PRINT_STYLES =
    "body { margin: 0; padding: 24px; font-family: system-ui, -apple-system, sans-serif; " +
    "background: white; color: black; }\n" +
    ".print-header { margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #ccc; }\n" +
    ".print-header h1 { font-size: 18px; font-weight: 600; margin: 0 0 4px; }\n" +
    ".print-header .print-meta { font-size: 11px; color: #666; }\n" +
    ".print-body { width: 100%; }\n" +
    ".print-body img { max-width: 100%; height: auto; display: block; }\n" +
    "@media print { .print-header { page-break-after: avoid; } }\n";

  const MSG_POPUP_BLOCKED = "Print blocked \u2014 please allow pop-ups for this site and try again.";

  // ---- Helpers ----

  function esc(str) {
    if (!str) return "";
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function fmtTime(ts) {
    if (!ts) return "";
    return new Date(ts).toLocaleString();
  }

  function fileUrl(a) {
    // Prefer the gallery's canonical fileUrl if exposed, otherwise build
    // an absolute URL (absolute is required for Blob-origin documents).
    const gs = window._galleryState;
    if (gs && gs.fileUrl) return gs.fileUrl(a);
    return window.location.origin + "/files/" + a.id + "/" + encodeURIComponent(a.filename);
  }

  function headerHtml(a) {
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
   * Opens a new window from an HTML string using a Blob URL.
   * Waits for load, then triggers the browser print dialog.
   */
  function openBlobAndPrint(html) {
    const blob = new Blob([html], { type: "text/html" });
    const blobUrl = URL.createObjectURL(blob);
    const w = window.open(blobUrl, "_blank", "width=900,height=700");
    if (!w) {
      URL.revokeObjectURL(blobUrl);
      alert(MSG_POPUP_BLOCKED);
      return;
    }

    function onReady() {
      URL.revokeObjectURL(blobUrl);
      w.focus();
      w.print();
    }

    w.addEventListener("load", onReady);
    // Guard against the load event having already fired
    if (w.document.readyState === "complete") onReady();
  }

  // ---- Per-type print strategies ----

  function printIframe(a) {
    const url = a.artifact_type === "notebook"
      ? "/api/notebooks/" + a.id + "/rendered"
      : fileUrl(a);

    const w = window.open(url, "_blank", "width=900,height=700");
    if (!w) {
      alert(MSG_POPUP_BLOCKED);
      return;
    }

    function onReady() {
      try {
        const s = w.document.createElement("style");
        s.textContent =
          "@media print { " +
          ".plotly-notifier, .modebar, .modebar-container { display: none !important; } " +
          ".jp-Toolbar, .jp-mod-toolbar { display: none !important; } " +
          "body { background: white !important; } " +
          "}";
        w.document.head.appendChild(s);
      } catch (_) { /* cross-origin guard — print as-is */ }
      w.focus();
      w.print();
    }

    w.addEventListener("load", onReady);
    if (w.document.readyState === "complete") onReady();
  }

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

  function printMarkdown(a) {
    const mdEl = document.getElementById("md-viewport");
    const rendered = mdEl && mdEl.querySelector(".osprey-md-rendered");

    if (!rendered) {
      fetch(fileUrl(a))
        .then(function (r) { return r.text(); })
        .then(function (text) {
          openBlobAndPrint(
            "<!DOCTYPE html><html><head><meta charset='UTF-8'>" +
            "<title>" + esc(a.title) + "</title>" +
            "<style>" + PRINT_STYLES +
            " pre { white-space: pre-wrap; font-family: monospace; }</style></head><body>" +
            headerHtml(a) +
            '<div class="print-body"><pre>' + esc(text) + "</pre></div></body></html>"
          );
        })
        .catch(function (err) {
          console.error("[print.js] Failed to fetch markdown:", err);
          alert("Could not load markdown source for printing.");
        });
      return;
    }

    // Clone rendered HTML (includes KaTeX output as DOM elements)
    const contentHtml = rendered.outerHTML;

    // Find KaTeX and highlight.js stylesheets from the host page
    let extraLinks = "";
    document.querySelectorAll('link[rel="stylesheet"]').forEach(function (link) {
      if (link.href && (link.href.indexOf("katex") !== -1 || link.id === "hljs-theme")) {
        extraLinks += '<link rel="stylesheet" href="' + esc(link.href) + '">';
      }
    });

    const html = "<!DOCTYPE html><html><head><meta charset='UTF-8'>" +
      "<title>" + esc(a.title) + "</title>" +
      extraLinks +
      "<style>" + PRINT_STYLES +
      ".osprey-md-rendered { font-family: system-ui, sans-serif; font-size: 14px; " +
      "  line-height: 1.7; color: #111; }\n" +
      ".osprey-md-rendered h1,.osprey-md-rendered h2,.osprey-md-rendered h3 { margin-top: 1.2em; }\n" +
      ".osprey-md-rendered pre,.osprey-md-rendered code { background: #f5f5f5; border-radius: 3px; " +
      "  padding: 2px 4px; font-size: 12px; }\n" +
      ".osprey-md-rendered pre { padding: 12px; overflow-x: auto; }\n" +
      ".osprey-md-rendered table { border-collapse: collapse; width: 100%; }\n" +
      ".osprey-md-rendered th,.osprey-md-rendered td { border: 1px solid #ddd; padding: 6px 10px; }\n" +
      "</style></head><body>" +
      headerHtml(a) +
      '<div class="print-body">' + contentHtml + "</div></body></html>";

    openBlobAndPrint(html);
  }

  function printTimeseries(a) {
    const chartEl = document.querySelector("#ts-viewport [data-ts-chart] .js-plotly-plot");

    if (!chartEl || typeof Plotly === "undefined") {
      alert("Chart not yet rendered. Please wait for it to load, then try again.");
      return;
    }

    Plotly.toImage(chartEl, { format: "png", width: 1200, height: 600 })
      .then(function (dataUrl) {
        const html = "<!DOCTYPE html><html><head><meta charset='UTF-8'>" +
          "<title>" + esc(a.title) + "</title>" +
          "<style>" + PRINT_STYLES + "</style></head><body>" +
          headerHtml(a) +
          '<div class="print-body"><img src="' + dataUrl +
          '" alt="' + esc(a.title) + ' chart"></div></body></html>';
        openBlobAndPrint(html);
      })
      .catch(function (err) {
        console.error("[print.js] Plotly.toImage failed:", err);
        alert("Could not capture chart for printing.");
      });
  }

  // ---- Dispatch ----

  function printArtifact(a) {
    if (!a) return;

    const isTimeseries =
      (a.metadata && a.metadata.data_type === "timeseries") ||
      a.category === "archiver_data";

    if (isTimeseries) { printTimeseries(a); return; }

    switch (a.artifact_type) {
      case "plot_html":
      case "table_html":
      case "dashboard_html":
      case "html":
      case "notebook":
      case "text":
      case "json":
        printIframe(a);
        break;
      case "plot_png":
      case "image":
        printImage(a);
        break;
      case "markdown":
        printMarkdown(a);
        break;
      default:
        printIframe(a);
    }
  }

  // ---- Button injection (mirrors logbook.js pattern) ----

  function createPrintBtn(a) {
    const btn = document.createElement("button");
    btn.className = "btn-action-icon print-action-btn";
    btn.title = "Print / Save as PDF";
    // PRINT_ICON is a hardcoded SVG constant — safe for innerHTML
    btn.innerHTML = PRINT_ICON;  // eslint-disable-line no-unsanitized/property
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      printArtifact(a);
    });
    return btn;
  }

  function injectPrintButton() {
    const gs = window._galleryState;
    if (!gs) return;

    // Remove stale buttons from previous render
    document.querySelectorAll(".print-action-btn").forEach(function (b) { b.remove(); });

    const selected = gs.getSelectedArtifact && gs.getSelectedArtifact();
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

  window.injectPrintButton = injectPrintButton;
})();
