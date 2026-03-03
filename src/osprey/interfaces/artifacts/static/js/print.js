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
   * Opens a new window from an HTML string and triggers the print dialog.
   *
   * Uses about:blank instead of Blob URLs because Blob URLs get their own
   * origin, which (a) breaks relative image URLs embedded in the HTML and
   * (b) gets blocked by browsers when opened from an iframed page.
   * about:blank windows inherit the opener's origin, avoiding both issues.
   */
  function openBlobAndPrint(html) {
    const w = window.open("about:blank", "_blank", "width=900,height=700");
    if (!w) {
      alert(MSG_POPUP_BLOCKED);
      return;
    }

    const parsed = new DOMParser().parseFromString(html, "text/html");

    // Initialize the about:blank document with a minimal shell.
    // Safe: writing a hardcoded empty document to our own about:blank window.
    w.document.open();  // eslint-disable-line no-restricted-syntax
    w.document.write("<!DOCTYPE html><html><head></head><body></body></html>");  // static content only
    w.document.close();

    // Transfer parsed content into the live document via DOM adoption
    while (parsed.head.childNodes.length) {
      w.document.head.appendChild(w.document.adoptNode(parsed.head.childNodes[0]));
    }
    while (parsed.body.childNodes.length) {
      w.document.body.appendChild(w.document.adoptNode(parsed.body.childNodes[0]));
    }

    // Wait for images/stylesheets to load before printing
    w.addEventListener("load", function () { w.focus(); w.print(); });
    if (w.document.readyState === "complete") { w.focus(); w.print(); }
  }

  // ---- Per-type print strategies ----

  function printIframe(a) {
    let url;
    if (a.artifact_type === "notebook") {
      url = "/api/notebooks/" + a.id + "/rendered";
    } else if (a.artifact_type === "markdown") {
      url = "/api/markdown/" + a.id + "/rendered";
    } else {
      url = fileUrl(a);
    }

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
    w.document.open();  // eslint-disable-line no-restricted-syntax
    w.document.write(  // static content only
      "<!DOCTYPE html><html><head><style>body{display:flex;align-items:center;" +
      "justify-content:center;height:100vh;font-family:system-ui;color:#666;}" +
      "</style></head><body><p>Capturing chart\u2026</p></body></html>"
    );
    w.document.close();

    Plotly.toImage(chartEl, { format: "png", width: 1200, height: 600 })
      .then(function (dataUrl) {
        var html = "<!DOCTYPE html><html><head><meta charset='UTF-8'>" +
          "<title>" + esc(a.title) + "</title>" +
          "<style>" + PRINT_STYLES + "</style></head><body>" +
          headerHtml(a) +
          '<div class="print-body"><img src="' + dataUrl +
          '" alt="' + esc(a.title) + ' chart"></div></body></html>';

        var parsed = new DOMParser().parseFromString(html, "text/html");

        // Replace the loading content with the chart
        // Safe: writing a hardcoded empty document to our own window.
        w.document.open();  // eslint-disable-line no-restricted-syntax
        w.document.write("<!DOCTYPE html><html><head></head><body></body></html>");  // static content only
        w.document.close();
        while (parsed.head.childNodes.length) {
          w.document.head.appendChild(w.document.adoptNode(parsed.head.childNodes[0]));
        }
        while (parsed.body.childNodes.length) {
          w.document.body.appendChild(w.document.adoptNode(parsed.body.childNodes[0]));
        }
        w.focus();
        w.print();
      })
      .catch(function (err) {
        console.error("[print.js] Plotly.toImage failed:", err);
        try { w.close(); } catch (_) {}
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
      case "markdown":
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
