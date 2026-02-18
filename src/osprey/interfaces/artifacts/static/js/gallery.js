/* OSPREY Artifact Gallery — client-side logic
 *
 * Three-domain navigation: Artifacts, Context, and Memory.
 * Each domain has Focus + Browse sub-views.
 * Stats is a shared global view.
 *
 * URL schema:
 *   #artifacts/focus    (default)
 *   #artifacts/browse
 *   #context/focus
 *   #context/browse
 *   #memory/focus
 *   #memory/browse
 *   #stats
 */

(function () {
  "use strict";

  // Embedded mode — hide logo when loaded inside web terminal iframe
  const params = new URLSearchParams(window.location.search);
  if (params.get('embedded') === 'true') {
    document.body.classList.add('embedded');
  }

  // ---- DOM refs ----

  const focusContainer = document.getElementById("focus-container");
  const focusEmpty = document.getElementById("focus-empty");
  const healthIndicator = document.getElementById("health-indicator");
  const refreshBtn = document.getElementById("refresh-btn");
  const headerCount = document.getElementById("header-count");
  const searchInput = document.getElementById("search");
  const sidebarBody = document.getElementById("sidebar-body");
  const previewEmpty = document.getElementById("preview-empty");
  const previewContent = document.getElementById("preview-content");
  const resizeHandle = document.getElementById("resize-handle");
  const sidebar = document.getElementById("browse-sidebar");

  // Context DOM refs
  const headerContextCount = document.getElementById("header-context-count");
  const ctxSearchInput = document.getElementById("ctx-search");
  const ctxSidebarBody = document.getElementById("ctx-sidebar-body");
  const ctxPreviewEmpty = document.getElementById("ctx-preview-empty");
  const ctxPreviewContent = document.getElementById("ctx-preview-content");
  const ctxResizeHandle = document.getElementById("ctx-resize-handle");
  const ctxSidebar = document.getElementById("ctx-sidebar");

  // Context Focus DOM refs
  const ctxFocusContainer = document.getElementById("ctx-focus-container");
  const ctxFocusEmpty = document.getElementById("ctx-focus-empty");

  let artifacts = [];
  let focusedArtifact = null;
  let selectedArtifact = null;   // currently previewed in browse
  let browseMode = "tree";       // "tree" | "activity"
  let sidebarLayout = "list";    // "list" | "gallery"
  let sessionStartTime = null;   // for "new" badges

  // Context state
  let contextEntries = [];
  let selectedContext = null;     // currently previewed in context browse
  let focusedContext = null;      // context entry in focus view
  let ctxBrowseMode = "tree";    // "tree" | "activity"
  let ctxSidebarLayout = "list"; // "list" | "gallery"

  // Memory DOM refs
  const headerMemoryCount = document.getElementById("header-memory-count");
  const memSearchInput = document.getElementById("mem-search");
  const memSidebarBody = document.getElementById("mem-sidebar-body");
  const memPreviewEmpty = document.getElementById("mem-preview-empty");
  const memPreviewContent = document.getElementById("mem-preview-content");
  const memResizeHandle = document.getElementById("mem-resize-handle");
  const memSidebarEl = document.getElementById("mem-sidebar");
  const memFocusContainer = document.getElementById("mem-focus-container");
  const memFocusEmpty = document.getElementById("mem-focus-empty");

  // Memory state
  let memoryEntries = [];
  let selectedMemory = null;
  let focusedMemory = null;
  let memBrowseMode = "tree";
  let memSidebarLayout = "list";

  // Domain-aware routing state
  let activeDomain = "artifacts";   // "artifacts" | "context" | "memory"
  let activeSubview = "focus";      // "focus" | "browse" | "stats"

  // Per-domain default and last-used subview tracking
  const domainDefaultSubview = {
    artifacts: "focus",
    context: "browse",
    memory: "browse",
  };
  let domainLastSubview = {
    artifacts: "focus",
    context: "browse",
    memory: "browse",
  };

  // ---- Resize Handle Factory ----

  function initSplitPaneResize(handle, sidebarEl) {
    let startX = 0;
    let startWidth = 0;

    function onMouseDown(e) {
      e.preventDefault();
      startX = e.clientX;
      startWidth = sidebarEl.getBoundingClientRect().width;
      handle.classList.add("dragging");
      document.body.classList.add("resizing");
      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
    }

    function onMouseMove(e) {
      const delta = e.clientX - startX;
      const newWidth = Math.max(48, Math.min(startWidth + delta, window.innerWidth - 48));
      sidebarEl.style.width = newWidth + "px";
    }

    function onMouseUp() {
      handle.classList.remove("dragging");
      document.body.classList.remove("resizing");
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
    }

    handle.addEventListener("mousedown", onMouseDown);

    // Touch support
    handle.addEventListener("touchstart", (e) => {
      const touch = e.touches[0];
      startX = touch.clientX;
      startWidth = sidebarEl.getBoundingClientRect().width;
      handle.classList.add("dragging");
      document.body.classList.add("resizing");

      function onTouchMove(e2) {
        const t = e2.touches[0];
        const delta = t.clientX - startX;
        const newWidth = Math.max(48, Math.min(startWidth + delta, window.innerWidth - 48));
        sidebarEl.style.width = newWidth + "px";
      }

      function onTouchEnd() {
        handle.classList.remove("dragging");
        document.body.classList.remove("resizing");
        document.removeEventListener("touchmove", onTouchMove);
        document.removeEventListener("touchend", onTouchEnd);
      }

      document.addEventListener("touchmove", onTouchMove, { passive: true });
      document.addEventListener("touchend", onTouchEnd);
    }, { passive: true });
  }

  // ---- View Routing (domain-aware) ----

  function initRouter() {
    // Domain buttons — click to activate the domain (recall last subview)
    document.querySelectorAll(".domain-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const domain = btn.dataset.domain;
        const subview = domainLastSubview[domain] || domainDefaultSubview[domain];
        navigateTo(`${domain}/${subview}`);
      });
    });

    // View-mode buttons (Focus / Browse) inside each domain segment
    document.querySelectorAll(".view-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation(); // don't bubble to domain-btn
        const segment = btn.closest(".domain-segment");
        const domain = segment.dataset.domain;
        navigateTo(`${domain}/${btn.dataset.subview}`);
      });
    });

    // Stats link (standalone)
    const statsLink = document.getElementById("stats-link");
    if (statsLink) {
      statsLink.addEventListener("click", (e) => {
        e.preventDefault();
        navigateTo("stats");
      });
    }

    // Hash change
    window.addEventListener("hashchange", () => {
      const hash = window.location.hash.replace("#", "") || "artifacts/focus";
      navigateTo(hash, true);
    });

    // Initial route
    const hash = window.location.hash.replace("#", "") || "artifacts/focus";
    navigateTo(hash);
  }

  function navigateTo(route, fromHashChange) {
    let domain, subview;

    if (route === "stats") {
      domain = activeDomain;
      subview = "stats";
    } else if (route.includes("/")) {
      [domain, subview] = route.split("/");
    } else {
      // Legacy/fallback: bare "focus" or "browse" → use active domain
      domain = activeDomain;
      subview = route;
    }

    // Validate
    if (!["artifacts", "context", "memory"].includes(domain)) domain = "artifacts";
    if (!["focus", "browse", "stats"].includes(subview)) subview = "focus";

    activeDomain = domain;
    activeSubview = subview;

    // Track last-used subview per domain (skip stats — it's shared)
    if (subview !== "stats") {
      domainLastSubview[domain] = subview;
    }

    // Update domain segments (expand active, collapse others)
    document.querySelectorAll(".domain-segment").forEach((seg) =>
      seg.classList.toggle("active", seg.dataset.domain === domain)
    );
    document.querySelector(".app-container").dataset.domain = domain;

    // Update view-mode buttons inside the active segment
    document.querySelectorAll(".view-btn").forEach((b) => b.classList.remove("active"));
    if (subview !== "stats") {
      const activeSegment = document.querySelector(`.domain-segment[data-domain="${domain}"]`);
      if (activeSegment) {
        const viewBtn = activeSegment.querySelector(`.view-btn[data-subview="${subview}"]`);
        if (viewBtn) viewBtn.classList.add("active");
      }
    }

    // Update stats link
    const statsLink = document.getElementById("stats-link");
    if (statsLink) statsLink.classList.toggle("active", subview === "stats");

    // Show target view
    const viewId = subview === "stats" ? "view-stats" : `view-${domain}-${subview}`;
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    const view = document.getElementById(viewId);
    if (view) view.classList.add("active");

    // Update URL hash
    if (!fromHashChange) {
      window.location.hash = subview === "stats" ? "stats" : `${domain}/${subview}`;
    }

    // Trigger data fetches
    if (domain === "artifacts" && subview === "focus") fetchFocus();
    if (domain === "artifacts" && subview === "browse") fetchArtifacts();
    if (domain === "context" && subview === "focus") fetchContextFocus();
    if (domain === "context" && subview === "browse") fetchContextEntries();
    if (domain === "memory" && subview === "focus") fetchMemoryFocus();
    if (domain === "memory" && subview === "browse") fetchMemoryEntries();
    if (subview === "stats") { fetchArtifacts(); fetchContextEntries(); fetchMemoryEntries(); renderStats(); }
  }

  function currentRoute() {
    return { domain: activeDomain, subview: activeSubview };
  }

  // ---- API ----

  async function fetchArtifacts() {
    try {
      const resp = await fetch("/api/artifacts");
      const data = await resp.json();
      artifacts = data.artifacts || [];
      if (!sessionStartTime && artifacts.length > 0) {
        sessionStartTime = new Date().toISOString();
      }
      updateHealth(true);
      updateHeaderCount();
      renderSidebar();
    } catch (err) {
      console.error("Failed to fetch artifacts:", err);
      updateHealth(false);
    }
  }

  async function fetchFocus() {
    try {
      const resp = await fetch("/api/focus");
      const data = await resp.json();
      focusedArtifact = data.artifact;
      updateHealth(true);
      renderFocus();
    } catch (err) {
      console.error("Failed to fetch focus:", err);
      updateHealth(false);
    }

    try {
      const resp = await fetch("/api/artifacts");
      const data = await resp.json();
      artifacts = data.artifacts || [];
      updateHeaderCount();
    } catch {
      // non-critical
    }
  }

  function updateHealth(connected) {
    const dot = healthIndicator.querySelector(".status-dot");
    const label = healthIndicator.querySelector(".status-label");
    if (connected) {
      dot.className = "status-dot healthy";
      label.textContent = "Connected";
    } else {
      dot.className = "status-dot error";
      label.textContent = "Disconnected";
    }
  }

  function updateHeaderCount() {
    headerCount.textContent = artifacts.length;
  }

  // ---- Rendering Helpers ----

  function fileUrl(a) {
    return `/files/${a.id}/${encodeURIComponent(a.filename)}`;
  }

  function typeBadge(type) {
    const labels = {
      plot_html: "Plotly",
      plot_png: "Matplotlib",
      table_html: "Table",
      html: "HTML",
      markdown: "Markdown",
      json: "JSON",
      image: "Image",
      text: "Text",
      file: "File",
      notebook: "Notebook",
    };
    return labels[type] || type;
  }

  function typeIcon(type) {
    const icons = {
      plot_html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 3v18h18"/><path d="M7 16l4-8 4 4 4-6"/></svg>',
      plot_png: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>',
      table_html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18"/></svg>',
      html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
      markdown: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="M6 8v8l3-3 3 3V8M17 12h-4m4 0l-2-2m2 2l-2 2"/></svg>',
      json: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M8 3H7a2 2 0 00-2 2v5a2 2 0 01-2 2 2 2 0 012 2v5a2 2 0 002 2h1M16 3h1a2 2 0 012 2v5a2 2 0 002 2 2 2 0 00-2 2v5a2 2 0 01-2 2h-1"/></svg>',
      image: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>',
      text: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>',
      file: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6"/></svg>',
      notebook: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 4h16v16H4z"/><path d="M8 2v4M8 18v4M4 8h16M12 12h4M12 15h3"/></svg>',
    };
    return icons[type] || icons.file;
  }

  const chevronSvg = '<svg class="tree-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>';

  // ---- Tool Helpers (context entries) ----

  function toolLabel(tool) {
    const labels = {
      channel_read: "Channel Read",
      channel_write: "Channel Write",
      archiver_read: "Archiver Read",
      python_execute: "Python Execute",
      channel_find: "Channel Find",
      memory_save: "Memory Save",
      memory_recall: "Memory Recall",
      ariel_search: "ARIEL Search",
      screen_capture: "Screen Capture",
    };
    return labels[tool] || tool;
  }

  function toolIcon(tool) {
    const icons = {
      channel_read: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 12s2-4 5-4 5 8 10 8 5-4 5-4"/></svg>',
      channel_write: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M17 3a2.83 2.83 0 114 4L7.5 20.5 2 22l1.5-5.5L17 3z"/></svg>',
      archiver_read: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 8v13H3V8M1 3h22v5H1zM10 12h4"/></svg>',
      python_execute: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
      channel_find: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>',
      memory_save: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>',
      memory_recall: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
      ariel_search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg>',
      screen_capture: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>',
    };
    return icons[tool] || '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>';
  }

  function dataTypeLabel(dt) {
    const labels = {
      timeseries: "Timeseries",
      channel_values: "Channel Values",
      channel_list: "Channel List",
      code_result: "Code Result",
      memory: "Memory",
      search_results: "Search Results",
      screenshot: "Screenshot",
    };
    return labels[dt] || dt;
  }

  function renderJsonHtml(obj, indent) {
    indent = indent || 0;
    const pad = "  ".repeat(indent);
    const padInner = "  ".repeat(indent + 1);

    if (obj === null) return '<span class="json-null">null</span>';
    if (typeof obj === "boolean") return `<span class="json-boolean">${obj}</span>`;
    if (typeof obj === "number") return `<span class="json-number">${obj}</span>`;
    if (typeof obj === "string") return `<span class="json-string">"${escapeHtml(obj)}"</span>`;

    if (Array.isArray(obj)) {
      if (obj.length === 0) return '<span class="json-bracket">[]</span>';
      const items = obj.map((v) => padInner + renderJsonHtml(v, indent + 1)).join('<span class="json-comma">,</span>\n');
      return '<span class="json-bracket">[</span>\n' + items + '\n' + pad + '<span class="json-bracket">]</span>';
    }

    if (typeof obj === "object") {
      const keys = Object.keys(obj);
      if (keys.length === 0) return '<span class="json-bracket">{}</span>';
      const entries = keys.map((k) =>
        padInner + '<span class="json-key">"' + escapeHtml(k) + '"</span>: ' + renderJsonHtml(obj[k], indent + 1)
      ).join('<span class="json-comma">,</span>\n');
      return '<span class="json-bracket">{</span>\n' + entries + '\n' + pad + '<span class="json-bracket">}</span>';
    }

    return escapeHtml(String(obj));
  }

  function formatSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(1) + " MB";
  }

  function formatTime(iso) {
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } catch {
      return iso;
    }
  }

  function formatDate(iso) {
    try {
      const d = new Date(iso);
      const today = new Date();
      const yesterday = new Date(today);
      yesterday.setDate(yesterday.getDate() - 1);

      if (d.toDateString() === today.toDateString()) return "Today";
      if (d.toDateString() === yesterday.toDateString()) return "Yesterday";
      return d.toLocaleDateString([], { month: "short", day: "numeric" });
    } catch {
      return "Unknown";
    }
  }

  function formatFullTime(iso) {
    try {
      return new Date(iso).toLocaleString();
    } catch {
      return iso;
    }
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function isNewThisSession(a) {
    if (!sessionStartTime) return false;
    return a.timestamp >= sessionStartTime;
  }

  function thumbnailHtml(a) {
    const url = fileUrl(a);
    switch (a.artifact_type) {
      case "plot_png":
      case "image":
        return `<img src="${url}" alt="" loading="lazy" />`;
      case "plot_html":
      case "table_html":
      case "html":
        return `<iframe src="${url}" sandbox="allow-scripts allow-same-origin" loading="lazy" tabindex="-1"></iframe>`;
      case "notebook":
        return `<iframe src="/api/notebooks/${a.id}/rendered" sandbox="allow-scripts allow-same-origin" loading="lazy" tabindex="-1"></iframe>`;
      default:
        return `<div class="thumb-placeholder">${typeIcon(a.artifact_type)}<span>${typeBadge(a.artifact_type)}</span></div>`;
    }
  }

  // ---- Artifact Focus View ----

  function renderFocus() {
    if (!focusedArtifact) {
      focusContainer.innerHTML = "";
      focusContainer.classList.add("hidden");
      focusEmpty.classList.remove("hidden");
      return;
    }

    focusEmpty.classList.add("hidden");
    focusContainer.classList.remove("hidden");

    const a = focusedArtifact;
    const url = fileUrl(a);

    let viewportContent = "";
    switch (a.artifact_type) {
      case "plot_html":
      case "table_html":
      case "html":
        viewportContent = `<iframe src="${url}" class="focus-iframe" sandbox="allow-scripts allow-same-origin"></iframe>`;
        break;
      case "notebook":
        viewportContent = `<iframe src="" class="focus-iframe notebook-interactive" data-notebook-id="${a.id}" sandbox="allow-scripts allow-same-origin allow-popups allow-forms allow-modals allow-downloads"></iframe>`;
        break;
      case "plot_png":
      case "image":
        viewportContent = `<img src="${url}" alt="${escapeHtml(a.title)}" class="focus-image" />`;
        break;
      case "markdown":
      case "text":
      case "json":
        viewportContent = `<iframe src="${url}" class="focus-iframe-dark"></iframe>`;
        break;
      default:
        viewportContent = `<div style="padding: var(--space-8); text-align: center;">
          <p>Download: <a href="${url}" target="_blank" class="download-link">${escapeHtml(a.filename)}</a></p>
        </div>`;
    }

    const sorted = [...artifacts].reverse();
    const idx = sorted.findIndex((x) => x.id === a.id);
    const hasPrev = idx > 0;
    const hasNext = idx >= 0 && idx < sorted.length - 1;

    focusContainer.innerHTML = `
      <div class="focus-title-row">
        <span class="focus-led"></span>
        <h2>${escapeHtml(a.title)}</h2>
        <span class="badge badge-${a.artifact_type}">${typeBadge(a.artifact_type)}</span>
      </div>
      ${a.description ? `<p class="focus-desc">${escapeHtml(a.description)}</p>` : ""}
      <div class="focus-viewport">
        ${viewportContent}
      </div>
      <div class="focus-footer">
        <div class="focus-nav">
          <button class="focus-nav-btn" id="focus-prev" ${hasPrev ? "" : "disabled"}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M15 18l-6-6 6-6"/>
            </svg>
            Prev
          </button>
          <button class="focus-nav-btn" id="focus-next" ${hasNext ? "" : "disabled"}>
            Next
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M9 18l6-6-6-6"/>
            </svg>
          </button>
        </div>
        <div class="focus-meta">
          <span class="meta-item">
            <span class="meta-label">Type</span>
            <span class="meta-value">${typeBadge(a.artifact_type)}</span>
          </span>
          <span class="meta-item">
            <span class="meta-label">Size</span>
            <span class="meta-value">${formatSize(a.size_bytes)}</span>
          </span>
          <span class="meta-item">
            <span class="meta-label">Created</span>
            <span class="meta-value">${formatFullTime(a.timestamp)}</span>
          </span>
        </div>
        <a href="${url}" target="_blank" class="download-link">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3"/>
          </svg>
          Open
        </a>
        <button class="btn btn-danger btn-sm" id="focus-delete" title="Delete artifact">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/>
          </svg>
          Delete
        </button>
      </div>
    `;

    const prevBtn = document.getElementById("focus-prev");
    const nextBtn = document.getElementById("focus-next");
    if (prevBtn && hasPrev) {
      prevBtn.addEventListener("click", () => {
        focusedArtifact = sorted[idx - 1];
        renderFocus();
      });
    }
    if (nextBtn && hasNext) {
      nextBtn.addEventListener("click", () => {
        focusedArtifact = sorted[idx + 1];
        renderFocus();
      });
    }
    document.getElementById("focus-delete")?.addEventListener("click", () => {
      if (!confirm(`Delete "${a.title}"? This cannot be undone.`)) return;
      fetch(`/api/artifacts/${a.id}`, { method: "DELETE" })
        .then(r => { if (!r.ok) throw new Error(); return r.json(); })
        .then(() => {
          artifacts = artifacts.filter(x => x.id !== a.id);
          focusedArtifact = null;
          updateHeaderCount();
          renderSidebar();
          renderFocus();
        })
        .catch(err => console.error("Delete failed:", err));
    });

    loadInteractiveNotebooks();
  }

  function loadInteractiveNotebooks() {
    document.querySelectorAll('.notebook-interactive[data-notebook-id]').forEach(iframe => {
      const id = iframe.dataset.notebookId;
      if (iframe.src && iframe.src !== '' && iframe.src !== window.location.href) return;
      fetch(`/api/notebooks/${id}/interactive`)
        .then(r => r.json())
        .then(data => { iframe.src = data.jupyter_url; })
        .catch(() => {
          // Fallback to static rendering if Jupyter container unavailable
          iframe.src = `/api/notebooks/${id}/rendered`;
        });
    });
  }

  function focusNavigate(delta) {
    if (!focusedArtifact || artifacts.length === 0) return;
    const sorted = [...artifacts].reverse();
    const idx = sorted.findIndex((x) => x.id === focusedArtifact.id);
    const newIdx = idx + delta;
    if (newIdx >= 0 && newIdx < sorted.length) {
      focusedArtifact = sorted[newIdx];
      renderFocus();
    }
  }

  // ---- Context Focus View ----

  async function fetchContextFocus() {
    try {
      const resp = await fetch("/api/context/focus");
      const data = await resp.json();
      focusedContext = data.entry;
      updateHealth(true);
      renderContextFocus();
    } catch (err) {
      console.error("Failed to fetch context focus:", err);
      updateHealth(false);
    }

    // Also fetch entries for prev/next + header count
    try {
      const resp = await fetch("/api/context");
      const data = await resp.json();
      contextEntries = data.entries || [];
      updateHeaderContextCount();
    } catch { /* non-critical */ }
  }

  async function renderContextFocus() {
    if (!focusedContext) {
      ctxFocusContainer.innerHTML = "";
      ctxFocusContainer.classList.add("hidden");
      ctxFocusEmpty.classList.remove("hidden");
      return;
    }
    ctxFocusEmpty.classList.add("hidden");
    ctxFocusContainer.classList.remove("hidden");

    const e = focusedContext;

    // Build summary and access_details card content
    const summaryHtml = e.summary
      ? Object.entries(e.summary).map(([k, v]) =>
          `<div><span style="color:var(--text-muted)">${escapeHtml(k)}:</span> ${escapeHtml(String(v))}</div>`
        ).join("")
      : "<div>No summary</div>";

    const accessHtml = e.access_details
      ? Object.entries(e.access_details).map(([k, v]) =>
          `<div><span style="color:var(--text-muted)">${escapeHtml(k)}:</span> ${escapeHtml(String(v))}</div>`
        ).join("")
      : "<div>No access details</div>";

    // Prev/next navigation
    const sorted = [...contextEntries].reverse();
    const idx = sorted.findIndex((x) => x.id === e.id);
    const hasPrev = idx > 0;
    const hasNext = idx >= 0 && idx < sorted.length - 1;

    ctxFocusContainer.innerHTML = `
      <div class="focus-title-row">
        <span class="focus-led amber"></span>
        <h2>${escapeHtml(e.description)}</h2>
        <span class="badge badge-${e.tool}">${toolLabel(e.tool)}</span>
      </div>
      <div class="focus-json-viewport">
        <div class="context-info-cards">
          <div class="context-info-card">
            <div class="context-info-card-title">Summary</div>
            <div class="context-info-card-body">${summaryHtml}</div>
          </div>
          <div class="context-info-card">
            <div class="context-info-card-title">Access Details</div>
            <div class="context-info-card-body">${accessHtml}</div>
          </div>
        </div>
        <div class="json-viewer" id="ctx-focus-json-viewer">
          <span class="text-muted">Loading data...</span>
        </div>
      </div>
      <div class="focus-footer">
        <div class="focus-nav">
          <button class="focus-nav-btn" id="ctx-focus-prev" ${hasPrev ? "" : "disabled"}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M15 18l-6-6 6-6"/>
            </svg>
            Prev
          </button>
          <button class="focus-nav-btn" id="ctx-focus-next" ${hasNext ? "" : "disabled"}>
            Next
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M9 18l6-6-6-6"/>
            </svg>
          </button>
          <button class="btn btn-danger btn-sm" id="ctx-focus-delete" title="Delete context entry">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/>
            </svg>
            Delete
          </button>
        </div>
        <div class="focus-meta">
          <span class="meta-item">
            <span class="meta-label">Tool</span>
            <span class="meta-value">${toolLabel(e.tool)}</span>
          </span>
          <span class="meta-item">
            <span class="meta-label">Data Type</span>
            <span class="meta-value">${dataTypeLabel(e.data_type)}</span>
          </span>
          <span class="meta-item">
            <span class="meta-label">Size</span>
            <span class="meta-value">${formatSize(e.size_bytes)}</span>
          </span>
          <span class="meta-item">
            <span class="meta-label">Created</span>
            <span class="meta-value">${formatFullTime(e.timestamp)}</span>
          </span>
        </div>
      </div>
    `;

    // Prev/next handlers
    const prevBtn = document.getElementById("ctx-focus-prev");
    const nextBtn = document.getElementById("ctx-focus-next");
    if (prevBtn && hasPrev) {
      prevBtn.addEventListener("click", () => {
        focusedContext = sorted[idx - 1];
        renderContextFocus();
      });
    }
    if (nextBtn && hasNext) {
      nextBtn.addEventListener("click", () => {
        focusedContext = sorted[idx + 1];
        renderContextFocus();
      });
    }
    document.getElementById("ctx-focus-delete")?.addEventListener("click", () => {
      if (!confirm(`Delete context entry #${e.id} "${e.description}"? This cannot be undone.`)) return;
      fetch(`/api/context/${e.id}`, { method: "DELETE" })
        .then(r => { if (!r.ok) throw new Error(); return r.json(); })
        .then(() => {
          contextEntries = contextEntries.filter(x => x.id !== e.id);
          focusedContext = null;
          updateHeaderContextCount();
          renderCtxSidebar();
          renderContextFocus();
        })
        .catch(err => console.error("Delete failed:", err));
    });

    // Lazy-fetch the JSON data
    const data = await fetchContextData(e.id);
    const viewer = document.getElementById("ctx-focus-json-viewer");
    if (viewer && data !== null) {
      viewer.innerHTML = renderJsonHtml(data, 0);
    } else if (viewer) {
      viewer.innerHTML = '<span class="text-muted">Failed to load data</span>';
    }
  }

  function contextFocusNavigate(delta) {
    if (!focusedContext || contextEntries.length === 0) return;
    const sorted = [...contextEntries].reverse();
    const idx = sorted.findIndex((x) => x.id === focusedContext.id);
    const newIdx = idx + delta;
    if (newIdx >= 0 && newIdx < sorted.length) {
      focusedContext = sorted[newIdx];
      renderContextFocus();
    }
  }

  // ---- Browse Sidebar ----

  function getFilteredArtifacts() {
    const query = searchInput.value.trim().toLowerCase();
    if (!query) return artifacts;
    return artifacts.filter(
      (a) =>
        a.title.toLowerCase().includes(query) ||
        a.filename.toLowerCase().includes(query) ||
        (a.description && a.description.toLowerCase().includes(query)) ||
        a.artifact_type.toLowerCase().includes(query)
    );
  }

  function renderSidebar() {
    const filtered = getFilteredArtifacts();

    if (filtered.length === 0) {
      sidebarBody.innerHTML = `
        <div class="sidebar-empty">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <rect x="3" y="3" width="7" height="7" rx="1"/>
            <rect x="14" y="3" width="7" height="7" rx="1"/>
            <rect x="3" y="14" width="7" height="7" rx="1"/>
            <rect x="14" y="14" width="7" height="7" rx="1"/>
          </svg>
          <span>${searchInput.value ? "No matches" : "No artifacts yet"}</span>
        </div>
      `;
      return;
    }

    if (browseMode === "tree") {
      renderTreeMode(filtered);
    } else {
      renderActivityMode(filtered);
    }
  }

  // ---- Gallery Card HTML (shared by both modes in gallery layout) ----

  function galleryCardHtml(a, i) {
    const sel = selectedArtifact && selectedArtifact.id === a.id ? " selected" : "";
    return `
      <div class="gallery-card${sel}"
           data-id="${a.id}"
           data-type="${a.artifact_type}"
           style="animation-delay: ${i * 30}ms">
        <div class="gallery-card-thumb">${thumbnailHtml(a)}</div>
        <div class="gallery-card-info">
          <div class="gallery-card-title" title="${escapeHtml(a.title)}">${escapeHtml(a.title)}</div>
          <div class="gallery-card-meta">
            <span class="gallery-card-type">${typeBadge(a.artifact_type)}</span>
            <span class="gallery-card-time">${formatTime(a.timestamp)}</span>
            <span class="gallery-card-size">${formatSize(a.size_bytes)}</span>
          </div>
        </div>
      </div>`;
  }

  // ---- Tree Mode (group by type) ----

  function renderTreeMode(items) {
    const groups = {};
    items.forEach((a) => {
      if (!groups[a.artifact_type]) groups[a.artifact_type] = [];
      groups[a.artifact_type].push(a);
    });

    const sortedTypes = Object.keys(groups).sort((a, b) => {
      const diff = groups[b].length - groups[a].length;
      return diff !== 0 ? diff : a.localeCompare(b);
    });

    const isGallery = sidebarLayout === "gallery";
    let html = "";
    let globalIdx = 0;

    sortedTypes.forEach((type) => {
      const typeArtifacts = groups[type];
      typeArtifacts.sort((a, b) => (b.timestamp || "").localeCompare(a.timestamp || ""));

      if (isGallery) {
        html += `
          <div class="tree-section" data-type="${type}">
            <div class="gallery-section-header" data-type="${type}">
              ${chevronSvg}
              <span class="tree-section-icon">${typeIcon(type)}</span>
              <span>${typeBadge(type)}</span>
              <span class="tree-section-count">${typeArtifacts.length}</span>
            </div>
            <div class="tree-section-items sidebar-gallery">
              ${typeArtifacts.map((a) => galleryCardHtml(a, globalIdx++)).join("")}
            </div>
          </div>`;
      } else {
        html += `
          <div class="tree-section" data-type="${type}">
            <div class="tree-section-header" data-type="${type}">
              ${chevronSvg}
              <span class="tree-section-icon">${typeIcon(type)}</span>
              <span>${typeBadge(type)}</span>
              <span class="tree-section-count">${typeArtifacts.length}</span>
            </div>
            <div class="tree-section-items">
              ${typeArtifacts
                .map(
                  (a, i) => `
                <div class="tree-item${selectedArtifact && selectedArtifact.id === a.id ? " selected" : ""}"
                     data-id="${a.id}"
                     style="animation-delay: ${i * 30}ms">
                  <span class="tree-item-icon">${typeIcon(a.artifact_type)}</span>
                  <span class="tree-item-name" title="${escapeHtml(a.title)}">${escapeHtml(a.title)}</span>
                  ${isNewThisSession(a) ? '<span class="tree-item-badge new">new</span>' : ""}
                  <span class="tree-item-size">${formatSize(a.size_bytes)}</span>
                </div>`
                )
                .join("")}
            </div>
          </div>`;
      }
    });

    sidebarBody.innerHTML = html;
    attachSidebarHandlers();
  }

  // ---- Activity Mode (chronological timeline) ----

  function renderActivityMode(items) {
    const sorted = [...items].sort((a, b) =>
      (b.timestamp || "").localeCompare(a.timestamp || "")
    );

    const dateGroups = {};
    sorted.forEach((a) => {
      const label = formatDate(a.timestamp);
      if (!dateGroups[label]) dateGroups[label] = [];
      dateGroups[label].push(a);
    });

    const isGallery = sidebarLayout === "gallery";
    let html = "";
    let itemIndex = 0;

    Object.entries(dateGroups).forEach(([label, group]) => {
      html += `<div class="timeline-group">`;
      html += `<div class="timeline-group-label">${label}</div>`;

      if (isGallery) {
        html += `<div class="sidebar-gallery">`;
        group.forEach((a) => {
          html += galleryCardHtml(a, itemIndex++);
        });
        html += `</div>`;
      } else {
        group.forEach((a) => {
          html += `
            <div class="timeline-item${selectedArtifact && selectedArtifact.id === a.id ? " selected" : ""}"
                 data-id="${a.id}"
                 data-type="${a.artifact_type}"
                 style="animation-delay: ${itemIndex * 25}ms">
              <span class="timeline-dot"></span>
              <div class="timeline-item-body">
                <div class="timeline-item-title" title="${escapeHtml(a.title)}">${escapeHtml(a.title)}</div>
                <div class="timeline-item-meta">
                  <span class="timeline-item-type">${typeBadge(a.artifact_type)}</span>
                  <span class="timeline-item-time">${formatTime(a.timestamp)}</span>
                </div>
              </div>
            </div>`;
          itemIndex++;
        });
      }

      html += `</div>`;
    });

    sidebarBody.innerHTML = html;
    attachSidebarHandlers();
  }

  function attachSidebarHandlers() {
    // Tree/gallery section toggle
    sidebarBody.querySelectorAll(".tree-section-header, .gallery-section-header").forEach((header) => {
      header.addEventListener("click", () => {
        header.parentElement.classList.toggle("collapsed");
      });
    });

    // Item click (tree, timeline, or gallery card)
    const clickables = ".tree-item, .timeline-item, .gallery-card";
    sidebarBody.querySelectorAll(clickables).forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.closest(".tree-section-header, .gallery-section-header")) return;

        const id = el.dataset.id;
        const a = artifacts.find((x) => x.id === id);
        if (a) {
          selectedArtifact = a;
          sidebarBody
            .querySelectorAll(clickables)
            .forEach((item) => item.classList.remove("selected"));
          el.classList.add("selected");
          renderPreview();
        }
      });
    });
  }

  // ---- Browse Preview Pane ----

  function renderPreview() {
    if (!selectedArtifact) {
      previewEmpty.classList.remove("hidden");
      previewContent.classList.add("hidden");
      return;
    }

    previewEmpty.classList.add("hidden");
    previewContent.classList.remove("hidden");

    const a = selectedArtifact;
    const url = fileUrl(a);

    let viewportHtml = "";
    switch (a.artifact_type) {
      case "plot_html":
      case "table_html":
      case "html":
        viewportHtml = `<iframe src="${url}" class="preview-iframe-light" sandbox="allow-scripts allow-same-origin"></iframe>`;
        break;
      case "notebook":
        viewportHtml = `<iframe src="/api/notebooks/${a.id}/rendered" class="preview-iframe-light" sandbox="allow-scripts allow-same-origin"></iframe>`;
        break;
      case "plot_png":
      case "image":
        viewportHtml = `<img src="${url}" alt="${escapeHtml(a.title)}" />`;
        break;
      case "markdown":
      case "text":
      case "json":
        viewportHtml = `<iframe src="${url}" class="preview-iframe-dark"></iframe>`;
        break;
      default:
        viewportHtml = `<div class="preview-download">
          <a href="${url}" target="_blank" class="btn btn-secondary">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
            </svg>
            Download ${escapeHtml(a.filename)}
          </a>
        </div>`;
    }

    previewContent.innerHTML = `
      <div class="preview-header">
        <div class="preview-header-left">
          <span class="badge badge-${a.artifact_type}">${typeBadge(a.artifact_type)}</span>
          <span class="preview-header-title">${escapeHtml(a.title)}</span>
        </div>
        <div class="preview-header-actions">
          <button class="btn btn-primary btn-sm" id="preview-set-focus" title="Set as focus artifact">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/>
              <circle cx="12" cy="12" r="3"/>
            </svg>
            Focus
          </button>
          <a href="${url}" target="_blank" class="btn btn-secondary btn-sm" title="Open in new tab">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3"/>
            </svg>
            Open
          </a>
          <button class="btn btn-danger btn-sm" id="preview-delete" title="Delete artifact">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/>
            </svg>
            Delete
          </button>
        </div>
      </div>
      ${a.description ? `<div class="preview-desc">${escapeHtml(a.description)}</div>` : ""}
      <div class="preview-meta-strip">
        <span class="preview-meta-item">
          <span class="preview-meta-label">Size</span>
          <span class="preview-meta-value">${formatSize(a.size_bytes)}</span>
        </span>
        <span class="preview-meta-item">
          <span class="preview-meta-label">Created</span>
          <span class="preview-meta-value">${formatFullTime(a.timestamp)}</span>
        </span>
        ${a.tool_source ? `
        <span class="preview-meta-item">
          <span class="preview-meta-label">Source</span>
          <span class="preview-meta-value">${escapeHtml(a.tool_source)}</span>
        </span>` : ""}
      </div>
      <div class="preview-viewport">
        ${viewportHtml}
      </div>
    `;

    document.getElementById("preview-set-focus").addEventListener("click", () => setAsFocus(a));
    document.getElementById("preview-delete").addEventListener("click", () => {
      if (!confirm(`Delete "${a.title}"? This cannot be undone.`)) return;
      fetch(`/api/artifacts/${a.id}`, { method: "DELETE" })
        .then(r => { if (!r.ok) throw new Error(); return r.json(); })
        .then(() => {
          artifacts = artifacts.filter(x => x.id !== a.id);
          if (selectedArtifact?.id === a.id) selectedArtifact = null;
          if (focusedArtifact?.id === a.id) focusedArtifact = null;
          updateHeaderCount();
          renderSidebar();
          renderPreview();
        })
        .catch(err => console.error("Delete failed:", err));
    });
  }

  async function setAsFocus(artifact) {
    try {
      const resp = await fetch("/api/focus", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ artifact_id: artifact.id }),
      });
      if (resp.ok) {
        focusedArtifact = artifact;
        navigateTo("artifacts/focus");
      }
    } catch (err) {
      console.error("Failed to set focus:", err);
    }
  }

  // ---- Context API ----

  async function fetchContextEntries() {
    try {
      const resp = await fetch("/api/context");
      const data = await resp.json();
      contextEntries = data.entries || [];
      updateHealth(true);
      updateHeaderContextCount();
      renderCtxSidebar();
    } catch (err) {
      console.error("Failed to fetch context entries:", err);
      updateHealth(false);
    }
  }

  async function fetchContextData(entryId) {
    try {
      const resp = await fetch(`/api/context/${entryId}/data`);
      return await resp.json();
    } catch (err) {
      console.error("Failed to fetch context data:", err);
      return null;
    }
  }

  function updateHeaderContextCount() {
    headerContextCount.textContent = contextEntries.length;
  }

  async function setContextFocus(entry) {
    try {
      const resp = await fetch("/api/context/focus", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ entry_id: entry.id }),
      });
      if (resp.ok) {
        focusedContext = entry;
        navigateTo("context/focus");
      }
    } catch (err) {
      console.error("Failed to set context focus:", err);
    }
  }

  // ---- Context Sidebar ----

  function getFilteredContextEntries() {
    const query = ctxSearchInput.value.trim().toLowerCase();
    if (!query) return contextEntries;
    return contextEntries.filter(
      (e) =>
        e.description.toLowerCase().includes(query) ||
        e.tool.toLowerCase().includes(query) ||
        e.data_type.toLowerCase().includes(query)
    );
  }

  function renderCtxSidebar() {
    const filtered = getFilteredContextEntries();

    if (filtered.length === 0) {
      ctxSidebarBody.innerHTML = `
        <div class="sidebar-empty">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <rect x="3" y="3" width="7" height="7" rx="1"/>
            <rect x="14" y="3" width="7" height="7" rx="1"/>
            <rect x="3" y="14" width="7" height="7" rx="1"/>
            <rect x="14" y="14" width="7" height="7" rx="1"/>
          </svg>
          <span>${ctxSearchInput.value ? "No matches" : "No context entries yet"}</span>
        </div>
      `;
      return;
    }

    if (ctxBrowseMode === "tree") {
      renderCtxTreeMode(filtered);
    } else {
      renderCtxActivityMode(filtered);
    }
  }

  function ctxGalleryCardHtml(entry, i) {
    const sel = selectedContext && selectedContext.id === entry.id ? " selected" : "";
    const summaryText = entry.summary
      ? Object.entries(entry.summary).map(([k, v]) => `${k}: ${v}`).join("\n")
      : entry.description;
    return `
      <div class="gallery-card${sel}"
           data-id="${entry.id}"
           data-tool="${entry.tool}"
           style="animation-delay: ${i * 30}ms">
        <div class="gallery-card-thumb">
          <div class="ctx-thumb-summary">${escapeHtml(summaryText)}</div>
        </div>
        <div class="gallery-card-info">
          <div class="gallery-card-title" title="${escapeHtml(entry.description)}">${escapeHtml(entry.description)}</div>
          <div class="gallery-card-meta">
            <span class="gallery-card-type">${toolLabel(entry.tool)}</span>
            <span class="gallery-card-time">${formatTime(entry.timestamp)}</span>
            <span class="gallery-card-size">${formatSize(entry.size_bytes)}</span>
          </div>
        </div>
      </div>`;
  }

  function renderCtxTreeMode(items) {
    const groups = {};
    items.forEach((e) => {
      if (!groups[e.tool]) groups[e.tool] = [];
      groups[e.tool].push(e);
    });

    const sortedTools = Object.keys(groups).sort((a, b) => {
      const diff = groups[b].length - groups[a].length;
      return diff !== 0 ? diff : a.localeCompare(b);
    });

    const isGallery = ctxSidebarLayout === "gallery";
    let html = "";
    let globalIdx = 0;

    sortedTools.forEach((tool) => {
      const toolEntries = groups[tool];
      toolEntries.sort((a, b) => (b.timestamp || "").localeCompare(a.timestamp || ""));

      if (isGallery) {
        html += `
          <div class="tree-section" data-tool="${tool}">
            <div class="gallery-section-header" data-tool="${tool}">
              ${chevronSvg}
              <span class="tree-section-icon">${toolIcon(tool)}</span>
              <span>${toolLabel(tool)}</span>
              <span class="tree-section-count">${toolEntries.length}</span>
            </div>
            <div class="tree-section-items sidebar-gallery">
              ${toolEntries.map((e) => ctxGalleryCardHtml(e, globalIdx++)).join("")}
            </div>
          </div>`;
      } else {
        html += `
          <div class="tree-section" data-tool="${tool}">
            <div class="tree-section-header" data-tool="${tool}">
              ${chevronSvg}
              <span class="tree-section-icon">${toolIcon(tool)}</span>
              <span>${toolLabel(tool)}</span>
              <span class="tree-section-count">${toolEntries.length}</span>
            </div>
            <div class="tree-section-items">
              ${toolEntries
                .map(
                  (e, i) => `
                <div class="tree-item${selectedContext && selectedContext.id === e.id ? " selected" : ""}"
                     data-id="${e.id}"
                     style="animation-delay: ${i * 30}ms">
                  <span class="tree-item-icon">${toolIcon(e.tool)}</span>
                  <span class="tree-item-name" title="${escapeHtml(e.description)}">${escapeHtml(e.description)}</span>
                  <span class="tree-item-size">${formatSize(e.size_bytes)}</span>
                </div>`
                )
                .join("")}
            </div>
          </div>`;
      }
    });

    ctxSidebarBody.innerHTML = html;
    attachCtxSidebarHandlers();
  }

  function renderCtxActivityMode(items) {
    const sorted = [...items].sort((a, b) =>
      (b.timestamp || "").localeCompare(a.timestamp || "")
    );

    const dateGroups = {};
    sorted.forEach((e) => {
      const label = formatDate(e.timestamp);
      if (!dateGroups[label]) dateGroups[label] = [];
      dateGroups[label].push(e);
    });

    const isGallery = ctxSidebarLayout === "gallery";
    let html = "";
    let itemIndex = 0;

    Object.entries(dateGroups).forEach(([label, group]) => {
      html += `<div class="timeline-group">`;
      html += `<div class="timeline-group-label">${label}</div>`;

      if (isGallery) {
        html += `<div class="sidebar-gallery">`;
        group.forEach((e) => {
          html += ctxGalleryCardHtml(e, itemIndex++);
        });
        html += `</div>`;
      } else {
        group.forEach((e) => {
          html += `
            <div class="timeline-item${selectedContext && selectedContext.id === e.id ? " selected" : ""}"
                 data-id="${e.id}"
                 data-tool="${e.tool}"
                 style="animation-delay: ${itemIndex * 25}ms">
              <span class="timeline-dot"></span>
              <div class="timeline-item-body">
                <div class="timeline-item-title" title="${escapeHtml(e.description)}">${escapeHtml(e.description)}</div>
                <div class="timeline-item-meta">
                  <span class="timeline-item-type">${toolLabel(e.tool)}</span>
                  <span class="timeline-item-time">${formatTime(e.timestamp)}</span>
                </div>
              </div>
            </div>`;
          itemIndex++;
        });
      }

      html += `</div>`;
    });

    ctxSidebarBody.innerHTML = html;
    attachCtxSidebarHandlers();
  }

  function attachCtxSidebarHandlers() {
    // Section collapse toggle
    ctxSidebarBody.querySelectorAll(".tree-section-header, .gallery-section-header").forEach((header) => {
      header.addEventListener("click", () => {
        header.parentElement.classList.toggle("collapsed");
      });
    });

    // Item click
    const clickables = ".tree-item, .timeline-item, .gallery-card";
    ctxSidebarBody.querySelectorAll(clickables).forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.closest(".tree-section-header, .gallery-section-header")) return;

        const id = parseInt(el.dataset.id, 10);
        const entry = contextEntries.find((x) => x.id === id);
        if (entry) {
          selectedContext = entry;
          ctxSidebarBody
            .querySelectorAll(clickables)
            .forEach((item) => item.classList.remove("selected"));
          el.classList.add("selected");
          renderCtxPreview();
        }
      });
    });
  }

  // ---- Context Preview Pane ----

  async function renderCtxPreview() {
    if (!selectedContext) {
      ctxPreviewEmpty.classList.remove("hidden");
      ctxPreviewContent.classList.add("hidden");
      return;
    }

    ctxPreviewEmpty.classList.add("hidden");
    ctxPreviewContent.classList.remove("hidden");

    const e = selectedContext;

    // Build summary and access_details card content
    const summaryHtml = e.summary
      ? Object.entries(e.summary).map(([k, v]) =>
          `<div><span style="color:var(--text-muted)">${escapeHtml(k)}:</span> ${escapeHtml(String(v))}</div>`
        ).join("")
      : "<div>No summary</div>";

    const accessHtml = e.access_details
      ? Object.entries(e.access_details).map(([k, v]) =>
          `<div><span style="color:var(--text-muted)">${escapeHtml(k)}:</span> ${escapeHtml(String(v))}</div>`
        ).join("")
      : "<div>No access details</div>";

    ctxPreviewContent.innerHTML = `
      <div class="preview-header">
        <div class="preview-header-left">
          <span class="badge badge-${e.tool}">${toolLabel(e.tool)}</span>
          <span class="preview-header-title">${escapeHtml(e.description)}</span>
        </div>
        <div class="preview-header-actions">
          <button class="btn btn-primary btn-sm" id="ctx-preview-set-focus" title="Set as focus context">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/>
              <circle cx="12" cy="12" r="3"/>
            </svg>
            Focus
          </button>
          <button class="btn btn-danger btn-sm" id="ctx-preview-delete" title="Delete context entry">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/>
            </svg>
            Delete
          </button>
        </div>
      </div>
      <div class="preview-meta-strip">
        <span class="preview-meta-item">
          <span class="preview-meta-label">Tool</span>
          <span class="preview-meta-value">${toolLabel(e.tool)}</span>
        </span>
        <span class="preview-meta-item">
          <span class="preview-meta-label">Data Type</span>
          <span class="preview-meta-value">${dataTypeLabel(e.data_type)}</span>
        </span>
        <span class="preview-meta-item">
          <span class="preview-meta-label">Size</span>
          <span class="preview-meta-value">${formatSize(e.size_bytes)}</span>
        </span>
        <span class="preview-meta-item">
          <span class="preview-meta-label">Created</span>
          <span class="preview-meta-value">${formatFullTime(e.timestamp)}</span>
        </span>
      </div>
      <div class="context-info-cards">
        <div class="context-info-card">
          <div class="context-info-card-title">Summary</div>
          <div class="context-info-card-body">${summaryHtml}</div>
        </div>
        <div class="context-info-card">
          <div class="context-info-card-title">Access Details</div>
          <div class="context-info-card-body">${accessHtml}</div>
        </div>
      </div>
      <div class="json-viewer" id="ctx-json-viewer">
        <span class="text-muted">Loading data...</span>
      </div>
    `;

    document.getElementById("ctx-preview-set-focus").addEventListener("click", () => setContextFocus(e));
    document.getElementById("ctx-preview-delete").addEventListener("click", () => {
      if (!confirm(`Delete context entry #${e.id} "${e.description}"? This cannot be undone.`)) return;
      fetch(`/api/context/${e.id}`, { method: "DELETE" })
        .then(r => { if (!r.ok) throw new Error(); return r.json(); })
        .then(() => {
          contextEntries = contextEntries.filter(x => x.id !== e.id);
          if (selectedContext?.id === e.id) selectedContext = null;
          if (focusedContext?.id === e.id) focusedContext = null;
          updateHeaderContextCount();
          renderCtxSidebar();
          renderCtxPreview();
        })
        .catch(err => console.error("Delete failed:", err));
    });

    // Lazy-fetch the JSON data
    const data = await fetchContextData(e.id);
    const viewer = document.getElementById("ctx-json-viewer");
    if (viewer && data !== null) {
      viewer.innerHTML = renderJsonHtml(data, 0);
    } else if (viewer) {
      viewer.innerHTML = '<span class="text-muted">Failed to load data</span>';
    }
  }

  // ---- Memory Type Helpers ----

  function memoryTypeIcon(type) {
    const icons = {
      note: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>',
      pin: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2a3 3 0 00-3 3c0 1.8 1.2 3.5 3 5 1.8-1.5 3-3.2 3-5a3 3 0 00-3-3z"/><path d="M12 10v12"/><path d="M8 22h8"/></svg>',
    };
    return icons[type] || icons.note;
  }

  function memoryTypeLabel(type) {
    return type === "pin" ? "Pin" : "Note";
  }

  // ---- Memory API ----

  async function fetchMemoryEntries() {
    try {
      const resp = await fetch("/api/memory");
      const data = await resp.json();
      memoryEntries = data.entries || [];
      updateHealth(true);
      updateHeaderMemoryCount();
      renderMemSidebar();
    } catch (err) {
      console.error("Failed to fetch memory entries:", err);
      updateHealth(false);
    }
  }

  async function fetchMemoryFocus() {
    try {
      const resp = await fetch("/api/memory/focus");
      const data = await resp.json();
      focusedMemory = data.entry;
      updateHealth(true);
      renderMemFocus();
    } catch (err) {
      console.error("Failed to fetch memory focus:", err);
      updateHealth(false);
    }

    // Also fetch entries for prev/next + header count
    try {
      const resp = await fetch("/api/memory");
      const data = await resp.json();
      memoryEntries = data.entries || [];
      updateHeaderMemoryCount();
    } catch { /* non-critical */ }
  }

  function updateHeaderMemoryCount() {
    headerMemoryCount.textContent = memoryEntries.length;
  }

  async function setMemoryFocus(entry) {
    try {
      const resp = await fetch("/api/memory/focus", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ memory_id: entry.id }),
      });
      if (resp.ok) {
        focusedMemory = entry;
        navigateTo("memory/focus");
      }
    } catch (err) {
      console.error("Failed to set memory focus:", err);
    }
  }

  // ---- Memory Focus View ----

  function renderMemFocus() {
    if (!focusedMemory) {
      memFocusContainer.innerHTML = "";
      memFocusContainer.classList.add("hidden");
      memFocusEmpty.classList.remove("hidden");
      return;
    }
    memFocusEmpty.classList.add("hidden");
    memFocusContainer.classList.remove("hidden");

    const m = focusedMemory;

    const tagsHtml = m.tags && m.tags.length > 0
      ? m.tags.map(t => `<span class="mem-tag-pill">${escapeHtml(t)}</span>`).join(" ")
      : "";

    const importanceHtml = m.importance === "important"
      ? '<span class="mem-importance-star" title="Important">&#9733;</span>'
      : "";

    let linkHtml = "";
    if (m.memory_type === "pin" && m.linked_label) {
      const linkDomain = m.linked_artifact_id ? "Artifact" : "Context";
      linkHtml = `<div style="margin-top: var(--space-3);">
        <span class="mem-link-badge" data-link-artifact="${m.linked_artifact_id || ""}" data-link-context="${m.linked_context_id || ""}">
          ${linkDomain}: "${escapeHtml(m.linked_label)}"
        </span>
      </div>`;
    }

    // Prev/next navigation
    const sorted = [...memoryEntries].reverse();
    const idx = sorted.findIndex((x) => x.id === m.id);
    const hasPrev = idx > 0;
    const hasNext = idx >= 0 && idx < sorted.length - 1;

    memFocusContainer.innerHTML = `
      <div class="focus-title-row">
        <span class="focus-led violet"></span>
        <h2>${memoryTypeLabel(m.memory_type)} #${m.id} ${importanceHtml}</h2>
        <span class="badge" style="color: var(--color-violet-light); background: var(--color-violet-glow); border: 1px solid rgba(159,122,234,0.25);">
          ${memoryTypeLabel(m.memory_type)}
        </span>
      </div>
      ${tagsHtml ? `<div style="margin: var(--space-2) 0; display: flex; gap: 4px; flex-wrap: wrap;">${tagsHtml}</div>` : ""}
      <div class="focus-json-viewport">
        <div class="context-info-cards">
          <div class="context-info-card" style="flex: 1;">
            <div class="context-info-card-title">${m.memory_type === "pin" ? "Annotation" : "Content"}</div>
            <div class="context-info-card-body" style="white-space: pre-wrap;">${escapeHtml(m.content)}</div>
          </div>
        </div>
        ${linkHtml}
      </div>
      <div class="focus-footer">
        <div class="focus-nav">
          <button class="focus-nav-btn" id="mem-focus-prev" ${hasPrev ? "" : "disabled"}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M15 18l-6-6 6-6"/>
            </svg>
            Prev
          </button>
          <button class="focus-nav-btn" id="mem-focus-next" ${hasNext ? "" : "disabled"}>
            Next
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M9 18l6-6-6-6"/>
            </svg>
          </button>
          <button class="btn btn-danger btn-sm" id="mem-focus-delete" title="Delete memory">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/>
            </svg>
            Delete
          </button>
        </div>
        <div class="focus-meta">
          <span class="meta-item">
            <span class="meta-label">Type</span>
            <span class="meta-value">${memoryTypeLabel(m.memory_type)}</span>
          </span>
          <span class="meta-item">
            <span class="meta-label">Importance</span>
            <span class="meta-value">${m.importance}</span>
          </span>
          <span class="meta-item">
            <span class="meta-label">Created</span>
            <span class="meta-value">${formatFullTime(m.timestamp)}</span>
          </span>
        </div>
      </div>
    `;

    // Prev/next handlers
    const prevBtn = document.getElementById("mem-focus-prev");
    const nextBtn = document.getElementById("mem-focus-next");
    if (prevBtn && hasPrev) {
      prevBtn.addEventListener("click", () => {
        focusedMemory = sorted[idx - 1];
        renderMemFocus();
      });
    }
    if (nextBtn && hasNext) {
      nextBtn.addEventListener("click", () => {
        focusedMemory = sorted[idx + 1];
        renderMemFocus();
      });
    }
    document.getElementById("mem-focus-delete")?.addEventListener("click", () => {
      if (!confirm(`Delete memory #${m.id}? This cannot be undone.`)) return;
      fetch(`/api/memory/${m.id}`, { method: "DELETE" })
        .then(r => { if (!r.ok) throw new Error(); return r.json(); })
        .then(() => {
          memoryEntries = memoryEntries.filter(x => x.id !== m.id);
          focusedMemory = null;
          updateHeaderMemoryCount();
          renderMemSidebar();
          renderMemFocus();
        })
        .catch(err => console.error("Delete failed:", err));
    });

    // Link badge click handler
    memFocusContainer.querySelectorAll(".mem-link-badge").forEach((badge) => {
      badge.addEventListener("click", () => navigateToLinked(m));
    });
  }

  function memoryFocusNavigate(delta) {
    if (!focusedMemory || memoryEntries.length === 0) return;
    const sorted = [...memoryEntries].reverse();
    const idx = sorted.findIndex((x) => x.id === focusedMemory.id);
    const newIdx = idx + delta;
    if (newIdx >= 0 && newIdx < sorted.length) {
      focusedMemory = sorted[newIdx];
      renderMemFocus();
    }
  }

  function navigateToLinked(entry) {
    if (entry.linked_artifact_id) {
      const art = artifacts.find(a => a.id === entry.linked_artifact_id);
      if (art) {
        focusedArtifact = art;
        navigateTo("artifacts/focus");
      }
    } else if (entry.linked_context_id) {
      const ctx = contextEntries.find(e => e.id === entry.linked_context_id);
      if (ctx) {
        focusedContext = ctx;
        navigateTo("context/focus");
      }
    }
  }

  // ---- Memory Sidebar ----

  function getFilteredMemoryEntries() {
    const query = memSearchInput.value.trim().toLowerCase();
    if (!query) return memoryEntries;
    return memoryEntries.filter(
      (m) =>
        m.content.toLowerCase().includes(query) ||
        m.memory_type.toLowerCase().includes(query) ||
        (m.tags && m.tags.some(t => t.toLowerCase().includes(query))) ||
        (m.linked_label && m.linked_label.toLowerCase().includes(query))
    );
  }

  function renderMemSidebar() {
    const filtered = getFilteredMemoryEntries();

    if (filtered.length === 0) {
      memSidebarBody.innerHTML = `
        <div class="sidebar-empty">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <rect x="3" y="3" width="7" height="7" rx="1"/>
            <rect x="14" y="3" width="7" height="7" rx="1"/>
            <rect x="3" y="14" width="7" height="7" rx="1"/>
            <rect x="14" y="14" width="7" height="7" rx="1"/>
          </svg>
          <span>${memSearchInput.value ? "No matches" : "No memories yet"}</span>
        </div>
      `;
      return;
    }

    if (memBrowseMode === "tree") {
      renderMemTreeMode(filtered);
    } else {
      renderMemActivityMode(filtered);
    }
  }

  function memGalleryCardHtml(m, i) {
    const sel = selectedMemory && selectedMemory.id === m.id ? " selected" : "";
    const importanceStar = m.importance === "important" ? " &#9733;" : "";
    const preview = m.content.length > 80 ? m.content.substring(0, 80) + "..." : m.content;
    return `
      <div class="gallery-card${sel}"
           data-id="${m.id}"
           data-memory-type="${m.memory_type}"
           style="animation-delay: ${i * 30}ms">
        <div class="gallery-card-thumb">
          <div class="ctx-thumb-summary">${escapeHtml(preview)}</div>
        </div>
        <div class="gallery-card-info">
          <div class="gallery-card-title" title="${escapeHtml(m.content)}">${memoryTypeLabel(m.memory_type)} #${m.id}${importanceStar}</div>
          <div class="gallery-card-meta">
            <span class="gallery-card-type">${memoryTypeLabel(m.memory_type)}</span>
            <span class="gallery-card-time">${formatTime(m.timestamp)}</span>
          </div>
        </div>
      </div>`;
  }

  function renderMemTreeMode(items) {
    const groups = {};
    items.forEach((m) => {
      if (!groups[m.memory_type]) groups[m.memory_type] = [];
      groups[m.memory_type].push(m);
    });

    // Sort: notes first, then pins
    const sortedTypes = Object.keys(groups).sort((a, b) => {
      const diff = groups[b].length - groups[a].length;
      return diff !== 0 ? diff : a.localeCompare(b);
    });

    const isGallery = memSidebarLayout === "gallery";
    let html = "";
    let globalIdx = 0;

    sortedTypes.forEach((type) => {
      const typeEntries = groups[type];
      typeEntries.sort((a, b) => (b.timestamp || "").localeCompare(a.timestamp || ""));

      const sectionLabel = type === "note" ? "Notes" : "Pins";

      if (isGallery) {
        html += `
          <div class="tree-section" data-memory-type="${type}">
            <div class="gallery-section-header" data-memory-type="${type}">
              ${chevronSvg}
              <span class="tree-section-icon">${memoryTypeIcon(type)}</span>
              <span>${sectionLabel}</span>
              <span class="tree-section-count">${typeEntries.length}</span>
            </div>
            <div class="tree-section-items sidebar-gallery">
              ${typeEntries.map((m) => memGalleryCardHtml(m, globalIdx++)).join("")}
            </div>
          </div>`;
      } else {
        html += `
          <div class="tree-section" data-memory-type="${type}">
            <div class="tree-section-header" data-memory-type="${type}">
              ${chevronSvg}
              <span class="tree-section-icon">${memoryTypeIcon(type)}</span>
              <span>${sectionLabel}</span>
              <span class="tree-section-count">${typeEntries.length}</span>
            </div>
            <div class="tree-section-items">
              ${typeEntries
                .map(
                  (m, i) => {
                    const importanceStar = m.importance === "important" ? '<span class="mem-importance-star">&#9733;</span>' : "";
                    const contentPreview = m.content.length > 50 ? m.content.substring(0, 50) + "..." : m.content;
                    return `
                <div class="tree-item${selectedMemory && selectedMemory.id === m.id ? " selected" : ""}"
                     data-id="${m.id}"
                     style="animation-delay: ${i * 30}ms">
                  <span class="tree-item-icon">${memoryTypeIcon(m.memory_type)}</span>
                  <span class="tree-item-name" title="${escapeHtml(m.content)}">${importanceStar}${escapeHtml(contentPreview)}</span>
                </div>`;
                  }
                )
                .join("")}
            </div>
          </div>`;
      }
    });

    memSidebarBody.innerHTML = html;
    attachMemSidebarHandlers();
  }

  function renderMemActivityMode(items) {
    const sorted = [...items].sort((a, b) =>
      (b.timestamp || "").localeCompare(a.timestamp || "")
    );

    const dateGroups = {};
    sorted.forEach((m) => {
      const label = formatDate(m.timestamp);
      if (!dateGroups[label]) dateGroups[label] = [];
      dateGroups[label].push(m);
    });

    const isGallery = memSidebarLayout === "gallery";
    let html = "";
    let itemIndex = 0;

    Object.entries(dateGroups).forEach(([label, group]) => {
      html += `<div class="timeline-group">`;
      html += `<div class="timeline-group-label">${label}</div>`;

      if (isGallery) {
        html += `<div class="sidebar-gallery">`;
        group.forEach((m) => {
          html += memGalleryCardHtml(m, itemIndex++);
        });
        html += `</div>`;
      } else {
        group.forEach((m) => {
          const contentPreview = m.content.length > 50 ? m.content.substring(0, 50) + "..." : m.content;
          html += `
            <div class="timeline-item${selectedMemory && selectedMemory.id === m.id ? " selected" : ""}"
                 data-id="${m.id}"
                 data-memory-type="${m.memory_type}"
                 style="animation-delay: ${itemIndex * 25}ms">
              <span class="timeline-dot"></span>
              <div class="timeline-item-body">
                <div class="timeline-item-title" title="${escapeHtml(m.content)}">${escapeHtml(contentPreview)}</div>
                <div class="timeline-item-meta">
                  <span class="timeline-item-type">${memoryTypeLabel(m.memory_type)}</span>
                  <span class="timeline-item-time">${formatTime(m.timestamp)}</span>
                </div>
              </div>
            </div>`;
          itemIndex++;
        });
      }

      html += `</div>`;
    });

    memSidebarBody.innerHTML = html;
    attachMemSidebarHandlers();
  }

  function attachMemSidebarHandlers() {
    // Section collapse toggle
    memSidebarBody.querySelectorAll(".tree-section-header, .gallery-section-header").forEach((header) => {
      header.addEventListener("click", () => {
        header.parentElement.classList.toggle("collapsed");
      });
    });

    // Item click
    const clickables = ".tree-item, .timeline-item, .gallery-card";
    memSidebarBody.querySelectorAll(clickables).forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.closest(".tree-section-header, .gallery-section-header")) return;

        const id = parseInt(el.dataset.id, 10);
        const entry = memoryEntries.find((x) => x.id === id);
        if (entry) {
          selectedMemory = entry;
          memSidebarBody
            .querySelectorAll(clickables)
            .forEach((item) => item.classList.remove("selected"));
          el.classList.add("selected");
          renderMemPreview();
        }
      });
    });
  }

  // ---- Memory Preview Pane ----

  function renderMemPreview() {
    if (!selectedMemory) {
      memPreviewEmpty.classList.remove("hidden");
      memPreviewContent.classList.add("hidden");
      return;
    }

    memPreviewEmpty.classList.add("hidden");
    memPreviewContent.classList.remove("hidden");

    const m = selectedMemory;

    const tagsHtml = m.tags && m.tags.length > 0
      ? m.tags.map(t => `<span class="mem-tag-pill">${escapeHtml(t)}</span>`).join(" ")
      : "";

    const importanceHtml = m.importance === "important"
      ? '<span class="mem-importance-star" title="Important">&#9733;</span>'
      : "";

    let linkHtml = "";
    if (m.memory_type === "pin" && m.linked_label) {
      const linkDomain = m.linked_artifact_id ? "Artifact" : "Context";
      linkHtml = `<div style="margin-top: var(--space-3);">
        <span class="mem-link-badge" data-link-artifact="${m.linked_artifact_id || ""}" data-link-context="${m.linked_context_id || ""}">
          ${linkDomain}: "${escapeHtml(m.linked_label)}"
        </span>
      </div>`;
    }

    memPreviewContent.innerHTML = `
      <div class="preview-header">
        <div class="preview-header-left">
          <span class="badge" style="color: var(--color-violet-light); background: var(--color-violet-glow); border: 1px solid rgba(159,122,234,0.25);">
            ${memoryTypeLabel(m.memory_type)}
          </span>
          <span class="preview-header-title">${memoryTypeLabel(m.memory_type)} #${m.id} ${importanceHtml}</span>
        </div>
        <div class="preview-header-actions">
          <button class="btn btn-primary btn-sm" id="mem-preview-set-focus" title="Set as focus memory">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/>
              <circle cx="12" cy="12" r="3"/>
            </svg>
            Focus
          </button>
          <button class="btn btn-danger btn-sm" id="mem-preview-delete" title="Delete memory">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/>
            </svg>
            Delete
          </button>
        </div>
      </div>
      <div class="preview-meta-strip">
        <span class="preview-meta-item">
          <span class="preview-meta-label">Type</span>
          <span class="preview-meta-value">${memoryTypeLabel(m.memory_type)}</span>
        </span>
        <span class="preview-meta-item">
          <span class="preview-meta-label">Importance</span>
          <span class="preview-meta-value">${m.importance}</span>
        </span>
        <span class="preview-meta-item">
          <span class="preview-meta-label">Created</span>
          <span class="preview-meta-value">${formatFullTime(m.timestamp)}</span>
        </span>
      </div>
      ${tagsHtml ? `<div style="margin: var(--space-2) var(--space-4); display: flex; gap: 4px; flex-wrap: wrap;">${tagsHtml}</div>` : ""}
      <div class="context-info-cards" style="margin: var(--space-3) var(--space-4);">
        <div class="context-info-card" style="flex: 1;">
          <div class="context-info-card-title">${m.memory_type === "pin" ? "Annotation" : "Content"}</div>
          <div class="context-info-card-body" style="white-space: pre-wrap;">${escapeHtml(m.content)}</div>
        </div>
      </div>
      ${linkHtml}
    `;

    document.getElementById("mem-preview-set-focus").addEventListener("click", () => setMemoryFocus(m));
    document.getElementById("mem-preview-delete").addEventListener("click", () => {
      if (!confirm(`Delete memory #${m.id}? This cannot be undone.`)) return;
      fetch(`/api/memory/${m.id}`, { method: "DELETE" })
        .then(r => { if (!r.ok) throw new Error(); return r.json(); })
        .then(() => {
          memoryEntries = memoryEntries.filter(x => x.id !== m.id);
          if (selectedMemory?.id === m.id) selectedMemory = null;
          if (focusedMemory?.id === m.id) focusedMemory = null;
          updateHeaderMemoryCount();
          renderMemSidebar();
          renderMemPreview();
        })
        .catch(err => console.error("Delete failed:", err));
    });

    // Link badge click handler
    memPreviewContent.querySelectorAll(".mem-link-badge").forEach((badge) => {
      badge.addEventListener("click", () => navigateToLinked(m));
    });
  }

  // ---- Stats View ----

  function renderStats() {
    const statsContent = document.getElementById("stats-content");
    if (!statsContent) return;

    const totalCount = artifacts.length;
    const totalSize = artifacts.reduce((sum, a) => sum + (a.size_bytes || 0), 0);
    const typeCounts = {};
    artifacts.forEach((a) => {
      typeCounts[a.artifact_type] = (typeCounts[a.artifact_type] || 0) + 1;
    });

    let html = `
      <div class="stat-card">
        <div class="stat-card-title">Total Artifacts</div>
        <div class="stat-card-value">${totalCount}</div>
        <div class="stat-card-subtitle">${Object.keys(typeCounts).length} types</div>
      </div>
      <div class="stat-card">
        <div class="stat-card-title">Total Size</div>
        <div class="stat-card-value">${formatSize(totalSize)}</div>
        <div class="stat-card-subtitle">across all artifacts</div>
      </div>
    `;

    const sortedTypes = Object.entries(typeCounts).sort((a, b) => b[1] - a[1]);
    sortedTypes.forEach(([type, count]) => {
      html += `
        <div class="stat-card">
          <div class="stat-card-title">${typeBadge(type)}</div>
          <div class="stat-card-value">${count}</div>
          <div class="stat-card-subtitle">${((count / totalCount) * 100).toFixed(0)}% of total</div>
        </div>
      `;
    });

    // Context stats section
    const ctxTotal = contextEntries.length;
    const ctxSize = contextEntries.reduce((sum, e) => sum + (e.size_bytes || 0), 0);
    const toolCounts = {};
    contextEntries.forEach((e) => {
      toolCounts[e.tool] = (toolCounts[e.tool] || 0) + 1;
    });

    html += `
      <div class="stats-section-divider">
        <h3>Data Context</h3>
        <p class="text-secondary">MCP tool output data stored during this session</p>
      </div>
      <div class="stat-card">
        <div class="stat-card-title">Total Entries</div>
        <div class="stat-card-value">${ctxTotal}</div>
        <div class="stat-card-subtitle">${Object.keys(toolCounts).length} tools</div>
      </div>
      <div class="stat-card">
        <div class="stat-card-title">Context Size</div>
        <div class="stat-card-value">${formatSize(ctxSize)}</div>
        <div class="stat-card-subtitle">across all entries</div>
      </div>
    `;

    Object.entries(toolCounts).sort((a, b) => b[1] - a[1]).forEach(([tool, count]) => {
      html += `
        <div class="stat-card">
          <div class="stat-card-title">${toolLabel(tool)}</div>
          <div class="stat-card-value">${count}</div>
          <div class="stat-card-subtitle">${ctxTotal > 0 ? ((count / ctxTotal) * 100).toFixed(0) : 0}% of context</div>
        </div>
      `;
    });

    // Memory stats section
    const memTotal = memoryEntries.length;
    const memNotes = memoryEntries.filter(m => m.memory_type === "note").length;
    const memPins = memoryEntries.filter(m => m.memory_type === "pin").length;
    const memImportant = memoryEntries.filter(m => m.importance === "important").length;
    const allTags = new Set();
    memoryEntries.forEach(m => (m.tags || []).forEach(t => allTags.add(t)));

    html += `
      <div class="stats-section-divider">
        <h3>Memory</h3>
        <p class="text-secondary">Saved notes and pins from this session</p>
      </div>
      <div class="stat-card">
        <div class="stat-card-title">Total Memories</div>
        <div class="stat-card-value">${memTotal}</div>
        <div class="stat-card-subtitle">${memNotes} notes, ${memPins} pins</div>
      </div>
      <div class="stat-card">
        <div class="stat-card-title">Important</div>
        <div class="stat-card-value">${memImportant}</div>
        <div class="stat-card-subtitle">marked as important</div>
      </div>
      <div class="stat-card">
        <div class="stat-card-title">Tags</div>
        <div class="stat-card-value">${allTags.size}</div>
        <div class="stat-card-subtitle">distinct tags</div>
      </div>
    `;

    statsContent.innerHTML = html;
  }

  // ---- Events ----

  // Search filter (browse)
  searchInput.addEventListener("input", debounce(() => renderSidebar(), 200));

  // Search filter (context)
  ctxSearchInput.addEventListener("input", debounce(() => renderCtxSidebar(), 200));

  // Search filter (memory)
  memSearchInput.addEventListener("input", debounce(() => renderMemSidebar(), 200));

  // Mode toggle — browse sidebar (non-ctx buttons only)
  document.querySelectorAll(".mode-btn:not(.ctx-mode-btn)").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.dataset.mode;
      if (mode === browseMode) return;
      browseMode = mode;
      document.querySelectorAll(".mode-btn:not(.ctx-mode-btn)").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderSidebar();
    });
  });

  // Mode toggle — context sidebar
  document.querySelectorAll(".ctx-mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.dataset.mode;
      if (mode === ctxBrowseMode) return;
      ctxBrowseMode = mode;
      document.querySelectorAll(".ctx-mode-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderCtxSidebar();
    });
  });

  // Layout toggle — browse sidebar (non-ctx buttons only)
  document.querySelectorAll(".layout-btn:not(.ctx-layout-btn)").forEach((btn) => {
    btn.addEventListener("click", () => {
      const layout = btn.dataset.layout;
      if (layout === sidebarLayout) return;
      sidebarLayout = layout;
      document.querySelectorAll(".layout-btn:not(.ctx-layout-btn)").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderSidebar();
    });
  });

  // Layout toggle — context sidebar
  document.querySelectorAll(".ctx-layout-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const layout = btn.dataset.layout;
      if (layout === ctxSidebarLayout) return;
      ctxSidebarLayout = layout;
      document.querySelectorAll(".ctx-layout-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderCtxSidebar();
    });
  });

  // Mode toggle — memory sidebar
  document.querySelectorAll(".mem-mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.dataset.mode;
      if (mode === memBrowseMode) return;
      memBrowseMode = mode;
      document.querySelectorAll(".mem-mode-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderMemSidebar();
    });
  });

  // Layout toggle — memory sidebar
  document.querySelectorAll(".mem-layout-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const layout = btn.dataset.layout;
      if (layout === memSidebarLayout) return;
      memSidebarLayout = layout;
      document.querySelectorAll(".mem-layout-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderMemSidebar();
    });
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    const { domain, subview } = currentRoute();
    const activeSearch = domain === "context" ? ctxSearchInput : searchInput;

    const searchInputs = [searchInput, ctxSearchInput, memSearchInput];
    if (e.key === "/" && !searchInputs.includes(document.activeElement)) {
      e.preventDefault();
      if (subview === "browse") {
        if (domain === "memory") memSearchInput.focus();
        else if (domain === "context") ctxSearchInput.focus();
        else searchInput.focus();
      } else {
        searchInput.focus();
      }
    }
    if (e.key === "Escape") {
      if (document.activeElement === searchInput) {
        searchInput.blur();
        searchInput.value = "";
        renderSidebar();
      } else if (document.activeElement === ctxSearchInput) {
        ctxSearchInput.blur();
        ctxSearchInput.value = "";
        renderCtxSidebar();
      } else if (document.activeElement === memSearchInput) {
        memSearchInput.blur();
        memSearchInput.value = "";
        renderMemSidebar();
      }
    }

    // Arrow keys in all focus views
    if (subview === "focus") {
      if (domain === "artifacts") {
        if (e.key === "ArrowLeft") focusNavigate(-1);
        if (e.key === "ArrowRight") focusNavigate(1);
      } else if (domain === "context") {
        if (e.key === "ArrowLeft") contextFocusNavigate(-1);
        if (e.key === "ArrowRight") contextFocusNavigate(1);
      } else if (domain === "memory") {
        if (e.key === "ArrowLeft") memoryFocusNavigate(-1);
        if (e.key === "ArrowRight") memoryFocusNavigate(1);
      }
    }
  });

  function debounce(fn, ms) {
    let timer;
    return function (...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  // ---- SSE (Server-Sent Events) ----

  let sseSource = null;

  function connectSSE() {
    if (sseSource) {
      sseSource.close();
      sseSource = null;
    }
    const source = new EventSource("/api/events");
    sseSource = source;
    source.onopen = () => updateHealth(true);
    source.onmessage = (event) => {
      updateHealth(true);
      let eventData = null;
      try { eventData = JSON.parse(event.data); } catch { /* ignore */ }
      const eventType = eventData && eventData.type;

      // Focus change — switch to focus view and re-fetch
      if (eventType === "focus" && eventData.domain) {
        const d = eventData.domain === "artifact" ? "artifacts" : eventData.domain;
        navigateTo(`${d}/focus`);
        // Force re-fetch even if already on this view
        if (d === "artifacts") fetchFocus();
        else if (d === "context") fetchContextFocus();
        else if (d === "memory") fetchMemoryFocus();
        return;
      }

      // Handle deletion events
      if (eventType === "artifact_deleted") {
        artifacts = artifacts.filter(a => a.id !== eventData.id);
        if (focusedArtifact?.id === eventData.id) focusedArtifact = null;
        if (selectedArtifact?.id === eventData.id) { selectedArtifact = null; renderPreview(); }
        updateHeaderCount();
        renderSidebar();
        const { domain: dd, subview: ds } = currentRoute();
        if (dd === "artifacts" && ds === "focus") renderFocus();
        return;
      }
      if (eventType === "context_deleted") {
        contextEntries = contextEntries.filter(e => e.id !== eventData.id);
        if (focusedContext?.id === eventData.id) focusedContext = null;
        if (selectedContext?.id === eventData.id) { selectedContext = null; renderCtxPreview(); }
        updateHeaderContextCount();
        renderCtxSidebar();
        const { domain: dd, subview: ds } = currentRoute();
        if (dd === "context" && ds === "focus") renderContextFocus();
        return;
      }
      if (eventType === "memory_deleted") {
        memoryEntries = memoryEntries.filter(e => e.id !== eventData.id);
        if (focusedMemory?.id === eventData.id) focusedMemory = null;
        if (selectedMemory?.id === eventData.id) { selectedMemory = null; renderMemPreview(); }
        updateHeaderMemoryCount();
        renderMemSidebar();
        const { domain: dd, subview: ds } = currentRoute();
        if (dd === "memory" && ds === "focus") renderMemFocus();
        return;
      }

      // Always update counts for relevant event types
      if (eventType === "context") {
        fetchContextEntries().catch(() => {});
      }
      if (eventType === "memory") {
        fetchMemoryEntries().catch(() => {});
      }
      if (eventType === "artifact" || !eventType) {
        fetchArtifacts().catch(() => {});
      }

      // Refresh active view
      const { domain, subview } = currentRoute();
      if (domain === "artifacts" && subview === "focus") fetchFocus();
      if (domain === "context" && subview === "focus") fetchContextFocus();
      if (domain === "memory" && subview === "focus") fetchMemoryFocus();
      if (subview === "stats") renderStats();
    };
    source.onerror = () => updateHealth(false);
  }

  function doRefresh() {
    refreshBtn.classList.add("refreshing");
    Promise.all([
      fetchArtifacts(),
      fetchContextEntries(),
      fetchMemoryEntries(),
    ]).finally(() => {
      refreshBtn.classList.remove("refreshing");
    });
    // Refresh active focus view
    const { domain, subview } = currentRoute();
    if (domain === "artifacts" && subview === "focus") fetchFocus();
    if (domain === "context" && subview === "focus") fetchContextFocus();
    if (domain === "memory" && subview === "focus") fetchMemoryFocus();
    if (subview === "stats") renderStats();
    // Reconnect SSE in case connection dropped
    connectSSE();
  }

  // ---- Resize forwarding for focus iframes ----

  function initFocusResizeForwarding() {
    // Plotly (and other charting libs) listen for `window.resize` on their own
    // window object.  When the gallery container resizes (browser window resize,
    // panel drag, or outer iframe resize), no resize event fires inside the
    // focus iframe.  A ResizeObserver on the focus containers detects the DOM
    // size change and dispatches a synthetic resize event into the embedded
    // iframe so charts re-render at the new size.
    const viewports = [focusContainer, ctxFocusContainer];
    viewports.forEach((container) => {
      if (!container) return;
      const observer = new ResizeObserver(() => {
        const iframe = container.querySelector(".focus-iframe, .focus-iframe-dark");
        if (iframe && iframe.contentWindow) {
          try {
            iframe.contentWindow.dispatchEvent(new Event("resize"));
          } catch {
            // cross-origin iframe — nothing we can do
          }
        }
      });
      observer.observe(container);
    });
  }

  // ---- Init ----

  initRouter();
  initSplitPaneResize(resizeHandle, sidebar);
  initSplitPaneResize(ctxResizeHandle, ctxSidebar);
  initSplitPaneResize(memResizeHandle, memSidebarEl);
  initFocusResizeForwarding();
  refreshBtn.addEventListener("click", doRefresh);
  fetchArtifacts();
  fetchContextEntries();
  fetchMemoryEntries();
  connectSSE();
})();
