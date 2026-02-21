/**
 * OSPREY Artifact Gallery — Unified Browse View
 *
 * Single gallery for all artifacts with type filtering, highlight/pin flags,
 * and inline timeseries rendering. No more domain switcher.
 */
(function () {
  "use strict";

  // ---- DOM Refs ----

  const headerCount = document.getElementById("header-count");
  const healthDot = document.getElementById("health-indicator");
  const refreshBtn = document.getElementById("refresh-btn");
  const filterBar = document.getElementById("filter-bar");
  const searchInput = document.getElementById("search");
  const sidebarBody = document.getElementById("sidebar-body");
  const sidebar = document.getElementById("browse-sidebar");
  const resizeHandle = document.getElementById("resize-handle");
  const previewEmpty = document.getElementById("preview-empty");
  const previewContent = document.getElementById("preview-content");

  // ---- State ----

  let artifacts = [];
  let selectedArtifact = null;
  let focusedArtifact = null;
  let browseMode = "tree"; // "tree" | "activity"
  let sidebarLayout = "list"; // "list" | "gallery"
  let activeFilter = "all"; // "all" | "highlighted" | "pinned" | "stats" | type string
  let activeView = "browse"; // "browse" | "stats"
  let typeRegistry = {};
  let _sessionStart = new Date().toISOString();

  // ---- Type Registry ----

  async function initTypeRegistry() {
    try {
      const resp = await fetch("/api/type-registry");
      typeRegistry = await resp.json();
    } catch (err) {
      console.error("Failed to load type registry:", err);
    }
  }

  function typeBadge(type) {
    const info =
      (typeRegistry.artifact_types && typeRegistry.artifact_types[type]) || {};
    return info.label || type.replace(/_/g, " ");
  }

  function typeIcon(type) {
    const icons = {
      plot_html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
      plot_png: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>',
      table_html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/></svg>',
      html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
      markdown: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6"/></svg>',
      text: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>',
      json: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6"/></svg>',
      image: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>',
      notebook: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>',
      dashboard_html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
    };
    return icons[type] || icons.text;
  }

  function typeColor(type) {
    const info =
      (typeRegistry.artifact_types && typeRegistry.artifact_types[type]) || {};
    return info.color || "#64748b";
  }

  function thumbnailHtml(a) {
    switch (a.artifact_type) {
      case "plot_png":
      case "image":
        return `<img src="${fileUrl(a)}" alt="" loading="lazy" />`;
      default:
        return `<div class="gallery-card-thumb-icon">${typeIcon(a.artifact_type)}</div>`;
    }
  }

  // ---- Utilities ----

  function escapeHtml(str) {
    const d = document.createElement("div");
    d.textContent = str || "";
    return d.innerHTML;
  }

  function formatSize(bytes) {
    if (!bytes) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    let i = 0;
    let size = bytes;
    while (size >= 1024 && i < units.length - 1) { size /= 1024; i++; }
    return `${size.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
  }

  function formatTime(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  }

  function formatFullTime(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      year: "numeric", month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit",
    });
  }

  function formatDate(iso) {
    if (!iso) return "Unknown";
    const d = new Date(iso);
    const now = new Date();
    if (d.toDateString() === now.toDateString()) return "Today";
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    if (d.toDateString() === yesterday.toDateString()) return "Yesterday";
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }

  function fileUrl(a) {
    return `/files/${a.id}/${encodeURIComponent(a.filename)}`;
  }

  function isNewThisSession(a) {
    return a.timestamp && a.timestamp >= _sessionStart;
  }

  function sendToTerminal(text) {
    try {
      if (window.parent && window.parent !== window) {
        window.parent.postMessage({ type: "paste-to-terminal", text }, "*");
      }
    } catch { /* cross-origin */ }
  }

  function updateHealth(ok) {
    if (healthDot) healthDot.className = ok ? "health-dot healthy" : "health-dot";
  }

  function updateHeaderCount() {
    if (headerCount) headerCount.textContent = artifacts.length;
  }

  const chevronSvg = '<svg class="tree-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18l6-6-6-6"/></svg>';

  // ---- Color pass: color badges by type ----

  function requestColorPass() {
    requestAnimationFrame(() => {
      document.querySelectorAll("[class*='badge-']").forEach((el) => {
        const cls = [...el.classList].find((c) => c.startsWith("badge-"));
        if (cls) {
          const type = cls.replace("badge-", "");
          const color = typeColor(type);
          el.style.color = color;
          el.style.borderColor = color;
        }
      });
    });
  }

  // ---- Split-Pane Resize ----

  function initSplitPaneResize(handle, sidebarEl) {
    if (!handle || !sidebarEl) return;
    let startX, startWidth;
    handle.addEventListener("mousedown", (e) => {
      e.preventDefault();
      startX = e.clientX;
      startWidth = sidebarEl.offsetWidth;
      const onMove = (ev) => {
        const delta = ev.clientX - startX;
        const newW = Math.max(180, Math.min(startWidth + delta, window.innerWidth * 0.6));
        sidebarEl.style.width = newW + "px";
      };
      const onUp = () => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
      };
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    });
  }

  // ---- Filter Bar ----

  function initFilterBar() {
    if (!filterBar) return;
    rebuildTypeChips();
    wireFilterChips();
  }

  /** Rebuild all conditional filter chips based on current artifacts. */
  function rebuildTypeChips() {
    // --- Highlighted / Pinned chips: show only when count > 0 ---
    const highlightedCount = artifacts.filter((a) => a.highlighted).length;
    const pinnedCount = artifacts.filter((a) => a.pinned).length;

    const highlightedChip = filterBar.querySelector('[data-filter="highlighted"]');
    const pinnedChip = filterBar.querySelector('[data-filter="pinned"]');

    if (highlightedChip) {
      highlightedChip.hidden = highlightedCount === 0;
      const countEl = highlightedChip.querySelector(".chip-count");
      if (countEl) countEl.textContent = highlightedCount || "";
    }
    if (pinnedChip) {
      pinnedChip.hidden = pinnedCount === 0;
      const countEl = pinnedChip.querySelector(".chip-count");
      if (countEl) countEl.textContent = pinnedCount || "";
    }

    // --- Type chips: show only types that have artifacts ---
    const typesContainer = document.getElementById("filter-type-chips");
    if (!typesContainer) return;

    const presentTypes = new Set(artifacts.map((a) => a.artifact_type));

    // If current filter no longer has artifacts, reset to "all"
    if (activeFilter === "highlighted" && highlightedCount === 0) {
      activeFilter = "all";
    } else if (activeFilter === "pinned" && pinnedCount === 0) {
      activeFilter = "all";
    } else if (activeFilter !== "all" && activeFilter !== "highlighted"
        && activeFilter !== "pinned" && !presentTypes.has(activeFilter)) {
      activeFilter = "all";
    }

    typesContainer.innerHTML = "";
    if (typeRegistry.artifact_types) {
      Object.entries(typeRegistry.artifact_types).forEach(([type, info]) => {
        if (!presentTypes.has(type)) return;
        const count = artifacts.filter((a) => a.artifact_type === type).length;
        const chip = document.createElement("button");
        chip.className = "filter-chip type-chip";
        chip.dataset.filter = type;
        chip.innerHTML = `<span class="chip-icon">${typeIcon(type)}</span>${info.label || type} <span class="chip-count">${count}</span>`;
        typesContainer.appendChild(chip);
      });
    }

    wireFilterChips();
    updateFilterBarActive();
  }

  function wireFilterChips() {
    filterBar.querySelectorAll(".filter-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const filter = chip.dataset.filter;

        if (filter === "stats") {
          activeView = activeView === "stats" ? "browse" : "stats";
          updateViewVisibility();
          updateFilterBarActive();
          if (activeView === "stats") renderStats();
          return;
        }

        activeView = "browse";
        activeFilter = filter;
        updateViewVisibility();
        updateFilterBarActive();
        renderSidebar();
      });
    });
  }

  function updateFilterBarActive() {
    filterBar.querySelectorAll(".filter-chip").forEach((chip) => {
      const f = chip.dataset.filter;
      if (f === "stats") {
        chip.classList.toggle("active", activeView === "stats");
      } else {
        chip.classList.toggle("active", activeView === "browse" && activeFilter === f);
      }
    });
  }

  function updateViewVisibility() {
    const browseView = document.getElementById("view-artifacts-browse");
    const statsView = document.getElementById("view-stats");
    if (browseView) browseView.classList.toggle("hidden", activeView !== "browse");
    if (statsView) statsView.classList.toggle("hidden", activeView !== "stats");
  }

  // ---- API ----

  async function fetchArtifacts() {
    try {
      const resp = await fetch("/api/artifacts");
      const data = await resp.json();
      artifacts = data.artifacts || [];
      updateHealth(true);
      updateHeaderCount();
      rebuildTypeChips();
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
      if (data.artifact) {
        focusedArtifact = data.artifact;
      }
    } catch (err) {
      console.error("Failed to fetch focus:", err);
    }
  }

  // ---- Filtering ----

  function getFilteredArtifacts() {
    let filtered = [...artifacts];

    // Apply filter
    if (activeFilter === "highlighted") {
      filtered = filtered.filter((a) => a.highlighted);
    } else if (activeFilter === "pinned") {
      filtered = filtered.filter((a) => a.pinned);
    } else if (activeFilter !== "all") {
      // Type filter
      filtered = filtered.filter((a) => a.artifact_type === activeFilter);
    }

    // Apply search
    const query = searchInput.value.trim().toLowerCase();
    if (query) {
      filtered = filtered.filter(
        (a) =>
          a.title.toLowerCase().includes(query) ||
          a.filename.toLowerCase().includes(query) ||
          (a.description && a.description.toLowerCase().includes(query)) ||
          a.artifact_type.toLowerCase().includes(query)
      );
    }

    // Sort: pinned first, then by timestamp descending
    filtered.sort((a, b) => {
      if (a.pinned && !b.pinned) return -1;
      if (!a.pinned && b.pinned) return 1;
      return (b.timestamp || "").localeCompare(a.timestamp || "");
    });

    return filtered;
  }

  // ---- Gallery Card HTML (shared by both modes in gallery layout) ----

  function galleryCardHtml(a, i) {
    const sel = selectedArtifact && selectedArtifact.id === a.id ? " selected" : "";
    const pinnedCls = a.pinned ? " pinned" : "";
    return `
      <div class="gallery-card${sel}${pinnedCls}"
           data-id="${a.id}"
           data-type="${a.artifact_type}"
           style="animation-delay: ${i * 30}ms">
        <div class="gallery-card-thumb">${thumbnailHtml(a)}</div>
        <div class="gallery-card-info">
          <div class="gallery-card-title" title="${escapeHtml(a.title)}">
            ${a.pinned ? '<span class="pin-indicator" title="Pinned">&#128204;</span>' : ""}
            ${a.highlighted ? '<span class="highlight-indicator" title="Highlighted">&#9733;</span>' : ""}
            ${escapeHtml(a.title)}
          </div>
          <div class="gallery-card-meta">
            <span class="gallery-card-type">${typeBadge(a.artifact_type)}</span>
            <span class="gallery-card-time">${formatTime(a.timestamp)}</span>
            <span class="gallery-card-size">${formatSize(a.size_bytes)}</span>
          </div>
        </div>
      </div>`;
  }

  // ---- Sidebar Rendering ----

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
          <span>${searchInput.value || activeFilter !== "all" ? "No matches" : "No artifacts yet"}</span>
        </div>
      `;
      return;
    }

    if (browseMode === "tree") {
      renderTreeMode(filtered);
    } else {
      renderActivityMode(filtered);
    }
    requestColorPass();
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
                <div class="tree-item${selectedArtifact && selectedArtifact.id === a.id ? " selected" : ""}${a.pinned ? " pinned" : ""}"
                     data-id="${a.id}"
                     style="animation-delay: ${i * 30}ms">
                  ${a.pinned ? '<span class="pin-indicator" title="Pinned">&#128204;</span>' : ""}
                  ${a.highlighted ? '<span class="highlight-indicator" title="Highlighted">&#9733;</span>' : ""}
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
    const dateGroups = {};
    items.forEach((a) => {
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
        group.forEach((a) => { html += galleryCardHtml(a, itemIndex++); });
        html += `</div>`;
      } else {
        group.forEach((a) => {
          html += `
            <div class="timeline-item${selectedArtifact && selectedArtifact.id === a.id ? " selected" : ""}${a.pinned ? " pinned" : ""}"
                 data-id="${a.id}"
                 data-type="${a.artifact_type}"
                 style="animation-delay: ${itemIndex * 25}ms">
              <span class="timeline-dot"></span>
              <div class="timeline-item-body">
                <div class="timeline-item-title" title="${escapeHtml(a.title)}">
                  ${a.pinned ? '<span class="pin-indicator">&#128204;</span>' : ""}
                  ${a.highlighted ? '<span class="highlight-indicator">&#9733;</span>' : ""}
                  ${escapeHtml(a.title)}
                </div>
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

    // Item click + drag-and-drop
    const clickables = ".tree-item, .timeline-item, .gallery-card";
    sidebarBody.querySelectorAll(clickables).forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.closest(".tree-section-header, .gallery-section-header")) return;
        const id = el.dataset.id;
        const a = artifacts.find((x) => x.id === id);
        if (a) {
          selectedArtifact = a;
          sidebarBody.querySelectorAll(clickables).forEach((item) => item.classList.remove("selected"));
          el.classList.add("selected");
          renderPreview();
        }
      });

      // Send-to-terminal: double-click
      el.addEventListener("dblclick", (e) => {
        e.preventDefault();
        const id = el.dataset.id;
        const a = artifacts.find((x) => x.id === id);
        if (!a) return;
        sendToTerminal(`Read this artifact file: osprey-workspace/artifacts/${a.filename}`);
      });
    });
  }

  // ---- Preview Pane ----

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

    // Check for timeseries data
    if (a.metadata && a.metadata.data_type === "timeseries" && a.metadata.data_file) {
      viewportHtml = `<div id="ts-viewport" class="ts-viewport-container"></div>`;
    } else {
      switch (a.artifact_type) {
        case "plot_html":
        case "table_html":
        case "dashboard_html":
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
    }

    previewContent.innerHTML = `
      <div class="preview-header">
        <div class="preview-header-left">
          <span class="badge badge-${a.artifact_type}">${typeBadge(a.artifact_type)}</span>
          <span class="preview-header-title">${escapeHtml(a.title)}</span>
        </div>
        <div class="preview-header-actions">
          <button class="btn btn-sm ${a.highlighted ? "btn-warning" : "btn-secondary"}" id="preview-toggle-highlight" title="${a.highlighted ? "Remove highlight" : "Highlight"}">
            &#9733; ${a.highlighted ? "Highlighted" : "Highlight"}
          </button>
          <button class="btn btn-sm ${a.pinned ? "btn-primary" : "btn-secondary"}" id="preview-toggle-pin" title="${a.pinned ? "Unpin" : "Pin"}">
            &#128204; ${a.pinned ? "Pinned" : "Pin"}
          </button>
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

    // Wire action buttons
    document.getElementById("preview-toggle-highlight").addEventListener("click", () => toggleHighlight(a));
    document.getElementById("preview-toggle-pin").addEventListener("click", () => togglePin(a));
    document.getElementById("preview-set-focus").addEventListener("click", () => setAsFocus(a));
    document.getElementById("preview-delete").addEventListener("click", () => {
      if (!confirm(`Delete "${a.title}"? This cannot be undone.`)) return;
      fetch(`/api/artifacts/${a.id}`, { method: "DELETE" })
        .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
        .then(() => {
          artifacts = artifacts.filter((x) => x.id !== a.id);
          if (selectedArtifact?.id === a.id) selectedArtifact = null;
          if (focusedArtifact?.id === a.id) focusedArtifact = null;
          updateHeaderCount();
          renderSidebar();
          renderPreview();
        })
        .catch((err) => console.error("Delete failed:", err));
    });

    // Render timeseries if applicable
    if (a.metadata && a.metadata.data_type === "timeseries" && a.metadata.data_file) {
      const tsEl = document.getElementById("ts-viewport");
      if (tsEl) renderTimeseriesView(tsEl, a);
    }

    requestColorPass();
    if (window.injectLogbookButtons) window.injectLogbookButtons();
  }

  // ---- Pin / Highlight / Focus actions ----

  async function togglePin(artifact) {
    const newPinned = !artifact.pinned;
    try {
      const resp = await fetch(`/api/artifacts/${artifact.id}/pin`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pinned: newPinned }),
      });
      if (resp.ok) {
        artifact.pinned = newPinned;
        renderSidebar();
        renderPreview();
      }
    } catch (err) {
      console.error("Failed to toggle pin:", err);
    }
  }

  async function toggleHighlight(artifact) {
    const newHighlighted = !artifact.highlighted;
    try {
      const resp = await fetch(`/api/artifacts/${artifact.id}/highlight`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ highlighted: newHighlighted }),
      });
      if (resp.ok) {
        artifact.highlighted = newHighlighted;
        renderSidebar();
        renderPreview();
      }
    } catch (err) {
      console.error("Failed to toggle highlight:", err);
    }
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
      }
    } catch (err) {
      console.error("Failed to set focus:", err);
    }
  }

  // ---- Timeseries Rendering ----

  let _plotlyLoaded = false;
  let _plotlyLoading = null;

  function ensurePlotlyLoaded() {
    if (_plotlyLoaded) return Promise.resolve();
    if (_plotlyLoading) return _plotlyLoading;
    _plotlyLoading = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "https://cdn.plot.ly/plotly-2.35.2.min.js";
      script.onload = () => { _plotlyLoaded = true; resolve(); };
      script.onerror = () => reject(new Error("Failed to load Plotly"));
      document.head.appendChild(script);
    });
    return _plotlyLoading;
  }

  const _tsColorway = [
    "#4fd1c5", "#d4a574", "#9f7aea", "#3b82f6",
    "#22c55e", "#f59e0b", "#ef4444", "#e879f9",
  ];

  function _esc(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function _tsFormatValue(num) {
    if (num == null) return "--";
    if (typeof num !== "number") return String(num);
    const abs = Math.abs(num);
    if (abs === 0) return "0";
    if (abs >= 1e6 || abs < 0.001) return num.toExponential(3);
    return num.toPrecision(5);
  }

  function _tsShortChannelName(name) {
    if (!name) return "";
    if (name.length <= 24) return name;
    const parts = name.split(":");
    if (parts.length >= 3) return parts[0] + ":...:" + parts[parts.length - 1];
    return name.slice(0, 10) + "..." + name.slice(-10);
  }

  function _tsShortTime(iso) {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      return d.toLocaleString(undefined, {
        month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
      });
    } catch { return String(iso); }
  }

  function _tsExportCSV(chartData) {
    const cols = chartData.columns || [];
    const rows = [["timestamp", ...cols].join(",")];
    chartData.index.forEach((ts, i) => {
      const vals = chartData.data[i] || [];
      rows.push([ts, ...vals].join(","));
    });
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `timeseries_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function _tsExportJSON(chartData) {
    const blob = new Blob([JSON.stringify(chartData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `timeseries_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function renderTimeseriesView(container, artifact) {
    container.innerHTML = '<div class="ts-loading">Loading timeseries data...</div>';
    try {
      const chartResp = await fetch(
        `/api/artifacts/${artifact.id}/data?format=chart&max_points=2000`
      );
      if (!chartResp.ok) throw new Error(`Chart fetch failed: ${chartResp.status}`);
      const chartData = await chartResp.json();
      const columns = chartData.columns || [];
      const summary = artifact.metadata || {};

      const visible = new Set(columns);

      let html = '<div class="ts-viewer">';

      // Info bar
      html += '<div class="ts-info-bar">';
      columns.forEach((c) => {
        html += `<span class="ts-badge ts-badge-channel"><span class="badge-label">CH</span> ${_esc(_tsShortChannelName(c))}</span>`;
      });
      html += `<span class="ts-badge ts-badge-rows"><span class="badge-label">Rows</span> ${chartData.total_rows.toLocaleString()}</span>`;
      if (chartData.downsampled) {
        html += `<span class="ts-badge ts-badge-downsampled"><span class="badge-label">Downsampled</span> ${chartData.returned_points.toLocaleString()} pts</span>`;
      }
      html += '</div>';

      // Toolbar
      html += '<div class="ts-toolbar">';
      html += '<div class="ts-toolbar-group">';
      columns.forEach((col, ci) => {
        const color = _tsColorway[ci % _tsColorway.length];
        html += `<button class="ts-ch-toggle" data-ch-index="${ci}" data-ch-name="${_esc(col)}" title="${_esc(col)}">`;
        html += `<span class="ts-ch-dot" style="background:${color}"></span>`;
        html += _esc(_tsShortChannelName(col));
        html += '</button>';
      });
      html += '</div>';
      html += '<span class="ts-toolbar-divider"></span>';
      html += '<div class="ts-toolbar-group">';
      html += '<button class="ts-action-btn" data-action="zoom-reset" title="Reset zoom">Reset Zoom</button>';
      html += '<button class="ts-action-btn ts-export-btn" data-action="export-csv" title="Export CSV">CSV</button>';
      html += '<button class="ts-action-btn ts-export-btn" data-action="export-json" title="Export JSON">JSON</button>';
      html += '</div>';
      html += '</div>';

      // Chart
      html += '<div class="ts-chart-container" data-ts-chart></div>';

      // Table
      html += '<div data-ts-table></div>';
      html += '</div>';
      container.innerHTML = html;

      // Wire events
      const chartEl = container.querySelector("[data-ts-chart]");

      container.querySelectorAll(".ts-ch-toggle").forEach((btn) => {
        btn.addEventListener("click", () => {
          const ci = parseInt(btn.dataset.chIndex, 10);
          const col = columns[ci];
          if (visible.has(col)) {
            if (visible.size <= 1) return;
            visible.delete(col);
            btn.classList.add("ts-ch-off");
          } else {
            visible.add(col);
            btn.classList.remove("ts-ch-off");
          }
          if (chartEl && chartEl.data) {
            const update = columns.map((c) => visible.has(c));
            Plotly.restyle(chartEl, { visible: update });
          }
        });
      });

      container.querySelectorAll(".ts-action-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          const action = btn.dataset.action;
          if (action === "zoom-reset" && chartEl) {
            Plotly.relayout(chartEl, { "xaxis.autorange": true, "yaxis.autorange": true });
          } else if (action === "export-csv") {
            _tsExportCSV(chartData);
          } else if (action === "export-json") {
            _tsExportJSON(chartData);
          }
        });
      });

      await renderTimeseriesChart(chartEl, chartData);

      const tableEl = container.querySelector("[data-ts-table]");
      await renderTimeseriesTable(tableEl, artifact.id, columns, 0);
    } catch (err) {
      console.error("Timeseries render failed:", err);
      container.innerHTML = '<span class="text-muted">Failed to load timeseries data</span>';
    }
  }

  async function renderTimeseriesChart(el, chartData) {
    await ensurePlotlyLoaded();
    if (!el) return;

    const traces = chartData.columns.map((col, ci) => ({
      x: chartData.index,
      y: chartData.data.map((row) => row[ci]),
      name: col,
      type: "scattergl",
      mode: "lines",
      hovertemplate: "%{y:.4g}<extra>%{fullData.name}</extra>",
    }));

    const layout = {
      paper_bgcolor: "#131c2e",
      plot_bgcolor: "#0b1120",
      font: { family: "'DM Mono', monospace", size: 11, color: "#8b9ab5" },
      margin: { t: 30, r: 20, b: 50, l: 60 },
      hovermode: "x unified",
      xaxis: { gridcolor: "rgba(100,116,139,0.1)", linecolor: "rgba(100,116,139,0.18)", tickfont: { size: 10 } },
      yaxis: { gridcolor: "rgba(100,116,139,0.1)", linecolor: "rgba(100,116,139,0.18)", tickfont: { size: 10 } },
      legend: { bgcolor: "rgba(19,28,46,0.85)", bordercolor: "rgba(100,116,139,0.18)", borderwidth: 1, font: { size: 10 } },
      colorway: _tsColorway,
    };

    Plotly.newPlot(el, traces, layout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
    });
  }

  const TS_TABLE_PAGE_SIZE = 50;

  async function renderTimeseriesTable(el, artifactId, columns, offset) {
    if (!el) return;
    el.innerHTML = '<div class="ts-loading">Loading table...</div>';

    try {
      const resp = await fetch(
        `/api/artifacts/${artifactId}/data?format=table&offset=${offset}&limit=${TS_TABLE_PAGE_SIZE}`
      );
      if (!resp.ok) throw new Error(`Table fetch failed: ${resp.status}`);
      const tableData = await resp.json();

      const totalPages = Math.ceil(tableData.total_rows / TS_TABLE_PAGE_SIZE);
      const currentPage = Math.floor(offset / TS_TABLE_PAGE_SIZE) + 1;

      let html = '<div class="ts-data-table-wrapper"><table class="ts-data-table">';
      html += '<thead><tr><th>Index</th>';
      columns.forEach((c) => { html += `<th>${_esc(c)}</th>`; });
      html += '</tr></thead><tbody>';

      tableData.index.forEach((idx, i) => {
        html += '<tr>';
        html += `<td class="ts-index-cell">${_esc(String(idx))}</td>`;
        const row = tableData.data[i] || [];
        row.forEach((val) => { html += `<td>${_esc(val == null ? "" : String(val))}</td>`; });
        html += '</tr>';
      });

      html += '</tbody></table></div>';

      html += '<div class="ts-pagination">';
      html += `<button class="btn btn-secondary btn-sm" data-ts-prev ${offset === 0 ? "disabled" : ""}>Prev</button>`;
      html += `<span class="ts-page-info">Page ${currentPage} of ${totalPages}</span>`;
      html += `<button class="btn btn-secondary btn-sm" data-ts-next ${offset + TS_TABLE_PAGE_SIZE >= tableData.total_rows ? "disabled" : ""}>Next</button>`;
      html += '</div>';

      el.innerHTML = html;

      el.querySelector("[data-ts-prev]")?.addEventListener("click", () => {
        renderTimeseriesTable(el, artifactId, columns, Math.max(0, offset - TS_TABLE_PAGE_SIZE));
      });
      el.querySelector("[data-ts-next]")?.addEventListener("click", () => {
        renderTimeseriesTable(el, artifactId, columns, offset + TS_TABLE_PAGE_SIZE);
      });
    } catch (err) {
      console.error("Table render failed:", err);
      el.innerHTML = '<span class="text-muted">Failed to load table data</span>';
    }
  }

  // ---- JSON Viewer ----

  function renderJsonHtml(obj, depth) {
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
      const items = keys.map((k) => `<div class="json-item"><span class="json-key">"${escapeHtml(k)}"</span>: ${renderJsonHtml(obj[k], depth + 1)}</div>`).join("");
      return `<span class="json-bracket">{</span><div class="json-indent">${items}</div><span class="json-bracket">}</span>`;
    }
    return escapeHtml(String(obj));
  }

  // ---- Stats View ----

  function renderStats() {
    const statsContent = document.getElementById("stats-content");
    if (!statsContent) return;

    const totalCount = artifacts.length;
    const totalSize = artifacts.reduce((sum, a) => sum + (a.size_bytes || 0), 0);
    const typeCounts = {};
    artifacts.forEach((a) => { typeCounts[a.artifact_type] = (typeCounts[a.artifact_type] || 0) + 1; });

    const highlightedCount = artifacts.filter((a) => a.highlighted).length;
    const pinnedCount = artifacts.filter((a) => a.pinned).length;

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
      <div class="stat-card">
        <div class="stat-card-title">Highlighted</div>
        <div class="stat-card-value">${highlightedCount}</div>
        <div class="stat-card-subtitle">agent-flagged items</div>
      </div>
      <div class="stat-card">
        <div class="stat-card-title">Pinned</div>
        <div class="stat-card-value">${pinnedCount}</div>
        <div class="stat-card-subtitle">quick-access items</div>
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

    statsContent.innerHTML = html;
  }

  // ---- Events ----

  if (searchInput) {
    searchInput.addEventListener("input", debounce(() => renderSidebar(), 200));
  }

  // Mode toggle (tree/activity)
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.dataset.mode;
      if (mode === browseMode) return;
      browseMode = mode;
      document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderSidebar();
    });
  });

  // Layout toggle (list/gallery)
  document.querySelectorAll(".layout-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const layout = btn.dataset.layout;
      if (layout === sidebarLayout) return;
      sidebarLayout = layout;
      document.querySelectorAll(".layout-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderSidebar();
    });
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== searchInput) {
      e.preventDefault();
      searchInput.focus();
    }
    if (e.key === "Escape" && document.activeElement === searchInput) {
      searchInput.blur();
      searchInput.value = "";
      renderSidebar();
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
    if (sseSource) { sseSource.close(); sseSource = null; }
    const source = new EventSource("/api/events");
    sseSource = source;
    source.onopen = () => updateHealth(true);
    source.onmessage = (event) => {
      updateHealth(true);
      let eventData = null;
      try { eventData = JSON.parse(event.data); } catch { return; }
      const eventType = eventData && eventData.type;

      if (eventType === "focus") {
        fetchFocus();
        return;
      }

      if (eventType === "artifact_deleted") {
        artifacts = artifacts.filter((a) => a.id !== eventData.id);
        if (focusedArtifact?.id === eventData.id) focusedArtifact = null;
        if (selectedArtifact?.id === eventData.id) { selectedArtifact = null; renderPreview(); }
        updateHeaderCount();
        rebuildTypeChips();
        renderSidebar();
        return;
      }

      if (eventType === "artifact_updated") {
        // Update the artifact in-place
        const idx = artifacts.findIndex((a) => a.id === eventData.id);
        if (idx >= 0) {
          artifacts[idx] = { ...artifacts[idx], ...eventData };
          if (selectedArtifact?.id === eventData.id) {
            selectedArtifact = artifacts[idx];
            renderPreview();
          }
          renderSidebar();
        }
        return;
      }

      if (eventType === "artifact" || !eventType) {
        fetchArtifacts().catch(() => {});
      }

      if (activeView === "stats") renderStats();
    };
    source.onerror = () => updateHealth(false);
  }

  function doRefresh() {
    refreshBtn.classList.add("refreshing");
    fetchArtifacts().finally(() => {
      refreshBtn.classList.remove("refreshing");
    });
    if (activeView === "stats") renderStats();
    connectSSE();
  }

  // ---- Init ----

  initSplitPaneResize(resizeHandle, sidebar);
  refreshBtn.addEventListener("click", doRefresh);
  initTypeRegistry().then(() => {
    initFilterBar();
    fetchArtifacts();
    fetchFocus();
    connectSSE();
  });
})();
