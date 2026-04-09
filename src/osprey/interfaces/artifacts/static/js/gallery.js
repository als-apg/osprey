/**
 * OSPREY Artifact Gallery — Unified Browse View
 *
 * Single gallery for all artifacts with type filtering, pin flag,
 * and inline timeseries rendering.
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
  let activeFilter = "all"; // "all" | "pinned" | type string
  let typeRegistry = {};
  let _sessionStart = new Date().toISOString();
  let isFullscreen = false;
  let _newArtifactsSinceFullscreen = 0;
  let currentSessionId = null;
  let showAllSessions = false;

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
      (typeRegistry.categories && typeRegistry.categories[type]) ||
      (typeRegistry.artifact_types && typeRegistry.artifact_types[type]) || {};
    return info.label || type.replace(/_/g, " ");
  }

  function typeIcon(type) {
    const icons = {
      // Artifact types
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
      // Category types
      archiver_data: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 5v14c0 1.66-4.03 3-9 3s-9-1.34-9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>',
      channel_values: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
      write_results: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>',
      code_output: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
      visualization: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><polyline points="7 14 11 10 15 14 19 8"/></svg>',
      dashboard: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
      document: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>',
      screenshot: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>',
      channel_finder: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
      search_results: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
      logbook_research: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>',
      literature_research: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>',
      wiki_research: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>',
      mml_research: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
      agent_response: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>',
      user_artifact: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
      diagnostic_report: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M16 4h2a2 2 0 012 2v14a2 2 0 01-2 2H6a2 2 0 01-2-2V6a2 2 0 012-2h2"/><rect x="8" y="2" width="8" height="4" rx="1"/><path d="M9 14l2 2 4-4"/></svg>',
    };
    return icons[type] || icons.text;
  }

  function typeColor(type) {
    const info =
      (typeRegistry.categories && typeRegistry.categories[type]) ||
      (typeRegistry.artifact_types && typeRegistry.artifact_types[type]) || {};
    return info.color || "#64748b";
  }

  function thumbnailHtml(a) {
    const url = fileUrl(a);
    switch (a.artifact_type) {
      case "plot_png":
      case "image":
        return `<img src="${url}" alt="" loading="lazy"
                 onerror="this.parentElement.classList.add('img-error')" />`;
      case "plot_html":
      case "table_html":
      case "dashboard_html":
      case "html":
        return `<iframe src="${url}" sandbox="allow-scripts allow-same-origin"
                 loading="lazy" tabindex="-1"></iframe>`;
      case "notebook":
        return `<iframe src="/api/notebooks/${a.id}/rendered"
                 sandbox="allow-scripts allow-same-origin"
                 loading="lazy" tabindex="-1"></iframe>`;
      default:
        if (a.summary && Object.keys(a.summary).length > 0) {
          const text = Object.entries(a.summary)
            .map(([k, v]) => `${k}: ${v}`)
            .join("\n");
          return `<div class="thumb-summary">${escapeHtml(text)}</div>`;
        }
        return `<div class="thumb-placeholder">${typeIcon(a.artifact_type)}<span>${typeBadge(a.artifact_type)}</span></div>`;
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

  /** URL for "Open in new tab" — uses rendered endpoints for types that
   *  browsers can't display natively (markdown, notebook). */
  function openUrl(a) {
    switch (a.artifact_type) {
      case "markdown": return `/api/markdown/${a.id}/rendered`;
      case "notebook": return `/api/notebooks/${a.id}/rendered`;
      default:         return fileUrl(a);
    }
  }

  function isNewThisSession(a) {
    return a.timestamp && a.timestamp >= _sessionStart;
  }

  function sendToTerminal(text) {
    try {
      if (window.parent && window.parent !== window) {
        window.parent.postMessage({ type: "osprey-paste-to-terminal", text }, "*");
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

  function _setTypeColorVars(el, color) {
    el.style.setProperty("--type-color", color);
    el.style.setProperty("--type-bg", color + "14");
    el.style.setProperty("--type-border", color + "40");
  }

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
      document.querySelectorAll(".tree-section[data-type]").forEach((el) => {
        const type = el.dataset.type;
        if (type) _setTypeColorVars(el, typeColor(type));
      });
      document.querySelectorAll(".gallery-card[data-type]").forEach((el) => {
        const type = el.dataset.type;
        if (type) _setTypeColorVars(el, typeColor(type));
      });
      document.querySelectorAll(".timeline-item[data-type]").forEach((el) => {
        const type = el.dataset.type;
        if (type) _setTypeColorVars(el, typeColor(type));
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
    // --- Pinned chip: show only when count > 0 ---
    const pinnedCount = artifacts.filter((a) => a.pinned).length;

    const pinnedChip = filterBar.querySelector('[data-filter="pinned"]');

    if (pinnedChip) {
      pinnedChip.hidden = pinnedCount === 0;
      const countEl = pinnedChip.querySelector(".chip-count");
      if (countEl) countEl.textContent = pinnedCount || "";
    }

    // --- Type chips: show only types that have artifacts ---
    const typesContainer = document.getElementById("filter-type-chips");
    if (!typesContainer) return;

    const presentTypes = new Set(artifacts.map((a) => a.category || a.artifact_type));

    // If current filter no longer has artifacts, reset to "all"
    if (activeFilter === "pinned" && pinnedCount === 0) {
      activeFilter = "all";
    } else if (activeFilter !== "all"
        && activeFilter !== "pinned" && !presentTypes.has(activeFilter)) {
      activeFilter = "all";
    }

    typesContainer.innerHTML = "";
    // Prefer category-based chips when registry provides categories
    const chipSource = typeRegistry.categories || typeRegistry.artifact_types;
    if (chipSource) {
      Object.entries(chipSource).forEach(([type, info]) => {
        if (!presentTypes.has(type)) return;
        const count = artifacts.filter((a) => (a.category || a.artifact_type) === type).length;
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
        activeFilter = chip.dataset.filter;
        updateFilterBarActive();
        renderSidebar();
      });
    });
  }

  function updateFilterBarActive() {
    filterBar.querySelectorAll(".filter-chip").forEach((chip) => {
      chip.classList.toggle("active", activeFilter === chip.dataset.filter);
    });
  }

  // ---- API ----

  function showErrorBanner(msg) {
    var banner = document.getElementById("error-banner");
    if (!banner) {
      banner = document.createElement("div");
      banner.id = "error-banner";
      banner.style.cssText =
        "position:fixed;top:0;left:0;right:0;z-index:9999;padding:12px 20px;" +
        "background:var(--color-error);color:#fff;font-size:14px;text-align:center;";
      document.body.prepend(banner);
    }
    banner.textContent = msg;
    banner.style.display = "block";
  }

  function hideErrorBanner() {
    var banner = document.getElementById("error-banner");
    if (banner) banner.style.display = "none";
  }

  async function fetchArtifacts() {
    try {
      let url = "/api/artifacts";
      if (currentSessionId && !showAllSessions) {
        url += "?session_id=" + encodeURIComponent(currentSessionId);
      }
      const resp = await fetch(url);
      if (!resp.ok) {
        const errText = await resp.text();
        showErrorBanner("API error (" + resp.status + "): " + errText);
        updateHealth(false);
        return;
      }
      hideErrorBanner();
      const data = await resp.json();
      artifacts = data.artifacts || [];
      updateHealth(true);
      updateHeaderCount();
      rebuildTypeChips();
      renderSidebar();
    } catch (err) {
      console.error("Failed to fetch artifacts:", err);
      showErrorBanner("Failed to fetch artifacts: " + err.message);
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
    if (activeFilter === "pinned") {
      filtered = filtered.filter((a) => a.pinned);
    } else if (activeFilter !== "all") {
      // Type filter
      filtered = filtered.filter((a) => a.category === activeFilter || a.artifact_type === activeFilter);
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
           data-type="${a.category || a.artifact_type}"
           style="animation-delay: ${i * 30}ms">
        <div class="gallery-card-thumb">${thumbnailHtml(a)}</div>
        <div class="gallery-card-info">
          <div class="gallery-card-title" title="${escapeHtml(a.title)}">
            ${a.pinned ? '<span class="pin-indicator" title="Pinned">&#128204;</span>' : ""}
            ${escapeHtml(a.title)}
          </div>
          <div class="gallery-card-meta">
            <span class="gallery-card-type">${typeBadge(a.category || a.artifact_type)}</span>
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
      const groupKey = a.category || a.artifact_type;
      if (!groups[groupKey]) groups[groupKey] = [];
      groups[groupKey].push(a);
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
                 data-type="${a.category || a.artifact_type}"
                 style="animation-delay: ${itemIndex * 25}ms">
              <span class="timeline-dot"></span>
              <div class="timeline-item-body">
                <div class="timeline-item-title" title="${escapeHtml(a.title)}">
                  ${a.pinned ? '<span class="pin-indicator">&#128204;</span>' : ""}
                  ${escapeHtml(a.title)}
                </div>
                <div class="timeline-item-meta">
                  <span class="timeline-item-type">${typeBadge(a.category || a.artifact_type)}</span>
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

    // Item click, double-click (fullscreen), drag-and-drop (send to terminal)
    const clickables = ".tree-item, .timeline-item, .gallery-card";
    sidebarBody.querySelectorAll(clickables).forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.closest(".tree-section-header, .gallery-section-header")) return;
        const id = el.dataset.id;
        const a = artifacts.find((x) => x.id === id);
        if (a) {
          const alreadySelected = selectedArtifact?.id === a.id;
          selectedArtifact = a;
          setAsFocus(a);
          sidebarBody.querySelectorAll(clickables).forEach((item) => item.classList.remove("selected"));
          el.classList.add("selected");
          if (!alreadySelected) renderPreview();
        }
      });

      // Fullscreen: double-click
      el.addEventListener("dblclick", (e) => {
        e.preventDefault();
        const id = el.dataset.id;
        const a = artifacts.find((x) => x.id === id);
        if (!a) return;
        selectedArtifact = a;
        enterFullscreen(a);
      });

      // Drag-and-drop: drag artifact to terminal to paste reference
      el.draggable = true;
      el.addEventListener("dragstart", (e) => {
        const id = el.dataset.id;
        const a = artifacts.find((x) => x.id === id);
        if (!a) return;
        const text = `Please have a look at _agent_data/artifacts/${a.filename}`;
        e.dataTransfer.setData("text/plain", text);
        e.dataTransfer.effectAllowed = "copy";
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
    if ((a.metadata && a.metadata.data_type === "timeseries" || a.category === "archiver_data") && (a.data_file || (a.metadata && a.metadata.data_file))) {
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
          viewportHtml = `<div id="md-viewport" class="md-preview-container"></div>`;
          break;
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
          <button class="fullscreen-back-btn" id="fullscreen-back" title="Exit fullscreen (Esc)">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
            Back
          </button>
          <span class="badge badge-${a.category || a.artifact_type}">${typeBadge(a.category || a.artifact_type)}</span>
          <span class="preview-header-title">${escapeHtml(a.title)}</span>
        </div>
        <div class="preview-header-actions">
          <span class="fullscreen-new-badge" id="fullscreen-new-badge" data-count="0"></span>
          <button class="btn-action-icon ${a.pinned ? "btn-action-pinned" : ""}" id="preview-toggle-pin" title="${a.pinned ? "Unpin" : "Pin"}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 17v5"/><path d="M5 9l2-1V4a1 1 0 011-1h8a1 1 0 011 1v4l2 1a1 1 0 01.4 1.6l-1.4 2a1 1 0 01-.8.4H6.8a1 1 0 01-.8-.4l-1.4-2A1 1 0 015 9z"/></svg>
          </button>
          <button class="btn-action-icon fullscreen-enter-btn" id="fullscreen-enter" title="Maximize (F)">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 3H5a2 2 0 00-2 2v3M21 8V5a2 2 0 00-2-2h-3M16 21h3a2 2 0 002-2v-3M3 16v3a2 2 0 002 2h3"/></svg>
          </button>
          <a href="${openUrl(a)}" target="_blank" class="btn-action-icon" title="Open in new tab">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3"/></svg>
          </a>
          <button class="btn-action-icon btn-action-danger" id="preview-delete" title="Delete artifact">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/></svg>
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
        <span class="preview-meta-path" id="preview-copy-path" title="Click to copy path">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
          <span class="preview-path-text">_agent_data/artifacts/${escapeHtml(a.filename)}</span>
        </span>
      </div>
      <div class="preview-viewport">
        ${viewportHtml}
      </div>
    `;

    // Wire action buttons
    document.getElementById("preview-toggle-pin").addEventListener("click", () => togglePin(a));
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

    // Copy path to clipboard
    const copyPathBtn = document.getElementById("preview-copy-path");
    if (copyPathBtn) {
      copyPathBtn.addEventListener("click", () => {
        const path = `_agent_data/artifacts/${a.filename}`;
        navigator.clipboard.writeText(path).then(() => {
          copyPathBtn.classList.add("copied");
          setTimeout(() => copyPathBtn.classList.remove("copied"), 1500);
        });
      });
    }

    // Fullscreen buttons
    const enterBtn = document.getElementById("fullscreen-enter");
    if (enterBtn) enterBtn.addEventListener("click", () => enterFullscreen());

    const backBtn = document.getElementById("fullscreen-back");
    if (backBtn) backBtn.addEventListener("click", () => exitFullscreen());

    const newBadge = document.getElementById("fullscreen-new-badge");
    if (newBadge) {
      newBadge.addEventListener("click", () => {
        // Navigate to newest artifact, stay in fullscreen
        if (artifacts.length > 0) {
          const sorted = [...artifacts].sort((a, b) => (b.timestamp || "").localeCompare(a.timestamp || ""));
          selectedArtifact = sorted[0];
          _newArtifactsSinceFullscreen = 0;
          renderPreview();
        }
      });
    }

    // Update badge if in fullscreen
    if (isFullscreen) updateNewArtifactBadge();

    // Render timeseries if applicable
    if ((a.metadata && a.metadata.data_type === "timeseries" || a.category === "archiver_data") && (a.data_file || (a.metadata && a.metadata.data_file))) {
      const tsEl = document.getElementById("ts-viewport");
      if (tsEl) renderTimeseriesView(tsEl, a);
    }

    // Render markdown inline
    if (a.artifact_type === "markdown") {
      const mdEl = document.getElementById("md-viewport");
      if (mdEl) renderMarkdownView(mdEl, a);
    }

    requestColorPass();
    if (window.injectLogbookButtons) window.injectLogbookButtons();
    if (window.injectPrintButton) window.injectPrintButton();
  }

  // ---- Pin / Focus actions ----

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

  // ---- Fullscreen Mode ----

  function enterFullscreen(artifact) {
    if (isFullscreen) return;
    if (artifact) {
      selectedArtifact = artifact;
    }
    if (!selectedArtifact) return;

    isFullscreen = true;
    _newArtifactsSinceFullscreen = 0;
    document.body.classList.add('fullscreen-mode');
    document.body.classList.add('fullscreen-entering');
    setTimeout(() => document.body.classList.remove('fullscreen-entering'), 700);

    updateNewArtifactBadge();
    renderPreview();

    // Resize iframes / Plotly charts
    requestAnimationFrame(() => {
      const iframe = previewContent.querySelector('iframe');
      if (iframe?.contentWindow) {
        try { iframe.contentWindow.dispatchEvent(new Event('resize')); } catch {}
      }
      const tsChart = document.getElementById('ts-viewport');
      if (tsChart && typeof Plotly !== 'undefined') {
        try { Plotly.Plots.resize(tsChart); } catch {}
      }
    });
  }

  function exitFullscreen() {
    if (!isFullscreen) return;
    isFullscreen = false;
    _newArtifactsSinceFullscreen = 0;
    document.body.classList.remove('fullscreen-mode');
    document.body.classList.remove('fullscreen-entering');

    renderPreview();
    renderSidebar();

    // Resize iframes / Plotly charts
    requestAnimationFrame(() => {
      const iframe = previewContent.querySelector('iframe');
      if (iframe?.contentWindow) {
        try { iframe.contentWindow.dispatchEvent(new Event('resize')); } catch {}
      }
      const tsChart = document.getElementById('ts-viewport');
      if (tsChart && typeof Plotly !== 'undefined') {
        try { Plotly.Plots.resize(tsChart); } catch {}
      }
      // Scroll selected item into view in sidebar
      if (selectedArtifact) {
        const sel = sidebarBody.querySelector(`[data-id="${selectedArtifact.id}"]`);
        if (sel) sel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    });
  }

  function updateNewArtifactBadge() {
    const badge = document.getElementById('fullscreen-new-badge');
    if (!badge) return;
    const count = _newArtifactsSinceFullscreen;
    badge.dataset.count = String(count);
    badge.textContent = count > 0 ? `${count} new` : '';
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

  // ---- Markdown Rendering ----

  let _markedConfigured = false;

  function configureMarked() {
    if (_markedConfigured) return;
    _markedConfigured = true;
    if (typeof marked === "undefined") return;

    const renderer = {
      code({ text, lang }) {
        const src = text ?? "";
        let highlighted = escapeHtml(src);
        if (typeof hljs !== "undefined" && src) {
          try {
            if (lang && hljs.getLanguage(lang)) {
              highlighted = hljs.highlight(src, { language: lang }).value;
            } else {
              highlighted = hljs.highlightAuto(src).value;
            }
          } catch { /* fallback to escaped */ }
        }
        return `<pre><code class="hljs${lang ? ` language-${lang}` : ""}">${highlighted}</code></pre>`;
      },
    };

    marked.use({ gfm: true, breaks: false, renderer });
  }

  /**
   * Render LaTeX math in markdown text using KaTeX.
   *
   * Strategy: extract $$...$$ (display) and $...$ (inline) blocks BEFORE
   * marked.js parses the text (to prevent marked from mangling LaTeX
   * special characters like _ and ^).  Replace with placeholders, run
   * marked.parse(), then swap KaTeX-rendered HTML back in.
   */
  function renderMathInMarkdown(text) {
    if (typeof katex === "undefined") return marked.parse(text);

    const placeholders = [];
    let idx = 0;

    function placeholder(html) {
      const key = `\x00MATH${idx++}\x00`;
      placeholders.push({ key, html });
      return key;
    }

    function renderKatex(expr, displayMode) {
      try {
        return katex.renderToString(expr.trim(), {
          displayMode,
          throwOnError: false,
          strict: false,
        });
      } catch {
        const cls = displayMode ? "katex-error-display" : "katex-error-inline";
        return `<span class="${cls}">${escapeHtml(expr)}</span>`;
      }
    }

    // Pass 1: display math $$...$$ (must come before inline $...$)
    text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, expr) =>
      placeholder(renderKatex(expr, true))
    );

    // Pass 2: inline math $...$ (not preceded/followed by digit to avoid $5)
    text = text.replace(/(?<!\$)(?<!\d)\$(?!\$)(.+?)(?<!\$)\$(?!\d)/g, (_, expr) =>
      placeholder(renderKatex(expr, false))
    );

    // Run marked on the placeholder-injected text
    let html;
    try {
      html = marked.parse(text);
    } catch {
      html = `<p>${escapeHtml(text)}</p>`;
    }

    // Swap placeholders back in (safe: KaTeX output is sanitized by the library)
    for (const { key, html: mathHtml } of placeholders) {
      html = html.replace(key, mathHtml);
    }

    return html;
  }

  async function renderMarkdownView(container, artifact) {
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

  // ---- Timeseries Rendering ----

  let _plotlyLoaded = false;
  let _plotlyLoading = null;

  function ensurePlotlyLoaded() {
    if (_plotlyLoaded) return Promise.resolve();
    if (_plotlyLoading) return _plotlyLoading;
    _plotlyLoading = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/static/js/vendor/plotly.min.js";
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

  const _tsThemes = {
    dark: {
      paper: "#131c2e", plot: "#0b1120", font: "#8b9ab5",
      grid: "rgba(100,116,139,0.1)", line: "rgba(100,116,139,0.18)",
      legendBg: "rgba(19,28,46,0.85)", legendBorder: "rgba(100,116,139,0.18)",
    },
    light: {
      paper: "#eef2f7", plot: "#f7f9fc", font: "#0c1322",
      grid: "rgba(0,0,0,0.08)", line: "rgba(0,0,0,0.12)",
      legendBg: "rgba(238,242,247,0.9)", legendBorder: "rgba(0,0,0,0.1)",
    },
  };

  function _currentTheme() {
    return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
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

    const t = _tsThemes[_currentTheme()] || _tsThemes.dark;

    const layout = {
      paper_bgcolor: t.paper,
      plot_bgcolor: t.plot,
      font: { family: "'DM Mono', monospace", size: 11, color: t.font },
      margin: { t: 30, r: 20, b: 50, l: 60 },
      hovermode: "x unified",
      xaxis: { gridcolor: t.grid, linecolor: t.line, tickfont: { size: 10 } },
      yaxis: { gridcolor: t.grid, linecolor: t.line, tickfont: { size: 10 } },
      legend: { bgcolor: t.legendBg, bordercolor: t.legendBorder, borderwidth: 1, font: { size: 10 } },
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

  // Keyboard shortcuts (priority order)
  document.addEventListener("keydown", (e) => {
    const tag = document.activeElement?.tagName;
    const isInput = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";

    // 1. Escape + fullscreen → exit fullscreen (highest priority)
    if (e.key === "Escape" && isFullscreen) {
      e.preventDefault();
      exitFullscreen();
      return;
    }

    // 2. Escape + search focused → clear search
    if (e.key === "Escape" && document.activeElement === searchInput) {
      searchInput.blur();
      searchInput.value = "";
      renderSidebar();
      return;
    }

    // 3. "/" (no input focused) → exit fullscreen first, then focus search
    if (e.key === "/" && !isInput) {
      e.preventDefault();
      if (isFullscreen) exitFullscreen();
      searchInput.focus();
      return;
    }

    // 4. "F" (no input focused, no modifiers) → toggle fullscreen
    if (e.key === "f" && !isInput && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      if (isFullscreen) {
        exitFullscreen();
      } else {
        enterFullscreen();
      }
      return;
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
        // Agent called artifact_focus — select that artifact in the gallery
        const focusId = eventData.id;
        const wantFullscreen = !!eventData.fullscreen;
        if (focusId) {
          const a = artifacts.find((x) => x.id === focusId);
          if (a) {
            selectedArtifact = a;
            focusedArtifact = a;
            renderSidebar();
            renderPreview();
            if (wantFullscreen) enterFullscreen(a);
            // Scroll the selected item into view
            requestAnimationFrame(() => {
              const sel = sidebarBody.querySelector(`[data-id="${focusId}"]`);
              if (sel) sel.scrollIntoView({ behavior: "smooth", block: "nearest" });
            });
          } else {
            // Artifact not yet in local list — refresh and retry
            fetchArtifacts().then(() => {
              const retry = artifacts.find((x) => x.id === focusId);
              if (retry) {
                selectedArtifact = retry;
                focusedArtifact = retry;
                renderSidebar();
                renderPreview();
                if (wantFullscreen) enterFullscreen(retry);
              }
            });
          }
        }
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
        if (isFullscreen) _newArtifactsSinceFullscreen++;
        fetchArtifacts().then(() => {
          if (isFullscreen) updateNewArtifactBadge();
        }).catch(() => {});
      }
    };
    source.onerror = () => updateHealth(false);
  }

  function doRefresh() {
    refreshBtn.classList.add("refreshing");
    fetchArtifacts().finally(() => {
      refreshBtn.classList.remove("refreshing");
    });
    connectSSE();
  }

  // ---- Theme change: forward to iframes + re-style timeseries charts ----

  function _onThemeChange(theme) {
    // Forward to all preview iframes (Plotly HTML artifacts)
    document.querySelectorAll(".preview-viewport iframe, .browse-preview-pane iframe").forEach((iframe) => {
      try { iframe.contentWindow.postMessage({ type: "osprey-theme-change", theme }, "*"); } catch {}
    });
    // Re-style any visible timeseries Plotly chart (target the actual Plotly
    // graph div inside the container, not the outer #ts-viewport wrapper)
    const tsChart = document.querySelector("#ts-viewport [data-ts-chart]");
    if (tsChart && typeof Plotly !== "undefined") {
      const t = _tsThemes[theme] || _tsThemes.dark;
      try {
        Plotly.relayout(tsChart, {
          paper_bgcolor: t.paper, plot_bgcolor: t.plot,
          "font.color": t.font,
          "xaxis.gridcolor": t.grid, "xaxis.linecolor": t.line,
          "yaxis.gridcolor": t.grid, "yaxis.linecolor": t.line,
          "legend.bgcolor": t.legendBg, "legend.bordercolor": t.legendBorder,
        });
      } catch {}
    }
  }

  // Listen for theme changes from the parent (Web Terminal) or self
  window.addEventListener("message", (e) => {
    if (e.data && e.data.type === "osprey-theme-change" && e.data.theme) {
      _onThemeChange(e.data.theme);
    }
    if (e.data && e.data.type === "osprey-session-change" && e.data.session_id) {
      currentSessionId = e.data.session_id;
      const btn = document.getElementById("all-sessions-btn");
      if (btn) btn.classList.remove("active");
      showAllSessions = false;
      fetchArtifacts();
    }
  });

  // Also observe data-theme attribute changes (covers non-postMessage scenarios)
  new MutationObserver((mutations) => {
    for (const m of mutations) {
      if (m.attributeName === "data-theme") {
        _onThemeChange(document.documentElement.getAttribute("data-theme") || "dark");
      }
    }
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  // ---- Expose state for logbook.js / print.js ----

  window._galleryState = {
    getFocusedArtifact() { return focusedArtifact; },
    getSelectedArtifact() { return selectedArtifact; },
    fileUrl,
  };

  // ---- Init ----

  initSplitPaneResize(resizeHandle, sidebar);
  refreshBtn.addEventListener("click", doRefresh);

  const allSessionsBtn = document.getElementById("all-sessions-btn");
  if (allSessionsBtn) {
    allSessionsBtn.addEventListener("click", () => {
      showAllSessions = !showAllSessions;
      allSessionsBtn.classList.toggle("active", showAllSessions);
      fetchArtifacts();
    });
  }
  initTypeRegistry().then(() => {
    initFilterBar();
    fetchArtifacts();
    fetchFocus();
    connectSSE();
  });
})();
